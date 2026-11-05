# ── utils/import_order.py ──
"""Import-ordering guards for this repository.

The problem
-----------
On Windows, importing **pyarrow after torch** corrupts the heap and kills the
interpreter with an access violation (``0xC0000374``, surfaced as
``Windows fatal exception: access violation``). There is no Python traceback
and no exception to catch — the process simply dies.

This was measured on the development machine, in a fresh venv:

    import torch, pandas     -> DIED rc=3221226356
    import torch, pyarrow    -> DIED rc=3221226356
    import pandas, torch     -> OK
    import pyarrow, torch    -> OK

So the incompatible pair is ``torch`` and ``pyarrow``, not ``torch`` and
``pandas`` as it first appears. pandas 2.x imports pyarrow unconditionally at
module scope (``pandas/compat/pyarrow.py``) to decide which Arrow-backed
features to enable, so *any* ``import pandas`` after torch drags pyarrow in and
dies with it.

Why it cannot simply be removed
-------------------------------
``pyarrow`` is a hard requirement of ``streamlit`` and ``mlflow``, both of which
this project depends on, and this pandas version offers no opt-out flag. The
conflict therefore has to be managed by ordering.

The two valid strategies
------------------------
1. **Avoid it.** Keep pandas and pyarrow out of processes that load torch. This
   is what ``utils/helpers.py`` and ``visualization/charts.py`` do, and it is
   the better option for pure training paths — they never need a DataFrame.
2. **Order it.** When a process genuinely needs both (the Streamlit app, the
   audit-chain export, the test suite), import pandas *before* torch. This
   module exists for that case.

``KMP_DUPLICATE_LIB_OK=TRUE`` does **not** help. It silences the OpenMP
duplicate-runtime guard but the underlying access violation still occurs.

Usage
-----
Call :func:`prefer_pandas_before_torch` at the earliest possible point, before
torch is imported::

    from utils.import_order import prefer_pandas_before_torch
    prefer_pandas_before_torch()

It is safe to call repeatedly and safe when pandas or torch is absent.
"""

from __future__ import annotations

import sys

__all__ = [
    "describe_import_state",
    "prefer_pandas_before_torch",
    "torch_already_loaded",
]


def torch_already_loaded() -> bool:
    """True when torch has already been imported in this process.

    Once true, importing pandas (and therefore pyarrow) is unsafe on Windows.
    """
    return "torch" in sys.modules


def prefer_pandas_before_torch() -> None:
    """Import pandas ahead of torch, if both are available.

    Safe to call multiple times: once pandas is in ``sys.modules`` this is a
    dictionary lookup. If torch is *already* loaded, importing pandas would be
    the very operation that crashes, so this deliberately does nothing and
    leaves the caller to avoid pandas instead.

    Nothing is raised when pandas is not installed — a process that does not
    need pandas should not be forced to install it.
    """
    if "pandas" in sys.modules:
        return
    if torch_already_loaded():
        # Loading pandas here would import pyarrow and kill the interpreter.
        # Callers in this state must avoid pandas entirely (see the module
        # docstring, strategy 1).
        return
    try:
        import pandas  # noqa: F401
    except ImportError:
        return


def describe_import_state() -> dict[str, bool]:
    """Snapshot of the imports that decide whether this process is at risk.

    Intended for diagnostics and tests, not for control flow.
    """
    return {
        "pandas_loaded": "pandas" in sys.modules,
        "pyarrow_loaded": "pyarrow" in sys.modules,
        "torch_loaded": torch_already_loaded(),
    }
