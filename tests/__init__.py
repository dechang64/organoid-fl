# ── tests/__init__.py ──
"""Test package bootstrap.

The guard below is load-bearing, not decoration. ``unittest`` imports this
package before any test module, which makes it the correct hook for forcing
pandas (and therefore pyarrow) to load *ahead of* torch.

Without it, whichever test module imports torch first poisons the process:
a later ``import pandas`` — for example from ``analysis.audit_engine``'s
DataFrame export — triggers a pyarrow access violation that kills the
interpreter with no traceback. Running the suite through pytest hits exactly
that, because pytest loads every module into one process.

See ``utils/import_order.py`` for the measurement and the full explanation.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.import_order import prefer_pandas_before_torch  # noqa: E402

prefer_pandas_before_torch()
