# ── tests/test_import_order.py ──
"""Regression guards for the pyarrow-after-torch crash.

The hazard is unusual enough to be worth pinning down with executable evidence
rather than a comment, because it is *invisible* in normal development: the
process dies with no traceback, no exception, and no failing assertion. A
refactor that quietly reorders two imports would reintroduce it and the failure
would look like a mysterious CI timeout.

The measurements these tests encode are:

    import torch, pandas     -> DIED
    import torch, pyarrow    -> DIED
    import pandas, torch     -> OK
    import pyarrow, torch    -> OK
"""

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.import_order import (
    describe_import_state,
    prefer_pandas_before_torch,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code: str) -> subprocess.CompletedProcess:
    """Execute ``code`` in a clean interpreter rooted at the repo."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
        check=False,
    )


def _have(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


@unittest.skipUnless(
    sys.platform == "win32",
    "this hazard is Windows-specific: the pyarrow/torch access violation comes "
    "from conflicting native runtimes that the Windows wheel layout produces. "
    "On Linux and macOS the same import order is harmless, so asserting a crash "
    "there would fail for the wrong reason.",
)
@unittest.skipUnless(
    _have("torch") and _have("pyarrow"),
    "torch and pyarrow are both required to exercise this crash",
)
class TestNativeLibraryConflictIsReal(unittest.TestCase):
    """Pin the underlying hazard so nobody 'simplifies' the guards away.

    Windows-only by necessity — see the skip reason. The guards themselves
    (``TestPreferPandasBeforeTorch``) run everywhere, which is the property that
    matters for portability: they are harmless when the conflict is absent.
    """

    def test_importing_pyarrow_after_torch_kills_the_process(self):
        proc = _run("import torch, pyarrow")
        self.assertNotEqual(
            proc.returncode, 0,
            "pyarrow no longer crashes this interpreter when imported after "
            "torch. If the environment changed, revisit utils/import_order.py "
            "and the guards in tests/ before deleting them.",
        )

    def test_importing_pandas_after_torch_kills_the_process(self):
        """pandas drags pyarrow in at module scope, which is why it dies too."""
        proc = _run("import torch, pandas")
        self.assertNotEqual(proc.returncode, 0)

    def test_the_reverse_order_is_safe(self):
        for code in ("import pandas, torch", "import pyarrow, torch"):
            with self.subTest(code=code):
                proc = _run(code)
                self.assertEqual(proc.returncode, 0, f"{code} died: {proc.stderr[-300:]}")


@unittest.skipUnless(_have("pandas"), "pandas is needed to observe the guard working")
class TestPreferPandasBeforeTorch(unittest.TestCase):
    def test_guard_is_idempotent_and_never_raises(self):
        prefer_pandas_before_torch()
        prefer_pandas_before_torch()
        self.assertIn("pandas", sys.modules)

    def test_guard_is_a_noop_once_torch_is_loaded(self):
        """Importing pandas at that point is the crash, so the guard must refuse.

        Run in a subprocess with a stubbed torch, so the assertion is meaningful
        regardless of whether the pytest process itself has already imported the
        real torch. Doing it in-process would make the test skip whenever any
        earlier module pulled torch in — which is most of the time.
        """
        code = (
            "import sys;"
            "sys.path.insert(0, '.');"
            "sys.modules['torch'] = object();"          # pretend torch is loaded
            "sys.modules.pop('pandas', None);"
            "from utils.import_order import prefer_pandas_before_torch;"
            "prefer_pandas_before_torch();"
            "print('REFUSED' if 'pandas' not in sys.modules else 'IMPORTED_PANDAS')"
        )
        proc = _run(code)
        self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr[-400:]}")
        self.assertIn(
            "REFUSED", proc.stdout,
            "the guard imported pandas while torch was loaded, which is the "
            "operation that crashes the interpreter",
        )

    def test_describe_import_state_reports_booleans(self):
        state = describe_import_state()
        self.assertEqual(
            set(state), {"pandas_loaded", "pyarrow_loaded", "torch_loaded"}
        )
        self.assertTrue(all(isinstance(v, bool) for v in state.values()))

    def test_a_process_using_the_guard_survives_mixed_imports(self):
        """End-to-end: the guard makes a pandas + torch process work."""
        code = (
            "import sys;"
            "sys.path.insert(0, '.');"
            "from utils.import_order import prefer_pandas_before_torch;"
            "prefer_pandas_before_torch();"
            "import torch;"
            "import pandas as pd;"
            "import analysis.audit_engine;"
            "print('SURVIVED', pd.__version__)"
        )
        proc = _run(code)
        self.assertEqual(
            proc.returncode, 0,
            f"a guarded process still died. rc={proc.returncode} "
            f"stderr={proc.stderr[-400:]}",
        )
        self.assertIn("SURVIVED", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
