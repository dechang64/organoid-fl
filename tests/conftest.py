# ── tests/conftest.py ──
"""pytest bootstrap.

pytest imports ``conftest.py`` before collecting test modules, so this is the
earliest hook available for forcing pandas ahead of torch. ``tests/__init__.py``
does the same job for ``unittest`` discovery; both are needed because the two
runners have different load orders.

Reason: importing pyarrow after torch crashes the interpreter on Windows with
an access violation (no traceback, no catchable exception). pandas 2.x imports
pyarrow at module scope, so a late ``import pandas`` — such as the one in
``analysis/audit_engine.to_dataframe`` — is enough to kill the whole run.
See ``utils/import_order.py`` for the measurement.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.import_order import prefer_pandas_before_torch  # noqa: E402

prefer_pandas_before_torch()
