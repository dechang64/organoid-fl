# ── analysis/pipeline/__init__.py ──
"""The write path and the projected read path over the M0 contracts.

``AnnotationService`` is the only writer callers should use; ``SQLiteStore`` and
``PersistentAuditChain`` are its organs. Importing this package pulls in the
contract layer, ``analysis.audit_spec`` and **numpy** (the geometry layer needs
it) — no pandas, no pyarrow, no torch. That matters because of this repository's
import-order guard: numpy before torch is harmless, pandas before torch is not,
and a verifier process should not have to load the numeric stack just to audit a
store.
"""

from analysis.pipeline.chain import ChainError, PersistentAuditChain
from analysis.pipeline.disagreement import (
    METRICS_VERSION,
    DisagreementComputation,
    DisagreementError,
    compute_disagreement,
)
from analysis.pipeline.geometry import (
    GEOMETRY_FORMAT,
    CoderPayload,
    GeometryError,
    match_payloads,
)
from analysis.pipeline.service import AnnotationService, ServiceError
from analysis.pipeline.store import SQLiteStore, StoreError
from analysis.pipeline.stratum import StratumRuleError, label_stratum

__all__ = [
    "GEOMETRY_FORMAT",
    "METRICS_VERSION",
    "AnnotationService",
    "ChainError",
    "CoderPayload",
    "DisagreementComputation",
    "DisagreementError",
    "GeometryError",
    "PersistentAuditChain",
    "SQLiteStore",
    "ServiceError",
    "StoreError",
    "StratumRuleError",
    "compute_disagreement",
    "label_stratum",
    "match_payloads",
]
