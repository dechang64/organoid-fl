# ── analysis/pipeline/__init__.py ──
"""M1 vertical slice: the write path over the M0 contracts.

``AnnotationService`` is the only writer callers should use; ``SQLiteStore``
and ``PersistentAuditChain`` are its organs. Importing this package pulls in
the contract layer and ``analysis.audit_spec`` only — no pandas, no torch — so
a verifier process can audit a store without paying for (or risking) the
numeric stack.
"""

from analysis.pipeline.chain import ChainError, PersistentAuditChain
from analysis.pipeline.service import AnnotationService, ServiceError
from analysis.pipeline.store import SQLiteStore, StoreError

__all__ = [
    "AnnotationService",
    "ChainError",
    "PersistentAuditChain",
    "SQLiteStore",
    "ServiceError",
    "StoreError",
]
