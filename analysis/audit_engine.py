# ── analysis/audit_engine.py ──
"""
Audit Chain Engine
==================
Python implementation of the audit chain, sharing one canonical hash
specification with the Rust implementation in ``src/blockchain.rs``.
Records FL operations with SHA-256 hashing for tamper evidence.

The preimage format lives in :mod:`analysis.audit_spec` and is the single
source of truth for both languages. Before that module existed the two sides
hashed different byte strings — Python used sorted-key JSON, Rust used a
colon-joined ``Debug`` rendering of an enum — so a chain written on one side
could not be verified on the other.
"""

import json
import time
from dataclasses import dataclass
from typing import Optional

from analysis.audit_spec import (
    SPEC_VERSION,
    canonical_hash,
    canonical_preimage,
    legacy_hash,
)


@dataclass
class AuditBlock:
    """A single block in the audit chain."""

    index: int
    timestamp: str
    operation: str
    details: dict
    prev_hash: str
    hash: str = ""
    #: Optional proof-of-work counter. Present because the Rust block struct has
    #: one and it takes part in the digest, so omitting it here would make the
    #: two sides disagree again.
    nonce: int = 0
    #: Which preimage rule produced :attr:`hash`.
    spec_version: str = SPEC_VERSION

    def compute_hash(self) -> str:
        """Compute the canonical SHA-256 hash of this block."""
        return canonical_hash(
            index=self.index,
            timestamp=self.timestamp,
            operation=self.operation,
            details=self.details,
            prev_hash=self.prev_hash,
            nonce=self.nonce,
        )

    def preimage(self) -> bytes:
        """The exact bytes hashed. Exposed so two implementations can be
        compared field by field when a digest disagrees."""
        return canonical_preimage(
            index=self.index,
            timestamp=self.timestamp,
            operation=self.operation,
            details=self.details,
            prev_hash=self.prev_hash,
            nonce=self.nonce,
        )

    def verify(self, allow_legacy: bool = True) -> bool:
        """Verify this block's hash integrity.

        Args:
            allow_legacy: also accept a digest produced by the pre-v1 Python
                formula, so chains written before the spec was unified remain
                verifiable. Verified-with-legacy should be treated as "valid but
                written by an older implementation", not as equivalent.
        """
        if self.hash == self.compute_hash():
            return True
        if allow_legacy:
            return self.hash == legacy_hash(
                self.index, self.timestamp, self.operation, self.details,
                self.prev_hash,
            )
        return False

    def uses_current_spec(self) -> bool:
        """True when this block's digest was produced by the canonical rule."""
        return self.hash == self.compute_hash()


class AuditEngine:
    """Python audit chain for FL operations."""

    def __init__(self, max_blocks: int = 1000):
        self.max_blocks = max_blocks
        self.chain: list[AuditBlock] = []
        self._create_genesis()

    def _create_genesis(self) -> None:
        genesis = AuditBlock(
            index=0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            operation="genesis",
            details={"message": "Organoid-FL Audit Chain initialized"},
            prev_hash="0",
        )
        genesis.hash = genesis.compute_hash()
        self.chain.append(genesis)

    def append(
        self, operation: str, details: Optional[dict] = None, nonce: int = 0
    ) -> AuditBlock:
        """Append a new block to the chain.

        Raises:
            SpecError: when ``details`` cannot be canonicalised identically in
                both languages (a float outside the safe domain, or a
                non-JSON type). Failing here is deliberate: a block whose digest
                depends on the writing language cannot be verified by the other
                implementation, so it must not enter the chain at all.
        """
        prev_block = self.chain[-1]
        block = AuditBlock(
            index=prev_block.index + 1,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            operation=operation,
            details=details or {},
            prev_hash=prev_block.hash,
            nonce=nonce,
        )
        block.hash = block.compute_hash()
        self.chain.append(block)

        # Evict old blocks if over limit
        if len(self.chain) > self.max_blocks:
            self.chain = self.chain[-self.max_blocks:]

        return block

    def verify_chain(self) -> bool:
        """Verify entire chain integrity."""
        for i in range(1, len(self.chain)):
            if self.chain[i].prev_hash != self.chain[i - 1].hash:
                return False
            if not self.chain[i].verify():
                return False
        return True

    def audit_spec_versions(self) -> dict:
        """Count blocks by which preimage rule produced their digest.

        A chain can legitimately contain both: blocks written before the spec
        was unified keep their legacy digests, and rewriting them would destroy
        the tamper evidence they exist to provide.
        """
        current = sum(1 for b in self.chain if b.uses_current_spec())
        return {
            "current_spec": current,
            "legacy_spec": len(self.chain) - current,
            "expected_spec": SPEC_VERSION,
        }

    def recent(self, n: int = 10) -> list[AuditBlock]:
        """Get the most recent n blocks."""
        return self.chain[-n:]

    def __len__(self) -> int:
        return len(self.chain)

    def get_stats(self) -> dict:
        """Return audit chain statistics."""
        return {
            "chain_length": len(self),
            "chain_valid": self.verify_chain(),
            "latest_hash": self.chain[-1].hash[:16] + "..." if self.chain else "",
            **self.audit_spec_versions(),
        }

    def to_dataframe(self):
        """Export chain as a DataFrame for display.

        The pandas import is lazy so that a process which only needs the hash
        chain never pays for it. That is only safe because the caller has
        already loaded pandas *before* torch — pandas 2.x pulls pyarrow in at
        module scope, and a late ``import pyarrow`` after torch is an access
        violation on Windows with no traceback. This function is exactly what
        used to kill the whole pytest run.

        Callers that mix pandas and torch must therefore either import pandas
        first or call :func:`utils.import_order.prefer_pandas_before_torch`.
        """
        from utils.import_order import describe_import_state

        state = describe_import_state()
        if state["torch_loaded"] and not state["pandas_loaded"]:
            raise RuntimeError(
                "to_dataframe() needs pandas, but torch is already loaded in "
                "this process. Importing pandas now would import pyarrow and "
                "crash the interpreter with no traceback on Windows. Import "
                "pandas before torch, or call "
                "utils.import_order.prefer_pandas_before_torch() at startup, "
                "or use the dict/list export instead of a DataFrame."
            )

        import pandas as pd
        rows = []
        for b in self.chain:
            rows.append({
                "Block": b.index,
                "Time": b.timestamp,
                "Operation": b.operation,
                "Details": json.dumps(b.details, ensure_ascii=False)[:80],
                "Hash": b.hash[:16] + "...",
                "Prev": b.prev_hash[:16] + "...",
            })
        return pd.DataFrame(rows)
