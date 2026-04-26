# ── analysis/audit_engine.py ──
"""
Audit Chain Engine
==================
Python simulation of the Rust blockchain audit chain.
Records FL operations with SHA-256 hashing for tamper evidence.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AuditBlock:
    """A single block in the audit chain."""
    index: int
    timestamp: str
    operation: str
    details: dict
    prev_hash: str
    hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of block contents."""
        data = {
            "index": self.index,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "details": self.details,
            "prev_hash": self.prev_hash,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def verify(self) -> bool:
        """Verify block hash integrity."""
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

    def append(self, operation: str, details: Optional[dict] = None) -> AuditBlock:
        """Append a new block to the chain."""
        prev_block = self.chain[-1]
        block = AuditBlock(
            index=prev_block.index + 1,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            operation=operation,
            details=details or {},
            prev_hash=prev_block.hash,
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

    def recent(self, n: int = 10) -> list[AuditBlock]:
        """Get the most recent n blocks."""
        return self.chain[-n:]

    def __len__(self) -> int:
        return len(self.chain)

    def get_stats(self) -> dict:
        return {
            "chain_length": len(self),
            "chain_valid": self.verify_chain(),
            "latest_hash": self.chain[-1].hash[:16] + "..." if self.chain else "",
        }

    def to_dataframe(self):
        """Export chain as list of dicts for display."""
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
