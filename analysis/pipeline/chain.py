# ── analysis/pipeline/chain.py ──
"""Persistent audit chain for the M1 slice.

``analysis.audit_engine.AuditEngine`` is the FL engine's in-memory chain and
deliberately truncates at ``max_blocks`` — fine for federated round telemetry,
wrong for an annotation audit trail, where the point of the chain is that it
*never* forgets. This class keeps every block, on disk, and verifies the whole
sequence on every read.

The hash itself is not re-implemented here. Each block is hashed by
:func:`analysis.audit_spec.canonical_hash`, the function the shared fixture
``tests/fixtures/audit_hash_vectors.json`` proves is byte-identical in Python,
Rust and JavaScript. M1's third exit criterion ("跨语言哈希一致") is therefore
inherited from the spec, not re-argued here; the chain-level vectors added to
that fixture additionally pin the *linkage* (each block's ``prev_hash`` being
the previous block's digest).
"""

from __future__ import annotations

import json
import time
from itertools import pairwise
from typing import Any, Optional

from analysis.audit_spec import SPEC_VERSION, canonical_hash


class ChainError(ValueError):
    """The chain on disk is not the chain that was written."""


class PersistentAuditChain:
    """Append-only, restart-surviving audit chain backed by ``SQLiteStore``."""

    def __init__(self, store) -> None:
        self._store = store
        self._ensure_genesis()

    # ── writing ──────────────────────────────────────────────────────

    def _ensure_genesis(self) -> None:
        if self.head() is not None:
            return
        genesis = {
            "index": 0,
            "timestamp": _utc_now(),
            "operation": "genesis",
            "details": {"message": "annotation pipeline audit chain initialised"},
            "prev_hash": "0",
            "nonce": 0,
        }
        genesis["hash"] = canonical_hash(
            index=genesis["index"],
            timestamp=genesis["timestamp"],
            operation=genesis["operation"],
            details=genesis["details"],
            prev_hash=genesis["prev_hash"],
            nonce=genesis["nonce"],
        )
        self._write(genesis)

    def append(
        self,
        operation: str,
        details: dict[str, Any],
        *,
        timestamp: Optional[str] = None,
        nonce: int = 0,
    ) -> dict[str, Any]:
        """Append one block; returns the stored block.

        ``timestamp`` is injectable so a replay or a test can produce a fully
        deterministic chain — the property the cross-language chain vectors
        rely on. Production callers omit it and get wall-clock time.

        Details must survive :func:`analysis.audit_spec.validate_details`,
        which runs inside ``canonical_hash``: a float outside the canonical
        domain raises rather than entering the chain, because a digest that
        depends on the writing language is not evidence.
        """
        head = self.head()
        if head is None:  # pragma: no cover - genesis is created in __init__
            raise ChainError("chain has no genesis block")
        block = {
            "index": head["index"] + 1,
            "timestamp": timestamp or _utc_now(),
            "operation": operation,
            "details": details,
            "prev_hash": head["hash"],
            "nonce": nonce,
        }
        block["hash"] = canonical_hash(
            index=block["index"],
            timestamp=block["timestamp"],
            operation=block["operation"],
            details=block["details"],
            prev_hash=block["prev_hash"],
            nonce=block["nonce"],
        )
        self._write(block)
        return block

    def _write(self, block: dict[str, Any]) -> None:
        self._store._conn.execute(
            "INSERT INTO audit_blocks "
            "(index_, timestamp, operation, details, prev_hash, nonce, hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                block["index"],
                block["timestamp"],
                block["operation"],
                json.dumps(block["details"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                block["prev_hash"],
                block["nonce"],
                block["hash"],
            ),
        )
        self._store._conn.commit()

    # ── reading ──────────────────────────────────────────────────────

    def head(self) -> Optional[dict[str, Any]]:
        row = self._store._conn.execute(
            "SELECT * FROM audit_blocks ORDER BY index_ DESC LIMIT 1"
        ).fetchone()
        return _row_to_block(row) if row is not None else None

    def blocks(self) -> list[dict[str, Any]]:
        rows = self._store._conn.execute(
            "SELECT * FROM audit_blocks ORDER BY index_ ASC"
        ).fetchall()
        return [_row_to_block(row) for row in rows]

    def verify(self) -> dict[str, Any]:
        """Recompute every digest and every linkage, from genesis.

        Returns a report rather than a bare bool so a failure says *which*
        block broke and whether it is a linkage break (a block was removed or
        reordered) or a digest break (a block was edited in place).
        """
        blocks = self.blocks()
        if not blocks:
            return {"length": 0, "valid": False, "reason": "empty chain"}
        problems: list[str] = []
        if blocks[0]["prev_hash"] != "0":
            problems.append(f"block 0 prev_hash is {blocks[0]['prev_hash'][:12]}…, expected '0'")
        for previous, block in pairwise(blocks):
            if block["prev_hash"] != previous["hash"]:
                problems.append(
                    f"block {block['index']} prev_hash does not match block "
                    f"{previous['index']} digest (linkage break)"
                )
        for block in blocks:
            expected = canonical_hash(
                index=block["index"],
                timestamp=block["timestamp"],
                operation=block["operation"],
                details=block["details"],
                prev_hash=block["prev_hash"],
                nonce=block["nonce"],
            )
            if block["hash"] != expected:
                problems.append(
                    f"block {block['index']} digest mismatch: stored "
                    f"{block['hash'][:12]}…, recomputed {expected[:12]}…"
                )
        return {
            "length": len(blocks),
            "valid": not problems,
            "head_hash": blocks[-1]["hash"],
            "spec_version": SPEC_VERSION,
            "problems": problems,
        }

    def operations(self) -> list[str]:
        return [block["operation"] for block in self.blocks()]


def _row_to_block(row: Any) -> dict[str, Any]:
    return {
        "index": int(row["index_"]),
        "timestamp": row["timestamp"],
        "operation": row["operation"],
        "details": json.loads(row["details"]),
        "prev_hash": row["prev_hash"],
        "nonce": int(row["nonce"]),
        "hash": row["hash"],
    }


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
