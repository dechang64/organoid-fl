# ── tests/test_audit_engine.py ──
"""Tests for the Audit Engine module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.audit_engine import AuditEngine, AuditBlock


class TestAuditBlock:
    def test_compute_hash(self):
        block = AuditBlock(
            index=0,
            timestamp="2026-01-01T00:00:00Z",
            operation="test",
            details={"key": "value"},
            prev_hash="0",
        )
        h = block.compute_hash()
        assert len(h) == 64  # SHA-256 hex
        assert h != ""

    def test_verify(self):
        block = AuditBlock(
            index=0,
            timestamp="2026-01-01T00:00:00Z",
            operation="test",
            details={"key": "value"},
            prev_hash="0",
        )
        block.hash = block.compute_hash()
        assert block.verify()

    def test_tamper_detection(self):
        block = AuditBlock(
            index=0,
            timestamp="2026-01-01T00:00:00Z",
            operation="test",
            details={"key": "value"},
            prev_hash="0",
        )
        block.hash = block.compute_hash()
        block.details = {"key": "tampered"}
        assert not block.verify()


class TestAuditEngine:
    def test_genesis(self):
        engine = AuditEngine()
        assert len(engine) == 1
        assert engine.chain[0].operation == "genesis"

    def test_append(self):
        engine = AuditEngine()
        engine.append("insert", {"vectors": 10})
        assert len(engine) == 2

    def test_chain_integrity(self):
        engine = AuditEngine()
        engine.append("insert", {"vectors": 10})
        engine.append("search", {"k": 5, "results": 3})
        engine.append("delete", {"count": 2})
        assert engine.verify_chain()

    def test_recent(self):
        engine = AuditEngine()
        for i in range(10):
            engine.append("insert", {"count": i})
        recent = engine.recent(3)
        assert len(recent) == 3
        assert recent[-1].index == 10  # Most recent (genesis=0, then 1-10)

    def test_max_blocks_eviction(self):
        engine = AuditEngine(max_blocks=5)
        for i in range(10):
            engine.append("insert", {"count": i})
        assert len(engine) <= 5

    def test_get_stats(self):
        engine = AuditEngine()
        engine.append("insert", {"vectors": 5})
        stats = engine.get_stats()
        assert stats["chain_length"] == 2
        assert stats["chain_valid"] is True

    def test_to_dataframe(self):
        engine = AuditEngine()
        engine.append("insert", {"vectors": 10})
        df = engine.to_dataframe()
        assert len(df) == 2
        assert "Block" in df.columns
        assert "Operation" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
