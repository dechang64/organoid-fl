# ── tests/test_vector_engine.py ──
"""Tests for the Vector Engine module."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.vector_engine import VectorEngine


class TestVectorEngine:
    def test_insert_and_search(self):
        engine = VectorEngine(dimension=64)
        vec = np.random.randn(64).astype(np.float32)
        engine.insert("v1", vec)

        results = engine.search(vec, k=1)
        assert len(results) == 1
        assert results[0][0] == "v1"
        assert results[0][1] > 0.99  # Self-similarity

    def test_bulk_insert(self):
        engine = VectorEngine(dimension=32)
        n = 100
        vectors = np.random.randn(n, 32).astype(np.float32)
        ids = [f"v{i}" for i in range(n)]

        count = engine.bulk_insert(ids, vectors)
        assert count == n
        assert len(engine) == n

    def test_search_k(self):
        engine = VectorEngine(dimension=16)
        for i in range(20):
            engine.insert(f"v{i}", np.random.randn(16).astype(np.float32))

        results = engine.search(np.random.randn(16).astype(np.float32), k=5)
        assert len(results) == 5
        # Results should be sorted by descending similarity
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]

    def test_delete(self):
        engine = VectorEngine(dimension=8)
        engine.insert("v1", np.random.randn(8).astype(np.float32))
        engine.insert("v2", np.random.randn(8).astype(np.float32))

        deleted = engine.delete(["v1"])
        assert deleted == 1
        assert len(engine) == 1

    def test_empty_search(self):
        engine = VectorEngine(dimension=8)
        results = engine.search(np.random.randn(8).astype(np.float32), k=5)
        assert results == []

    def test_dimension_mismatch(self):
        engine = VectorEngine(dimension=8)
        with pytest.raises(ValueError):
            engine.insert("v1", np.random.randn(16).astype(np.float32))

    def test_get_stats(self):
        engine = VectorEngine(dimension=128)
        engine.insert("v1", np.random.randn(128).astype(np.float32))
        stats = engine.get_stats()
        assert stats["total_vectors"] == 1
        assert stats["dimension"] == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
