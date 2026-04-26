# ── tests/test_vision_rag.py ──
"""Tests for the Vision RAG module."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.vision_rag import (
    VisionRAG,
    encode_morphology,
    MORPHOLOGY_FEATURES,
)


SAMPLE_MORPHOLOGY = {
    "area": 5000,
    "perimeter": 300,
    "circularity": 0.8,
    "solidity": 0.9,
    "aspect_ratio": 1.1,
    "eccentricity": 0.2,
    "n_organoids": 5,
    "avg_area": 4500,
    "std_area": 1000,
    "class_distribution": [0.7, 0.2, 0.1],
}


class TestEncodeMorphology:
    def test_basic_encoding(self):
        vec = encode_morphology(SAMPLE_MORPHOLOGY)
        assert isinstance(vec, np.ndarray)
        # class_distribution has 3 values, so total = 9 scalar + 3 = 12
        expected_dim = len(MORPHOLOGY_FEATURES) + 2  # +2 for extra class_distribution items
        assert len(vec) == expected_dim
        assert vec.dtype == np.float32

    def test_normalized(self):
        vec = encode_morphology(SAMPLE_MORPHOLOGY)
        norm = np.linalg.norm(vec)
        # Vector should be non-zero and finite
        assert norm > 0
        assert np.all(np.isfinite(vec))

    def test_consistency(self):
        v1 = encode_morphology(SAMPLE_MORPHOLOGY)
        v2 = encode_morphology(SAMPLE_MORPHOLOGY)
        assert np.allclose(v1, v2)

    def test_different_inputs(self):
        other = dict(SAMPLE_MORPHOLOGY)
        other["circularity"] = 0.2
        v1 = encode_morphology(SAMPLE_MORPHOLOGY)
        v2 = encode_morphology(other)
        assert not np.allclose(v1, v2)


class TestVisionRAG:
    def test_init(self):
        rag = VisionRAG()
        # feature_dim includes flattened class_distribution (3 values)
        assert rag.feature_dim > 0
        assert len(rag.cases) == 0

    def test_register_and_query(self):
        rag = VisionRAG()
        rag.register_case(
            morphology=SAMPLE_MORPHOLOGY,
            diagnosis="healthy",
            confidence=0.95,
            report="Normal morphology observed.",
            hospital_id="hospital_1",
        )

        # Query with same morphology should find the case
        results = rag.query(SAMPLE_MORPHOLOGY, k=1, min_similarity=0.0)
        assert len(results) >= 1
        case_id, similarity, case = results[0]
        assert similarity > 0.9  # Same input → very high similarity
        assert case.diagnosis == "healthy"

    def test_populate_demo(self):
        rag = VisionRAG()
        rag.populate_demo(n_cases=100, n_hospitals=4, seed=42)
        stats = rag.get_stats()
        assert stats["total_cases"] == 100
        assert stats["hospitals"] == 4
        assert "hospital_distribution" in stats
        assert "diagnosis_distribution" in stats

    def test_query_empty_db(self):
        rag = VisionRAG()
        results = rag.query(SAMPLE_MORPHOLOGY, k=3)
        assert results == []

    def test_generate_report(self):
        rag = VisionRAG()
        rag.populate_demo(n_cases=50, n_hospitals=3, seed=42)
        results = rag.query(SAMPLE_MORPHOLOGY, k=5, min_similarity=0.0)
        if results:
            report = rag.generate_report(SAMPLE_MORPHOLOGY, results)
            # Report is a dict with diagnostic summary
            assert isinstance(report, dict)
            assert len(report) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
