# ── tests/test_detector.py ──
"""Tests for the YOLOv11 Detector module."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.detector import OrganoidDetector, Detection


class TestDetection:
    def test_creation(self):
        d = Detection(
            bbox=[10, 20, 100, 200],
            class_name="healthy",
            class_id=0,
            confidence=0.95,
            cx=55, cy=110,
            width=90, height=180,
            area=16200,
        )
        assert d.class_name == "healthy"
        assert d.confidence == 0.95
        assert d.area == 16200

    def test_to_dict(self):
        d = Detection(
            bbox=[0, 0, 50, 50],
            class_name="early_stage",
            class_id=1,
            confidence=0.88,
            cx=25, cy=25,
            width=50, height=50,
            area=2500,
        )
        d_dict = d.to_dict()
        assert d_dict["class_name"] == "early_stage"
        assert d_dict["confidence"] == 0.88
        assert "bbox" in d_dict


class TestOrganoidDetector:
    def test_init(self):
        det = OrganoidDetector(model_size="n")
        assert det.model_size == "n"
        assert det.CLASS_NAMES == ["healthy", "early_stage", "late_stage"]

    def test_count_by_class(self):
        det = OrganoidDetector(model_size="n")
        detections = [
            Detection([0,0,50,50], "healthy", 0, 0.9, 25, 25, 50, 50, 2500),
            Detection([0,0,50,50], "healthy", 0, 0.8, 25, 25, 50, 50, 2500),
            Detection([0,0,50,50], "late_stage", 2, 0.7, 25, 25, 50, 50, 2500),
        ]
        counts = det.count_by_class(detections)
        assert counts == {"healthy": 2, "late_stage": 1}

    def test_summary_empty(self):
        det = OrganoidDetector(model_size="n")
        summary = det.summary([])
        assert summary["total"] == 0
        assert summary["avg_confidence"] == 0

    def test_summary_nonempty(self):
        det = OrganoidDetector(model_size="n")
        detections = [
            Detection([0,0,50,50], "healthy", 0, 0.9, 25, 25, 50, 50, 2500),
            Detection([0,0,100,100], "early_stage", 1, 0.7, 50, 50, 100, 100, 10000),
        ]
        summary = det.summary(detections)
        assert summary["total"] == 2
        assert summary["avg_confidence"] == 0.8
        assert summary["avg_area"] == 6250
        assert summary["min_area"] == 2500
        assert summary["max_area"] == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
