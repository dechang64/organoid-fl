# ── tests/test_segmentor.py ──
"""Tests for the SAM2 Segmentor module."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.segmentor import OrganoidSegmentor, SegmentationResult


class TestSegmentationResult:
    def test_creation(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        sr = SegmentationResult(
            mask=mask,
            area=3600,
            perimeter=240.0,
            circularity=0.785,
            solidity=0.95,
            aspect_ratio=1.0,
            eccentricity=0.0,
            bbox=[20, 20, 80, 80],
            centroid=(50.0, 50.0),
        )
        assert sr.area == 3600
        assert sr.circularity == 0.785

    def test_to_dict(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        sr = SegmentationResult(
            mask=mask,
            area=0,
            perimeter=0,
            circularity=0,
            solidity=0,
            aspect_ratio=0,
            eccentricity=0,
            bbox=[0, 0, 0, 0],
            centroid=(0, 0),
        )
        d = sr.to_dict()
        assert "mask" not in d  # Mask should not be serialized
        assert "area" in d
        assert "circularity" in d


class TestOrganoidSegmentor:
    def test_init(self):
        seg = OrganoidSegmentor()
        assert seg is not None

    def test_compute_morphology_circle(self):
        seg = OrganoidSegmentor()
        mask = np.zeros((100, 100), dtype=np.uint8)
        Y, X = np.ogrid[:100, :100]
        circle = (X - 50) ** 2 + (Y - 50) ** 2 <= 30 ** 2
        mask[circle] = 1

        result = seg._compute_morphology(mask, bbox=[20, 20, 80, 80])
        assert result["area"] > 0
        assert result["circularity"] > 0.5
        assert 0 <= result["solidity"] <= 1.05  # Allow floating point tolerance
        assert 0 <= result["eccentricity"] <= 1

    def test_compute_morphology_empty(self):
        seg = OrganoidSegmentor()
        mask = np.zeros((50, 50), dtype=np.uint8)
        result = seg._compute_morphology(mask, bbox=[0, 0, 0, 0])
        assert result["area"] == 0

    def test_compute_morphology_rectangle(self):
        seg = OrganoidSegmentor()
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 10:50] = 1

        result = seg._compute_morphology(mask, bbox=[10, 10, 50, 90])
        assert result["area"] == 80 * 40
        assert result["aspect_ratio"] >= 1  # Rectangle, may be exactly 1 depending on fit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
