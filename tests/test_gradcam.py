# ── tests/test_gradcam.py ──
"""Tests for the Grad-CAM explainability module."""

import sys
import os
import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.gradcam import GradCAM, generate_explanation_report


class DummyConvModel(nn.Module):
    """Simple CNN for testing Grad-CAM."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TestGradCAM:
    def test_init_with_target_layer(self):
        model = DummyConvModel()
        cam = GradCAM(model, target_layer=model.conv2)
        assert cam.target_layer == model.conv2

    def test_generate_heatmap(self):
        model = DummyConvModel()
        model.eval()
        cam = GradCAM(model, target_layer=model.conv2)

        x = torch.randn(1, 3, 224, 224)
        heatmap = cam.generate(x, target_class=0)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2
        assert heatmap.shape[0] == 224
        assert heatmap.shape[1] == 224
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1

    def test_overlay(self):
        model = DummyConvModel()
        model.eval()
        cam = GradCAM(model, target_layer=model.conv2)

        x = torch.randn(1, 3, 224, 224)
        heatmap = cam.generate(x, target_class=0)
        # overlay(image, heatmap) — image first, heatmap second
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        overlay = cam.overlay(image, heatmap)
        assert overlay.shape == (224, 224, 3)

    def test_different_target_classes(self):
        model = DummyConvModel()
        model.eval()
        cam = GradCAM(model, target_layer=model.conv2)

        x = torch.randn(1, 3, 224, 224)
        h0 = cam.generate(x, target_class=0)
        h1 = cam.generate(x, target_class=1)
        h2 = cam.generate(x, target_class=2)

        # Different classes should produce different heatmaps
        assert not np.allclose(h0, h1) or not np.allclose(h1, h2)


class TestExplanationReport:
    def test_healthy_report(self):
        morphology = {
            "circularity": 0.85,
            "eccentricity": 0.15,
            "solidity": 0.92,
        }
        heatmap = np.random.rand(224, 224).astype(np.float32)
        report = generate_explanation_report(
            heatmap=heatmap,
            target_class="healthy",
            class_names=["healthy", "early_stage", "late_stage"],
            confidence=0.95,
            morphology=morphology,
        )
        assert "healthy" in report.lower()
        assert "circularity" in report.lower()

    def test_late_stage_report(self):
        morphology = {
            "circularity": 0.25,
            "eccentricity": 0.85,
            "solidity": 0.55,
        }
        heatmap = np.random.rand(224, 224).astype(np.float32)
        report = generate_explanation_report(
            heatmap=heatmap,
            target_class="late_stage",
            class_names=["healthy", "early_stage", "late_stage"],
            confidence=0.88,
            morphology=morphology,
        )
        assert "late_stage" in report.lower()
        assert "irregular" in report.lower()

    def test_no_morphology(self):
        heatmap = np.random.rand(224, 224).astype(np.float32)
        report = generate_explanation_report(
            heatmap=heatmap,
            target_class="early_stage",
            class_names=["healthy", "early_stage", "late_stage"],
            confidence=0.75,
            morphology=None,
        )
        assert "early_stage" in report.lower()
        assert "Morphological Evidence" not in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
