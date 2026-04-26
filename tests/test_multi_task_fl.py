# ── tests/test_multi_task_fl.py ──
"""Tests for the Multi-Task FL Engine module."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.multi_task_fl import MultiTaskFLEngine


class TestMultiTaskFLEngine:
    def test_init(self):
        engine = MultiTaskFLEngine(
            input_dim=64,
            num_classes=2,
            hidden_dim=16,
            lr=0.01,
            local_epochs=1,
        )
        assert engine.input_dim == 64
        assert engine.num_classes == 2

    def test_basic_training(self):
        engine = MultiTaskFLEngine(
            input_dim=64,
            num_classes=3,
            hidden_dim=16,
            lr=0.01,
            local_epochs=1,
        )
        features = np.random.randn(300, 64).astype(np.float32)
        labels = np.random.randint(0, 3, 300).astype(np.int64)

        history = engine.run(
            features, labels,
            n_clients=3,
            rounds=3,
        )

        assert len(history) == 3
        assert all("val_acc" in h for h in history)
        assert all("val_loss" in h for h in history)
        assert all("client_metrics" in h for h in history)
        # Accuracy should be non-trivial
        assert history[-1]["val_acc"] > 0.1

    def test_different_client_counts(self):
        engine = MultiTaskFLEngine(
            input_dim=32,
            num_classes=2,
            hidden_dim=8,
            lr=0.01,
            local_epochs=1,
        )
        features = np.random.randn(200, 32).astype(np.float32)
        labels = np.random.randint(0, 2, 200).astype(np.int64)

        history = engine.run(
            features, labels,
            n_clients=5,
            rounds=2,
        )

        assert len(history) == 2
        assert len(history[0]["client_metrics"]) == 5

    def test_classifier_head_dimensions(self):
        engine = MultiTaskFLEngine(input_dim=128, num_classes=5, hidden_dim=32)
        import torch
        x = torch.randn(2, 128)
        out = engine.classifier(x)
        assert out.shape == (2, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
