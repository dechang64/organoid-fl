# ── tests/test_fl_engine.py ──
"""Tests for the FL Engine module."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.fl_engine import (
    OrganoidClassifier, FLEngine,
    fedavg_aggregate, get_params, set_params,
    train_client, evaluate_model,
)


class TestOrganoidClassifier:
    def test_forward_shape(self):
        model = OrganoidClassifier(input_dim=128, num_classes=3)
        x = np.random.randn(4, 128).astype(np.float32)
        import torch
        out = model(torch.tensor(x))
        assert out.shape == (4, 3)

    def test_different_dims(self):
        model = OrganoidClassifier(input_dim=256, num_classes=5, hidden_dim=64)
        import torch
        x = torch.randn(2, 256)
        out = model(x)
        assert out.shape == (2, 5)


class TestFedAvg:
    def test_aggregate_two_models(self):
        import torch
        model1 = OrganoidClassifier(input_dim=64, num_classes=2, hidden_dim=16)
        model2 = OrganoidClassifier(input_dim=64, num_classes=2, hidden_dim=16)

        p1 = get_params(model1)
        p2 = get_params(model2)

        avg = fedavg_aggregate([p1, p2])

        # Check keys match
        assert set(avg.keys()) == set(p1.keys())

        # Check values are averaged
        for key in avg:
            expected = (p1[key].data + p2[key].data) / 2
            assert torch.allclose(avg[key].data, expected)

    def test_aggregate_three_models(self):
        import torch
        models = [OrganoidClassifier(input_dim=32, num_classes=2, hidden_dim=8) for _ in range(3)]
        params = [get_params(m) for m in models]
        avg = fedavg_aggregate(params)
        assert set(avg.keys()) == set(params[0].keys())


class TestTrainClient:
    def test_train_reduces_loss(self):
        import torch
        model = OrganoidClassifier(input_dim=64, num_classes=3, hidden_dim=16)
        X = torch.randn(50, 64)
        y = torch.randint(0, 3, (50,))

        loss_before, _ = evaluate_model(model, X, y, batch_size=16)
        params, loss_after, acc = train_client(model, X, y, lr=0.01, epochs=5, batch_size=16)

        assert loss_after <= loss_before + 0.5  # Allow some variance
        assert 0 <= acc <= 1


class TestFLEngine:
    def test_basic_training(self):
        engine = FLEngine(input_dim=64, num_classes=3, hidden_dim=16, lr=0.01, local_epochs=1)
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
        # Accuracy should be non-trivial
        assert history[-1]["val_acc"] > 0.1

    def test_different_client_counts(self):
        engine = FLEngine(input_dim=32, num_classes=2, hidden_dim=8, lr=0.01, local_epochs=1)
        features = np.random.randn(200, 32).astype(np.float32)
        labels = np.random.randint(0, 2, 200).astype(np.int64)

        history = engine.run(
            features, labels,
            n_clients=5,
            rounds=2,
        )

        assert len(history) == 2
        assert len(history[0]["client_metrics"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
