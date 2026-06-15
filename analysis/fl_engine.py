# ── analysis/fl_engine.py ──
"""
Federated Learning Engine
=========================
Core FL simulation: FedAvg aggregation with per-client training.
Pure PyTorch — no external FL framework dependency.

Supports:
- FedAvg (McMahan 2017)
- Configurable Non-IID data splits
- Per-round metrics tracking
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from typing import Optional
import time


class OrganoidClassifier(nn.Module):
    """MLP classifier for organoid stage classification."""

    def __init__(self, input_dim: int = 512, num_classes: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP classifier."""
        return self.net(x)


def get_params(model: nn.Module) -> OrderedDict:
    """Get model parameters as OrderedDict."""
    return OrderedDict(model.named_parameters())


def set_params(model: nn.Module, params: OrderedDict) -> None:
    """Load parameters into model."""
    model.load_state_dict(params)


def fedavg_aggregate(
    client_params: list[OrderedDict],
    client_weights: list[int] | None = None,
) -> OrderedDict:
    """FedAvg: weighted average of client parameters (McMahan 2017).

    Args:
        client_params: list of client parameter dicts.
        client_weights: number of samples per client. If None, uses
            unweighted average (assumes equal data split).
    """
    if client_weights is None:
        # Unweighted fallback (equal split)
        avg = OrderedDict()
        for key in client_params[0].keys():
            avg[key] = torch.stack([p[key].data for p in client_params]).mean(dim=0)
        return avg

    # Weighted FedAvg: w_k = |D_k| / |D|
    total = sum(client_weights)
    avg = OrderedDict()
    for key in client_params[0].keys():
        weighted_sum = torch.zeros_like(client_params[0][key].data)
        for params, w in zip(client_params, client_weights):
            weighted_sum += params[key].data * (w / total)
        avg[key] = weighted_sum
    return avg


def train_client(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.001,
    epochs: int = 2,
    batch_size: int = 32,
) -> tuple[OrderedDict, float, float]:
    """Train a single client locally.

    Returns:
        (params, train_loss, train_accuracy)
    """
    model.train()
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss, correct, total = 0.0, 0, 0
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            _, pred = torch.max(out, 1)
            correct += (pred == y_batch).sum().item()
            total += len(y_batch)

    return get_params(model), total_loss / total, correct / total


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
) -> tuple[float, float]:
    """Evaluate model on data.

    Returns:
        (loss, accuracy)
    """
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            out = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += loss.item() * len(y_batch)
            _, pred = torch.max(out, 1)
            correct += (pred == y_batch).sum().item()
            total += len(y_batch)

    return total_loss / total, correct / total


class FLEngine:
    """Federated Learning simulation engine.

    Usage:
        engine = FLEngine(input_dim=512, num_classes=3)
        history = engine.run(features, labels, n_clients=3, rounds=10)
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 3,
        hidden_dim: int = 128,
        lr: float = 0.001,
        local_epochs: int = 2,
        batch_size: int = 32,
        non_iid: float = 0.0,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.non_iid = non_iid
        self.seed = seed

    def run(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_clients: int = 3,
        rounds: int = 10,
        val_split: float = 0.2,
        progress_callback: Optional[callable] = None,
    ) -> list[dict]:
        """Run federated learning simulation.

        Args:
            progress_callback: called after each round with (round, metrics_dict)

        Returns:
            List of per-round metrics dicts
        """
        from utils.helpers import split_federated_data

        # Split data
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(len(features))
        features, labels = features[idx], labels[idx]

        val_size = int(len(features) * val_split)
        val_X, val_y = features[:val_size], labels[:val_size]
        train_X, train_y = features[val_size:], labels[val_size:]

        # Split training data across clients
        client_data = split_federated_data(
            train_X, train_y, n_clients, non_iid=self.non_iid, seed=self.seed
        )

        # Initialize global model
        global_model = OrganoidClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
        )
        global_params = get_params(global_model)

        history = []
        for rnd in range(1, rounds + 1):
            t0 = time.time()
            client_params_list = []
            client_metrics = []

            for cid, (cX, cy) in enumerate(client_data):
                client_model = OrganoidClassifier(
                    input_dim=self.input_dim,
                    num_classes=self.num_classes,
                    hidden_dim=self.hidden_dim,
                )
                set_params(client_model, global_params)
                params, train_loss, train_acc = train_client(
                    client_model, cX, cy,
                    lr=self.lr,
                    epochs=self.local_epochs,
                    batch_size=self.batch_size,
                )
                client_params_list.append(params)
                client_metrics.append({
                    "client": cid + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "n_samples": len(cX),
                })

            # Aggregate with sample weighting
            client_weights = [m["n_samples"] for m in client_metrics]
            global_params = fedavg_aggregate(client_params_list, client_weights)

            # Evaluate global model
            set_params(global_model, global_params)
            val_loss, val_acc = evaluate_model(global_model, val_X, val_y, self.batch_size)

            elapsed = time.time() - t0
            round_metrics = {
                "round": rnd,
                "avg_train_loss": np.mean([m["train_loss"] for m in client_metrics]),
                "avg_train_acc": np.mean([m["train_acc"] for m in client_metrics]),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "elapsed": elapsed,
                "client_metrics": client_metrics,
            }
            history.append(round_metrics)

            if progress_callback:
                progress_callback(rnd, round_metrics)

        # Store final model
        self.global_model = global_model
        self.global_params = global_params
        self.history = history

        return history
