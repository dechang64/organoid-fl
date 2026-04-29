# ── analysis/multi_task_fl.py ──
"""
Multi-Task Federated Learning Engine
=====================================
Federated learning with multiple vision models running in parallel.

Architecture:
  Client-side:
    1. YOLOv11  → detection weights (trainable)
    2. DINOv2   → frozen backbone, train classifier head
    3. SAM2     → frozen backbone, train prompt encoder

  Server-side (FedAvg):
    1. Average YOLO detection head + FPN weights
    2. Average classifier head weights
    3. Average SAM2 prompt encoder weights

This enables each hospital to contribute to detection, classification,
AND segmentation simultaneously — without sharing any patient images.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from typing import Optional, Callable
import time


class MultiTaskFLEngine:
    """Federated learning engine with detection + classification + segmentation.

    Each client trains locally on its own data. Only model weights
    (gradients) are shared with the server for aggregation.
    """

    def __init__(
        self,
        input_dim: int = 768,       # DINOv2 base output dim
        num_classes: int = 3,
        hidden_dim: int = 128,
        lr: float = 0.001,
        local_epochs: int = 2,
        batch_size: int = 32,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.seed = seed

        # Classification head (sits on top of DINOv2)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        # Task-specific state
        self.detector_weights: Optional[OrderedDict] = None
        self.segmentor_weights: Optional[OrderedDict] = None
        self.history: list[dict] = []

    def _split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_clients: int,
        val_split: float = 0.2,
    ) -> tuple:
        """Split data across clients with validation set."""
        rng = np.random.RandomState(self.seed)
        n = len(features)
        indices = rng.permutation(n)

        # Validation split (from last portion)
        val_size = int(n * val_split)
        val_idx = indices[-val_size:]
        train_idx = indices[:-val_size]

        # Split training data across clients (relative indices)
        n_train = len(train_idx)
        chunk_size = n_train // n_clients
        client_splits = []
        for i in range(n_clients):
            start = i * chunk_size
            end = start + chunk_size if i < n_clients - 1 else n_train
            client_splits.append(np.arange(start, end))

        return (
            features[train_idx], labels[train_idx],
            features[val_idx], labels[val_idx],
            client_splits,
        )

    def _train_classifier_client(
        self,
        client_model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[OrderedDict, float, float]:
        """Train classification head on one client's features."""
        optimizer = optim.Adam(client_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        client_model.train()
        total_loss, correct, total = 0.0, 0, 0

        for _ in range(self.local_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                out = client_model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * len(xb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(xb)

        params = OrderedDict(client_model.state_dict())
        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return params, avg_loss, accuracy

    @staticmethod
    def _evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 64) -> tuple[float, float]:
        """Evaluate model accuracy and loss."""
        model.eval()
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size)

        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                out = model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item() * len(xb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(xb)

        return total_loss / max(total, 1), correct / max(total, 1)

    @staticmethod
    def _fedavg(params_list: list[OrderedDict]) -> OrderedDict:
        """FedAvg: average parameters across clients."""
        avg = OrderedDict()
        for key in params_list[0]:
            avg[key] = torch.stack([p[key] for p in params_list]).mean(dim=0)
        return avg

    def run(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_clients: int = 3,
        rounds: int = 10,
        val_split: float = 0.2,
        progress_callback: Optional[Callable] = None,
        # Optional: pass pre-computed detector/segmentor weights per client
        detector_weights_fn: Optional[Callable] = None,
        segmentor_weights_fn: Optional[Callable] = None,
    ) -> list[dict]:
        """Run multi-task federated learning.

        Args:
            features: (N, input_dim) feature vectors
            labels: (N,) class labels
            n_clients: number of hospitals/clients
            rounds: FL communication rounds
            val_split: fraction of data for validation
            progress_callback: optional callback(round, metrics)
            detector_weights_fn: callable(client_id) → detector weights dict
            segmentor_weights_fn: callable(client_id) → segmentor weights dict

        Returns:
            Training history list
        """
        train_X, train_y, val_X, val_y, client_splits = self._split_data(
            features, labels, n_clients, val_split
        )

        train_X_t = torch.tensor(train_X, dtype=torch.float32)
        train_y_t = torch.tensor(train_y, dtype=torch.long)
        val_X_t = torch.tensor(val_X, dtype=torch.float32)
        val_y_t = torch.tensor(val_y, dtype=torch.long)

        history = []
        t_total_start = time.time()

        for rnd in range(rounds):
            t0 = time.time()

            # ── Per-client training ──
            client_params_list = []
            client_metrics = []
            det_weights_list = []
            seg_weights_list = []

            for cid, split_idx in enumerate(client_splits):
                idx_tensor = torch.tensor(split_idx, dtype=torch.long)
                cX = train_X_t[idx_tensor]
                cy = train_y_t[idx_tensor]

                # Clone classifier head
                client_model = nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(self.hidden_dim, self.num_classes),
                )
                # Load current global weights
                client_model.load_state_dict(OrderedDict(self.classifier.state_dict()))

                params, train_loss, train_acc = self._train_classifier_client(
                    client_model, cX, cy
                )
                client_params_list.append(params)
                client_metrics.append({
                    "client": cid + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                })

                # Collect detector/segmentor weights if available
                if detector_weights_fn:
                    dw = detector_weights_fn(cid)
                    if dw:
                        det_weights_list.append(dw)
                if segmentor_weights_fn:
                    sw = segmentor_weights_fn(cid)
                    if sw:
                        seg_weights_list.append(sw)

            # ── FedAvg aggregation ──
            global_cls_params = self._fedavg(client_params_list)
            self.classifier.load_state_dict(global_cls_params)

            # Aggregate detector weights
            if det_weights_list:
                self.detector_weights = self._fedavg(det_weights_list)

            # Aggregate segmentor weights
            if seg_weights_list:
                self.segmentor_weights = self._fedavg(seg_weights_list)

            # ── Evaluate ──
            val_loss, val_acc = self._evaluate(self.classifier, val_X_t, val_y_t, self.batch_size)
            elapsed = time.time() - t0

            round_metrics = {
                "round": rnd,
                "avg_train_loss": np.mean([m["train_loss"] for m in client_metrics]),
                "avg_train_acc": np.mean([m["train_acc"] for m in client_metrics]),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "elapsed": elapsed,
                "client_metrics": client_metrics,
                "detector_aggregated": len(det_weights_list) > 0,
                "segmentor_aggregated": len(seg_weights_list) > 0,
            }
            history.append(round_metrics)

            if progress_callback:
                progress_callback(rnd, round_metrics)

        self.history = history
        return history
