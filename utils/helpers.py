# ── utils/helpers.py ──
"""Utility functions for Organoid-FL platform."""

import numpy as np
import pandas as pd
from typing import Optional


def generate_synthetic_features(
    n_samples: int = 600,
    dim: int = 512,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic organoid features for demo purposes.

    Returns:
        (features, labels, class_names)
    """
    rng = np.random.RandomState(seed)
    class_names = ["healthy", "early_stage", "late_stage"][:n_classes]

    # Create cluster centers with separation
    centers = rng.randn(n_classes, dim).astype(np.float32) * 2.0
    features_list, labels_list = [], []

    samples_per_class = n_samples // n_classes
    for i, cls in enumerate(class_names):
        cls_features = centers[i] + rng.randn(samples_per_class, dim).astype(np.float32) * 0.8
        features_list.append(cls_features)
        labels_list.append(np.full(samples_per_class, i, dtype=np.int64))

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Shuffle
    idx = rng.permutation(len(features))
    return features[idx], labels[idx], class_names


def split_federated_data(
    features: np.ndarray,
    labels: np.ndarray,
    n_clients: int = 3,
    non_iid: float = 0.0,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split data across clients, optionally with Non-IID distribution.

    Args:
        non_iid: 0.0 = IID, 1.0 = fully Non-IID (each client gets mostly one class)
    """
    rng = np.random.RandomState(seed)
    n_classes = len(np.unique(labels))
    indices_per_class = [np.where(labels == c)[0] for c in range(n_classes)]

    client_data = [[] for _ in range(n_clients)]

    if non_iid > 0:
        # Sort indices by class, then distribute with skew
        for cls_indices in indices_per_class:
            rng.shuffle(cls_indices)
            # Primary client for this class
            primary = rng.randint(n_clients)
            split_point = int(len(cls_indices) * (0.3 + 0.7 * non_iid))
            client_data[primary].extend(cls_indices[:split_point].tolist())
            remaining = cls_indices[split_point:]
            # Distribute remaining across other clients
            for idx in remaining:
                other = rng.choice([c for c in range(n_clients) if c != primary])
                client_data[other].append(idx)
    else:
        # IID: shuffle and split evenly
        all_indices = rng.permutation(len(features))
        chunk_size = len(features) // n_clients
        for i in range(n_clients):
            client_data[i] = all_indices[i * chunk_size : (i + 1) * chunk_size].tolist()

    return [(features[indices], labels[indices]) for indices in client_data]


def compute_client_distribution(labels: np.ndarray, n_clients: int, class_names: list[str]) -> pd.DataFrame:
    """Compute per-client class distribution as a DataFrame."""
    n_samples = len(labels)
    chunk_size = n_samples // n_clients
    rows = []
    for cid in range(n_clients):
        start = cid * chunk_size
        end = start + chunk_size if cid < n_clients - 1 else n_samples
        client_labels = labels[start:end]
        for cls_id, cls_name in enumerate(class_names):
            count = (client_labels == cls_id).sum()
            rows.append({"Client": f"Client {cid + 1}", "Class": cls_name, "Count": count})
    return pd.DataFrame(rows)


def format_accuracy(acc: float) -> str:
    """Format accuracy as percentage string."""
    return f"{acc * 100:.2f}%"


def format_loss(loss: float) -> str:
    """Format loss value."""
    return f"{loss:.4f}"
