# ── utils/helpers.py ──
"""Utility functions for the Organoid-FL platform.

Import policy — do not "tidy" the pandas import
-----------------------------------------------
This module deliberately does **not** import pandas at module scope, and that
is load-bearing rather than untidy.

The actual mechanism — note it is not the one you would guess — is that
importing **pyarrow after torch** corrupts the heap and kills the interpreter
with an access violation on Windows, with no traceback to catch:

    >>> import torch;      import pandas     # process dies silently
    >>> import pandas;     import torch      # fine

``torch`` and ``pyarrow`` are the incompatible pair. pandas 2.x imports pyarrow
at module scope (``pandas/compat/pyarrow.py``) to decide which Arrow-backed
features to enable, so any ``import pandas`` after torch drags pyarrow in and
dies with it. ``KMP_DUPLICATE_LIB_OK=TRUE`` does not rescue this — it only
suppresses the OpenMP guard while the access violation still occurs.

``analysis.fl_engine`` reaches this module from inside its training loop, so a
top-level pandas import here made every ``FLEngine.run()`` call die the moment
it started.

The rule is therefore *not* "move the pandas import into the function that
needs it". That helps only when that function runs before torch; once torch is
loaded, a late ``import pandas`` is precisely the crash. ``analysis/audit_engine``
did exactly that and took the whole pytest run down with it. The two options are:

1. keep pandas out of the process entirely — what this module does, and the
   right choice for a pure training path that never needs a DataFrame; or
2. import pandas *before* torch, via
   :func:`utils.import_order.prefer_pandas_before_torch`.

Keep ``TestNoPandasOnTheTorchPath`` in ``tests/test_fl_engine_gated.py`` and
``tests/test_import_order.py`` passing before changing any import here.
"""

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # typing only — must not execute at import time
    import pandas as pd

    from utils.taxonomy import Taxonomy

__all__ = [
    "compute_client_distribution",
    "format_accuracy",
    "format_loss",
    "generate_synthetic_features",
    "split_federated_data",
]


def generate_synthetic_features(
    n_samples: int = 600,
    dim: int = 512,
    n_classes: Optional[int] = None,
    seed: int = 42,
    taxonomy: "Taxonomy | str | None" = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic organoid features for demo and test purposes.

    Class names come from the taxonomy registry rather than being written out
    here. This function previously hard-coded ``["healthy", "early_stage",
    "late_stage"]`` — a fourth copy of the same placeholder schema that matched
    no dataset in the repository.

    Args:
        n_samples: total samples to generate.
        dim: feature dimensionality.
        n_classes: how many classes to generate. Defaults to the taxonomy's
            full class count.
        taxonomy: class schema; see :func:`utils.taxonomy.resolve_taxonomy`.

    Returns:
        ``(features, labels, class_names)`` with labels in ``0..n_classes-1``.
    """
    from utils.taxonomy import resolve_taxonomy

    tax = resolve_taxonomy(taxonomy)
    available = list(tax.keys)
    if n_classes is None:
        n_classes = len(available)
    if not 1 <= n_classes <= len(available):
        raise ValueError(
            f"n_classes must be between 1 and {len(available)} for taxonomy "
            f"{tax.name!r}, got {n_classes}"
        )
    class_names = available[:n_classes]

    rng = np.random.RandomState(seed)

    # Create cluster centers with separation
    centers = rng.randn(n_classes, dim).astype(np.float32) * 2.0
    features_list, labels_list = [], []

    samples_per_class = n_samples // n_classes
    for i in range(n_classes):
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
    """Split data across clients, optionally with a Non-IID distribution.

    Args:
        non_iid: 0.0 = IID, 1.0 = fully Non-IID (each client gets mostly one class)

    Returns:
        One ``(features, labels)`` pair per client, in client-index order.
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


def compute_client_distribution(
    labels: np.ndarray,
    n_clients: int,
    class_names: Sequence[str],
) -> "pd.DataFrame":
    """Compute per-client class distribution as a DataFrame.

    pandas is imported here, not at module scope — see the module docstring for
    why that matters on Windows.
    """
    import pandas as pd

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
