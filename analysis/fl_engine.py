from __future__ import annotations
# ── analysis/fl_engine.py ──
"""
Federated Learning Engine
=========================
Core FL simulation: FedAvg aggregation with per-client training.
Pure PyTorch — no external FL framework dependency.

Supports:
- FedAvg (McMahan 2017), sample-weighted or uniform
- FedProx proximal term (Li 2020)
- Performance-gated aggregation — ``none`` / ``soft`` / ``hard`` / ``local``
  (see :mod:`analysis.gating`)
- Configurable Non-IID data splits
- Per-round metrics tracking

On gating
---------
``docs/experiment_tracker.md:39-56`` records that on mouse-liver organoid data,
holding the data, model and round count fixed and changing *only* the
aggregation policy moved mAP50 from **0.30** (unconditional) to **0.75**
(EWA-weighted) to **0.92** (accept-on-improvement). That policy previously
existed only inside ``scripts/mouse_liver/fl_sequential.py``. It is now
available here through :meth:`FLEngine.run_sequential`, and
:meth:`FLEngine.run` is unchanged for existing callers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from typing import Callable, Optional, Sequence
import time

from analysis.aggregation import (
    AggregationStrategy,
    fedavg as _fedavg_weighted,
    fedavg_unweighted as _fedavg_unweighted,
)
from analysis.gating import GateStrategy, PerformanceGate


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

    Delegates to :func:`analysis.aggregation.fedavg` so there is a single
    implementation of the averaging rule in the codebase.

    Args:
        client_params: list of client parameter dicts.
        client_weights: number of samples per client. If None, uses
            unweighted average (assumes equal data split).
    """
    if client_weights is None:
        return _fedavg_unweighted(client_params)
    return _fedavg_weighted(client_params, client_weights)


def train_client(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.001,
    epochs: int = 2,
    batch_size: int = 32,
    global_params: Optional[OrderedDict] = None,
    prox_mu: float = 0.0,
) -> tuple[OrderedDict, float, float]:
    """Train a single client locally.

    Args:
        global_params: current global parameters. Required when ``prox_mu > 0``.
        prox_mu: FedProx proximal coefficient (Li 2020). ``0.0`` is plain
            FedAvg. A positive value adds ``(mu/2) * ||w - w_global||²`` to the
            local objective, anchoring the client to the global model and
            damping drift on small or shifted local datasets.

    Returns:
        (params, train_loss, train_accuracy)
    """
    if prox_mu and global_params is None:
        raise ValueError("train_client(prox_mu>0) requires global_params to compute the proximal term")

    model.train()
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Snapshot the anchor so the proximal term is measured against the global
    # model as it was at the start of this client's update, not a moving target.
    #
    # The anchor must come from ``global_params``, not from the model's current
    # weights. Taking it from ``model`` made the term read
    # ``||w - w_local_start||``, which is zero at step 0 and only measures
    # *local* drift — it does nothing to keep clients near the global model when
    # a caller passes global parameters that differ from the model's own. In the
    # usual flow the two coincide because the engine calls
    # ``set_params(client_model, global_params)`` first, which is exactly why
    # the mistake stayed invisible.
    anchor = None
    if prox_mu and global_params is not None:
        anchor = OrderedDict(
            (name, global_params[name].detach().clone())
            for name, _ in model.named_parameters()
            if name in global_params
        )

    total_loss, correct, total = 0.0, 0, 0
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)

            if anchor:
                proximal = None
                for name, param in model.named_parameters():
                    if name not in anchor:
                        continue
                    delta = param - anchor[name].to(param.device)
                    term = (delta * delta).sum()
                    proximal = term if proximal is None else proximal + term
                if proximal is not None:
                    loss = loss + proximal * (prox_mu / 2.0)

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

    Reproducibility: ``seed`` drives *both* the numpy data split and the torch
    RNG (model initialisation, ``DataLoader`` shuffling, dropout). Seeding only
    the split — as this class previously did — left every run starting from a
    different initialisation, so two runs of the same configuration could differ
    by several accuracy points. Callers that need bit-identical reruns should
    keep ``torch.use_deterministic_algorithms`` in mind as well.
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
        prox_mu: float = 0.0,
        strategy: AggregationStrategy | str = AggregationStrategy.FEDAVG,
        n_clients: int = 3,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.non_iid = non_iid
        self.seed = seed
        self.prox_mu = prox_mu
        self.strategy = AggregationStrategy.parse(strategy)
        #: Default client count for :meth:`run_sequential`; :meth:`run` still
        #: takes ``n_clients`` per call.
        self.num_clients_hint = n_clients

    def _seed_torch(self) -> None:
        """Make the torch-side randomness follow :attr:`seed`.

        Without this the engine is not reproducible: the data split is seeded
        but model weights, batch order and dropout are drawn from the process
        global RNG, which depends on whatever ran before.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a hard dependency here
            return
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU-only path
            torch.cuda.manual_seed_all(self.seed)


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

        self._seed_torch()

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

        if self.strategy is AggregationStrategy.FEDAVG_UNWEIGHTED:
            self._warn_unweighted()

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
                    global_params=global_params,
                    prox_mu=self.prox_mu,
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
            if self.strategy is AggregationStrategy.FEDAVG_UNWEIGHTED:
                global_params = fedavg_aggregate(client_params_list)
            else:
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

    # ─────────────────────────────────────────────────────────────────────
    # Sequential-chain, performance-gated federation
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _warn_unweighted() -> None:
        import warnings

        warnings.warn(
            "AggregationStrategy.FEDAVG_UNWEIGHTED gives every client equal say "
            "regardless of data volume. On unbalanced sites this over-weights the "
            "smallest one. Use FEDAVG with real sample counts unless you are "
            "deliberately reproducing a legacy result.",
            RuntimeWarning,
            stacklevel=3,
        )

    def run_sequential(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        gate: GateStrategy | str | PerformanceGate = GateStrategy.HARD,
        gate_margin: float = 0.0,
        gate_ewa_warmup_rounds: int = 2,
        client_order: Optional[Sequence[int]] = None,
        rounds: int = 10,
        val_split: float = 0.2,
        higher_is_better: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> list[dict]:
        """Sequential-chain FL with a performance gate.

        Each round visits clients one at a time; after training locally on the
        current global model, a client's update is *gated* before being folded
        in. The signal is that client's post-update accuracy on the shared
        validation split.

        This is the regime in which the recorded 0.30 → 0.75 → 0.92 spread
        appears: with an unconditional average the global model is dragged
        towards every site in turn, while a gate keeps a good model good.

        Args:
            gate: a :class:`~analysis.gating.PerformanceGate`, a gate strategy
                name, or a :class:`~analysis.gating.GateStrategy`.
            gate_margin: acceptance margin for ``hard``/``local``.
            gate_ewa_warmup_rounds: warmup rounds for ``soft``.
            client_order: visitation order. ``None`` follows client index order.
            higher_is_better: set ``False`` when gating on a loss.

        Returns:
            Per-round metric dicts, each carrying the gate decisions so the run
            can be replayed and audited.
        """
        from utils.helpers import split_federated_data
        from analysis.aggregation import blend as _blend

        if isinstance(gate, PerformanceGate):
            gate_obj = gate
        else:
            gate_obj = PerformanceGate(
                strategy=gate,
                margin=gate_margin,
                ewa_warmup_rounds=gate_ewa_warmup_rounds,
                higher_is_better=higher_is_better,
            )
        # Keep the explicitly-passed margin in sync when a bare strategy was given.
        if not isinstance(gate, PerformanceGate):
            gate_obj.margin = gate_margin

        self._seed_torch()

        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(len(features))
        features, labels = features[idx], labels[idx]

        val_size = int(len(features) * val_split)
        val_X, val_y = features[:val_size], labels[:val_size]
        train_X, train_y = features[val_size:], labels[val_size:]

        n_clients = self.num_clients_hint or 3
        client_data = split_federated_data(
            train_X, train_y, n_clients, non_iid=self.non_iid, seed=self.seed
        )

        order = list(client_order) if client_order is not None else list(range(n_clients))
        if sorted(order) != list(range(n_clients)):
            raise ValueError(
                f"client_order must be a permutation of 0..{n_clients - 1}, got {order}"
            )

        global_model = OrganoidClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
        )
        global_params = get_params(global_model)

        history: list[dict] = []
        for rnd in range(rounds):
            t0 = time.time()
            round_decisions: list[dict] = []

            for cid in order:
                cX, cy = client_data[cid]
                client_model = OrganoidClassifier(
                    input_dim=self.input_dim,
                    num_classes=self.num_classes,
                    hidden_dim=self.hidden_dim,
                )
                set_params(client_model, global_params)
                candidate_params, train_loss, train_acc = train_client(
                    client_model, cX, cy,
                    lr=self.lr,
                    epochs=self.local_epochs,
                    batch_size=self.batch_size,
                    global_params=global_params,
                    prox_mu=self.prox_mu,
                )

                # Signal: the candidate's own accuracy on the shared validation
                # split. Using the shared split keeps the comparison across
                # clients meaningful — per-client local splits are not.
                set_params(client_model, candidate_params)
                _, signal = evaluate_model(client_model, val_X, val_y, self.batch_size)

                decision = gate_obj.decide(
                    signal,
                    local_samples=len(cX),
                    round_index=rnd,
                    node_name=f"client_{cid}",
                )

                if decision.weight_local > 0.0:
                    if decision.weight_global > 0.0:
                        global_params = _blend(
                            candidate_params, global_params, decision.weight_local
                        )
                    else:
                        global_params = candidate_params

                set_params(global_model, global_params)
                val_loss, val_acc = evaluate_model(global_model, val_X, val_y, self.batch_size)

                round_decisions.append({
                    "client": cid + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "signal": signal,
                    "decision": decision.to_dict(),
                    "global_val_acc_after": val_acc,
                })

            elapsed = time.time() - t0
            accepted = sum(1 for d in round_decisions if d["decision"]["accept"])
            round_metrics = {
                "round": rnd + 1,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "elapsed": elapsed,
                "gate_strategy": gate_obj.strategy.value,
                "accepted": accepted,
                "rejected": len(round_decisions) - accepted,
                "global_signal": gate_obj.global_signal,
                "client_metrics": round_decisions,
            }
            history.append(round_metrics)

            if progress_callback:
                progress_callback(rnd + 1, round_metrics)

        self.global_model = global_model
        self.global_params = global_params
        self.history = history
        return history
