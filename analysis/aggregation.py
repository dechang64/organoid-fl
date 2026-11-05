# ── analysis/aggregation.py ──
"""Federated parameter aggregation strategies.

Framework-agnostic on purpose
-----------------------------
Every function here works on a plain mapping of ``name -> array_like``. Both
``torch.Tensor`` and ``numpy.ndarray`` satisfy the arithmetic used, so the
FedAvg / FedProx / gated logic can be unit-tested without a GPU, a torch
install, or a training run. The original platform engines were only reachable
through a full training loop, which is why the aggregation policy was never
independently verified.

What is covered
---------------
* :func:`fedavg` — sample-weighted mean (McMahan 2017).
* :func:`fedavg_unweighted` — plain mean, kept because the original
  ``analysis/multi_task_fl.py:154-160`` used it and existing results depend on
  being able to reproduce it.
* :func:`blend` — two-way weighted merge, the primitive the gating policies
  and EWA build on.
* :func:`fedprox_penalty` — the proximal term for local objectives (Li 2020),
  which the README cited but the repository never implemented.
* :func:`aggregate` — one entry point that applies a strategy plus an optional
  :class:`~analysis.gating.GateDecision`.
"""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from analysis.gating import GateDecision

__all__ = [
    "AggregationError",
    "AggregationReport",
    "AggregationStrategy",
    "aggregate",
    "aggregate_with_report",
    "blend",
    "fedavg",
    "fedavg_unweighted",
    "fedprox_penalty",
    "normalise_weights",
]


class AggregationError(ValueError):
    """Raised when client parameter sets are incompatible or weights are invalid."""


class AggregationStrategy(str, Enum):
    """Which aggregation rule to apply."""

    #: Sample-weighted FedAvg — ``w_k = |D_k| / sum(|D|)``.
    FEDAVG = "fedavg"
    #: Plain mean over clients, no weighting. Legacy behaviour.
    FEDAVG_UNWEIGHTED = "fedavg_unweighted"
    #: FedAvg with the local objective carrying a proximal term (used at the
    #: client, not the server; listed here so a run can declare it).
    FEDPROX = "fedprox"

    @classmethod
    def parse(cls, value: "AggregationStrategy | str | None") -> "AggregationStrategy":
        if value is None:
            return cls.FEDAVG
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            valid = ", ".join(s.value for s in cls)
            raise AggregationError(
                f"Unknown aggregation strategy {value!r}; expected one of: {valid}"
            ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _scale(value: Any, factor: float) -> Any:
    if factor == 1.0:
        return value
    return value * factor


def _add(a: Any, b: Any) -> Any:
    return a + b


def _zeros_like(value: Any) -> Any:
    """Zero of the same kind, preserving the array type (torch vs numpy)."""
    try:
        import torch  # type: ignore

        if isinstance(value, torch.Tensor):
            return torch.zeros_like(value)
    except ImportError:  # pragma: no cover
        pass
    return value * 0.0


def _is_float_like(value: Any) -> bool:
    """True when averaging this value is meaningful."""
    try:
        import torch  # type: ignore

        if isinstance(value, torch.Tensor):
            return bool(torch.is_floating_point(value))
    except ImportError:  # pragma: no cover
        pass

    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        name = str(dtype)
        # Integer buffers such as torch's ``num_batches_tracked`` must not be
        # averaged — a fractional batch counter is meaningless and torch will
        # raise when it is loaded back into the model.
        return "float" in name or "complex" in name
    return isinstance(value, float)


def _params_only(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Drop the ``_meta`` key so an :class:`AggregationReport` can be re-aggregated."""
    if "_meta" in params:
        return {k: v for k, v in params.items() if k != "_meta"}
    return params


def validate_compatible(params_list: Sequence[Mapping[str, Any]]) -> None:
    """Ensure every client exposes the same parameter keys and shapes."""
    if not params_list:
        raise AggregationError("Cannot aggregate an empty client list")

    cleaned = [_params_only(p) for p in params_list]
    reference = cleaned[0]
    ref_keys = set(reference.keys())
    for i, params in enumerate(cleaned[1:], start=1):
        keys = set(params.keys())
        if keys != ref_keys:
            missing = sorted(ref_keys - keys)
            extra = sorted(keys - ref_keys)
            raise AggregationError(
                f"Client {i} parameter keys differ. missing={missing[:5]} extra={extra[:5]}"
            )
        for key in reference:
            a, b = reference[key], params[key]
            shape_a = getattr(a, "shape", None)
            shape_b = getattr(b, "shape", None)
            if shape_a is not None and shape_a != shape_b:
                raise AggregationError(
                    f"Client {i} parameter {key!r} has shape {tuple(shape_b)} "
                    f"but client 0 has {tuple(shape_a)}"
                )


def normalise_weights(weights: Sequence[float], n: Optional[int] = None) -> list[float]:
    """Return non-negative weights summing to 1.

    Raises:
        AggregationError: on a length mismatch, a negative weight, or a total
            of zero. Silently substituting a uniform distribution here would
            hide a data accounting bug that changes the meaning of the result.
    """
    values = [float(w) for w in weights]
    if n is not None and len(values) != n:
        raise AggregationError(f"Got {len(values)} weights for {n} clients")
    if not values:
        raise AggregationError("Weight list is empty")
    if any(w < 0 for w in values):
        raise AggregationError(f"Weights must be non-negative, got {values}")
    total = sum(values)
    if total <= 0:
        raise AggregationError(
            "Weights sum to zero; every client would be ignored. Supply real "
            "sample counts or use fedavg_unweighted."
        )
    return [w / total for w in values]


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation primitives
# ─────────────────────────────────────────────────────────────────────────────


def fedavg(
    client_params: Sequence[Mapping[str, Any]],
    client_weights: Optional[Sequence[int | float]] = None,
) -> "OrderedDict[str, Any]":
    """Sample-weighted FedAvg (McMahan 2017).

    Args:
        client_params: one parameter mapping per client.
        client_weights: samples per client. When omitted a uniform mean is used,
            which is only correct for an equal split.

    Returns:
        Aggregated parameters in the reference client's key order.

    Non-floating entries (integer buffers) are copied from the first client
    rather than averaged.
    """
    if not client_params:
        raise AggregationError("Cannot aggregate an empty client list")
    client_params = [_params_only(p) for p in client_params]
    validate_compatible(client_params)

    if client_weights is None:
        return fedavg_unweighted(client_params)

    weights = normalise_weights(client_weights, n=len(client_params))
    reference = client_params[0]

    averaged: "OrderedDict[str, Any]" = OrderedDict()
    for key, ref_value in reference.items():
        if not _is_float_like(ref_value):
            averaged[key] = ref_value
            continue
        accumulator = None
        for params, w in zip(client_params, weights, strict=True):
            contribution = _scale(params[key], w)
            accumulator = contribution if accumulator is None else _add(accumulator, contribution)
        averaged[key] = accumulator

    return averaged


def fedavg_unweighted(
    client_params: Sequence[Mapping[str, Any]],
) -> "OrderedDict[str, Any]":
    """Plain mean over clients — the legacy behaviour of ``multi_task_fl._fedavg``.

    Kept explicitly because existing recorded results were produced with it.
    Note that on badly unbalanced clients this over-weights small sites; prefer
    :func:`fedavg` with real sample counts for new work.
    """
    if not client_params:
        raise AggregationError("Cannot aggregate an empty client list")
    validate_compatible(client_params)

    n = len(client_params)
    reference = client_params[0]

    averaged: "OrderedDict[str, Any]" = OrderedDict()
    for key, ref_value in reference.items():
        if not _is_float_like(ref_value):
            averaged[key] = ref_value
            continue
        accumulator = None
        for params in client_params:
            contribution = _scale(params[key], 1.0 / n)
            accumulator = contribution if accumulator is None else _add(accumulator, contribution)
        averaged[key] = accumulator

    return averaged


def blend(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    weight_primary: float,
) -> "OrderedDict[str, Any]":
    """Weighted merge ``weight_primary * primary + (1 - weight_primary) * secondary``.

    This is the operation every gating policy reduces to: ``HARD`` picks a
    weight of 1 or 0, ``NONE`` and ``SOFT`` compute an intermediate one.
    """
    if not 0.0 <= weight_primary <= 1.0:
        raise AggregationError(f"weight_primary must be in [0, 1], got {weight_primary}")
    primary = _params_only(primary)
    secondary = _params_only(secondary)
    validate_compatible([primary, secondary])

    w_secondary = 1.0 - weight_primary
    merged: "OrderedDict[str, Any]" = OrderedDict()
    for key, value in primary.items():
        if not _is_float_like(value):
            merged[key] = value
            continue
        if weight_primary == 0.0:
            merged[key] = secondary[key]
        elif w_secondary == 0.0:
            merged[key] = value
        else:
            merged[key] = _add(
                _scale(value, weight_primary),
                _scale(secondary[key], w_secondary),
            )
    return merged


def fedprox_penalty(
    local_params: Mapping[str, Any],
    global_params: Mapping[str, Any],
    mu: float,
) -> Any:
    """Proximal term ``(mu / 2) * ||w_local - w_global||^2`` (Li 2020, FedProx).

    Add this to a client's local loss during training. It anchors each site to
    the global model, which is what stops a client with a small or shifted
    dataset from drifting away on its own objective — the ``mu = 0`` case is
    exactly plain FedAvg.

    Returns a scalar of the same array type as the inputs.

    Raises:
        AggregationError: when ``mu`` is negative or parameter sets mismatch.
    """
    if mu < 0:
        raise AggregationError(f"FedProx mu must be non-negative, got {mu}")
    local_params = _params_only(local_params)
    global_params = _params_only(global_params)
    validate_compatible([local_params, global_params])

    total = None
    for key, value in local_params.items():
        if not _is_float_like(value):
            continue
        delta = value - global_params[key]
        squared = (delta * delta).sum()
        total = squared if total is None else total + squared

    if total is None:
        # No floating parameters — return a zero consistent with the first value.
        first = next(iter(local_params.values()))
        return _zeros_like(first)
    return total * (mu / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Report + high-level entry point
# ─────────────────────────────────────────────────────────────────────────────


class AggregationReport(dict):
    """Aggregation outcome plus the numbers that explain it.

    Subclasses ``dict`` so existing call sites that expect plain parameters keep
    working, while ``report["_meta"]`` carries the audit trail.
    """

    @property
    def meta(self) -> dict:
        return self.get("_meta", {})

    @property
    def params(self) -> dict:
        """Just the parameters, with metadata stripped."""
        return {k: v for k, v in self.items() if k != "_meta"}


def aggregate_with_report(
    client_params: Sequence[Mapping[str, Any]],
    *,
    strategy: AggregationStrategy | str = AggregationStrategy.FEDAVG,
    client_weights: Optional[Sequence[int | float]] = None,
    gate_decision: Optional[GateDecision] = None,
    previous_global: Optional[Mapping[str, Any]] = None,
) -> AggregationReport:
    """Aggregate clients, optionally through a gate, and explain the result.

    Flow:

    1. Merge the clients with the chosen strategy to get the *candidate*.
    2. If a :class:`GateDecision` is supplied, merge the candidate with the
       previous global model using the decision's weights. A decision with
       ``weight_local == 0`` returns the previous global model unchanged.
    """
    strat = AggregationStrategy.parse(strategy)

    if strat is AggregationStrategy.FEDAVG_UNWEIGHTED:
        candidate = fedavg_unweighted(client_params)
    else:
        candidate = fedavg(client_params, client_weights)

    meta: dict = {
        "strategy": strat.value,
        "n_clients": len(client_params),
        "client_weights": (
            [round(w, 6) for w in normalise_weights(client_weights, len(client_params))]
            if client_weights
            else None
        ),
        "gated": gate_decision is not None,
    }

    if gate_decision is None:
        result: "OrderedDict[str, Any]" = OrderedDict(candidate)
        meta["applied_weights"] = {"candidate": 1.0, "previous_global": 0.0}
    else:
        if previous_global is None:
            if gate_decision.weight_global > 0.0:
                raise AggregationError(
                    f"Gate decided weight_global={gate_decision.weight_global} but no "
                    "previous_global was supplied to merge against"
                )
            result = OrderedDict(candidate)
        else:
            result = blend(
                candidate, previous_global, gate_decision.weight_local
            )

        meta["applied_weights"] = {
            "candidate": gate_decision.weight_local,
            "previous_global": gate_decision.weight_global,
        }
        meta["gate"] = gate_decision.to_dict()

    report = AggregationReport(result)
    report["_meta"] = meta
    return report


def aggregate(
    client_params: Sequence[Mapping[str, Any]],
    *,
    strategy: AggregationStrategy | str = AggregationStrategy.FEDAVG,
    client_weights: Optional[Sequence[int | float]] = None,
    gate_decision: Optional[GateDecision] = None,
    previous_global: Optional[Mapping[str, Any]] = None,
) -> "OrderedDict[str, Any]":
    """Aggregate and return parameters only.

    Thin wrapper over :func:`aggregate_with_report` for call sites that do not
    need the audit trail.
    """
    return OrderedDict(
        aggregate_with_report(
            client_params,
            strategy=strategy,
            client_weights=client_weights,
            gate_decision=gate_decision,
            previous_global=previous_global,
        ).params
    )
