# ── analysis/gating.py ──
"""Performance gating for federated aggregation.

Why this exists
---------------
``docs/experiment_tracker.md:39-56`` records the single most important result
this project has produced:

    same data, same model, same round count — only the aggregation policy differs

        gate=none (unconditional)   mAP50 0.30
        gate=soft (EWA weighted)    mAP50 0.75
        gate=hard (accept on gain)  mAP50 0.92

A three-fold difference in detection quality from the aggregation rule alone.
Across federated sites with different scanners and staining protocols, the
unconditional average is actively harmful: it drags a strong site's model
towards a weak site's distribution.

Despite that, the platform's own engine — ``analysis/fl_engine.py`` — shipped
only plain FedAvg, and this logic lived exclusively in
``scripts/mouse_liver/fl_sequential.py:336-407``, a YOLO-training script that
needs a GPU, Ultralytics and the mouse-liver batches to run at all. Nothing in
``analysis/`` could reproduce the result, so the product could not expose it.

This module lifts the policy into the platform as a small, testable,
framework-free component, preserving the exact semantics of the original
script (see the per-strategy notes below).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

__all__ = [
    "GateDecision",
    "GateStrategy",
    "PerformanceGate",
    "UnknownGateError",
]


class UnknownGateError(ValueError):
    """Raised when an unrecognised gate strategy is requested."""


class GateStrategy(str, Enum):
    """How a candidate update is weighed against the current global model.

    Semantics are ported verbatim from ``scripts/mouse_liver/fl_sequential.py``
    so the platform can reproduce the recorded numbers.

    ``NONE``
        Unconditional. Blend by *data volume* alone:
        ``w_local = n_local / (n_local + global_count)``. This is what produced
        the 0.30 mAP50 baseline.
    ``SOFT``
        Performance-weighted exponential moving average. The first
        ``ewa_warmup_rounds`` rounds fall back to the volume-weighted blend,
        then weights become
        ``w_local = sig_local / (sig_local + sig_global)``. Produced 0.75.
    ``HARD``
        Accept the candidate only when it beats the current global signal by
        more than ``margin``; otherwise keep the global model untouched.
        Produced 0.92.
    ``LOCAL``
        No global gate at all: each node keeps its own best model, and the
        global model is the average of those per-node bests. This is the
        "sovereignty follows keys" pattern — a site that never gets a good
        global model still keeps its own.
    """

    NONE = "none"
    SOFT = "soft"
    HARD = "hard"
    LOCAL = "local"

    @classmethod
    def parse(cls, value: "GateStrategy | str | None") -> "GateStrategy":
        if value is None:
            return cls.NONE
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            valid = ", ".join(s.value for s in cls)
            raise UnknownGateError(
                f"Unknown gate strategy {value!r}; expected one of: {valid}"
            ) from exc


@dataclass(frozen=True)
class GateDecision:
    """The outcome of one gating decision, with its reasoning.

    Every field needed to audit *why* a model changed is carried here, so a
    training log can be replayed and argued about without re-running anything.
    """

    strategy: GateStrategy
    accept: bool
    #: Coefficient on the incoming local/candidate parameters.
    weight_local: float
    #: Coefficient on the retained global parameters.
    weight_global: float
    reason: str
    signal_local: Optional[float] = None
    signal_global: Optional[float] = None
    margin: float = 0.0
    round_index: int = 0
    node_name: Optional[str] = None

    @property
    def is_blend(self) -> bool:
        """True when both sides contribute (a weighted merge, not a replace/keep)."""
        return self.weight_local > 0.0 and self.weight_global > 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "accept": self.accept,
            "weight_local": round(self.weight_local, 6),
            "weight_global": round(self.weight_global, 6),
            "reason": self.reason,
            "signal_local": self.signal_local,
            "signal_global": self.signal_global,
            "margin": self.margin,
            "round_index": self.round_index,
            "node_name": self.node_name,
        }


@dataclass
class PerformanceGate:
    """Stateful gate. One instance per federated run.

    Args:
        strategy: which policy to apply.
        margin: ``HARD``/``LOCAL`` acceptance threshold. Must be beaten
            *strictly*, matching the original ``signal_local > cached + margin``.
        ewa_warmup_rounds: ``SOFT`` only — rounds of volume-weighted blending
            before performance weighting starts. Two in the original script.
        higher_is_better: set ``False`` for losses so the comparisons flip.
        eps: floor for performance signals, preventing a zero-valued signal
            from zeroing its own weight.
        min_signal: when a signal is at or below this, ``HARD`` refuses the
            update regardless of margin. Guards the case where both local and
            global are effectively broken and a tiny absolute gain would
            otherwise hand the global model to a single bad site.
    """

    strategy: GateStrategy = GateStrategy.NONE
    margin: float = 0.0
    ewa_warmup_rounds: int = 2
    higher_is_better: bool = True
    eps: float = 1e-8
    min_signal: Optional[float] = None

    # ── accumulated state ──
    round_index: int = 0
    _global_signal: Optional[float] = None
    _global_count: int = 0
    _initialised: bool = False
    _node_best: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.strategy = GateStrategy.parse(self.strategy)
        if self.ewa_warmup_rounds < 0:
            raise ValueError("ewa_warmup_rounds must be non-negative")

    # ── introspection ────────────────────────────────────────────────────

    @property
    def global_signal(self) -> Optional[float]:
        """Last known global performance signal, or ``None`` before init."""
        return self._global_signal

    @property
    def global_sample_count(self) -> int:
        return self._global_count

    @property
    def node_best(self) -> dict[str, float]:
        """Per-node best signal, populated only by :attr:`GateStrategy.LOCAL`."""
        return dict(self._node_best)

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    def reset(self) -> None:
        """Clear all accumulated state so the gate can be reused."""
        self.round_index = 0
        self._global_signal = None
        self._global_count = 0
        self._initialised = False
        self._node_best.clear()

    # ── decisions ────────────────────────────────────────────────────────

    def _better(self, candidate: float, reference: float, margin: float) -> bool:
        """``candidate`` beats ``reference + margin`` under the configured polarity."""
        if self.higher_is_better:
            return candidate > reference + margin
        return candidate < reference - margin

    def decide(
        self,
        signal_local: float,
        *,
        signal_global: Optional[float] = None,
        local_samples: Optional[int] = None,
        global_samples: Optional[int] = None,
        round_index: Optional[int] = None,
        node_name: Optional[str] = None,
    ) -> GateDecision:
        """Decide how to fold one candidate into the global model.

        Args:
            signal_local: the candidate's measured performance.
            signal_global: authoritative global performance if the caller just
                measured it. When omitted the gate uses its cached value, which
                mirrors the original script (the cache is refreshed at round
                end).
            local_samples: this contribution's data volume, for ``NONE`` and
                the ``SOFT`` warmup.
            global_samples: volume already accumulated in the global model.
            round_index: 0-based round, drives the ``SOFT`` warmup.
            node_name: required by ``LOCAL``.

        Returns:
            A :class:`GateDecision`. ``weight_local == 0`` always means "keep
            the existing global model unchanged".
        """
        rnd = self.round_index if round_index is None else round_index
        current_global = signal_global if signal_global is not None else self._global_signal

        handler = {
            GateStrategy.NONE: self._decide_none,
            GateStrategy.SOFT: self._decide_soft,
            GateStrategy.HARD: self._decide_hard,
            GateStrategy.LOCAL: self._decide_local,
        }[self.strategy]

        decision = handler(
            signal_local=signal_local,
            signal_global=current_global,
            local_samples=local_samples,
            global_samples=global_samples,
            round_index=rnd,
            node_name=node_name,
        )

        # ── advance state ──
        self.round_index = rnd + 1
        if decision.accept:
            self._initialised = True
            if self.strategy in (GateStrategy.NONE, GateStrategy.HARD):
                self._global_signal = signal_local
                self._global_count = (
                    (self._global_count or global_samples or 0) + (local_samples or 0)
                )
            elif self.strategy is GateStrategy.SOFT:
                self._global_signal = (
                    decision.weight_local * signal_local
                    + decision.weight_global * (current_global or 0.0)
                )
                self._global_count = (
                    (self._global_count or global_samples or 0) + (local_samples or 0)
                )
        elif current_global is not None:
            self._global_signal = current_global

        return decision

    # ── per-strategy implementations ─────────────────────────────────────

    def _decide_none(
        self,
        *,
        signal_local: float,
        signal_global: Optional[float],
        local_samples: Optional[int],
        global_samples: Optional[int],
        round_index: int,
        node_name: Optional[str],
    ) -> GateDecision:
        n_local = int(local_samples or 0)
        n_global = int(global_samples if global_samples is not None else self._global_count)

        if not self._initialised or (n_local + n_global) == 0:
            return GateDecision(
                strategy=self.strategy, accept=True, weight_local=1.0, weight_global=0.0,
                reason="unconditional: no accumulated global model yet, adopt candidate",
                signal_local=signal_local, signal_global=signal_global,
                margin=self.margin, round_index=round_index, node_name=node_name,
            )

        w_local = n_local / (n_local + n_global)
        return GateDecision(
            strategy=self.strategy, accept=True,
            weight_local=w_local, weight_global=1.0 - w_local,
            reason=(
                f"unconditional volume-weighted blend: {n_local} new / "
                f"{n_local + n_global} total samples"
            ),
            signal_local=signal_local, signal_global=signal_global,
            margin=self.margin, round_index=round_index, node_name=node_name,
        )

    def _decide_soft(
        self,
        *,
        signal_local: float,
        signal_global: Optional[float],
        local_samples: Optional[int],
        global_samples: Optional[int],
        round_index: int,
        node_name: Optional[str],
    ) -> GateDecision:
        n_local = int(local_samples or 0)
        n_global = int(global_samples if global_samples is not None else self._global_count)

        warming_up = (not self._initialised) or (round_index < self.ewa_warmup_rounds)
        if warming_up:
            if not self._initialised or (n_local + n_global) == 0:
                return GateDecision(
                    strategy=self.strategy, accept=True, weight_local=1.0, weight_global=0.0,
                    reason="EWA warmup: no global model yet, adopt candidate",
                    signal_local=signal_local, signal_global=signal_global,
                    margin=self.margin, round_index=round_index, node_name=node_name,
                )
            w_local = n_local / (n_local + n_global)
            return GateDecision(
                strategy=self.strategy, accept=True,
                weight_local=w_local, weight_global=1.0 - w_local,
                reason=(
                    f"EWA warmup (round {round_index} < {self.ewa_warmup_rounds}): "
                    "volume-weighted"
                ),
                signal_local=signal_local, signal_global=signal_global,
                margin=self.margin, round_index=round_index, node_name=node_name,
            )

        # Performance-weighted EWA.
        s_local = max(abs(signal_local), self.eps)
        s_global = max(abs(signal_global or 0.0), self.eps)
        total = s_local + s_global
        w_local, w_global = s_local / total, s_global / total

        return GateDecision(
            strategy=self.strategy, accept=True,
            weight_local=w_local, weight_global=w_global,
            reason=(
                f"performance-weighted EWA: sig_local={signal_local:.4f}, "
                f"sig_global={signal_global if signal_global is not None else float('nan'):.4f}"
            ),
            signal_local=signal_local, signal_global=signal_global,
            margin=self.margin, round_index=round_index, node_name=node_name,
        )

    def _decide_hard(
        self,
        *,
        signal_local: float,
        signal_global: Optional[float],
        local_samples: Optional[int],
        global_samples: Optional[int],
        round_index: int,
        node_name: Optional[str],
    ) -> GateDecision:
        if not self._initialised or signal_global is None:
            return GateDecision(
                strategy=self.strategy, accept=True, weight_local=1.0, weight_global=0.0,
                reason="hard gate: no global model yet, adopt candidate as initial global",
                signal_local=signal_local, signal_global=signal_global,
                margin=self.margin, round_index=round_index, node_name=node_name,
            )

        if self.min_signal is not None:
            below_floor = (
                signal_local <= self.min_signal
                if self.higher_is_better
                else signal_local >= self.min_signal
            )
            if below_floor:
                return GateDecision(
                    strategy=self.strategy, accept=False, weight_local=0.0, weight_global=1.0,
                    reason=(
                        f"hard gate: candidate {signal_local:.4f} is at or below the "
                        f"absolute floor {self.min_signal:.4f}; a relative margin is not "
                        "trustworthy in this regime"
                    ),
                    signal_local=signal_local, signal_global=signal_global,
                    margin=self.margin, round_index=round_index, node_name=node_name,
                )

        if self._better(signal_local, signal_global, self.margin):
            return GateDecision(
                strategy=self.strategy, accept=True, weight_local=1.0, weight_global=0.0,
                reason=(
                    f"hard gate: {signal_local:.4f} beats {signal_global:.4f} "
                    f"+ margin {self.margin}"
                ),
                signal_local=signal_local, signal_global=signal_global,
                margin=self.margin, round_index=round_index, node_name=node_name,
            )

        return GateDecision(
            strategy=self.strategy, accept=False, weight_local=0.0, weight_global=1.0,
            reason=(
                f"hard gate: {signal_local:.4f} does not beat {signal_global:.4f} "
                f"+ margin {self.margin}; keeping global"
            ),
            signal_local=signal_local, signal_global=signal_global,
            margin=self.margin, round_index=round_index, node_name=node_name,
        )

    def _decide_local(
        self,
        *,
        signal_local: float,
        signal_global: Optional[float],
        local_samples: Optional[int],
        global_samples: Optional[int],
        round_index: int,
        node_name: Optional[str],
    ) -> GateDecision:
        if not node_name:
            raise UnknownGateError(
                "GateStrategy.LOCAL requires node_name so each site's best model "
                "can be tracked independently"
            )

        best = self._node_best.get(node_name)
        if best is None or self._better(signal_local, best, self.margin):
            self._node_best[node_name] = signal_local
            return GateDecision(
                strategy=self.strategy, accept=True, weight_local=1.0, weight_global=0.0,
                reason=(
                    f"local gate: {node_name} stored new best ({signal_local:.4f}"
                    + (f", previous {best:.4f}" if best is not None else "")
                    + ")"
                ),
                signal_local=signal_local, signal_global=signal_global,
                margin=self.margin, round_index=round_index, node_name=node_name,
            )

        return GateDecision(
            strategy=self.strategy, accept=False, weight_local=0.0, weight_global=1.0,
            reason=(
                f"local gate: {node_name} keeps its stored best ({best:.4f}); "
                f"candidate {signal_local:.4f} did not improve"
            ),
            signal_local=signal_local, signal_global=signal_global,
            margin=self.margin, round_index=round_index, node_name=node_name,
        )
