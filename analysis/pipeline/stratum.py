# ── analysis/pipeline/stratum.py ──
"""Difficulty strata at ingestion (PRD 4.7, 7.2, D-08; question Q-07).

The rule is not decided yet, and this module does not decide it
---------------------------------------------------------------
Q-07 asks *which signals* determine a sample's difficulty stratum. That is a
methodological decision with a permanent consequence: PRD 4.8 notes that a
stratum not recorded at ingestion cannot be reconstructed later, so the label has
to be written before the sample is worked on. Writing a rule now would freeze an
answer the project has not given — and because the label is *also* the key of the
held-out split (D-08), a wrong rule silently moves the hardest samples into the
easy stratum and nothing reports a problem.

So what ships is the mechanism: a rule registry, mandatory signals, a version
stamp, and a snapshot of the signals that produced each label. The one rule
included here is explicitly marked ``provisional`` and is refused unless the
caller says so by name. That way the ingestion path has something to refuse, the
tests have something to exercise, and the decision stays where it belongs.

What "no default" means in code
--------------------------------
There is no ``EASY`` fallback anywhere. A missing signal raises; an unknown rule
raises; a provisional rule raises unless acknowledged. ``StratumLabel`` itself
already refuses an empty ``signal_snapshot``, so a label recorded without
evidence cannot be constructed even by a caller that bypasses this module.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from analysis.contracts.entities import ContractValidationError, StratumLabel
from analysis.contracts.enums import Stratum

__all__ = [
    "STRATUM_PROVISIONAL_NOTE",
    "STRATUM_RULES",
    "StratumRule",
    "StratumRuleError",
    "label_stratum",
    "rule",
]


class StratumRuleError(ValueError):
    """Signals, rule, or the acknowledgement of a provisional rule is missing."""


#: Recorded in the PRD and in the ingestion log next to every provisional label,
#: so a dataset that used one is identifiable after Q-07 is answered.
STRATUM_PROVISIONAL_NOTE = (
    "Q-07 未定：本规则的信号与阈值均为占位，落盘时记录 rule_version 与 signal_snapshot，"
    "Q-07 定案后需按新规则重打并保留旧标签"
)


@dataclass(frozen=True)
class StratumRule:
    """One way of turning ingestion signals into a stratum."""

    version: str
    provisional: bool
    required_signals: tuple[str, ...]
    decide: Callable[[Mapping[str, float]], Stratum]
    notes: str = ""


def _range(signals: Mapping[str, float], name: str, low: float, high: float) -> float:
    value = signals[name]
    if not math.isfinite(value):
        raise StratumRuleError(f"signal {name!r} is not finite ({value!r})")
    if value < low or value > high:
        raise StratumRuleError(
            f"signal {name!r} = {value} is outside [{low}, {high}]; a value out of "
            "range means the feature extractor and the rule disagree about units, and "
            "guessing which one is right would mislabel the sample permanently"
        )
    return value


def _provisional_v1(signals: Mapping[str, float]) -> Stratum:
    """Ordered decision, most-specific first.

    Order matters and is part of the rule: a clustered sample is hard regardless
    of how good its contrast is, because the difficulty is deciding where one
    organoid ends. Evaluating contrast first would file it as easy.
    """
    contrast = _range(signals, "contrast", 0.0, 1.0)
    object_count = _range(signals, "object_count", 0.0, 10_000.0)
    clustered = _range(signals, "clustered_fraction", 0.0, 1.0)
    size_cv = _range(signals, "size_cv", 0.0, 10.0)
    if clustered >= 0.40:
        return Stratum.CLUSTERED
    if contrast <= 0.25:
        return Stratum.LOW_CONTRAST
    if object_count >= 25 or size_cv >= 0.60:
        return Stratum.BORDERLINE
    return Stratum.EASY


#: Registry. Keys are persisted ids; a project registers its own rule here after
#: Q-07 is answered rather than editing this module.
STRATUM_RULES: dict[str, StratumRule] = {
    "provisional_v1": StratumRule(
        version="provisional_v1",
        provisional=True,
        required_signals=("contrast", "object_count", "clustered_fraction", "size_cv"),
        decide=_provisional_v1,
        notes=STRATUM_PROVISIONAL_NOTE,
    )
}


def rule(name: str) -> StratumRule:
    try:
        return STRATUM_RULES[name]
    except KeyError as exc:
        raise StratumRuleError(
            f"{name!r} is not a registered stratum rule; known: {sorted(STRATUM_RULES)}"
        ) from exc


def label_stratum(
    sample_id: str,
    signals: Mapping[str, Any],
    *,
    rule_name: str = "provisional_v1",
    allow_provisional: bool = False,
    labeled_at: Optional[str] = None,
) -> StratumLabel:
    """Turn ingestion signals into a :class:`StratumLabel`.

    Args:
        signals: whatever the feature extractor measured. Every signal the rule
            declares must be present — a rule that fills in a missing signal is a
            rule that labels on partial evidence.
        allow_provisional: must be ``True`` to use a rule marked provisional.
            This is the Q-07 gate: it makes "we are running on an undecided rule"
            an explicit act with a name attached, not a default.
    """
    if not isinstance(signals, Mapping):
        raise StratumRuleError(f"signals must be a mapping, got {type(signals).__name__}")
    selected = rule(rule_name)
    if selected.provisional and not allow_provisional:
        raise StratumRuleError(
            f"stratum rule {selected.version!r} is provisional and Q-07 is still open: "
            f"{selected.notes}. Pass allow_provisional=True to accept a provisional "
            "label, or register the project's own rule."
        )
    missing = [name for name in selected.required_signals if name not in signals]
    if missing:
        raise StratumRuleError(
            f"missing signal(s) {missing} required by rule {selected.version!r}. D-08: "
            "a stratum that cannot be derived must fail ingestion — defaulting to "
            "'easy' would file the hardest samples as the easiest and nothing would "
            "report it."
        )
    unknown = sorted(set(signals) - set(selected.required_signals))
    numeric: dict[str, float] = {}
    for name in selected.required_signals:
        value = signals[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StratumRuleError(f"signal {name!r} must be numeric, got {type(value).__name__}")
        numeric[name] = float(value)

    stratum = selected.decide(numeric)
    try:
        return StratumLabel(
            sample_id=sample_id,
            stratum=stratum,
            signal_snapshot={
                **numeric,
                "rule_version": selected.version,
                "rule_provisional": selected.provisional,
                "extra_signals": {name: signals[name] for name in unknown},
            },
            labeled_at=labeled_at or _now(),
            rule_version=selected.version,
        )
    except ContractValidationError as exc:
        raise StratumRuleError(str(exc)) from exc


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
