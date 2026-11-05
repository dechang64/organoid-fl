# ── analysis/pipeline/disagreement.py ──
"""Disagreement analysis for one sample (PRD 4.5, S3).

Why this is a pure function and not a service method
----------------------------------------------------
The same computation has to run in three places: online when the Nth blind read
lands (P-02, latency ≤ 60 s), offline when a dataset is re-scored under a new
metric version, and in tests. A pure function over payload bytes is the only
shape that can be all three without the offline run silently using different
rules than the online one.

What is deliberately absent
---------------------------
``controversy_score`` is a weighted composite of four signals in the PRD. Two of
them are not computable yet:

* *与预标注差异* needs the pre-annotation payload. M0 recorded that PRD 7.2 has
  no ``PreAnnotation`` entity, so there is nothing to compare against; the
  projection layer can only say whether a ranking exists.
* *自评低置信度* needs per-instance confidence, which the payload format allows
  but the canvas does not always collect.

An absent component is recorded as absent and the remaining weights are
renormalised, with the weights written into the snapshot. The alternative —
treating a missing signal as zero disagreement — would score a sample with no
pre-annotation as *less* controversial than an otherwise identical one, and the
Top-10% cosign trigger would then depend on which fields the client happened to
send.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from analysis.contracts.hashing import content_hash
from analysis.pipeline.geometry import (
    BOOTSTRAP_SEED,
    CoderPayload,
    GeometryError,
    bootstrap_alpha_ci,
    build_units,
    match_payloads,
    positive_negative_agreement,
    raw_agreement,
)

__all__ = [
    "CONTROVERSY_PROFILE",
    "METRICS_VERSION",
    "DisagreementComputation",
    "DisagreementError",
    "compute_disagreement",
    "decode_region_map",
    "revision_set_hash",
]

#: Version of the metric definitions *and* the snapshot's shape. A stored
#: snapshot names the version that produced it, so a dataset scored under two
#: versions can be told apart instead of being averaged together.
METRICS_VERSION = "organoid-metrics-v1"

#: Named weight profile. Numbers live here rather than in the PRD because the
#: PRD does not fix them; naming the profile keeps a score comparable only with
#: scores from the same profile, which the snapshot records.
CONTROVERSY_PROFILE = "composite-v1"

WEIGHTS: Mapping[str, float] = {
    "iou": 0.50,
    "attribute": 0.30,
    "confidence": 0.10,
    "preannotation": 0.10,
}


class DisagreementError(ValueError):
    """The sample cannot be analysed as requested."""


@dataclass(frozen=True)
class DisagreementComputation:
    """Everything one computation produces, before it is persisted."""

    snapshot: Mapping[str, Any]
    controversy_score: float
    regions: bytes
    revision_set_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot": dict(self.snapshot),
            "controversy_score": self.controversy_score,
            "revision_set_hash": self.revision_set_hash,
        }


def revision_set_hash(payloads: Mapping[str, bytes]) -> str:
    """Identity of the input set.

    Recomputation is skipped when this is unchanged, so a second call after an
    unrelated edit does not append a duplicate audit event — but a supplement
    *does* change the hash, which is what makes "recompute after supplements"
    work without a dirty flag.
    """
    return content_hash(
        {
            "coders": sorted(payloads),
            "payloads": {coder: content_hash({"bytes": data.hex()}) for coder, data in payloads.items()},
        }
    )


def compute_disagreement(
    sample_id: str,
    payloads: Mapping[str, bytes],
    *,
    iou_threshold: float = 0.5,
    resamples: int = 1000,
    seed: int = BOOTSTRAP_SEED,
) -> DisagreementComputation:
    """Compare two or more coders' products for one sample.

    Raises:
        DisagreementError: with fewer than two coders, or when the payloads are
            not comparable (different canvas, unparseable geometry).
    """
    if len(payloads) < 2:
        raise DisagreementError(
            f"sample {sample_id!r} has {len(payloads)} coder product(s); disagreement "
            "needs at least two, and reporting 1.0 agreement for a single coder is the "
            "most expensive kind of wrong number"
        )
    try:
        parsed = {coder: CoderPayload.parse(data) for coder, data in payloads.items()}
    except GeometryError as exc:
        raise DisagreementError(f"sample {sample_id!r}: {exc}") from exc

    coders = sorted(parsed)
    pairs: dict[str, Any] = {}
    pairwise_iou: list[float] = []
    for i, coder_a in enumerate(coders):
        for coder_b in coders[i + 1 :]:
            result = match_payloads(parsed[coder_a], parsed[coder_b], iou_threshold=iou_threshold)
            pairs[f"{coder_a}|{coder_b}"] = result.to_dict()
            # A pair with no matched object at all is total disagreement, not a
            # missing value: both coders answered, they just answered differently.
            pairwise_iou.append(result.iou_mean if result.iou_mean is not None else 0.0)

    units = build_units(parsed, iou_threshold=iou_threshold)
    unit_values: list[list[Optional[str]]] = [
        [unit.values.get(coder) for coder in coders] for unit in units
    ]
    alpha = bootstrap_alpha_ci(unit_values, resamples=resamples, seed=seed)

    all_confidence = [value for payload in parsed.values() for value in payload.confidence_values()]
    iou_component = 1.0 - (sum(pairwise_iou) / len(pairwise_iou))
    attribute_component = None if alpha["alpha"] is None else 1.0 - float(alpha["alpha"])
    confidence_component = None if not all_confidence else 1.0 - (sum(all_confidence) / len(all_confidence))

    components: dict[str, Optional[float]] = {
        "iou": _clamp(iou_component),
        "attribute": None if attribute_component is None else _clamp(attribute_component),
        "confidence": None if confidence_component is None else _clamp(confidence_component),
        # No PreAnnotation entity in the contract yet (see module docstring).
        "preannotation": None,
    }
    missing = {
        "preannotation": "no pre-annotation payload in the contract (M0 gap, PRD 7.2)",
    }
    if components["confidence"] is None:
        missing["confidence"] = "no coder supplied per-instance confidence"
    if components["attribute"] is None:
        missing["attribute"] = "no unit has two coder values, so alpha is undefined"

    score, weights_used = _composite(components)
    regions, region_stats = _region_map(parsed)

    snapshot: dict[str, Any] = {
        "metrics_version": METRICS_VERSION,
        "controversy_profile": CONTROVERSY_PROFILE,
        "coders": coders,
        "iou_threshold": iou_threshold,
        "pairs": pairs,
        "units": {
            "count": len(units),
            "ambiguous_coders": sum(len(unit.ambiguous) for unit in units),
            "values": {unit.unit_id: dict(unit.values) for unit in units if len(unit.values) > 1},
        },
        "alpha": alpha,
        "agreement": {
            "raw": raw_agreement(unit_values),
            "positive_negative": positive_negative_agreement(unit_values),
        },
        "controversy": {
            "score": score,
            "components": dict(components),
            "weights": weights_used,
            "missing_components": missing,
        },
        "regions": region_stats,
    }
    return DisagreementComputation(
        snapshot=snapshot,
        controversy_score=score,
        regions=regions,
        revision_set_hash=revision_set_hash(payloads),
    )


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _composite(
    components: Mapping[str, Optional[float]],
) -> tuple[float, dict[str, float]]:
    """Weighted mean over the components that exist, renormalised.

    Returns the score and the weights actually used, which go into the snapshot:
    a score without its weights cannot be compared with a later one, and the
    renormalisation is the reason two samples' scores are not on the same scale
    when their available components differ.
    """
    available = {name: value for name, value in components.items() if value is not None}
    if not available:
        raise DisagreementError("no controversy component is computable")
    total = sum(WEIGHTS[name] for name in available)
    weights = {name: WEIGHTS[name] / total for name in available}
    score = sum(weights[name] * float(value) for name, value in available.items())
    return _clamp(score), weights


def _region_map(parsed: Mapping[str, CoderPayload]) -> tuple[bytes, dict[str, Any]]:
    """Pixels where the coders do not all agree, run-length encoded.

    Stored as run-lengths rather than a bitmap because the map is overwhelmingly
    two-valued at its edges (all coders agree, or all but one): RLE of the vote
    counts is a few hundred integers for a 200×200 canvas, while a PNG would need
    a binary dependency to write.
    """
    coders = sorted(parsed)
    shapes = {(payload.height, payload.width) for payload in parsed.values()}
    if len(shapes) != 1:
        raise DisagreementError(f"coders disagree about the canvas: {sorted(shapes)}")
    height, width = next(iter(shapes))
    votes = np.zeros((height, width), dtype=np.uint8)
    for payload in parsed.values():
        votes += payload.union_mask().astype(np.uint8)

    n = len(coders)
    disputed = (votes > 0) & (votes < n)
    union = votes > 0
    flat = votes.reshape(-1).tolist()
    runs: list[list[int]] = []
    for value in flat:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])

    payload = {
        "format": "organoid-dispute-map-v1",
        "width": width,
        "height": height,
        "coders": coders,
        "runs": runs,
    }
    stats = {
        "disputed_pixels": int(disputed.sum()),
        "union_pixels": int(union.sum()),
        "disputed_share": float(disputed.sum() / union.sum()) if union.any() else 0.0,
        "runs": len(runs),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"), stats


def decode_region_map(regions: bytes) -> np.ndarray:
    """Rebuild the vote map from :func:`_region_map` output.

    Exposed because the adjudication view has to draw the map, and because a test
    that only round-trips the bytes would not prove the runs describe the pixels
    the analysis claims.
    """
    data = json.loads(regions.decode("utf-8"))
    values: list[int] = []
    for value, count in data["runs"]:
        values.extend([value] * count)
    expected = data["height"] * data["width"]
    if len(values) != expected:
        raise DisagreementError(
            f"run-length map decodes to {len(values)} pixels, expected {expected}"
        )
    return np.array(values, dtype=np.uint8).reshape(data["height"], data["width"])

