# ── analysis/pipeline/geometry.py ──
"""Geometry payloads and the segmentation metrics of PRD 4.5 / 6.1.

Why this module owns the payload format
---------------------------------------
Everything downstream — disagreement analysis, the metric matrix, adjudication,
training export — compares annotation *geometry*. If each of those read a
different payload shape, the first one would silently treat a missing ``instances``
key as "no objects" and report perfect agreement on an empty pair of masks. So the
payload format is defined here, once, with a version string, and every reader in
the pipeline goes through :meth:`CoderPayload.parse`.

Why IoU matching before metrics
--------------------------------
Dice compares two masks that are supposed to be the same object. Deciding *which*
object corresponds to which is therefore part of the measurement, not a
pre-processing detail: match greedily by descending IoU at a fixed threshold
(0.5, PRD 6.1) and report unmatched instances as FP/FN. Reporting Dice over
"whatever happens to be at the same index" is how an annotation tool ends up with
a 0.95 Dice on a sample the experts disagreed about completely.

Why no scipy
------------
``scipy`` is not a declared dependency of this repository (``requirements.txt``
lists numpy only). HD95 needs surface distances, which a KD-tree makes easy and
which numpy broadcasting makes easy enough at annotation resolution: surface
pixels of a 200×200 mask number in the hundreds, so the full distance matrix is a
few hundred kilobytes. :func:`_subsample` bounds the worst case for large masks
instead of adding a dependency for it.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "GEOMETRY_FORMAT",
    "CoderPayload",
    "GeometryError",
    "InstanceGeometry",
    "MatchPair",
    "MatchResult",
    "Unit",
    "bootstrap_alpha_ci",
    "build_units",
    "hausdorff95",
    "krippendorff_alpha_nominal",
    "match_payloads",
    "positive_negative_agreement",
    "rasterize_polygon",
    "raw_agreement",
]

#: Bumped whenever the accepted payload shape changes. The reader refuses any
#: other value rather than guessing: a payload format that silently tolerates
#: drift is how two nodes end up measuring different things under the same name.
GEOMETRY_FORMAT = "organoid-geometry-v1"

#: Pixels per side above which the canvas is rejected. Annotation canvases are
#: tiled well below this; the limit exists so a malformed payload cannot make the
#: rasteriser allocate gigabytes.
MAX_CANVAS = 8192

#: Above this many surface pixels, HD95 subsamples deterministically. The cap
#: bounds the O(n·m) distance matrix without changing the reported statistic by
#: more than the rasterisation error itself.
MAX_SURFACE_POINTS = 4000


class GeometryError(ValueError):
    """A payload is not annotatable geometry, or two payloads cannot be compared."""


# ─────────────────────────────────────────────────────────────────────────────
# Payload
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InstanceGeometry:
    """One annotated object, in canvas pixel coordinates."""

    id: str
    class_key: str
    polygon: tuple[tuple[float, float], ...]
    confidence: Optional[float] = None


@dataclass(frozen=True)
class CoderPayload:
    """One coder's complete product for one sample.

    ``instances`` may be empty: "there is nothing in this field" is a legitimate
    annotation and must be distinguishable from "this coder has not answered".
    The latter is represented by the *absence of the payload*, not by an empty
    instance list, which is why the caller passes a mapping keyed by coder.
    """

    width: int
    height: int
    instances: tuple[InstanceGeometry, ...]
    masks: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    # ── construction ────────────────────────────────────────────────

    @classmethod
    def parse(cls, payload: bytes) -> "CoderPayload":
        """Read a geometry payload, refusing anything ambiguous.

        Strict on purpose. A tolerant reader that defaults ``width`` to 512 would
        produce a mask that is silently the wrong scale, and every downstream
        metric would then be a number rather than a measurement.
        """
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GeometryError(f"payload is not UTF-8 JSON: {exc}") from exc
        if not isinstance(data, Mapping):
            raise GeometryError(f"payload must be a JSON object, got {type(data).__name__}")

        version = data.get("format")
        if version != GEOMETRY_FORMAT:
            raise GeometryError(
                f"payload format {version!r} is not {GEOMETRY_FORMAT!r}; refusing to "
                "guess at field meanings"
            )
        unknown = set(data) - {"format", "width", "height", "instances"}
        if unknown:
            raise GeometryError(
                f"payload has unknown top-level key(s) {sorted(unknown)}; a tolerant "
                "reader would drop them and the information would be lost without an error"
            )

        width = _positive_int("width", data.get("width"))
        height = _positive_int("height", data.get("height"))
        if width > MAX_CANVAS or height > MAX_CANVAS:
            raise GeometryError(f"canvas {width}×{height} exceeds the {MAX_CANVAS} limit")

        raw_instances = data.get("instances")
        if not isinstance(raw_instances, list):
            raise GeometryError("instances must be a JSON array (empty is allowed)")

        instances: list[InstanceGeometry] = []
        masks: dict[str, np.ndarray] = {}
        seen: set[str] = set()
        for index, raw in enumerate(raw_instances):
            instance = _parse_instance(raw, index, width, height)
            if instance.id in seen:
                raise GeometryError(f"duplicate instance id {instance.id!r}")
            seen.add(instance.id)
            mask = rasterize_polygon(instance.polygon, width, height)
            if not mask.any():
                raise GeometryError(
                    f"instance {instance.id!r} rasterises to an empty mask; a zero-area "
                    "object is a data error, not an empty annotation"
                )
            instances.append(instance)
            masks[instance.id] = mask
        return cls(width=width, height=height, instances=tuple(instances), masks=masks)

    # ── queries ─────────────────────────────────────────────────────

    def mask(self, instance_id: str) -> np.ndarray:
        try:
            return self.masks[instance_id]
        except KeyError as exc:
            raise GeometryError(f"no mask for instance {instance_id!r}") from exc

    def union_mask(self) -> np.ndarray:
        """Foreground of this coder's whole product (any object)."""
        out = np.zeros((self.height, self.width), dtype=bool)
        for mask in self.masks.values():
            out |= mask
        return out

    @property
    def class_keys(self) -> tuple[str, ...]:
        return tuple(i.class_key for i in self.instances)

    def confidence_values(self) -> tuple[float, ...]:
        return tuple(i.confidence for i in self.instances if i.confidence is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": GEOMETRY_FORMAT,
            "width": self.width,
            "height": self.height,
            "instances": [
                {
                    "id": i.id,
                    "class_key": i.class_key,
                    "polygon": [[x, y] for x, y in i.polygon],
                    **({"confidence": i.confidence} if i.confidence is not None else {}),
                }
                for i in self.instances
            ],
        }


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GeometryError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise GeometryError(f"{name} must be positive, got {value}")
    return value


def _parse_instance(raw: Any, index: int, width: int, height: int) -> InstanceGeometry:
    if not isinstance(raw, Mapping):
        raise GeometryError(f"instances[{index}] must be an object")
    instance_id = raw.get("id")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise GeometryError(f"instances[{index}].id must be a non-empty string")
    class_key = raw.get("class_key")
    if not isinstance(class_key, str) or not class_key.strip():
        raise GeometryError(f"instances[{index}].class_key must be a non-empty string")

    confidence = raw.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise GeometryError(f"instances[{index}].confidence must be a number")
        if not (0.0 <= float(confidence) <= 1.0):
            raise GeometryError(f"instances[{index}].confidence must be within [0, 1]")

    if "polygon" in raw:
        points = _parse_polygon(raw["polygon"], index, width, height)
    elif "rect" in raw:
        points = _parse_rect(raw["rect"], index, width, height)
    else:
        raise GeometryError(f"instances[{index}] needs either 'polygon' or 'rect'")
    return InstanceGeometry(
        id=instance_id,
        class_key=class_key,
        polygon=points,
        confidence=None if confidence is None else float(confidence),
    )


def _parse_rect(raw: Any, index: int, width: int, height: int) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise GeometryError(f"instances[{index}].rect must be [x0, y0, x1, y1]")
    try:
        x0, y0, x1, y1 = (float(v) for v in raw)
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"instances[{index}].rect must be numeric") from exc
    if x1 <= x0 or y1 <= y0:
        raise GeometryError(f"instances[{index}].rect must have x1 > x0 and y1 > y0")
    _check_bounds([(x0, y0), (x1, y1)], index, width, height)
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _parse_polygon(raw: Any, index: int, width: int, height: int) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, (list, tuple)):
        raise GeometryError(f"instances[{index}].polygon must be an array of points")
    if len(raw) < 3:
        raise GeometryError(f"instances[{index}].polygon needs at least 3 points, got {len(raw)}")
    points: list[tuple[float, float]] = []
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise GeometryError(f"instances[{index}].polygon points must be [x, y]")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise GeometryError(f"instances[{index}].polygon has a non-numeric point") from exc
        points.append((x, y))
    _check_bounds(points, index, width, height)
    return tuple(points)


def _check_bounds(points: Sequence[tuple[float, float]], index: int, width: int, height: int) -> None:
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise GeometryError(f"instances[{index}] has a non-finite coordinate")
        if x < 0 or x > width or y < 0 or y > height:
            raise GeometryError(
                f"instances[{index}] has a point ({x}, {y}) outside the {width}×{height} "
                "canvas. Clamping would silently move the annotation, so this is refused."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Rasterisation
# ─────────────────────────────────────────────────────────────────────────────


def rasterize_polygon(
    points: Sequence[tuple[float, float]], width: int, height: int
) -> np.ndarray:
    """Fill a polygon by even-odd scanline at pixel centres.

    Pixel centres rather than pixel corners, because every other measurement in
    this repository (``analysis.morphometrics``) treats a pixel as a unit square
    and its centre as the sample point; mixing conventions would put a half-pixel
    bias into every Dice comparison.
    """
    mask = np.zeros((height, width), dtype=bool)
    pts = list(points)
    n = len(pts)
    centres_y = np.arange(height) + 0.5
    centres_x = np.arange(width) + 0.5
    for row, y in enumerate(centres_y):
        crossings: list[float] = []
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if (y1 <= y < y2) or (y2 <= y < y1):
                t = (y - y1) / (y2 - y1)
                crossings.append(x1 + t * (x2 - x1))
        if not crossings:
            continue
        crossings.sort()
        if len(crossings) % 2:
            raise GeometryError(
                "polygon has an odd number of scanline crossings; it is not a simple "
                "closed ring and its area is ambiguous"
            )
        for k in range(0, len(crossings), 2):
            lo, hi = crossings[k], crossings[k + 1]
            mask[row] |= (centres_x >= lo) & (centres_x < hi)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Matching and per-pair metrics
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MatchPair:
    """Two instances judged to be the same object."""

    a: str
    b: str
    iou: float
    dice: float
    hd95: float
    assd: float
    same_class: bool


@dataclass(frozen=True)
class MatchResult:
    """The complete comparison of two coders' products for one sample."""

    pairs: tuple[MatchPair, ...]
    unmatched_a: tuple[str, ...]
    unmatched_b: tuple[str, ...]
    iou_threshold: float

    @property
    def tp(self) -> int:
        return len(self.pairs)

    @property
    def fp(self) -> int:
        return len(self.unmatched_b)

    @property
    def fn(self) -> int:
        return len(self.unmatched_a)

    @property
    def detection_precision(self) -> Optional[float]:
        denom = self.tp + self.fp
        return None if denom == 0 else self.tp / denom

    @property
    def detection_recall(self) -> Optional[float]:
        denom = self.tp + self.fn
        return None if denom == 0 else self.tp / denom

    @property
    def detection_f1(self) -> Optional[float]:
        denom = 2 * self.tp + self.fp + self.fn
        return None if denom == 0 else 2 * self.tp / denom

    def _mean(self, values: Iterable[float]) -> Optional[float]:
        collected = list(values)
        return None if not collected else sum(collected) / len(collected)

    @property
    def dice_mean(self) -> Optional[float]:
        return self._mean(p.dice for p in self.pairs)

    @property
    def iou_mean(self) -> Optional[float]:
        return self._mean(p.iou for p in self.pairs)

    @property
    def hd95_mean(self) -> Optional[float]:
        return self._mean(p.hd95 for p in self.pairs)

    @property
    def assd_mean(self) -> Optional[float]:
        return self._mean(p.assd for p in self.pairs)

    @property
    def class_agreement(self) -> Optional[float]:
        if not self.pairs:
            return None
        return sum(1 for p in self.pairs if p.same_class) / len(self.pairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "detection_precision": self.detection_precision,
            "detection_recall": self.detection_recall,
            "detection_f1": self.detection_f1,
            "dice_mean": self.dice_mean,
            "iou_mean": self.iou_mean,
            "hd95_mean": self.hd95_mean,
            "assd_mean": self.assd_mean,
            "class_agreement": self.class_agreement,
            "unmatched_a": list(self.unmatched_a),
            "unmatched_b": list(self.unmatched_b),
        }


def dice_and_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float]:
    inter = int(np.count_nonzero(mask_a & mask_b))
    total = int(np.count_nonzero(mask_a)) + int(np.count_nonzero(mask_b))
    union = total - inter
    dice = 1.0 if total == 0 else 2.0 * inter / total
    iou = 1.0 if union == 0 else inter / union
    return dice, iou


def _surface(mask: np.ndarray) -> np.ndarray:
    """Pixels of ``mask`` with at least one 4-neighbour outside the mask."""
    padded = np.pad(mask, 1, constant_values=False)
    h, w = mask.shape
    interior = np.ones_like(mask, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        interior &= padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    return mask & ~interior


def _surface_points(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(_surface(mask))
    if rows.size == 0:  # a single-pixel or fully-degenerate mask
        rows, cols = np.nonzero(mask)
    return np.column_stack((rows, cols)).astype(float)


def _subsample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    stride = math.ceil(len(points) / limit)
    return points[::stride]


def hausdorff95(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float]:
    """(HD95, ASSD) in pixels, both directions pooled.

    Pooling rather than taking the max of two directed percentiles: the
    directed 95th percentile of A→B and B→A answer different questions, and
    taking the max of two 95th percentiles is not the 95th percentile of the
    Hausdorff distribution. Pooling is the convention in the segmentation
    literature and is what the bootstrap below assumes.
    """
    a = _subsample(_surface_points(mask_a), MAX_SURFACE_POINTS)
    b = _subsample(_surface_points(mask_b), MAX_SURFACE_POINTS)
    if len(a) == 0 or len(b) == 0:
        raise GeometryError("cannot measure a surface distance against an empty mask")
    distances = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    forward = distances.min(axis=1)
    backward = distances.min(axis=0)
    pooled = np.concatenate((forward, backward))
    return float(np.percentile(pooled, 95)), float(pooled.mean())


def match_payloads(
    a: CoderPayload, b: CoderPayload, *, iou_threshold: float = 0.5
) -> MatchResult:
    """Greedy one-to-one matching at a fixed IoU threshold (PRD 6.1).

    Greedy rather than optimal assignment: Hungarian matching optimises total
    IoU, which can pair a strong match with a weak one to improve the sum. For
    detection metrics the threshold is the decision, so taking matches in
    descending IoU order is both cheaper and closer to what a human reviewer
    would call the correspondence.
    """
    if (a.width, a.height) != (b.width, b.height):
        raise GeometryError(
            f"canvases differ ({a.width}×{a.height} vs {b.width}×{b.height}); "
            "comparing them would rescale one annotation without saying so"
        )
    candidates: list[tuple[float, str, str]] = []
    for ia in a.instances:
        mask_a = a.mask(ia.id)
        for ib in b.instances:
            _dice, iou = dice_and_iou(mask_a, b.mask(ib.id))
            if iou >= iou_threshold:
                candidates.append((iou, ia.id, ib.id))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_a: set[str] = set()
    used_b: set[str] = set()
    pairs: list[MatchPair] = []
    class_of_a = {i.id: i.class_key for i in a.instances}
    class_of_b = {i.id: i.class_key for i in b.instances}
    for iou, id_a, id_b in candidates:
        if id_a in used_a or id_b in used_b:
            continue
        used_a.add(id_a)
        used_b.add(id_b)
        mask_a, mask_b = a.mask(id_a), b.mask(id_b)
        dice, _ = dice_and_iou(mask_a, mask_b)
        hd95, assd = hausdorff95(mask_a, mask_b)
        pairs.append(
            MatchPair(
                a=id_a,
                b=id_b,
                iou=iou,
                dice=dice,
                hd95=hd95,
                assd=assd,
                same_class=class_of_a[id_a] == class_of_b[id_b],
            )
        )
    return MatchResult(
        pairs=tuple(pairs),
        unmatched_a=tuple(sorted({i.id for i in a.instances} - used_a)),
        unmatched_b=tuple(sorted({i.id for i in b.instances} - used_b)),
        iou_threshold=iou_threshold,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Attribute agreement
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Unit:
    """One *object* as seen by however many coders annotated it.

    Units are the rows of the attribute analysis. Building them from IoU-linked
    instances rather than by index is what lets a coder omit an object without
    the omission turning into a spurious attribute disagreement: the unit simply
    has a missing value for that coder, which is exactly what Krippendorff's α
    is defined to handle.
    """

    unit_id: int
    members: Mapping[str, tuple[str, ...]]
    values: Mapping[str, str]
    ambiguous: tuple[str, ...] = ()

    @property
    def coders(self) -> tuple[str, ...]:
        return tuple(sorted(self.values))


def build_units(
    payloads: Mapping[str, CoderPayload], *, iou_threshold: float = 0.5
) -> tuple[Unit, ...]:
    """Link instances across coders by IoU, then read off one value per coder."""
    coders = sorted(payloads)
    if not coders:
        return ()
    shapes = {(p.width, p.height) for p in payloads.values()}
    if len(shapes) > 1:
        raise GeometryError(f"coders disagree about the canvas size: {sorted(shapes)}")

    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(x: tuple[str, str], y: tuple[str, str]) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for payload_coder, payload in payloads.items():
        for instance in payload.instances:
            find((payload_coder, instance.id))

    for i, coder_a in enumerate(coders):
        for coder_b in coders[i + 1 :]:
            payload_a, payload_b = payloads[coder_a], payloads[coder_b]
            for instance_a in payload_a.instances:
                mask_a = payload_a.mask(instance_a.id)
                for instance_b in payload_b.instances:
                    _dice, iou = dice_and_iou(mask_a, payload_b.mask(instance_b.id))
                    if iou >= iou_threshold:
                        union((coder_a, instance_a.id), (coder_b, instance_b.id))

    grouped: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for node in list(parent):
        grouped[find(node)].append(node)

    class_key_of: dict[tuple[str, str], str] = {}
    for coder, payload in payloads.items():
        for instance in payload.instances:
            class_key_of[(coder, instance.id)] = instance.class_key

    units: list[Unit] = []
    for unit_id, nodes in enumerate(sorted(grouped.values())):
        members: dict[str, list[str]] = defaultdict(list)
        for coder, instance_id in sorted(nodes):
            members[coder].append(instance_id)
        values: dict[str, str] = {}
        ambiguous: list[str] = []
        for coder, ids in sorted(members.items()):
            keys = [class_key_of[(coder, instance_id)] for instance_id in ids]
            if len(ids) == 1:
                values[coder] = keys[0]
                continue
            counts = Counter(keys)
            top = max(counts.values())
            winners = sorted(k for k, v in counts.items() if v == top)
            values[coder] = winners[0]
            ambiguous.append(coder)
        units.append(
            Unit(
                unit_id=unit_id,
                members={c: tuple(sorted(ids)) for c, ids in sorted(members.items())},
                values=values,
                ambiguous=tuple(sorted(ambiguous)),
            )
        )
    return tuple(units)


def krippendorff_alpha_nominal(
    units: Sequence[Sequence[Optional[str]]],
) -> Optional[float]:
    """Nominal Krippendorff's α over units × coders, with missing values.

    Returns ``None`` when fewer than two units have two values. One pairable
    unit is not a small sample of agreement, it is no evidence at all — and both
    "α = 1.0" and "α = 0.0" would be invented numbers that a downstream threshold
    would happily consume. Zero is additionally indistinguishable from "coders
    disagree completely", the opposite reading of the same value. ``1.0`` is
    returned when observed disagreement is zero *and* expected disagreement is
    zero (all coders used one category throughout), which is Krippendorff's own
    convention.
    """
    pairable: list[list[str]] = []
    for unit in units:
        values = [v for v in unit if v is not None]
        if len(values) >= 2:
            pairable.append(values)
    if len(pairable) < 2:
        return None

    coincidence: dict[tuple[str, str], float] = defaultdict(float)
    for values in pairable:
        m = len(values)
        for i, vi in enumerate(values):
            for j, vj in enumerate(values):
                if i != j:
                    coincidence[(vi, vj)] += 1.0 / (m - 1)
    n = sum(coincidence.values())
    if n == 0:
        return None
    marginals: Counter[str] = Counter()
    for (vi, _vj), weight in coincidence.items():
        marginals[vi] += weight
    observed = sum(w for (vi, vj), w in coincidence.items() if vi != vj) / n
    expected = (
        sum(
            marginals[vi] * marginals[vj]
            for vi in marginals
            for vj in marginals
            if vi != vj
        )
        / (n * (n - 1))
    )
    if expected == 0:
        return 1.0
    return 1.0 - observed / expected


#: Fixed seed. PRD 4.5 requires the bootstrap interval to be reproducible, which
#: it is not if the seed is drawn from the clock.
BOOTSTRAP_SEED = 20261105


def bootstrap_alpha_ci(
    units: Sequence[Sequence[Optional[str]]],
    *,
    resamples: int = 1000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Percentile bootstrap of α over units. Reproducible for a fixed seed."""
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    point = krippendorff_alpha_nominal(units)
    if point is None:
        return {
            "alpha": None,
            "ci_low": None,
            "ci_high": None,
            "resamples": 0,
            "seed": seed,
            "note": "undefined: no unit has two coders",
        }
    rng = np.random.default_rng(seed)
    pairable = [list(u) for u in units]
    samples: list[float] = []
    index = np.arange(len(pairable))
    for _ in range(resamples):
        draw = rng.choice(index, size=len(pairable), replace=True)
        value = krippendorff_alpha_nominal([pairable[i] for i in draw])
        if value is not None:
            samples.append(value)
    if not samples:
        return {
            "alpha": point,
            "ci_low": None,
            "ci_high": None,
            "resamples": 0,
            "seed": seed,
            "note": "every resample was undefined",
        }
    return {
        "alpha": point,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "resamples": len(samples),
        "seed": seed,
    }


def raw_agreement(units: Sequence[Sequence[Optional[str]]]) -> Optional[float]:
    """Share of units on which every coder that annotated agreed.

    Reported next to α rather than instead of it (PRD 6.1): raw agreement says
    how often people agreed, α says how much of that is better than chance.
    """
    counted = [u for u in units if len([v for v in u if v is not None]) >= 2]
    if not counted:
        return None
    agree = sum(1 for u in counted if len({v for v in u if v is not None}) == 1)
    return agree / len(counted)


def positive_negative_agreement(
    units: Sequence[Sequence[Optional[str]]],
) -> dict[str, Any]:
    """Per-class positive / negative agreement (Byrt et al.), macro-averaged.

    Reported because κ (and α) collapse when one category dominates — the Kappa
    paradox PRD 6.1 names. Positive agreement answers "when the object is there,
    do coders agree it is there"; negative agreement answers the same for
    absence, and the two diverging is the signal that a single coefficient hides.
    """
    labels = sorted({v for unit in units for v in unit if v is not None})
    per_class: dict[str, dict[str, Optional[float]]] = {}
    for label in labels:
        a = b = c = d = 0
        for unit in units:
            present = [v for v in unit if v is not None]
            if len(present) < 2:
                continue
            positive = [v == label for v in present]
            for i in range(len(positive)):
                for j in range(i + 1, len(positive)):
                    pi, pj = positive[i], positive[j]
                    if pi and pj:
                        a += 1
                    elif pi or pj:
                        b += 1 if pi else 0
                        c += 1 if pj else 0
                    else:
                        d += 1
        pos_denom = 2 * a + b + c
        neg_denom = 2 * d + b + c
        per_class[label] = {
            "positive": None if pos_denom == 0 else 2 * a / pos_denom,
            "negative": None if neg_denom == 0 else 2 * d / neg_denom,
            "co_positive_pairs": a,
            "discordant_pairs": b + c,
            "co_negative_pairs": d,
        }
    positives = [v["positive"] for v in per_class.values() if v["positive"] is not None]
    negatives = [v["negative"] for v in per_class.values() if v["negative"] is not None]
    return {
        "per_class": per_class,
        "macro_positive": None if not positives else sum(positives) / len(positives),
        "macro_negative": None if not negatives else sum(negatives) / len(negatives),
    }
