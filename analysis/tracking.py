# ── analysis/tracking.py ──
"""Instance tracking across time — the time-lapse dimension.

Why this matters more for organoids than for cells
--------------------------------------------------
A single organoid field of view is not a snapshot. Organoid assays are
routinely acquired as time-lapse series because the biologically interesting
events are *transitions*: a cystic non-budding organoid acquiring its first
bud, a bud resolving into a spheroid, a lineage expanding. A classification
label on one frame answers "what is this now"; the scientific question is
usually "what did it become, and how fast".

None of that is reachable without identity. The repository had no tracking
module at all: ``analysis/detector.py`` and ``analysis/segmentor.py`` both
returned per-image objects with no link across frames, so the time axis could
be analysed but not the objects on it.

What this provides
------------------
Deterministic, dependency-light linking of instances between time points:

* matching on mask IoU when masks exist, falling back to physical centroid
  distance when they do not;
* tolerance for frames where an object was missed by the detector
  (``max_frame_gap``), which matters because a mitotic or dying organoid
  frequently drops below threshold for a frame or two;
* division-event detection, recording lineage in :attr:`Track.lineage_parent`
  and :attr:`Track.division_events`.

Identity is sticky: once an instance is assigned a ``track_id`` the assignment
is stable, so re-running with different parameters does not silently relabel a
lineage.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence

import numpy as np

from analysis.schema import Instance, Track, new_id

__all__ = [
    "TrackingConfig",
    "TrackingResult",
    "centroid_distance",
    "iou",
    "link_tracks",
]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrackingConfig:
    """Linking thresholds.

    Distances can be expressed in physical units, which is the point of
    carrying a pixel size at all: ``max_centroid_dist_um=5`` means "an organoid
    cannot plausibly have moved more than 5 µm between frames", a statement
    that stays true if the acquisition is rescaled. A pixel-space threshold
    silently changes meaning whenever the objective or binning changes.
    """

    #: Minimum mask IoU for a mask-based match.
    min_iou: float = 0.1
    #: Maximum centroid displacement in micrometres. Requires pixel_size_um.
    #: When no scale is supplied, set :attr:`max_centroid_dist_px` instead —
    #: otherwise distance gating is silently disabled and only IoU links
    #: objects. :func:`link_tracks` warns when that happens.
    max_centroid_dist_um: Optional[float] = 5.0
    #: Maximum centroid displacement in pixels, used when no scale is known.
    max_centroid_dist_px: Optional[float] = None
    #: How many consecutive missed frames a track may survive.
    max_frame_gap: int = 1
    #: Detect one-parent-to-two-children division events.
    allow_division: bool = True
    #: A daughter must be no larger than this fraction of the parent. Two
    #: daughters each as large as the parent is a matching artefact, not
    #: biology — the default of 0.8 lets each daughter be slightly large
    #: (segmentation bleed) while still rejecting an equal-size "split".
    division_max_child_area_ratio: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_iou <= 1.0:
            raise ValueError(f"min_iou must be in [0, 1], got {self.min_iou}")
        if self.max_frame_gap < 0:
            raise ValueError("max_frame_gap must be non-negative")


@dataclass
class TrackingResult:
    """Tracks plus the new instance records that reference them."""

    tracks: list[Track] = field(default_factory=list)
    #: ``instance_id -> track_id`` for every linked instance.
    assignments: dict[str, str] = field(default_factory=dict)
    #: Instance records with ``track_id`` (and lineage links) populated.
    instances: list[Instance] = field(default_factory=list)
    #: ``(parent_track_id, child_track_id, t_index)`` for each division.
    divisions: list[tuple[str, str, int]] = field(default_factory=list)

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    @property
    def n_divisions(self) -> int:
        return len(self.divisions)

    @property
    def mean_length(self) -> float:
        if not self.tracks:
            return 0.0
        return sum(t.length for t in self.tracks) / len(self.tracks)

    def to_dict(self) -> dict:
        return {
            "n_tracks": self.n_tracks,
            "n_divisions": self.n_divisions,
            "mean_length": round(self.mean_length, 3),
            "tracks": [t.to_dict() for t in self.tracks],
            "divisions": [
                {"parent": p, "child": c, "t_index": t} for p, c, t in self.divisions
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────


def _mask_of(instance: Instance) -> Optional[np.ndarray]:
    mask = instance.mask
    if mask is None:
        return None
    arr = np.asarray(mask)
    return arr.astype(bool) if arr.ndim == 2 else None


def iou(a: Instance, b: Instance) -> Optional[float]:
    """Intersection-over-union of two instance masks.

    Returns ``None`` when either instance has no usable mask, so callers can
    distinguish "no overlap" (0.0) from "cannot be computed".
    """
    ma, mb = _mask_of(a), _mask_of(b)
    if ma is None or mb is None:
        return None
    if ma.shape != mb.shape:
        return None
    union = int(np.logical_or(ma, mb).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(ma, mb).sum()) / union


def centroid_distance(a: Instance, b: Instance, *, pixel_size_um: Optional[float] = None) -> float:
    """Euclidean centroid distance, in µm when a scale is available.

    Falls back to the bbox centre when an instance has no measurement, so
    detector-only pipelines can still be tracked.
    """
    pa = _centroid_of(a)
    pb = _centroid_of(b)
    if pa is None or pb is None:
        return float("inf")
    dist_px = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
    return dist_px * pixel_size_um if pixel_size_um else dist_px


def _centroid_of(instance: Instance) -> Optional[tuple[float, float]]:
    if instance.measurement is not None and instance.measurement.centroid_px is not None:
        return instance.measurement.centroid_px
    if instance.bbox_px is not None:
        x0, y0, x1, y1 = instance.bbox_px
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    mask = _mask_of(instance)
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        return (float(xs.mean()), float(ys.mean()))
    return None


def _area_of(instance: Instance) -> float:
    if instance.measurement is not None and instance.measurement.area_px:
        return float(instance.measurement.area_px)
    mask = _mask_of(instance)
    if mask is not None:
        return float(mask.sum())
    if instance.bbox_px is not None:
        x0, y0, x1, y1 = instance.bbox_px
        return float(max(0.0, x1 - x0) * max(0.0, y1 - y0))
    return 0.0


def _pair_cost(
    a: Instance,
    b: Instance,
    *,
    config: TrackingConfig,
    pixel_size_um: Optional[float],
) -> Optional[float]:
    """Cost in ``[0, 2]`` for linking ``a`` to ``b``, or ``None`` if ineligible.

    A pair is eligible if **either** the masks overlap enough **or** the
    centroid displacement is plausible. Requiring both would drop genuinely
    moving objects: a young organoid can drift more than one radius between
    frames, so its IoU with its own previous position falls below any useful
    threshold while the trajectory is still perfectly unambiguous.

    Overlap matches are preferred by construction — they score below 1.0 while
    distance-only matches score at or above 1.0 — so a better geometric
    candidate always wins.
    """
    overlap = iou(a, b)

    dist = centroid_distance(a, b, pixel_size_um=pixel_size_um)
    limit = (
        config.max_centroid_dist_um
        if (pixel_size_um is not None and config.max_centroid_dist_um is not None)
        else config.max_centroid_dist_px
    )

    overlap_ok = overlap is not None and overlap >= config.min_iou
    distance_ok = (
        limit is not None and np.isfinite(dist) and dist <= limit
    )

    if not overlap_ok and not distance_ok:
        return None

    if overlap_ok:
        # 0.0 (identical masks) .. just under 1.0
        return float(1.0 - overlap)

    # Distance-only fallback, ranked strictly after any overlap match.
    return float(1.0 + min(dist / limit, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Linking
# ─────────────────────────────────────────────────────────────────────────────


def link_tracks(
    frames: Mapping[int, Sequence[Instance]],
    *,
    config: Optional[TrackingConfig] = None,
    pixel_size_um: Optional[float] = None,
    specimen_id: Optional[str] = None,
) -> TrackingResult:
    """Link instances across time points into tracks.

    Args:
        frames: ``{t_index: [Instance, ...]}``. Need not be contiguous.
        config: linking thresholds.
        pixel_size_um: scale used for physical distance gates.
        specimen_id: carried onto each produced :class:`Track`.

    Returns:
        A :class:`TrackingResult` holding the tracks, the id assignments, and
        new :class:`Instance` records with ``track_id`` populated (the inputs
        are frozen and left untouched).
    """
    cfg = config or TrackingConfig()
    if not frames:
        return TrackingResult()

    if (
        pixel_size_um is None
        and cfg.max_centroid_dist_um is not None
        and cfg.max_centroid_dist_px is None
    ):
        warnings.warn(
            "link_tracks: max_centroid_dist_um is set but no pixel_size_um was "
            "supplied and max_centroid_dist_px is unset, so distance gating is "
            "disabled and objects will only be linked by mask IoU. Pass "
            "pixel_size_um, or set max_centroid_dist_px for a pixel-space gate.",
            RuntimeWarning,
            stacklevel=2,
        )

    times = sorted(frames)

    # track_id -> (last instance, last t index)
    #
    # This used to carry a third element, a count of consecutive missed frames.
    # It was incremented whenever an empty frame aged the tracks but never read:
    # eligibility is decided by ``t - last_t`` against :attr:`max_frame_gap`,
    # which is strictly more general because it also covers frames that were
    # delivered but in which the object was not detected. Two competing notions
    # of "gap" where only one is consulted is a trap for the next reader, so the
    # unused counter is gone.
    active: dict[str, tuple[Instance, int]] = {}
    # track_id -> the child instances created from a division, pending adoption
    created: dict[str, Track] = {}
    updated: dict[str, Instance] = {}
    divisions: list[tuple[str, str, int]] = []

    for t in times:
        instances = [i for i in frames[t]]
        if not instances:
            # A frame with no detections ages every active track.
            continue

        # ── build and sort all eligible (track, instance) pairs by cost ──
        pairs: list[tuple[float, str, int]] = []
        for tid, (last_instance, last_t) in active.items():
            if t - last_t - 1 > cfg.max_frame_gap:
                continue
            for idx, candidate in enumerate(instances):
                if candidate.z_index != last_instance.z_index:
                    continue
                cost = _pair_cost(
                    last_instance, candidate, config=cfg, pixel_size_um=pixel_size_um
                )
                if cost is not None:
                    pairs.append((cost, tid, idx))

        pairs.sort(key=lambda p: (p[0], p[1], p[2]))

        matched_tracks: set[str] = set()
        matched_instances: dict[int, str] = {}
        handled_idx: set[int] = set()
        # A track that would take two or more instances in the same frame is a
        # division event, not a continuation.
        extra_hits: dict[str, list[int]] = {}

        # `pairs` is already sorted by cost, so the cost itself is not consulted
        # here — only the ordering it produced. Keeping it in the pattern that
        # way makes the tuple shape obvious without implying a use.
        for _cost, tid, idx in pairs:
            if idx in matched_instances:
                continue
            if tid not in matched_tracks:
                matched_tracks.add(tid)
                matched_instances[idx] = tid
                continue
            if cfg.allow_division:
                extra_hits.setdefault(tid, []).append(idx)

        # ── continuation: exactly one candidate matched this track ──
        for idx, tid in matched_instances.items():
            if tid in extra_hits:
                # Divisions are handled below — every daughter becomes its own
                # track, including the one that happened to match first. Cell
                # lineage convention: a divided parent does not continue.
                continue
            inst = instances[idx]
            last_instance, _ = active[tid]
            created[tid].add(inst.id, t)
            updated[inst.id] = replace(
                inst,
                track_id=tid,
                parent_instance_id=last_instance.id,
            )
            active[tid] = (inst, t)
            handled_idx.add(idx)

        # ── division: one parent, several daughters in one frame ──
        for tid, extra_idxs in extra_hits.items():
            parent_track = created[tid]
            parent_instance, _ = active[tid]
            parent_area = _area_of(parent_instance)

            daughter_idxs = [i for i, owner in matched_instances.items() if owner == tid]
            daughter_idxs.extend(extra_idxs)

            for idx in daughter_idxs:
                daughter = instances[idx]
                daughter_area = _area_of(daughter)

                # A "division" whose daughters are each as large as the parent
                # is a matching artefact. Leave those instances unhandled so
                # they start independent tracks.
                if (
                    parent_area > 0
                    and daughter_area > parent_area * cfg.division_max_child_area_ratio
                ):
                    continue

                child_track_id = new_id("trk")
                child_track = Track(
                    id=child_track_id,
                    specimen_id=specimen_id,
                    lineage_parent=tid,
                )
                child_track.add(daughter.id, t)
                created[child_track_id] = child_track
                parent_track.record_division(t, child_track_id)
                divisions.append((tid, child_track_id, t))
                updated[daughter.id] = replace(
                    daughter,
                    track_id=child_track_id,
                    parent_instance_id=parent_instance.id,
                )
                active[child_track_id] = (daughter, t)
                handled_idx.add(idx)

            # The parent track terminates at the division.
            del active[tid]

        # ── unmatched instances start new tracks ──
        for idx, inst in enumerate(instances):
            if idx in handled_idx:
                continue

            # Sticky identity: an instance that already carries a track id keeps
            # it, so re-running on previously tracked data is idempotent rather
            # than relabelling the same lineage with fresh ids on every pass.
            tid = inst.track_id or new_id("trk")
            track = created.get(tid)
            if track is None:
                track = Track(id=tid, specimen_id=specimen_id)
                created[tid] = track
            # Either way the instance is appended to the track the annotation
            # named — the carrier of the id is asserting identity, so it is not
            # this function's place to overrule it.
            active[tid] = (inst, t)
            track.add(inst.id, t)
            updated[inst.id] = replace(inst, track_id=tid)
            handled_idx.add(idx)

        # ── retire tracks that have been silent too long ──
        stale = [
            tid for tid, (_, last_t) in active.items()
            if t - last_t > cfg.max_frame_gap
        ]
        for tid in stale:
            del active[tid]

    tracks = sorted(
        created.values(),
        key=lambda tr: (tr.start_t, tr.end_t, tr.id),
    )

    assignments = {iid: inst.track_id for iid, inst in updated.items() if inst.track_id}

    # Preserve every input instance, adding the tracked variants in place of
    # their originals so callers get a complete, consistent list.
    ordered: list[Instance] = []
    for t in times:
        for inst in frames[t]:
            ordered.append(updated.get(inst.id, inst))

    return TrackingResult(
        tracks=tracks,
        assignments=assignments,
        instances=ordered,
        divisions=divisions,
    )
