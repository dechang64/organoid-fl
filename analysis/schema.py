# ── analysis/schema.py ──
"""Organoid-native data model.

Why this module exists
----------------------
The platform's original object model was two flat 2-D dataclasses:

    analysis/detector.py:26-41    Detection(bbox, class_name, confidence, ...)
    analysis/segmentor.py:27-38   SegmentationResult(mask, area, circularity, ...)

Neither could express anything that makes an organoid specimen different from
a generic cell image:

* no **channel** identity, so multi-channel fluorescence collapses to RGB
* no **Z index**, so confocal stacks collapse to a single plane
* no **time index**, so time-lapse (the core organoid assay) is unsupported
* no **physical scale**, so ``area`` is a pixel count and every morphology
  metric is unit-less — circularity and solidity cannot be compared across
  microscopes or magnifications
* no **provenance**, so a model pre-annotation, a human edit, and a value
  propagated from a neighbouring frame are indistinguishable
* no **track identity**, so an instance cannot be followed across time

This module supplies those dimensions. Values are immutable dataclasses so a
measurement cannot be silently mutated after a review is signed off.

Physical units
--------------
Every geometric field on :class:`Measurement` is in micrometres (``_um``) or
square micrometres (``_um2``); pixel-valued companions carry an explicit
``_px`` suffix. If :attr:`Image.pixel_size_um` is unknown the geometry is left
as ``None`` rather than silently reported in pixels under a ``_um`` name —
guessing here is how unit errors reach a publication.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional

__all__ = [
    "ChannelSpec",
    "Image",
    "Instance",
    "Measurement",
    "Modality",
    "Provenance",
    "ReviewStatus",
    "Specimen",
    "Track",
    "new_id",
]


def new_id(prefix: str) -> str:
    """Short, collision-safe identifier such as ``inst_3f9a2c1d``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class Provenance(str, Enum):
    """Where an annotation came from.

    Kept explicit so a downstream analysis can say "trained on human labels
    only" or "evaluated against adjudicated consensus" without re-deriving it
    from logs. This matters because model pre-annotations and human edits have
    very different error profiles and must not be silently pooled.
    """

    HUMAN = "human"
    MODEL_PREANNOTATION = "model_preannotation"
    PROPAGATED = "propagated"
    CONSENSUS = "consensus"
    IMPORTED = "imported"
    SYNTHETIC = "synthetic"
    #: PRD 7.4. For sites that insist the model picks the final label. Recorded
    #: honestly so the cost is measurable, and excluded from ground truth: a
    #: label the model chose cannot be evidence about that model.
    MODEL_SELECTED = "model_selected"

    @property
    def is_ground_truth(self) -> bool:
        """True only for provenance that may serve as evaluation ground truth."""
        return self in (Provenance.HUMAN, Provenance.CONSENSUS, Provenance.IMPORTED)


class ReviewStatus(str, Enum):
    """Position in the annotation review lifecycle."""

    UNREVIEWED = "unreviewed"
    PRE_ANNOTATED = "pre_annotated"
    ANNOTATED = "annotated"
    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"
    ADJUDICATED = "adjudicated"
    REJECTED = "rejected"

    @property
    def is_signoff(self) -> bool:
        """True once the instance has passed independent review."""
        return self in (ReviewStatus.REVIEWED, ReviewStatus.ADJUDICATED)


class Modality(str, Enum):
    """Acquisition modality, which constrains what the data can support."""

    BRIGHTFIELD = "brightfield"
    PHASE_CONTRAST = "phase_contrast"
    FLUORESCENCE = "fluorescence"
    CONFOCAL = "confocal"
    LIGHT_SHEET = "light_sheet"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Channels
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChannelSpec:
    """One acquisition channel.

    ``name`` is the stable identifier persisted in records; ``wavelength_nm``
    is the emission/excitation centre when known, and ``stain`` records what
    was actually labelled (e.g. ``"DAPI"``, ``"GFP"``). Keeping the stain
    string is what lets a downstream task ask for "the nuclei channel" without
    depending on channel ordering, which differs across microscopes.
    """

    name: str
    index: int
    stain: Optional[str] = None
    wavelength_nm: Optional[float] = None
    is_reference: bool = False
    lookup_table: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("ChannelSpec.name must be a non-empty string")
        if self.index < 0:
            raise ValueError(f"ChannelSpec.index must be non-negative, got {self.index}")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "stain": self.stain,
            "wavelength_nm": self.wavelength_nm,
            "is_reference": self.is_reference,
            "lookup_table": self.lookup_table,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChannelSpec":
        return cls(
            name=str(payload["name"]),
            index=int(payload["index"]),
            stain=payload.get("stain"),
            wavelength_nm=payload.get("wavelength_nm"),
            is_reference=bool(payload.get("is_reference", False)),
            lookup_table=payload.get("lookup_table"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Measurement:
    """Calibrated morphology and intensity for a single instance.

    All ``_um`` / ``_um2`` fields are ``None`` when the parent image had no
    pixel size, and pixel-valued ``_px`` companions are always populated. A
    consumer that requires physical units must check and skip, not guess.
    """

    # ── geometry, physical units ──
    area_um2: Optional[float] = None
    perimeter_um: Optional[float] = None
    equivalent_diameter_um: Optional[float] = None
    major_axis_um: Optional[float] = None
    minor_axis_um: Optional[float] = None

    # ── geometry, dimensionless (require physical calibration to be meaningful) ──
    circularity: Optional[float] = None
    solidity: Optional[float] = None
    aspect_ratio: Optional[float] = None
    eccentricity: Optional[float] = None

    # ── geometry, raw pixels ──
    area_px: int = 0
    perimeter_px: Optional[float] = None

    # ── location ──
    centroid_px: Optional[tuple[float, float]] = None
    centroid_um: Optional[tuple[float, float]] = None
    z_index: int = 0
    t_index: int = 0

    # ── intensity, per channel name, computed on the ORIGINAL bit depth ──
    # Summed / averaged over the instance mask. Never computed on a
    # display-normalised 8-bit copy: that would destroy the dynamic range that
    # fluorescence quantification depends on.
    mean_intensity: Mapping[str, float] = field(default_factory=dict)
    integral_intensity: Mapping[str, float] = field(default_factory=dict)
    max_intensity: Mapping[str, float] = field(default_factory=dict)

    # ── mask composition ──
    is_multi_component: bool = False
    n_components: int = 1
    largest_component_fraction: Optional[float] = None
    has_holes: bool = False
    euler_number: Optional[int] = None

    # ── quality gates ──
    touches_border: bool = False

    def to_dict(self) -> dict:
        return {
            "area_um2": self.area_um2,
            "perimeter_um": self.perimeter_um,
            "equivalent_diameter_um": self.equivalent_diameter_um,
            "major_axis_um": self.major_axis_um,
            "minor_axis_um": self.minor_axis_um,
            "circularity": self.circularity,
            "solidity": self.solidity,
            "aspect_ratio": self.aspect_ratio,
            "eccentricity": self.eccentricity,
            "area_px": self.area_px,
            "perimeter_px": self.perimeter_px,
            "centroid_px": list(self.centroid_px) if self.centroid_px else None,
            "centroid_um": list(self.centroid_um) if self.centroid_um else None,
            "z_index": self.z_index,
            "t_index": self.t_index,
            "mean_intensity": dict(self.mean_intensity),
            "integral_intensity": dict(self.integral_intensity),
            "max_intensity": dict(self.max_intensity),
            "is_multi_component": self.is_multi_component,
            "n_components": self.n_components,
            "largest_component_fraction": self.largest_component_fraction,
            "has_holes": self.has_holes,
            "euler_number": self.euler_number,
            "touches_border": self.touches_border,
        }

    @property
    def is_physically_calibrated(self) -> bool:
        """True when a pixel size was available and physical geometry is populated."""
        return self.area_um2 is not None

    @property
    def is_single_object(self) -> bool:
        """True when the mask held exactly one connected component.

        Morphology ratios on a merged multi-object mask are not interpretable,
        so downstream feature use should gate on this.
        """
        return not self.is_multi_component


# ─────────────────────────────────────────────────────────────────────────────
# Instance
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Instance:
    """A single annotated organoid occurrence within one image (one Z/T plane).

    ``mask`` is excluded from :meth:`to_dict` — masks are large binary arrays
    and belong in the imaging store, not in annotation metadata.
    """

    id: str
    image_id: str
    class_key: str
    taxonomy_name: str
    specimen_id: Optional[str] = None

    z_index: int = 0
    t_index: int = 0

    bbox_px: Optional[tuple[float, float, float, float]] = None
    bbox_um: Optional[tuple[float, float, float, float]] = None
    mask: Any = None  # np.ndarray | None, deliberately untyped to avoid a hard numpy dependency

    measurement: Optional[Measurement] = None

    track_id: Optional[str] = None
    parent_instance_id: Optional[str] = None
    child_instance_ids: tuple[str, ...] = ()

    provenance: Provenance = Provenance.HUMAN
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    annotator: Optional[str] = None
    confidence: Optional[float] = None
    #: Model that assisted the final decision, when one did (PRD 7.4).
    #:
    #: Kept separate from ``provenance`` on purpose. A human decision reached
    #: with a model's ranking on screen is still a human decision — provenance
    #: stays HUMAN — but the assistance has to be countable, otherwise the
    #: auxiliary-path bias monitor (5.6.4) has nothing to count. Recording it as
    #: CONSENSUS would be a category error and would make the label pass the
    #: ground-truth gate as if the model had not been involved.
    assisted_by: Optional[str] = None
    #: True when the signed label matched the model's top-ranked revision.
    model_rank_agreement: Optional[bool] = None

    created_at: str = field(default_factory=_utc_now_iso)
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Instance.id must be non-empty")
        if not self.image_id:
            raise ValueError("Instance.image_id must be non-empty")
        if not self.class_key:
            raise ValueError("Instance.class_key must be non-empty")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Instance.confidence must be in [0, 1], got {self.confidence}")
        if self.z_index < 0 or self.t_index < 0:
            raise ValueError("Instance.z_index / t_index must be non-negative")
        if not isinstance(self.provenance, Provenance):
            object.__setattr__(self, "provenance", Provenance(self.provenance))
        if not isinstance(self.review_status, ReviewStatus):
            object.__setattr__(self, "review_status", ReviewStatus(self.review_status))
        if not isinstance(self.child_instance_ids, tuple):
            object.__setattr__(self, "child_instance_ids", tuple(self.child_instance_ids))
        if self.assisted_by is not None and not str(self.assisted_by).strip():
            raise ValueError("Instance.assisted_by must be a non-empty model id when set")
        if self.assisted_by is not None and self.provenance is Provenance.CONSENSUS:
            raise ValueError(
                "Instance.assisted_by is set on a CONSENSUS label (PRD 7.4). A label "
                "produced with a model's ranking on screen is HUMAN-assisted, not "
                "consensus: recording it as consensus lets a model-influenced label "
                "pass the ground-truth gate, which silently disables the quality gate."
            )

    # ── derived views ────────────────────────────────────────────────────

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    @property
    def is_tracked(self) -> bool:
        return self.track_id is not None

    @property
    def is_ground_truth(self) -> bool:
        """True only when provenance *and* review both clear the bar.

        A model pre-annotation that a human reviewed is still not independent
        ground truth for evaluating that same model, so provenance is checked
        first and the review gate applied on top.
        """
        return self.provenance.is_ground_truth and self.review_status.is_signoff

    @property
    def area_um2(self) -> Optional[float]:
        return self.measurement.area_um2 if self.measurement else None

    def with_mask(self, mask: Any) -> "Instance":
        return replace(self, mask=mask)

    def to_dict(self) -> dict:
        """Serialise metadata only — ``mask`` and its ndarray are omitted."""
        return {
            "id": self.id,
            "image_id": self.image_id,
            "specimen_id": self.specimen_id,
            "class_key": self.class_key,
            "taxonomy_name": self.taxonomy_name,
            "z_index": self.z_index,
            "t_index": self.t_index,
            "bbox_px": list(self.bbox_px) if self.bbox_px else None,
            "bbox_um": list(self.bbox_um) if self.bbox_um else None,
            "measurement": self.measurement.to_dict() if self.measurement else None,
            "track_id": self.track_id,
            "parent_instance_id": self.parent_instance_id,
            "child_instance_ids": list(self.child_instance_ids),
            "provenance": self.provenance.value,
            "review_status": self.review_status.value,
            "annotator": self.annotator,
            "confidence": self.confidence,
            "assisted_by": self.assisted_by,
            "model_rank_agreement": self.model_rank_agreement,
            "created_at": self.created_at,
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Image
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Image:
    """Acquisition metadata for one field of view.

    ``pixel_size_um`` and ``z_step_um`` should be taken from the microscope
    metadata. When the source file carries no calibration, leave them ``None``:
    downstream code will report ``None`` rather than silently treating pixels
    as micrometres.
    """

    id: str
    path: str
    specimen_id: Optional[str] = None

    width_px: int = 0
    height_px: int = 0

    channels: tuple[ChannelSpec, ...] = ()
    modality: Modality = Modality.UNKNOWN
    bit_depth: int = 8

    n_z: int = 1
    n_t: int = 1

    pixel_size_um: Optional[float] = None
    z_step_um: Optional[float] = None
    time_interval_min: Optional[float] = None

    plate: Optional[str] = None
    well: Optional[str] = None
    acquisition_time: Optional[str] = None
    objective: Optional[str] = None
    site: Optional[str] = None
    checksum: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Image.id must be non-empty")
        if not self.path:
            raise ValueError("Image.path must be non-empty")
        if self.width_px < 0 or self.height_px < 0:
            raise ValueError("Image dimensions must be non-negative")
        if self.n_z < 1 or self.n_t < 1:
            raise ValueError("Image.n_z and Image.n_t must be >= 1")
        if self.bit_depth < 1:
            raise ValueError("Image.bit_depth must be >= 1")
        if self.pixel_size_um is not None and self.pixel_size_um <= 0:
            raise ValueError("Image.pixel_size_um must be positive when set")
        if self.z_step_um is not None and self.z_step_um <= 0:
            raise ValueError("Image.z_step_um must be positive when set")
        if self.time_interval_min is not None and self.time_interval_min <= 0:
            raise ValueError("Image.time_interval_min must be positive when set")

        channels = tuple(self.channels)
        if channels:
            indices = sorted(c.index for c in channels)
            if indices != list(range(len(channels))):
                raise ValueError(
                    f"Image.channels indices must be contiguous from 0, got {indices}"
                )
        object.__setattr__(self, "channels", channels)
        if not isinstance(self.modality, Modality):
            object.__setattr__(self, "modality", Modality(self.modality))

    # ── derived views ────────────────────────────────────────────────────

    @property
    def n_channels(self) -> int:
        return len(self.channels) if self.channels else 1

    @property
    def channel_names(self) -> tuple[str, ...]:
        if self.channels:
            return tuple(c.name for c in self.channels)
        return ("C0",)

    @property
    def is_3d(self) -> bool:
        """True for a Z-stack — a single time point with multiple planes."""
        return self.n_z > 1

    @property
    def is_timelapse(self) -> bool:
        """True when more than one time point was acquired."""
        return self.n_t > 1

    @property
    def is_4d(self) -> bool:
        """True for a Z-stack acquired over time."""
        return self.is_3d and self.is_timelapse

    @property
    def is_multichannel(self) -> bool:
        return self.n_channels > 1

    @property
    def is_calibrated(self) -> bool:
        """True when physical scale is known, so ``_um`` geometry is meaningful."""
        return self.pixel_size_um is not None

    @property
    def extent_um(self) -> Optional[tuple[float, float]]:
        """Field-of-view width and height in micrometres, or ``None``."""
        if not self.is_calibrated or not (self.width_px and self.height_px):
            return None
        return (self.width_px * self.pixel_size_um, self.height_px * self.pixel_size_um)

    @property
    def z_extent_um(self) -> Optional[float]:
        if self.z_step_um is None or not self.is_3d:
            return None
        return (self.n_z - 1) * self.z_step_um

    @property
    def duration_min(self) -> Optional[float]:
        if self.time_interval_min is None or not self.is_timelapse:
            return None
        return (self.n_t - 1) * self.time_interval_min

    @property
    def plane_count(self) -> int:
        """Total Z x T x C planes in the acquisition."""
        return self.n_z * self.n_t * self.n_channels

    def channel(self, name: str) -> ChannelSpec:
        """Return a channel by name, case-insensitively.

        Raises:
            KeyError: when the channel is absent — callers must not silently
                fall back to channel 0, because that is how a nuclear stain
                ends up being measured as if it were a membrane marker.
        """
        needle = name.strip().lower()
        for c in self.channels:
            if c.name.lower() == needle:
                return c
        for c in self.channels:
            if c.stain and c.stain.lower() == needle:
                return c
        raise KeyError(f"Image {self.id} has no channel {name!r}; available: {self.channel_names}")

    def channel_of_stain(self, stain: str) -> Optional[ChannelSpec]:
        """Return the first channel matching a stain name, or ``None``."""
        needle = stain.strip().lower()
        for c in self.channels:
            if c.stain and c.stain.lower() == needle:
                return c
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "specimen_id": self.specimen_id,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "channels": [c.to_dict() for c in self.channels],
            "modality": self.modality.value,
            "bit_depth": self.bit_depth,
            "n_z": self.n_z,
            "n_t": self.n_t,
            "pixel_size_um": self.pixel_size_um,
            "z_step_um": self.z_step_um,
            "time_interval_min": self.time_interval_min,
            "plate": self.plate,
            "well": self.well,
            "acquisition_time": self.acquisition_time,
            "objective": self.objective,
            "site": self.site,
            "checksum": self.checksum,
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Specimen
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Specimen:
    """A biological sample: one culture, donor, or organoid line.

    This is the federated learning *site's* unit of provenance. Aggregation
    statistics are only comparable between nodes when the specimen context
    lines up, so batch / protocol / passage are recorded explicitly rather
    than being inferred from file naming.
    """

    id: str
    species: Optional[str] = None
    tissue: Optional[str] = None
    donor_id: Optional[str] = None
    line_name: Optional[str] = None
    batch: Optional[str] = None
    passage: Optional[int] = None
    protocol: Optional[str] = None
    treatment: Optional[str] = None
    site: Optional[str] = None
    created_at: str = field(default_factory=_utc_now_iso)
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Specimen.id must be non-empty")
        if self.passage is not None and self.passage < 0:
            raise ValueError("Specimen.passage must be non-negative")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "species": self.species,
            "tissue": self.tissue,
            "donor_id": self.donor_id,
            "line_name": self.line_name,
            "batch": self.batch,
            "passage": self.passage,
            "protocol": self.protocol,
            "treatment": self.treatment,
            "site": self.site,
            "created_at": self.created_at,
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Track
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Track:
    """An instance followed through time.

    Mutable by design: a track is accumulated as frames are linked, unlike the
    frozen records above which are written once and then reviewed.
    """

    id: str
    specimen_id: Optional[str] = None
    instance_ids: list[str] = field(default_factory=list)
    start_t: int = 0
    end_t: int = 0

    #: Id of the track this one split off from, when a division was detected.
    lineage_parent: Optional[str] = None
    #: Ids of tracks this one divided into.
    children: list[str] = field(default_factory=list)
    #: ``t_index`` values at which a division event was inferred.
    division_events: list[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Number of frames the track spans, inclusive."""
        return len(self.instance_ids)

    @property
    def duration_frames(self) -> int:
        return max(0, self.end_t - self.start_t)

    @property
    def is_division(self) -> bool:
        return bool(self.division_events)

    @property
    def is_root(self) -> bool:
        return self.lineage_parent is None

    def add(self, instance_id: str, t_index: int) -> None:
        """Append an instance and advance the tracked time span."""
        self.instance_ids.append(instance_id)
        if self.length == 1:
            self.start_t = self.end_t = t_index
        else:
            self.start_t = min(self.start_t, t_index)
            self.end_t = max(self.end_t, t_index)

    def record_division(self, t_index: int, child_track_id: str) -> None:
        """Record a division event and link the child track.

        Both are keyed rather than appended blindly: a division that produces
        two daughters at the same time point is **one** event, and recording it
        twice would double-count it in any downstream rate calculation.
        """
        if t_index not in self.division_events:
            self.division_events.append(t_index)
        if child_track_id not in self.children:
            self.children.append(child_track_id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "specimen_id": self.specimen_id,
            "instance_ids": list(self.instance_ids),
            "start_t": self.start_t,
            "end_t": self.end_t,
            "length": self.length,
            "lineage_parent": self.lineage_parent,
            "children": list(self.children),
            "division_events": list(self.division_events),
        }
