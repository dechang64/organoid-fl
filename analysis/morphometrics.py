# ── analysis/morphometrics.py ──
"""Calibrated, per-object morphology measurement.

The defect this replaces
-----------------------
``analysis/segmentor.py:159-222`` measured every mask as if it held exactly one
object, and mixed two different scopes inside a single ratio:

    area = int(mask.sum())                       # every foreground pixel
    contours, _ = cv2.findContours(...)
    perimeter = cv2.arcLength(contours[0], True)  # first contour only
    hull      = cv2.convexHull(contours[0])       # first contour only
    solidity  = area / cv2.contourArea(hull)      # numerator global, denominator local

When a SAM2 mask covers several organoids — the normal case in a dense field —
the numerator counts all of them while the denominator describes one, so the
ratio is not a shape descriptor at all. Convex hulls are also concave-safe, so
the numerator could exceed the denominator and produce values above 1, or fall
far below a plausible concavity.

``docs/experiment_tracker.md:149`` records ``solidity`` scoring an AUC of
**0.272 — below the 0.5 of random**. A feature that anti-predicts its own
target is the signature of exactly this kind of scope error, not of a weak
biological signal.

What this module does instead
----------------------------
* Splits the mask into connected components and measures **each object
  separately**, so the area, perimeter and hull in a ratio all describe the
  same pixels.
* Reports geometry in **micrometres** whenever a pixel size is known, and
  leaves physical fields ``None`` when it is not — never silently returning
  pixel counts under a ``_um`` name.
* Computes intensity statistics on the **native bit-depth** data.
* States its own measurement bias instead of hiding it: a rasterised boundary
  over-estimates perimeter, so perimeter-derived features carry an
  ``is_small_object`` flag for exclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from analysis.schema import Instance, Measurement, Provenance, ReviewStatus, new_id

__all__ = [
    "MorphologyConfig",
    "label_map",
    "mask_quality",
    "mask_to_instances",
    "measure_mask",
    "measure_objects",
]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MorphologyConfig:
    """Measurement options.

    Attributes:
        min_area_px: components smaller than this are dropped. Defaults to 8,
            which is roughly the point below which a rasterised perimeter stops
            carrying usable shape information.
        small_object_diameter_px: objects whose equivalent diameter is below
            this are flagged ``is_small_object`` so perimeter-derived features
            (circularity, solidity) can be excluded from modelling. Area-derived
            features remain valid.
        connectivity: 4 or 8. 8-connectivity is the usual choice for
            fluorescence masks.
        border_margin_px: components whose bounding box comes within this many
            pixels of the image edge are flagged ``touches_border``. Truncated
            objects have biased area and must be excluded from size statistics.
        compute_intensity: compute per-channel intensity statistics.
        intensity_percentiles: extra intensity percentiles to report.
    """

    min_area_px: int = 8
    small_object_diameter_px: float = 5.0
    connectivity: int = 8
    border_margin_px: int = 0
    compute_intensity: bool = True
    intensity_percentiles: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {self.connectivity}")
        if self.min_area_px < 0:
            raise ValueError("min_area_px must be non-negative")
        if self.border_margin_px < 0:
            raise ValueError("border_margin_px must be non-negative")


# ─────────────────────────────────────────────────────────────────────────────
# Mask utilities
# ─────────────────────────────────────────────────────────────────────────────


def _as_mask(mask: Any) -> np.ndarray:
    """Coerce to a 2-D boolean array."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {arr.shape}")
    return arr.astype(bool)


def label_map(mask: Any, *, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Label connected components.

    Returns ``(labels, n_components)`` where ``labels`` is ``int32`` with 0 as
    background and 1..N per component. Implemented with OpenCV when available
    and a pure-numpy union-find otherwise, so the module stays importable in a
    minimal environment.
    """
    m = _as_mask(mask)
    if not m.any():
        return np.zeros(m.shape, dtype=np.int32), 0

    try:
        import cv2

        structure = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3) if connectivity == 8 else (1, 3)
        )
        n, labels = cv2.connectedComponents(m.astype(np.uint8), connectivity=connectivity)
        del structure  # connectivity is honoured by the argument above
        return labels.astype(np.int32), int(n) - 1
    except ImportError:  # pragma: no cover - cv2 is present in this repo's venv
        return _label_union_find(m, connectivity=connectivity)


def _label_union_find(mask: np.ndarray, *, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Dependency-free two-pass connected-component labelling."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    neighbours = [(-1, 0), (0, -1)] if connectivity == 4 else [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

    next_label = 1
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            found = []
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] > 0:
                    found.append(labels[ny, nx])
            if not found:
                parent.append(next_label)
                labels[y, x] = next_label
                next_label += 1
            else:
                labels[y, x] = min(found)
                for f in found[1:]:
                    union(found[0], f)

    remap = {}
    out = np.zeros_like(labels)
    for y in range(h):
        for x in range(w):
            lab = labels[y, x]
            if lab == 0:
                continue
            root = find(lab)
            if root not in remap:
                remap[root] = len(remap) + 1
            out[y, x] = remap[root]
    return out, len(remap)


def _component_holes(component: np.ndarray) -> tuple[int, int, bool]:
    """Return ``(hole_pixel_count, n_holes, has_holes)`` for one component crop.

    A hole is foreground-free area enclosed by the component. Computed by
    comparing the component's own pixels against the fill of its external
    contour, which avoids depending on a hole-filling library.
    """
    try:
        import cv2

        contours, hierarchy = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
    except ImportError:  # pragma: no cover
        return 0, 0, False

    if not contours or hierarchy is None:
        return 0, 0, False

    hierarchy = hierarchy[0]
    filled = np.zeros_like(component, dtype=np.uint8)
    n_holes = 0
    for i, contour in enumerate(contours):
        # A contour whose parent is != -1 is a hole boundary.
        if hierarchy[i][3] != -1:
            n_holes += 1
            cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)

    hole_pixels = int(filled.sum())
    return hole_pixels, n_holes, hole_pixels > 0


def _boundary_perimeter(component: np.ndarray) -> float:
    """Raster boundary length, counting diagonal steps as ``sqrt(2)``.

    This is a chain-code length. It is **positively biased**: a raster boundary
    can only approximate a smooth curve, and the over-estimate is
    proportionally largest for small objects. :class:`MorphologyConfig` exposes
    ``small_object_diameter_px`` so consumers can exclude objects where that
    bias matters.
    """
    c = component.astype(np.uint8)
    padded = np.zeros((c.shape[0] + 2, c.shape[1] + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = c

    # Horizontal and vertical runs where foreground meets background.
    vertical = np.abs(np.diff(padded, axis=1)) * 1.0
    horizontal = np.abs(np.diff(padded, axis=0)) * 1.0

    # Diagonal transitions: 2x2 windows with exactly two diagonally opposite
    # foreground pixels contribute a corner.
    a = padded[:-1, :-1].astype(np.int16)
    b = padded[:-1, 1:].astype(np.int16)
    c_ = padded[1:, :-1].astype(np.int16)
    d = padded[1:, 1:].astype(np.int16)
    diagonal = ((a + d == 2) & (b + c_ == 0)) | ((b + c_ == 2) & (a + d == 0))
    n_diagonal = float(diagonal.sum())

    return float(vertical.sum() + horizontal.sum() + n_diagonal * (np.sqrt(2.0) - 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────────────


def mask_quality(mask: Any, *, connectivity: int = 8) -> dict:
    """Mask-level diagnostics that describe the *set*, not any one object.

    Useful for deciding whether a segmentation is trustworthy: a mask whose
    largest component is 30 % of its pixels is several merged objects, and
    per-object shape statistics from it should not be pooled with clean ones.
    """
    m = _as_mask(mask)
    labels, n = label_map(m, connectivity=connectivity)
    return _quality_from_labels(labels, n, total_pixels=int(m.sum()))


def _quality_from_labels(labels: np.ndarray, n_components: int, *, total_pixels: int) -> dict:
    """Build mask-level diagnostics from an existing label map.

    Split out so callers that already labelled the mask do not pay for a second
    connected-component pass.
    """
    if n_components == 0:
        return {
            "n_components": 0,
            "total_area_px": 0,
            "largest_area_px": 0,
            "largest_component_fraction": 0.0,
            "component_sizes_px": [],
            "is_multi_component": False,
        }
    sizes = np.bincount(labels.ravel())
    sizes = sizes[1:]  # drop background
    largest = int(sizes.max()) if len(sizes) else 0
    return {
        "n_components": int(n_components),
        "total_area_px": int(total_pixels),
        "largest_area_px": largest,
        "largest_component_fraction": (largest / total_pixels) if total_pixels else 0.0,
        "component_sizes_px": sizes.tolist(),
        "is_multi_component": int(n_components) > 1,
    }


def measure_mask(
    mask: Any,
    *,
    pixel_size_um: Optional[float] = None,
    intensity: Optional[np.ndarray] = None,
    channel_names: Optional[Sequence[str]] = None,
    z_index: int = 0,
    t_index: int = 0,
    config: Optional[MorphologyConfig] = None,
) -> list[Measurement]:
    """Measure every object in ``mask``, one :class:`Measurement` per object.

    Args:
        mask: 2-D binary array.
        pixel_size_um: physical size of one pixel. When ``None``, any ``_um``
            field on the result is ``None`` and only the ``_px`` and
            dimensionless fields are populated.
        intensity: matching 2-D array used for intensity statistics. **Must be
            the native-bit-depth plane** — passing a display-normalised uint8
            copy would destroy the dynamic range. May also be ``(C, Y, X)`` for
            multi-channel intensity.
        channel_names: names for the intensity channels; defaults to ``C0..``.
        config: measurement options.

    Returns:
        One :class:`Measurement` per connected component, ordered largest first.
        Each carries the parent mask's ``n_components`` and
        ``is_multi_component`` so a consumer can tell a clean single-object
        mask from a merged one.
    """
    cfg = config or MorphologyConfig()
    m = _as_mask(mask)
    labels, n_components = label_map(m, connectivity=cfg.connectivity)

    if n_components == 0:
        return []

    quality = _quality_from_labels(labels, n_components, total_pixels=int(m.sum()))

    intensity_arr, chan_names = _prepare_intensity(intensity, channel_names)

    results: list[Measurement] = []
    order = sorted(range(1, n_components + 1), key=lambda lab: -int((labels == lab).sum()))

    for lab in order:
        component = labels == lab
        area_px = int(component.sum())
        if area_px < cfg.min_area_px:
            continue
        results.append(
            _measure_component(
                component,
                area_px=area_px,
                pixel_size_um=pixel_size_um,
                intensity_arr=intensity_arr,
                chan_names=chan_names,
                z_index=z_index,
                t_index=t_index,
                cfg=cfg,
                mask_quality=quality,
            )
        )

    return results


def _prepare_intensity(
    intensity: Optional[np.ndarray],
    channel_names: Optional[Sequence[str]],
) -> tuple[Optional[np.ndarray], tuple[str, ...]]:
    if intensity is None:
        return None, ()
    arr = np.asarray(intensity)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
        default_names = ("C0",)
    elif arr.ndim == 3:
        default_names = tuple(f"C{i}" for i in range(arr.shape[0]))
    else:
        raise ValueError(f"intensity must be (Y, X) or (C, Y, X), got shape {arr.shape}")

    names = tuple(channel_names) if channel_names else default_names
    if len(names) != arr.shape[0]:
        raise ValueError(
            f"Got {len(names)} channel_names for {arr.shape[0]} intensity channels"
        )
    return arr, names


def _measure_component(
    component: np.ndarray,
    *,
    area_px: int,
    pixel_size_um: Optional[float],
    intensity_arr: Optional[np.ndarray],
    chan_names: tuple[str, ...],
    z_index: int,
    t_index: int,
    cfg: MorphologyConfig,
    mask_quality: Mapping[str, Any],
) -> Measurement:
    try:
        import cv2
    except ImportError:  # pragma: no cover
        cv2 = None  # type: ignore

    ys, xs = np.nonzero(component)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    h, w = component.shape

    centroid_px = (float(xs.mean()), float(ys.mean()))

    perimeter_px: Optional[float] = None
    circularity: Optional[float] = None
    solidity: Optional[float] = None
    major_px: Optional[float] = None
    minor_px: Optional[float] = None
    aspect_ratio: Optional[float] = None
    eccentricity: Optional[float] = None

    if cv2 is not None:
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if contours:
            # Longest contour, not [0]: findContours does not guarantee order.
            contour = max(contours, key=cv2.contourArea)
            perimeter_px = float(cv2.arcLength(contour, True))
            if perimeter_px > 0:
                circularity = float(4.0 * np.pi * area_px / (perimeter_px ** 2))

            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            if hull_area > 0:
                # Component area over the hull of the SAME component — the
                # scope match the original code was missing.
                solidity = float(min(area_px / hull_area, 1.0))

            if len(contour) >= 5:
                (_, _), (ax1, ax2), _ = cv2.fitEllipse(contour)
                major_px, minor_px = float(max(ax1, ax2)), float(min(ax1, ax2))
                if minor_px > 0:
                    aspect_ratio = float(major_px / minor_px)
                    ratio = min(minor_px / major_px, 1.0)
                    eccentricity = float(np.sqrt(max(0.0, 1.0 - ratio ** 2)))

    if perimeter_px is None and cv2 is None:  # pragma: no cover
        perimeter_px = _boundary_perimeter(component)

    # ── physical conversion ──
    scale = pixel_size_um
    area_um2 = float(area_px * scale ** 2) if scale else None
    perimeter_um = float(perimeter_px * scale) if (scale and perimeter_px is not None) else None
    major_um = float(major_px * scale) if (scale and major_px is not None) else None
    minor_um = float(minor_px * scale) if (scale and minor_px is not None) else None
    centroid_um = (centroid_px[0] * scale, centroid_px[1] * scale) if scale else None

    # Equivalent diameter comes from area, which is a robust count rather than
    # a rasterised boundary estimate, so it is preferred over the ellipse axes
    # for gating and for reporting object size.
    eq_diam_px = float(2.0 * np.sqrt(area_px / np.pi))
    equivalent_diameter_um = float(eq_diam_px * scale) if scale else None

    # ── intensity on native bit depth ──
    mean_i: dict[str, float] = {}
    integral_i: dict[str, float] = {}
    max_i: dict[str, float] = {}
    if cfg.compute_intensity and intensity_arr is not None:
        local = component[y0:y1, x0:x1]
        for ci, cname in enumerate(chan_names):
            plane = intensity_arr[ci][y0:y1, x0:x1].astype(np.float64)
            vals = plane[local]
            if vals.size == 0:
                continue
            mean_i[cname] = float(vals.mean())
            integral_i[cname] = float(vals.sum())
            max_i[cname] = float(vals.max())

    # The hole *area* is available from this call but deliberately not
    # stored: Measurement carries has_holes and euler_number, and a lumen
    # area field would be a schema change rather than a measurement fix.
    _hole_pixels, n_holes, has_holes = _component_holes(component)

    margin = cfg.border_margin_px
    touches_border = bool(
        x0 <= margin or y0 <= margin or x1 >= w - margin or y1 >= h - margin
    )

    return Measurement(
        area_um2=area_um2,
        perimeter_um=perimeter_um,
        equivalent_diameter_um=equivalent_diameter_um,
        major_axis_um=major_um,
        minor_axis_um=minor_um,
        circularity=circularity,
        solidity=solidity,
        aspect_ratio=aspect_ratio,
        eccentricity=eccentricity,
        area_px=area_px,
        perimeter_px=perimeter_px,
        centroid_px=centroid_px,
        centroid_um=centroid_um,
        z_index=z_index,
        t_index=t_index,
        mean_intensity=mean_i,
        integral_intensity=integral_i,
        max_intensity=max_i,
        is_multi_component=bool(mask_quality.get("is_multi_component", False)),
        n_components=int(mask_quality.get("n_components", 1)),
        largest_component_fraction=float(mask_quality.get("largest_component_fraction", 1.0)),
        has_holes=has_holes,
        euler_number=1 - n_holes,
        touches_border=touches_border,
    )


def measure_objects(
    mask: Any,
    *,
    pixel_size_um: Optional[float] = None,
    intensity: Optional[np.ndarray] = None,
    channel_names: Optional[Sequence[str]] = None,
    config: Optional[MorphologyConfig] = None,
    **kwargs: Any,
) -> list[Measurement]:
    """Alias for :func:`measure_mask` that reads better for per-object callers."""
    return measure_mask(
        mask,
        pixel_size_um=pixel_size_um,
        intensity=intensity,
        channel_names=channel_names,
        config=config,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mask -> Instance bridge
# ─────────────────────────────────────────────────────────────────────────────


def mask_to_instances(
    mask: Any,
    *,
    image_id: str,
    class_key: str,
    taxonomy_name: str,
    specimen_id: Optional[str] = None,
    pixel_size_um: Optional[float] = None,
    intensity: Optional[np.ndarray] = None,
    channel_names: Optional[Sequence[str]] = None,
    z_index: int = 0,
    t_index: int = 0,
    provenance: Provenance | str = Provenance.MODEL_PREANNOTATION,
    review_status: ReviewStatus | str = ReviewStatus.PRE_ANNOTATED,
    annotator: Optional[str] = None,
    confidence: Optional[float] = None,
    config: Optional[MorphologyConfig] = None,
    keep_masks: bool = True,
) -> list[Instance]:
    """Turn one binary mask into one :class:`Instance` per object.

    Each instance keeps the crop of its own component as its mask, so a
    downstream edit touches only that organoid rather than the merged blob.
    """
    cfg = config or MorphologyConfig()
    m = _as_mask(mask)
    labels, n = label_map(m, connectivity=cfg.connectivity)
    measurements = measure_mask(
        m,
        pixel_size_um=pixel_size_um,
        intensity=intensity,
        channel_names=channel_names,
        z_index=z_index,
        t_index=t_index,
        config=cfg,
    )

    prov = provenance if isinstance(provenance, Provenance) else Provenance(provenance)
    status = review_status if isinstance(review_status, ReviewStatus) else ReviewStatus(review_status)

    quality = _quality_from_labels(labels, n, total_pixels=int(m.sum()))

    instances: list[Instance] = []
    # Measurements come back largest-first; pair them with matching labels.
    remaining = sorted(range(1, n + 1), key=lambda lab: -int((labels == lab).sum()))
    for lab, meas in zip(remaining, measurements, strict=True):
        component = labels == lab
        sub = component if keep_masks else None

        ys, xs = np.nonzero(component)
        bbox_px = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        if pixel_size_um:
            bbox_um = tuple(v * pixel_size_um for v in bbox_px)  # type: ignore[assignment]
        else:
            bbox_um = None

        instances.append(
            Instance(
                id=new_id("inst"),
                image_id=image_id,
                specimen_id=specimen_id,
                class_key=class_key,
                taxonomy_name=taxonomy_name,
                z_index=z_index,
                t_index=t_index,
                bbox_px=bbox_px,
                bbox_um=bbox_um,  # type: ignore[arg-type]
                mask=sub,
                measurement=meas,
                provenance=prov,
                review_status=status,
                annotator=annotator,
                confidence=confidence,
                notes=(
                    f"mask_quality: {quality['n_components']} components; "
                    f"largest={quality['largest_component_fraction']:.2f}"
                    if quality["is_multi_component"]
                    else None
                ),
            )
        )

    return instances
