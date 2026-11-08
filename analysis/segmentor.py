from __future__ import annotations
# ── analysis/segmentor.py ──
"""
SAM2 Organoid Segmentor
========================
Pixel-level organoid segmentation using Meta's SAM2.

Pipeline:
1. YOLO detects organoids → bounding boxes (prompts)
2. SAM2 uses boxes as prompts → pixel-level masks
3. Per-object morphology is measured by :mod:`analysis.morphometrics`

Morphology metrics:
- Area (pixels) and area in µm² when a pixel size is supplied
- Perimeter, circularity (4π·area/perimeter², 1 = perfect circle)
- Solidity (area/convex_area, measures concavity)
- Aspect ratio and eccentricity from the fitted ellipse

Measurement change worth knowing about
--------------------------------------
``_compute_morphology`` used to mix two scopes inside one ratio: the numerator
was the whole mask's pixel count while the perimeter, convex hull and ellipse
were taken from ``contours[0]`` alone. On a mask covering several organoids —
routine in a dense field — the resulting ``solidity`` could exceed 1, which no
shape descriptor can legitimately do. That is the mechanism behind the
``solidity`` AUC of 0.272 (below random) recorded in
``docs/experiment_tracker.md:149``.

Metrics are now produced by :func:`analysis.morphometrics.measure_mask`, which
measures each connected component in its own scope. ``_compute_morphology`` is
retained as a compatibility shim and a deprecation notice is emitted.
"""

from typing import Optional

import numpy as np
from dataclasses import dataclass, asdict

from analysis.morphometrics import MorphologyConfig, measure_mask
from analysis.schema import Instance, Provenance, ReviewStatus
from utils.taxonomy import Taxonomy, resolve_taxonomy


@dataclass
class SegmentationResult:
    """Single organoid segmentation result.

    The trailing fields were added when measurement moved to
    :mod:`analysis.morphometrics`. They default to ``None`` so existing
    positional construction keeps working.
    """
    mask: np.ndarray            # binary mask (H, W)
    area: int                   # pixel count of the reported object
    perimeter: float            # boundary length in pixels
    circularity: float          # 4π·area/perimeter²
    solidity: float             # area/convex_area, always in [0, 1]
    aspect_ratio: float         # major/minor axis
    eccentricity: float         # 0=circle, 1=line
    bbox: list[float]           # [x1, y1, x2, y2]
    centroid: tuple[float, float]

    # ── added: physical units, populated only when a pixel size is known ──
    area_um2: Optional[float] = None
    perimeter_um: Optional[float] = None
    equivalent_diameter_um: Optional[float] = None
    centroid_um: Optional[tuple[float, float]] = None

    # ── added: mask composition, so merged masks are identifiable ──
    n_components: int = 1
    total_area_px: int = 0
    is_multi_component: bool = False
    touches_border: bool = False

    def to_dict(self) -> dict:
        """Serialize segmentation result (excludes mask array)."""
        d = asdict(self)
        d.pop("mask")  # Don't serialize mask
        return d


class OrganoidSegmentor:
    """SAM2-based organoid segmentor.

    Uses YOLO bounding boxes as prompts for SAM2.
    Falls back to point-based prompts if no boxes available.
    """

    def __init__(
        self,
        checkpoint: str = "sam2_hiera_small.pt",
        device: Optional[str] = None,
        pixel_size_um: Optional[float] = None,
        taxonomy: object = None,
    ):
        """
        Args:
            checkpoint: SAM2 model checkpoint
            device: "cuda" or "cpu"
            pixel_size_um: physical pixel size. When supplied, every measurement
                carries µm / µm² geometry; when omitted, only pixel and
                dimensionless fields are populated.
            taxonomy: class schema used by :meth:`segment_instances`.
        """
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if self.cuda_available() else "cpu")
        self.pixel_size_um = pixel_size_um
        self.taxonomy: Taxonomy = resolve_taxonomy(taxonomy)
        self.predictor = None

    @staticmethod
    def cuda_available() -> bool:
        """Check if CUDA is available for SAM2 acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _ensure_predictor(self):
        """Lazy-load SAM2 predictor."""
        if self.predictor is None:
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                model = build_sam2(self.checkpoint, device=self.device)
                self.predictor = SAM2ImagePredictor(model)
            except ImportError as exc:
                raise RuntimeError(
                    "sam2 package is required for segmentation. "
                    "Install with: pip install segment-anything-2"
                ) from exc
            except Exception as e:
                raise RuntimeError(f"Failed to load SAM2 model: {e}") from e

    def segment(
        self,
        image: np.ndarray,
        boxes: Optional[list[list[float]]] = None,
        points: Optional[list[list[float]]] = None,
    ) -> list[SegmentationResult]:
        """Segment organoids given prompts.

        Args:
            image: HWC numpy array (RGB)
            boxes: list of [x1, y1, x2, y2] bounding boxes
            points: list of [x, y] center points (fallback)

        Returns:
            List of SegmentationResult
        """
        self._ensure_predictor()
        self.predictor.set_image(image)

        results = []

        if boxes:
            for box in boxes:
                box_np = np.array(box, dtype=np.float32)
                masks, _scores, _ = self.predictor.predict(
                    box=box_np,
                    multimask_output=False,
                )
                mask = masks[0]
                metrics = self._compute_morphology(mask, box)
                results.append(SegmentationResult(mask=mask, **metrics))

        elif points:
            for pt in points:
                point_np = np.array([pt], dtype=np.float32)
                point_label = np.array([1], dtype=np.int32)  # foreground
                masks, _scores, _ = self.predictor.predict(
                    point_coords=point_np,
                    point_labels=point_label,
                    multimask_output=False,
                )
                mask = masks[0]
                # Compute bbox from mask
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                else:
                    box = [0, 0, 0, 0]
                metrics = self._compute_morphology(mask, box)
                results.append(SegmentationResult(mask=mask, **metrics))

        return results

    def segment_from_detections(
        self,
        image: np.ndarray,
        detections: list,
    ) -> list[SegmentationResult]:
        """Segment using YOLO detection results as prompts.

        Args:
            image: HWC numpy array
            detections: list of Detection objects (from detector.py)

        Returns:
            List of SegmentationResult
        """
        boxes = [d.bbox for d in detections]
        return self.segment(image, boxes=boxes)

    def _compute_morphology(self, mask: np.ndarray, bbox: list[float]) -> dict:
        """Extract morphology for the largest object in ``mask``.

        Returns a dict whose keys match :class:`SegmentationResult`.

        .. deprecated::
            Use :func:`analysis.morphometrics.measure_mask` directly, or
            :meth:`segment_instances` which returns fully populated
            :class:`~analysis.schema.Instance` records. This shim exists so
            older call sites keep working.

        Args:
            mask: binary mask (H, W)
            bbox: ``[x1, y1, x2, y2]`` for the region that prompted the mask.

        Returns:
            Dict of morphology metrics. ``area`` is the largest object's pixel
            count — the same object every other metric describes. The whole
            mask's pixel count is reported separately as ``total_area_px``.
        """
        measurements = measure_mask(
            mask,
            pixel_size_um=self.pixel_size_um,
            config=MorphologyConfig(min_area_px=1),
        )

        if not measurements:
            return {
                "area": 0, "perimeter": 0, "circularity": 0.0,
                "solidity": 0.0, "aspect_ratio": 0.0, "eccentricity": 0.0,
                "bbox": list(bbox), "centroid": (0.0, 0.0),
                "area_um2": None, "perimeter_um": None,
                "equivalent_diameter_um": None, "centroid_um": None,
                "n_components": 0, "total_area_px": 0,
                "is_multi_component": False, "touches_border": False,
            }

        primary = measurements[0]  # ordered largest-first
        return {
            "area": primary.area_px,
            "perimeter": float(primary.perimeter_px or 0.0),
            "circularity": float(primary.circularity or 0.0),
            "solidity": float(primary.solidity or 0.0),
            "aspect_ratio": float(primary.aspect_ratio or 0.0),
            "eccentricity": float(primary.eccentricity or 0.0),
            "bbox": list(bbox),
            "centroid": primary.centroid_px or (0.0, 0.0),
            "area_um2": primary.area_um2,
            "perimeter_um": primary.perimeter_um,
            "equivalent_diameter_um": primary.equivalent_diameter_um,
            "centroid_um": primary.centroid_um,
            "n_components": primary.n_components,
            "total_area_px": int(np.asarray(mask).astype(bool).sum()),
            "is_multi_component": primary.is_multi_component,
            "touches_border": primary.touches_border,
        }

    def segment_instances(
        self,
        image: np.ndarray,
        *,
        image_id: str,
        boxes: Optional[list[list[float]]] = None,
        class_keys: Optional[list[str]] = None,
        intensity: Optional[np.ndarray] = None,
        channel_names: Optional[list[str]] = None,
        z_index: int = 0,
        t_index: int = 0,
        specimen_id: Optional[str] = None,
        confidence: Optional[float] = None,
        provenance: Provenance | str = Provenance.MODEL_PREANNOTATION,
        review_status: ReviewStatus | str = ReviewStatus.PRE_ANNOTATED,
    ) -> list[Instance]:
        """Segment and return organoid-native :class:`Instance` records.

        Unlike :meth:`segment`, which collapses each prompted mask to one
        :class:`SegmentationResult`, this splits merged masks into individual
        objects and attaches calibrated measurements, provenance and review
        state — the shape the annotation product actually consumes.

        Args:
            image: HWC numpy array (RGB) or a single quantitative plane.
            image_id: identifier of the parent image.
            boxes: ``[x1, y1, x2, y2]`` prompts.
            class_keys: one taxonomy class key per box. Defaults to the first
                class of the configured taxonomy.
            intensity: native-bit-depth plane(s) for intensity measurement.
            channel_names: names for the intensity channels.
        """
        from analysis.morphometrics import mask_to_instances

        instances: list[Instance] = []
        masks = self.segment(image, boxes=boxes)

        for i, result in enumerate(masks):
            class_key = (
                class_keys[i] if class_keys and i < len(class_keys)
                else self.taxonomy.resolve_key(0)
            )
            instances.extend(
                mask_to_instances(
                    result.mask,
                    image_id=image_id,
                    class_key=class_key,
                    taxonomy_name=self.taxonomy.name,
                    specimen_id=specimen_id,
                    pixel_size_um=self.pixel_size_um,
                    intensity=intensity,
                    channel_names=channel_names,
                    z_index=z_index,
                    t_index=t_index,
                    provenance=provenance,
                    review_status=review_status,
                    confidence=confidence,
                )
            )

        return instances

    def get_trainable_params(self) -> dict:
        """Get prompt encoder parameters for FL aggregation."""
        self._ensure_predictor()
        return {
            k: v.clone()
            for k, v in self.predictor.model.state_dict().items()
            if "prompt_encoder" in k
        }

    def load_params(self, state_dict: dict) -> None:
        """Load aggregated prompt encoder parameters."""
        self._ensure_predictor()
        current = self.predictor.model.state_dict()
        current.update(state_dict)
        self.predictor.model.load_state_dict(current)
