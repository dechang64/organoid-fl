from __future__ import annotations
# ── analysis/detector.py ──
"""
YOLOv11 Organoid Detector
==========================
Object detection for organoid images using Ultralytics YOLOv11.

Capabilities:
- Detect individual organoids in microscopy images
- Classify using the platform taxonomy (see ``utils.taxonomy``)
- Count organoids per image
- Measure size (bounding box area)

Class schema
------------
This module previously hard-coded ``["healthy", "early_stage", "late_stage"]``
— a three-class placeholder that matched **no dataset in the repository**. Every
real ``data.yaml`` (``data/organoid_patches``, ``data/intestinal_organoid``, all
twelve ``fl_detect_results/*/*``) declares four classes:
``organoid0 / organoid1 / organoid3 / spheroid``.

Because ``detect()`` indexed that three-element list with the model's ``cls_id``,
a genuine fourth class fell through to the string ``"class_3"`` and every
detection from a 4-class model was mislabelled. Class names now come from the
taxonomy registry, and an out-of-range index is reported rather than invented.

Models:
- yolo11n.pt  → 3.2M params, fastest
- yolo11s.pt  → 9.4M params, balanced
- yolo11m.pt  → 20.1M params, accurate
"""

import warnings
from typing import Any, ClassVar, Optional

import numpy as np
from dataclasses import dataclass, asdict

from utils.taxonomy import DEFAULT_TAXONOMY, Taxonomy, TaxonomyError, resolve_taxonomy


@dataclass
class Detection:
    """Single organoid detection result."""
    bbox: list[float]          # [x1, y1, x2, y2]
    class_name: str            # taxonomy class key, e.g. "organoid0", "spheroid"
    class_id: int
    confidence: float
    cx: float                  # center x
    cy: float                  # center y
    width: float
    height: float
    area: float                # bbox area in pixels²
    label: Optional[str] = None        # human-facing label from the taxonomy
    taxonomy_name: Optional[str] = None  # which taxonomy produced class_name

    def to_dict(self) -> dict:
        """Serialize detection result to dictionary."""
        return asdict(self)


class OrganoidDetector:
    """YOLOv11-based organoid detector.

    Args:
        model_size: ``"n"``, ``"s"`` or ``"m"`` — the Ultralytics YOLOv11 variant.
        device: ``"cuda"``, ``"cpu"`` or ``None`` to auto-detect.
        taxonomy: class schema. Accepts a registered name, a
            :class:`~utils.taxonomy.Taxonomy`, a parsed ``data.yaml`` mapping, or
            a path to one. Defaults to the registry default
            (``organoid_4class``).
        on_unknown_class: what to do when a model emits a class index outside
            the taxonomy. ``"warn"`` (default) keeps the pipeline running while
            making the mismatch visible; ``"raise"`` fails fast, which is the
            right choice for a validated run; ``"passthrough"`` records the raw
            index with no complaint, for legacy behaviour.
    """

    #: Class keys of the registry's default taxonomy.
    #:
    #: **Deprecated as a source of truth** — kept because callers reference it.
    #: It is now generated from the registry, so it can no longer drift from the
    #: datasets. Prefer :attr:`OrganoidDetector.class_names` on an instance,
    #: which honours the ``taxonomy`` you actually configured.
    CLASS_NAMES: ClassVar[list[str]] = list(resolve_taxonomy(DEFAULT_TAXONOMY).keys)

    def __init__(
        self,
        model_size: str = "n",
        device: Optional[str] = None,
        taxonomy: Any = None,
        on_unknown_class: str = "warn",
    ):
        if on_unknown_class not in ("warn", "raise", "passthrough"):
            raise ValueError(
                "on_unknown_class must be 'warn', 'raise' or 'passthrough', "
                f"got {on_unknown_class!r}"
            )
        self.model_size = model_size
        self.model_name = f"yolo11{model_size}.pt"
        self.device = device
        self.model = None
        self.taxonomy: Taxonomy = resolve_taxonomy(taxonomy)
        self.on_unknown_class = on_unknown_class

    @property
    def class_names(self) -> list[str]:
        """Class keys of this detector's configured taxonomy."""
        return list(self.taxonomy.keys)

    @property
    def n_classes(self) -> int:
        return self.taxonomy.n_classes

    def _ensure_model(self):
        """Lazy-load model to avoid startup cost."""
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_name)
                if self.device:
                    self.model.to(self.device)
            except ImportError as exc:
                raise RuntimeError(
                    "ultralytics package is required for YOLO detection. "
                    "Install with: pip install ultralytics"
                ) from exc
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load YOLO model '{self.model_name}': {e}"
                ) from e

    def detect(self, image, conf_threshold: float = 0.25) -> list[Detection]:
        """Detect organoids in an image.

        Args:
            image: file path, PIL Image, or numpy array (HWC)
            conf_threshold: minimum confidence score

        Returns:
            List of Detection objects
        """
        self._ensure_model()
        results = self.model(image, conf=conf_threshold, verbose=False)

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().tolist()
                cls_id = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1

                class_key, label = self._resolve_class(cls_id)

                detections.append(Detection(
                    bbox=xyxy,
                    class_name=class_key,
                    class_id=cls_id,
                    confidence=conf,
                    cx=(x1 + x2) / 2,
                    cy=(y1 + y2) / 2,
                    width=w,
                    height=h,
                    area=w * h,
                    label=label,
                    taxonomy_name=self.taxonomy.name,
                ))

        return detections

    def _resolve_class(self, cls_id: int) -> tuple[str, Optional[str]]:
        """Map a model class index onto the taxonomy.

        Returns ``(class_key, label)``. Unknown indices are surfaced according to
        :attr:`on_unknown_class` instead of being silently renamed.
        """
        try:
            key = self.taxonomy.resolve_key(self.taxonomy.validate_index(cls_id))
            return key, self.taxonomy.key_to_class[key.lower()].label
        except TaxonomyError as exc:
            if self.on_unknown_class == "raise":
                raise
            if self.on_unknown_class == "warn":
                warnings.warn(
                    f"Detector taxonomy {self.taxonomy.name!r} has "
                    f"{self.taxonomy.n_classes} classes ({self.taxonomy.keys}) but the "
                    f"model emitted class index {cls_id}. Check that the model was "
                    f"trained against this schema. Original error: {exc}",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return f"class_{cls_id}", None

    def train_local(
        self,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        lr: float = 0.01,
        **kwargs,
    ) -> dict:
        """Train detection model on local data.

        Args:
            data_yaml: path to YOLO data.yaml config
            epochs: training epochs
            imgsz: input image size
            batch: batch size
            lr: learning rate

        Returns:
            Training results dict
        """
        self._ensure_model()
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr,
            verbose=False,
            **kwargs,
        )
        return {
            "epochs": epochs,
            "final_map": results.results_dict.get("metrics/mAP50(B)", 0),
            "final_map50_95": results.results_dict.get("metrics/mAP50-95(B)", 0),
        }

    def export_weights(self) -> dict:
        """Export model weights for FedAvg aggregation.

        Returns:
            OrderedDict of parameter name → tensor
        """
        self._ensure_model()
        from collections import OrderedDict
        return OrderedDict(self.model.model.state_dict())

    def load_weights(self, state_dict: dict) -> None:
        """Load aggregated weights from server.

        Args:
            state_dict: aggregated parameter dict
        """
        self._ensure_model()
        self.model.model.load_state_dict(state_dict)

    def get_trainable_params(self) -> dict:
        """Alias for export_weights (FL interface compatibility)."""
        return self.export_weights()

    def count_by_class(self, detections: list[Detection]) -> dict[str, int]:
        """Count detections per class."""
        counts = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    def summary(self, detections: list[Detection]) -> dict:
        """Generate detection summary statistics."""
        if not detections:
            return {"total": 0, "classes": {}, "avg_confidence": 0, "avg_area": 0}

        class_counts = self.count_by_class(detections)
        return {
            "total": len(detections),
            "classes": class_counts,
            "avg_confidence": np.mean([d.confidence for d in detections]),
            "avg_area": np.mean([d.area for d in detections]),
            "min_area": min(d.area for d in detections),
            "max_area": max(d.area for d in detections),
        }
