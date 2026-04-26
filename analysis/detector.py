# ── analysis/detector.py ──
"""
YOLOv11 Organoid Detector
==========================
Object detection for organoid images using Ultralytics YOLOv11.

Capabilities:
- Detect individual organoids in microscopy images
- Classify: healthy / early_stage / late_stage
- Count organoids per image
- Measure size (bounding box area)
- Compute circularity from detected regions

Models:
- yolo11n.pt  → 3.2M params, fastest
- yolo11s.pt  → 9.4M params, balanced
- yolo11m.pt  → 20.1M params, accurate
"""

import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class Detection:
    """Single organoid detection result."""
    bbox: list[float]          # [x1, y1, x2, y2]
    class_name: str            # "healthy", "early_stage", "late_stage"
    class_id: int
    confidence: float
    cx: float                  # center x
    cy: float                  # center y
    width: float
    height: float
    area: float                # bbox area in pixels²

    def to_dict(self) -> dict:
        return asdict(self)


class OrganoidDetector:
    """YOLOv11-based organoid detector."""

    CLASS_NAMES = ["healthy", "early_stage", "late_stage"]

    def __init__(self, model_size: str = "n", device: Optional[str] = None):
        """
        Args:
            model_size: "n" (nano), "s" (small), "m" (medium)
            device: "cuda", "cpu", or None (auto-detect)
        """
        self.model_size = model_size
        self.model_name = f"yolo11{model_size}.pt"
        self.device = device
        self.model = None

    def _ensure_model(self):
        """Lazy-load model to avoid startup cost."""
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            if self.device:
                self.model.to(self.device)

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

                detections.append(Detection(
                    bbox=xyxy,
                    class_name=self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else f"class_{cls_id}",
                    class_id=cls_id,
                    confidence=conf,
                    cx=(x1 + x2) / 2,
                    cy=(y1 + y2) / 2,
                    width=w,
                    height=h,
                    area=w * h,
                ))

        return detections

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
