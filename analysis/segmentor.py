# ── analysis/segmentor.py ──
"""
SAM2 Organoid Segmentor
========================
Pixel-level organoid segmentation using Meta's SAM2.

Pipeline:
1. YOLO detects organoids → bounding boxes (prompts)
2. SAM2 uses boxes as prompts → pixel-level masks
3. Extract morphology metrics from masks

Morphology metrics:
- Area (pixel count)
- Perimeter (boundary length)
- Circularity (4π·area/perimeter², 1 = perfect circle)
- Solidity (area/convex_area, measures concavity)
- Aspect ratio (major/minor axis from ellipse fit)
- Eccentricity (0 = circle, 1 = line)
"""

import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class SegmentationResult:
    """Single organoid segmentation result."""
    mask: np.ndarray            # binary mask (H, W)
    area: int                   # pixel count
    perimeter: float            # boundary length
    circularity: float          # 4π·area/perimeter²
    solidity: float             # area/convex_area
    aspect_ratio: float         # major/minor axis
    eccentricity: float         # 0=circle, 1=line
    bbox: list[float]           # [x1, y1, x2, y2]
    centroid: tuple[float, float]

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

    def __init__(self, checkpoint: str = "sam2_hiera_small.pt", device: Optional[str] = None):
        """
        Args:
            checkpoint: SAM2 model checkpoint
            device: "cuda" or "cpu"
        """
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if self.cuda_available() else "cpu")
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
            except ImportError:
                raise RuntimeError(
                    "sam2 package is required for segmentation. "
                    "Install with: pip install segment-anything-2"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load SAM2 model: {e}")

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
                masks, scores, _ = self.predictor.predict(
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
                masks, scores, _ = self.predictor.predict(
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
        """Extract morphological features from binary mask.

        Args:
            mask: binary mask (H, W)
            bbox: [x1, y1, x2, y2]

        Returns:
            Dict of morphology metrics
        """
        area = int(mask.sum())

        if area == 0:
            return {
                "area": 0, "perimeter": 0, "circularity": 0,
                "solidity": 0, "aspect_ratio": 0, "eccentricity": 0,
                "bbox": bbox, "centroid": (0.0, 0.0),
            }

        # Perimeter via contour detection
        try:
            import cv2
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

                # Solidity (convexity)
                hull = cv2.convexHull(contours[0])
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                # Ellipse fit
                if len(contours[0]) >= 5:
                    ellipse = cv2.fitEllipse(contours[0])
                    major, minor = ellipse[1]
                    aspect_ratio = major / minor if minor > 0 else 0
                    eccentricity = np.sqrt(1 - (minor / major) ** 2) if major > 0 else 0
                else:
                    aspect_ratio = 1.0
                    eccentricity = 0.0
            else:
                perimeter = circularity = solidity = aspect_ratio = eccentricity = 0.0
        except ImportError:
            # Fallback without cv2
            perimeter = circularity = solidity = aspect_ratio = eccentricity = 0.0

        # Centroid
        ys, xs = np.where(mask)
        centroid = (float(np.mean(xs)), float(np.mean(ys)))

        return {
            "area": area,
            "perimeter": float(perimeter),
            "circularity": float(circularity),
            "solidity": float(solidity),
            "aspect_ratio": float(aspect_ratio),
            "eccentricity": float(eccentricity),
            "bbox": bbox,
            "centroid": centroid,
        }

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
