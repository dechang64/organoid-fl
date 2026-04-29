# ── analysis/feature_extractor_v2.py ──
"""
DINOv2 Feature Extractor
=========================
Self-supervised ViT feature extraction — replaces ResNet-18.

Advantages:
- 768-dim features (vs 512-dim ResNet)
- No labels needed for pretraining (self-supervised)
- Better global context via ViT attention
- State-of-the-art on medical imaging benchmarks

Models:
- facebook/dinov2-vits14   → 384-dim, 22M params (fastest)
- facebook/dinov2-base     → 768-dim, 86M params (recommended)
- facebook/dinov2-large    → 1024-dim, 300M params (best quality)
- facebook/dinov2-giant    → 1536-dim, 1.1B params (heavy)
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from typing import Optional
from torchvision import transforms


class DINOv2Extractor(nn.Module):
    """DINOv2-based feature extractor using CLS token."""

    MODEL_DIMS = {
        "vits14": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }

    def __init__(self, model_name: str = "facebook/dinov2-base", device: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.variant = model_name.split("/")[-1] if "/" in model_name else model_name
        self.dim = self.MODEL_DIMS.get(self.variant, 768)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy import to avoid hard dependency
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.to(self.device)
        self.backbone.eval()

        # Standard transform for DINOv2
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token embedding.

        Args:
            x: [B, 3, 224, 224] tensor

        Returns:
            [B, dim] feature vectors
        """
        x = x.to(self.device)
        outputs = self.backbone(x)
        return outputs.last_hidden_state[:, 0]  # CLS token

    def extract(self, image_path: str) -> np.ndarray:
        """Extract features from a single image.

        Args:
            image_path: path to image file

        Returns:
            (dim,) numpy array
        """
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0)
        feat = self.forward(x)
        return feat.squeeze(0).cpu().numpy()

    def extract_batch(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        """Extract features from multiple images.

        Args:
            image_paths: list of image file paths
            batch_size: batch size for inference

        Returns:
            (N, dim) numpy array
        """
        features = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(self.transform(img))
            x = torch.stack(batch_tensors)
            batch_feats = self.forward(x)
            features.append(batch_feats.cpu().numpy())

        return np.concatenate(features, axis=0)

    def get_trainable_params(self) -> dict:
        """Get backbone parameters for FedAvg aggregation.

        Note: In typical FL setup, DINOv2 backbone is frozen
        and only the classification head is trained.
        """
        return {k: v.clone() for k, v in self.backbone.state_dict().items()}

    def load_params(self, state_dict: dict) -> None:
        """Load aggregated parameters."""
        self.backbone.load_state_dict(state_dict)


class ResNet18Extractor(nn.Module):
    """Legacy ResNet-18 feature extractor (kept for backward compatibility)."""

    def __init__(self):
        super().__init__()
        from torchvision import models
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.dim = 512
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector from image tensor."""
        return self.features(x).squeeze(-1).squeeze(-1)  # [B, 512]

    def extract(self, image_path: str) -> np.ndarray:
        """Extract 512-dim feature vector from a single image file."""
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0)
        feat = self.forward(x)
        return feat.squeeze(0).cpu().numpy()


def get_extractor(model_type: str = "dinov2", **kwargs):
    """Factory function for feature extractors.

    Args:
        model_type: "dinov2" or "resnet18"
        **kwargs: passed to extractor constructor

    Returns:
        Feature extractor instance
    """
    if model_type == "dinov2":
        return DINOv2Extractor(**kwargs)
    elif model_type == "resnet18":
        return ResNet18Extractor()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
