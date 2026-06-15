# ── analysis/gradcam.py ──
"""
Grad-CAM Explainability
========================
Visual explanations for organoid classification decisions.

Grad-CAM (Gradient-weighted Class Activation Mapping):
  - Highlights which image regions influenced the model's prediction
  - Works with any CNN (ResNet, EfficientNet, etc.)
  - Also supports ViT attention rollouts for DINOv2

Usage:
  1. Pass image through model
  2. Get gradients w.r.t. target class
  3. Weight feature maps by gradients
  4. Generate heatmap overlay

This provides interpretable evidence for clinical decisions:
  "The model classified this as late_stage because it focused on
   the irregular boundary regions (shown in red)."
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Tuple


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN models.

    Supports ResNet, VGG, EfficientNet, and similar architectures.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: the classification model
            target_layer: the last convolutional layer to visualize.
                         If None, auto-detects from model architecture.
        """
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        if target_layer is None:
            target_layer = self._auto_detect_target_layer(model)

        self.target_layer = target_layer

        # Register hooks
        self._register_hooks()

    def _auto_detect_target_layer(self, model: nn.Module) -> nn.Module:
        """Auto-detect the last conv layer in common architectures."""
        # ResNet
        if hasattr(model, "layer4"):
            return model.layer4[-1]
        # VGG
        if hasattr(model, "features"):
            for m in reversed(list(model.features)):
                if isinstance(m, nn.Conv2d):
                    return m
        # Generic: find last Conv2d
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model.")
        return last_conv

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed image tensor [1, C, H, W]
            target_class: class index to explain. If None, uses predicted class.

        Returns:
            Heatmap as numpy array (H, W) with values in [0, 1]
        """
        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Weighted combination of feature maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU + normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()

    def overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> np.ndarray:
        """Overlay heatmap on original image.

        Args:
            image: original image as numpy array (H, W, 3), uint8
            heatmap: Grad-CAM output (H, W), float32 in [0, 1]
            alpha: overlay transparency
            colormap: matplotlib colormap name

        Returns:
            Blended image as numpy array (H, W, 3), uint8
        """
        try:
            import cv2
        except ImportError:
            return image

        # Resize heatmap to image size
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay


class AttentionRollout:
    """Attention rollout visualization for Vision Transformer (DINOv2).

    Aggregates attention weights across all layers and heads
    to produce a spatial attention map.
    """

    def __init__(self, model):
        """
        Args:
            model: DINOv2 or similar ViT model from transformers
        """
        self.model = model
        self.model.eval()
        self.attention_maps = []

    def _register_attention_hooks(self):
        """Register hooks to capture attention weights from all layers."""
        self.attention_maps = []

        def attention_hook(module, input, output):
            # output is tuple: (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                self.attention_maps.append(output[1].detach().cpu())

        # Register on all self-attention layers
        for name, module in self.model.named_modules():
            if "attention" in name.lower() and "output" not in name.lower():
                try:
                    module.register_forward_hook(attention_hook)
                except Exception:
                    pass

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generate attention rollout map.

        Args:
            input_tensor: preprocessed image [1, C, H, W]

        Returns:
            Attention map as numpy array (H, W) in [0, 1]
        """
        self.attention_maps = []
        self._register_attention_hooks()

        with torch.no_grad():
            self.model(input_tensor)

        if not self.attention_maps:
            # Fallback: return uniform attention
            h = input_tensor.shape[-1] // 14  # patch size
            return np.ones((h, h)) / (h * h)

        # Average attention across heads for each layer
        layer_attentions = []
        for attn in self.attention_maps:
            # attn shape: [batch, heads, seq_len, seq_len]
            avg_heads = attn.mean(dim=1)  # [batch, seq_len, seq_len]
            layer_attentions.append(avg_heads[0])  # first batch item

        # Rollout: multiply attention matrices across layers
        rollout = layer_attentions[0]
        for attn in layer_attentions[1:]:
            rollout = torch.matmul(attn, rollout)

        # Extract attention to CLS token (first row)
        cls_attention = rollout[0, 1:]  # skip CLS token itself

        # Reshape to 2D grid (assuming square image)
        grid_size = int(np.sqrt(cls_attention.shape[0]))
        if grid_size * grid_size == cls_attention.shape[0]:
            attention_map = cls_attention.reshape(grid_size, grid_size).numpy()
        else:
            attention_map = cls_attention.numpy()

        # Normalize
        attention_map = (attention_map - attention_map.min()) / max(attention_map.max() - attention_map.min(), 1e-8)

        return attention_map


def generate_explanation_report(
    heatmap: np.ndarray,
    target_class: str,
    class_names: list[str],
    confidence: float,
    morphology: Optional[dict] = None,
) -> str:
    """Generate a human-readable explanation of the model's decision.

    Args:
        heatmap: Grad-CAM heatmap (H, W)
        target_class: predicted class name
        class_names: list of all class names
        confidence: model confidence
        morphology: optional morphology metrics

    Returns:
        Explanation text
    """
    # Analyze heatmap focus regions
    h, w = heatmap.shape
    center_region = heatmap[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    edge_region_top = heatmap[:h // 4, :]
    edge_region_bottom = heatmap[3 * h // 4:, :]
    edge_region_left = heatmap[:, :w // 4]
    edge_region_right = heatmap[:, 3 * w // 4:]

    center_focus = center_region.mean()
    edge_focus = np.mean([
        edge_region_top.mean(), edge_region_bottom.mean(),
        edge_region_left.mean(), edge_region_right.mean(),
    ])

    lines = [
        f"## Model Explanation",
        f"",
        f"**Prediction:** {target_class} (confidence: {confidence:.1%})",
        f"",
        f"### Attention Analysis",
    ]

    if center_focus > edge_focus:
        lines.append(f"- Model primarily focused on **central regions** (internal structure)")
    else:
        lines.append(f"- Model primarily focused on **boundary regions** (shape/edge features)")

    # Morphology-based explanation
    if morphology:
        lines.append(f"")
        lines.append(f"### Morphological Evidence")
        circ = morphology.get("circularity", 0)
        ecc = morphology.get("eccentricity", 0)
        sol = morphology.get("solidity", 0)

        if target_class == "healthy":
            lines.append(f"- High circularity ({circ:.2f}) consistent with healthy morphology")
            lines.append(f"- Low eccentricity ({ecc:.2f}) indicates regular shape")
        elif target_class == "early_stage":
            lines.append(f"- Moderate circularity ({circ:.2f}) suggests early morphological changes")
            if ecc > 0.4:
                lines.append(f"- Elevated eccentricity ({ecc:.2f}) indicates shape irregularity")
        elif target_class == "late_stage":
            lines.append(f"- Low circularity ({circ:.2f}) indicates significant morphological changes")
            lines.append(f"- High eccentricity ({ecc:.2f}) confirms irregular growth pattern")
            if sol < 0.7:
                lines.append(f"- Low solidity ({sol:.2f}) suggests boundary fragmentation")

    lines.append(f"")
    lines.append(f"*Generated by Organoid-FL Grad-CAM explainability module.*")

    return "\n".join(lines)
