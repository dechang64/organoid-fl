"""
Feature extractor using ResNet18.

Extracts 512-dim feature vectors from organoid images.
Uses pretrained ResNet18 backbone (ImageNet), frozen.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class ResNet18Extractor(nn.Module):
    """ResNet18 feature extractor (removes final FC layer)."""
    
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the classification head
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.eval()
    
    def forward(self, x):
        with torch.no_grad():
            return self.features(x).squeeze(-1).squeeze(-1)  # [B, 512]


# Standard transform for ResNet18
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_features(image_path, model):
    """Extract feature vector from a single image."""
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)  # [1, 3, 224, 224]
    feature = model(tensor)  # [1, 512]
    return feature.squeeze(0).numpy()  # [512]


def extract_dataset(data_dir, model, output_path=None):
    """Extract features from all images in a directory."""
    data_dir = Path(data_dir)
    features = []
    labels = []
    paths = []
    
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    print(f"Extracting features from {data_dir}")
    print(f"Classes: {classes}")
    
    total = sum(len(list((data_dir / c).glob("*.png"))) for c in classes)
    count = 0
    
    for cls in classes:
        cls_dir = data_dir / cls
        for img_path in sorted(cls_dir.glob("*.png")):
            feat = extract_features(img_path, model)
            features.append(feat)
            labels.append(class_to_idx[cls])
            paths.append(str(img_path))
            count += 1
            if count % 50 == 0:
                print(f"  {count}/{total}")
    
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    print(f"Extracted {len(features)} features, shape: {features.shape}")
    
    if output_path:
        np.savez(output_path, features=features, labels=labels, paths=paths, classes=classes)
        print(f"Saved to {output_path}")
    
    return features, labels, paths, classes


if __name__ == "__main__":
    model = ResNet18Extractor()
    
    # Generate data if not exists
    data_dir = Path("/home/z/my-project/organoid-fl/fl/data")
    if not data_dir.exists():
        from generate_data import generate_dataset
        generate_dataset(str(data_dir), n_per_class=200, img_size=128)
    
    extract_dataset(str(data_dir), model, "/home/z/my-project/organoid-fl/fl/features.npz")
