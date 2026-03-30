"""
Generate synthetic organoid-like images for federated learning simulation.

Creates circular/elliptical blob images that mimic organoid microscopy:
- Multiple cell-like structures per image
- Varying sizes, colors, textures
- 3 classes: healthy, early-stage, late-stage
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import random
from pathlib import Path


def generate_organoid_image(size=128, stage="healthy", seed=None):
    """Generate a synthetic organoid-like microscopy image."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Background: dark microscopy-like
    bg_color = (10, 12, 15)
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Parameters by stage
    params = {
        "healthy": {
            "n_organoids": random.randint(2, 5),
            "size_range": (15, 35),
            "color_range": ((80, 180, 100), (140, 220, 160)),  # green-ish
            "irregularity": 0.1,
            "n_cells": random.randint(3, 8),
        },
        "early_stage": {
            "n_organoids": random.randint(3, 7),
            "size_range": (10, 25),
            "color_range": ((180, 160, 80), (220, 200, 120)),  # yellow-ish
            "irregularity": 0.25,
            "n_cells": random.randint(5, 12),
        },
        "late_stage": {
            "n_organoids": random.randint(1, 4),
            "size_range": (20, 45),
            "color_range": ((180, 80, 80), (220, 120, 100)),  # red-ish
            "irregularity": 0.4,
            "n_cells": random.randint(8, 20),
        },
    }
    
    p = params[stage]
    
    for _ in range(p["n_organoids"]):
        # Random position
        cx = random.randint(20, size - 20)
        cy = random.randint(20, size - 20)
        r = random.randint(*p["size_range"])
        
        # Color interpolation
        c_lo, c_hi = p["color_range"]
        color = tuple(random.randint(lo, hi) for lo, hi in zip(c_lo, c_hi))
        
        # Draw irregular ellipse (organoid shape)
        irregularity = p["irregularity"]
        points = []
        n_points = 36
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            rx = r * (1 + irregularity * random.uniform(-1, 1))
            ry = r * (1 + irregularity * random.uniform(-1, 1))
            x = cx + rx * np.cos(angle)
            y = cy + ry * np.sin(angle)
            points.append((x, y))
        
        draw.polygon(points, fill=color)
        
        # Add cell-like internal structures
        for _ in range(p["n_cells"]):
            cell_cx = cx + random.randint(-r//2, r//2)
            cell_cy = cy + random.randint(-r//2, r//2)
            cell_r = random.randint(2, max(3, r // 5))
            cell_color = tuple(min(255, c + random.randint(-20, 40)) for c in color)
            draw.ellipse(
                [cell_cx - cell_r, cell_cy - cell_r, cell_cx + cell_r, cell_cy + cell_r],
                fill=cell_color,
            )
    
    # Add noise (microscopy artifacts)
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 3, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    
    # Slight blur to simulate microscopy focus
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return img


def generate_dataset(output_dir, n_per_class=200, img_size=128):
    """Generate full synthetic dataset."""
    classes = ["healthy", "early_stage", "late_stage"]
    output_dir = Path(output_dir)
    
    for cls in classes:
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(n_per_class):
            img = generate_organoid_image(size=img_size, stage=cls, seed=i)
            img.save(cls_dir / f"{cls}_{i:04d}.png")
    
    print(f"Generated {n_per_class * len(classes)} images in {output_dir}")
    return output_dir


if __name__ == "__main__":
    generate_dataset("/home/z/my-project/organoid-fl/fl/data", n_per_class=200, img_size=128)
