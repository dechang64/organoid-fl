# ── data/synthetic.py ──
"""
Synthetic Data Generator
========================
Generate synthetic organoid-like images and features for demo purposes.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import random
from typing import Optional


def generate_organoid_image(size: int = 128, stage: str = "healthy", seed: Optional[int] = None) -> Image.Image:
    """Generate a synthetic organoid-like microscopy image."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    bg_color = (10, 12, 15)
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    params = {
        "healthy": {
            "n_organoids": random.randint(2, 5),
            "size_range": (15, 35),
            "color_range": ((80, 180, 100), (140, 220, 160)),
            "irregularity": 0.1,
            "n_cells": random.randint(3, 8),
        },
        "early_stage": {
            "n_organoids": random.randint(1, 3),
            "size_range": (20, 45),
            "color_range": ((180, 160, 60), (220, 200, 100)),
            "irregularity": 0.3,
            "n_cells": random.randint(5, 12),
        },
        "late_stage": {
            "n_organoids": random.randint(1, 2),
            "size_range": (25, 55),
            "color_range": ((180, 60, 60), (220, 100, 80)),
            "irregularity": 0.5,
            "n_cells": random.randint(8, 20),
        },
    }

    p = params.get(stage, params["healthy"])

    for _ in range(p["n_organoids"]):
        cx = random.randint(int(size * 0.2), int(size * 0.8))
        cy = random.randint(int(size * 0.2), int(size * 0.8))
        r = random.randint(*p["size_range"])
        color = tuple(random.randint(p["color_range"][0][i], p["color_range"][1][i]) for i in range(3))

        # Draw irregular blob
        points = []
        n_points = 24
        for j in range(n_points):
            angle = 2 * np.pi * j / n_points
            dr = r * (1 + p["irregularity"] * (random.random() - 0.5))
            px = cx + int(dr * np.cos(angle))
            py = cy + int(dr * np.sin(angle))
            points.append((px, py))
        draw.polygon(points, fill=color)

        # Draw cells
        for _ in range(p["n_cells"]):
            cell_angle = random.uniform(0, 2 * np.pi)
            cell_dist = random.uniform(0, r * 0.7)
            cell_cx = cx + int(cell_dist * np.cos(cell_angle))
            cell_cy = cy + int(cell_dist * np.sin(cell_angle))
            cell_r = random.randint(2, 5)
            cell_color = tuple(min(255, c + random.randint(-20, 20)) for c in color)
            draw.ellipse(
                [cell_cx - cell_r, cell_cy - cell_r, cell_cx + cell_r, cell_cy + cell_r],
                fill=cell_color,
            )

    # Add noise
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 3, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    return img


def generate_dataset(
    output_dir: str,
    n_per_class: int = 200,
    img_size: int = 128,
    classes: Optional[list[str]] = None,
) -> Path:
    """Generate full synthetic organoid dataset."""
    if classes is None:
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
