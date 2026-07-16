#!/usr/bin/env python3
"""
Crop organoid bbox patches from YOLO detection dataset → classification dataset.

Input:  intestinal_organoid/OrganoidDataset/  (840 imgs, 23K YOLO annotations)
Output: organoid_patches/  (23K+ cropped patches, 4-class classification)

Usage:
    python3 crop_patches.py [--min-size 20] [--padding 4]
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
from collections import Counter

CLASS_NAMES = {
    0: "organoid0",   # cystic non-budding
    1: "organoid1",   # early organoid
    2: "organoid3",   # late organoid
    3: "spheroid",
}


def parse_yolo_label(label_path, img_w, img_h):
    """Parse YOLO format label → list of (cls, x1, y1, x2, y2) in pixels."""
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            boxes.append((cls, int(x1), int(y1), int(x2), int(y2)))
    return boxes


def crop_and_save(img, boxes, output_dir, img_stem, min_size=20, padding=4):
    """Crop each bbox and save as classification patch."""
    img_w, img_h = img.size
    counts = Counter()

    for i, (cls, x1, y1, x2, y2) in enumerate(boxes):
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_w, x2 + padding)
        y2 = min(img_h, y2 + padding)

        # Skip tiny boxes
        bw = x2 - x1
        bh = y2 - y1
        if bw < min_size or bh < min_size:
            counts["skipped_tiny"] += 1
            continue

        # Crop
        patch = img.crop((x1, y1, x2, y2))

        # Save
        cls_name = CLASS_NAMES.get(cls, f"class{cls}")
        cls_dir = output_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        patch_path = cls_dir / f"{img_stem}_{i:04d}.jpg"
        patch.save(patch_path, quality=95)
        counts[cls_name] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description="Crop organoid patches for classification")
    parser.add_argument("--input", type=str,
                        default="intestinal_organoid/OrganoidDataset",
                        help="Input YOLO dataset root")
    parser.add_argument("--output", type=str,
                        default="organoid_patches",
                        help="Output classification dataset root")
    parser.add_argument("--min-size", type=int, default=20,
                        help="Min bbox size in pixels (skip smaller)")
    parser.add_argument("--padding", type=int, default=4,
                        help="Padding around bbox in pixels")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation split ratio")
    args = parser.parse_args()

    base = Path(__file__).parent
    input_dir = base / args.input
    output_dir = base / args.output

    if not input_dir.exists():
        print(f"Input not found: {input_dir}")
        sys.exit(1)

    # Prepare output dirs
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    total_counts = Counter()
    img_count = 0
    total_boxes = 0

    # Process train and val splits
    for split in ["train", "val"]:
        img_dir = input_dir / split / "images"
        lbl_dir = input_dir / split / "labels"

        if not img_dir.exists():
            print(f"Skipping {split}: {img_dir} not found")
            continue

        out_split = val_dir if split == "val" else train_dir

        # For train split, further split into train/val by image index
        images = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])

        print(f"\nProcessing {split}: {len(images)} images")

        for idx, img_name in enumerate(images):
            img_stem = Path(img_name).stem
            img_path = img_dir / img_name
            lbl_path = lbl_dir / f"{img_stem}.txt"

            if not lbl_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
            boxes = parse_yolo_label(lbl_path, img_w, img_h)
            total_boxes += len(boxes)

            # For train split: move ~val_ratio to val
            if split == "train" and idx % int(1 / args.val_ratio) == 0:
                target_dir = val_dir
            else:
                target_dir = out_split

            counts = crop_and_save(img, boxes, target_dir, img_stem,
                                   min_size=args.min_size, padding=args.padding)

            for k, v in counts.items():
                total_counts[k] += v

            img_count += 1
            if img_count % 100 == 0:
                print(f"  {img_count} images processed...")

    # Summary
    print("\n" + "=" * 60)
    print("  Organoid Patch Classification Dataset")
    print("=" * 60)

    valid_classes = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]

    for cls_name in valid_classes:
        train_c = len(list((train_dir / cls_name).glob("*.jpg"))) if (train_dir / cls_name).exists() else 0
        val_c = len(list((val_dir / cls_name).glob("*.jpg"))) if (val_dir / cls_name).exists() else 0
        print(f"  {cls_name:15s}: train={train_c:6d}  val={val_c:5d}  total={train_c + val_c:6d}")

    train_total = sum(len(list((train_dir / c).glob("*.jpg"))) for c in valid_classes if (train_dir / c).exists())
    val_total = sum(len(list((val_dir / c).glob("*.jpg"))) for c in valid_classes if (val_dir / c).exists())
    print(f"  {'TOTAL':15s}: train={train_total:6d}  val={val_total:5d}  total={train_total + val_total:6d}")
    print(f"\n  Source images: {img_count}")
    print(f"  Source bboxes: {total_boxes}")
    print(f"  Skipped (tiny): {total_counts.get('skipped_tiny', 0)}")
    print(f"\n  Output: {output_dir}")

    # Generate data.yaml for Ultralytics classification
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"# Organoid Patch Classification Dataset\n")
        f.write(f"# Auto-generated by crop_patches.py\n")
        f.write(f"path: {output_dir.resolve()}\n")
        f.write(f"train: train\n")
        f.write(f"val: val\n\n")
        f.write(f"nc: {len(valid_classes)}\n")
        f.write(f"names: {valid_classes}\n")
    print(f"  Config: {yaml_path}")

    # Also generate a summary JSON
    import json
    summary = {
        "dataset": "organoid_patches",
        "source": "Zenodo 6768583 (Intestinal Organoid)",
        "task": "classification",
        "num_classes": len(valid_classes),
        "classes": {c: {"train": len(list((train_dir / c).glob("*.jpg"))),
                         "val": len(list((val_dir / c).glob("*.jpg")))}
                    for c in valid_classes if (train_dir / c).exists()},
        "train_total": train_total,
        "val_total": val_total,
        "source_images": img_count,
        "source_bboxes": total_boxes,
        "skipped_tiny": total_counts.get("skipped_tiny", 0),
        "min_size": args.min_size,
        "padding": args.padding,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
