#!/usr/bin/env python3
"""
MultiOrg Tiling Script (v2 - fixed)
- 6384x5720 16-bit TIFF → 640x640 RGB patches
- Polygon JSON annotations → YOLO bbox format
- Explicitly selects Annotator_A
- Output: YOLO-compatible dataset

Usage:
    # Quick test (first 5 images)
    python multiorg_tiling.py --max-images 5

    # Full run
    python multiorg_tiling.py --src D:\\datasets\\mutliorg\\MultiOrg_v2 --dst D:\\datasets\\MultiOrg_YOLO

    # Train after tiling
    yolo detect train model=yolo12s.pt data="D:\\datasets\\MultiOrg_YOLO\\data.yaml" epochs=100 imgsz=640 batch=8
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np


CLASS_MAP = {"Normal": 0, "Macros": 1}
CLASS_NAMES = ["Normal", "Macros"]


def load_annotations(json_path):
    """Load polygon annotations from JSON.
    Format: {'0': [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    annotations = []
    for key, polygon in data.items():
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = x_max - x_min
        h = y_max - y_min
        if w < 2 or h < 2:
            continue
        annotations.append({
            'x_min': x_min, 'y_min': y_min,
            'x_max': x_max, 'y_max': y_max,
            'w': w, 'h': h,
        })
    return annotations


def bbox_to_yolo(ann, tile_x, tile_y, tile_size):
    """Convert absolute bbox to YOLO normalized format within a tile."""
    x_min = max(0, ann['x_min'] - tile_x)
    y_min = max(0, ann['y_min'] - tile_y)
    x_max = min(tile_size, ann['x_max'] - tile_x)
    y_max = min(tile_size, ann['y_max'] - tile_y)

    w = x_max - x_min
    h = y_max - y_min
    if w < 2 or h < 2:
        return None

    xc = (x_min + x_max) / 2.0 / tile_size
    yc = (y_min + y_max) / 2.0 / tile_size
    nw = w / tile_size
    nh = h / tile_size

    return (
        max(0, min(1, xc)),
        max(0, min(1, yc)),
        max(0, min(1, nw)),
        max(0, min(1, nh)),
    )


def convert_tiff_to_rgb(tiff_path):
    """Convert 16-bit TIFF → 8-bit RGB PIL Image (3-channel)."""
    im = Image.open(tiff_path)
    arr = np.array(im, dtype=np.float64)
    vmin, vmax = arr.min(), arr.max()
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin) * 255.0
    else:
        arr = np.zeros_like(arr)
    arr8 = arr.astype(np.uint8)
    # Stack grayscale 3 times → RGB
    rgb = np.stack([arr8, arr8, arr8], axis=-1)
    return Image.fromarray(rgb, mode='RGB')


def find_files(img_dir):
    """Find TIFF and Annotator_A JSON in a directory.
    Returns (tiff_path, json_path) or (None, None).
    """
    tiff_file = None
    json_file = None
    json_candidates = []

    for f in os.listdir(img_dir):
        f_lower = f.lower()
        full = os.path.join(img_dir, f)
        if f_lower.endswith('.tiff') or f_lower.endswith('.tif'):
            tiff_file = full
        elif f_lower.endswith('.json'):
            json_candidates.append(full)

    if tiff_file is None:
        return None, None

    # Prefer Annotator_A, then Annotator_0, then any
    for jf in json_candidates:
        if 'annotator_a' in os.path.basename(jf).lower():
            json_file = jf
            break
    if json_file is None:
        for jf in json_candidates:
            if 'annotator_0' in os.path.basename(jf).lower():
                json_file = jf
                break
    if json_file is None and json_candidates:
        json_file = json_candidates[0]

    return tiff_file, json_file


def process_image(img_dir, output_img_dir, output_lbl_dir, class_id,
                  tile_size, stride, min_objects=1):
    """Process one image → patches."""
    img_name = os.path.basename(img_dir)
    tiff_file, json_file = find_files(img_dir)

    if tiff_file is None:
        return 0

    img_pil = convert_tiff_to_rgb(tiff_file)
    img_w, img_h = img_pil.size

    annotations = []
    if json_file and os.path.exists(json_file):
        annotations = load_annotations(json_file)

    if not annotations:
        return 0

    patch_count = 0

    for ty in range(0, img_h, stride):
        for tx in range(0, img_w, stride):
            tile_w = min(tile_size, img_w - tx)
            tile_h = min(tile_size, img_h - ty)

            if tile_w < tile_size // 2 or tile_h < tile_size // 2:
                continue

            # Annotations whose center falls in this tile
            tile_annotations = []
            for ann in annotations:
                cx = (ann['x_min'] + ann['x_max']) / 2.0
                cy = (ann['y_min'] + ann['y_max']) / 2.0
                if tx <= cx < tx + tile_w and ty <= cy < ty + tile_h:
                    tile_annotations.append(ann)

            if len(tile_annotations) < min_objects:
                continue

            # Crop & pad to full tile_size
            tile_img = img_pil.crop((tx, ty, tx + tile_w, ty + tile_h))
            if tile_w < tile_size or tile_h < tile_size:
                padded = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                padded.paste(tile_img, (0, 0))
                tile_img = padded

            # YOLO labels
            yolo_lines = []
            for ann in tile_annotations:
                result = bbox_to_yolo(ann, tx, ty, tile_size)
                if result is not None:
                    xc, yc, w, h = result
                    yolo_lines.append(
                        f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                    )

            if not yolo_lines:
                continue

            patch_name = f"{img_name}_tx{tx}_ty{ty}"
            tile_img.save(os.path.join(output_img_dir, f"{patch_name}.png"))
            with open(os.path.join(output_lbl_dir, f"{patch_name}.txt"), 'w') as f:
                f.write('\n'.join(yolo_lines))

            patch_count += 1

    return patch_count


def process_split(src_dir, dst_dir, split, tile_size, stride,
                  min_objects=1, max_images=None):
    """Process train or test split."""
    split_dir = os.path.join(src_dir, split)
    if not os.path.isdir(split_dir):
        print(f"[WARN] {split_dir} not found")
        return 0, {0: 0, 1: 0}

    out_img = os.path.join(dst_dir, split, 'images', split)
    out_lbl = os.path.join(dst_dir, split, 'labels', split)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    total = 0
    class_counts = {0: 0, 1: 0}
    img_count = 0

    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\n  {split}/{class_name} (id={class_id})...")
        class_img_count = 0

        for plate_name in sorted(os.listdir(class_dir)):
            plate_dir = os.path.join(class_dir, plate_name)
            if not os.path.isdir(plate_dir):
                continue
            for img_dir_name in sorted(os.listdir(plate_dir)):
                img_dir = os.path.join(plate_dir, img_dir_name)
                if not os.path.isdir(img_dir):
                    continue

                if max_images and class_img_count >= max_images:
                    break

                n = process_image(
                    img_dir, out_img, out_lbl, class_id,
                    tile_size, stride, min_objects
                )
                total += n
                class_counts[class_id] += n
                class_img_count += 1
                img_count += 1

                if img_count % 20 == 0:
                    print(f"    [{img_count} images processed] "
                          f"running total: {total} patches")

            if max_images and class_img_count >= max_images:
                break

    return total, class_counts


def create_data_yaml(dst_dir):
    """Create YOLO data.yaml."""
    dst_fwd = dst_dir.replace('\\', '/')
    yaml_content = f"""# MultiOrg YOLO Dataset
# Generated by multiorg_tiling.py v2
path: {dst_fwd}
train: train/images/train
val: test/images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path = os.path.join(dst_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nCreated {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description='MultiOrg Tiling + YOLO Conversion (v2)'
    )
    parser.add_argument(
        '--src',
        default=r'D:\datasets\mutliorg\MultiOrg_v2',
        help='Source MultiOrg_v2 directory'
    )
    parser.add_argument(
        '--dst',
        default=r'D:\datasets\MultiOrg_YOLO',
        help='Output YOLO dataset directory'
    )
    parser.add_argument('--tile', type=int, default=640)
    parser.add_argument('--stride', type=int, default=640)
    parser.add_argument('--min-objects', type=int, default=1)
    parser.add_argument(
        '--max-images', type=int, default=None,
        help='Limit images per split (for quick test)'
    )
    args = parser.parse_args()

    print(f"Source:  {args.src}")
    print(f"Output:  {args.dst}")
    print(f"Tile: {args.tile}, Stride: {args.stride}")
    if args.max_images:
        print(f"[QUICK TEST] Max {args.max_images} images per split")
    print("=" * 60)

    all_counts = {}
    for split in ['train', 'test']:
        n, cc = process_split(
            args.src, args.dst, split,
            args.tile, args.stride, args.min_objects,
            args.max_images
        )
        all_counts[split] = {'total': n, 'classes': cc}
        print(f"\n  => {split}: {n} patches "
              f"(Normal={cc[0]}, Macros={cc[1]})")

    yaml_path = create_data_yaml(args.dst)

    # Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    for split, info in all_counts.items():
        total = info['total']
        cc = info['classes']
        if total > 0:
            ratio = cc[0] / total * 100
            print(f"  {split:6s}: {total:6d} patches | "
                  f"Normal={cc[0]} ({ratio:.1f}%) | "
                  f"Macros={cc[1]} ({100-ratio:.1f}%)")
        else:
            print(f"  {split:6s}: 0 patches")

    grand_total = sum(v['total'] for v in all_counts.values())
    print(f"\n  Total: {grand_total} patches")

    print("\n" + "=" * 60)
    print("OUTPUT STRUCTURE:")
    print(f"  {args.dst}/")
    print(f"  ├── train/images/train/*.png  (RGB 640x640)")
    print(f"  ├── train/labels/train/*.txt  (YOLO format)")
    print(f"  ├── test/images/test/*.png")
    print(f"  ├── test/labels/test/*.txt")
    print(f"  └── data.yaml")
    print(f"\nTrain command:")
    print(f'  yolo detect train model=yolo12s.pt '
          f'data="{yaml_path}" epochs=100 imgsz=640 batch=8')


if __name__ == '__main__':
    main()
