#!/usr/bin/env python3
"""
MultiOrg Tiling Script
- 6384x5720 16-bit TIFF → 640x640 patches
- Polygon JSON annotations → YOLO bbox format
- Output: YOLO-compatible dataset on D: drive

Usage:
    python multiorg_tiling.py --src D:\datasets\mutliorg\MultiOrg_v2 --dst D:\datasets\MultiOrg_YOLO --tile 640 --stride 640
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np


# Class mapping: folder name → YOLO class id
CLASS_MAP = {
    "Normal": 0,
    "Macros": 1,
}

CLASS_NAMES = ["Normal", "Macros"]


def load_annotations(json_path):
    """Load polygon annotations from JSON file.
    Format: {'0': [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = []
    for key, polygon in data.items():
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
    
    xc = max(0, min(1, xc))
    yc = max(0, min(1, yc))
    nw = max(0, min(1, nw))
    nh = max(0, min(1, nh))
    
    return (xc, yc, nw, nh)


def convert_tiff_to_8bit(tiff_path):
    """Convert 16-bit TIFF to 8-bit PIL Image."""
    im = Image.open(tiff_path)
    arr = np.array(im, dtype=np.float64)
    vmin, vmax = arr.min(), arr.max()
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin) * 255.0
    else:
        arr = np.zeros_like(arr)
    return Image.fromarray(arr.astype(np.uint8), mode='L')


def process_image(img_dir, output_img_dir, output_lbl_dir, class_id, tile_size, stride, min_objects=1):
    """Process one image directory → patches."""
    img_name = os.path.basename(img_dir)
    
    tiff_file = None
    json_file = None
    for f in os.listdir(img_dir):
        if f.endswith('.tiff') or f.endswith('.tif'):
            tiff_file = os.path.join(img_dir, f)
        elif f.endswith('.json'):
            json_file = os.path.join(img_dir, f)
    
    if tiff_file is None:
        return 0
    
    img_pil = convert_tiff_to_8bit(tiff_file)
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
            
            # Find annotations whose center falls in this tile
            tile_annotations = []
            for ann in annotations:
                cx = (ann['x_min'] + ann['x_max']) / 2.0
                cy = (ann['y_min'] + ann['y_max']) / 2.0
                if tx <= cx < tx + tile_w and ty <= cy < ty + tile_h:
                    tile_annotations.append(ann)
            
            if len(tile_annotations) < min_objects:
                continue
            
            # Crop & pad
            tile_img = img_pil.crop((tx, ty, tx + tile_w, ty + tile_h))
            if tile_w < tile_size or tile_h < tile_size:
                padded = Image.new('L', (tile_size, tile_size), 0)
                padded.paste(tile_img, (0, 0))
                tile_img = padded
            
            # YOLO labels
            yolo_lines = []
            for ann in tile_annotations:
                result = bbox_to_yolo(ann, tx, ty, tile_size)
                if result is not None:
                    xc, yc, w, h = result
                    yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            
            if not yolo_lines:
                continue
            
            patch_name = f"{img_name}_tx{tx}_ty{ty}"
            tile_img.save(os.path.join(output_img_dir, f"{patch_name}.png"))
            with open(os.path.join(output_lbl_dir, f"{patch_name}.txt"), 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            patch_count += 1
    
    return patch_count


def process_split(src_dir, dst_dir, split, tile_size, stride, min_objects=1):
    """Process train or test split."""
    split_dir = os.path.join(src_dir, split)
    if not os.path.isdir(split_dir):
        print(f"[WARN] {split_dir} not found")
        return 0
    
    total = 0
    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        out_img = os.path.join(dst_dir, split, 'images', split)
        out_lbl = os.path.join(dst_dir, split, 'labels', split)
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        
        print(f"\n  {split}/{class_name} (id={class_id})...")
        
        for plate_name in sorted(os.listdir(class_dir)):
            plate_dir = os.path.join(class_dir, plate_name)
            if not os.path.isdir(plate_dir):
                continue
            for img_dir_name in sorted(os.listdir(plate_dir)):
                img_dir = os.path.join(plate_dir, img_dir_name)
                if not os.path.isdir(img_dir):
                    continue
                n = process_image(img_dir, out_img, out_lbl, class_id, tile_size, stride, min_objects)
                total += n
    
    return total


def create_data_yaml(dst_dir):
    """Create YOLO data.yaml."""
    # Use forward slashes for YOLO compatibility
    dst_fwd = dst_dir.replace('\\', '/')
    yaml_content = f"""# MultiOrg YOLO Dataset
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
    parser = argparse.ArgumentParser(description='MultiOrg Tiling + YOLO Conversion')
    parser.add_argument('--src', default=r'D:\datasets\mutliorg\MultiOrg_v2')
    parser.add_argument('--dst', default=r'D:\datasets\MultiOrg_YOLO')
    parser.add_argument('--tile', type=int, default=640)
    parser.add_argument('--stride', type=int, default=640)
    parser.add_argument('--min-objects', type=int, default=1)
    args = parser.parse_args()
    
    print(f"Source: {args.src}")
    print(f"Output: {args.dst}")
    print(f"Tile: {args.tile}, Stride: {args.stride}")
    print("=" * 60)
    
    for split in ['train', 'test']:
        n = process_split(args.src, args.dst, split, args.tile, args.stride, args.min_objects)
        print(f"\n  => {split}: {n} patches")
    
    yaml_path = create_data_yaml(args.dst)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  {args.dst}/")
    print(f"  ├── train/images/train/*.png")
    print(f"  ├── train/labels/train/*.txt")
    print(f"  ├── test/images/test/*.png")
    print(f"  ├── test/labels/test/*.txt")
    print(f"  └── data.yaml")
    print(f'\nTrain: yolo detect train model=yolo12n.pt data="{yaml_path}" epochs=100 imgsz=640 batch=8')


if __name__ == '__main__':
    main()
