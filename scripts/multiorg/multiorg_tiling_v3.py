#!/usr/bin/env python3
"""
MultiOrg Tiling v3 — 单类 organoid + 512 patch + 丢弃边界 bbox + 多标注者支持

关键修正（vs v2 tiling）:
1. 单类 organoid（class 0），不是 Normal/Macros 两类
2. 512×512 patch（不是 640）
3. 丢弃跨 patch 边界的 bbox（只保留中心在 patch 内的）
4. 支持选择标注者（Annotator_A/B/C）用于多标注者实验
5. 输出 YOLO 格式（RF-DETR 兼容）

Usage:
    # 基础版（单标注者 A）
    python multiorg_tiling_v3.py --src D:\\datasets\\mutliorg\\MultiOrg_v2 --dst D:\\datasets\\MultiOrg_v3_512

    # 多标注者版（生成 3 套标签）
    python multiorg_tiling_v3.py --src D:\\datasets\\mutliorg\\MultiOrg_v2 --dst D:\\datasets\\MultiOrg_v3_512_multi --multi-rater
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np


TILE_SIZE = 640
STRIDE = 640  # 无重叠
CLASS_ID = 0  # 单类 organoid
CLASS_NAMES = ["organoid"]


def load_annotations(json_path):
    """Load polygon annotations from JSON.
    MultiOrg format: {'0': [[y1,x1],[y2,x2],[y3,x3],[y4,x4]], ...}
    NOTE: [row, col] = [y, x] order (napari convention)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    annotations = []
    for key, polygon in data.items():
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        xs = [p[1] for p in polygon]  # col = x
        ys = [p[0] for p in polygon]  # row = y
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
            'cx': (x_min + x_max) / 2.0,
            'cy': (y_min + y_max) / 2.0,
        })
    return annotations


def bbox_to_yolo(ann, tile_x, tile_y, tile_size, drop_boundary=True):
    """Convert bbox to YOLO format within a tile.

    Args:
        drop_boundary: If True, drop bboxes whose center is outside the tile.
                       If False, clip bboxes at tile boundary (old behavior).
    """
    if drop_boundary:
        # 只保留中心在 patch 内的 bbox
        cx = ann['cx'] - tile_x
        cy = ann['cy'] - tile_y
        if not (0 <= cx < tile_size and 0 <= cy < tile_size):
            return None
        # 使用完整 bbox（可能跨边界），但裁剪到 patch 范围
        x_min = max(0, ann['x_min'] - tile_x)
        y_min = max(0, ann['y_min'] - tile_y)
        x_max = min(tile_size, ann['x_max'] - tile_x)
        y_max = min(tile_size, ann['y_max'] - tile_y)
    else:
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
    """Convert 16-bit TIFF → 8-bit RGB PIL Image."""
    im = Image.open(tiff_path)
    if im.mode == 'I;16' or im.mode == 'I':
        arr = np.array(im, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax > vmin:
            arr8 = ((arr - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        else:
            arr8 = np.zeros_like(arr, dtype=np.uint8)
        rgb = np.stack([arr8, arr8, arr8], axis=-1)
        return Image.fromarray(rgb, mode='RGB')
    return im.convert('RGB')


def find_annotation_files(img_dir):
    """Find all annotation JSON files in a directory.
    Returns dict: {'annotator_a': path, 'annotator_b': path, ...}
    """
    result = {}
    for f in os.listdir(img_dir):
        if not f.lower().endswith('.json'):
            continue
        lower = f.lower()
        full_path = os.path.join(img_dir, f)
        if 'annotator_a' in lower:
            result['annotator_a'] = full_path
        elif 'annotator_b' in lower:
            result['annotator_b'] = full_path
        elif 'annotator_c' in lower:
            result['annotator_c'] = full_path
        elif 't0' in lower:
            result.setdefault('t0', full_path)
        elif 't1_a' in lower or 't1a' in lower:
            result['t1_a'] = full_path
        elif 't1_b' in lower or 't1b' in lower:
            result['t1_b'] = full_path
        else:
            result.setdefault('other', full_path)
    return result


def find_tiff(img_dir):
    """Find TIFF file in a directory."""
    for f in os.listdir(img_dir):
        if f.lower().endswith(('.tiff', '.tif')):
            return os.path.join(img_dir, f)
    return None


def process_image_multi_rater(img_dir, output_img_dir, output_lbl_dirs,
                               tile_size, stride, min_objects=1,
                               split='', plate_name='', img_name='',
                               drop_boundary=True):
    """Process one image with multiple annotators.
    output_lbl_dirs: dict of {'annotator_a': path, 'annotator_b': path, ...}
    drop_boundary: True for train (clean signal), False for val (keep all bboxes, clip to edge)
    """
    tiff_file = find_tiff(img_dir)
    if tiff_file is None:
        return 0

    ann_files = find_annotation_files(img_dir)
    if not ann_files:
        return 0

    # 检查输出目录中是否有 'any' 模式（单标注者合并）
    is_any_mode = 'any' in output_lbl_dirs

    # 加载所有标注
    all_annotations = {}
    for ann_key, ann_path in ann_files.items():
        all_annotations[ann_key] = load_annotations(ann_path)

    if not all_annotations:
        return 0

    # 使用第一个标注者的 bbox 来决定哪些 patch 有内容
    primary_key = list(all_annotations.keys())[0]
    primary_annotations = all_annotations[primary_key]

    img_pil = convert_tiff_to_rgb(tiff_file)
    img_w, img_h = img_pil.size

    patch_count = 0

    for ty in range(0, img_h, stride):
        for tx in range(0, img_w, stride):
            tile_w = min(tile_size, img_w - tx)
            tile_h = min(tile_size, img_h - ty)
            if tile_w < tile_size // 2 or tile_h < tile_size // 2:
                continue

            # 检查 primary 标注是否有中心在此 patch 的 bbox
            tile_anns_primary = [
                ann for ann in primary_annotations
                if tx <= ann['cx'] < tx + tile_w and ty <= ann['cy'] < ty + tile_h
            ]
            if len(tile_anns_primary) < min_objects:
                continue

            # 裁剪 & 填充
            tile_img = img_pil.crop((tx, ty, tx + tile_w, ty + tile_h))
            if tile_w < tile_size or tile_h < tile_size:
                padded = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                padded.paste(tile_img, (0, 0))
                tile_img = padded

            patch_name = f"{split}_{plate_name}_{img_name}_tx{tx}_ty{ty}"
            tile_img.save(os.path.join(output_img_dir, f"{patch_name}.png"))

            # 为每个标注者生成标签
            if is_any_mode:
                # 单标注者模式：只使用 primary 标注者（第一个找到的）
                # 避免合并多套标注导致 GT 重复
                # - train: 每图只有1个标注者(A或B)，primary 就是它
                # - test: 每图有3套(t0/t1_a/t1_b)，只用 t0 避免重复
                yolo_lines = []
                for ann in primary_annotations:
                    result = bbox_to_yolo(ann, tx, ty, tile_size, drop_boundary=drop_boundary)
                    if result is not None:
                        xc, yc, w, h = result
                        yolo_lines.append(
                            f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                        )
                lbl_path = os.path.join(output_lbl_dirs['any'], f"{patch_name}.txt")
                with open(lbl_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
            else:
                # 多标注者模式：每个标注者一个标签目录
                for ann_key, annotations in all_annotations.items():
                    if ann_key not in output_lbl_dirs:
                        continue
                    yolo_lines = []
                    for ann in annotations:
                        result = bbox_to_yolo(ann, tx, ty, tile_size, drop_boundary=drop_boundary)
                        if result is not None:
                            xc, yc, w, h = result
                            yolo_lines.append(
                                f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                            )
                    lbl_path = os.path.join(output_lbl_dirs[ann_key], f"{patch_name}.txt")
                    with open(lbl_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))

            patch_count += 1

    return patch_count


def process_split(src_dir, dst_dir, split, tile_size, stride,
                  min_objects=1, multi_rater=False):
    """Process train or test split.
    
    Both train and val use drop_boundary=True:
    - train: clean signal (only center-in-patch bboxes)
    - val: each organoid evaluated only in patch where its center lies
      (closest to real sliding-window inference — edge organoids detected by neighboring patch)
    """
    drop_boundary = True  # both train and val
    split_dir = os.path.join(src_dir, split)
    if not os.path.isdir(split_dir):
        print(f"[WARN] {split_dir} not found")
        return 0

    out_img = os.path.join(dst_dir, split, 'images')
    os.makedirs(out_img, exist_ok=True)

    # 输出标签目录
    output_lbl_dirs = {}
    if multi_rater:
        # 多标注者：每个标注者一个 labels 子目录
        # 先探查实际有哪些标注者
        ann_types = set()
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for plate in os.listdir(class_dir):
                plate_dir = os.path.join(class_dir, plate)
                if not os.path.isdir(plate_dir):
                    continue
                for img_dir_name in os.listdir(plate_dir):
                    img_dir = os.path.join(plate_dir, img_dir_name)
                    if not os.path.isdir(img_dir):
                        continue
                    ann_files = find_annotation_files(img_dir)
                    ann_types.update(ann_files.keys())
                    if ann_types:
                        break
                if ann_types:
                    break
            if ann_types:
                break

        for ann_type in sorted(ann_types):
            lbl_dir = os.path.join(dst_dir, split, f'labels_{ann_type}')
            os.makedirs(lbl_dir, exist_ok=True)
            output_lbl_dirs[ann_type] = lbl_dir
    else:
        # 单标注者：用所有可用标注（train 中 A 和 B 是不同图片，合并到同一目录）
        lbl_dir = os.path.join(dst_dir, split, 'labels')
        os.makedirs(lbl_dir, exist_ok=True)
        # 标记为 'any' — process_image_multi_rater 会用找到的第一个标注
        output_lbl_dirs['any'] = lbl_dir

    total = 0
    img_count = 0

    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\n  {split}/{class_name}...")
        class_img_count = 0

        for plate_name in sorted(os.listdir(class_dir)):
            plate_dir = os.path.join(class_dir, plate_name)
            if not os.path.isdir(plate_dir):
                continue
            for img_dir_name in sorted(os.listdir(plate_dir)):
                img_dir = os.path.join(plate_dir, img_dir_name)
                if not os.path.isdir(img_dir):
                    continue

                n = process_image_multi_rater(
                    img_dir, out_img, output_lbl_dirs,
                    tile_size, stride, min_objects,
                    split=split, plate_name=plate_name, img_name=img_dir_name,
                    drop_boundary=drop_boundary
                )
                total += n
                img_count += 1
                class_img_count += 1

                if img_count % 20 == 0:
                    print(f"    [{img_count} images] {total} patches so far")

    return total


def create_data_yaml(dst_dir):
    """Create YOLO data.yaml for single-class."""
    dst_fwd = dst_dir.replace('\\', '/')
    yaml_content = f"""# MultiOrg YOLO Dataset v3 (single-class, 512px)
# Generated by multiorg_tiling_v3.py
path: {dst_fwd}
train: train/images
val: test/images

nc: 1
names: {CLASS_NAMES}
"""
    yaml_path = os.path.join(dst_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nCreated {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description='MultiOrg Tiling v3 — single-class, 512px, drop boundary bbox'
    )
    parser.add_argument(
        '--src',
        default=r'D:\datasets\mutliorg\MultiOrg_v2',
        help='Source MultiOrg_v2 directory'
    )
    parser.add_argument(
        '--dst',
        default=r'D:\datasets\MultiOrg_v3_512',
        help='Output YOLO dataset directory'
    )
    parser.add_argument('--tile', type=int, default=TILE_SIZE)
    parser.add_argument('--stride', type=int, default=STRIDE)
    parser.add_argument('--min-objects', type=int, default=1)
    parser.add_argument('--multi-rater', action='store_true',
                        help='Generate separate labels for each annotator')
    args = parser.parse_args()

    print(f"Source:  {args.src}")
    print(f"Output:  {args.dst}")
    print(f"Tile: {args.tile}, Stride: {args.stride}")
    print(f"Class: single (organoid)")
    print(f"Multi-rater: {args.multi_rater}")
    print("=" * 60)

    all_counts = {}
    for split in ['train', 'test']:
        n = process_split(
            args.src, args.dst, split,
            args.tile, args.stride, args.min_objects,
            args.multi_rater
        )
        all_counts[split] = n
        print(f"\n  => {split}: {n} patches")

    yaml_path = create_data_yaml(args.dst)

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    for split, n in all_counts.items():
        print(f"  {split:6s}: {n:6d} patches")
    print(f"\n  Total: {sum(all_counts.values())} patches")
    print(f"\n  Train command (YOLO):")
    print(f'  yolo detect train model=yolo12s.pt '
          f'data="{yaml_path}" epochs=400 imgsz={args.tile} batch=8')
    print(f"\n  Train command (RF-DETR):")
    print(f'  python train_rfdetr.py --data "{yaml_path}" '
          f'--model base --epochs 200 --imgsz {args.tile}')


if __name__ == '__main__':
    main()
