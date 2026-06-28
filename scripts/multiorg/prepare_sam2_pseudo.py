r"""
方案 A：自蒸馏——用 SAM2 zero-shot 生成伪 mask 替代 GT 4点多边形

数据流：
1. GT 4点多边形 → bbox（位置 prompt）
2. SAM2 zero-shot 用 bbox 做 prompt → 伪 mask（精确轮廓）
3. 伪 mask 作为微调 supervision（替代粗糙的 4点多边形 mask）

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate

    python scripts\multiorg\prepare_sam2_pseudo.py ^
        --src D:\datasets\mutliorg\MultiOrg_v2\train ^
        --dst data\multiorg_sam2_pseudo ^
        --sam2-checkpoint sam2_checkpoints\sam2_hiera_small.pt ^
        --annotator Annotator_A
"""
import os
import sys
import json
import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


def load_sam2_predictor(checkpoint, device='cuda'):
    """加载 SAM2 zero-shot predictor"""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    for cfg in ['sam2_hiera_s', 'sam2_hiera_small']:
        try:
            model = build_sam2(cfg, checkpoint, device=device)
            return SAM2ImagePredictor(model)
        except Exception:
            continue
    raise RuntimeError(f"Could not load SAM2 with checkpoint={checkpoint}")


def load_tiff_rgb(tiff_path):
    """16-bit TIFF → 8-bit RGB (用 tifffile 避免 PIL segfault)"""
    try:
        import tifffile
        arr = tifffile.imread(tiff_path)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = arr.astype(np.float32)
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax > vmin:
            arr8 = ((arr - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        else:
            arr8 = np.zeros_like(arr, dtype=np.uint8)
        rgb = np.stack([arr8, arr8, arr8], axis=-1)
        return rgb
    except ImportError:
        im = Image.open(tiff_path)
        if im.mode == 'I;16' or im.mode == 'I':
            arr = np.array(im, dtype=np.float64)
            vmin, vmax = float(arr.min()), float(arr.max())
            if vmax > vmin:
                arr8 = ((arr - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            else:
                arr8 = np.zeros_like(arr, dtype=np.uint8)
            return np.stack([arr8, arr8, arr8], axis=-1)
        return np.array(im.convert('RGB'))


def load_annotations(json_path):
    """加载 napari JSON 标注 → list of polygons"""
    with open(json_path) as f:
        data = json.load(f)
    polygons = []
    for key, shape in data.items():
        if isinstance(shape, dict) and 'vertices' in shape:
            polygons.append(shape['vertices'])
        elif isinstance(shape, list):
            polygons.append(shape)
    return polygons


def polygon_to_bbox(polygon, img_h, img_w):
    """napari 多边形 [[y,x],...] → bbox [x1,y1,x2,y2]"""
    xs = [int(p[1]) for p in polygon]  # col = x
    ys = [int(p[0]) for p in polygon]  # row = y
    x1 = max(0, min(xs))
    y1 = max(0, min(ys))
    x2 = min(img_w, max(xs))
    y2 = min(img_h, max(ys))
    return [x1, y1, x2, y2]


def process_split(src_dir, dst_dir, predictor, annotator='Annotator_A', split_name='all', max_images=None, device='cuda'):
    """处理一个 split：用 SAM2 zero-shot 生成伪 mask"""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir) / split_name
    (dst_path / 'images').mkdir(parents=True, exist_ok=True)
    (dst_path / 'masks').mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(src_path.rglob('*.tiff'))
    if max_images:
        tiff_files = tiff_files[:max_images]

    print(f"  Found {len(tiff_files)} TIFF files")

    manifest = []
    for i, img_path in enumerate(tiff_files):
        img_stem = img_path.stem
        class_name = img_path.parent.parent.parent.name
        out_name = f"{class_name}_{img_path.parent.parent.name}_{img_path.parent.name}_{img_stem}"
        img_out = dst_path / 'images' / f'{out_name}.png'
        mask_out = dst_path / 'masks' / f'{out_name}.png'

        if img_out.exists() and mask_out.exists():
            print(f"  [{i+1}/{len(tiff_files)}] cached, skip")
            continue

        if i % 10 == 0:
            print(f"  [{i+1}/{len(tiff_files)}] {img_path.name} ...", end='', flush=True)

        # 找标注
        json_name = f"{img_stem}_{annotator}.json"
        json_path = img_path.parent / json_name
        if not json_path.exists():
            for alt in ['Annotator_A', 'Annotator_B', 'Annotator_C']:
                alt_path = img_path.parent / f"{img_stem}_{alt}.json"
                if alt_path.exists():
                    json_path = alt_path
                    break
            else:
                if i % 10 == 0:
                    print(f" no annotation, skip")
                continue

        # 加载图片
        img_rgb = load_tiff_rgb(img_path)
        h, w = img_rgb.shape[:2]

        # 加载标注 → bbox（只用位置，不用多边形形状）
        polygons = load_annotations(json_path)
        if not polygons:
            continue

        bboxes = [polygon_to_bbox(p, h, w) for p in polygons]

        # SAM2 zero-shot 生成伪 mask
        predictor.set_image(img_rgb)
        instance_map = np.zeros((h, w), dtype=np.uint16)
        instances = []
        inst_id = 0  # 连续编号，跳过空 mask 的 instance
        for bbox in bboxes:
            inst_id += 1
            box = np.array(bbox, dtype=np.float32)
            try:
                masks, scores, _ = predictor.predict(box=box, multimask_output=False)
                mask_full = masks[0].astype(np.uint16)  # full image size
            except Exception:
                # fallback: bbox 矩形（全图尺寸，只填 bbox 区域）
                mask_full = np.zeros((h, w), dtype=np.uint16)
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                mask_full[y1:y2, x1:x2] = 1

            ys, xs = np.where(mask_full > 0)
            if len(ys) > 0:
                instance_map[mask_full > 0] = inst_id
                instances.append({
                    'bbox': bbox,
                    'area': int(len(ys)),
                })
            else:
                inst_id -= 1  # 回退编号，保持连续
            del mask_full

        if not instances:
            if i % 10 == 0:
                print(f" 0 instances, skip")
            continue

        # 保存
        cv2.imwrite(str(img_out), img_rgb)
        cv2.imwrite(str(mask_out), instance_map.astype(np.uint16))

        manifest.append({
            'image': str(img_out),
            'mask': str(mask_out),
            'n_instances': len(instances),
            'instances': instances,
        })

        if i % 10 == 0:
            print(f" {len(instances)} instances (pseudo mask)")

    print(f"  Processed {len(manifest)}/{len(tiff_files)} images")

    with open(dst_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=lambda x: int(x) if isinstance(x, (np.integer,)) else str(x))

    return manifest


def main():
    parser = argparse.ArgumentParser(description='Prepare SAM2 pseudo-label data (self-distillation)')
    parser.add_argument('--src', required=True, help='MultiOrg train directory')
    parser.add_argument('--dst', required=True, help='Output directory')
    parser.add_argument('--sam2-checkpoint', default='sam2_checkpoints/sam2_hiera_small.pt')
    parser.add_argument('--annotator', default='Annotator_A')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"SAM2 Pseudo-Label Data Preparation (Self-Distillation)")
    print(f"{'='*60}")
    print(f"  Source: {args.src}")
    print(f"  Output: {args.dst}")
    print(f"  SAM2: {args.sam2_checkpoint}")
    print(f"  Annotator: {args.annotator}")

    # 加载 SAM2
    print(f"\nLoading SAM2 (zero-shot)...")
    predictor = load_sam2_predictor(args.sam2_checkpoint, args.device)
    print(f"  SAM2 loaded")

    # 清理旧目录
    import shutil
    for subdir in ['all', 'train', 'val']:
        old = Path(args.dst) / subdir
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)

    # 处理所有图片
    all_manifest = process_split(args.src, args.dst, predictor, args.annotator, 'all', args.max_images, args.device)

    # 随机划分 train/val
    random.seed(42)
    random.shuffle(all_manifest)
    n_val = max(1, int(len(all_manifest) * args.val_ratio))
    val_manifest = all_manifest[:n_val]
    train_manifest = all_manifest[n_val:]

    # 移动文件
    for split, manifest in [('train', train_manifest), ('val', val_manifest)]:
        split_dir = Path(args.dst) / split
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'masks').mkdir(parents=True, exist_ok=True)
        for item in manifest:
            src_img = Path(item['image'])
            src_mask = Path(item['mask'])
            dst_img = split_dir / 'images' / src_img.name
            dst_mask = split_dir / 'masks' / src_mask.name
            if dst_img.exists():
                dst_img.unlink()
            if dst_mask.exists():
                dst_mask.unlink()
            shutil.move(str(src_img), str(dst_img))
            shutil.move(str(src_mask), str(dst_mask))
            item['image'] = str(dst_img)
            item['mask'] = str(dst_mask)

        with open(split_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2, default=lambda x: int(x) if isinstance(x, (np.integer,)) else str(x))

        print(f"  {split}: {len(manifest)} images, {sum(m['n_instances'] for m in manifest)} instances")

    shutil.rmtree(Path(args.dst) / 'all', ignore_errors=True)

    print(f"\nDone! Pseudo-label data ready at {args.dst}")
    print(f"  Train: {len(train_manifest)} images")
    print(f"  Val: {len(val_manifest)} images")


if __name__ == '__main__':
    main()
