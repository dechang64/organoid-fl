r"""
MultiOrg SAM2 微调：多边形 GT → mask → mask_decoder finetune

策略：
- 冻结 image_encoder + memory_attention + memory_encoder（大模型部分）
- 只解冻 sam_mask_decoder + sam_prompt_encoder（轻量部分）
- Box prompt（从 GT mask 提取 bbox）→ SAM2 → mask → BCE/Dice loss vs GT mask
- 3060 12GB 可行（image_encoder no_grad，只 mask_decoder backward）

数据流：
1. napari 多边形 [[y1,x1],...,[y4,x4]] → cv2.fillPoly → binary mask
2. 每个 organoid 一个 mask instance → 提取 bbox 作为 box prompt
3. 随机划分 train/val（按 image 分，不按 instance 分）

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl

    # Step 1: 准备数据（多边形 → mask）
    python scripts\multiorg\prepare_sam2_data.py ^
        --src D:\datasets\mutliorg\MultiOrg_v2\train ^
        --dst data\multiorg_sam2 ^
        --annotator Annotator_A

    # Step 2: 微调 SAM2
    python scripts\multiorg\finetune_sam2.py ^
        --data data\multiorg_sam2 ^
        --checkpoint sam2_checkpoints\sam2_hiera_small.pt ^
        --dst runs\sam2_finetune ^
        --epochs 5 --lr 1e-5 --batch-size 4

    # Step 3: 用微调后的 SAM2 跑形态学过滤
    python scripts\multiorg\multiorg_sam2.py ^
        --weights output\checkpoint_best_regular.pth ^
        --sam2-checkpoint runs\sam2_finetune\sam2_finetuned.pt ^
        --finetuned-sam2 ...
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

def napari_polygon_to_mask(polygons, height, width):
    """napari 多边形 [[y1,x1],...,[y4,x4]] → binary mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        # napari (row, col) = (y, x)，cv2 需要 (x, y)
        pts = np.array([[int(x), int(y)] for y, x in poly], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def load_annotations(json_path, annotator_key=None):
    """加载 napari JSON 标注
    
    Args:
        json_path: 标注文件路径
        annotator_key: 指定标注者（如 'Annotator_A'），None 则取第一个
    Returns:
        list of polygons, each polygon = [[y1,x1],...,[y4,x4]]
    """
    with open(json_path) as f:
        data = json.load(f)
    
    polygons = []
    for key, shape in data.items():
        if isinstance(shape, dict) and 'vertices' in shape:
            verts = shape['vertices']
            # napari (row, col) = (y, x)
            polygons.append(verts)
        elif isinstance(shape, list):
            # 直接就是顶点列表
            polygons.append(shape)
    return polygons


def mask_to_instances(mask):
    """binary mask → instance masks (每个连通域一个)"""
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    instances = []
    for i in range(1, num_labels):  # 0 是背景
        inst_mask = (labels == i).astype(np.uint8)
        ys, xs = np.where(inst_mask > 0)
        if len(ys) < 5:  # 太小的跳过
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        instances.append({
            'mask': inst_mask,
            'bbox': [x1, y1, x2, y2],
            'area': len(ys),
        })
    return instances


def process_split(src_dir, dst_dir, annotator='Annotator_A', split_name='train', max_images=None):
    """处理一个 split（train 或 test）"""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir) / split_name
    (dst_path / 'images').mkdir(parents=True, exist_ok=True)
    (dst_path / 'masks').mkdir(parents=True, exist_ok=True)
    
    # 收集所有图片
    tiff_files = sorted(src_path.rglob('*.tiff'))
    if max_images:
        tiff_files = tiff_files[:max_images]
    
    print(f"  Found {len(tiff_files)} TIFF files to process")
    
    manifest = []
    for i, img_path in enumerate(tiff_files):
        # 先检查是否已处理（在加载图片之前！）
        img_stem = img_path.stem
        out_name = f"{img_path.parent.parent.name}_{img_path.parent.name}_{img_stem}"
        img_out = dst_path / 'images' / f'{out_name}.png'
        mask_out = dst_path / 'masks' / f'{out_name}.png'

        if img_out.exists() and mask_out.exists():
            manifest.append({
                'image': str(img_out),
                'mask': str(mask_out),
                'n_instances': 0,
                'instances': [],
            })
            if i % 50 == 0:
                print(f"  [{i+1}/{len(tiff_files)}] cached")
            continue

        if i % 10 == 0:
            print(f"  [{i+1}/{len(tiff_files)}] {img_path.name} ...", end='', flush=True)
        # 找对应标注
        json_name = f"{img_stem}_{annotator}.json"
        json_path = img_path.parent / json_name
        
        if not json_path.exists():
            # 尝试其他标注者
            for alt in ['Annotator_A', 'Annotator_B', 'Annotator_C']:
                alt_path = img_path.parent / f"{img_stem}_{alt}.json"
                if alt_path.exists():
                    json_path = alt_path
                    break
            else:
                if i % 10 == 0:
                    print(f" no annotation, skip")
                continue
        else:
            if i % 10 == 0:
                print(f" processing...", end='', flush=True)
        
        # 加载图片
        img = Image.open(img_path)
        # 16-bit → 8-bit
        if img.mode == 'I;16':
            arr = np.array(img)
            vmin, vmax = arr.min(), arr.max()
            if vmax > vmin:
                arr = ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            img8 = np.stack([arr, arr, arr], axis=-1)  # RGB
        else:
            img8 = np.array(img.convert('RGB'))
        
        h, w = img8.shape[:2]
        
        # 加载标注 → mask
        polygons = load_annotations(json_path)
        if not polygons:
            continue
        
        mask = napari_polygon_to_mask(polygons, h, w)
        instances = mask_to_instances(mask)
        
        if not instances:
            if i % 10 == 0:
                print(f" 0 instances, skip")
            continue
        
        # 保存（out_name/img_out/mask_out 已在循环开头定义）
        cv2.imwrite(str(img_out), img8)
        # mask 保存为单通道，每个 instance 一个文件
        # 实际上我们保存 instance map（每个像素 = instance id）
        instance_map = np.zeros((h, w), dtype=np.uint16)
        for i, inst in enumerate(instances):
            instance_map[inst['mask'] > 0] = i + 1
        
        cv2.imwrite(str(mask_out), instance_map.astype(np.uint16))
        
        manifest.append({
            'image': str(img_out),
            'mask': str(mask_out),
            'n_instances': len(instances),
            'instances': [{'bbox': inst['bbox'], 'area': inst['area']} for inst in instances],
        })
        
        if i % 10 == 0:
            print(f" {len(instances)} instances OK")
    
    print(f"  Processed {len(manifest)}/{len(tiff_files)} images")
    
    # 保存 manifest
    with open(dst_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=lambda x: int(x) if isinstance(x, (np.integer,)) else str(x))
    
    print(f"  {split_name}: {len(manifest)} images, {sum(m['n_instances'] for m in manifest)} instances")
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Prepare MultiOrg data for SAM2 finetune')
    parser.add_argument('--src', required=True, help='MultiOrg train or test directory')
    parser.add_argument('--dst', required=True, help='Output directory')
    parser.add_argument('--annotator', default='Annotator_A', help='Annotator key')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation split ratio')
    args = parser.parse_args()
    
    print(f"Source: {args.src}")
    print(f"Output: {args.dst}")
    print(f"Annotator: {args.annotator}")
    print(f"Val ratio: {args.val_ratio}")
    print("=" * 60)
    
    # 处理所有图片
    # 清理旧目录（避免之前部分移动导致文件不一致）
    import shutil
    for subdir in ['all', 'train', 'val']:
        old = Path(args.dst) / subdir
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)

    all_manifest = process_split(args.src, args.dst, args.annotator, 'all', args.max_images)
    
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
        import shutil
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
    
    # 删除 all 目录
    import shutil
    shutil.rmtree(Path(args.dst) / 'all', ignore_errors=True)
    
    print(f"\nDone! Data ready at {args.dst}")
    print(f"  Train: {len(train_manifest)} images")
    print(f"  Val: {len(val_manifest)} images")


if __name__ == '__main__':
    main()
