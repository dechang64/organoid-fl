r"""
多数据集联合 SupCon 训练：合并 MultiOrg + 鼠肝 + Intestinal crops

目标：验证多数据集联合训练能否学到通用 organoid 原语（解决跨域失效问题）

流程：
1. 合并三个数据集的 crops + metadata 到统一目录
2. 用 slot_supcon.py 训练
3. 分别在每个数据集上评估（同域 + 跨域）

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl

    # Step 1: 合并数据集
    python scripts\mouse_liver\merge_datasets.py

    # Step 2: 训练（GPU）
    python scripts\multiorg\slot_supcon.py --metadata data\merged_crops\crop_metadata.json --crops-dir data\merged_crops\crops --output-dir results\supcon_merged --device cuda:0 --epochs 50

    # Step 3: 跨域评估（每个数据集单独）
    python scripts\mouse_liver\cross_domain_eval.py --checkpoint results\supcon_merged\best.pt --metadata data\mouse_crops\b1\crop_metadata.json --crops-dir data\mouse_crops\b1\crops --device cuda:0 --tag merged_b1
    python scripts\mouse_liver\cross_domain_eval.py --checkpoint results\supcon_merged\best.pt --metadata data\mouse_crops\b2\crop_metadata.json --crops-dir data\mouse_crops\b2\crops --device cuda:0 --tag merged_b2
    python scripts\mouse_liver\cross_domain_eval.py --checkpoint results\supcon_merged\best.pt --metadata data\mouse_crops\b3\crop_metadata.json --crops-dir data\mouse_crops\b3\crops --device cuda:0 --tag merged_b3
    python scripts\mouse_liver\cross_domain_eval.py --checkpoint results\supcon_merged\best.pt --metadata data\intestinal_crops\val\crop_metadata.json --crops-dir data\intestinal_crops\val\crops --device cuda:0 --tag merged_intestinal
"""
import json
import os
import shutil
from pathlib import Path


def merge_dataset(name, metadata_path, crops_dir, dst_crops_dir, prefix):
    """合并单个数据集的 metadata + crops。"""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    merged = []
    for entry in metadata:
        cache_key = entry.get('cache_key', '')
        if not cache_key:
            continue
        
        # Add prefix to avoid name collisions across datasets
        new_cache_key = f"{prefix}_{cache_key}"
        
        # Copy crop file
        src_crop = os.path.join(crops_dir, f"{cache_key}.png")
        dst_crop = os.path.join(dst_crops_dir, f"{new_cache_key}.png")
        
        if os.path.exists(src_crop):
            if not os.path.exists(dst_crop):
                shutil.copy2(src_crop, dst_crop)
        else:
            # Try crop_path from metadata
            alt_path = entry.get('crop_path', '')
            if alt_path and os.path.exists(alt_path):
                if not os.path.exists(dst_crop):
                    shutil.copy2(alt_path, dst_crop)
            else:
                print(f"  [WARN] Crop not found: {src_crop}")
                continue
        
        # Update entry
        entry['cache_key'] = new_cache_key
        entry['crop_path'] = dst_crop
        entry['dataset'] = prefix
        merged.append(entry)
    
    print(f"  {name}: {len(merged)} crops (from {len(metadata)} metadata entries)")
    return merged


def main():
    project_root = Path(__file__).parent.parent.parent
    dst_dir = project_root / 'data' / 'merged_crops'
    dst_crops = dst_dir / 'crops'
    dst_crops.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Merging datasets for joint SupCon training")
    print("=" * 60)
    
    all_metadata = []
    
    # 1. MultiOrg (100 crops)
    print("\n[1/3] MultiOrg")
    multiorg_meta = project_root / 'results' / 'phase2_vlm_100' / 'vlm_results.json'
    multiorg_crops = project_root / 'results' / 'phase2_vlm_100' / 'crops'
    if multiorg_meta.exists():
        all_metadata.extend(
            merge_dataset('MultiOrg', multiorg_meta, multiorg_crops, dst_crops, 'multiorg')
        )
    else:
        print("  [SKIP] MultiOrg metadata not found")
    
    # 2. Mouse Liver (B1 + B2 + B3)
    print("\n[2/3] Mouse Liver")
    for batch in ['b1', 'b2', 'b3']:
        meta_path = project_root / 'data' / 'mouse_crops' / batch / 'crop_metadata.json'
        crops_dir = project_root / 'data' / 'mouse_crops' / batch / 'crops'
        if meta_path.exists():
            all_metadata.extend(
                merge_dataset(f'Mouse {batch}', meta_path, crops_dir, dst_crops, f'mouse_{batch}')
            )
        else:
            print(f"  [SKIP] Mouse {batch} metadata not found")
    
    # 3. Intestinal
    print("\n[3/3] Intestinal")
    intestinal_meta = project_root / 'data' / 'intestinal_crops' / 'val' / 'crop_metadata.json'
    intestinal_crops = project_root / 'data' / 'intestinal_crops' / 'val' / 'crops'
    if intestinal_meta.exists():
        all_metadata.extend(
            merge_dataset('Intestinal', intestinal_meta, intestinal_crops, dst_crops, 'intestinal')
        )
    else:
        print("  [SKIP] Intestinal metadata not found")
    
    # Save merged metadata
    meta_path = dst_dir / 'crop_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    # Stats
    tp = sum(1 for x in all_metadata if x.get('matched'))
    fp = len(all_metadata) - tp
    datasets = {}
    for x in all_metadata:
        ds = x.get('dataset', 'unknown')
        if ds not in datasets:
            datasets[ds] = {'tp': 0, 'fp': 0}
        if x.get('matched'):
            datasets[ds]['tp'] += 1
        else:
            datasets[ds]['fp'] += 1
    
    print(f"\n{'='*60}")
    print(f"Merged dataset summary")
    print(f"{'='*60}")
    print(f"  Total: {len(all_metadata)} crops (TP={tp}, FP={fp})")
    print(f"  Crops dir: {dst_crops}")
    print(f"  Metadata: {meta_path}")
    print(f"\n  Per-dataset breakdown:")
    for ds, counts in sorted(datasets.items()):
        print(f"    {ds}: {counts['tp']} TP + {counts['fp']} FP = {counts['tp']+counts['fp']} total")
    
    print(f"\nNext: train with")
    print(f"  python scripts\\multiorg\\slot_supcon.py \\")
    print(f"    --metadata data\\merged_crops\\crop_metadata.json \\")
    print(f"    --crops-dir data\\merged_crops\\crops \\")
    print(f"    --output-dir results\\supcon_merged \\")
    print(f"    --device cuda:0 --epochs 50")


if __name__ == '__main__':
    main()
