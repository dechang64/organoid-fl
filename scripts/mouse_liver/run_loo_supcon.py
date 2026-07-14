r"""
留一法跨域评估：真正的跨域泛化测试

三个数据集轮流留一个做测试集，用另外两个训练：
- L1: Train on MultiOrg + Intestinal → Test on Mouse Liver (B1+B2+B3)
- L2: Train on MultiOrg + Mouse Liver → Test on Intestinal
- L3: Train on Mouse Liver + Intestinal → Test on MultiOrg

这样测试集完全不参与训练，才是真正的跨域泛化。

Usage (冬生本地 GPU):
    cd C:\\Users\\decha\\organoid-fl
    python scripts\\mouse_liver\\run_loo_supcon.py --device cuda:0

Usage (云 VM CPU):
    cd /home/z/my-project/organoid-fl
    python3 scripts/mouse_liver/run_loo_supcon.py --device cpu --epochs 20
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(str(PROJECT_ROOT))

# Dataset configs
DATASETS = {
    'multiorg': {
        'metadata': 'results/phase2_vlm_100/vlm_results.json',
        'crops_dir': 'results/phase2_vlm_100/crops',
        'name_prefix': 'multiorg',
    },
    'mouse_b1': {
        'metadata': 'data/mouse_crops/b1/crop_metadata.json',
        'crops_dir': 'data/mouse_crops/b1/crops',
        'name_prefix': 'mouse_b1',
    },
    'mouse_b2': {
        'metadata': 'data/mouse_crops/b2/crop_metadata.json',
        'crops_dir': 'data/mouse_crops/b2/crops',
        'name_prefix': 'mouse_b2',
    },
    'mouse_b3': {
        'metadata': 'data/mouse_crops/b3/crop_metadata.json',
        'crops_dir': 'data/mouse_crops/b3/crops',
        'name_prefix': 'mouse_b3',
    },
    'intestinal': {
        'metadata': 'data/intestinal_crops/val/crop_metadata.json',
        'crops_dir': 'data/intestinal_crops/val/crops',
        'name_prefix': 'intestinal',
    },
}

# Leave-one-out configs: (test_dataset, [train_datasets])
LOO_CONFIGS = [
    ('mouse_b1', ['multiorg', 'mouse_b2', 'mouse_b3', 'intestinal']),
    ('mouse_b2', ['multiorg', 'mouse_b1', 'mouse_b3', 'intestinal']),
    ('mouse_b3', ['multiorg', 'mouse_b1', 'mouse_b2', 'intestinal']),
    ('intestinal', ['multiorg', 'mouse_b1', 'mouse_b2', 'mouse_b3']),
]


def load_metadata(meta_path):
    """Load metadata, handle different formats."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Normalize: ensure each entry has 'matched' and 'cache_key'
    normalized = []
    for item in data:
        if not item.get('crop_exists', True):
            continue
        entry = {
            'cache_key': item.get('cache_key', ''),
            'image': item.get('image', ''),
            'det_idx': item.get('det_idx', 0),
            'bbox': item.get('bbox', [0, 0, 0, 0]),
            'rfdetr_conf': item.get('rfdetr_conf', item.get('confidence', 0.5)),
            'matched': item.get('matched', False),
            'match_iou': item.get('match_iou', 0.0),
            'area': 0, 'circularity': 0, 'solidity': 0, 'aspect_ratio': 0,
            'crop_path': item.get('crop_path', ''),
            'image_size': [0, 0],
            'n_gt': 0,
        }
        normalized.append(entry)
    return normalized


def prepare_loo_split(train_datasets, test_dataset, output_dir):
    """Prepare merged metadata + symlink/copy crops for train and test."""
    output_dir = Path(output_dir)
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    train_crops = train_dir / 'crops'
    test_crops = test_dir / 'crops'
    
    for d in [train_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
        (d / 'crops').mkdir(exist_ok=True)
    
    # Merge train metadata
    train_meta = []
    for ds_name in train_datasets:
        ds = DATASETS[ds_name]
        if not os.path.exists(ds['metadata']):
            print(f"  [WARN] {ds_name} metadata not found: {ds['metadata']}")
            continue
        items = load_metadata(ds['metadata'])
        # Copy crops
        for item in items:
            src_crop = os.path.join(ds['crops_dir'], item['cache_key'] + '.png')
            dst_crop = train_crops / (item['cache_key'] + '.png')
            if os.path.exists(src_crop) and not dst_crop.exists():
                try:
                    shutil.copy2(src_crop, dst_crop)
                except Exception as e:
                    print(f"  [WARN] copy failed: {e}")
            # Rename cache_key to avoid collision
            item['cache_key'] = f"{ds['name_prefix']}_{item['cache_key']}"
            # Re-copy with new name
            if os.path.exists(src_crop):
                dst_crop2 = train_crops / (item['cache_key'] + '.png')
                if not dst_crop2.exists():
                    shutil.copy2(src_crop, dst_crop2)
            # Update image name for split logic
            item['image'] = f"{ds['name_prefix']}/{item['image']}"
            train_meta.append(item)
    
    # Test metadata (leave-one-out)
    test_meta = []
    ds = DATASETS[test_dataset]
    if os.path.exists(ds['metadata']):
        items = load_metadata(ds['metadata'])
        for item in items:
            src_crop = os.path.join(ds['crops_dir'], item['cache_key'] + '.png')
            dst_crop = test_crops / (item['cache_key'] + '.png')
            if os.path.exists(src_crop) and not dst_crop.exists():
                try:
                    shutil.copy2(src_crop, dst_crop)
                except:
                    pass
            test_meta.append(item)
    
    # Save metadata
    with open(train_dir / 'crop_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(train_meta, f, indent=2, ensure_ascii=False)
    with open(test_dir / 'crop_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(test_meta, f, indent=2, ensure_ascii=False)
    
    tp_train = sum(1 for x in train_meta if x.get('matched'))
    tp_test = sum(1 for x in test_meta if x.get('matched'))
    print(f"  Train: {len(train_meta)} crops (TP={tp_train}, FP={len(train_meta)-tp_train})")
    print(f"  Test:  {len(test_meta)} crops (TP={tp_test}, FP={len(test_meta)-tp_test})")
    
    return train_dir, test_dir


def train_supcon(metadata_path, crops_dir, output_dir, device, epochs):
    """Train SupCon using slot_supcon.py."""
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'multiorg'))
    import slot_supcon
    
    os.makedirs(output_dir, exist_ok=True)
    
    sys.argv = [
        'slot_supcon.py',
        '--metadata', str(metadata_path),
        '--crops-dir', str(crops_dir),
        '--output-dir', str(output_dir),
        '--num-slots', '8',
        '--dim-slots', '128',
        '--proj-dim', '256',
        '--temperature', '0.07',
        '--supcon-weight', '0.1',
        '--epochs', str(epochs),
        '--batch-size', '32',
        '--device', str(device),
        '--num-workers', '0',
    ]
    
    print(f"  Training SupCon → {output_dir}")
    slot_supcon.main()
    
    ckpt = os.path.join(output_dir, 'best.pt')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def eval_cross_domain(checkpoint, metadata, crops_dir, device, tag):
    """Run cross_domain_eval.py."""
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'mouse_liver'))
    from cross_domain_eval import load_model, SimpleCropDataset
    from sklearn.metrics import roc_auc_score, average_precision_score
    from torch.utils.data import DataLoader
    import torch
    import numpy as np
    
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint, dev)
    dataset = SimpleCropDataset(metadata, crops_dir)
    
    if len(dataset) == 0:
        print(f"  [ERROR] No crops for {tag}")
        return None
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    all_scores, all_confs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(dev)
            logits, _, _ = model(images, return_embeddings=True)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_scores.extend(probs.cpu().numpy())
            all_confs.extend(batch['confidence'].numpy())
            all_labels.extend(batch['label'].numpy())
    
    scores = np.array(all_scores)
    confs = np.array(all_confs)
    labels = np.array(all_labels)
    
    n_tp = labels.sum()
    n_fp = len(labels) - n_tp
    slot_auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5
    conf_auc = roc_auc_score(labels, confs) if len(set(labels)) > 1 else 0.5
    slot_ap = average_precision_score(labels, scores) if len(set(labels)) > 1 else 0
    conf_ap = average_precision_score(labels, confs) if len(set(labels)) > 1 else 0
    
    result = {
        'tag': tag,
        'n_crops': len(labels),
        'n_tp': int(n_tp),
        'n_fp': int(n_fp),
        'slot_auc': float(slot_auc),
        'conf_auc': float(conf_auc),
        'slot_ap': float(slot_ap),
        'conf_ap': float(conf_ap),
    }
    
    print(f"  {tag}: {len(labels)} crops (TP={n_tp}, FP={n_fp})")
    print(f"    Slot AUC: {slot_auc:.4f}  AP: {slot_ap:.4f}")
    print(f"    Conf AUC: {conf_auc:.4f}  AP: {conf_ap:.4f}")
    print(f"    Δ AUC:    {slot_auc - conf_auc:+.4f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Leave-one-out cross-domain SupCon eval')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output-base', default='results/loo_supcon')
    args = parser.parse_args()
    
    t0 = time.time()
    print("=" * 60)
    print("  Leave-One-Out Cross-Domain SupCon Evaluation")
    print("  True cross-domain: test set NOT in training")
    print("=" * 60)
    
    all_results = {}
    
    for test_ds, train_dss in LOO_CONFIGS:
        tag = f"loo_test_{test_ds}"
        print(f"\n{'='*60}")
        print(f"  LOO: train on {train_dss} → test on {test_ds}")
        print(f"{'='*60}")
        
        output_dir = Path(args.output_base) / test_ds
        merged_dir = output_dir / 'merged'
        
        # Prepare split
        print("\n[1/3] Preparing leave-one-out split...")
        train_dir, test_dir = prepare_loo_split(train_dss, test_ds, merged_dir)
        
        # Train
        print("\n[2/3] Training SupCon...")
        ckpt = train_supcon(
            train_dir / 'crop_metadata.json',
            train_dir / 'crops',
            str(output_dir / 'supcon'),
            args.device,
            args.epochs
        )
        
        # Eval on held-out dataset (TRUE cross-domain)
        print("\n[3/3] Evaluating on held-out dataset (true cross-domain)...")
        result = eval_cross_domain(
            ckpt,
            str(test_dir / 'crop_metadata.json'),
            str(test_dir / 'crops'),
            args.device,
            tag
        )
        all_results[test_ds] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("  Leave-One-Out Summary (TRUE cross-domain)")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'Crops':>7} {'Slot AUC':>10} {'Conf AUC':>10} {'Δ AUC':>8}")
    print("-" * 55)
    for ds, r in all_results.items():
        if r:
            print(f"{ds:<15} {r['n_crops']:>7} {r['slot_auc']:>10.4f} {r['conf_auc']:>10.4f} {r['slot_auc']-r['conf_auc']:>+8.4f}")
    
    # Compare with same-domain (§9.5) and single-domain (§9.3)
    print(f"\n{'='*60}")
    print("  Comparison: Single-domain vs Same-domain(merged) vs LOO(true cross)")
    print(f"{'='*60}")
    single = {'mouse_b1': 0.29, 'mouse_b2': 0.51, 'mouse_b3': 0.54, 'intestinal': 0.67}
    merged = {'mouse_b1': 0.74, 'mouse_b2': 0.70, 'mouse_b3': 0.85, 'intestinal': 0.74}
    conf   = {'mouse_b1': 0.91, 'mouse_b2': 0.98, 'mouse_b3': 0.92, 'intestinal': 0.92}
    
    print(f"{'Dataset':<15} {'Single':>8} {'Merged':>8} {'LOO':>8} {'Conf':>8}")
    print("-" * 50)
    for ds in ['mouse_b1', 'mouse_b2', 'mouse_b3', 'intestinal']:
        loo = all_results.get(ds, {}).get('slot_auc', 0) if all_results.get(ds) else 0
        print(f"{ds:<15} {single[ds]:>8.2f} {merged[ds]:>8.2f} {loo:>8.2f} {conf[ds]:>8.2f}")
    
    # Save
    summary_path = Path(args.output_base) / 'loo_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
