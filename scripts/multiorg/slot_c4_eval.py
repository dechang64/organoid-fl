"""
Phase 11 C4: Slot encoder → 检测 mAP 闭环评估

方案 7.6: "C4 关键验证：对比训练的 slot encoder 能否提升检测 mAP"

流程：
1. 加载 SupCon 训练的 slot model（best.pt）
2. 对 metadata 中每个 crop：DINOv2 → Slot Attention → slot score
3. 用 slot score 调整 RF-DETR confidence（多种策略）
4. 用 sklearn average_precision_score 计算 PR-AUC（=AP50 近似）
5. 搜索最优 α 和 threshold

注意：真正的 mAP 需要 GT 标注框（annotations.json），这里用 metadata 的
matched 标签（IoU≥0.5 = TP）做 PR-AUC 近似。如果有 GT JSON 可后续补充。

Usage:
    python slot_c4_eval.py --checkpoint results/supcon_xxx/best.pt \
        --metadata data/ctm_crops/ctm_metadata.json \
        --crops-dir data/ctm_crops \
        --device cuda:0
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slot_supcon import SlotSupConModel

try:
    from ctm.ctm_dataset import OrganoidCTMDataset
    from torch.utils.data import DataLoader
except ImportError:
    OrganoidCTMDataset = None
    DataLoader = None


def compute_ap(labels, scores):
    """PR-AUC = Average Precision. Uses sklearn, not self-written."""
    if len(set(labels)) < 2:
        return 0.0
    return average_precision_score(labels, scores)


def compute_pr_auc(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def load_model(checkpoint_path, device, config=None):
    """Load SupCon model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if config is None:
        config = ckpt.get('config', {})

    model = SlotSupConModel(
        num_slots=config.get('num_slots', 8),
        dim_slots=config.get('dim_slots', 128),
        num_iters=config.get('num_iters', 3),
        n_classes=2,
        img_size=config.get('img_size', 224),
        proj_dim=config.get('proj_dim', 256),
    )

    # Load trainable params only (DINOv2 is frozen, reconstructed by timm)
    if 'model_state_dict' in ckpt:
        trainable_sd = ckpt['model_state_dict']
    elif 'trainable_state_dict' in ckpt:
        trainable_sd = ckpt['trainable_state_dict']
    else:
        trainable_sd = ckpt

    # Manual match: only load keys that exist in both model and checkpoint
    # (backbone keys are in model but not in checkpoint by design)
    model_sd = model.state_dict()
    loaded, skipped = 0, []
    for k in trainable_sd:
        if k in model_sd and trainable_sd[k].shape == model_sd[k].shape:
            model_sd[k] = trainable_sd[k]
            loaded += 1
        else:
            skipped.append(k)
    model.load_state_dict(model_sd)
    print(f"Loaded {loaded}/{len(trainable_sd)} checkpoint params")
    if skipped:
        print(f"Skipped {len(skipped)} (shape mismatch or not in model): {skipped[:3]}...")

    model.to(device)
    model.eval()
    return model


def extract_slot_scores(model, metadata_path, crops_dir, device,
                        img_size=224, batch_size=32, num_workers=4):
    """
    Run model on all crops in metadata, return slot scores + confs + labels.

    Returns:
        scores: np.array [N] - slot classifier probability of TP
        confs: np.array [N] - RF-DETR confidence
        labels: np.array [N] - 1 if matched (TP), 0 if FP
        image_names: list [N] - image name for each crop
        bboxes: list [N] - bbox for each crop
    """
    if OrganoidCTMDataset is None:
        raise ImportError("Cannot import OrganoidCTMDataset from ctm/ctm_dataset.py")

    # Use test split
    dataset = OrganoidCTMDataset(
        metadata_path, crops_dir, 'test',
        img_size=img_size, augment=False, balance=False, seed=42,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)

    all_scores = []
    all_confs = []
    all_labels = []
    all_image_names = []
    all_bboxes = []

    # Build index mapping: dataset index -> metadata entry
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    # dataset.indices contains indices into all_dets
    # We need to map back to metadata entries
    # OrganoidCTMDataset builds all_dets from metadata in order
    # so dataset.indices[i] -> metadata[dataset.indices[i]]

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch['image'].to(device)
            logits, _, _ = model(images, return_embeddings=False)
            probs = F.softmax(logits, dim=-1)[:, 1]  # P(TP)

            start = batch_idx * batch_size
            end = start + len(images)

            for i in range(len(images)):
                idx = dataset.indices[start + i]
                entry = meta_data[idx]
                all_scores.append(probs[i].item())
                all_confs.append(entry.get('rfdetr_conf',
                                          entry.get('confidence', 0.5)))
                all_labels.append(1 if entry.get('matched', False) else 0)
                all_image_names.append(entry.get('image', ''))
                all_bboxes.append(entry.get('bbox', [0, 0, 0, 0]))

    all_scores = np.array(all_scores)
    all_confs = np.array(all_confs)
    all_labels = np.array(all_labels)

    return all_scores, all_confs, all_labels, all_image_names, all_bboxes


def evaluate_strategies(scores, confs, labels):
    """
    Evaluate different confidence adjustment strategies.

    Strategies:
    1. Baseline: conf only
    2. Hard filter: conf * (score >= threshold)
    3. Soft penalize: conf * (1 - alpha * (1 - score))
    4. Geometric mean: conf^0.8 * score^0.2
    5. LR meta: logistic regression on [score, conf]
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    results = {}

    # 1. Baseline
    base_ap = compute_ap(labels, confs)
    results['baseline'] = {'ap': base_ap, 'strategy': 'conf_only'}

    # 2. Hard filter: sweep threshold
    best_thr, best_ap_filter = None, 0
    for thr in np.arange(0.1, 0.9, 0.05):
        filtered = confs * (scores >= thr)
        ap = compute_ap(labels, filtered)
        if ap > best_ap_filter:
            best_ap_filter = ap
            best_thr = thr
    results['hard_filter'] = {'ap': best_ap_filter, 'threshold': best_thr,
                               'delta': best_ap_filter - base_ap}

    # 3. Soft penalize: sweep alpha
    best_alpha, best_ap_soft = None, 0
    for alpha in np.arange(0.0, 1.01, 0.1):
        adjusted = confs * (1 - alpha * (1 - scores))
        ap = compute_ap(labels, adjusted)
        if ap > best_ap_soft:
            best_ap_soft = ap
            best_alpha = alpha
    results['soft_penalize'] = {'ap': best_ap_soft, 'alpha': best_alpha,
                                 'delta': best_ap_soft - base_ap}

    # 4. Geometric mean: sweep weight
    best_w, best_ap_geom = None, 0
    for w in np.arange(0.0, 0.51, 0.05):
        adjusted = confs ** (1 - w) * np.clip(scores, 1e-6, 1) ** w
        ap = compute_ap(labels, adjusted)
        if ap > best_ap_geom:
            best_ap_geom = ap
            best_w = w
    results['geometric_mean'] = {'ap': best_ap_geom, 'w_slot': best_w,
                                  'delta': best_ap_geom - base_ap}

    # 5. LR meta (5-fold CV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_features = np.column_stack([scores, confs])
    meta_probs = np.zeros(len(labels))
    for train_idx, test_idx in skf.split(meta_features, labels):
        lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        lr.fit(meta_features[train_idx], labels[train_idx])
        meta_probs[test_idx] = lr.predict_proba(meta_features[test_idx])[:, 1]
    meta_ap = compute_ap(labels, meta_probs)
    results['lr_meta'] = {'ap': meta_ap, 'delta': meta_ap - base_ap}

    # Find best
    best_strategy = max(results.values(), key=lambda x: x.get('ap', 0))
    results['best'] = {
        'strategy': best_strategy['strategy'] if 'strategy' in best_strategy else
                    max(results.keys(), key=lambda k: results[k].get('ap', 0)),
        'ap': best_strategy.get('ap', 0),
        'delta': best_strategy.get('ap', 0) - base_ap,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 11 C4: Slot → detection AP eval')
    parser.add_argument('--checkpoint', required=True, help='Path to best.pt')
    parser.add_argument('--metadata', required=True, help='ctm_metadata.json')
    parser.add_argument('--crops-dir', required=True, help='Crops directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join('results',
                                       f'c4_eval_{Path(args.checkpoint).parent.name}')

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")

    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, device)

    # Extract slot scores
    print("\nExtracting slot scores...")
    scores, confs, labels, image_names, bboxes = extract_slot_scores(
        model, args.metadata, args.crops_dir, device,
        img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    n_tp = int(labels.sum())
    n_fp = len(labels) - n_tp
    print(f"Crops: {len(scores)} (TP={n_tp}, FP={n_fp})")

    # Individual AUC
    slot_auc = roc_auc_score(labels, scores)
    conf_auc = roc_auc_score(labels, confs)
    slot_ap = compute_ap(labels, scores)
    conf_ap = compute_ap(labels, confs)
    print(f"\nSlot:  AUC={slot_auc:.4f}  AP={slot_ap:.4f}")
    print(f"Conf:  AUC={conf_auc:.4f}  AP={conf_ap:.4f}")

    # Evaluate strategies
    print("\n" + "=" * 60)
    print("Strategy evaluation (PR-AUC = AP50)")
    print("=" * 60)

    results = evaluate_strategies(scores, confs, labels)

    print(f"\n  Baseline (conf only):     AP={results['baseline']['ap']:.4f}")
    print(f"  Hard filter:              AP={results['hard_filter']['ap']:.4f}  "
          f"(thr={results['hard_filter']['threshold']:.2f}, "
          f"Δ={results['hard_filter']['delta']:+.4f})")
    print(f"  Soft penalize:            AP={results['soft_penalize']['ap']:.4f}  "
          f"(α={results['soft_penalize']['alpha']:.1f}, "
          f"Δ={results['soft_penalize']['delta']:+.4f})")
    print(f"  Geometric mean:           AP={results['geometric_mean']['ap']:.4f}  "
          f"(w={results['geometric_mean']['w_slot']:.2f}, "
          f"Δ={results['geometric_mean']['delta']:+.4f})")
    print(f"  LR meta (5-fold CV):      AP={results['lr_meta']['ap']:.4f}  "
          f"(Δ={results['lr_meta']['delta']:+.4f})")

    best = results['best']
    print(f"\n  Best: {best['strategy']}  AP={best['ap']:.4f}  "
          f"(Δ={best['delta']:+.4f})")

    # Per-image breakdown (top 5 images with most crops)
    print("\n" + "=" * 60)
    print("Per-image breakdown (top 5 by crop count)")
    print("=" * 60)
    from collections import defaultdict
    per_image = defaultdict(lambda: {'scores': [], 'confs': [], 'labels': []})
    for i, img in enumerate(image_names):
        per_image[img]['scores'].append(scores[i])
        per_image[img]['confs'].append(confs[i])
        per_image[img]['labels'].append(labels[i])

    sorted_images = sorted(per_image.items(), key=lambda x: len(x[1]['labels']), reverse=True)
    print(f"{'Image':<35} {'N':>5} {'TP':>4} {'FP':>4} {'Slot AUC':>9} {'Conf AUC':>9}")
    print("-" * 75)
    for img, data in sorted_images[:5]:
        n = len(data['labels'])
        tp = sum(data['labels'])
        fp = n - tp
        l = np.array(data['labels'])
        s = np.array(data['scores'])
        c = np.array(data['confs'])
        sa = roc_auc_score(l, s) if len(set(l)) > 1 else 0.5
        ca = roc_auc_score(l, c) if len(set(l)) > 1 else 0.5
        print(f"{img:<35} {n:5d} {tp:4d} {fp:4d} {sa:9.4f} {ca:9.4f}")

    # Save
    output = {
        'checkpoint': args.checkpoint,
        'n_crops': len(scores),
        'n_tp': n_tp,
        'n_fp': n_fp,
        'slot_auc': float(slot_auc),
        'conf_auc': float(conf_auc),
        'slot_ap': float(slot_ap),
        'conf_ap': float(conf_ap),
        'strategies': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating))
                           else vv for kk, vv in v.items()}
                       for k, v in results.items()},
        'n_images': len(per_image),
    }
    with open(os.path.join(args.output_dir, 'c4_results.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/c4_results.json")


if __name__ == '__main__':
    main()
