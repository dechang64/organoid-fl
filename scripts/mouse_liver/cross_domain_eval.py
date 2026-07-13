"""
跨域 Slot 评估：用 MultiOrg best.pt 直接评估鼠肝 crops

和 slot_c4_eval.py 的区别：
- 不用 OrganoidCTMDataset（避免 train/val/test split）
- 直接加载全部 crops 做推理
- 输出和 MultiOrg C4 相同格式的结果

Usage (冬生本地):
    # 先跑 generate_mouse_crops.py 生成 crops + metadata
    # 然后用 MultiOrg best.pt 评估：
    
    python scripts/mouse_liver/cross_domain_eval.py \\
        --checkpoint results/supcon_8s_d128_p256_t0.07_b0.1_20260713_003826/best.pt \\
        --metadata data/mouse_crops/b1/crop_metadata.json \\
        --crops-dir data/mouse_crops/b1/crops \\
        --device cuda:0 \\
        --tag b1

    # 对 B2/B3 重复（换 --metadata / --crops-dir / --tag）
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'multiorg'))
from slot_supcon import SlotSupConModel


def compute_pr_auc(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


class SimpleCropDataset(Dataset):
    """直接从 metadata + crops 目录加载，不做 train/val/test split。"""
    def __init__(self, metadata_path, crops_dir, img_size=224):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.dets = json.load(f)
        self.crops_dir = crops_dir
        self.img_size = img_size
        print(f"[Dataset] {len(self.dets)} crops from metadata")

    def __len__(self):
        return len(self.dets)

    def __getitem__(self, idx):
        det = self.dets[idx]
        # Try crop_path first, then fallback to cache_key.png
        crop_path = det.get('crop_path', '')
        if not crop_path or not os.path.exists(crop_path):
            # Fallback: crops_dir / cache_key.png
            crop_path = os.path.join(self.crops_dir, det['cache_key'] + '.png')
        
        img = Image.open(crop_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        
        # Normalize (DINOv2 expects ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        
        return {
            'image': img,
            'label': 1 if det.get('matched', False) else 0,
            'confidence': det.get('rfdetr_conf', det.get('confidence', 0.5)),
            'image_name': det.get('image', ''),
            'bbox': det.get('bbox', [0, 0, 0, 0]),
            'det_idx': det.get('det_idx', 0),
        }


def load_model(checkpoint_path, device):
    """Load SupCon model from MultiOrg checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    
    model = SlotSupConModel(
        num_slots=config.get('num_slots', 8),
        dim_slots=config.get('dim_slots', 128),
        num_iters=config.get('num_iters', 3),
        n_classes=2,
        img_size=config.get('img_size', 224),
        proj_dim=config.get('proj_dim', 256),
    )
    
    # Load trainable params (backbone reconstructed by timm)
    if 'model_state_dict' in ckpt:
        trainable_sd = ckpt['model_state_dict']
    elif 'trainable_state_dict' in ckpt:
        trainable_sd = ckpt['trainable_state_dict']
    else:
        trainable_sd = ckpt
    
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


def evaluate_strategies(scores, confs, labels):
    """4 种 slot→conf 过滤策略。"""
    results = {}
    base_ap = average_precision_score(labels, confs)
    results['baseline'] = {'ap': base_ap, 'delta': 0.0}
    
    # 1. Hard filter: drop if slot < threshold
    best_ap, best_thr = base_ap, 0.0
    for thr in np.arange(0.05, 0.95, 0.05):
        filtered = confs.copy()
        filtered[scores < thr] = 0
        ap = average_precision_score(labels, filtered)
        if ap > best_ap:
            best_ap, best_thr = ap, thr
    results['hard_filter'] = {'ap': best_ap, 'delta': best_ap - base_ap, 'threshold': float(best_thr)}
    
    # 2. Soft penalize: conf *= (1 - α*(1-slot))
    best_ap, best_alpha = base_ap, 0.0
    for alpha in np.arange(0.1, 1.0, 0.1):
        filtered = confs * (1 - alpha * (1 - scores))
        ap = average_precision_score(labels, filtered)
        if ap > best_ap:
            best_ap, best_alpha = ap, alpha
    results['soft_penalize'] = {'ap': best_ap, 'delta': best_ap - base_ap, 'alpha': float(best_alpha)}
    
    # 3. Geometric mean: conf^w * slot^(1-w)
    best_ap, best_w = base_ap, 0.0
    for w in np.arange(0.1, 1.0, 0.1):
        eps = 1e-8
        filtered = np.power(confs + eps, w) * np.power(scores + eps, 1 - w)
        ap = average_precision_score(labels, filtered)
        if ap > best_ap:
            best_ap, best_w = ap, w
    results['geometric_mean'] = {'ap': best_ap, 'delta': best_ap - base_ap, 'w_slot': float(1 - best_w)}
    
    # 4. LR meta (5-fold)
    meta_features = np.column_stack([scores, confs])
    skf = StratifiedKFold(n_splits=min(5, len(set(labels))), shuffle=True, random_state=42)
    meta_probs = np.zeros(len(labels))
    for train_idx, test_idx in skf.split(meta_features, labels):
        lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        lr.fit(meta_features[train_idx], labels[train_idx])
        meta_probs[test_idx] = lr.predict_proba(meta_features[test_idx])[:, 1]
    lr_ap = average_precision_score(labels, meta_probs)
    results['lr_meta'] = {'ap': lr_ap, 'delta': lr_ap - base_ap}
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Cross-domain Slot Evaluation (MultiOrg → Mouse Liver)')
    parser.add_argument('--checkpoint', required=True, help='MultiOrg SupCon best.pt')
    parser.add_argument('--metadata', required=True, help='Mouse liver crop_metadata.json')
    parser.add_argument('--crops-dir', required=True, help='Mouse liver crops directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tag', default='mouse_liver', help='Tag for output dir')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.output_dir is None:
        args.output_dir = f'results/cross_domain_{args.tag}'
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model (trained on MultiOrg)
    print(f"\n{'='*60}")
    print(f"Cross-domain eval: MultiOrg model → {args.tag}")
    print(f"{'='*60}")
    model = load_model(args.checkpoint, device)
    
    # Load mouse liver crops
    dataset = SimpleCropDataset(args.metadata, args.crops_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    
    # Run inference
    all_scores = []
    all_confs = []
    all_labels = []
    all_embeddings = []
    all_image_names = []
    
    print(f"\nRunning inference on {len(dataset)} crops...")
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            logits, _, embeddings = model(images, return_embeddings=True)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            
            all_scores.extend(probs.cpu().numpy())
            all_confs.extend(batch['confidence'].numpy())
            all_labels.extend(batch['label'].numpy())
            all_embeddings.append(embeddings.cpu().numpy())
            all_image_names.extend(batch['image_name'])
    
    scores = np.array(all_scores)
    confs = np.array(all_confs)
    labels = np.array(all_labels)
    embeddings = np.concatenate(all_embeddings)
    
    n_tp = labels.sum()
    n_fp = len(labels) - n_tp
    
    print(f"\nResults for {args.tag}:")
    print(f"  Total crops: {len(labels)}")
    print(f"  TP: {n_tp}, FP: {n_fp}")
    print(f"  TP rate: {n_tp/len(labels)*100:.1f}%")
    
    # AUC
    slot_auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5
    conf_auc = roc_auc_score(labels, confs) if len(set(labels)) > 1 else 0.5
    slot_ap = average_precision_score(labels, scores)
    conf_ap = average_precision_score(labels, confs)
    
    print(f"\n  Slot AUC:  {slot_auc:.4f}  AP: {slot_ap:.4f}")
    print(f"  Conf AUC:  {conf_auc:.4f}  AP: {conf_ap:.4f}")
    print(f"  Δ AUC:     {slot_auc - conf_auc:+.4f}")
    print(f"  Δ AP:      {slot_ap - conf_ap:+.4f}")
    
    # Strategies
    results = evaluate_strategies(scores, confs, labels)
    
    print(f"\n{'Strategy':<20} {'AP':>8} {'Δ AP':>8} {'Param':>10}")
    print("-" * 50)
    for name, r in results.items():
        param = ''
        if 'threshold' in r: param = f"thr={r['threshold']:.2f}"
        elif 'alpha' in r: param = f"α={r['alpha']:.1f}"
        elif 'w_slot' in r: param = f"w={r['w_slot']:.2f}"
        elif name == 'lr_meta': param = '5-fold'
        print(f"{name:<20} {r['ap']:8.4f} {r['delta']:+8.4f} {param:>10}")
    
    # Per-image breakdown
    print(f"\nPer-image breakdown:")
    print(f"{'Image':<25} {'N':>5} {'TP':>4} {'FP':>4} {'Slot AUC':>9} {'Conf AUC':>9}")
    print("-" * 65)
    per_image = {}
    for i, img in enumerate(all_image_names):
        if img not in per_image:
            per_image[img] = {'scores': [], 'confs': [], 'labels': []}
        per_image[img]['scores'].append(scores[i])
        per_image[img]['confs'].append(confs[i])
        per_image[img]['labels'].append(labels[i])
    
    for img in sorted(per_image.keys()):
        d = per_image[img]
        n = len(d['labels'])
        tp = sum(d['labels'])
        fp = n - tp
        s = np.array(d['scores'])
        c = np.array(d['confs'])
        l = np.array(d['labels'])
        sa = roc_auc_score(l, s) if len(set(l)) > 1 else 0.5
        ca = roc_auc_score(l, c) if len(set(l)) > 1 else 0.5
        print(f"{img:<25} {n:5d} {tp:4d} {fp:4d} {sa:9.4f} {ca:9.4f}")
    
    # Save embeddings for Phase 10 federated
    np.save(os.path.join(args.output_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(args.output_dir, 'labels.npy'), labels)
    np.save(os.path.join(args.output_dir, 'confs.npy'), confs)
    
    # Save results
    output = {
        'tag': args.tag,
        'checkpoint': args.checkpoint,
        'n_crops': len(labels),
        'n_tp': int(n_tp),
        'n_fp': int(n_fp),
        'slot_auc': float(slot_auc),
        'conf_auc': float(conf_auc),
        'slot_ap': float(slot_ap),
        'conf_ap': float(conf_ap),
        'strategies': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating))
                           else vv for kk, vv in v.items()}
                       for k, v in results.items()},
    }
    with open(os.path.join(args.output_dir, 'cross_domain_results.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
