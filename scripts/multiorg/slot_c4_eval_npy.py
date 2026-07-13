"""
Phase 11 C4: Slot→检测 mAP 闭环评估（从 .npy 直接加载，无需 crops 目录）

用已保存的 test_embeddings.npy / test_labels.npy / test_confs.npy
评估 slot 过滤能否提升检测 PR-AUC（=AP50 近似）。

Usage:
    python slot_c4_eval_npy.py \
        --embeddings upload/xxx_test_embeddings.npy \
        --labels upload/xxx_test_labels.npy \
        --confs upload/xxx_test_confs.npy
"""
import argparse
import json
import os
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def compute_pr_auc(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def evaluate_strategies(scores, confs, labels):
    """评估 4 种 slot→conf 过滤策略，搜索最优参数。"""
    results = {}
    base_ap = average_precision_score(labels, confs)
    base_pr = compute_pr_auc(labels, confs)
    results['baseline'] = {'ap': base_ap, 'pr_auc': base_pr}

    # 1. Hard filter: drop if slot_prob < threshold
    best_ap, best_thr = base_ap, 0.0
    for thr in np.arange(0.05, 0.95, 0.05):
        mask = scores >= thr
        if mask.sum() < 10:
            continue
        filtered_confs = confs.copy()
        filtered_confs[~mask] = 0  # 置零 = NMS 掉
        ap = average_precision_score(labels, filtered_confs)
        if ap > best_ap:
            best_ap, best_thr = ap, thr
    results['hard_filter'] = {
        'ap': best_ap, 'pr_auc': compute_pr_auc(labels, confs * (scores >= best_thr)),
        'threshold': best_thr, 'delta': best_ap - base_ap,
    }

    # 2. Soft penalize: conf *= (1 - α*(1 - slot_prob))
    best_ap2, best_alpha = base_ap, 0.0
    for alpha in np.arange(0.1, 1.0, 0.1):
        adjusted = confs * (1 - alpha * (1 - scores))
        ap = average_precision_score(labels, adjusted)
        if ap > best_ap2:
            best_ap2, best_alpha = ap, alpha
    adjusted = confs * (1 - best_alpha * (1 - scores))
    results['soft_penalize'] = {
        'ap': best_ap2, 'pr_auc': compute_pr_auc(labels, adjusted),
        'alpha': best_alpha, 'delta': best_ap2 - base_ap,
    }

    # 3. Geometric mean: conf^w * slot^(1-w)
    best_ap3, best_w = base_ap, 0.0
    eps = 1e-8
    for w in np.arange(0.1, 0.95, 0.05):
        combined = np.power(confs + eps, w) * np.power(scores + eps, 1 - w)
        ap = average_precision_score(labels, combined)
        if ap > best_ap3:
            best_ap3, best_w = ap, w
    combined = np.power(confs + eps, best_w) * np.power(scores + eps, 1 - best_w)
    results['geometric_mean'] = {
        'ap': best_ap3, 'pr_auc': compute_pr_auc(labels, combined),
        'w_slot': best_w, 'delta': best_ap3 - base_ap,
    }

    # 4. LR meta: 5-fold LR on [slot_prob, conf]
    meta_features = np.column_stack([scores, confs])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_probs = np.zeros(len(labels))
    for train_idx, test_idx in skf.split(meta_features, labels):
        lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        lr.fit(meta_features[train_idx], labels[train_idx])
        meta_probs[test_idx] = lr.predict_proba(meta_features[test_idx])[:, 1]
    results['lr_meta'] = {
        'ap': average_precision_score(labels, meta_probs),
        'pr_auc': compute_pr_auc(labels, meta_probs),
        'delta': average_precision_score(labels, meta_probs) - base_ap,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='C4: Slot→detection mAP (from .npy)')
    parser.add_argument('--embeddings', required=True, help='test_embeddings.npy (B,256)')
    parser.add_argument('--labels', required=True, help='test_labels.npy')
    parser.add_argument('--confs', required=True, help='test_confs.npy')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    labels = np.load(args.labels)
    confs = np.load(args.confs)
    print(f"Embeddings: {embeddings.shape}")
    print(f"Labels: {labels.shape} (TP={labels.sum()}, FP={len(labels)-labels.sum()})")
    print(f"Confs: {confs.shape}")

    if args.output_dir is None:
        args.output_dir = 'results/c4_eval_npy'
    os.makedirs(args.output_dir, exist_ok=True)

    # Compute slot scores from embeddings (5-fold LR)
    print("\n=== Computing slot scores (5-fold LR on embeddings) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    slot_scores = np.zeros(len(labels))
    for train_idx, test_idx in skf.split(embeddings, labels):
        lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        lr.fit(embeddings[train_idx], labels[train_idx])
        slot_scores[test_idx] = lr.predict_proba(embeddings[test_idx])[:, 1]

    slot_auc = roc_auc_score(labels, slot_scores)
    conf_auc = roc_auc_score(labels, confs)
    slot_ap = average_precision_score(labels, slot_scores)
    conf_ap = average_precision_score(labels, confs)
    print(f"Slot ROC-AUC:  {slot_auc:.4f}  PR-AUC: {slot_ap:.4f}")
    print(f"Conf ROC-AUC:  {conf_auc:.4f}  PR-AUC: {conf_ap:.4f}")

    # Evaluate strategies
    print("\n=== Evaluating filtering strategies ===")
    results = evaluate_strategies(slot_scores, confs, labels)

    print(f"\n{'Strategy':<20} {'AP':>8} {'PR-AUC':>8} {'Δ AP':>8} {'Param':>10}")
    print("-" * 60)
    print(f"{'Baseline (conf)':<20} {results['baseline']['ap']:8.4f} {results['baseline']['pr_auc']:8.4f} {'---':>8} {'---':>10}")
    print(f"{'Hard filter':<20} {results['hard_filter']['ap']:8.4f} {results['hard_filter']['pr_auc']:8.4f} {results['hard_filter']['delta']:+8.4f} {'thr=' + f'{results['hard_filter']['threshold']:.2f}':>10}")
    print(f"{'Soft penalize':<20} {results['soft_penalize']['ap']:8.4f} {results['soft_penalize']['pr_auc']:8.4f} {results['soft_penalize']['delta']:+8.4f} {'α=' + f'{results['soft_penalize']['alpha']:.1f}':>10}")
    print(f"{'Geometric mean':<20} {results['geometric_mean']['ap']:8.4f} {results['geometric_mean']['pr_auc']:8.4f} {results['geometric_mean']['delta']:+8.4f} {'w=' + f'{results['geometric_mean']['w_slot']:.2f}':>10}")
    print(f"{'LR meta':<20} {results['lr_meta']['ap']:8.4f} {results['lr_meta']['pr_auc']:8.4f} {results['lr_meta']['delta']:+8.4f} {'5-fold':>10}")

    best = max(results.values(), key=lambda x: x['ap'])
    print(f"\nBest: AP={best['ap']:.4f} (Δ={best['delta']:+.4f})")

    output = {
        'slot_auc': float(slot_auc),
        'conf_auc': float(conf_auc),
        'slot_ap': float(slot_ap),
        'conf_ap': float(conf_ap),
        'strategies': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating))
                           else vv for kk, vv in v.items()}
                       for k, v in results.items()},
        'n_samples': len(labels),
    }
    with open(os.path.join(args.output_dir, 'c4_results.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/c4_results.json")


if __name__ == '__main__':
    main()
