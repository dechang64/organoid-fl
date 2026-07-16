#!/usr/bin/env python3
"""
Slot Attention + RF-DETR confidence 组合 AUC 分析

评估 slot embeddings 和 confidence 是否互补：
1. Slot alone (logistic regression on 1024-dim slots)
2. Confidence alone (1-dim)
3. Combined: (slot_prob + conf) / 2
4. Combined: logistic regression on [slot_prob, conf]
5. Combined: logistic regression on [slots_flattened, conf]

Usage:
    python slot_confidence_combined.py \
        --slots upload/xxx_test_slots.npy \
        --labels upload/xxx_test_labels.npy \
        --confs upload/xxx_test_confs.npy
"""

import argparse
import json
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slots', required=True, help='test_slots.npy path')
    parser.add_argument('--labels', required=True, help='test_labels.npy path')
    parser.add_argument('--confs', required=True, help='test_confs.npy path')
    parser.add_argument('--output-dir', default='results/slot_conf_combined',
                        help='output directory')
    args = parser.parse_args()

    # Load
    slots = np.load(args.slots)  # (N, 8, 128)
    labels = np.load(args.labels)  # (N,)
    confs = np.load(args.confs)  # (N,)

    print(f"Slots shape: {slots.shape}")
    print(f"Labels shape: {labels.shape}  (TP={labels.sum()}, FP={len(labels)-labels.sum()})")
    print(f"Confs shape: {confs.shape}")
    print()

    # Flatten slots
    N = slots.shape[0]
    slots_flat = slots.reshape(N, -1)  # (N, 1024)
    print(f"Slots flattened: {slots_flat.shape}")
    print()

    # === 1. Individual AUC ===
    print("=" * 60)
    print("Individual AUC")
    print("=" * 60)

    # Confidence
    conf_roc = roc_auc_score(labels, confs)
    conf_pr = average_precision_score(labels, confs)
    print(f"RF-DETR confidence:  ROC-AUC={conf_roc:.4f}  PR-AUC={conf_pr:.4f}")

    # Slot (logistic regression, 5-fold CV)
    scaler = StandardScaler()
    slots_scaled = scaler.fit_transform(slots_flat)

    # 5-fold CV for slot
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    slot_probs = np.zeros(N)
    for train_idx, test_idx in skf.split(slots_scaled, labels):
        lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        lr.fit(slots_scaled[train_idx], labels[train_idx])
        slot_probs[test_idx] = lr.predict_proba(slots_scaled[test_idx])[:, 1]

    slot_roc = roc_auc_score(labels, slot_probs)
    slot_pr = average_precision_score(labels, slot_probs)
    print(f"Slot (5-fold CV LR):  ROC-AUC={slot_roc:.4f}  PR-AUC={slot_pr:.4f}")
    print()

    # === 2. Combined: simple average ===
    print("=" * 60)
    print("Combined: Simple Average")
    print("=" * 60)

    combined_avg = (slot_probs + confs) / 2.0
    avg_roc = roc_auc_score(labels, combined_avg)
    avg_pr = average_precision_score(labels, combined_avg)
    print(f"Avg (slot+conf)/2:    ROC-AUC={avg_roc:.4f}  PR-AUC={avg_pr:.4f}")
    print()

    # === 3. Combined: LR on [slot_prob, conf] ===
    print("=" * 60)
    print("Combined: LR on [slot_prob, conf]")
    print("=" * 60)

    meta_features = np.column_stack([slot_probs, confs])
    meta_scaler = StandardScaler()
    meta_scaled = meta_scaler.fit_transform(meta_features)

    meta_probs = np.zeros(N)
    for train_idx, test_idx in skf.split(meta_scaled, labels):
        lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        lr.fit(meta_scaled[train_idx], labels[train_idx])
        meta_probs[test_idx] = lr.predict_proba(meta_scaled[test_idx])[:, 1]

    meta_roc = roc_auc_score(labels, meta_probs)
    meta_pr = average_precision_score(labels, meta_probs)
    print(f"LR(slot_prob, conf):  ROC-AUC={meta_roc:.4f}  PR-AUC={meta_pr:.4f}")
    print()

    # === 4. Combined: LR on [slots_flattened, conf] ===
    print("=" * 60)
    print("Combined: LR on [slots_flattened(1024), conf]")
    print("=" * 60)

    combined_features = np.column_stack([slots_scaled, confs])
    combined_probs = np.zeros(N)
    for train_idx, test_idx in skf.split(combined_features, labels):
        lr = LogisticRegression(max_iter=5000, C=0.1, solver='lbfgs')
        lr.fit(combined_features[train_idx], labels[train_idx])
        combined_probs[test_idx] = lr.predict_proba(combined_features[test_idx])[:, 1]

    combined_roc = roc_auc_score(labels, combined_probs)
    combined_pr = average_precision_score(labels, combined_probs)
    print(f"LR(slots+conf):       ROC-AUC={combined_roc:.4f}  PR-AUC={combined_pr:.4f}")
    print()

    # === 5. Weighted combinations (scan weights) ===
    print("=" * 60)
    print("Weighted combination scan: w * slot_prob + (1-w) * conf")
    print("=" * 60)
    print(f"{'w_slot':>8}  {'ROC-AUC':>8}  {'PR-AUC':>8}  {'vs conf':>8}")
    print("-" * 40)

    best_w = 0.0
    best_roc = 0.0
    best_pr = 0.0
    for w in np.arange(0.0, 1.05, 0.05):
        weighted = w * slot_probs + (1 - w) * confs
        roc = roc_auc_score(labels, weighted)
        pr = average_precision_score(labels, weighted)
        delta = roc - conf_roc
        print(f"{w:8.2f}  {roc:8.4f}  {pr:8.4f}  {delta:+8.4f}")
        if roc > best_roc:
            best_roc = roc
            best_pr = pr
            best_w = w

    print()
    print(f"Best weight: w_slot={best_w:.2f}  ROC-AUC={best_roc:.4f}  PR-AUC={best_pr:.4f}")
    print(f"  vs RF-DETR alone: ROC {best_roc - conf_roc:+.4f}  PR {best_pr - conf_pr:+.4f}")
    print()

    # === 6. Correlation analysis ===
    print("=" * 60)
    print("Correlation between slot_prob and conf")
    print("=" * 60)
    from scipy.stats import pearsonr, spearmanr
    r_pearson, p_pearson = pearsonr(slot_probs, confs)
    r_spearman, p_spearman = spearmanr(slot_probs, confs)
    print(f"Pearson:  r={r_pearson:.4f}  p={p_pearson:.6f}")
    print(f"Spearman: r={r_spearman:.4f}  p={p_spearman:.6f}")
    print(f"  → {'高度相关(冗余)' if abs(r_pearson) > 0.8 else '中等相关(有一定互补)' if abs(r_pearson) > 0.5 else '低相关(强互补)'}")
    print()

    # === Summary ===
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<35} {'ROC-AUC':>8} {'PR-AUC':>8} {'vs conf(ROC)':>13}")
    print("-" * 65)
    print(f"{'RF-DETR confidence':<35} {conf_roc:8.4f} {conf_pr:8.4f} {'baseline':>13}")
    print(f"{'Slot (5-fold CV LR)':<35} {slot_roc:8.4f} {slot_pr:8.4f} {slot_roc-conf_roc:+13.4f}")
    print(f"{'Avg (slot+conf)/2':<35} {avg_roc:8.4f} {avg_pr:8.4f} {avg_roc-conf_roc:+13.4f}")
    print(f"{'LR(slot_prob, conf)':<35} {meta_roc:8.4f} {meta_pr:8.4f} {meta_roc-conf_roc:+13.4f}")
    print(f"{'LR(slots_1024 + conf)':<35} {combined_roc:8.4f} {combined_pr:8.4f} {combined_roc-conf_roc:+13.4f}")
    print(f"{'Best weighted (w={:.2f})'.format(best_w):<35} {best_roc:8.4f} {best_pr:8.4f} {best_roc-conf_roc:+13.4f}")
    print()

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        'individual': {
            'rf_detr_confidence': {'roc_auc': float(conf_roc), 'pr_auc': float(conf_pr)},
            'slot_5fold_lr': {'roc_auc': float(slot_roc), 'pr_auc': float(slot_pr)},
        },
        'combined': {
            'avg': {'roc_auc': float(avg_roc), 'pr_auc': float(avg_pr)},
            'lr_meta': {'roc_auc': float(meta_roc), 'pr_auc': float(meta_pr)},
            'lr_full': {'roc_auc': float(combined_roc), 'pr_auc': float(combined_pr)},
            'best_weighted': {
                'w_slot': float(best_w),
                'roc_auc': float(best_roc),
                'pr_auc': float(best_pr),
            },
        },
        'correlation': {
            'pearson_r': float(r_pearson),
            'pearson_p': float(p_pearson),
            'spearman_r': float(r_spearman),
            'spearman_p': float(p_spearman),
        },
    }
    with open(os.path.join(args.output_dir, 'combined_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_dir}/combined_results.json")


if __name__ == '__main__':
    main()
