"""
Phase 1: MultiOrg 形态学特征提取与 TP/FP 可分性分析

Usage:
    cd C:\\Users\\decha\\organoid-fl
    python scripts\\multiorg\\phase1_morphology_analysis.py --json results\\multiorg_sam2_zeroshot\\multiorg_sam2_results.json

输出:
    results/phase1_multiorg_tsne.png
    results/phase1_multiorg_analysis.png
    results/phase1_multiorg_summary.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract_primitives(results_json):
    """从 MultiOrg SAM2 results JSON 提取 primitive 向量"""
    with open(results_json, encoding='utf-8') as f:
        data = json.load(f)
    
    primitives = []
    for img_data in data.get('per_image', []):
        img_name = img_data['image']
        tp_bbox = img_data.get('tp_bbox', 0)
        fp_bbox = img_data.get('fp_bbox', 0)
        
        detections = img_data.get('detections', [])
        
        # Bug fix: detections 可能不是 TP-first 排序
        # 检查每个 detection 是否有 matched/is_tp 字段
        if detections and 'matched' in detections[0]:
            # 有 matched 字段，直接用
            for det in detections:
                label = 'TP' if det.get('matched', False) else 'FP'
                primitives.append(_make_primitive(img_name, label, det))
        elif detections and 'is_tp' in detections[0]:
            for det in detections:
                label = 'TP' if det.get('is_tp', False) else 'FP'
                primitives.append(_make_primitive(img_name, label, det))
        else:
            # Fallback: 假设 TP 在前（按 confidence 降序匹配 IoU）
            # 但这不一定准确，打印警告
            if tp_bbox > 0 and fp_bbox > 0:
                print(f"  [WARN] {img_name}: no matched/is_tp field, assuming TP-first order")
            for i, det in enumerate(detections):
                label = 'TP' if i < tp_bbox else 'FP'
                primitives.append(_make_primitive(img_name, label, det))
    
    return primitives


def _make_primitive(img_name, label, det):
    """从 detection dict 提取形态学特征"""
    # MultiOrg SAM2: morphology fields directly on det (not nested)
    morph = det.get('morphology', det)
    if 'area' not in morph:
        morph = det
    
    # eccentricity might not exist in MultiOrg (only mouse liver has it)
    ecc = float(morph.get('eccentricity', 0))
    
    return {
        'image': img_name,
        'label': label,
        'area': float(morph.get('area', 0)),
        'perimeter': float(morph.get('perimeter', 0)),
        'circularity': float(morph.get('circularity', 0)),
        'solidity': float(morph.get('solidity', 0)),
        'aspect_ratio': float(morph.get('aspect_ratio', 0)),
        'eccentricity': ecc,
        'confidence': float(morph.get('confidence', det.get('confidence', 0))),
    }


def analyze(primitives, output_dir):
    """Run full analysis"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features = ['area', 'perimeter', 'circularity', 'solidity', 'aspect_ratio', 'eccentricity', 'confidence']
    labels_en = ['Area', 'Perimeter', 'Circularity', 'Solidity', 'Aspect Ratio', 'Eccentricity', 'Confidence']
    
    X = np.array([[p[f] for f in features] for p in primitives])
    y = np.array([1 if p['label']=='TP' else 0 for p in primitives])
    
    n_tp = sum(y)
    n_fp = len(y) - n_tp
    print(f"Total: {len(y)} primitives ({n_tp} TP, {n_fp} FP)")
    
    if n_fp == 0:
        print("No FP detected — cannot do TP/FP analysis")
        return
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # t-SNE — perplexity scales with dataset size
    n = len(X)
    perplexity = min(30, max(5, n - 1))  # Bug fix: was min(5, n-1), too low for large data
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_2d = tsne.fit_transform(X_scaled)
    
    tp_mask = y == 1
    fp_mask = y == 0
    
    # === Plot 1: t-SNE ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.scatter(X_2d[tp_mask, 0], X_2d[tp_mask, 1], c='green', label='TP', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax.scatter(X_2d[fp_mask, 0], X_2d[fp_mask, 1], c='red', label='FP', s=50, alpha=0.6, marker='x', linewidths=1.5)
    ax.set_title('t-SNE: TP vs FP', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    
    ax = axes[1]
    # Color by area
    areas = X[:, 0]
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=np.log10(areas + 1), cmap='viridis', s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
    plt.colorbar(scatter, ax=ax, label='log10(Area)')
    ax.set_title('t-SNE: By Area', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase1_multiorg_tsne.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {output_dir / 'phase1_multiorg_tsne.png'}")
    
    # === Plot 2: p-values + classification ===
    p_values = []
    tp_medians = []
    fp_medians = []
    for i, feat in enumerate(features):
        tp_vals = X[tp_mask, i]
        fp_vals = X[fp_mask, i]
        tp_medians.append(np.median(tp_vals) if len(tp_vals) > 0 else 0)
        fp_medians.append(np.median(fp_vals) if len(fp_vals) > 0 else 0)
        if len(tp_vals) > 0 and len(fp_vals) > 0:
            try:
                u, p = mannwhitneyu(tp_vals, fp_vals, alternative='two-sided')
            except ValueError:
                p = 1.0  # all values identical
        else:
            p = 1.0
        p_values.append(p)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    
    # p-values
    ax = axes[0]
    colors = ['#EF4444' if p < 0.05 else '#94A3B8' for p in p_values]
    ax.barh(range(len(features)), [-np.log10(p) for p in p_values], color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(x=-np.log10(0.05), color='#EF4444', linestyle='--', linewidth=1.5, label='p=0.05')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(labels_en, fontsize=11)
    ax.set_xlabel('-log10(p-value)', fontsize=12)
    ax.set_title('TP vs FP: Mann-Whitney U Test', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    for i, p in enumerate(p_values):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(-np.log10(p) + 0.1, i, sig, va='center', fontsize=11, fontweight='bold')
    
    # Classification — use 5-fold CV for large datasets, LOO for small
    n_samples = len(y)
    if n_samples > 200:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_name = "5-fold"
    else:
        cv = LeaveOneOut()
        cv_name = "LOO"
    
    results = {'Baseline (all TP)': y.mean()}
    
    # Morphology only
    X_morph = X_scaled[:, :6]  # exclude confidence
    scores = cross_val_score(LogisticRegression(random_state=42, max_iter=1000), X_morph, y, cv=cv, scoring='accuracy')
    results['Morphology only (LR)'] = scores.mean()
    
    scores = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=42), X_morph, y, cv=cv, scoring='accuracy')
    results['Morphology only (RF)'] = scores.mean()
    
    # Confidence only
    X_conf = X_scaled[:, 6:7]
    scores = cross_val_score(LogisticRegression(random_state=42), X_conf, y, cv=cv, scoring='accuracy')
    results['Confidence only (LR)'] = scores.mean()
    
    # All features
    scores = cross_val_score(LogisticRegression(random_state=42, max_iter=1000), X_scaled, y, cv=cv, scoring='accuracy')
    results['All features (LR)'] = scores.mean()
    
    names = list(results.keys())
    accs = list(results.values())
    colors2 = ['#94A3B8' if 'Baseline' in n else '#60A5FA' if 'Morphology' in n else '#3B82F6' for n in names]
    bars = ax.bar(range(len(names)), accs, color=colors2, edgecolor='black', linewidth=0.5)
    ax.axhline(y=results['Baseline (all TP)'], color='#EF4444', linestyle='--', linewidth=1.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(' (', '\n(') for n in names], fontsize=9)
    ax.set_ylabel(f'{cv_name} Accuracy', fontsize=12)
    ax.set_title(f'Classification: Can Morphology Separate TP/FP? ({cv_name})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.savefig(output_dir / 'phase1_multiorg_analysis.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {output_dir / 'phase1_multiorg_analysis.png'}")
    
    # === Summary ===
    print(f"\n=== Feature Statistics (TP vs FP) ===")
    for i, feat in enumerate(features):
        sig = '***' if p_values[i] < 0.001 else '**' if p_values[i] < 0.01 else '*' if p_values[i] < 0.05 else 'ns'
        print(f"  {feat:15s}: TP median={tp_medians[i]:.3f}, FP median={fp_medians[i]:.3f}, p={p_values[i]:.4f} {sig}")
    
    print(f"\n=== Classification (LOO) ===")
    for name, acc in results.items():
        print(f"  {name}: {acc:.3f}")
    
    summary = {
        "phase": "Phase 1 MultiOrg: 形态学特征提取与 TP/FP 可分性分析",
        "total_primitives": int(len(y)),
        "tp": int(n_tp),
        "fp": int(n_fp),
        "features": features,
        "p_values": dict(zip(features, [f"{p:.4f}" for p in p_values])),
        "tp_medians": dict(zip(features, [f"{m:.3f}" for m in tp_medians])),
        "fp_medians": dict(zip(features, [f"{m:.3f}" for m in fp_medians])),
        "classification_loo": {k: f"{v:.3f}" for k, v in results.items()},
    }
    
    with open(output_dir / 'phase1_multiorg_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else str(o))
    print(f"\nSaved: {output_dir / 'phase1_multiorg_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description='MultiOrg Phase 1: Morphology analysis')
    parser.add_argument('--json', required=True, help='Path to multiorg_sam2_results.json')
    parser.add_argument('--output', default='results', help='Output directory')
    args = parser.parse_args()
    
    print(f"Loading: {args.json}")
    primitives = extract_primitives(args.json)
    print(f"Extracted {len(primitives)} primitives")
    
    analyze(primitives, args.output)


if __name__ == '__main__':
    main()
