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
        for i, det in enumerate(detections):
            # TP = first tp_bbox detections, FP = rest
            label = 'TP' if i < tp_bbox else 'FP'
            
            morph = det.get('morphology', det)  # might be nested
            if 'area' not in morph:
                morph = det  # try direct
            
            primitive = {
                'image': img_name,
                'label': label,
                'area': float(morph.get('area', 0)),
                'perimeter': float(morph.get('perimeter', 0)),
                'circularity': float(morph.get('circularity', 0)),
                'solidity': float(morph.get('solidity', 0)),
                'aspect_ratio': float(morph.get('aspect_ratio', 0)),
                'eccentricity': float(morph.get('eccentricity', 0)),
                'confidence': float(morph.get('confidence', det.get('confidence', 0))),
            }
            primitives.append(primitive)
    
    return primitives


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
    
    # t-SNE
    perplexity = min(5, len(X) - 1)
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
        tp_medians.append(np.median(tp_vals))
        fp_medians.append(np.median(fp_vals))
        u, p = mannwhitneyu(tp_vals, fp_vals, alternative='two-sided')
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
    
    # Classification
    ax = axes[1]
    loo = LeaveOneOut()
    
    results = {'Baseline (all TP)': y.mean()}
    
    # Morphology only
    X_morph = X_scaled[:, :6]  # exclude confidence
    scores = cross_val_score(LogisticRegression(random_state=42, max_iter=1000), X_morph, y, cv=loo, scoring='accuracy')
    results['Morphology only (LR)'] = scores.mean()
    
    scores = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=42), X_morph, y, cv=loo, scoring='accuracy')
    results['Morphology only (RF)'] = scores.mean()
    
    # Confidence only
    X_conf = X_scaled[:, 6:7]
    scores = cross_val_score(LogisticRegression(random_state=42), X_conf, y, cv=loo, scoring='accuracy')
    results['Confidence only (LR)'] = scores.mean()
    
    # All features
    scores = cross_val_score(LogisticRegression(random_state=42, max_iter=1000), X_scaled, y, cv=loo, scoring='accuracy')
    results['All features (LR)'] = scores.mean()
    
    names = list(results.keys())
    accs = list(results.values())
    colors2 = ['#94A3B8' if 'Baseline' in n else '#60A5FA' if 'Morphology' in n else '#3B82F6' for n in names]
    bars = ax.bar(range(len(names)), accs, color=colors2, edgecolor='black', linewidth=0.5)
    ax.axhline(y=results['Baseline (all TP)'], color='#EF4444', linestyle='--', linewidth=1.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(' (', '\n(') for n in names], fontsize=9)
    ax.set_ylabel('LOO Accuracy', fontsize=12)
    ax.set_title('Classification: Can Morphology Separate TP/FP?', fontsize=14, fontweight='bold')
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
        "total_primitives": len(y),
        "tp": n_tp,
        "fp": n_fp,
        "features": features,
        "p_values": dict(zip(features, [f"{p:.4f}" for p in p_values])),
        "tp_medians": dict(zip(features, [f"{m:.3f}" for m in tp_medians])),
        "fp_medians": dict(zip(features, [f"{m:.3f}" for m in fp_medians])),
        "classification_loo": {k: f"{v:.3f}" for k, v in results.items()},
    }
    
    with open(output_dir / 'phase1_multiorg_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
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
