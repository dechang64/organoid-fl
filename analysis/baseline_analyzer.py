#!/usr/bin/env python3
"""
MultiOrg Baseline Analysis Script
- Parse YOLOv12s training results from runs/detect/baseline_test/v12s_1280/
- Generate comprehensive analysis: convergence, per-class, confusion, FL readiness
- Output: baseline_analysis.json + baseline_report.md

Usage:
    python baseline_analyzer.py --results C:/Users/decha/organoid-fl/runs/detect/baseline_test/v12s_1280
"""

import os
import json
import argparse
import csv
from pathlib import Path
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze MultiOrg baseline YOLO results')
    parser.add_argument('--results', type=str, 
                        default='runs/detect/baseline_test/v12s_1280',
                        help='Path to YOLO results directory')
    parser.add_argument('--output', type=str, default='baseline_analysis',
                        help='Output directory for analysis')
    return parser.parse_args()


def load_results_csv(csv_path):
    """Load Ultralytics results.csv → list of dicts."""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys
            row = {k.strip(): v.strip() for k, v in row.items()}
            results.append(row)
    return results


def parse_metrics(results):
    """Extract key metrics from results.csv."""
    epochs = []
    train_box_loss = []
    train_cls_loss = []
    train_dfl_loss = []
    val_box_loss = []
    val_cls_loss = []
    val_dfl_loss = []
    precision = []
    recall = []
    mAP50 = []
    mAP50_95 = []
    lr = []
    
    for row in results:
        epochs.append(int(row.get('epoch', 0)))
        train_box_loss.append(float(row.get('train/box_loss', 0)))
        train_cls_loss.append(float(row.get('train/cls_loss', 0)))
        train_dfl_loss.append(float(row.get('train/dfl_loss', 0)))
        val_box_loss.append(float(row.get('val/box_loss', 0)))
        val_cls_loss.append(float(row.get('val/cls_loss', 0)))
        val_dfl_loss.append(float(row.get('val/dfl_loss', 0)))
        precision.append(float(row.get('metrics/precision(B)', 0)))
        recall.append(float(row.get('metrics/recall(B)', 0)))
        mAP50.append(float(row.get('metrics/mAP50(B)', 0)))
        mAP50_95.append(float(row.get('metrics/mAP50-95(B)', 0)))
        lr.append(float(row.get('lr/pg0', 0)))
    
    return {
        'epochs': epochs,
        'train_box_loss': train_box_loss,
        'train_cls_loss': train_cls_loss,
        'train_dfl_loss': train_dfl_loss,
        'val_box_loss': val_box_loss,
        'val_cls_loss': val_cls_loss,
        'val_dfl_loss': val_dfl_loss,
        'precision': precision,
        'recall': recall,
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'lr': lr,
    }


def analyze_convergence(metrics):
    """Analyze training convergence patterns."""
    mAP50 = metrics['mAP50']
    mAP50_95 = metrics['mAP50_95']
    val_box = metrics['val_box_loss']
    epochs = metrics['epochs']
    
    # Best epoch
    best_map50_epoch = mAP50.index(max(mAP50)) + 1
    best_map5095_epoch = mAP50_95.index(max(mAP50_95)) + 1
    
    # Convergence speed (epoch reaching 95% of final mAP50)
    final_map50 = mAP50[-1]
    target_95 = final_map50 * 0.95
    conv_epoch = None
    for i, m in enumerate(mAP50):
        if m >= target_95:
            conv_epoch = i + 1
            break
    
    # Overfitting detection (val loss starts increasing)
    min_val_loss_epoch = val_box.index(min(val_box)) + 1
    val_loss_trend = 'increasing' if val_box[-1] > min(val_box) * 1.1 else 'stable'
    
    # Late-stage improvement (last 20 epochs)
    late_improvement = mAP50[-1] - mAP50[-20] if len(mAP50) >= 20 else 0
    
    return {
        'best_mAP50_epoch': best_map50_epoch,
        'best_mAP50': max(mAP50),
        'best_mAP5095_epoch': best_map5095_epoch,
        'best_mAP5095': max(mAP50_95),
        'convergence_epoch_95': conv_epoch,
        'final_mAP50': final_map50,
        'final_mAP5095': mAP50_95[-1],
        'min_val_box_loss_epoch': min_val_loss_epoch,
        'val_loss_trend': val_loss_trend,
        'late_stage_improvement': late_improvement,
        'total_epochs': len(epochs),
    }


def analyze_per_class(val_output_text):
    """Parse per-class validation output."""
    lines = val_output_text.strip().split('\n')
    classes = {}
    
    for line in lines:
        parts = line.split()
        if len(parts) >= 7:
            cls_name = parts[0]
            if cls_name in ['all', 'organoid0', 'organoid1', 'organoid3', 'spheroid']:
                try:
                    classes[cls_name] = {
                        'images': int(parts[1]),
                        'instances': int(parts[2]),
                        'precision': float(parts[3]),
                        'recall': float(parts[4]),
                        'mAP50': float(parts[5]),
                        'mAP5095': float(parts[6]),
                    }
                except (ValueError, IndexError):
                    continue
    return classes


def analyze_fl_readiness(per_class, convergence):
    """Evaluate readiness for FL experiments."""
    all_metrics = per_class.get('all', {})
    
    # Class imbalance assessment
    instances = {k: v['instances'] for k, v in per_class.items() if k != 'all'}
    max_inst = max(instances.values()) if instances else 1
    min_inst = min(instances.values()) if instances else 1
    imbalance_ratio = max_inst / min_inst if min_inst > 0 else float('inf')
    
    # mAP spread (for EWA signal quality)
    map50_values = [v['mAP50'] for k, v in per_class.items() if k != 'all']
    map50_spread = max(map50_values) - min(map50_values) if map50_values else 0
    
    map5095_values = [v['mAP5095'] for k, v in per_class.items() if k != 'all']
    map5095_spread = max(map5095_values) - min(map5095_values) if map5095_values else 0
    
    # FL readiness score
    score = 0
    reasons = []
    
    if all_metrics.get('mAP50', 0) > 0.85:
        score += 25
        reasons.append("mAP50 > 0.85: strong baseline")
    else:
        reasons.append(f"mAP50 = {all_metrics.get('mAP50', 0):.3f}: need improvement")
    
    if imbalance_ratio < 5:
        score += 25
        reasons.append(f"Class imbalance ratio {imbalance_ratio:.1f}x: manageable")
    else:
        reasons.append(f"Class imbalance ratio {imbalance_ratio:.1f}x: high, FL will amplify")
    
    if map5095_spread > 0.05:
        score += 25
        reasons.append(f"mAP50-95 spread {map5095_spread:.3f}: good EWA signal")
    else:
        reasons.append(f"mAP50-95 spread {map5095_spread:.3f}: EWA signal may be weak")
    
    if convergence.get('val_loss_trend') == 'stable':
        score += 25
        reasons.append("Val loss stable: no severe overfitting")
    else:
        reasons.append("Val loss increasing: possible overfitting")
    
    return {
        'score': score,
        'imbalance_ratio': imbalance_ratio,
        'class_instances': instances,
        'mAP50_spread': map50_spread,
        'mAP5095_spread': map5095_spread,
        'reasons': reasons,
        'fl_recommendation': 'ready' if score >= 75 else 'cautious' if score >= 50 else 'not_ready',
    }


def generate_report(convergence, per_class, fl_readiness, val_text):
    """Generate markdown analysis report."""
    report = f"""# MultiOrg Baseline Analysis Report

## Training Summary

| Metric | Value |
|--------|-------|
| Total Epochs | {convergence['total_epochs']} |
| Best mAP50 | {convergence['best_mAP50']:.4f} (epoch {convergence['best_mAP50_epoch']}) |
| Best mAP50-95 | {convergence['best_mAP5095']:.4f} (epoch {convergence['best_mAP5095_epoch']}) |
| Final mAP50 | {convergence['final_mAP50']:.4f} |
| Final mAP50-95 | {convergence['final_mAP5095']:.4f} |
| Convergence (95% of final) | Epoch {convergence.get('convergence_epoch_95', 'N/A')} |
| Val Loss Trend | {convergence['val_loss_trend']} |
| Late-stage improvement | {convergence['late_stage_improvement']:.4f} |

## Per-Class Performance

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|-------|--------|-----------|---|---|-------|----------|
"""
    for cls_name in ['all', 'organoid0', 'organoid1', 'organoid3', 'spheroid']:
        if cls_name in per_class:
            c = per_class[cls_name]
            report += f"| {cls_name} | {c['images']} | {c['instances']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['mAP50']:.3f} | {c['mAP5095']:.3f} |\n"
    
    report += f"""
## FL Readiness Assessment

**Score: {fl_readiness['score']}/100 — {fl_readiness['fl_recommendation'].upper()}**

"""
    for reason in fl_readiness['reasons']:
        report += f"- {reason}\n"
    
    report += f"""
### Class Distribution
"""
    for cls, n in fl_readiness['class_instances'].items():
        report += f"- {cls}: {n} instances\n"
    
    report += f"""
### EWA Signal Quality
- mAP50 spread: {fl_readiness['mAP50_spread']:.4f} ({'good' if fl_readiness['mAP50_spread'] > 0.05 else 'weak'})
- mAP50-95 spread: {fl_readiness['mAP5095_spread']:.4f} ({'good' if fl_readiness['mAP5095_spread'] > 0.05 else 'weak'})
- Imbalance ratio: {fl_readiness['imbalance_ratio']:.1f}x

## FL Experiment Recommendations

### Phase 1: Non-IID Partitioning
- **Client A**: organoid0-dominant (natural majority class)
- **Client B**: organoid1 + organoid3 (mixed developmental stages)
- **Client C**: spheroid-dominant (minority class specialist)
- **Non-IID ratio**: 0.8 (moderate) or 0.95 (extreme)

### Phase 2: EWA Active Aggregation
- **Signal**: mAP50-95 (spread = {fl_readiness['mAP5095_spread']:.4f}, {'sufficient' if fl_readiness['mAP5095_spread'] > 0.05 else 'may need more extreme non-IID'})
- **Warmup**: 2 rounds FedAvg before EWA
- **Compare**: FedAvg vs EWA vs FedProx (mu=0.01, 0.1, 1.0)

### Model Selection for FL
- **v12s** (current): 9.2M params, 48.5h for 100 epochs — too heavy for FL
- **v12n**: ~2M params, ~10h estimated — recommended for FL
- **imgsz**: 640 (vs current 1280) — 4x faster, acceptable mAP50-95 trade-off

## Raw Validation Output
```
{val_text}
```

---
Generated by baseline_analyzer.py
"""
    return report


def main():
    args = parse_args()
    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing: {results_dir}")
    
    # Load results.csv
    csv_path = results_dir / 'results.csv'
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return
    
    results = load_results_csv(csv_path)
    metrics = parse_metrics(results)
    convergence = analyze_convergence(metrics)
    
    print(f"  Epochs: {convergence['total_epochs']}")
    print(f"  Best mAP50: {convergence['best_mAP50']:.4f} (epoch {convergence['best_mAP50_epoch']})")
    print(f"  Best mAP50-95: {convergence['best_mAP5095']:.4f}")
    
    # Validation output (from user-provided text or args.txt)
    val_text = ""
    val_path = results_dir / 'val_output.txt'
    if val_path.exists():
        with open(val_path, 'r', encoding='utf-8') as f:
            val_text = f.read()
    else:
        # Use default from known results
        val_text = """                   all         84       2469      0.823      0.813      0.885      0.624
             organoid0         77       1295      0.861      0.853      0.915       0.58
             organoid1         70        548      0.739      0.781      0.841      0.609
             organoid3         54        401       0.82      0.866      0.923      0.689
              spheroid         48        225      0.874      0.751      0.862       0.62"""
    
    per_class = analyze_per_class(val_text)
    fl_readiness = analyze_fl_readiness(per_class, convergence)
    
    print(f"  FL Readiness: {fl_readiness['score']}/100 — {fl_readiness['fl_recommendation']}")
    
    # Save JSON
    analysis = {
        'convergence': convergence,
        'per_class': per_class,
        'fl_readiness': fl_readiness,
        'val_output': val_text,
    }
    json_path = output_dir / 'baseline_analysis.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {json_path}")
    
    # Save report
    report = generate_report(convergence, per_class, fl_readiness, val_text)
    md_path = output_dir / 'baseline_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved: {md_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("  ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Best mAP50:    {convergence['best_mAP50']:.4f} (epoch {convergence['best_mAP50_epoch']})")
    print(f"  Best mAP50-95: {convergence['best_mAP5095']:.4f} (epoch {convergence['best_mAP5095_epoch']})")
    print(f"  FL Readiness:  {fl_readiness['score']}/100 — {fl_readiness['fl_recommendation']}")
    print(f"  Imbalance:     {fl_readiness['imbalance_ratio']:.1f}x")
    print(f"  mAP50-95 spread: {fl_readiness['mAP5095_spread']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
