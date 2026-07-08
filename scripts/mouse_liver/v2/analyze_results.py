"""
Mouse Liver Organoid v2 — Results Analyzer

Usage:
    cd C:\\Users\\decha\\organoid-fl
    python scripts\\mouse_liver\\v2\\analyze_results.py

Outputs:
    runs/mouse_liver_v2/summary.json   — all metrics in one JSON
    runs/mouse_liver_v2/summary.txt    — formatted tables
    console                            — same tables printed
"""

import json
import os
import sys
from pathlib import Path

# ============================================================
# Config
# ============================================================
OUTPUT_BASE = Path("runs/mouse_liver_v2")
BATCHES = ["b1", "b2", "b3"]
FL_TAGS = ["F1", "F2", "F3", "F4"]


# ============================================================
# Helpers
# ============================================================
def load_json(path):
    """Safe JSON loader."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"  [WARN] Failed to load {path}: {e}")
        return None


def fmt(val, suffix=""):
    """Format a float to 2 decimal places."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{val:.2f}{suffix}"
    return str(val) + suffix


def pct(val):
    """Format as percentage."""
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def get_bbox_metrics(eval_json):
    """Extract bbox TP/FP/FN/P/R/F1 from eval_test.json."""
    if not eval_json:
        return None
    bbox = eval_json.get("bbox", {})
    return {
        "tp": bbox.get("tp", 0),
        "fp": bbox.get("fp", 0),
        "fn": bbox.get("fn", 0),
        "precision": bbox.get("precision", 0),
        "recall": bbox.get("recall", 0),
        "f1": bbox.get("f1", 0),
        "n_images": eval_json.get("n_images", 0),
        "resolution": eval_json.get("resolution", "N/A"),
    }


def get_summary_metrics(json_data, key="summary"):
    """Extract P/R/F1 from traditional/sam2 JSON."""
    if not json_data:
        return None
    s = json_data.get(key, {})
    return {
        "tp": s.get("total_tp", 0),
        "fp": s.get("total_fp", 0),
        "fn": s.get("total_fn", 0),
        "precision": s.get("precision", 0),
        "recall": s.get("recall", 0),
        "f1": s.get("f1", 0),
        "n_images": s.get("n_images", 0),
        "method": s.get("method", "N/A"),
    }


# ============================================================
# Collectors
# ============================================================
def collect_baseline_ceiling():
    """Table 1: B1/B2/B3 full + central."""
    results = {}
    for b in BATCHES:
        results[b] = get_bbox_metrics(
            load_json(OUTPUT_BASE / b / "full" / "eval_test.json")
        )

    # Central: evaluate.py --tag central 会在 b1/b2/b3 下各生成 central/eval_test.json
    # 也可能在 central/ 下有 eval_test.json
    central_results = []
    for b in BATCHES:
        r = get_bbox_metrics(
            load_json(OUTPUT_BASE / b / "central" / "eval_test.json")
        )
        if r and r["n_images"] > 0:
            central_results.append((b, r))

    # 合并 central 的 TP/FP/FN
    if central_results:
        total_tp = sum(r["tp"] for _, r in central_results)
        total_fp = sum(r["fp"] for _, r in central_results)
        total_fn = sum(r["fn"] for _, r in central_results)
        total_img = sum(r["n_images"] for _, r in central_results)
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results["central"] = {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": prec, "recall": rec, "f1": f1,
            "n_images": total_img,
            "resolution": central_results[0][1]["resolution"],
        }
    else:
        # 也尝试 central/eval_test.json
        results["central"] = get_bbox_metrics(
            load_json(OUTPUT_BASE / "central" / "eval_test.json")
        )

    return results


def collect_cross_domain():
    """Table 2: zeroshot vs fewshot for B2/B3."""
    results = {}
    for b in ["b2", "b3"]:
        results[b] = {
            "zeroshot": get_bbox_metrics(
                load_json(OUTPUT_BASE / b / f"b1_to_{b}_zeroshot" / "eval_test.json")
            ),
            "fewshot": get_bbox_metrics(
                load_json(OUTPUT_BASE / b / "fewshot" / "eval_test.json")
            ),
            "full": get_bbox_metrics(
                load_json(OUTPUT_BASE / b / "full" / "eval_test.json")
            ),
        }
    return results


def collect_traditional():
    """Table 3: Traditional CV per batch."""
    results = {}
    for b in BATCHES:
        results[b] = get_summary_metrics(
            load_json(OUTPUT_BASE / b / "traditional" / "traditional_results.json")
        )
    return results


def collect_sam2():
    """Table 4: SAM2 segmentation per batch."""
    results = {}
    for b in BATCHES:
        results[b] = {
            "full": get_summary_metrics(
                load_json(OUTPUT_BASE / b / "sam2_full" / "sam2_results.json")
            ),
        }
    # fewshot only for b2/b3
    for b in ["b2", "b3"]:
        results[b]["fewshot"] = get_summary_metrics(
            load_json(OUTPUT_BASE / b / "sam2_fewshot" / "sam2_results.json")
        )
    return results


def collect_fl():
    """Table 5: FL experiments."""
    results = {}
    for tag in FL_TAGS:
        fl_json = load_json(OUTPUT_BASE / "fl" / tag / "fl_results.json")
        if not fl_json:
            results[tag] = None
            continue

        rounds = fl_json.get("rounds", [])
        if not rounds:
            results[tag] = None
            continue

        final = rounds[-1]
        best_round = max(rounds, key=lambda r: r.get("global_mAP50-95", 0))

        results[tag] = {
            "tag": tag,
            "gate": fl_json.get("gate", "N/A"),
            "order": fl_json.get("order", "N/A"),
            "num_rounds": fl_json.get("num_rounds", 0),
            "final_mAP50": final.get("global_mAP50", 0),
            "final_mAP50_95": final.get("global_mAP50-95", 0),
            "best_mAP50": best_round.get("global_mAP50", 0),
            "best_mAP50_95": best_round.get("global_mAP50-95", 0),
            "best_round": best_round.get("round", 0),
            "rounds": rounds,
        }
    return results


def collect_training_configs():
    """Bonus: collect training_config.json from each model dir."""
    configs = {}
    model_dirs = [
        "b1/full", "b2/full", "b3/full", "central",
        "b2/fewshot", "b3/fewshot",
    ]
    for d in model_dirs:
        cfg = load_json(OUTPUT_BASE / d / "training_config.json")
        if cfg:
            key = d.replace("/", "_")
            configs[key] = {
                "epochs": cfg.get("epochs", "N/A"),
                "resolution": cfg.get("resolution", "N/A"),
                "batch_size": cfg.get("batch_size", "N/A"),
                "grad_accum_steps": cfg.get("grad_accum_steps", "N/A"),
                "early_stopping": cfg.get("early_stopping", "N/A"),
                "early_stopping_patience": cfg.get("early_stopping_patience", "N/A"),
            }
    return configs


# ============================================================
# Formatters
# ============================================================
def fmt_table_1_baseline(data):
    """Table 1: Baseline + Ceiling."""
    lines = []
    lines.append("=" * 70)
    lines.append("Table 1: Baseline & Ceiling (RF-DETR bbox detection)")
    lines.append("=" * 70)
    header = f"{'Model':<12} {'TP':>6} {'FP':>6} {'FN':>6} {'P':>8} {'R':>8} {'F1':>8} {'Imgs':>6}"
    lines.append(header)
    lines.append("-" * 70)

    order = ["b1", "b2", "b3", "central"]
    labels = {"b1": "B1 (6)", "b2": "B2 (6)", "b3": "B3 (12)", "central": "Central (24)"}

    for key in order:
        r = data.get(key)
        if r:
            lines.append(
                f"{labels[key]:<12} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
                f"{fmt(r['precision']):>8} {fmt(r['recall']):>8} {fmt(r['f1']):>8} "
                f"{r['n_images']:>6}"
            )
        else:
            lines.append(f"{labels[key]:<12} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>6}")

    lines.append("")
    return "\n".join(lines)


def fmt_table_2_transfer(data):
    """Table 2: Cross-domain transfer."""
    lines = []
    lines.append("=" * 70)
    lines.append("Table 2: Cross-Domain Transfer (B1 -> B2/B3)")
    lines.append("=" * 70)
    header = f"{'Target':<8} {'Mode':<12} {'TP':>6} {'FP':>6} {'FN':>6} {'P':>8} {'R':>8} {'F1':>8}"
    lines.append(header)
    lines.append("-" * 70)

    for b in ["b2", "b3"]:
        for mode, label in [("full", "Full (6)"), ("zeroshot", "Zeroshot"), ("fewshot", "Fewshot (3)")]:
            r = data.get(b, {}).get(mode)
            if r:
                lines.append(
                    f"{b.upper():<8} {label:<12} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
                    f"{fmt(r['precision']):>8} {fmt(r['recall']):>8} {fmt(r['f1']):>8}"
                )
            else:
                lines.append(f"{b.upper():<8} {label:<12} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        lines.append("")

    lines.append("")
    return "\n".join(lines)


def fmt_table_3_traditional(data):
    """Table 3: Traditional CV vs RF-DETR."""
    lines = []
    lines.append("=" * 70)
    lines.append("Table 3: Traditional CV (Otsu) vs RF-DETR (Full)")
    lines.append("=" * 70)
    header = f"{'Batch':<6} {'Method':<14} {'TP':>6} {'FP':>6} {'FN':>6} {'P':>8} {'R':>8} {'F1':>8}"
    lines.append(header)
    lines.append("-" * 70)

    # Load RF-DETR full results for comparison
    rf_data = collect_baseline_ceiling()

    for b in BATCHES:
        # Traditional
        r = data.get(b)
        if r:
            lines.append(
                f"{b.upper():<6} {'Traditional':<14} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
                f"{fmt(r['precision']):>8} {fmt(r['recall']):>8} {fmt(r['f1']):>8}"
            )
        # RF-DETR
        rf = rf_data.get(b)
        if rf:
            lines.append(
                f"{b.upper():<6} {'RF-DETR':<14} {rf['tp']:>6} {rf['fp']:>6} {rf['fn']:>6} "
                f"{fmt(rf['precision']):>8} {fmt(rf['recall']):>8} {fmt(rf['f1']):>8}"
            )
        lines.append("")

    lines.append("")
    return "\n".join(lines)


def fmt_table_4_sam2(data):
    """Table 4: SAM2 segmentation."""
    lines = []
    lines.append("=" * 70)
    lines.append("Table 4: SAM2 Segmentation (zero-shot, bbox-guided)")
    lines.append("=" * 70)
    header = f"{'Batch':<6} {'Mode':<12} {'TP':>6} {'FP':>6} {'FN':>6} {'P':>8} {'R':>8} {'F1':>8}"
    lines.append(header)
    lines.append("-" * 70)

    for b in BATCHES:
        for mode in ["full", "fewshot"]:
            if mode == "fewshot" and b not in ["b2", "b3"]:
                continue
            r = data.get(b, {}).get(mode)
            label = "Full" if mode == "full" else "Fewshot"
            if r:
                lines.append(
                    f"{b.upper():<6} {label:<12} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
                    f"{fmt(r['precision']):>8} {fmt(r['recall']):>8} {fmt(r['f1']):>8}"
                )
            else:
                lines.append(f"{b.upper():<6} {label:<12} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        lines.append("")

    lines.append("")
    return "\n".join(lines)


def fmt_table_5_fl(data):
    """Table 5: FL experiments."""
    lines = []
    lines.append("=" * 70)
    lines.append("Table 5: Federated Learning (4 strategies)")
    lines.append("=" * 70)
    header = f"{'Tag':<6} {'Gate':<8} {'Order':<12} {'Rounds':>7} {'Final mAP50':>14} {'Final mAP50-95':>16} {'Best mAP50-95':>16} {'Best@Rd':>8}"
    lines.append(header)
    lines.append("-" * 70)

    for tag in FL_TAGS:
        r = data.get(tag)
        if r:
            lines.append(
                f"{tag:<6} {r['gate']:<8} {r['order']:<12} {r['num_rounds']:>7} "
                f"{fmt(r['final_mAP50']):>14} {fmt(r['final_mAP50_95']):>16} "
                f"{fmt(r['best_mAP50_95']):>16} {r['best_round']:>8}"
            )
        else:
            lines.append(f"{tag:<6} {'N/A':<8} {'N/A':<12} {'N/A':>7} {'N/A':>14} {'N/A':>16} {'N/A':>16} {'N/A':>8}")

    lines.append("")
    return "\n".join(lines)


def fmt_table_6_fl_convergence(data):
    """Table 6: FL per-round convergence."""
    lines = []
    lines.append("=" * 70)
    lines.append("Table 6: FL Per-Round Convergence (mAP50 / mAP50-95)")
    lines.append("=" * 70)

    for tag in FL_TAGS:
        r = data.get(tag)
        if not r or not r.get("rounds"):
            lines.append(f"{tag}: N/A")
            continue

        lines.append(f"\n{tag} (gate={r['gate']}, order={r['order']}):")
        header = f"  {'Round':>6} {'mAP50':>10} {'mAP50-95':>10}"
        lines.append(header)
        for rd in r["rounds"]:
            lines.append(
                f"  {rd['round']:>6} {fmt(rd.get('global_mAP50', 0)):>10} "
                f"{fmt(rd.get('global_mAP50-95', 0)):>10}"
            )

    lines.append("")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
def main():
    if not OUTPUT_BASE.exists():
        print(f"[ERROR] {OUTPUT_BASE} not found")
        print("Run this from the organoid-fl repo root:")
        print("  cd C:\\Users\\decha\\organoid-fl")
        print("  python scripts\\mouse_liver\\v2\\analyze_results.py")
        sys.exit(1)

    print("Collecting results from", OUTPUT_BASE.resolve())
    print()

    # Collect
    baseline = collect_baseline_ceiling()
    transfer = collect_cross_domain()
    traditional = collect_traditional()
    sam2 = collect_sam2()
    fl = collect_fl()
    configs = collect_training_configs()

    # Format tables
    tables = [
        fmt_table_1_baseline(baseline),
        fmt_table_2_transfer(transfer),
        fmt_table_3_traditional(traditional),
        fmt_table_4_sam2(sam2),
        fmt_table_5_fl(fl),
        fmt_table_6_fl_convergence(fl),
    ]

    output_text = "\n\n".join(tables)

    # Print to console
    print(output_text)

    # Save summary.txt
    txt_path = OUTPUT_BASE / "summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"\nSaved: {txt_path}")

    # Save summary.json (all raw data)
    json_path = OUTPUT_BASE / "summary.json"
    all_data = {
        "baseline_ceiling": baseline,
        "cross_domain": transfer,
        "traditional_cv": traditional,
        "sam2": sam2,
        "fl": {tag: ({k: v for k, v in r.items() if k != "rounds"} if r else None) for tag, r in fl.items()},
        "fl_convergence": {tag: (r["rounds"] if r else None) for tag, r in fl.items()},
        "training_configs": configs,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {json_path}")

    print("\nDone! Send me summary.txt and summary.json.")


if __name__ == "__main__":
    main()
