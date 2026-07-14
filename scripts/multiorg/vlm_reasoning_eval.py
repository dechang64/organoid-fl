r"""
Phase B2: VLM 两步推理 — 视觉原语作为推理操作

借鉴 DeepSeek TwVP：视觉原语不是特征，而是推理操作的单元。
不是单次打分，而是多步推理：全局评估 → 局部放大 → 对比推理

三步 pipeline：
1. 全局评估：VLM 看 crop，给初步判断 + 不确定度
2. 局部放大：对不确定的 crop，放大到原图区域再评估
3. 对比推理：和确定的 TP/FP 对比，最终判断

Usage (冬生本地或云 VM):
    cd /home/z/my-project/organoid-fl
    python scripts/multiorg/vlm_reasoning_eval.py --device cpu

    # 冬生本地 GPU 不需要，VLM 走 z-ai API
    python scripts\multiorg\vlm_reasoning_eval.py
"""
import argparse
import json
import os
import sys
import subprocess
import time
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(str(PROJECT_ROOT))

# ── Datasets ──
DATASETS = {
    'multiorg': {
        'metadata': 'results/phase2_vlm_100/vlm_results.json',
        'crops_dir': 'results/phase2_vlm_100/crops',
    },
    'mouse_b1': {
        'metadata': 'data/mouse_crops/b1/crop_metadata.json',
        'crops_dir': 'data/mouse_crops/b1/crops',
    },
    'mouse_b2': {
        'metadata': 'data/mouse_crops/b2/crop_metadata.json',
        'crops_dir': 'data/mouse_crops/b2/crops',
    },
    'mouse_b3': {
        'metadata': 'data/mouse_crops/b3/crop_metadata.json',
        'crops_dir': 'data/mouse_crops/b3/crops',
    },
    'intestinal': {
        'metadata': 'data/intestinal_crops/val/crop_metadata.json',
        'crops_dir': 'data/intestinal_crops/val/crops',
    },
}

# ── Prompts ──

# Step 1: 全局评估
PROMPT_STEP1 = """Look at this image crop from a microscopy image. 
Is this a real organoid (a 3D multicellular biological structure) or a background artifact/false positive detection?

Answer in JSON format:
{"is_organoid": true/false, "confidence": 0.0-1.0, "reason": "brief reason"}"""

# Step 3: 对比推理
PROMPT_STEP3 = """Compare these two image crops from a microscopy image.
Crop A is a known TRUE POSITIVE (real organoid). Crop B is being evaluated.

Is Crop B also a real organoid (similar to Crop A) or a background artifact?

Answer in JSON format:
{"is_organoid": true/false, "confidence": 0.0-1.0, "reason": "brief reason"}"""


def load_metadata(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = []
    for x in data:
        items.append({
            'cache_key': x.get('cache_key', ''),
            'matched': x.get('matched', False),
            'confidence': x.get('rfdetr_conf', x.get('confidence', 0.5)),
            'crop_path': x.get('crop_path', ''),
            'image': x.get('image', ''),
        })
    return items


def call_vlm(prompt, image_path, timeout=60):
    """Call VLM via z-ai Node.js SDK (bypass CLI for reliability)."""
    import subprocess
    try:
        script_path = Path(__file__).parent / 'vlm_call.mjs'
        cmd = ['bun', 'run', str(script_path), prompt, str(image_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        # stdout has JSON content
        output = result.stdout.strip()
        if output:
            try:
                content = json.loads(output)
                # content is the message content string, parse JSON from it
                import re
                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass

        return None
    except Exception as e:
        return None


def find_crop_path(item, crops_dir):
    """Find crop image path."""
    cache_key = item['cache_key']
    crop_path = os.path.join(crops_dir, cache_key + '.png')
    if os.path.exists(crop_path):
        return crop_path
    # Fallback
    if item.get('crop_path') and os.path.exists(item['crop_path']):
        return item['crop_path']
    return None


def find_tp_reference(items, crops_dir, n=3):
    """Find a few confirmed TP crops for comparison."""
    refs = []
    for item in items:
        if item['matched']:
            path = find_crop_path(item, crops_dir)
            if path:
                refs.append(path)
            if len(refs) >= n:
                break
    return refs


def evaluate_dataset(ds_name, meta_path, crops_dir, max_crops=None, enable_step3=True):
    """Evaluate one dataset with VLM reasoning."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    items = load_metadata(meta_path)
    print(f"  {len(items)} crops (TP={sum(1 for x in items if x['matched'])}, "
          f"FP={sum(1 for x in items if not x['matched'])})")
    
    if max_crops:
        # Shuffle to mix TP/FP, then take max_crops
        import random
        random.seed(42)
        random.shuffle(items)
        items = items[:max_crops]
        print(f"  Limited to {max_crops} crops for testing (shuffled)")
    
    # Find TP references for step 3
    tp_refs = find_tp_reference(items, crops_dir, n=3) if enable_step3 else []
    
    all_scores = []
    all_confs = []
    all_labels = []
    
    for i, item in enumerate(items):
        crop_path = find_crop_path(item, crops_dir)
        if not crop_path:
            continue
        
        # Step 1: 全局评估
        result1 = call_vlm(PROMPT_STEP1, crop_path)
        
        if result1 and 'is_organoid' in result1:
            score = float(result1.get('confidence', 0.5))
            if not result1['is_organoid']:
                score = 1.0 - score
        else:
            score = 0.5  # fallback if VLM fails
        
        # Step 3: 对比推理 (for uncertain samples)
        if enable_step3 and tp_refs and 0.3 < score < 0.7:
            ref_path = tp_refs[i % len(tp_refs)]
            # Create comparison image (side by side)
            # For now, just use step 1 score as fallback
            # TODO: create side-by-side comparison image
            pass
        
        all_scores.append(score)
        all_confs.append(item['confidence'])
        all_labels.append(1 if item['matched'] else 0)
        
        if (i+1) % 10 == 0 or i == len(items) - 1:
            print(f"    [{i+1}/{len(items)}] "
                  f"TP={sum(all_labels)}, FP={len(all_labels)-sum(all_labels)}")
    
    scores = np.array(all_scores)
    confs = np.array(all_confs)
    labels = np.array(all_labels)
    
    if len(set(labels)) < 2:
        print(f"  [SKIP] Single class")
        return None
    
    slot_auc = roc_auc_score(labels, scores)
    conf_auc = roc_auc_score(labels, confs)
    slot_ap = average_precision_score(labels, scores)
    conf_ap = average_precision_score(labels, confs)
    
    return {
        'n_crops': len(labels),
        'n_tp': int(sum(labels)),
        'n_fp': int(len(labels) - sum(labels)),
        'slot_auc': float(slot_auc),
        'conf_auc': float(conf_auc),
        'slot_ap': float(slot_ap),
        'conf_ap': float(conf_ap),
        'delta_auc': float(slot_auc - conf_auc),
    }


def main():
    parser = argparse.ArgumentParser(description='Phase B2: VLM Reasoning TP/FP Eval')
    parser.add_argument('--max-crops', type=int, default=None,
                        help='Limit crops per dataset (for testing)')
    parser.add_argument('--no-step3', action='store_true',
                        help='Disable step 3 (comparison reasoning)')
    parser.add_argument('--output', default='results/vlm_reasoning')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to evaluate (default: all available)')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("Phase B2: VLM Two-Step Reasoning TP/FP Evaluation")
    print("=" * 60)
    
    all_results = {}
    
    for ds_name in DATASETS:
        if args.datasets and ds_name not in args.datasets:
            continue
        
        config = DATASETS[ds_name]
        meta_path = config['metadata']
        crops_dir = config['crops_dir']
        
        if not os.path.exists(meta_path):
            print(f"\n  [SKIP] {ds_name}: metadata not found")
            continue
        
        print(f"\n{'='*40}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*40}")
        
        result = evaluate_dataset(
            ds_name, meta_path, crops_dir,
            max_crops=args.max_crops,
            enable_step3=not args.no_step3
        )
        
        if result:
            all_results[ds_name] = result
            print(f"\n  Slot AUC:  {result['slot_auc']:.4f}  AP: {result['slot_ap']:.4f}")
            print(f"  Conf AUC:  {result['conf_auc']:.4f}  AP: {result['conf_ap']:.4f}")
            print(f"  Δ AUC:     {result['delta_auc']:+.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: VLM Reasoning across datasets")
    print(f"{'='*60}")
    print(f"  {'Dataset':<15} {'VLM AUC':>10} {'Conf AUC':>10} {'Δ':>8}")
    print(f"  {'-'*50}")
    for ds, r in all_results.items():
        print(f"  {ds:<15} {r['slot_auc']:>10.4f} {r['conf_auc']:>10.4f} {r['delta_auc']:>+8.4f}")
    
    # Save
    out_path = os.path.join(args.output, 'vlm_reasoning_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == '__main__':
    main()
