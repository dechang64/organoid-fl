r"""
Phase A4: CLIP Zero-Shot TP/FP 评估

用 CLIP 文本-视觉对齐空间做 zero-shot 评估：
- 对每个 crop，用 CLIP 编码图像和两组文本 prompt（"a real organoid" vs "background artifact"）
- 比较相似度，AUC > 0.50 说明 CLIP 语义空间能区分 TP/FP
- 不需要训练，直接评估，1 天完成

如果 AUC > 0.50 → CLIP 特征有跨域信号 → 推进 A1（CLIP + SupCon 联合训练）
如果 AUC < 0.50 → CLIP 也无法区分 → 转向 B2（VLM 两步推理）

Usage:
    cd /home/z/my-project/organoid-fl
    python scripts/multiorg/clip_zeroshot_eval.py

    # 冬生本地 GPU:
    python scripts\multiorg\clip_zeroshot_eval.py --device cuda:0
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image

# ── CLIP prompt sets ──

# 5 组 prompt 对，从简单到专业
PROMPT_PAIRS = [
    {
        'name': 'basic',
        'tp': 'a real organoid',
        'fp': 'a background artifact',
    },
    {
        'name': 'descriptive',
        'tp': 'a round organoid structure with clear boundaries',
        'fp': 'a blurry background region without clear structure',
    },
    {
        'name': 'medical',
        'tp': 'a bright-field microscopy image of an organoid',
        'fp': 'a bright-field microscopy image of background debris',
    },
    {
        'name': 'morphological',
        'tp': 'a circular organoid with smooth membrane',
        'fp': 'an irregular non-organoid structure',
    },
    {
        'name': 'contextual',
        'tp': 'an organoid cell cluster in a culture dish',
        'fp': 'an empty region in a culture dish',
    },
]

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
    # Intestinal too large for cloud VM, add flag to skip
}


def load_metadata(meta_path):
    """Load metadata and return list of {cache_key, matched, confidence}."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items = []
    for d in data:
        items.append({
            'cache_key': d.get('cache_key', ''),
            'matched': d.get('matched', False),
            'confidence': d.get('rfdetr_conf', d.get('confidence', 0.5)),
            'crop_path': d.get('crop_path', ''),
        })
    return items


def encode_images_and_text(model, preprocess, tokenizer, device, crops_dir, items, prompts, batch_size=32):
    """Encode all crops and text prompts with CLIP."""
    import torch

    # Encode text prompts (once)
    tokenized = tokenizer(prompts)
    tokenized = tokenized.to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Encode images in batches
    all_image_features = []
    labels = []
    confs = []

    valid_items = []
    for item in items:
        crop_path = os.path.join(crops_dir, item['cache_key'] + '.png')
        if not os.path.exists(crop_path) and item.get('crop_path'):
            crop_path = item['crop_path']
        if not os.path.exists(crop_path):
            continue
        valid_items.append((item, crop_path))

    print(f"  {len(valid_items)}/{len(items)} crops found")

    for i in range(0, len(valid_items), batch_size):
        batch_items = valid_items[i:i+batch_size]
        images = []
        for item, crop_path in batch_items:
            img = Image.open(crop_path).convert('RGB')
            images.append(preprocess(img))

        image_tensor = torch.stack(images).to(device)

        with torch.no_grad():
            img_features = model.encode_image(image_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            all_image_features.append(img_features.cpu())
            labels.extend([1 if item['matched'] else 0 for item, _ in batch_items])
            confs.extend([item['confidence'] for item, _ in batch_items])

        if (i // batch_size) % 10 == 0:
            print(f"    {i+len(batch_items)}/{len(valid_items)}")

    image_features = torch.cat(all_image_features, dim=0)
    return image_features, text_features, np.array(labels), np.array(confs)


def evaluate_prompt_pair(image_features, text_features, labels, confs, tp_idx, fp_idx):
    """Evaluate one prompt pair: compute similarity and AUC."""
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Ensure both on same device (image_features is CPU, text_features is GPU)
    text_feat = text_features.cpu()

    # Cosine similarity to TP and FP text
    sim_tp = (image_features @ text_feat[tp_idx]).numpy()
    sim_fp = (image_features @ text_feat[fp_idx]).numpy()

    # Slot score = sim_tp - sim_fp (higher = more likely TP)
    slot_scores = sim_tp - sim_fp

    if len(set(labels)) < 2:
        return None

    slot_auc = roc_auc_score(labels, slot_scores)
    slot_ap = average_precision_score(labels, slot_scores)
    conf_auc = roc_auc_score(labels, confs)
    conf_ap = average_precision_score(labels, confs)

    return {
        'slot_auc': float(slot_auc),
        'slot_ap': float(slot_ap),
        'conf_auc': float(conf_auc),
        'conf_ap': float(conf_ap),
        'delta_auc': float(slot_auc - conf_auc),
    }


def main():
    parser = argparse.ArgumentParser(description='CLIP Zero-Shot TP/FP Evaluation')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model', default='ViT-B-16', help='CLIP model architecture')
    parser.add_argument('--pretrained', default='openai', help='Pretrained weights')
    parser.add_argument('--output', default='results/clip_zeroshot')
    parser.add_argument('--include-intestinal', action='store_true',
                        help='Include intestinal dataset (2744 crops, slow on CPU)')
    args = parser.parse_args()

    import torch
    import open_clip

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  CLIP Zero-Shot TP/FP Evaluation")
    print(f"  Model: {args.model} ({args.pretrained})")
    print(f"  Device: {device}")
    print("=" * 60)

    # Load CLIP
    print(f"\nLoading CLIP {args.model}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    print(f"  Loaded")

    # All text prompts (flat list: [tp1, fp1, tp2, fp2, ...])
    all_prompts = []
    prompt_indices = []
    for pair in PROMPT_PAIRS:
        tp_idx = len(all_prompts)
        all_prompts.append(pair['tp'])
        fp_idx = len(all_prompts)
        all_prompts.append(pair['fp'])
        prompt_indices.append((pair['name'], tp_idx, fp_idx))

    # Datasets to evaluate
    datasets = dict(DATASETS)
    if args.include_intestinal:
        datasets['intestinal'] = {
            'metadata': 'data/intestinal_crops/val/crop_metadata.json',
            'crops_dir': 'data/intestinal_crops/val/crops',
        }

    all_results = {}

    for ds_name, ds_config in datasets.items():
        meta_path = ds_config['metadata']
        crops_dir = ds_config['crops_dir']

        if not os.path.exists(meta_path):
            print(f"\n  [{ds_name}] Skip — metadata not found: {meta_path}")
            continue

        print(f"\n{'='*40}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*40}")

        items = load_metadata(meta_path)
        print(f"  {len(items)} crops (TP={sum(1 for x in items if x['matched'])}, "
              f"FP={sum(1 for x in items if not x['matched'])})")

        if len(items) == 0:
            print(f"  [SKIP] No crops found for {ds_name}")
            continue

        image_features, text_features, labels, confs = encode_images_and_text(
            model, preprocess, tokenizer, device, crops_dir, items, all_prompts
        )

        print(f"\n  Results per prompt pair:")
        print(f"  {'Prompt':<20} {'Slot AUC':>10} {'Slot AP':>10} {'Conf AUC':>10} {'Δ AUC':>8}")
        print(f"  {'-'*60}")

        ds_results = {}
        for name, tp_idx, fp_idx in prompt_indices:
            r = evaluate_prompt_pair(image_features, text_features, labels, confs, tp_idx, fp_idx)
            if r is None:
                print(f"  {name:<20}  N/A (single class)")
                continue
            ds_results[name] = r
            print(f"  {name:<20} {r['slot_auc']:>10.4f} {r['slot_ap']:>10.4f} "
                  f"{r['conf_auc']:>10.4f} {r['delta_auc']:>+8.4f}")

        # Best prompt
        best = max(ds_results.values(), key=lambda x: x['slot_auc'])
        best_name = [k for k, v in ds_results.items() if v == best][0]
        print(f"\n  Best prompt: {best_name} (AUC={best['slot_auc']:.4f})")
        print(f"  vs Conf AUC: {best['conf_auc']:.4f} (Δ={best['delta_auc']:+.4f})")

        all_results[ds_name] = {
            'n_crops': len(labels),
            'n_tp': int(labels.sum()),
            'n_fp': int(len(labels) - labels.sum()),
            'best_prompt': best_name,
            'best_slot_auc': best['slot_auc'],
            'conf_auc': best['conf_auc'],
            'all_prompts': ds_results,
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary: CLIP Zero-Shot across datasets")
    print(f"{'='*60}")
    print(f"  {'Dataset':<15} {'Best Prompt':<20} {'Slot AUC':>10} {'Conf AUC':>10} {'Δ':>8}")
    print(f"  {'-'*65}")

    for ds_name, r in all_results.items():
        print(f"  {ds_name:<15} {r['best_prompt']:<20} {r['best_slot_auc']:>10.4f} "
              f"{r['conf_auc']:>10.4f} {r['best_slot_auc']-r['conf_auc']:>+8.4f}")

    # Decision
    best_aucs = [r['best_slot_auc'] for r in all_results.values()]
    avg_clip = np.mean(best_aucs)
    print(f"\n  Average CLIP AUC: {avg_clip:.4f}")
    if avg_clip > 0.60:
        print("  → CLIP has signal! Proceed to A1 (CLIP + SupCon joint training)")
    elif avg_clip > 0.50:
        print("  → CLIP has weak signal. Consider A1 with prompt tuning")
    else:
        print("  → CLIP zero-shot fails. Turn to B2 (VLM two-step reasoning)")

    # Save
    out_path = os.path.join(args.output, 'clip_zeroshot_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == '__main__':
    main()
