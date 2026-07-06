r"""
鼠肝新 test set 泛化评估：用之前 few-shot 微调的 RF-DETR + zero-shot SAM2
在新的 10 对图上评估，和之前 10 对图的结果对比

Pipeline（和之前完全一致，保证可比性）:
  1. 从标注图提取红色轮廓 → GT bboxes
  2. RF-DETR 检测（few-shot weights）
  3. SAM2 分割（zero-shot, box prompt → mask）
  4. IoU > 0.5 匹配 → TP/FP/FN
  5. 形态学分析（circularity/solidity/aspect_ratio/confidence）

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate
    python scripts\mouse_liver\eval_new_testset.py ^
        --weights runs\mouse_liver_fewshot\checkpoint_best_regular.pth ^
        --src "D:\path\to\new_orig_images" ^
        --annot "D:\path\to\new_annotated_images" ^
        --sam2-checkpoint sam2_hiera_small.pt ^
        --dst results\mouse_liver_new_testset
"""
import argparse
import os
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def load_rfdetr(weights_path, model_variant='small'):
    from rfdetr import RFDETRSmall, RFDETRNano, RFDETRBase
    model_map = {'nano': RFDETRNano, 'small': RFDETRSmall, 'base': RFDETRBase}
    ModelClass = model_map.get(model_variant, RFDETRSmall)
    return ModelClass(pretrain_weights=weights_path, num_classes=1)


def load_sam2(checkpoint='sam2_hiera_small.pt', device='cuda'):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import torch as _torch

    # Handle finetuned checkpoint format
    ckpt_data = _torch.load(checkpoint, map_location='cpu', weights_only=False)
    if isinstance(ckpt_data, dict) and 'model_state_dict' in ckpt_data:
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        tmp_ckpt = os.path.join(tmp_dir, 'model_finetuned.pt')
        _torch.save({'model': ckpt_data['model_state_dict']}, tmp_ckpt)
        checkpoint = tmp_ckpt

    last_err = None
    for cfg in ['sam2_hiera_s', 'sam2_hiera_small']:
        try:
            model = build_sam2(cfg, checkpoint, device=device)
            return SAM2ImagePredictor(model)
        except Exception as e:
            last_err = e
            print(f"    [SAM2] config '{cfg}' failed: {e}")
            continue
    raise RuntimeError(f"Could not load SAM2: {last_err}")


def extract_gt_from_annot(annot_img):
    """从人工红线标注图提取 GT bboxes 和 masks"""
    r, g, b = annot_img[:,:,0].astype(int), annot_img[:,:,1].astype(int), annot_img[:,:,2].astype(int)
    red_mask = (r > 150) & (r - g > 50) & (r - b > 50)
    red_uint8 = (red_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(red_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 500]

    gt_bboxes = []
    gt_masks = []
    full_mask = np.zeros(annot_img.shape[:2], dtype=np.uint8)
    for c in valid:
        x, y, w, h = cv2.boundingRect(c)
        gt_bboxes.append([x, y, x + w, y + h])
        mask = np.zeros(annot_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [c], 255)
        gt_masks.append(mask)
        cv2.fillPoly(full_mask, [c], 255)

    return gt_bboxes, gt_masks, full_mask


def compute_morphology(mask, bbox):
    area = int(mask.sum())
    if area == 0:
        return {"area": 0, "perimeter": 0, "circularity": 0, "solidity": 0,
                "aspect_ratio": 0, "bbox": bbox}

    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area": area, "perimeter": 0, "circularity": 0, "solidity": 0,
                "aspect_ratio": 0, "bbox": bbox}

    c = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    if len(c) >= 5:
        ellipse = cv2.fitEllipse(c)
        major, minor = ellipse[1]
        aspect_ratio = major / minor if minor > 0 else 0
    else:
        aspect_ratio = 1.0

    return {
        "area": area, "perimeter": float(perimeter),
        "circularity": float(circularity), "solidity": float(solidity),
        "aspect_ratio": float(aspect_ratio), "bbox": bbox,
    }


def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0


def bbox_iou(box1, box2):
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)


def main():
    parser = argparse.ArgumentParser(description='Mouse liver new test set evaluation')
    parser.add_argument('--weights', required=True, help='RF-DETR few-shot checkpoint')
    parser.add_argument('--src', required=True, help='Original images directory')
    parser.add_argument('--annot', required=True, help='Annotated images directory (red line)')
    parser.add_argument('--sam2-checkpoint', default='sam2_hiera_small.pt')
    parser.add_argument('--dst', default='results/mouse_liver_new_testset')
    parser.add_argument('--model-variant', default='small', choices=['nano', 'small', 'base'])
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-sam2', action='store_true', help='Skip SAM2, use bbox only')
    parser.add_argument('--resolution', type=int, default=None, help='RF-DETR inference resolution (default: model default)')
    parser.add_argument('--resize-to', type=int, nargs=2, default=None, metavar=('W', 'H'),
                        help='Resize images to WxH before processing (e.g. --resize-to 2592 1944)')
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    # Pair images by sorted order
    orig_files = sorted(f for f in os.listdir(args.src) if f.lower().endswith(('.jpg', '.png', '.jpeg')))
    annot_files = sorted(f for f in os.listdir(args.annot) if f.lower().endswith(('.jpg', '.png', '.jpeg')))

    if len(orig_files) != len(annot_files):
        print(f"[WARN] orig={len(orig_files)} vs annot={len(annot_files)} mismatch!")

    n_pairs = min(len(orig_files), len(annot_files))
    print(f"\n{'='*60}")
    print(f"Mouse Liver New Test Set Evaluation")
    print(f"{'='*60}")
    print(f"  Pairs: {n_pairs}")
    print(f"  RF-DETR weights: {args.weights}")
    print(f"  SAM2 checkpoint: {args.sam2_checkpoint}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Device: {args.device}")
    print()

    # Load models
    print("Loading RF-DETR...")
    det_model = load_rfdetr(args.weights, args.model_variant)

    sam2_predictor = None
    if not args.no_sam2:
        print("Loading SAM2...")
        try:
            sam2_predictor = load_sam2(args.sam2_checkpoint, args.device)
            print("  SAM2 loaded (zero-shot)")
        except Exception as e:
            print(f"  SAM2 load failed: {e}")
            print("  Falling back to bbox-only mode")

    # Process each pair
    all_results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    all_tp_morphs, all_fp_morphs = [], []  # for morphology analysis

    for i in range(n_pairs):
        orig_path = os.path.join(args.src, orig_files[i])
        annot_path = os.path.join(args.annot, annot_files[i])

        img_pil = Image.open(orig_path)
        img_np = np.array(img_pil.convert('RGB'))
        h, w = img_np.shape[:2]

        annot_np = np.array(Image.open(annot_path).convert('RGB'))

        # Resize if requested (e.g. --resize-to 2592 1944 to match training resolution)
        if args.resize_to:
            tgt_w, tgt_h = args.resize_to
            # Resize original
            img_pil = img_pil.resize((tgt_w, tgt_h), Image.LANCZOS)
            img_np = np.array(img_pil.convert('RGB'))
            h, w = img_np.shape[:2]
            # Resize annotation (keep red lines sharp)
            annot_pil = Image.fromarray(annot_np).resize((tgt_w, tgt_h), Image.LANCZOS)
            annot_np = np.array(annot_pil.convert('RGB'))

        # 1. Extract GT from annotation
        gt_bboxes, gt_masks, gt_full_mask = extract_gt_from_annot(annot_np)

        # 2. RF-DETR detection
        if args.resolution:
            dets = det_model.predict(img_pil, threshold=args.threshold, shape=(args.resolution, args.resolution))
        else:
            dets = det_model.predict(img_pil, threshold=args.threshold)

        # 3. SAM2 segmentation
        seg_results = []
        if sam2_predictor and len(dets.xyxy) > 0:
            sam2_predictor.set_image(img_np)
            for di in range(len(dets.xyxy)):
                box = dets.xyxy[di].astype(np.float32)
                masks, scores, _ = sam2_predictor.predict(box=box, multimask_output=False)
                mask = masks[0]
                morph = compute_morphology(mask, box.tolist())
                morph['confidence'] = float(dets.confidence[di])
                morph['mask'] = mask
                seg_results.append(morph)
        elif len(dets.xyxy) > 0:
            for di in range(len(dets.xyxy)):
                x1, y1, x2, y2 = dets.xyxy[di].astype(int)
                mask = np.zeros((h, w), dtype=bool)
                mask[y1:y2, x1:x2] = True
                morph = compute_morphology(mask, dets.xyxy[di].tolist())
                morph['confidence'] = float(dets.confidence[di])
                morph['mask'] = mask
                seg_results.append(morph)

        # 4. Match detections to GT (mask IoU > 0.5)
        tp = 0
        matched = set()
        tp_morphs = []
        fp_morphs = []

        for seg in seg_results:
            best_iou, best_gi = 0, -1
            for gi, gt_mask in enumerate(gt_masks):
                if gi in matched:
                    continue
                # Use mask IoU (SAM2 mask vs GT contour mask)
                iou = mask_iou(seg['mask'], gt_mask)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou > 0.5 and best_gi >= 0:
                tp += 1
                matched.add(best_gi)
                tp_morphs.append({k: v for k, v in seg.items() if k != 'mask'})
            else:
                fp_morphs.append({k: v for k, v in seg.items() if k != 'mask'})

        fp = len(seg_results) - tp
        fn = len(gt_bboxes) - len(matched)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_tp_morphs.extend(tp_morphs)
        all_fp_morphs.extend(fp_morphs)

        # 5. Visualization
        vis = img_np.copy()
        # SAM2 masks (green)
        for seg in seg_results:
            mask = seg['mask'].astype(bool)
            color = np.array([0, 255, 0], dtype=np.uint8)
            vis[mask] = (vis[mask] * 0.5 + color * 0.5).astype(np.uint8)
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 200, 0), 3)
        # GT (red)
        for gt in gt_bboxes:
            cv2.rectangle(vis, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (255, 0, 0), 3)
        cv2.putText(vis, f'Green=pred  Red=GT  TP={tp} FP={fp} FN={fn}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        vis_path = os.path.join(args.dst, f'vis_{i:02d}.jpg')
        cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        per_img = {
            'index': i,
            'orig_file': orig_files[i],
            'annot_file': annot_files[i],
            'image_size': [w, h],
            'n_gt': len(gt_bboxes),
            'n_det': len(seg_results),
            'tp': tp, 'fp': fp, 'fn': fn,
        }
        all_results.append(per_img)
        print(f"  [{i+1}/{n_pairs}] {orig_files[i][:30]}...: gt={len(gt_bboxes)} det={len(seg_results)} TP={tp} FP={fp} FN={fn}")

    # Summary
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\n{'='*60}")
    print(f"NEW TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Images: {n_pairs}")
    print(f"  Method: {'RF-DETR + SAM2' if sam2_predictor else 'RF-DETR only'}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision: {prec:.4f} ({prec*100:.1f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.1f}%)")
    print(f"  F1:        {f1:.4f} ({f1*100:.1f}%)")

    # Morphology analysis: can we distinguish TP from FP?
    print(f"\n{'='*60}")
    print(f"MORPHOLOGY ANALYSIS (TP vs FP)")
    print(f"{'='*60}")
    if all_tp_morphs and all_fp_morphs:
        for feat in ['confidence', 'circularity', 'solidity', 'aspect_ratio']:
            tp_vals = [m.get(feat, 0) for m in all_tp_morphs]
            fp_vals = [m.get(feat, 0) for m in all_fp_morphs]
            tp_mean = np.mean(tp_vals) if tp_vals else 0
            fp_mean = np.mean(fp_vals) if fp_vals else 0
            print(f"  {feat:15s}: TP={tp_mean:.3f} (n={len(tp_vals)})  FP={fp_mean:.3f} (n={len(fp_vals)})  "
                  f"diff={tp_mean - fp_mean:+.3f}")
    else:
        print(f"  TP={len(all_tp_morphs)}  FP={len(all_fp_morphs)} — not enough for analysis")

    # Compare with previous results
    print(f"\n{'='*60}")
    print(f"COMPARISON WITH PREVIOUS RESULTS")
    print(f"{'='*60}")
    print(f"  Previous (10 pairs, 2592x1944): F1=93.9%  TP=23 FP=3 FN=0")
    print(f"  New      (10 pairs, 4000x3000): F1={f1*100:.1f}%  TP={total_tp} FP={total_fp} FN={total_fn}")

    # Save JSON
    output = {
        'summary': {
            'n_images': n_pairs,
            'method': 'rf_detr_fewshot + sam2_zeroshot' if sam2_predictor else 'rf_detr_fewshot',
            'weights': args.weights,
            'sam2_checkpoint': args.sam2_checkpoint,
            'threshold': args.threshold,
            'total_tp': total_tp, 'total_fp': total_fp, 'total_fn': total_fn,
            'precision': prec, 'recall': rec, 'f1': f1,
        },
        'per_image': all_results,
        'morphology_tp': all_tp_morphs,
        'morphology_fp': all_fp_morphs,
    }
    out_path = os.path.join(args.dst, 'new_testset_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Report: {out_path}")
    print(f"  Vis:    {args.dst}/vis_*.jpg")


if __name__ == '__main__':
    main()
