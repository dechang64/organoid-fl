r"""
鼠肝 SAM2 分割：RF-DETR 检测 → SAM2 分割 → 轮廓输出
把检测结果作为 SAM2 的 box prompt，输出像素级 mask，和人工红线标注对比

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate
    pip install segment-anything-2  (如果没装)

    python scripts\mouse_liver\sam2_segment.py --weights runs\mouse_liver_fewshot\checkpoint_best_regular.pth --src scripts\mouse_liver\yolo_format\images --gt scripts\mouse_liver\yolo_format\labels --dst results\sam2_segment --threshold 0.25
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
    model = build_sam2(checkpoint, device=device)
    return SAM2ImagePredictor(model)

def compute_morphology(mask, bbox):
    """从 mask 计算形态学指标"""
    area = int(mask.sum())
    if area == 0:
        return {"area": 0, "perimeter": 0, "circularity": 0, "solidity": 0, "aspect_ratio": 0, "eccentricity": 0}
    
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"area": area, "perimeter": 0, "circularity": 0, "solidity": 0, "aspect_ratio": 0, "eccentricity": 0}
    
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
        eccentricity = np.sqrt(1 - (minor / major) ** 2) if major > 0 else 0
    else:
        aspect_ratio = 1.0
        eccentricity = 0.0
    
    return {
        "area": area, "perimeter": float(perimeter), "circularity": float(circularity),
        "solidity": float(solidity), "aspect_ratio": float(aspect_ratio), "eccentricity": float(eccentricity),
    }

def mask_iou(mask1, mask2):
    """两个 mask 的 IoU"""
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0

def load_gt_mask(label_path, img_h, img_w):
    """从 YOLO 格式标注生成 mask（将 bbox 区域作为 GT mask 的近似）"""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, xc, yc, bw, bh = parts
                    xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
                    x1 = int((xc - bw/2) * img_w)
                    y1 = int((yc - bh/2) * img_h)
                    x2 = int((xc + bw/2) * img_w)
                    y2 = int((yc + bh/2) * img_h)
                    mask[y1:y2, x1:x2] = 255  # bbox 填充
    return mask

def load_gt_mask_from_annot(annot_img):
    """从人工红线标注图提取 mask"""
    r, g, b = annot_img[:,:,0].astype(int), annot_img[:,:,1].astype(int), annot_img[:,:,2].astype(int)
    red_mask = (r > 150) & (r - g > 50) & (r - b > 50)
    red_uint8 = (red_mask * 255).astype(np.uint8)
    # Fill contours
    contours, _ = cv2.findContours(red_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 500]
    mask = np.zeros_like(red_uint8)
    cv2.fillPoly(mask, valid, 255)
    return mask, valid

def main():
    parser = argparse.ArgumentParser(description='SAM2 segmentation on mouse liver organoids')
    parser.add_argument('--weights', required=True, help='RF-DETR checkpoint')
    parser.add_argument('--src', required=True, help='Image directory')
    parser.add_argument('--gt', default=None, help='YOLO label directory')
    parser.add_argument('--annot', default=None, help='Human annotation image directory (red line)')
    parser.add_argument('--dst', default='results/sam2_segment')
    parser.add_argument('--model-variant', default='small', choices=['nano', 'small', 'base'])
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--sam2-checkpoint', default='sam2_hiera_small.pt')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    os.makedirs(args.dst, exist_ok=True)
    
    # Load models
    print("Loading RF-DETR...")
    det_model = load_rfdetr(args.weights, args.model_variant)
    
    print("Loading SAM2...")
    try:
        sam2_predictor = load_sam2(args.sam2_checkpoint, args.device)
    except Exception as e:
        print(f"SAM2 load failed: {e}")
        print("Falling back to contour-based segmentation (no SAM2)")
        sam2_predictor = None
    
    img_dir = Path(args.src)
    images = sorted(img_dir.glob('*.jpg'))
    
    all_results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for i, img_path in enumerate(images):
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert('RGB'))
        h, w = img_np.shape[:2]
        
        # 1. RF-DETR detection
        dets = det_model.predict(img_pil, threshold=args.threshold)
        
        # 2. SAM2 segmentation
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
            # Fallback: use bbox as mask
            for di in range(len(dets.xyxy)):
                x1, y1, x2, y2 = dets.xyxy[di].astype(int)
                mask = np.zeros((h, w), dtype=bool)
                mask[y1:y2, x1:x2] = True
                morph = compute_morphology(mask, dets.xyxy[di].tolist())
                morph['confidence'] = float(dets.confidence[di])
                morph['mask'] = mask
                seg_results.append(morph)
        
        # 3. Load GT
        gt_bboxes = []
        if args.gt:
            lbl_path = Path(args.gt) / (img_path.stem + '.txt')
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls, xc, yc, bw, bh = parts
                            xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
                            x1 = (xc - bw/2) * w
                            y1 = (yc - bh/2) * h
                            x2 = (xc + bw/2) * w
                            y2 = (yc + bh/2) * h
                            gt_bboxes.append([x1, y1, x2, y2])
        
        # 4. Match detections to GT
        tp = 0
        matched = set()
        for seg in seg_results:
            best_iou, best_gi = 0, -1
            for gi, gt in enumerate(gt_bboxes):
                if gi in matched:
                    continue
                dx1, dy1, dx2, dy2 = seg['bbox']
                ix1, iy1 = max(dx1, gt[0]), max(dy1, gt[1])
                ix2, iy2 = min(dx2, gt[2]), min(dy2, gt[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1) * (iy2-iy1)
                    da = (dx2-dx1) * (dy2-dy1)
                    ga = (gt[2]-gt[0]) * (gt[3]-gt[1])
                    iou = inter / (da + ga - inter)
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
            if best_iou > 0.5 and best_gi >= 0:
                tp += 1
                matched.add(best_gi)
        fp = len(seg_results) - tp
        fn = len(gt_bboxes) - len(matched)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # 5. Visualization
        vis = img_np.copy()
        # Draw SAM2 masks (green filled)
        for seg in seg_results:
            mask = seg['mask']
            color = np.array([0, 255, 0], dtype=np.uint8)
            vis[mask] = vis[mask] * 0.5 + color * 0.5
            # Draw contour
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 200, 0), 3)
        
        # Draw GT (red)
        for gt in gt_bboxes:
            cv2.rectangle(vis, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (255, 0, 0), 3)
        
        cv2.putText(vis, f'Green=SAM2 mask  Red=GT bbox', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        vis_path = Path(args.dst) / f'seg_{img_path.name}'
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        # Also save contour-only version (like human annotation)
        contour_vis = img_np.copy()
        for seg in seg_results:
            mask_uint8 = seg['mask'].astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_vis, contours, -1, (255, 0, 0), 5)  # Blue contour like human red line
        
        contour_path = Path(args.dst) / f'contour_{img_path.name}'
        cv2.imwrite(str(contour_path), cv2.cvtColor(contour_vis, cv2.COLOR_RGB2BGR))
        
        # Per-image result
        img_result = {
            'image': img_path.name,
            'n_detections': len(seg_results),
            'n_gt': len(gt_bboxes),
            'tp': tp, 'fp': fp, 'fn': fn,
            'morphology': [{k: v for k, v in seg.items() if k != 'mask'} for seg in seg_results],
        }
        all_results.append(img_result)
        
        print(f"  [{i+1}/{len(images)}] {img_path.name}: det={len(seg_results)} gt={len(gt_bboxes)} TP={tp} FP={fp} FN={fn}")
    
    # Summary
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"SAM2 SEGMENTATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Images: {len(images)}")
    print(f"  Method: {'SAM2' if sam2_predictor else 'bbox fallback'}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision: {prec:.4f} ({prec*100:.1f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.1f}%)")
    print(f"  F1:        {f1:.4f} ({f1*100:.1f}%)")
    
    output = {
        'summary': {
            'n_images': len(images),
            'method': 'sam2' if sam2_predictor else 'bbox_fallback',
            'total_tp': total_tp, 'total_fp': total_fp, 'total_fn': total_fn,
            'precision': prec, 'recall': rec, 'f1': f1,
        },
        'per_image': all_results,
    }
    out_path = Path(args.dst) / 'sam2_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Report: {out_path}")
    print(f"\n  Visualization:")
    print(f"    seg_*.jpg: SAM2 mask (green) + GT bbox (red)")
    print(f"    contour_*.jpg: SAM2 contour only (like human annotation)")

if __name__ == '__main__':
    main()
