r"""
零样本检测：用 MultiOrg 训练的 RF-DETR 模型直接检测鼠肝类器官
不需要训练，直接推理看效果

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate
    python scripts\mouse_liver\zeroshot_detect.py --weights output\checkpoint_best_regular.pth --model-variant small --src scripts\mouse_liver\yolo_format\images --gt scripts\mouse_liver\yolo_format\labels --dst results\zeroshot --sahi --windows 640

    # 直接推理（不切片）
    python scripts\mouse_liver\zeroshot_detect.py --weights output\checkpoint_best_regular.pth --model-variant small --src scripts\mouse_liver\yolo_format\images --gt scripts\mouse_liver\yolo_format\labels --dst results\zeroshot_direct --threshold 0.5
"""
import argparse
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

def load_model(weights_path, model_variant='small'):
    from rfdetr import RFDETRSmall, RFDETRNano, RFDETRBase
    model_map = {
        'nano': RFDETRNano,
        'small': RFDETRSmall,
        'base': RFDETRBase,
    }
    ModelClass = model_map.get(model_variant, RFDETRSmall)
    model = ModelClass(pretrain_weights=weights_path, num_classes=1)
    return model

def detect_single(model, img_path, threshold=0.5):
    """单图直接推理"""
    import supervision as sv
    from PIL import Image
    img = Image.open(img_path)
    detections = model.predict(img, threshold=threshold)
    return detections

def detect_sahi(model, img_path, window=640, overlap=0.3, conf=0.25, merge='soft_nms', sf=0.3):
    """SAHI 切片推理"""
    import supervision as sv
    from PIL import Image
    import cv2
    
    img = np.array(Image.open(img_path).convert('RGB'))
    h, w = img.shape[:2]
    
    # Sliding window
    stride = int(window * (1 - overlap))
    all_dets = []
    
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(x0 + window, w)
            y1 = min(y0 + window, h)
            patch = Image.fromarray(img[y0:y1, x0:x1])
            
            dets = model.predict(patch, threshold=conf)
            
            # Offset detections to original image coordinates
            if len(dets.xyxy) > 0:
                dets.xyxy[:, 0] += x0
                dets.xyxy[:, 1] += y0
                dets.xyxy[:, 2] += x0
                dets.xyxy[:, 3] += y0
                all_dets.append(dets)
    
    # Merge detections manually (avoid sv.Detections.merge metadata conflict)
    if not all_dets:
        return sv.Detections.empty()
    
    all_xyxy = np.concatenate([d.xyxy for d in all_dets], axis=0)
    all_conf = np.concatenate([d.confidence for d in all_dets], axis=0)
    all_cls = np.concatenate([d.class_id for d in all_dets], axis=0)
    
    merged = sv.Detections(xyxy=all_xyxy, confidence=all_conf, class_id=all_cls)
    
    # Apply NMS or Soft-NMS
    if merge == 'nms':
        merged = sv.Detections(
            xyxy=merged.xyxy,
            confidence=merged.confidence,
            class_id=merged.class_id,
        )
        merged = sv.NMS(threshold=0.5).apply(merged)
    elif merge == 'soft_nms':
        # Simple soft-NMS: reduce confidence for overlapping boxes
        boxes = merged.xyxy
        scores = merged.confidence.copy()
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j:
                    continue
                xi1, yi1, xi2, yi2 = boxes[i]
                xj1, yj1, xj2, yj2 = boxes[j]
                xi_area = (xi2-xi1) * (yi2-yi1)
                xj_area = (xj2-xj1) * (yj2-yj1)
                inter_x1 = max(xi1, xj1)
                inter_y1 = max(yi1, yj1)
                inter_x2 = min(xi2, xj2)
                inter_y2 = min(yi2, yj2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter = (inter_x2-inter_x1) * (inter_y2-inter_y1)
                    iou = inter / (xi_area + xj_area - inter)
                    if iou > 0.3:
                        scores[j] *= (1 - iou)
        merged = sv.Detections(
            xyxy=merged.xyxy,
            confidence=scores,
            class_id=merged.class_id,
        )
    
    # Score filter
    if sf > 0:
        keep = merged.confidence >= sf
        merged = merged[keep]
    
    return merged

def evaluate(detections, gt_path, img_w, img_h):
    """评估检测结果与 GT 的匹配"""
    # Load GT (YOLO format)
    bboxes = []
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, xc, yc, w, h = parts
                    xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                    x1 = (xc - w/2) * img_w
                    y1 = (yc - h/2) * img_h
                    x2 = (xc + w/2) * img_w
                    y2 = (yc + h/2) * img_h
                    bboxes.append([x1, y1, x2, y2])
    return bboxes

def main():
    parser = argparse.ArgumentParser(description='Zero-shot detection on mouse liver organoids')
    parser.add_argument('--weights', required=True, help='RF-DETR checkpoint path')
    parser.add_argument('--src', required=True, help='Source image directory')
    parser.add_argument('--dst', default='./results/zeroshot', help='Output directory')
    parser.add_argument('--model-variant', default='small', choices=['nano', 'small', 'base'])
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--gt', default=None, help='GT label directory (YOLO format)')
    parser.add_argument('--sahi', action='store_true', help='Use SAHI sliding window')
    parser.add_argument('--windows', type=int, nargs='+', default=[640])
    parser.add_argument('--overlap', type=float, default=0.3)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--merge', default='soft_nms', choices=['nms', 'soft_nms'])
    parser.add_argument('--score-filter', type=float, default=0.3)
    args = parser.parse_args()
    
    os.makedirs(args.dst, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model_variant} from {args.weights}")
    model = load_model(args.weights, args.model_variant)
    
    # Get images
    img_dir = Path(args.src)
    images = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
    print(f"Found {len(images)} images")
    
    results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for i, img_path in enumerate(images):
        from PIL import Image
        img = Image.open(img_path)
        w, h = img.size
        
        if args.sahi:
            dets = detect_sahi(model, img_path, args.windows[0], args.overlap, 
                              args.conf, args.merge, args.score_filter)
        else:
            dets = detect_single(model, img_path, args.threshold)
        
        n_det = len(dets.xyxy)
        
        # Load GT if available
        gt_path = None
        if args.gt:
            gt_name = img_path.stem
            for ext in ['.txt']:
                candidate = Path(args.gt) / f'{gt_name}{ext}'
                if candidate.exists():
                    gt_path = candidate
                    break
        
        gt_bboxes = []
        if gt_path:
            gt_bboxes = evaluate(dets, gt_path, w, h)
        
        # Match detections to GT (IoU > 0.5)
        tp = 0
        fp = 0
        matched = set()
        if len(dets.xyxy) > 0 and gt_bboxes:
            for det_idx in range(len(dets.xyxy)):
                dx1, dy1, dx2, dy2 = dets.xyxy[det_idx]
                best_iou = 0
                best_gt = -1
                for gt_idx, (gx1, gy1, gx2, gy2) in enumerate(gt_bboxes):
                    if gt_idx in matched:
                        continue
                    inter_x1 = max(dx1, gx1)
                    inter_y1 = max(dy1, gy1)
                    inter_x2 = min(dx2, gx2)
                    inter_y2 = min(dy2, gy2)
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter = (inter_x2-inter_x1) * (inter_y2-inter_y1)
                        det_area = (dx2-dx1) * (dy2-dy1)
                        gt_area = (gx2-gx1) * (gy2-gy1)
                        iou = inter / (det_area + gt_area - inter)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = gt_idx
                if best_iou > 0.5 and best_gt >= 0:
                    tp += 1
                    matched.add(best_gt)
                else:
                    fp += 1
            fn = len(gt_bboxes) - len(matched)
        else:
            fp = n_det
            fn = len(gt_bboxes)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"  [{i+1}/{len(images)}] {img_path.name}: det={n_det} gt={len(gt_bboxes)} TP={tp} FP={fp} FN={fn} P={prec:.2f} R={rec:.2f}")
        
        results.append({
            'image': img_path.name,
            'n_detections': n_det,
            'n_gt': len(gt_bboxes),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': prec, 'recall': rec,
        })
    
    # Summary
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    summary = {
        'n_images': len(images),
        'total_tp': total_tp, 'total_fp': total_fp, 'total_fn': total_fn,
        'precision': prec, 'recall': rec, 'f1': f1,
        'method': 'sahi' if args.sahi else 'direct',
        'model': args.model_variant,
        'threshold': args.threshold if not args.sahi else args.conf,
        'score_filter': args.score_filter if args.sahi else 0,
    }
    
    print(f"\n{'='*60}")
    print(f"ZERO-SHOT DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Images: {len(images)}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision: {prec:.4f} ({prec*100:.1f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.1f}%)")
    print(f"  F1:        {f1:.4f} ({f1*100:.1f}%)")
    print(f"  Method: {'SAHI' if args.sahi else 'Direct inference'}")
    
    # Save results
    output = {'summary': summary, 'per_image': results}
    out_path = Path(args.dst) / 'zeroshot_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Report: {out_path}")

if __name__ == '__main__':
    main()
