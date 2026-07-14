r"""
Intestinal organoid 跨域 crops 生成：YOLOv12 检测 → GT 匹配 → bbox crops + metadata

和 generate_mouse_crops.py 格式兼容，生成相同的 crop_metadata.json，
这样 cross_domain_eval.py 可以直接用 MultiOrg supcon best.pt 做跨域评估。

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl

    # 用 yolo12s intestinal checkpoint 生成 val set crops
    python scripts\mouse_liver\generate_intestinal_crops.py --weights runs\detect\train\weights\best.pt --data-root data\intestinal_organoid\OrganoidDataset --split val --dst data\intestinal_crops\val
"""
import argparse
import json
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def load_yolo_labels(label_path, img_w, img_h):
    """Load YOLO format labels and convert to [x1, y1, x2, y2] pixel coords.
    
    YOLO format: class x_center y_center width height (normalized)
    """
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            bboxes.append([x1, y1, x2, y2])
    return bboxes


def bbox_iou(box1, box2):
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-8)


def match_detections_to_gt(dets, gt_bboxes, iou_threshold=0.3):
    """Match detections to GT, return matched flags and IoUs."""
    matched_flags = [False] * len(dets)
    matched_ious = [0.0] * len(dets)
    gt_used = [False] * len(gt_bboxes)
    
    for di, det in enumerate(dets):
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_bboxes):
            if gt_used[gi]:
                continue
            iou = bbox_iou(det['bbox'], gt)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_threshold and best_gi >= 0:
            matched_flags[di] = True
            matched_ious[di] = best_iou
            gt_used[best_gi] = True
    
    return matched_flags, matched_ious


def crop_and_resize(img_arr, bbox, target_size=224):
    """Crop bbox region and resize to target_size × target_size."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img_arr.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    crop = img_arr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(
        description='Generate intestinal organoid crops for cross-domain slot eval'
    )
    parser.add_argument('--weights', required=True,
                        help='YOLOv12 checkpoint path (e.g. runs/detect/train/weights/best.pt)')
    parser.add_argument('--data-root', required=True,
                        help='Dataset root (contains val/images, val/labels)')
    parser.add_argument('--split', default='val', choices=['val', 'train'],
                        help='Dataset split to process')
    parser.add_argument('--dst', required=True,
                        help='Output directory for crops + metadata')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--imgsz', type=int, default=1088,
                        help='YOLO inference image size (intestinal trained at 1088)')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='Crop output size (224 for DINOv2)')
    args = parser.parse_args()
    
    os.makedirs(args.dst, exist_ok=True)
    crops_dir = Path(args.dst) / 'crops'
    crops_dir.mkdir(exist_ok=True)
    
    # Load YOLOv12
    from ultralytics import YOLO
    print(f"Loading YOLOv12 from {args.weights}...")
    model = YOLO(args.weights)
    
    # Find images
    img_dir = Path(args.data_root) / args.split / 'images'
    label_dir = Path(args.data_root) / args.split / 'labels'
    images = sorted(img_dir.glob('*.jpeg')) + sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
    print(f"Found {len(images)} images in {img_dir}")
    
    all_dets = []
    total_tp, total_fp = 0, 0
    
    for img_idx, img_path in enumerate(images):
        # YOLO inference
        results = model(str(img_path), conf=args.conf, verbose=False, imgsz=args.imgsz)
        result = results[0]
        
        det_bboxes = result.boxes.xyxy.cpu().numpy().tolist() if len(result.boxes) > 0 else []
        det_confs = result.boxes.conf.cpu().numpy().tolist() if len(result.boxes) > 0 else []
        
        # Load image for cropping
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert('RGB'))
        h, w = img_np.shape[:2]
        
        # Load GT (YOLO format)
        label_path = label_dir / (img_path.stem + '.txt')
        gt_bboxes = load_yolo_labels(str(label_path), w, h)
        
        # Match
        det_list = [{'bbox': bb, 'confidence': cf} for bb, cf in zip(det_bboxes, det_confs)]
        matched_flags, matched_ious = match_detections_to_gt(det_list, gt_bboxes)
        
        n_tp = sum(matched_flags)
        n_fp = len(matched_flags) - n_tp
        total_tp += n_tp
        total_fp += n_fp
        
        print(f"  [{img_idx+1}/{len(images)}] {img_path.name}: "
              f"det={len(det_list)} gt={len(gt_bboxes)} TP={n_tp} FP={n_fp}")
        
        # Save crops
        for di, det in enumerate(det_list):
            cache_key = f"intestinal_{img_path.stem}_{di}"
            crop = crop_and_resize(img_np, det['bbox'], args.crop_size)
            crop_path = crops_dir / f"{cache_key}.png"
            success, buf = cv2.imencode('.png', cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            if success:
                buf.tofile(str(crop_path))
            
            entry = {
                'cache_key': cache_key,
                'image': f"intestinal/{img_path.stem}",
                'det_idx': di,
                'bbox': det['bbox'],
                'rfdetr_conf': det['confidence'],  # keep field name for compatibility
                'matched': bool(matched_flags[di]),
                'match_iou': float(matched_ious[di]),
                'area': 0, 'circularity': 0, 'solidity': 0, 'aspect_ratio': 0,
                'crop_path': str(crop_path),
                'image_size': [w, h],
                'n_gt': len(gt_bboxes),
            }
            all_dets.append(entry)
        
        # Free memory
        del img_np, results, result
        import gc; gc.collect()
    
    meta_path = Path(args.dst) / 'crop_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_dets, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: intestinal_{args.split}")
    print(f"{'='*60}")
    print(f"  Images: {len(images)}")
    print(f"  Total detections: {len(all_dets)}")
    print(f"  TP: {total_tp}")
    print(f"  FP: {total_fp}")
    print(f"  TP rate: {total_tp/len(all_dets)*100:.1f}%" if all_dets else "  N/A")
    print(f"  Crops: {crops_dir}")
    print(f"  Metadata: {meta_path}")


if __name__ == '__main__':
    main()
