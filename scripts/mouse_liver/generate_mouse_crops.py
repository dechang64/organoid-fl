"""
鼠肝跨域 crops 生成：RF-DETR 检测 → GT 匹配 → bbox crops + metadata

生成和 MultiOrg ctm_metadata.json 格式兼容的 metadata，这样
slot_supcon/slot_c4_eval 可以直接用 MultiOrg best.pt 做跨域评估。

Usage (冬生本地):
    cd C:\\Users\\decha\\organoid-fl
    
    # B1 (10 张, 2592×1944)
    python scripts\\mouse_liver\\generate_mouse_crops.py
        --batch b1
        --weights runs\\mouse_liver_v2\\b1\\full\\checkpoint_best_regular.pth
        --src mouse_liver_data_correct\\batch1\\images
        --annot mouse_liver_data_correct\\batch1\\annotated
        --annotations mouse_liver_data_correct\\batch1\\annotations.json
        --dst data\\mouse_crops\\b1
        --resolution 544
    
    # B2 (10 张, 4000×3000)
    python scripts\\mouse_liver\\generate_mouse_crops.py
        --batch b2
        --weights runs\\mouse_liver_v2\\b2\\full\\checkpoint_best_regular.pth
        --src mouse_liver_data_correct\\batch2\\images
        --annot mouse_liver_data_correct\\batch2\\annotated
        --annotations mouse_liver_data_correct\\batch2\\annotations.json
        --dst data\\mouse_crops\\b2
        --resolution 768
    
    # B3 (20 张, 4000×3000)
    python scripts\\mouse_liver\\generate_mouse_crops.py
        --batch b3
        --weights runs\\mouse_liver_v2\\b3\\full\\checkpoint_best_regular.pth
        --src mouse_liver_data_correct\\batch3\\images
        --annot mouse_liver_data_correct\\batch3\\annotated
        --annotations mouse_liver_data_correct\\batch3\\annotations.json
        --dst data\\mouse_crops\\b3
        --resolution 768

Output:
    {dst}/crops/
        {batch}_{image}_{det_idx}.png    # bbox crop (224×224 ready for DINOv2)
    {dst}/crop_metadata.json             # 和 MultiOrg 格式兼容
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_rfdetr(weights_path, model_variant='small'):
    """Load RF-DETR model."""
    from rfdetr import RFDETRSmall, RFDETRNano, RFDETRBase
    model_map = {'nano': RFDETRNano, 'small': RFDETRSmall, 'base': RFDETRBase}
    ModelClass = model_map.get(model_variant, RFDETRSmall)
    return ModelClass(pretrain_weights=weights_path, num_classes=1)


def load_gt_bboxes(annotations_path, image_name):
    """Load GT bboxes from annotations.json for a specific image.
    
    Returns list of [x1, y1, x2, y2] in pixel coords.
    """
    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for entry in data:
        if entry['image'] == image_name:
            bboxes = []
            for bb in entry['bboxes']:
                x, y, w, h = bb['x'], bb['y'], bb['w'], bb['h']
                bboxes.append([x, y, x + w, y + h])
            return bboxes
    return []


def bbox_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)


def match_detections_to_gt(dets, gt_bboxes, iou_threshold=0.5):
    """Match RF-DETR detections to GT bboxes.
    
    Returns list of (matched: bool, match_iou: float) for each detection.
    """
    matched_flags = [False] * len(dets)
    matched_ious = [0.0] * len(dets)
    gt_used = [False] * len(gt_bboxes)
    
    # Sort detections by confidence (descending) for greedy matching
    det_order = sorted(range(len(dets)), key=lambda i: -dets[i]['confidence'])
    
    for di in det_order:
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_bboxes):
            if gt_used[gi]:
                continue
            iou = bbox_iou(dets[di]['bbox'], gt)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_threshold and best_gi >= 0:
            matched_flags[di] = True
            matched_ious[di] = best_iou
            gt_used[best_gi] = True
    
    return matched_flags, matched_ious


def crop_and_resize(img_arr, bbox, target_size=224):
    """Crop bbox region from image and resize to target_size×target_size.
    
    Args:
        img_arr: numpy array [H, W, 3] uint8
        bbox: [x1, y1, x2, y2] in pixel coords
        target_size: output size (224 for DINOv2)
    
    Returns:
        crop: [target_size, target_size, 3] uint8
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_arr.shape[1], x2)
    y2 = min(img_arr.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        # Invalid bbox, return black image
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    crop = img_arr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Resize to 224×224 (DINOv2 input size)
    crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return crop


def compute_morphology(crop_mask=None, bbox=None):
    """Compute morphology features (placeholder, not used in cross-domain eval)."""
    return {
        'area': 0, 'perimeter': 0, 'circularity': 0,
        'solidity': 0, 'aspect_ratio': 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate mouse liver crops for cross-domain slot eval'
    )
    parser.add_argument('--batch', required=True, choices=['b1', 'b2', 'b3'],
                        help='Batch identifier')
    parser.add_argument('--weights', required=True,
                        help='RF-DETR checkpoint path')
    parser.add_argument('--src', required=True,
                        help='Source image directory')
    parser.add_argument('--annotations', required=True,
                        help='annotations.json path (GT bboxes)')
    parser.add_argument('--dst', required=True,
                        help='Output directory')
    parser.add_argument('--resolution', type=int, default=None,
                        help='RF-DETR inference resolution (e.g. 544 for B1, 768 for B2/B3)')
    parser.add_argument('--threshold', type=float, default=0.25,
                        help='RF-DETR detection threshold')
    parser.add_argument('--model-variant', default='small',
                        choices=['nano', 'small', 'base'])
    parser.add_argument('--crop-size', type=int, default=224,
                        help='Crop output size (224 for DINOv2)')
    args = parser.parse_args()
    
    os.makedirs(args.dst, exist_ok=True)
    crops_dir = Path(args.dst) / 'crops'
    crops_dir.mkdir(exist_ok=True)
    
    # Load RF-DETR
    print(f"Loading RF-DETR ({args.model_variant}) from {args.weights}...")
    det_model = load_rfdetr(args.weights, args.model_variant)
    
    # Load annotations
    with open(args.annotations, 'r', encoding='utf-8') as f:
        annot_data = json.load(f)
    print(f"Annotations: {len(annot_data)} images")
    
    # Process each image
    img_dir = Path(args.src)
    images = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
    print(f"Found {len(images)} images in {args.src}")
    
    all_dets = []
    total_tp, total_fp = 0, 0
    
    for img_idx, img_path in enumerate(images):
        # Load image
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert('RGB'))
        h, w = img_np.shape[:2]
        
        # RF-DETR detection
        if args.resolution:
            dets = det_model.predict(
                img_pil, threshold=args.threshold,
                shape=(args.resolution, args.resolution)
            )
        else:
            dets = det_model.predict(img_pil, threshold=args.threshold)
        
        # Get detection bboxes and confidences
        det_bboxes = dets.xyxy.tolist() if hasattr(dets, 'xyxy') else []
        det_confs = dets.confidence.tolist() if hasattr(dets, 'confidence') else []
        
        # Load GT
        gt_bboxes = load_gt_bboxes(args.annotations, img_path.name)
        
        # Match detections to GT
        det_list = [
            {'bbox': bb, 'confidence': cf}
            for bb, cf in zip(det_bboxes, det_confs)
        ]
        matched_flags, matched_ious = match_detections_to_gt(det_list, gt_bboxes)
        
        n_tp = sum(matched_flags)
        n_fp = len(matched_flags) - n_tp
        total_tp += n_tp
        total_fp += n_fp
        
        print(f"  [{img_idx+1}/{len(images)}] {img_path.name}: "
              f"det={len(det_list)} gt={len(gt_bboxes)} TP={n_tp} FP={n_fp}")
        
        # Save crops
        for di, det in enumerate(det_list):
            cache_key = f"{args.batch}_{img_path.stem}_{di}"
            crop = crop_and_resize(img_np, det['bbox'], args.crop_size)
            crop_path = crops_dir / f"{cache_key}.png"
            cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            
            # Metadata entry (compatible with MultiOrg ctm_metadata.json)
            entry = {
                'cache_key': cache_key,
                'image': f"{args.batch}/{img_path.stem}",
                'det_idx': di,
                'bbox': det['bbox'],
                'rfdetr_conf': det['confidence'],
                'matched': bool(matched_flags[di]),
                'match_iou': float(matched_ious[di]),
                'area': 0,  # not computed for cross-domain
                'circularity': 0,
                'solidity': 0,
                'aspect_ratio': 0,
                'crop_path': str(crop_path),
                'image_size': [w, h],
                'n_gt': len(gt_bboxes),
            }
            all_dets.append(entry)
    
    # Save metadata
    meta_path = Path(args.dst) / 'crop_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_dets, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.batch}")
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
