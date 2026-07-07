"""
传统CV基线：阈值分割 + 连通域分析 + 形态学过滤
不需要 GPU，不需要训练，直接在云 VM 上跑

Usage:
    python traditional_cv.py --src /path/to/小鼠-肝-4X --gt /path/to/yolo_format/labels --dst ./results/traditional
"""
import argparse
import os
import json
import cv2


def cv2_imread_unicode(path):
    """cv2.imread 在 Windows 上不支持中文路径, 用 imdecode + fromfile 替代"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

import numpy as np
from pathlib import Path
from PIL import Image

def detect_organoids(img_path, min_area=5000, max_area=500000):
    """传统CV检测：阈值分割 + 形态学 + 连通域"""
    img = cv2_imread_unicode(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 类器官比背景暗（mean~178，类器官更暗）
    # 用 Otsu 自适应阈值
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 形态学操作：开运算去噪 + 闭运算填洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 连通域分析
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    bboxes = []
    for i in range(1, n_labels):  # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        
        # 过滤太细长的（类器官近似圆形）
        aspect = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 999
        if aspect > 3.0:
            continue
        
        # 过滤边缘的（可能是不完整目标）
        margin = 20
        if x < margin or y < margin or x+bw > w-margin or y+bh > h-margin:
            continue
        
        bboxes.append({'x': int(x), 'y': int(y), 'w': int(bw), 'h': int(bh), 'area': int(area), 'confidence': 1.0})
    
    return bboxes

def load_gt(gt_path, img_w, img_h):
    bboxes = []
    if os.path.exists(gt_path):
        with open(gt_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, xc, yc, w, h = parts
                    xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                    x1 = int((xc - w/2) * img_w)
                    y1 = int((yc - h/2) * img_h)
                    bw = int(w * img_w)
                    bh = int(h * img_h)
                    bboxes.append({'x': int(x1), 'y': int(y1), 'w': int(bw), 'h': int(bh)})
    return bboxes

def iou(box1, box2):
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x']+box1['w'], box2['x']+box2['w'])
    y2 = min(box1['y']+box1['h'], box2['y']+box2['h'])
    if x2 <= x1 or y2 <= y1:
        return 0
    inter = (x2-x1) * (y2-y1)
    area1 = box1['w'] * box1['h']
    area2 = box2['w'] * box2['h']
    return inter / (area1 + area2 - inter)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--gt', default=None)
    parser.add_argument('--dst', default='./results/traditional')
    parser.add_argument('--min-area', type=int, default=5000)
    parser.add_argument('--max-area', type=int, default=500000)
    args = parser.parse_args()
    
    os.makedirs(args.dst, exist_ok=True)
    
    img_dir = Path(args.src)
    images = sorted(img_dir.glob('*.jpg'))
    
    results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for i, img_path in enumerate(images):
        img = Image.open(img_path)
        w, h = img.size
        
        dets = detect_organoids(img_path, args.min_area, args.max_area)
        
        # Load GT
        gt_bboxes = []
        if args.gt:
            gt_name = f'image_{i:02d}.txt'
            gt_path = Path(args.gt) / gt_name
            gt_bboxes = load_gt(gt_path, w, h)
        
        # Match
        tp = 0
        matched = set()
        for det in dets:
            best_iou = 0
            best_gt = -1
            for gi, gt in enumerate(gt_bboxes):
                if gi in matched:
                    continue
                v = iou(det, gt)
                if v > best_iou:
                    best_iou = v
                    best_gt = gi
            if best_iou > 0.5 and best_gt >= 0:
                tp += 1
                matched.add(best_gt)
        fp = len(dets) - tp
        fn = len(gt_bboxes) - len(matched)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0
        rec = tp/(tp+fn) if (tp+fn) > 0 else 0
        
        print(f"  [{i+1}/{len(images)}] {img_path.name}: det={len(dets)} gt={len(gt_bboxes)} TP={tp} FP={fp} FN={fn} P={prec:.2f} R={rec:.2f}")
        
        results.append({
            'image': img_path.name,
            'n_detections': len(dets),
            'n_gt': len(gt_bboxes),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': prec, 'recall': rec,
            'detections': dets,
        })
    
    prec = total_tp/(total_tp+total_fp) if (total_tp+total_fp) > 0 else 0
    rec = total_tp/(total_tp+total_fn) if (total_tp+total_fn) > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"TRADITIONAL CV DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Images: {len(images)}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision: {prec:.4f} ({prec*100:.1f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.1f}%)")
    print(f"  F1:        {f1:.4f} ({f1*100:.1f}%)")
    
    output = {
        'summary': {'n_images': len(images), 'total_tp': total_tp, 'total_fp': total_fp, 
                     'total_fn': total_fn, 'precision': prec, 'recall': rec, 'f1': f1,
                     'method': 'traditional_cv_otsu'},
        'per_image': results
    }
    out_path = Path(args.dst) / 'traditional_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"  Report: {out_path}")

if __name__ == '__main__':
    main()
