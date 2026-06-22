#!/usr/bin/env python3
"""
SAHI + 双尺度滑动窗口推理 for MultiOrg

关键设计:
1. 全图滑动窗口推理（不是在 patch 上评估）
2. 双尺度: 512 + 2048 (大目标+小目标)
3. WBF (Weighted Box Fusion) 合并重叠检测
4. 支持 YOLOv12 和 RF-DETR

Usage:
    # YOLOv12 推理
    python sahi_inference.py --model yolo --weights path/to/best.pt --src D:\\datasets\\mutliorg\\MultiOrg_v2\\test --dst ./results/sahi

    # RF-DETR 推理
    python sahi_inference.py --model rfdetr --weights path/to/checkpoint.pt --src D:\\datasets\\mutliorg\\MultiOrg_v2\\test --dst ./results/sahi
"""

import os
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image


def convert_tiff_to_rgb(tiff_path):
    """16-bit TIFF → 8-bit RGB"""
    im = Image.open(tiff_path)
    if im.mode == 'I;16' or im.mode == 'I':
        arr = np.array(im, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax > vmin:
            arr8 = ((arr - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        else:
            arr8 = np.zeros_like(arr, dtype=np.uint8)
        rgb = np.stack([arr8, arr8, arr8], axis=-1)
        return Image.fromarray(rgb, mode='RGB')
    return im.convert('RGB')


def sliding_windows(img_w, img_h, window_size, overlap=0.5):
    """生成滑动窗口坐标列表"""
    stride = int(window_size * (1 - overlap))
    windows = []
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x_end = min(x + window_size, img_w)
            y_end = min(y + window_size, img_h)
            if x_end - x < window_size // 2 or y_end - y < window_size // 2:
                continue
            windows.append((x, y, x_end, y_end))
    return windows


def weighted_box_fusion(detections, iou_threshold=0.5):
    """Weighted Box Fusion 合并重叠检测。

    detections: list of (x1, y1, x2, y2, score, model_size)
    Returns: fused list of (x1, y1, x2, y2, score)
    """
    if not detections:
        return []

    # 按分数排序
    sorted_det = sorted(detections, key=lambda d: -d[4])
    used = [False] * len(sorted_det)
    fused = []

    for i, det in enumerate(sorted_det):
        if used[i]:
            continue
        used[i] = True
        cluster = [det]

        for j in range(i + 1, len(sorted_det)):
            if used[j]:
                continue
            iou = compute_iou(det[:4], sorted_det[j][:4])
            if iou >= iou_threshold:
                used[j] = True
                cluster.append(sorted_det[j])

        # 加权平均
        total_score = sum(c[4] for c in cluster)
        if total_score > 0:
            x1 = sum(c[0] * c[4] for c in cluster) / total_score
            y1 = sum(c[1] * c[4] for c in cluster) / total_score
            x2 = sum(c[2] * c[4] for c in cluster) / total_score
            y2 = sum(c[3] * c[4] for c in cluster) / total_score
            avg_score = total_score / len(cluster)
            fused.append((x1, y1, x2, y2, avg_score))

    return fused


def compute_iou(box1, box2):
    """IoU for (x1,y1,x2,y2) format"""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def load_yolo_model(weights_path, device='cuda:0'):
    """加载 YOLO 模型"""
    from ultralytics import YOLO
    model = YOLO(weights_path)
    return model


def load_rfdetr_model(weights_path):
    """加载 RF-DETR 模型"""
    from rfdetr import RFDETRBase
    model = RFDETRBase(pretrain_weights=weights_path)
    return model


def detect_yolo_patch(model, img_pil, conf=0.25):
    """YOLO 检测单个 patch"""
    results = model.predict(img_pil, conf=conf, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = box.conf[0].cpu().numpy()
            detections.append((float(x1), float(y1), float(x2), float(y2), float(score)))
    return detections


def detect_rfdetr_patch(model, img_array, conf=0.25):
    """RF-DETR 检测单个 patch"""
    detections_raw = model.predict(img_array, threshold=conf)
    detections = []
    if hasattr(detections_raw, 'xyxy'):
        boxes = detections_raw.xyxy
        scores = detections_raw.confidence
        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes[i]
            detections.append((float(x1), float(y1), float(x2), float(y2), float(scores[i])))
    elif isinstance(detections_raw, dict):
        boxes = detections_raw.get('boxes', [])
        scores = detections_raw.get('scores', [])
        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes[i]
            detections.append((float(x1), float(y1), float(x2), float(y2), float(scores[i])))
    return detections


def inference_image(model, model_type, img_path, window_sizes=(512, 2048),
                    overlap=0.5, conf=0.25, device='cuda:0'):
    """对一张大图进行多尺度滑动窗口推理"""
    img_pil = convert_tiff_to_rgb(img_path)
    img_w, img_h = img_pil.size

    all_detections = []

    for ws in window_sizes:
        if ws > max(img_w, img_h):
            continue  # 跳过比图还大的窗口

        windows = sliding_windows(img_w, img_h, ws, overlap)

        for (x, y, x_end, y_end) in windows:
            tile = img_pil.crop((x, y, x_end, y_end))

            if model_type == 'yolo':
                dets = detect_yolo_patch(model, tile, conf)
            elif model_type == 'rfdetr':
                tile_arr = np.array(tile)
                dets = detect_rfdetr_patch(model, tile_arr, conf)
            else:
                continue

            # 转换到全图坐标
            for x1, y1, x2, y2, score in dets:
                all_detections.append((
                    x + x1, y + y1, x + x2, y + y2,
                    score, ws
                ))

    # WBF 融合
    fused = weighted_box_fusion(all_detections, iou_threshold=0.5)

    return fused, (img_w, img_h)


def load_ground_truth(json_path, img_w, img_h):
    """加载 ground truth (napari [row,col] format)"""
    import json as json_mod
    with open(json_path, 'r') as f:
        data = json_mod.load(f)

    gts = []
    for key, polygon in data.items():
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        xs = [p[1] for p in polygon]  # col = x
        ys = [p[0] for p in polygon]  # row = y
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        gts.append((x_min, y_min, x_max, y_max))

    return gts


def compute_ap_full(detections, ground_truths, iou_threshold=0.5):
    """计算 AP（单 IoU 阈值）+ 返回 TP/FP/FN。
    
    使用面积插值法（COCO 风格），不是 11-point。
    """
    if not ground_truths:
        return 0.0, 0, 0, 0  # ap, tp, fp, fn

    if not detections:
        return 0.0, 0, 0, len(ground_truths)

    det_sorted = sorted(detections, key=lambda d: -d[4])
    n_gt = len(ground_truths)
    matched_gt = [False] * n_gt
    tp_list = []
    fp_list = []

    for det in det_sorted:
        best_iou = 0
        best_gt = -1
        for i, gt in enumerate(ground_truths):
            if matched_gt[i]:
                continue
            iou = compute_iou(det[:4], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_iou >= iou_threshold and best_gt >= 0:
            matched_gt[best_gt] = True
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    recalls = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum + 1e-16)

    # 面积插值
    # 在 recall 单调递增方向取 precision 最大值
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

    tp = int(tp_cum[-1]) if len(tp_cum) > 0 else 0
    fp = int(fp_cum[-1]) if len(fp_cum) > 0 else 0
    fn = n_gt - tp

    return float(ap), tp, fp, fn


def compute_map_range(detections, ground_truths):
    """计算 mAP@0.5:0.05:0.95（COCO 标准 10 个阈值平均）。"""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for t in iou_thresholds:
        ap, _, _, _ = compute_ap_full(detections, ground_truths, iou_threshold=t)
        aps.append(ap)
    return float(np.mean(aps))


def find_annotation_for_image(img_dir, annotator='t1_b'):
    """找到图像对应的标注文件。
    
    annotator: 't0' | 't1_a' | 't1_b' | 'annotator_a' | 'annotator_b' | 'any'
    """
    # 精确匹配
    target = annotator.lower()
    for f in os.listdir(img_dir):
        if not f.lower().endswith('.json'):
            continue
        if target in f.lower():
            return os.path.join(img_dir, f)
    
    # fallback: any json
    if annotator == 'any':
        for f in os.listdir(img_dir):
            if f.lower().endswith('.json'):
                return os.path.join(img_dir, f)
    return None


def process_test_set(model, model_type, src_dir, dst_dir,
                     window_sizes=(640,), overlap=0.5, conf=0.25,
                     device='cuda:0', annotator='t1_b'):
    """处理整个 test set，用指定标注者评估。"""
    os.makedirs(dst_dir, exist_ok=True)

    all_results = []
    total_ap50 = 0
    total_ap5095 = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    n_images = 0

    for class_name in sorted(os.listdir(src_dir)):
        class_dir = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for plate in sorted(os.listdir(class_dir)):
            plate_dir = os.path.join(class_dir, plate)
            if not os.path.isdir(plate_dir):
                continue

            for img_dir_name in sorted(os.listdir(plate_dir)):
                img_dir = os.path.join(plate_dir, img_dir_name)
                if not os.path.isdir(img_dir):
                    continue

                # 找 TIFF
                tiff_file = None
                for f in os.listdir(img_dir):
                    if f.lower().endswith(('.tiff', '.tif')):
                        tiff_file = os.path.join(img_dir, f)
                        break

                if tiff_file is None:
                    continue

                # 找标注（指定标注者）
                gt_path = find_annotation_for_image(img_dir, annotator=annotator)

                print(f"  [{n_images+1:3d}] {class_name}/{plate}/{img_dir_name}...", end=" ")

                start = time.time()
                detections, (img_w, img_h) = inference_image(
                    model, model_type, tiff_file,
                    window_sizes, overlap, conf, device
                )
                elapsed = time.time() - start

                # 评估
                ap50, tp, fp, fn = 0.0, 0, 0, 0
                ap5095 = 0.0
                if gt_path:
                    gts = load_ground_truth(gt_path, img_w, img_h)
                    ap50, tp, fp, fn = compute_ap_full(detections, gts, iou_threshold=0.5)
                    ap5095 = compute_map_range(detections, gts)

                total_ap50 += ap50
                total_ap5095 += ap5095
                total_tp += tp
                total_fp += fp
                total_fn += fn
                n_images += 1

                print(f"AP50={ap50:.4f} AP50-95={ap5095:.4f} TP={tp} FP={fp} FN={fn} ({len(detections)} dets, {elapsed:.1f}s)")

                all_results.append({
                    'image': f"{class_name}/{plate}/{img_dir_name}",
                    'n_detections': len(detections),
                    'n_gt': tp + fn,
                    'ap50': ap50,
                    'ap5095': ap5095,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'time_s': elapsed,
                })

    # 汇总
    mean_ap50 = total_ap50 / n_images if n_images > 0 else 0
    mean_ap5095 = total_ap5095 / n_images if n_images > 0 else 0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    summary = {
        'n_images': n_images,
        'annotator': annotator,
        'mean_ap50': mean_ap50,
        'mean_ap5095': mean_ap5095,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'window_sizes': list(window_sizes),
        'overlap': overlap,
        'conf_threshold': conf,
        'model_type': model_type,
    }

    report_path = os.path.join(dst_dir, 'sahi_results.json')
    with open(report_path, 'w') as f:
        json.dump({'summary': summary, 'per_image': all_results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SAHI INFERENCE SUMMARY (annotator={annotator})")
    print(f"{'='*60}")
    print(f"  Images: {n_images}")
    print(f"  mAP@0.5:      {mean_ap50:.4f} ({mean_ap50*100:.1f}%)")
    print(f"  mAP@0.5:0.95: {mean_ap5095:.4f} ({mean_ap5095*100:.1f}%)")
    print(f"  Precision:    {overall_precision:.4f} ({overall_precision*100:.1f}%)")
    print(f"  Recall:       {overall_recall:.4f} ({overall_recall*100:.1f}%)")
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"  Windows: {window_sizes}, Overlap: {overlap}, Conf: {conf}")
    print(f"  Report: {report_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='SAHI Multi-Scale Sliding Window Inference for MultiOrg'
    )
    parser.add_argument('--model', required=True, choices=['yolo', 'rfdetr'])
    parser.add_argument('--weights', required=True, help='Model weights path')
    parser.add_argument('--src', required=True, help='Test set directory (MultiOrg_v2/test)')
    parser.add_argument('--dst', default='./results/sahi', help='Output directory')
    parser.add_argument('--windows', type=int, nargs='+', default=[640],
                        help='Window sizes (default: 640, matching training resolution)')
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--annotator', default='t1_b',
                        choices=['t0', 't1_a', 't1_b', 'annotator_a', 'annotator_b', 'any'],
                        help='Which annotator labels to evaluate against (default: t1_b)')
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Weights: {args.weights}")
    print(f"Source: {args.src}")
    print(f"Annotator: {args.annotator}")
    print(f"Windows: {args.windows}, Overlap: {args.overlap}, Conf: {args.conf}")
    print("=" * 60)

    # 加载模型
    if args.model == 'yolo':
        model = load_yolo_model(args.weights, args.device)
    else:
        model = load_rfdetr_model(args.weights)

    # 推理
    process_test_set(
        model, args.model, args.src, args.dst,
        tuple(args.windows), args.overlap, args.conf, args.device,
        annotator=args.annotator
    )


if __name__ == '__main__':
    main()
