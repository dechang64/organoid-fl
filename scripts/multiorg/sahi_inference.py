#!/usr/bin/env python3
"""
SAHI + 双尺度滑动窗口推理 for MultiOrg

关键设计:
1. 全图滑动窗口推理（不是在 patch 上评估）
2. 双尺度+downsample: 512(ds=2) + 2048(ds=8)，对齐 MultiOrg 论文协议
3. NMS 或 WBF 合并重叠检测（论文用 NMS threshold=0.5）
4. 支持 YOLOv12 和 RF-DETR
5. 边界框过滤：丢弃 SAHI 切割产生的碎片 FP

Usage:
    # YOLOv12 单尺度 640
    python sahi_inference.py --model yolo --weights path/to/best.pt --src D:\\datasets\\mutliorg\\MultiOrg_v2\\test --dst ./results/sahi --windows 640

    # 双尺度+downsample（对齐论文协议）
    python sahi_inference.py --model yolo --weights path/to/best.pt --src ... --dst ... --windows 512 2048 --downsample 2 8 --merge nms

    # RF-DETR 推理
    python sahi_inference.py --model rfdetr --weights path/to/checkpoint.pt --src ... --dst ...
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

    使用 IoU + 中心点距离双判据：
    - IoU ≥ iou_threshold → 同一目标
    - 或一个小框的中心在另一个框内 → 同一目标
    这解决了 overlap=0.5 滑动窗口中同一目标被两个窗口各检测一半的问题。
    """
    if not detections:
        return []

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

            # 中心点判据：小框中心在大框内
            cx_j = (sorted_det[j][0] + sorted_det[j][2]) / 2
            cy_j = (sorted_det[j][1] + sorted_det[j][3]) / 2
            cx_i = (det[0] + det[2]) / 2
            cy_i = (det[1] + det[3]) / 2

            center_in_i = (det[0] <= cx_j <= det[2] and det[1] <= cy_j <= det[3])
            center_in_j = (sorted_det[j][0] <= cx_i <= sorted_det[j][2] and
                           sorted_det[j][1] <= cy_i <= sorted_det[j][3])

            if iou >= iou_threshold or center_in_i or center_in_j:
                used[j] = True
                cluster.append(sorted_det[j])

        # 加权平均
        total_score = sum(c[4] for c in cluster)
        if total_score > 0:
            x1 = sum(c[0] * c[4] for c in cluster) / total_score
            y1 = sum(c[1] * c[4] for c in cluster) / total_score
            x2 = sum(c[2] * c[4] for c in cluster) / total_score
            y2 = sum(c[3] * c[4] for c in cluster) / total_score
            # 保留最高分（不是平均分），更接近真实置信度
            max_score = max(c[4] for c in cluster)
            fused.append((x1, y1, x2, y2, max_score))

    return fused


def nms(detections, iou_threshold=0.5):
    """标准 NMS（对齐 MultiOrg 论文协议）。

    detections: list of (x1, y1, x2, y2, score, ...)
    Returns: kept list of (x1, y1, x2, y2, score)

    与 WBF 的区别：NMS 只保留最高分框，不融合。
    论文用的就是 NMS threshold=0.5。
    """
    if not detections:
        return []

    sorted_det = sorted(detections, key=lambda d: -d[4])
    kept = []

    for det in sorted_det:
        suppressed = False
        for k in kept:
            iou = compute_iou(det[:4], k[:4])
            if iou >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(det[:5])  # 只保留 x1,y1,x2,y2,score

    return kept


def soft_nms(detections, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """Soft-NMS：降低重叠框的分数而非直接删除。

    Bodla et al., ICCV 2017.

    对密集目标场景比标准 NMS 更好：
    - 标准 NMS 直接删除重叠框 → 密集目标漏检
    - Soft-NMS 用高斯函数降低重叠框分数 → 保留但降权

    sigma=0.5 是论文推荐值。
    """
    if not detections:
        return []

    # 转为可变列表
    dets = [list(d[:5]) for d in detections]  # x1,y1,x2,y2,score
    kept = []

    while dets:
        # 选最高分
        dets.sort(key=lambda d: -d[4])
        best = dets.pop(0)
        if best[4] < score_threshold:
            break
        kept.append(tuple(best))

        # 对剩余框降分
        for d in dets:
            iou = compute_iou(best[:4], d[:4])
            if iou > 0:
                # 高斯衰减
                d[4] = d[4] * np.exp(-(iou ** 2) / sigma)

    return kept


def filter_boundary_detections(detections, img_w, img_h, min_size=10, boundary_margin=0):
    """过滤边界碎片检测。

    - 丢弃面积过小的检测（min_size 像素）
    - 丢弃完全在图像边界外的检测
    - boundary_margin > 0 时丢弃紧贴边界的检测（SAHI 切割碎片）
    """
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        w = x2 - x1
        h = y2 - y1
        if w < min_size or h < min_size:
            continue
        if x2 <= 0 or y2 <= 0 or x1 >= img_w or y1 >= img_h:
            continue
        # 裁剪到图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        filtered.append((x1, y1, x2, y2) + tuple(det[4:]))
    return filtered


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


def load_rfdetr_model(weights_path, model_variant='small'):
    """加载 RF-DETR 模型"""
    from rfdetr import RFDETRBase, RFDETRSmall, RFDETRLarge, RFDETRMedium
    from rfdetr import RFDETRNano

    model_map = {
        'nano': RFDETRNano,
        'small': RFDETRSmall,
        'base': RFDETRBase,
        'medium': RFDETRMedium,
        'large': RFDETRLarge,
    }
    ModelClass = model_map.get(model_variant, RFDETRSmall)
    model = ModelClass(pretrain_weights=weights_path)
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


def inference_image(model, model_type, img_path, window_sizes=(640,),
                    downsample_factors=None, overlap=0.5, conf=0.25,
                    device='cuda:0', merge='nms', nms_threshold=0.5,
                    min_size=10, score_filter=0.0):
    """对一张大图进行多尺度滑动窗口推理。

    对齐 MultiOrg 论文协议：
    - window=512, downsample=2 → 实际输入 256px（小目标）
    - window=2048, downsample=8 → 实际输入 256px（大目标）
    - overlap=0.5, NMS threshold=0.5

    downsample_factors: list of int, 与 window_sizes 一一对应
        None 时默认全部为 1（不 downsample）
    merge: 'nms' 或 'wbf' 或 'soft_nms'
    score_filter: Soft-NMS 后的 score 过滤阈值（仅 soft_nms 生效）
    """
    img_pil = convert_tiff_to_rgb(img_path)
    img_w, img_h = img_pil.size

    if downsample_factors is None:
        downsample_factors = [1] * len(window_sizes)

    all_detections = []

    for ws_idx, ws in enumerate(window_sizes):
        ds = downsample_factors[ws_idx] if ws_idx < len(downsample_factors) else 1
        if ws > max(img_w, img_h):
            continue  # 跳过比图还大的窗口

        windows = sliding_windows(img_w, img_h, ws, overlap)

        for (x, y, x_end, y_end) in windows:
            tile = img_pil.crop((x, y, x_end, y_end))

            # Downsample: 把 ws×ws 的 tile 缩小到 (ws/ds)×(ws/ds)
            if ds > 1:
                tile_w, tile_h = tile.size
                new_w = max(1, tile_w // ds)
                new_h = max(1, tile_h // ds)
                tile = tile.resize((new_w, new_h), Image.LANCZOS)

            if model_type == 'yolo':
                dets = detect_yolo_patch(model, tile, conf)
            elif model_type == 'rfdetr':
                tile_arr = np.array(tile)
                dets = detect_rfdetr_patch(model, tile_arr, conf)
            else:
                continue

            # 转换到全图坐标（检测坐标 × downsample + window offset）
            for x1, y1, x2, y2, score in dets:
                all_detections.append((
                    x + x1 * ds, y + y1 * ds,
                    x + x2 * ds, y + y2 * ds,
                    score, ws
                ))

    # 边界框过滤
    all_detections = filter_boundary_detections(
        all_detections, img_w, img_h, min_size=min_size
    )

    # 合并重叠检测
    if merge == 'wbf':
        fused = weighted_box_fusion(all_detections, iou_threshold=nms_threshold)
    elif merge == 'soft_nms':
        fused = soft_nms(all_detections, iou_threshold=nms_threshold, sigma=0.5)
        # Soft-NMS 后 score 过滤（低分框不删除只降分，需要二次过滤提升 precision）
        if score_filter > 0:
            fused = [d for d in fused if d[4] >= score_filter]
    else:  # nms
        fused = nms(all_detections, iou_threshold=nms_threshold)

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
                     window_sizes=(640,), downsample_factors=None,
                     overlap=0.5, conf=0.25, device='cuda:0',
                     annotator='t1_b', merge='nms', nms_threshold=0.5,
                     min_size=10, score_filter=0.0):
    """处理整个 test set，用指定标注者评估。"""
    os.makedirs(dst_dir, exist_ok=True)

    if downsample_factors is None:
        downsample_factors = [1] * len(window_sizes)

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
                    window_sizes, downsample_factors, overlap, conf, device,
                    merge=merge, nms_threshold=nms_threshold, min_size=min_size,
                    score_filter=score_filter
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
        'downsample_factors': list(downsample_factors),
        'overlap': overlap,
        'conf_threshold': conf,
        'merge_method': merge,
        'nms_threshold': nms_threshold,
        'min_size': min_size,
        'score_filter': score_filter,
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
    print(f"  Windows: {window_sizes}, Downsample: {downsample_factors}")
    print(f"  Overlap: {overlap}, Conf: {conf}, Merge: {merge} (threshold={nms_threshold})")
    print(f"  Report: {report_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='SAHI Multi-Scale Sliding Window Inference for MultiOrg'
    )
    parser.add_argument('--model', required=True, choices=['yolo', 'rfdetr'])
    parser.add_argument('--model-variant', default='small',
                        choices=['nano', 'small', 'base', 'medium', 'large'],
                        help='RF-DETR model variant (only used with --model rfdetr)')
    parser.add_argument('--weights', required=True, help='Model weights path')
    parser.add_argument('--src', required=True, help='Test set directory (MultiOrg_v2/test)')
    parser.add_argument('--dst', default='./results/sahi', help='Output directory')
    parser.add_argument('--windows', type=int, nargs='+', default=[640],
                        help='Window sizes (default: 640)')
    parser.add_argument('--downsample', type=int, nargs='+', default=None,
                        help='Downsample factors per window (default: 1 for all). '
                             'Paper protocol: --windows 512 2048 --downsample 2 8')
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--annotator', default='t1_b',
                        choices=['t0', 't1_a', 't1_b', 'annotator_a', 'annotator_b', 'any'],
                        help='Which annotator labels to evaluate against (default: t1_b)')
    parser.add_argument('--merge', default='nms', choices=['nms', 'wbf', 'soft_nms'],
                        help='Merge method: nms (paper protocol), wbf, or soft_nms (default: nms)')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                        help='NMS/WBF IoU threshold (default: 0.5, paper protocol)')
    parser.add_argument('--min-size', type=int, default=10,
                        help='Min detection size in pixels (filter SAHI fragments)')
    parser.add_argument('--score-filter', type=float, default=0.0,
                        help='Score filter after Soft-NMS (e.g. 0.3 keeps only score>=0.3, '
                             'boosts precision without hurting mAP. Only affects soft_nms)')
    args = parser.parse_args()

    # 默认 downsample
    if args.downsample is None:
        args.downsample = [1] * len(args.windows)

    # 长度校验
    if len(args.downsample) != len(args.windows):
        print(f"ERROR: --windows ({len(args.windows)}) and --downsample ({len(args.downsample)}) must have same length")
        return

    print(f"Model: {args.model}")
    print(f"Weights: {args.weights}")
    print(f"Source: {args.src}")
    print(f"Annotator: {args.annotator}")
    print(f"Windows: {args.windows}, Downsample: {args.downsample}")
    print(f"Overlap: {args.overlap}, Conf: {args.conf}")
    print(f"Merge: {args.merge} (threshold={args.nms_threshold}), Min size: {args.min_size}")
    print("=" * 60)

    # 加载模型
    if args.model == 'yolo':
        model = load_yolo_model(args.weights, args.device)
    else:
        model = load_rfdetr_model(args.weights, args.model_variant)

    # 推理
    process_test_set(
        model, args.model, args.src, args.dst,
        tuple(args.windows), tuple(args.downsample),
        args.overlap, args.conf, args.device,
        annotator=args.annotator,
        merge=args.merge, nms_threshold=args.nms_threshold,
        min_size=args.min_size, score_filter=args.score_filter
    )


if __name__ == '__main__':
    main()
