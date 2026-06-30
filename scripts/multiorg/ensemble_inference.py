r"""
Ensemble Inference for MultiOrg — RF-DETR + YOLOv12 双模型融合

策略：
  1. 两个模型分别跑 SAHI 滑动窗口推理（各自独立）
  2. 对每张图的两组检测结果做融合：
     - intersection: 只有当两个模型都检测到（IoU > match_threshold）才保留，score 取平均
     - union: 两个模型的检测全部保留，匹配上的 score 取平均，未匹配的按 score × penalty 保留
  3. 评估 ensemble 结果 vs 单模型 baseline

用法（Windows，单行命令）：
cd C:\Users\decha\organoid-fl
python scripts\multiorg\ensemble_inference.py --rfdetr-weights output\checkpoint_best_regular.pth --rfdetr-variant small --yolo-weights D:\datasets\MultiOrg_v4_640\runs\multiorg_v5_12s_freebies-2\best.pt --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\ensemble --strategy intersection --match-iou 0.5

依赖：
    pip install rfdetr ultralytics supervision tifffile
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from PIL import Image

# 复用 sahi_inference 的函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sahi_inference import (
    convert_tiff_to_rgb,
    sliding_windows,
    load_yolo_model,
    load_rfdetr_model,
    detect_yolo_patch,
    detect_rfdetr_patch,
    filter_boundary_detections,
    compute_iou,
    compute_ap_full,
    compute_map_range,
    load_ground_truth,
    find_annotation_for_image,
    nms,
    soft_nms,
)


def inference_single_model(model, model_type, img_pil, img_w, img_h,
                           window_sizes=(512,), downsample_factors=None,
                           overlap=0.3, conf=0.25, merge='soft_nms',
                           nms_threshold=0.5, min_size=10, score_filter=0.0):
    """单模型 SAHI 推理（复用 sahi_inference.inference_image 逻辑）

    Args:
        img_pil: 已转换好的 PIL Image（避免重复读取 16-bit TIFF）
        img_w, img_h: 图像尺寸

    Returns:
        detections: list of (x1, y1, x2, y2, score)
    """
    if downsample_factors is None:
        downsample_factors = [1] * len(window_sizes)

    all_detections = []

    for ws_idx, ws in enumerate(window_sizes):
        ds = downsample_factors[ws_idx] if ws_idx < len(downsample_factors) else 1
        if ws > max(img_w, img_h):
            continue

        windows = sliding_windows(img_w, img_h, ws, overlap)

        for (x, y, x_end, y_end) in windows:
            tile = img_pil.crop((x, y, x_end, y_end))

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

            for x1, y1, x2, y2, score in dets:
                all_detections.append((
                    x + x1 * ds, y + y1 * ds,
                    x + x2 * ds, y + y2 * ds,
                    score, ws
                ))

    all_detections = filter_boundary_detections(
        all_detections, img_w, img_h, min_size=min_size
    )

    if merge == 'soft_nms':
        fused = soft_nms(all_detections, iou_threshold=nms_threshold, sigma=0.5)
        if score_filter > 0:
            fused = [d for d in fused if d[4] >= score_filter]
    else:
        fused = nms(all_detections, iou_threshold=nms_threshold)

    # 只返回 (x1,y1,x2,y2,score)
    return [(d[0], d[1], d[2], d[3], d[4]) for d in fused]


def ensemble_intersection(dets_a, dets_b, match_iou=0.5):
    """Intersection 融合：只有两个模型都检测到的 box 保留

    匹配策略：贪心，按 score 降序，IoU > match_iou 视为同一目标
    Score 融合：取平均（也可取 min/max，这里用平均更稳健）

    Returns:
        fused: list of (x1, y1, x2, y2, score)
    """
    if not dets_a or not dets_b:
        return []

    # 按 score 降序
    dets_a_sorted = sorted(dets_a, key=lambda d: -d[4])
    dets_b_sorted = sorted(dets_b, key=lambda d: -d[4])

    matched_b = [False] * len(dets_b_sorted)
    fused = []

    for det_a in dets_a_sorted:
        best_iou = 0
        best_b_idx = -1
        for j, det_b in enumerate(dets_b_sorted):
            if matched_b[j]:
                continue
            iou = compute_iou(det_a[:4], det_b[:4])
            if iou > best_iou:
                best_iou = iou
                best_b_idx = j

        if best_iou >= match_iou and best_b_idx >= 0:
            matched_b[best_b_idx] = True
            det_b = dets_b_sorted[best_b_idx]
            # box 取平均
            x1 = (det_a[0] + det_b[0]) / 2
            y1 = (det_a[1] + det_b[1]) / 2
            x2 = (det_a[2] + det_b[2]) / 2
            y2 = (det_a[3] + det_b[3]) / 2
            # score 取平均
            score = (det_a[4] + det_b[4]) / 2
            fused.append((x1, y1, x2, y2, score))

    return fused


def ensemble_union(dets_a, dets_b, match_iou=0.5, unmatched_penalty=0.7,
                   post_nms_iou=0.5):
    """Union 融合：所有检测保留，匹配上的 score 取平均，未匹配的 score × penalty

    这个策略比 intersection 宽松，保留 recall 但降低未匹配检测的置信度

    Args:
        unmatched_penalty: 未匹配检测的 score 乘数（0.7 = 降 30%）
        post_nms_iou: 后置 NMS 的 IoU 阈值，合并未匹配的重叠框
                      （两个模型对同一目标 IoU<match_iou 时会产生重复框）
    """
    if not dets_a and not dets_b:
        return []
    if not dets_a:
        return [(d[0], d[1], d[2], d[3], d[4] * unmatched_penalty) for d in dets_b]
    if not dets_b:
        return [(d[0], d[1], d[2], d[3], d[4] * unmatched_penalty) for d in dets_a]

    dets_a_sorted = sorted(dets_a, key=lambda d: -d[4])
    dets_b_sorted = sorted(dets_b, key=lambda d: -d[4])

    matched_b = [False] * len(dets_b_sorted)
    matched_a = [False] * len(dets_a_sorted)
    fused = []

    for i, det_a in enumerate(dets_a_sorted):
        best_iou = 0
        best_b_idx = -1
        for j, det_b in enumerate(dets_b_sorted):
            if matched_b[j]:
                continue
            iou = compute_iou(det_a[:4], det_b[:4])
            if iou > best_iou:
                best_iou = iou
                best_b_idx = j

        if best_iou >= match_iou and best_b_idx >= 0:
            matched_a[i] = True
            matched_b[best_b_idx] = True
            det_b = dets_b_sorted[best_b_idx]
            x1 = (det_a[0] + det_b[0]) / 2
            y1 = (det_a[1] + det_b[1]) / 2
            x2 = (det_a[2] + det_b[2]) / 2
            y2 = (det_a[3] + det_b[3]) / 2
            score = (det_a[4] + det_b[4]) / 2
            fused.append((x1, y1, x2, y2, score))

    # 未匹配的 A 检测（B 没检测到）
    for i, det_a in enumerate(dets_a_sorted):
        if not matched_a[i]:
            fused.append((det_a[0], det_a[1], det_a[2], det_a[3], det_a[4] * unmatched_penalty))

    # 未匹配的 B 检测（A 没检测到）
    for j, det_b in enumerate(dets_b_sorted):
        if not matched_b[j]:
            fused.append((det_b[0], det_b[1], det_b[2], det_b[3], det_b[4] * unmatched_penalty))

    # 后置 NMS：合并未匹配的重叠框
    # （两个模型对同一目标 IoU<match_iou 时会各自保留一个框）
    fused = nms(fused, iou_threshold=post_nms_iou)

    # 按 score 降序
    fused.sort(key=lambda d: -d[4])
    return fused


def ensemble_wbf(dets_a, dets_b, img_w, img_h, iou_thr=0.55, weights=None):
    """Weighted Boxes Fusion (WBF) — Solovyev et al., IVC 2021

    SOTA ensemble 方法。所有 box 按 score 聚类，fused box = score 加权平均坐标，
    fused score = sum(scores) * T / N (T=cluster中box数, N=模型数)。

    比 intersection/union 优势：
    - 单模型检测到的框也保留（score × 1/N），不丢弃 → 保 recall
    - 坐标用 score 加权平均（非简单平均）→ 高置信度框贡献更大
    - cluster 机制天然去重，不需要后置 NMS

    Args:
        dets_a, dets_b: [(x1,y1,x2,y2,score), ...] 像素坐标
        img_w, img_h: 图像尺寸（用于归一化到 [0,1]）
        iou_thr: cluster 匹配的 IoU 阈值（WBF 论文推荐 0.55）
        weights: 每个模型的权重，None=[1,1] 等权

    Returns:
        fused: [(x1,y1,x2,y2,score), ...] 像素坐标
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        raise ImportError(
            "WBF requires ensemble-boxes: pip install ensemble-boxes"
        )

    if not dets_a and not dets_b:
        return []

    # 归一化到 [0,1]
    def normalize(dets):
        boxes = [[d[0]/img_w, d[1]/img_h, d[2]/img_w, d[3]/img_h] for d in dets]
        scores = [d[4] for d in dets]
        labels = [0] * len(dets)
        return boxes, scores, labels

    boxes_a, scores_a, labels_a = normalize(dets_a)
    boxes_b, scores_b, labels_b = normalize(dets_b)

    boxes_list = [boxes_a, boxes_b]
    scores_list = [scores_a, scores_b]
    labels_list = [labels_a, labels_b]

    if weights is None:
        weights = [1, 1]

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=0.0  # 保留所有框（WBF 通过 score 调节，不删框）
    )

    # 反归一化回像素坐标
    result = []
    for box, score in zip(fused_boxes, fused_scores):
        x1 = box[0] * img_w
        y1 = box[1] * img_h
        x2 = box[2] * img_w
        y2 = box[3] * img_h
        result.append((x1, y1, x2, y2, float(score)))

    result.sort(key=lambda d: -d[4])
    return result


def process_ensemble(rfdetr_model, yolo_model, src_dir, dst_dir,
                     rfdetr_config, yolo_config,
                     strategy='intersection', match_iou=0.5,
                     annotator='t1_b', unmatched_penalty=0.7):
    """处理整个 test set"""
    os.makedirs(dst_dir, exist_ok=True)

    all_results = []
    totals = {
        'rfdetr': {'ap50': 0, 'ap5095': 0, 'tp': 0, 'fp': 0, 'fn': 0},
        'yolo': {'ap50': 0, 'ap5095': 0, 'tp': 0, 'fp': 0, 'fn': 0},
        'ensemble': {'ap50': 0, 'ap5095': 0, 'tp': 0, 'fp': 0, 'fn': 0},
    }
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

                tiff_file = None
                for f in os.listdir(img_dir):
                    if f.lower().endswith(('.tiff', '.tif')):
                        tiff_file = os.path.join(img_dir, f)
                        break

                if tiff_file is None:
                    continue

                gt_path = find_annotation_for_image(img_dir, annotator=annotator)

                print(f"  [{n_images+1:3d}] {class_name}/{plate}/{img_dir_name}...", end=" ")

                start = time.time()

                # 只读一次 16-bit TIFF（避免重复读取大文件）
                img_pil = convert_tiff_to_rgb(tiff_file)
                img_w, img_h = img_pil.size

                # RF-DETR 推理
                rfdetr_dets = inference_single_model(
                    rfdetr_model, 'rfdetr', img_pil, img_w, img_h, **rfdetr_config
                )

                # YOLO 推理
                yolo_dets = inference_single_model(
                    yolo_model, 'yolo', img_pil, img_w, img_h, **yolo_config
                )

                # Ensemble 融合
                if strategy == 'intersection':
                    ensemble_dets = ensemble_intersection(rfdetr_dets, yolo_dets, match_iou)
                elif strategy == 'union':
                    ensemble_dets = ensemble_union(rfdetr_dets, yolo_dets, match_iou, unmatched_penalty)
                elif strategy == 'wbf':
                    ensemble_dets = ensemble_wbf(rfdetr_dets, yolo_dets, img_w, img_h,
                                                 iou_thr=match_iou)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                elapsed = time.time() - start

                # 评估三个结果
                gts = []
                if gt_path:
                    gts = load_ground_truth(gt_path, img_w, img_h)

                results = {}
                for name, dets in [('rfdetr', rfdetr_dets), ('yolo', yolo_dets), ('ensemble', ensemble_dets)]:
                    if gts:
                        ap50, tp, fp, fn = compute_ap_full(dets, gts, iou_threshold=0.5)
                        ap5095 = compute_map_range(dets, gts)
                    else:
                        ap50, tp, fp, fn, ap5095 = 0, 0, 0, 0, 0
                    results[name] = {'ap50': ap50, 'ap5095': ap5095, 'tp': tp, 'fp': fp, 'fn': fn, 'n_dets': len(dets)}
                    totals[name]['ap50'] += ap50
                    totals[name]['ap5095'] += ap5095
                    totals[name]['tp'] += tp
                    totals[name]['fp'] += fp
                    totals[name]['fn'] += fn

                n_images += 1

                r = results['rfdetr']
                y = results['yolo']
                e = results['ensemble']
                print(f"RF:{r['ap50']:.3f}/{r['n_dets']}dets "
                      f"YOLO:{y['ap50']:.3f}/{y['n_dets']}dets "
                      f"ENS:{e['ap50']:.3f}/{e['n_dets']}dets "
                      f"({elapsed:.1f}s)")

                all_results.append({
                    'image': f"{class_name}/{plate}/{img_dir_name}",
                    'rfdetr': results['rfdetr'],
                    'yolo': results['yolo'],
                    'ensemble': results['ensemble'],
                    'n_gt': len(gts),
                    'time_s': elapsed,
                })

    # 汇总
    summary = {}
    for name in ['rfdetr', 'yolo', 'ensemble']:
        t = totals[name]
        mean_ap50 = t['ap50'] / n_images if n_images > 0 else 0
        mean_ap5095 = t['ap5095'] / n_images if n_images > 0 else 0
        precision = t['tp'] / (t['tp'] + t['fp']) if (t['tp'] + t['fp']) > 0 else 0
        recall = t['tp'] / (t['tp'] + t['fn']) if (t['tp'] + t['fn']) > 0 else 0
        summary[name] = {
            'mean_ap50': mean_ap50,
            'mean_ap5095': mean_ap5095,
            'precision': precision,
            'recall': recall,
            'tp': t['tp'], 'fp': t['fp'], 'fn': t['fn'],
        }

    report = {
        'summary': summary,
        'config': {
            'strategy': strategy,
            'match_iou': match_iou,
            'rfdetr_config': rfdetr_config,
            'yolo_config': yolo_config,
            'annotator': annotator,
            'unmatched_penalty': unmatched_penalty if strategy == 'union' else None,
        },
        'per_image': all_results,
    }

    report_path = os.path.join(dst_dir, 'ensemble_results.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # 打印对比表
    print(f"\n{'='*70}")
    print(f"ENSEMBLE RESULTS (strategy={strategy}, annotator={annotator})")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'mAP50':>10} {'mAP50:95':>10} {'Prec':>8} {'Recall':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"{'-'*70}")
    for name in ['rfdetr', 'yolo', 'ensemble']:
        s = summary[name]
        print(f"{name:<12} {s['mean_ap50']*100:>9.2f}% {s['mean_ap5095']*100:>9.2f}% "
              f"{s['precision']*100:>7.2f}% {s['recall']*100:>7.2f}% "
              f"{s['tp']:>6} {s['fp']:>6} {s['fn']:>6}")
    print(f"\n  Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='RF-DETR + YOLOv12 Ensemble Inference')
    parser.add_argument('--rfdetr-weights', required=True, help='RF-DETR checkpoint path')
    parser.add_argument('--rfdetr-variant', default='small',
                        choices=['nano', 'small', 'base', 'medium', 'large'])
    parser.add_argument('--yolo-weights', required=True, help='YOLOv12 checkpoint path')
    parser.add_argument('--src', required=True, help='Test set dir (MultiOrg_v2/test)')
    parser.add_argument('--dst', default='./results/ensemble', help='Output directory')
    parser.add_argument('--strategy', default='wbf',
                        choices=['intersection', 'union', 'wbf'],
                        help='Ensemble strategy: wbf (SOTA, default) / intersection (strict) / union (loose)')
    parser.add_argument('--match-iou', type=float, default=0.3,
                        help='IoU threshold for matching detections between models (0.3 for cross-model)')
    parser.add_argument('--unmatched-penalty', type=float, default=0.7,
                        help='Score multiplier for unmatched detections (union strategy only)')
    parser.add_argument('--annotator', default='t1_b',
                        choices=['t0', 't1_a', 't1_b', 'annotator_a', 'annotator_b', 'any'])

    # RF-DETR SAHI config
    parser.add_argument('--rfdetr-windows', type=int, nargs='+', default=[512])
    parser.add_argument('--rfdetr-overlap', type=float, default=0.3)
    parser.add_argument('--rfdetr-conf', type=float, default=0.25)
    parser.add_argument('--rfdetr-merge', default='soft_nms', choices=['nms', 'soft_nms'])
    parser.add_argument('--rfdetr-score-filter', type=float, default=0.0)

    # YOLO SAHI config
    parser.add_argument('--yolo-windows', type=int, nargs='+', default=[512])
    parser.add_argument('--yolo-overlap', type=float, default=0.3)
    parser.add_argument('--yolo-conf', type=float, default=0.25)
    parser.add_argument('--yolo-merge', default='soft_nms', choices=['nms', 'soft_nms'])
    parser.add_argument('--yolo-score-filter', type=float, default=0.0)

    parser.add_argument('--min-size', type=int, default=10)
    parser.add_argument('--nms-threshold', type=float, default=0.5)
    args = parser.parse_args()

    rfdetr_config = {
        'window_sizes': tuple(args.rfdetr_windows),
        'overlap': args.rfdetr_overlap,
        'conf': args.rfdetr_conf,
        'merge': args.rfdetr_merge,
        'nms_threshold': args.nms_threshold,
        'min_size': args.min_size,
        'score_filter': args.rfdetr_score_filter,
    }
    yolo_config = {
        'window_sizes': tuple(args.yolo_windows),
        'overlap': args.yolo_overlap,
        'conf': args.yolo_conf,
        'merge': args.yolo_merge,
        'nms_threshold': args.nms_threshold,
        'min_size': args.min_size,
        'score_filter': args.yolo_score_filter,
    }

    print(f"{'='*70}")
    print(f"Ensemble Inference: RF-DETR + YOLOv12")
    print(f"{'='*70}")
    print(f"  RF-DETR: {args.rfdetr_weights} ({args.rfdetr_variant})")
    print(f"    windows={rfdetr_config['window_sizes']}, overlap={rfdetr_config['overlap']}, "
          f"conf={rfdetr_config['conf']}, merge={rfdetr_config['merge']}")
    print(f"  YOLO:    {args.yolo_weights}")
    print(f"    windows={yolo_config['window_sizes']}, overlap={yolo_config['overlap']}, "
          f"conf={yolo_config['conf']}, merge={yolo_config['merge']}")
    print(f"  Strategy: {args.strategy}, match_iou={args.match_iou}")
    if args.strategy == 'union':
        print(f"  Unmatched penalty: {args.unmatched_penalty}")
    print(f"  Source: {args.src}")
    print(f"  Annotator: {args.annotator}")
    print()

    # 加载模型
    print("Loading RF-DETR model...")
    rfdetr_model = load_rfdetr_model(args.rfdetr_weights, args.rfdetr_variant)
    # 显式设置 num_classes=1（避免从 checkpoint 推断的 warning）
    if hasattr(rfdetr_model, 'num_classes'):
        rfdetr_model.num_classes = 1
    print("Loading YOLO model...")
    yolo_model = load_yolo_model(args.yolo_weights)
    print("Models loaded.\n")

    process_ensemble(
        rfdetr_model, yolo_model, args.src, args.dst,
        rfdetr_config, yolo_config,
        strategy=args.strategy, match_iou=args.match_iou,
        annotator=args.annotator, unmatched_penalty=args.unmatched_penalty,
    )


if __name__ == '__main__':
    main()
