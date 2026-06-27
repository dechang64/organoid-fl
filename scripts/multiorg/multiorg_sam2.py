r"""
MultiOrg SAM2 Pipeline: SAHI 检测 → SAM2 分割 → 形态学过滤

在 SAHI bbox 检测基础上，用 SAM2 获取像素级 mask，计算形态学特征
（circularity, solidity, aspect_ratio），用这些特征过滤 FP。

鼠肝验证：confidence < 0.65 且 circularity < 0.71 → FP
MultiOrg 目标：砍掉 SAHI 的 11K+ FP，提升 precision 和 mAP

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate

    # 先跑 5 张测试（约 2-3 分钟）
    python scripts\multiorg\multiorg_sam2.py ^
        --weights output\small_512\checkpoint_best_regular.pth ^
        --model-variant small ^
        --src D:\datasets\mutliorg\MultiOrg_v2\test ^
        --dst results\multiorg_sam2 ^
        --windows 512 --overlap 0.3 --conf 0.25 --score-filter 0.3 ^
        --sam2-checkpoint sam2_hiera_small.pt ^
        --max-images 5

    # 全量跑（55 张，约 20-30 分钟）
    python scripts\multiorg\multiorg_sam2.py ^
        --weights output\small_512\checkpoint_best_regular.pth ^
        --model-variant small ^
        --src D:\datasets\mutliorg\MultiOrg_v2\test ^
        --dst results\multiorg_sam2 ^
        --windows 512 --overlap 0.3 --conf 0.25 --score-filter 0.3 ^
        --sam2-checkpoint sam2_hiera_small.pt

    # 不用 SAM2（只用 bbox morphology 做对照）
    python scripts\multiorg\multiorg_sam2.py ^
        --weights output\small_512\checkpoint_best_regular.pth ^
        --model-variant small ^
        --src D:\datasets\mutliorg\MultiOrg_v2\test ^
        --dst results\multiorg_sam2_nosam2 ^
        --windows 512 --overlap 0.3 --conf 0.25 --score-filter 0.3 ^
        --no-sam2 ^
        --max-images 5
"""
import argparse
import os
import sys
import json
import time
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# 添加脚本目录到 path（为了 import sahi_inference 的函数）
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from sahi_inference import (
    convert_tiff_to_rgb, sliding_windows, filter_boundary_detections,
    detect_rfdetr_patch, soft_nms, nms, weighted_box_fusion,
    load_ground_truth, compute_iou, compute_ap_full, compute_map_range,
    find_annotation_for_image, load_rfdetr_model
)


# ============================================================
# SAM2
# ============================================================

def load_sam2(checkpoint='sam2_hiera_small.pt', device='cuda'):
    """加载 SAM2 模型"""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import torch as _torch

    # 我们的 finetune checkpoint 格式: {'trainable_state_dict':..., 'model_state_dict':..., 'epoch':..., 'val_iou':...}
    # build_sam2 期望的格式: {'model': state_dict, ...} 或直接 state_dict
    # 需要先提取 model_state_dict，存成临时 checkpoint 给 build_sam2 加载
    ckpt_data = _torch.load(checkpoint, map_location='cpu', weights_only=False)
    if isinstance(ckpt_data, dict) and 'model_state_dict' in ckpt_data:
        # finetuned checkpoint — 转换成 build_sam2 能读的格式
        import tempfile, os
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
    raise RuntimeError(f"Could not load SAM2 with checkpoint={checkpoint}. Last error: {last_err}")


# ============================================================
# 形态学特征
# ============================================================

def compute_morphology(mask, bbox):
    """从 SAM2 mask 计算形态学指标。
    
    鼠肝验证有效特征：
    - circularity: 4πA/P², 1=完美圆。TP > 0.7, FP < 0.7
    - solidity: A/A_hull, 1=凸形。TP > 0.95, FP < 0.92
    - aspect_ratio: 宽/高, 1=等轴。TP 0.85-1.0, FP < 0.7
    - confidence: RF-DETR 分数。TP > 0.65, FP < 0.65
    """
    area = int(mask.sum())
    if area == 0:
        return {
            "area": 0, "perimeter": 0, "circularity": 0, "solidity": 0,
            "aspect_ratio": 0, "bbox": bbox, "has_mask": False
        }

    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "area": area, "perimeter": 0, "circularity": 0, "solidity": 0,
            "aspect_ratio": 0, "bbox": bbox, "has_mask": True
        }

    c = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # 用 bbox 的 aspect_ratio（更稳定 than ellipse）
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    aspect_ratio = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0

    return {
        "area": area,
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "solidity": float(solidity),
        "aspect_ratio": float(aspect_ratio),
        "bbox": [float(x) for x in bbox],
        "has_mask": True,
    }


def bbox_to_mask(bbox, img_h, img_w):
    """bbox → 矩形 mask（不用 SAM2 时的 fallback）"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    mask = np.zeros((img_h, img_w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


# ============================================================
# SAHI 检测（带 SAM2）
# ============================================================

def sahi_detect_with_sam2(model, model_type, img_path, sam2_predictor,
                          window_sizes=(512,), downsample_factors=None,
                          overlap=0.3, conf=0.25, device='cuda:0',
                          merge='soft_nms', nms_threshold=0.5,
                          min_size=10, score_filter=0.0,
                          use_sam2=True, max_det_per_image=500):
    """SAHI 检测 + SAM2 分割 + 形态学特征

    Returns:
        detections: list of dict, 每个含 bbox, confidence, morphology
        img_size: (w, h)
    """
    img_pil = convert_tiff_to_rgb(img_path)
    img_w, img_h = img_pil.size
    img_np = np.array(img_pil)

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

            if model_type == 'rfdetr':
                tile_arr = np.array(tile)
                dets = detect_rfdetr_patch(model, tile_arr, conf)
            else:
                # yolo fallback
                from sahi_inference import detect_yolo_patch
                dets = detect_yolo_patch(model, tile, conf)

            # 转换到全图坐标
            for x1, y1, x2, y2, score in dets:
                all_detections.append((
                    x + x1 * ds, y + y1 * ds,
                    x + x2 * ds, y + y2 * ds,
                    score, ws
                ))

    # 边界过滤
    all_detections = filter_boundary_detections(
        all_detections, img_w, img_h, min_size=min_size
    )

    # 合并
    if merge == 'wbf':
        fused = weighted_box_fusion(all_detections, iou_threshold=nms_threshold)
    elif merge == 'soft_nms':
        fused = soft_nms(all_detections, iou_threshold=nms_threshold, sigma=0.5)
        if score_filter > 0:
            fused = [d for d in fused if d[4] >= score_filter]
    else:
        fused = nms(all_detections, iou_threshold=nms_threshold)

    # 限制检测数量（防止 SAM2 跑太久）
    if len(fused) > max_det_per_image:
        fused = sorted(fused, key=lambda d: -d[4])[:max_det_per_image]
        print(f"    ⚠️ 检测数 {len(fused)} 超过上限 {max_det_per_image}，只取 top-{max_det_per_image}")

    # SAM2 分割
    detections = []
    if use_sam2 and sam2_predictor and len(fused) > 0:
        sam2_predictor.set_image(img_np)
        for det in fused:
            x1, y1, x2, y2, score = det[:5]
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            try:
                masks, scores, _ = sam2_predictor.predict(box=box, multimask_output=False)
                mask = masks[0]
            except Exception as e:
                # SAM2 失败时用 bbox mask
                mask = bbox_to_mask([x1, y1, x2, y2], img_h, img_w)

            morph = compute_morphology(mask, [x1, y1, x2, y2])
            morph['confidence'] = float(score)
            morph['window'] = int(det[5]) if len(det) > 5 else 0
            detections.append(morph)
    else:
        # 不用 SAM2，只用 bbox
        for det in fused:
            x1, y1, x2, y2, score = det[:5]
            morph = compute_morphology(
                bbox_to_mask([x1, y1, x2, y2], img_h, img_w),
                [x1, y1, x2, y2]
            )
            morph['confidence'] = float(score)
            morph['window'] = int(det[5]) if len(det) > 5 else 0
            detections.append(morph)

    return detections, (img_w, img_h)


# ============================================================
# 形态学过滤
# ============================================================

def filter_by_morphology(detections, min_circularity=0.0, min_solidity=0.0,
                         min_confidence=0.0, min_aspect_ratio=0.0):
    """用形态学特征过滤检测"""
    filtered = []
    for det in detections:
        if det.get('confidence', 0) < min_confidence:
            continue
        if det.get('circularity', 0) < min_circularity:
            continue
        if det.get('solidity', 0) < min_solidity:
            continue
        if det.get('aspect_ratio', 0) < min_aspect_ratio:
            continue
        filtered.append(det)
    return filtered


def evaluate_detections(detections, gts, iou_threshold=0.5):
    """评估检测结果（detections 是带 morphology 的 list）"""
    if not gts:
        return 0.0, 0.0, 0, 0, 0

    det_boxes = [(d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['confidence'])
                 for d in detections]
    ap50, tp, fp, fn = compute_ap_full(det_boxes, gts, iou_threshold=iou_threshold)
    ap5095 = compute_map_range(det_boxes, gts)
    return ap50, ap5095, tp, fp, fn


def try_morphology_filters(all_image_results):
    """尝试不同形态学过滤组合，返回最优配置"""
    filters = [
        # (name, min_conf, min_circ, min_solid, min_ar)
        ("baseline (no filter)",        0.0, 0.0, 0.0, 0.0),
        ("conf>=0.4",                   0.4, 0.0, 0.0, 0.0),
        ("conf>=0.5",                   0.5, 0.0, 0.0, 0.0),
        ("conf>=0.6",                   0.6, 0.0, 0.0, 0.0),
        ("circ>=0.3",                   0.0, 0.3, 0.0, 0.0),
        ("circ>=0.4",                   0.0, 0.4, 0.0, 0.0),
        ("circ>=0.5",                   0.0, 0.5, 0.0, 0.0),
        ("solid>=0.90",                 0.0, 0.0, 0.90, 0.0),
        ("solid>=0.95",                 0.0, 0.0, 0.95, 0.0),
        ("ar>=0.5",                     0.0, 0.0, 0.0, 0.5),
        ("ar>=0.6",                     0.0, 0.0, 0.0, 0.6),
        ("conf>=0.4 + circ>=0.3",       0.4, 0.3, 0.0, 0.0),
        ("conf>=0.4 + solid>=0.90",     0.4, 0.0, 0.90, 0.0),
        ("conf>=0.4 + ar>=0.5",         0.4, 0.0, 0.0, 0.5),
        ("conf>=0.5 + circ>=0.3",       0.5, 0.3, 0.0, 0.0),
        ("conf>=0.5 + solid>=0.90",     0.5, 0.0, 0.90, 0.0),
        ("conf>=0.4 + circ>=0.3 + solid>=0.90", 0.4, 0.3, 0.90, 0.0),
        ("conf>=0.4 + circ>=0.3 + ar>=0.5",     0.4, 0.3, 0.0, 0.5),
        ("conf>=0.5 + circ>=0.3 + solid>=0.90", 0.5, 0.3, 0.90, 0.0),
    ]

    results = []
    for name, mc, mci, ms, ma in filters:
        total_tp = total_fp = total_fn = 0
        total_ap50 = 0.0
        n_images = 0

        for img_res in all_image_results:
            filtered = filter_by_morphology(
                img_res['detections'],
                min_circularity=mci, min_solidity=ms,
                min_confidence=mc, min_aspect_ratio=ma
            )
            ap50, ap5095, tp, fp, fn = evaluate_detections(filtered, img_res['gts'])
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_ap50 += ap50
            n_images += 1

        mean_ap50 = total_ap50 / n_images if n_images > 0 else 0
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        results.append({
            'filter': name,
            'min_conf': mc, 'min_circ': mci, 'min_solid': ms, 'min_ar': ma,
            'mAP50': mean_ap50,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        })

    return results


# ============================================================
# 可视化
# ============================================================

def visualize_sam2(img_np, detections, gts, dst_path, max_det=50):
    """可视化：SAM2 mask (green) + GT bbox (red)"""
    vis = img_np.copy()
    # 只画前 max_det 个（防止太密）
    for det in detections[:max_det]:
        bbox = det['bbox']
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # 画 bbox
        conf = det.get('confidence', 0)
        color = (0, 255, 0) if conf >= 0.5 else (0, 200, 255)  # 绿=高置信度, 橙=低置信度
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 标注 confidence + circularity
        circ = det.get('circularity', 0)
        label = f"{conf:.2f} c={circ:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # GT (red)
    for gt in gts:
        cv2.rectangle(vis, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (255, 0, 0), 2)

    cv2.imwrite(str(dst_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='MultiOrg SAM2: SAHI detection + SAM2 segmentation + morphology filtering'
    )
    parser.add_argument('--weights', required=True, help='RF-DETR checkpoint path')
    parser.add_argument('--model-variant', default='small', choices=['nano', 'small', 'base', 'medium', 'large'])
    parser.add_argument('--src', required=True, help='MultiOrg test directory')
    parser.add_argument('--dst', default='results/multiorg_sam2', help='Output directory')
    parser.add_argument('--windows', type=int, nargs='+', default=[512])
    parser.add_argument('--downsample', type=int, nargs='+', default=None)
    parser.add_argument('--overlap', type=float, default=0.3)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--annotator', default='t1_b')
    parser.add_argument('--merge', default='soft_nms', choices=['nms', 'wbf', 'soft_nms'])
    parser.add_argument('--nms-threshold', type=float, default=0.5)
    parser.add_argument('--min-size', type=int, default=10)
    parser.add_argument('--score-filter', type=float, default=0.3)
    parser.add_argument('--sam2-checkpoint', default='sam2_hiera_small.pt')
    parser.add_argument('--no-sam2', action='store_true', help='Skip SAM2, use bbox-only morphology')
    parser.add_argument('--max-images', type=int, default=0, help='Max images (0=all)')
    parser.add_argument('--max-det-per-image', type=int, default=500,
                        help='Max detections per image (prevent SAM2 timeout)')
    parser.add_argument('--save-vis', action='store_true', help='Save visualization images')
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    use_sam2 = not args.no_sam2

    if args.downsample is None:
        args.downsample = [1] * len(args.windows)

    print(f"{'='*60}")
    print(f"MultiOrg SAM2 Pipeline")
    print(f"{'='*60}")
    print(f"  Model: RF-DETR {args.model_variant}")
    print(f"  Weights: {args.weights}")
    print(f"  SAM2: {'ON' if use_sam2 else 'OFF (bbox-only)'}")
    print(f"  Source: {args.src}")
    print(f"  Windows: {args.windows}, Downsample: {args.downsample}")
    print(f"  Overlap: {args.overlap}, Conf: {args.conf}")
    print(f"  Merge: {args.merge}, Score filter: {args.score_filter}")
    print(f"  Max det/image: {args.max_det_per_image}")
    print(f"{'='*60}")

    # 加载模型
    print("Loading RF-DETR...")
    det_model = load_rfdetr_model(args.weights, args.model_variant)

    sam2_predictor = None
    if use_sam2:
        print("Loading SAM2...")
        try:
            sam2_predictor = load_sam2(args.sam2_checkpoint, args.device)
            print("  SAM2 loaded successfully")
        except Exception as e:
            print(f"  SAM2 load failed: {e}")
            print("  Falling back to bbox-only mode")
            use_sam2 = False

    # 遍历测试集
    all_image_results = []
    n_processed = 0

    for class_name in sorted(os.listdir(args.src)):
        class_dir = os.path.join(args.src, class_name)
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

                # 找标注
                gt_path = find_annotation_for_image(img_dir, annotator=args.annotator)

                img_label = f"{class_name}/{plate}/{img_dir_name}"
                print(f"\n  [{n_processed+1}] {img_label}")

                # 加载 GT
                gts = []
                if gt_path:
                    # 先获取图片尺寸
                    img_pil = convert_tiff_to_rgb(tiff_file)
                    img_w, img_h = img_pil.size
                    gts = load_ground_truth(gt_path, img_w, img_h)
                    print(f"    GT: {len(gts)} organoids")

                # SAHI + SAM2
                start = time.time()
                detections, (img_w, img_h) = sahi_detect_with_sam2(
                    det_model, 'rfdetr', tiff_file, sam2_predictor,
                    window_sizes=tuple(args.windows),
                    downsample_factors=tuple(args.downsample),
                    overlap=args.overlap, conf=args.conf, device=args.device,
                    merge=args.merge, nms_threshold=args.nms_threshold,
                    min_size=args.min_size, score_filter=args.score_filter,
                    use_sam2=use_sam2, max_det_per_image=args.max_det_per_image
                )
                elapsed = time.time() - start

                # 评估
                ap50, ap5095, tp, fp, fn = evaluate_detections(detections, gts)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0

                print(f"    Det: {len(detections)}  TP={tp}  FP={fp}  FN={fn}")
                print(f"    AP50={ap50:.4f}  AP50-95={ap5095:.4f}  Prec={prec:.3f}  Rec={rec:.3f}  ({elapsed:.1f}s)")

                # 可视化
                if args.save_vis and n_processed < 10:
                    img_pil = convert_tiff_to_rgb(tiff_file)
                    img_np = np.array(img_pil)
                    vis_path = Path(args.dst) / f'vis_{n_processed:03d}_{class_name}_{img_dir_name}.jpg'
                    visualize_sam2(img_np, detections, gts, vis_path)

                all_image_results.append({
                    'image': img_label,
                    'detections': detections,
                    'gts': gts,
                    'ap50': ap50,
                    'ap5095': ap5095,
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'time_s': elapsed,
                })

                n_processed += 1
                if args.max_images > 0 and n_processed >= args.max_images:
                    break
            if args.max_images > 0 and n_processed >= args.max_images:
                break
        if args.max_images > 0 and n_processed >= args.max_images:
            break

    # ============================================================
    # 汇总 + 形态学过滤分析
    # ============================================================

    print(f"\n{'='*60}")
    print(f"BASELINE (no morphology filter)")
    print(f"{'='*60}")

    total_tp = sum(r['tp'] for r in all_image_results)
    total_fp = sum(r['fp'] for r in all_image_results)
    total_fn = sum(r['fn'] for r in all_image_results)
    mean_ap50 = sum(r['ap50'] for r in all_image_results) / len(all_image_results) if all_image_results else 0
    mean_ap5095 = sum(r['ap5095'] for r in all_image_results) / len(all_image_results) if all_image_results else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"  Images: {len(all_image_results)}")
    print(f"  mAP@0.5:      {mean_ap50:.4f} ({mean_ap50*100:.1f}%)")
    print(f"  mAP@0.5:0.95: {mean_ap5095:.4f} ({mean_ap5095*100:.1f}%)")
    print(f"  Precision:    {prec:.4f} ({prec*100:.1f}%)")
    print(f"  Recall:       {rec:.4f} ({rec*100:.1f}%)")
    print(f"  F1:           {f1:.4f} ({f1*100:.1f}%)")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")

    # 形态学过滤分析
    print(f"\n{'='*60}")
    print(f"MORPHOLOGY FILTER ANALYSIS")
    print(f"{'='*60}")

    filter_results = try_morphology_filters(all_image_results)
    print(f"{'Filter':<45} {'mAP50':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'TP':>5} {'FP':>6} {'FN':>5}")
    print("-" * 95)
    for fr in filter_results:
        print(f"{fr['filter']:<45} {fr['mAP50']*100:>6.1f}% {fr['precision']*100:>6.1f}% {fr['recall']*100:>6.1f}% {fr['f1']*100:>6.1f}% {fr['tp']:>5} {fr['fp']:>6} {fr['fn']:>5}")

    # 找最优 mAP50
    best_map = max(filter_results, key=lambda x: x['mAP50'])
    best_f1 = max(filter_results, key=lambda x: x['f1'])
    print(f"\n  Best mAP50: {best_map['filter']} → {best_map['mAP50']*100:.1f}%")
    print(f"  Best F1:    {best_f1['filter']} → {best_f1['f1']*100:.1f}%")

    # 形态学分布统计
    all_dets_flat = [d for r in all_image_results for d in r['detections']]
    if all_dets_flat:
        print(f"\n{'='*60}")
        print(f"MORPHOLOGY DISTRIBUTION ({len(all_dets_flat)} detections)")
        print(f"{'='*60}")
        for key in ['confidence', 'circularity', 'solidity', 'aspect_ratio', 'area']:
            vals = [d.get(key, 0) for d in all_dets_flat if key in d and d.get(key) is not None]
            if vals:
                vals_arr = np.array(vals)
                print(f"  {key:15s}: min={vals_arr.min():.3f}  p25={np.percentile(vals_arr,25):.3f}  "
                      f"median={np.median(vals_arr):.3f}  p75={np.percentile(vals_arr,75):.3f}  max={vals_arr.max():.3f}")

    # 保存报告
    report = {
        'config': {
            'model': 'rfdetr',
            'model_variant': args.model_variant,
            'weights': args.weights,
            'sam2': use_sam2,
            'sam2_checkpoint': args.sam2_checkpoint if use_sam2 else None,
            'windows': args.windows,
            'downsample': args.downsample,
            'overlap': args.overlap,
            'conf': args.conf,
            'merge': args.merge,
            'score_filter': args.score_filter,
            'annotator': args.annotator,
            'n_images': len(all_image_results),
        },
        'baseline': {
            'mAP50': mean_ap50,
            'mAP5095': mean_ap5095,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        },
        'filter_analysis': filter_results,
        'best_map50': best_map,
        'best_f1': best_f1,
        'per_image': [{
            'image': r['image'],
            'n_det': len(r['detections']),
            'n_gt': len(r['gts']),
            'ap50': r['ap50'],
            'ap5095': r['ap5095'],
            'tp': r['tp'], 'fp': r['fp'], 'fn': r['fn'],
            'time_s': r['time_s'],
            'detections': r['detections'],  # 完整 per-detection data
        } for r in all_image_results],
    }

    report_path = Path(args.dst) / 'multiorg_sam2_results.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")


if __name__ == '__main__':
    main()
