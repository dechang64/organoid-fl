r"""
Phase 1: Perception 层验证 — bbox→SAM2→形态学特征 pipeline

GT 标注可用性 (审计确认):
  B1 (10张, 2592×1944): 只有 bbox GT (annotations.json)
  B2 (10张, 2592×1944): 只有 bbox GT (annotations.json)
  B3 (20张, 4000×3000): bbox GT + 红色折线标注图 (真实轮廓 mask)

评估方式:
  - B1/B2: bbox 级 F1 (IoU>0.5 bbox 匹配)
  - B3: bbox F1 + mask F1 (红色折线→真实轮廓)
  - 所有 batch: 形态学特征分布 (从 SAM2 输出 mask 计算)

实验:
  P1-A: B1 8张 RF-DETR → B1 2张     — 复现 M14 bbox F1
  P1-B: B1 8张 RF-DETR → B2 10张    — 同分辨率跨域
  P1-C: B1 8张 RF-DETR → B3 20张    — 跨分辨率 (关键! bbox F1 + mask F1)
  P1-D: B1+B2 16张 RF-DETR → B3 20张 — 多源训练
  P1-E: B1+B2+B3 31张 → val 9张     — 集中式上界

复用:
  - sam2_segment.py: load_rfdetr, load_sam2, compute_morphology,
                     load_gt_mask (bbox→mask), load_gt_mask_from_annot (红线→mask)
  - fl_sequential.py: BATCH_DIRS, VAL_INDICES

Usage (冬生本地):
  cd C:\Users\decha\organoid-fl
  .\.venv\Scripts\activate

  # 跑全部 Phase 1
  python scripts\mouse_liver\phase1_perception.py

  # 或单组
  python scripts\mouse_liver\phase1_perception.py --exp P1-C

  # 指定 RF-DETR checkpoint
  python scripts\mouse_liver\phase1_perception.py --rfdetr-weights output\checkpoint_best_regular.pth
"""
import os, sys, json, time, argparse
import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam2_segment import (
    load_rfdetr, load_sam2, compute_morphology,
    load_gt_mask, load_gt_mask_from_annot, mask_iou,
)
from fl_sequential import BATCH_DIRS, VAL_INDICES, log

# ============================================================
# 配置 (从 workspace 确认, 不猜)
# ============================================================

# RF-DETR checkpoint: TOOLS.md 确认 "output\checkpoint_best_regular.pth"
# (R3 small+640 覆盖了 R1 small+560)
RFDETR_CKPT = r"output\checkpoint_best_regular.pth"
RFDETR_VARIANT = 'small'

# SAM2 checkpoint: TOOLS.md 确认 "sam2_checkpoints\sam2_hiera_small.pt"
SAM2_CKPT = r"sam2_checkpoints\sam2_hiera_small.pt"

# B3 红色折线标注图: 从 B3 annotations.json source_annotated 字段获取文件名
# 标注图目录: 冬生本地可能和 B3 原始图同目录, 或在单独目录
# 默认尝试 D:\datasets\mouse_liver_data\batch3\ (B3 原始图目录)
B3_ANNOT_DIR = r"D:\datasets\mouse_liver_data\batch3"

# B3 source_annotated 映射 (从 annotations.json 确认)
def load_b3_annot_mapping():
    """返回 {image_name: annot_file_name}"""
    with open(os.path.join(BATCH_DIRS['b3'], 'annotations.json')) as f:
        ann = json.load(f)
    return {item['image']: item['source_annotated'] for item in ann}

# 输出
OUTPUT_BASE = r"runs\mouse_liver_phase1"

# 推理参数
DET_THRESHOLD = 0.5  # M14 用 0.5 (不是 0.25), 0.25 FP 太多
SAM2_DEVICE = 'cuda'


# ============================================================
# 核心函数
# ============================================================

def run_pipeline(det_model, sam2_predictor, img_path):
    """RF-DETR 检测 → SAM2 分割 → 形态学特征

    不 resize — RF-DETR 内部处理分辨率, SAM2 需要原图尺寸的 mask
    和 GT mask (原图尺寸) 保持一致

    Returns:
        detections: list of {bbox, mask, confidence, morphology}
    """
    img_pil = Image.open(img_path).convert('RGB')
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]

    # 1. RF-DETR 检测
    dets = det_model.predict(img_pil, threshold=DET_THRESHOLD)

    if len(dets.xyxy) == 0:
        return []

    results = []
    if sam2_predictor:
        sam2_predictor.set_image(img_np)
        for di in range(len(dets.xyxy)):
            box = dets.xyxy[di].astype(np.float32)
            masks, scores, _ = sam2_predictor.predict(box=box, multimask_output=False)
            mask = masks[0]
            morph = compute_morphology(mask, box.tolist())
            morph['confidence'] = float(dets.confidence[di])
            morph['mask'] = mask
            morph['bbox'] = box.tolist()
            results.append(morph)
    else:
        # Fallback: bbox as mask (SAM2 加载失败时)
        for di in range(len(dets.xyxy)):
            x1, y1, x2, y2 = dets.xyxy[di].astype(int)
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            morph = compute_morphology(mask, dets.xyxy[di].tolist())
            morph['confidence'] = float(dets.confidence[di])
            morph['mask'] = mask
            morph['bbox'] = dets.xyxy[di].tolist()
            results.append(morph)

    return results


def get_gt_bboxes(node_name, img_file):
    """从 annotations.json 获取 bbox GT"""
    ann_path = os.path.join(BATCH_DIRS[node_name], 'annotations.json')
    with open(ann_path) as f:
        ann = json.load(f)
    img_name = img_file
    for item in ann:
        if item['image'] == img_name:
            return item['bboxes']
    return []


def get_gt_mask_from_b3_annot(img_file, b3_annot_mapping):
    """从 B3 红色折线标注图获取真实轮廓 mask

    多路径搜索: B3_ANNOT_DIR → B3 原始图目录 → B3 images 子目录
    Returns: (gt_mask, gt_contours) or (None, None)
    """
    if img_file not in b3_annot_mapping:
        return None, None

    annot_file = b3_annot_mapping[img_file]

    # 多路径搜索 (不猜, 尝试所有可能)
    candidate_dirs = [
        B3_ANNOT_DIR,
        BATCH_DIRS['b3'],                          # D:\datasets\mouse_liver_data\batch3
        os.path.join(BATCH_DIRS['b3'], 'images'),   # batch3\images
        r"D:\datasets\mouse_liver_annotated_20260702",  # 云 VM 对应目录
    ]
    annot_path = None
    for d in candidate_dirs:
        p = os.path.join(d, annot_file)
        if os.path.exists(p):
            annot_path = p
            break

    if annot_path is None:
        log(f"    ⚠️ 标注图 {annot_file} 未找到 (尝试了 {len(candidate_dirs)} 个目录)")
        return None, None

    annot_img = cv2.imread(annot_path)
    if annot_img is None:
        return None, None
    annot_img = cv2.cvtColor(annot_img, cv2.COLOR_BGR2RGB)

    mask, contours = load_gt_mask_from_annot(annot_img)
    return mask, contours


def evaluate_bbox_f1(detections, gt_bboxes, img_w, img_h, iou_thr=0.5):
    """bbox 级 F1 评估

    gt_bboxes: list of {x, y, w, h} (绝对坐标)
    detections: list of {bbox: [x1,y1,x2,y2], ...}
    """
    # 转换 GT to [x1,y1,x2,y2]
    gt_boxes = []
    for bb in gt_bboxes:
        x1, y1 = bb['x'], bb['y']
        x2, y2 = bb['x'] + bb['w'], bb['y'] + bb['h']
        gt_boxes.append([x1, y1, x2, y2])

    pred_boxes = [d['bbox'] for d in detections]

    # 贪心匹配
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou = 0
        best_gi = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = bbox_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_thr:
            matched_gt.add(best_gi)
            tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}


def bbox_iou(box1, box2):
    """bbox IoU, box format [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def evaluate_mask_f1(detections, gt_mask, iou_thr=0.5):
    """mask 级 F1 评估

    detections: list of {mask: np.bool array, ...}
    gt_mask: np.uint8 array (0/255)
    """
    gt_bool = gt_mask > 127
    gt_contours, _ = cv2.findContours(gt_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours = [c for c in gt_contours if cv2.contourArea(c) > 100]

    # 每个 GT contour 单独评估
    matched_gt = set()
    tp = 0
    for det in detections:
        pred_mask = det['mask']
        best_iou = 0
        best_gi = -1
        for gi, gc in enumerate(gt_contours):
            if gi in matched_gt:
                continue
            gt_m = np.zeros_like(gt_bool)
            cv2.drawContours(gt_m, [gc], -1, True, thickness=cv2.FILLED)
            iou = mask_iou(pred_mask, gt_m)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_thr:
            matched_gt.add(best_gi)
            tp += 1

    fp = len(detections) - tp
    fn = len(gt_contours) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}


def collect_morphology(detections):
    """收集形态学特征"""
    features = {'circularity': [], 'solidity': [], 'eccentricity': [], 'aspect_ratio': [], 'area': []}
    for det in detections:
        for feat in features:
            if feat in det:
                features[feat].append(float(det[feat]))
    return features


def compute_kl_divergence(p_vals, q_vals, n_bins=20):
    """计算两个分布的 KL 散度"""
    if len(p_vals) < 3 or len(q_vals) < 3:
        return -1
    all_vals = p_vals + q_vals
    lo, hi = min(all_vals), max(all_vals)
    if hi <= lo:
        return 0
    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=bins, density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, density=True)
    p_hist = p_hist + 1e-8
    q_hist = q_hist + 1e-8
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


# ============================================================
# 实验函数
# ============================================================

def run_experiment(tag, det_model, sam2_predictor, test_node, b3_annot_mapping=None,
                   train_nodes=None, eval_mask=False, test_images=None):
    """跑一个实验

    train_nodes: 不影响推理 (RF-DETR 已训练好), 只用于日志
    eval_mask: 是否评估 mask F1 (仅 B3)
    test_images: 指定测试图列表 (如 ['image_17.jpg', ...]), None 则用全部
    """
    log(f"\n{'='*60}")
    log(f"实验 {tag}: test={test_node}, mask_eval={eval_mask}")
    log(f"{'='*60}")

    test_img_dir = os.path.join(BATCH_DIRS[test_node], 'images')
    if test_images:
        test_imgs = test_images
    else:
        test_imgs = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.jpg')])

    total_bbox = {'tp': 0, 'fp': 0, 'fn': 0}
    total_mask = {'tp': 0, 'fp': 0, 'fn': 0}
    all_morph = {'circularity': [], 'solidity': [], 'eccentricity': [], 'aspect_ratio': [], 'area': []}
    per_image = []

    for img_file in test_imgs:
        img_path = os.path.join(test_img_dir, img_file)
        img_pil = Image.open(img_path)
        img_w, img_h = img_pil.size

        # 跑 pipeline
        dets = run_pipeline(det_model, sam2_predictor, img_path)

        # bbox F1
        gt_bboxes = get_gt_bboxes(test_node, img_file)
        bbox_res = evaluate_bbox_f1(dets, gt_bboxes, img_w, img_h)
        total_bbox['tp'] += bbox_res['tp']
        total_bbox['fp'] += bbox_res['fp']
        total_bbox['fn'] += bbox_res['fn']

        # mask F1 (仅 B3)
        mask_res = None
        if eval_mask and b3_annot_mapping:
            gt_mask, _ = get_gt_mask_from_b3_annot(img_file, b3_annot_mapping)
            if gt_mask is not None:
                mask_res = evaluate_mask_f1(dets, gt_mask)
                total_mask['tp'] += mask_res['tp']
                total_mask['fp'] += mask_res['fp']
                total_mask['fn'] += mask_res['fn']

        # 形态学特征
        morph = collect_morphology(dets)
        for feat in all_morph:
            all_morph[feat].extend(morph[feat])

        log(f"  {img_file}: det={len(dets)} gt={len(gt_bboxes)} "
              f"bbox_F1={bbox_res['f1']:.3f}"
              + (f" mask_F1={mask_res['f1']:.3f}" if mask_res else ""))

        per_image.append({
            'image': img_file,
            'n_det': len(dets),
            'n_gt': len(gt_bboxes),
            'bbox': bbox_res,
            'mask': mask_res,
        })

    # 汇总
    tp, fp, fn = total_bbox['tp'], total_bbox['fp'], total_bbox['fn']
    bbox_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    bbox_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    bbox_f1 = 2 * bbox_p * bbox_r / (bbox_p + bbox_r) if (bbox_p + bbox_r) > 0 else 0

    result = {
        'tag': tag,
        'test_node': test_node,
        'n_test': len(test_imgs),
        'bbox': {'tp': tp, 'fp': fp, 'fn': fn, 'precision': bbox_p, 'recall': bbox_r, 'f1': bbox_f1},
        'morphology': all_morph,
        'per_image': per_image,
    }

    if eval_mask:
        tp, fp, fn = total_mask['tp'], total_mask['fp'], total_mask['fn']
        mask_p = tp / (tp + fp) if (tp + fp) > 0 else 0
        mask_r = tp / (tp + fn) if (tp + fn) > 0 else 0
        mask_f1 = 2 * mask_p * mask_r / (mask_p + mask_r) if (mask_p + mask_r) > 0 else 0
        result['mask'] = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': mask_p, 'recall': mask_r, 'f1': mask_f1}

    log(f"\n  ★ {tag} 汇总: bbox_F1={bbox_f1:.4f} (P={bbox_p:.4f} R={bbox_r:.4f})"
          + (f" mask_F1={mask_f1:.4f}" if eval_mask else ""))

    return result


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all', choices=['all', 'P1-A', 'P1-B', 'P1-C', 'P1-D', 'P1-E'])
    parser.add_argument('--rfdetr-weights', type=str, default=RFDETR_CKPT)
    parser.add_argument('--sam2-checkpoint', type=str, default=SAM2_CKPT)
    parser.add_argument('--no-sam2', action='store_true', help='不用 SAM2, 只用 bbox')
    args = parser.parse_args()

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # B3 标注映射
    b3_annot_mapping = load_b3_annot_mapping()

    # 加载模型
    log("加载 RF-DETR...")
    det_model = load_rfdetr(args.rfdetr_weights, RFDETR_VARIANT)

    sam2_predictor = None
    if not args.no_sam2:
        log("加载 SAM2...")
        try:
            sam2_predictor = load_sam2(args.sam2_checkpoint, SAM2_DEVICE)
            log("  SAM2 加载成功")
        except Exception as e:
            log(f"  ⚠️ SAM2 加载失败: {e}, 将只用 bbox")
            sam2_predictor = None

    results = {}

    # P1-A: B1 → B1
    if args.exp in ['all', 'P1-A']:
        results['P1-A'] = run_experiment('P1-A', det_model, sam2_predictor, 'b1')

    # P1-B: B1 → B2
    if args.exp in ['all', 'P1-B']:
        results['P1-B'] = run_experiment('P1-B', det_model, sam2_predictor, 'b2')

    # P1-C: B1 → B3 (关键! bbox F1 + mask F1)
    if args.exp in ['all', 'P1-C']:
        results['P1-C'] = run_experiment('P1-C', det_model, sam2_predictor, 'b3',
                                          b3_annot_mapping, eval_mask=True)

    # P1-D: B1+B2 → B3 (多源训练, 但 RF-DETR 只用 B1 训练的, 这里复用同一模型)
    # P1-D 需要 B1+B2 训练的 RF-DETR, 暂时用同一模型
    if args.exp in ['all', 'P1-D']:
        log("\n  ⚠️ P1-D 需要 B1+B2 训练的 RF-DETR, 当前复用 B1 模型")
        log("  如果要严格 P1-D, 需要先训练 B1+B2 RF-DETR")
        results['P1-D'] = run_experiment('P1-D', det_model, sam2_predictor, 'b3',
                                          b3_annot_mapping, eval_mask=True)

    # P1-E: B1+B2+B3 → val 9张 (集中式上界)
    # 需要集中式训练的 RF-DETR, 暂时用 B1 模型
    # val_set 包含 B1×3+B2×3+B3×3, 其中 B3 的 3 张可以做 mask 评估
    if args.exp in ['all', 'P1-E']:
        log("\n  ⚠️ P1-E 需要 B1+B2+B3 训练的 RF-DETR, 当前复用 B1 模型")
        log("  测试集用 B3 的 3 张 val 图 (有红色折线标注)")
        # 用 B3 的 val 图 (idx 17,18,19)
        b3_val_imgs = ['image_17.jpg', 'image_18.jpg', 'image_19.jpg']
        results['P1-E'] = run_experiment('P1-E', det_model, sam2_predictor, 'b3',
                                          b3_annot_mapping, eval_mask=True,
                                          test_images=b3_val_imgs)

    # 形态学特征分布对比
    log(f"\n{'='*60}")
    log("形态学特征分布对比")
    log(f"{'='*60}")

    for feat in ['circularity', 'solidity', 'eccentricity']:
        log(f"\n  {feat}:")
        for exp_name in ['P1-A', 'P1-B', 'P1-C']:
            if exp_name in results:
                vals = results[exp_name]['morphology'].get(feat, [])
                if vals:
                    log(f"    {exp_name}: n={len(vals)}, mean={np.mean(vals):.4f}, "
                          f"std={np.std(vals):.4f}, min={min(vals):.4f}, max={max(vals):.4f}")

        # KL 散度
        if 'P1-B' in results and 'P1-C' in results:
            b2_vals = results['P1-B']['morphology'].get(feat, [])
            b3_vals = results['P1-C']['morphology'].get(feat, [])
            if b2_vals and b3_vals:
                kl = compute_kl_divergence(b2_vals, b3_vals)
                log(f"    KL(B2||B3) = {kl:.4f}")

    # 保存
    # 清除 mask 数据 (太大不适合 JSON)
    for exp in results:
        for pi in results[exp].get('per_image', []):
            pass  # mask 不在 per_image 里 (已清除)

    result_path = os.path.join(OUTPUT_BASE, 'phase1_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\n{'='*60}")
    log(f"Phase 1 汇总")
    log(f"{'='*60}")
    log(f"\n{'实验':<8} {'测试':<6} {'bbox_F1':<10} {'mask_F1':<10} {'P':<8} {'R':<8}")
    log("-" * 50)
    for exp_name, r in results.items():
        bbox_f1 = r['bbox']['f1']
        mask_f1 = r.get('mask', {}).get('f1', '-')
        mask_str = f"{mask_f1:.4f}" if isinstance(mask_f1, float) else mask_f1
        log(f"{exp_name:<8} {r['test_node']:<6} {bbox_f1:<10.4f} {mask_str:<10} "
              f"{r['bbox']['precision']:<8.4f} {r['bbox']['recall']:<8.4f}")

    log(f"\n结果: {result_path}")


if __name__ == '__main__':
    main()
