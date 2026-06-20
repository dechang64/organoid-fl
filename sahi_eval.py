"""
SAHI 切片推理评估脚本
用法: python sahi_eval.py
依赖: pip install sahi ultralytics
"""
import json
from pathlib import Path
import yaml
import torch
import numpy as np
from PIL import Image as PILImage
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ============ 配置 ============
MODEL_PATH = r"C:\Users\decha\organoid-fl\runs\detect\intestinal_12s_1088_freebies\weights\best.pt"
DATA_YAML = r"C:\Users\decha\organoid-fl\data\intestinal_organoid\OrganoidDataset\data.yaml"
SLICE_SIZE = 512
OVERLAP_RATIO = 0.2
CONF_THRESHOLD = 0.25
DEVICE = "cuda:0"

# ============ 加载模型 ============
print("加载模型...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=CONF_THRESHOLD,
    device=DEVICE,
)

# ============ 读取 data.yaml ============
with open(DATA_YAML, 'r') as f:
    data_cfg = yaml.safe_load(f)
base_path = Path(data_cfg.get('path', str(Path(DATA_YAML).parent)))
val_rel = data_cfg.get('val', 'val')
# 去掉可能的 ../ 或绝对前缀
if val_rel.startswith('../'):
    val_rel = val_rel[3:]
img_dir = base_path / val_rel / "images"
label_dir = base_path / val_rel / "labels"
class_names = data_cfg.get('names', {0: 'organoid0', 1: 'organoid1', 2: 'organoid2', 3: 'organoid3'})

print(f"图片目录: {img_dir}")
print(f"标签目录: {label_dir}")
print(f"类别: {class_names}")

# ============ 收集图片 ============
img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg')])
print(f"共 {len(img_files)} 张验证图片\n")

# ============ IoU 工具 ============
def box_iou(box1, box2):
    """box1: [N,4] xyxy, box2: [M,4] xyxy -> [N,M]"""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    return inter / (area1[:, None] + area2 - inter + 1e-9)

# ============ AP 计算（单类单IoU） ============
def compute_ap_single(img_preds, img_gts, cls_id, iou_threshold):
    """
    img_preds: list of dict {cls_id: [[x1,y1,x2,y2,score], ...]} per image
    img_gts:   list of dict {cls_id: [[x1,y1,x2,y2], ...]} per image
    返回 AP@该IoU
    """
    scores_list = []
    tp_fp_list = []
    n_gt_total = 0

    for idx in range(len(img_preds)):
        preds = img_preds[idx].get(cls_id, [])
        gts = img_gts[idx].get(cls_id, [])
        n_gt_total += len(gts)

        if len(gts) == 0:
            for p in preds:
                scores_list.append(p[4])
                tp_fp_list.append(0)
            continue

        gt_boxes = torch.tensor(gts, dtype=torch.float32)
        used = torch.zeros(len(gts))

        # 按 score 降序
        pred_scores = [p[4] for p in preds]
        order = torch.argsort(torch.tensor(pred_scores), descending=True)

        for oi in order:
            scores_list.append(pred_scores[oi])
            pred_box = torch.tensor([preds[oi][:4]], dtype=torch.float32)
            ious = box_iou(pred_box, gt_boxes)[0]
            best_iou, best_gt = ious.max(0)
            if best_iou >= iou_threshold and not used[best_gt]:
                tp_fp_list.append(1)
                used[best_gt] = 1
            else:
                tp_fp_list.append(0)

    if n_gt_total == 0:
        return 0.0, 0

    scores = torch.tensor(scores_list)
    tp_fp = torch.tensor(tp_fp_list)
    order = torch.argsort(scores, descending=True)
    tp_fp = tp_fp[order]

    tp_cum = torch.cumsum(tp_fp, 0)
    fp_cum = torch.cumsum(1 - tp_fp, 0)
    recall = tp_cum / n_gt_total
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)

    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recall >= t
        ap += precision[mask].max().item() if mask.any() else 0
    return ap / 101, n_gt_total

# ============ SAHI 推理 + 收集 GT ============
all_preds = []  # [{cls_id: [[x1,y1,x2,y2,score], ...]}, ...]
all_gts = []    # [{cls_id: [[x1,y1,x2,y2], ...]}, ...]

print("开始 SAHI 切片推理...")
for i, img_file in enumerate(img_files):
    # --- SAHI 推理 ---
    result = get_sliced_prediction(
        str(img_file),
        detection_model,
        slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO, overlap_width_ratio=OVERLAP_RATIO,
        postprocess_type="NMS",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        verbose=0,
    )

    pred_dict = {}
    for obj in result.object_prediction_list:
        b = obj.bbox
        cid = obj.category.id
        pred_dict.setdefault(cid, []).append(
            [b.minx, b.miny, b.maxx, b.maxy, obj.score.value]
        )
    all_preds.append(pred_dict)

    # --- 解析 GT (YOLO txt 格式) ---
    label_file = label_dir / (img_file.stem + ".txt")
    gt_dict = {}
    if label_file.exists():
        with PILImage.open(img_file) as im:
            img_w, img_h = im.size
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1 = (cx - w/2) * img_w
                y1 = (cy - h/2) * img_h
                x2 = (cx + w/2) * img_w
                y2 = (cy + h/2) * img_h
                gt_dict.setdefault(cls, []).append([x1, y1, x2, y2])
    all_gts.append(gt_dict)

    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(img_files)}] {img_file.name}")

print(f"\n推理完成，开始计算 mAP...\n")

# ============ 计算 mAP ============
results = {}
iou_thresholds = np.arange(0.5, 1.0, 0.05)

for cls_id in sorted(class_names.keys()):
    cls_name = class_names[cls_id]

    # mAP@50
    ap50, n_gt = compute_ap_single(all_preds, all_gts, cls_id, 0.5)

    # mAP@50:95
    aps = []
    for iou_th in iou_thresholds:
        ap, _ = compute_ap_single(all_preds, all_gts, cls_id, iou_th)
        aps.append(ap)
    map5095 = float(np.mean(aps))

    results[cls_name] = {
        'mAP50': round(ap50, 4),
        'mAP50-95': round(map5095, 4),
        'n_gt': n_gt,
    }
    print(f"  {cls_name:>12s}: mAP50={ap50:.4f}  mAP50-95={map5095:.4f}  (n_gt={n_gt})")

# 汇总（按 n_gt 加权的宏平均）
all_map50 = float(np.mean([r['mAP50'] for r in results.values()]))
all_map5095 = float(np.mean([r['mAP50-95'] for r in results.values()]))
print(f"\n  {'all':>12s}: mAP50={all_map50:.4f}  mAP50-95={all_map5095:.4f}")
print(f"\n  Baseline (v1 best.pt 直接推理): mAP50=0.8810  mAP50-95=0.6210")
print(f"  提升: mAP50 {all_map50-0.8810:+.4f}  mAP50-95 {all_map5095-0.6210:+.4f}")

# ============ 保存 ============
output = {
    'method': f'SAHI(slice={SLICE_SIZE},overlap={OVERLAP_RATIO})',
    'model': MODEL_PATH,
    'n_images': len(img_files),
    'overall': {'mAP50': round(all_map50, 4), 'mAP50-95': round(all_map5095, 4)},
    'per_class': results,
    'baseline': {'mAP50': 0.8810, 'mAP50-95': 0.6210},
}
out_path = Path(__file__).parent / 'sahi_results.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到 {out_path}")
