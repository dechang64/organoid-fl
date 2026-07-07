r"""
鼠肝 organoid 评估脚本 — bbox P/R/F1

从 test set 的 YOLO labels 读取 GT, 用 RF-DETR 检测
计算 bbox P/R/F1 (IoU=0.5)

mask 评估需要标注图 (红色轮廓), 用 eval_new_testset.py 单独跑

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate

    python scripts\mouse_liver\v2\evaluate.py --batch b1 --weights runs\mouse_liver_v2\b1\full\checkpoint_best_regular.pth --data-root D:\datasets\mouse_liver_split

输出:
    runs\mouse_liver_v2\{batch}\{mode}\eval_test.json
"""
import argparse
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image


BATCH_RESOLUTION = {
    'b1': 544,
    'b2': 768,
    'b3': 768,
}


def load_yolo_gt(label_path, img_w, img_h):
    """从 YOLO label 文件读取 GT bboxes"""
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
    with open(label_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, xc, yc, bw, bh = parts
                xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
                x1 = (xc - bw / 2) * img_w
                y1 = (yc - bh / 2) * img_h
                x2 = (xc + bw / 2) * img_w
                y2 = (yc + bh / 2) * img_h
                gt_boxes.append([x1, y1, x2, y2])
    return gt_boxes


def bbox_iou(box1, box2):
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)


def match_detections(det_boxes, det_scores, gt_boxes, iou_thr=0.5):
    """贪心匹配 TP/FP/FN"""
    # 按置信度排序检测
    indices = np.argsort(det_scores)[::-1]
    matched_gt = set()
    tp = 0
    for di in indices:
        best_iou, best_gi = 0, -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = bbox_iou(det_boxes[di], gb)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou > iou_thr and best_gi >= 0:
            tp += 1
            matched_gt.add(best_gi)
    fp = len(det_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, matched_gt


def evaluate_batch(batch_name, weights_path, data_root, output_base,
                   threshold=0.25, model_variant='small', tag=None, resolution_override=None):
    """评估单个 batch 的 test set"""
    from rfdetr import RFDETRSmall, RFDETRNano, RFDETRBase
    model_map = {'nano': RFDETRNano, 'small': RFDETRSmall, 'base': RFDETRBase}
    ModelClass = model_map.get(model_variant, RFDETRSmall)

    test_dir = Path(data_root) / batch_name / 'test'
    test_img_dir = test_dir / 'images'
    test_lbl_dir = test_dir / 'labels'

    if not test_img_dir.exists():
        raise FileNotFoundError(f"Test images not found: {test_img_dir}")

    # resolution: 用 override 或默认 BATCH_RESOLUTION
    resolution = resolution_override if resolution_override else BATCH_RESOLUTION[batch_name]

    print(f"\n{'='*60}")
    print(f"Evaluating {batch_name} test set")
    print(f"{'='*60}")
    print(f"  Weights: {weights_path}")
    print(f"  Resolution: {resolution}")
    print(f"  Test images: {test_img_dir}")

    # Load RF-DETR
    print("Loading RF-DETR...")
    det_model = ModelClass(pretrain_weights=weights_path, num_classes=1)

    # Process each test image
    images = sorted(test_img_dir.glob('*.jpg')) + sorted(test_img_dir.glob('*.png'))
    total_tp, total_fp, total_fn = 0, 0, 0
    per_image = []

    for img_path in images:
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert('RGB'))
        h, w = img_np.shape[:2]

        # Load GT
        lbl_path = test_lbl_dir / (img_path.stem + '.txt')
        gt_boxes = load_yolo_gt(str(lbl_path), w, h)

        # RF-DETR detection
        dets = det_model.predict(img_pil, threshold=threshold, shape=(resolution, resolution))

        det_boxes = []
        det_scores = []
        if hasattr(dets, 'xyxy') and len(dets.xyxy) > 0:
            for di in range(len(dets.xyxy)):
                det_boxes.append(list(dets.xyxy[di]))
                det_scores.append(float(dets.confidence[di]) if hasattr(dets, 'confidence') else 1.0)

        # Match TP/FP/FN
        tp, fp, fn, matched_gt = match_detections(det_boxes, det_scores, gt_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_image.append({
            'image': img_path.name,
            'n_gt': len(gt_boxes),
            'n_det': len(det_boxes),
            'tp': tp, 'fp': fp, 'fn': fn,
        })

    # Summary
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    results = {
        'batch': batch_name,
        'weights': weights_path,
        'resolution': resolution,
        'n_images': len(images),
        'threshold': threshold,
        'bbox': {
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'precision': prec, 'recall': rec, 'f1': f1,
        },
        'per_image': per_image,
    }

    # 判断输出目录: 用 tag 或从 weights_path 推断
    if tag:
        mode = tag
    else:
        mode = 'full' if 'full' in str(weights_path).replace('/', os.sep) else 'fewshot'

    # Save results
    output_dir = Path(output_base) / batch_name / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'eval_test.json'

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"RESULTS: {batch_name} ({mode})")
    print(f"{'='*60}")
    print(f"  Images: {len(images)}")
    print(f"  BBox: P={prec*100:.1f}%  R={rec*100:.1f}%  F1={f1*100:.1f}%")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Results: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate RF-DETR bbox on test set')
    parser.add_argument('--batch', required=True, choices=['b1', 'b2', 'b3', 'all'])
    parser.add_argument('--weights', required=True, help='RF-DETR checkpoint')
    parser.add_argument('--data-root', default=r'D:\datasets\mouse_liver_split')
    parser.add_argument('--output', default='runs/mouse_liver_v2')
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--model-variant', default='small', choices=['nano', 'small', 'base'])
    parser.add_argument('--tag', default=None, help='Custom tag for output dir')
    parser.add_argument('--resolution', type=int, default=None, help='Override inference resolution (e.g. 640 for central)')
    args = parser.parse_args()

    batches = ['b1', 'b2', 'b3'] if args.batch == 'all' else [args.batch]

    for batch in batches:
        evaluate_batch(batch, args.weights, args.data_root, args.output,
                       args.threshold, args.model_variant, args.tag, args.resolution)


if __name__ == '__main__':
    main()
