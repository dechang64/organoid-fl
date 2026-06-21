#!/usr/bin/env python3
"""
多标注者共识标签生成 + CLOD 风格标签噪声清洗

功能:
1. 读取多标注者标签（t0/Annotator_A, t1_A, t1_B）
2. IoU 投票生成共识标签：
   - 高 IoU (>0.5) 的 bbox 跨标注者配对 → 共识 bbox（高置信度）
   - 仅一个标注者标记的 bbox → 不确定 bbox（低置信度）
   - 多标注者都不标记的区域 → 可能漏标
3. 输出清洗后的标签 + 噪声报告

Usage:
    python label_consensus.py --data D:\\datasets\\MultiOrg_v3_512_multi

    # 仅 train set
    python label_consensus.py --data D:\\datasets\\MultiOrg_v3_512_multi --split train
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import glob


def parse_yolo_label(lbl_path):
    """解析 YOLO 格式标签文件
    Returns: list of (class_id, xc, yc, w, h) tuples
    """
    bboxes = []
    if not os.path.exists(lbl_path):
        return bboxes
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                bboxes.append((cls, xc, yc, w, h))
    return bboxes


def yolo_to_xyxy(xc, yc, w, h):
    """YOLO → (x_min, y_min, x_max, y_max) normalized"""
    return (xc - w/2, yc - h/2, xc + w/2, yc + h/2)


def iou(box1, box2):
    """计算 IoU (normalized coords)"""
    x1a, y1a, x2a, y2a = yolo_to_xyxy(*box1[1:])
    x1b, y1b, x2b, y2b = yolo_to_xyxy(*box2[1:])

    ix1 = max(x1a, x1b)
    iy1 = max(y1a, y1b)
    ix2 = min(x2a, x2b)
    iy2 = min(y2a, y2b)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area1 = (x2a - x1a) * (y2a - y1a)
    area2 = (x2b - x1b) * (y2b - y1b)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def build_consensus(annotator_labels, iou_threshold=0.5):
    """多标注者共识标签生成。

    Args:
        annotator_labels: dict of {'annotator_a': [bboxes], 'annotator_b': [bboxes], ...}
        iou_threshold: IoU 阈值，超过此值视为同一目标

    Returns:
        consensus_bboxes: list of (cls, xc, yc, w, h, confidence, n_annotators)
        noise_report: dict with statistics
    """
    annotators = list(annotator_labels.keys())
    n_annotators = len(annotators)

    if n_annotators == 0:
        return [], {'total': 0, 'consensus': 0, 'single': 0, 'conflict': 0}

    if n_annotators == 1:
        # 只有一个标注者，全部视为 single
        bboxes = annotator_labels[annotators[0]]
        consensus = [(b[0], b[1], b[2], b[3], b[4], 0.5, 1) for b in bboxes]
        return consensus, {
            'total': len(bboxes),
            'consensus': 0,
            'single': len(bboxes),
            'conflict': 0,
            'n_annotators': 1,
        }

    # 多标注者匹配
    # 策略: 以第一个标注者为基准，逐一与其他标注者匹配
    matched = {a: set() for a in annotators}  # 已匹配的 bbox 索引
    consensus = []
    unmatched_all = {a: [] for a in annotators}  # 各标注者未匹配的 bbox

    # 第一个标注者的每个 bbox 去找其他标注者中 IoU 最高的
    base_annotator = annotators[0]
    base_bboxes = annotator_labels[base_annotator]

    for i, base_bbox in enumerate(base_bboxes):
        matches = [(base_annotator, i, base_bbox, 1.0)]  # (annotator, idx, bbox, iou)
        matched[base_annotator].add(i)

        for other_ann in annotators[1:]:
            if matched[other_ann]:
                other_indices = list(set(range(len(annotator_labels[other_ann]))) - matched[other_ann])
            else:
                other_indices = range(len(annotator_labels[other_ann]))

            best_iou = 0
            best_idx = -1
            for j in other_indices:
                if j in matched[other_ann]:
                    continue
                cur_iou = iou(base_bbox, annotator_labels[other_ann][j])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_idx = j

            if best_idx >= 0 and best_iou >= iou_threshold:
                matches.append((other_ann, best_idx, annotator_labels[other_ann][best_idx], best_iou))
                matched[other_ann].add(best_idx)

        # 计算共识 bbox
        n_matched = len(matches)
        if n_matched >= 2:
            # 多标注者同意 → 高置信度
            avg_xc = sum(m[2][1] for m in matches) / n_matched
            avg_yc = sum(m[2][2] for m in matches) / n_matched
            avg_w = sum(m[2][3] for m in matches) / n_matched
            avg_h = sum(m[2][4] for m in matches) / n_matched
            cls = matches[0][2][0]  # 使用第一个的类别
            confidence = n_matched / n_annotators
            consensus.append((cls, avg_xc, avg_yc, avg_w, avg_h, confidence, n_matched))
        else:
            # 仅一个标注者 → 低置信度（但保留）
            confidence = 1.0 / n_annotators
            consensus.append((base_bbox[0], base_bbox[1], base_bbox[2],
                            base_bbox[3], base_bbox[4], confidence, 1))

    # 收集未匹配的 bbox（其他标注者标记但 base 没有）
    for other_ann in annotators[1:]:
        for j in range(len(annotator_labels[other_ann])):
            if j not in matched[other_ann]:
                bbox = annotator_labels[other_ann][j]
                confidence = 1.0 / n_annotators
                consensus.append((bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], confidence, 1))

    # 噪声报告
    consensus_count = sum(1 for c in consensus if c[6] >= 2)
    single_count = sum(1 for c in consensus if c[6] == 1)

    return consensus, {
        'total': len(consensus),
        'consensus': consensus_count,
        'single': single_count,
        'n_annotators': n_annotators,
    }


def process_split(data_dir, split, iou_threshold=0.5):
    """处理一个 split 的所有 patch。"""
    img_dir = os.path.join(data_dir, split, 'images')
    if not os.path.exists(img_dir):
        print(f"[WARN] {img_dir} not found")
        return

    # 找到所有标注者标签目录
    lbl_dirs = {}
    for d in os.listdir(os.path.join(data_dir, split)):
        if d.startswith('labels'):
            ann_key = d.replace('labels', '').strip('_') or 'default'
            lbl_dirs[ann_key] = os.path.join(data_dir, split, d)

    if not lbl_dirs:
        print(f"[WARN] No label directories found in {data_dir}/{split}/")
        return

    print(f"\n  Annotators found: {list(lbl_dirs.keys())}")

    # 输出目录
    consensus_dir = os.path.join(data_dir, split, 'labels_consensus')
    os.makedirs(consensus_dir, exist_ok=True)

    # 处理每个 patch
    images = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    total_stats = {'total': 0, 'consensus': 0, 'single': 0, 'n_annotators': 0}
    per_patch_report = []

    for img_path in images:
        stem = Path(img_path).stem

        # 加载各标注者标签
        annotator_labels = {}
        for ann_key, lbl_dir in lbl_dirs.items():
            lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
            bboxes = parse_yolo_label(lbl_path)
            if bboxes:
                annotator_labels[ann_key] = bboxes

        if not annotator_labels:
            # 空标签也写一个空文件
            open(os.path.join(consensus_dir, f"{stem}.txt"), 'w').close()
            continue

        # 生成共识标签
        consensus, stats = build_consensus(annotator_labels, iou_threshold)

        # 写入共识标签
        consensus_path = os.path.join(consensus_dir, f"{stem}.txt")
        with open(consensus_path, 'w') as f:
            for cls, xc, yc, w, h, conf, n_ann in consensus:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # 更新统计
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

        per_patch_report.append({
            'patch': stem,
            'n_bboxes': len(consensus),
            'n_consensus': stats['consensus'],
            'n_single': stats['single'],
        })

    # 报告
    print(f"\n  Consensus labels generated: {consensus_dir}")
    print(f"  Total bboxes: {total_stats['total']}")
    print(f"  Consensus (≥2 annotators): {total_stats['consensus']} "
          f"({total_stats['consensus']/max(1,total_stats['total'])*100:.1f}%)")
    print(f"  Single annotator only: {total_stats['single']} "
          f"({total_stats['single']/max(1,total_stats['total'])*100:.1f}%)")

    return total_stats, per_patch_report


def main():
    parser = argparse.ArgumentParser(
        description='MultiOrg Multi-Rater Label Consensus + CLOD-style Cleaning'
    )
    parser.add_argument('--data', required=True,
                        help='MultiOrg v3 dataset directory (with labels_* subdirs)')
    parser.add_argument('--split', default='all', choices=['all', 'train', 'test'])
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for consensus matching')
    args = parser.parse_args()

    print(f"Dataset: {args.data}")
    print(f"IoU threshold: {args.iou_threshold}")
    print("=" * 60)

    splits = ['train', 'test'] if args.split == 'all' else [args.split]
    all_stats = {}

    for split in splits:
        print(f"\n--- Processing {split} ---")
        stats, report = process_split(args.data, split, args.iou_threshold)
        all_stats[split] = stats

    # 保存汇总报告
    report_path = os.path.join(args.data, 'consensus_report.json')
    with open(report_path, 'w') as f:
        json.dump({
            'iou_threshold': args.iou_threshold,
            'stats': all_stats,
        }, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Report saved: {report_path}")

    # 输出 data.yaml for consensus labels
    yaml_path = os.path.join(args.data, 'data_consensus.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"""# MultiOrg Consensus Labels (CLOD-cleaned)
path: {args.data.replace(chr(92), '/')}
train: train/images
val: test/images

nc: 1
names: ["organoid"]

# Use consensus labels:
# train: train/labels_consensus
# val: test/labels_consensus
""")
    print(f"Data YAML: {yaml_path}")
    print(f"\nTrain with consensus labels:")
    print(f"  yolo detect train model=yolo12s.pt "
          f"data={yaml_path} epochs=400 imgsz=512 batch=8")


if __name__ == '__main__':
    main()
