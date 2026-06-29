r"""
FP 标注遗漏分析 + napari 导出

从检测结果中提取高 confidence FP，分析它们到最近 GT 的距离，
导出 napari 可导入的 JSON 格式供人工目视确认。

用法（Windows，单行命令）：
cd C:\Users\decha\organoid-fl
python scripts\multiorg\fp_missed_annotation_analysis.py --results-json results\multiorg_sam2_zeroshot\multiorg_sam2_results.json --data-root D:\datasets\mutliorg\MultiOrg_v2 --annotator t1_b --output-dir results\fp_missed_analysis --iou-threshold 0.5

输出：
    1. fp_gt_distance_analysis.json  - FP 到最近 GT 的距离统计
    2. missed_candidates_napari.json  - napari 可导入的漏标候选（每张图）
    3. missed_candidates_summary.json - 汇总统计
    4. fp_distance_hist.png           - 距离分布直方图
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def find_annotation_for_image(img_dir, annotator='t1_b'):
    """找到图像对应的标注文件（和 sahi_inference.py 一致）"""
    target = annotator.lower()
    for f in os.listdir(img_dir):
        if not f.lower().endswith('.json'):
            continue
        if target in f.lower():
            return os.path.join(img_dir, f)
    if annotator == 'any':
        for f in os.listdir(img_dir):
            if f.lower().endswith('.json'):
                return os.path.join(img_dir, f)
    return None


def load_gt_bboxes(json_path):
    """加载 GT 标注（napari [row,col] 格式）→ bbox list + polygon list
    
    返回:
        bboxes: [(x_min, y_min, x_max, y_max), ...]
        centers: [(cx, cy), ...]
        polygons: [[[y1,x1],[y2,x2],...], ...]  (原始 napari 格式)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    bboxes = []
    centers = []
    polygons = []
    
    for key, polygon in data.items():
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        # napari: [row, col] = [y, x]
        ys = [p[0] for p in polygon]
        xs = [p[1] for p in polygon]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        bboxes.append((x_min, y_min, x_max, y_max))
        centers.append((cx, cy))
        polygons.append(polygon)
    
    return bboxes, centers, polygons


def compute_iou(box1, box2):
    """IoU of two boxes (x_min, y_min, x_max, y_max)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def bbox_to_napari_polygon(bbox, padding=0):
    """bbox (x_min, y_min, x_max, y_max) → napari 4点多边形 [row, col] = [y, x]"""
    x_min, y_min, x_max, y_max = bbox
    if padding > 0:
        x_min -= padding
        y_min -= padding
        x_max += padding
        y_max += padding
    # napari 格式: [row, col] = [y, x]
    return [
        [y_min, x_min],
        [y_min, x_max],
        [y_max, x_max],
        [y_max, x_min],
    ]


def analyze_fp_to_gt(results, data_root, annotator, iou_threshold):
    """分析每个 FP 到最近 GT 的关系"""
    
    # 检测 data_root/test 还是 data_root
    test_dir = os.path.join(data_root, 'test')
    if not os.path.isdir(test_dir):
        test_dir = data_root
    
    all_fp_analysis = []
    missed_candidates = defaultdict(list)
    
    for img_res in results['per_image']:
        img_label = img_res['image']
        class_name, plate, img_name = img_label.split('/')
        
        img_dir = os.path.join(test_dir, class_name, plate, img_name)
        if not os.path.isdir(img_dir):
            print(f"  [WARN] Dir not found: {img_dir}")
            continue
        
        gt_path = find_annotation_for_image(img_dir, annotator)
        if not gt_path:
            print(f"  [WARN] No GT for: {img_label}")
            continue
        
        gt_bboxes, gt_centers, gt_polygons = load_gt_bboxes(gt_path)
        
        # 从 detections 中用 IoU 匹配区分 TP/FP
        detections = img_res['detections']
        det_sorted = sorted(detections, key=lambda d: -d['confidence'])
        
        n_gt = len(gt_bboxes)
        matched_gt = [False] * n_gt
        
        tp_dets = []
        fp_dets = []
        
        for det in det_sorted:
            det_bbox = tuple(det['bbox'])
            best_iou = 0
            best_gt = -1
            for i, gt_bbox in enumerate(gt_bboxes):
                if matched_gt[i]:
                    continue
                iou = compute_iou(det_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = i
            
            if best_iou >= iou_threshold and best_gt >= 0:
                matched_gt[best_gt] = True
                tp_dets.append(det)
            else:
                fp_dets.append(det)
        
        # 对每个 FP，找最近的 GT
        for fp in fp_dets:
            fp_bbox = fp['bbox']
            fp_cx = (fp_bbox[0] + fp_bbox[2]) / 2
            fp_cy = (fp_bbox[1] + fp_bbox[3]) / 2
            fp_w = fp_bbox[2] - fp_bbox[0]
            fp_h = fp_bbox[3] - fp_bbox[1]
            fp_diag = np.sqrt(fp_w**2 + fp_h**2)
            
            # 到所有 GT 的距离
            min_dist = float('inf')
            min_gt_idx = -1
            best_iou_with_any_gt = 0
            
            for i, (gt_cx, gt_cy) in enumerate(gt_centers):
                dist = np.sqrt((fp_cx - gt_cx)**2 + (fp_cy - gt_cy)**2)
                if dist < min_dist:
                    min_dist = dist
                    min_gt_idx = i
                    best_iou_with_any_gt = compute_iou(fp_bbox, gt_bboxes[i])
            
            # 判断分类
            if min_gt_idx >= 0:
                gt_w = gt_bboxes[min_gt_idx][2] - gt_bboxes[min_gt_idx][0]
                gt_h = gt_bboxes[min_gt_idx][3] - gt_bboxes[min_gt_idx][1]
                gt_size = max(gt_w, gt_h)
                dist_ratio = min_dist / gt_size if gt_size > 0 else float('inf')
            else:
                gt_size = 0
                dist_ratio = float('inf')
            
            # 分类逻辑:
            # - IoU 0.3-0.5: "partial overlap" — 可能是标注位置偏
            # - 距离 < fp_diag: "near miss" — 靠近已知 organoid
            # - 距离 < 2*fp_diag: "vicinity" — 在 organoid 附近
            # - 距离 > 2*fp_diag: "isolated" — 真正的孤立 FP
            if best_iou_with_any_gt >= 0.3:
                category = "partial_overlap"
            elif min_dist < fp_diag:
                category = "near_miss"
            elif min_dist < 2 * fp_diag:
                category = "vicinity"
            else:
                category = "isolated"
            
            entry = {
                'image': img_label,
                'bbox': fp_bbox,
                'confidence': fp['confidence'],
                'area': fp.get('area', 0),
                'circularity': fp.get('circularity', 0),
                'nearest_gt_dist': min_dist,
                'nearest_gt_iou': best_iou_with_any_gt,
                'dist_ratio': dist_ratio,
                'fp_diag': fp_diag,
                'category': category,
            }
            all_fp_analysis.append(entry)
            
            # 收集漏标候选：高 confidence + partial_overlap 或 near_miss
            if fp['confidence'] > 0.4 and category in ('partial_overlap', 'near_miss'):
                missed_candidates[img_label].append({
                    'bbox': fp_bbox,
                    'confidence': fp['confidence'],
                    'napari_polygon': bbox_to_napari_polygon(fp_bbox),
                    'nearest_gt_dist': min_dist,
                    'nearest_gt_iou': best_iou_with_any_gt,
                    'category': category,
                })
    
    return all_fp_analysis, missed_candidates


def export_napari_json(missed_candidates, output_path):
    """导出 napari 可导入的 JSON
    
    格式和原始 GT 一致:
    {
        "0": [[y1,x1],[y2,x2],[y3,x3],[y4,x4]],
        "1": [[y1,x1],[y2,x2],[y3,x3],[y4,x4]],
        ...
    }
    
    每张图一个文件，方便在 napari 中逐张查看
    """
    os.makedirs(output_path, exist_ok=True)
    
    total_exported = 0
    for img_label, candidates in missed_candidates.items():
        if not candidates:
            continue
        
        # napari JSON 格式
        napari_data = {}
        for i, cand in enumerate(candidates):
            napari_data[str(i)] = cand['napari_polygon']
        
        # 文件名: class_plate_image_missed.json
        safe_name = img_label.replace('/', '_')
        out_file = os.path.join(output_path, f"{safe_name}_missed.json")
        with open(out_file, 'w') as f:
            json.dump(napari_data, f, indent=2)
        
        total_exported += len(candidates)
    
    return total_exported


def plot_distance_hist(all_fp_analysis, output_path):
    """画 FP-to-GT 距离分布直方图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plot")
        return
    
    dists = [fp['nearest_gt_dist'] for fp in all_fp_analysis]
    dist_ratios = [fp['dist_ratio'] for fp in all_fp_analysis if fp['dist_ratio'] < 10]
    ious = [fp['nearest_gt_iou'] for fp in all_fp_analysis]
    confs = [fp['confidence'] for fp in all_fp_analysis]
    categories = [fp['category'] for fp in all_fp_analysis]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 距离分布
    ax = axes[0, 0]
    ax.hist(dists, bins=100, color='steelblue', edgecolor='white', alpha=0.7)
    ax.set_xlabel('Distance to nearest GT (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('FP-to-GT Distance Distribution')
    ax.set_yscale('log')
    
    # 2. 距离/GT size 比值
    ax = axes[0, 1]
    ax.hist(dist_ratios, bins=100, color='coral', edgecolor='white', alpha=0.7)
    ax.axvline(x=1.0, color='red', linestyle='--', label='dist = GT size')
    ax.axvline(x=2.0, color='orange', linestyle='--', label='dist = 2x GT size')
    ax.set_xlabel('Distance / GT size')
    ax.set_ylabel('Count')
    ax.set_title('Distance Ratio (dist / nearest GT size)')
    ax.set_yscale('log')
    ax.legend()
    
    # 3. IoU 分布
    ax = axes[1, 0]
    ax.hist(ious, bins=50, color='forestgreen', edgecolor='white', alpha=0.7)
    ax.axvline(x=0.3, color='red', linestyle='--', label='IoU=0.3 (partial)')
    ax.axvline(x=0.5, color='orange', linestyle='--', label='IoU=0.5 (TP threshold)')
    ax.set_xlabel('IoU with nearest GT')
    ax.set_ylabel('Count')
    ax.set_title('FP-GT IoU Distribution')
    ax.set_yscale('log')
    ax.legend()
    
    # 4. 分类统计
    ax = axes[1, 1]
    cat_counts = {}
    for c in categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    colors = {'partial_overlap': 'red', 'near_miss': 'orange', 'vicinity': 'yellow', 'isolated': 'gray'}
    bars = ax.bar(cat_counts.keys(), cat_counts.values(), 
                   color=[colors.get(k, 'blue') for k in cat_counts.keys()])
    ax.set_ylabel('Count')
    ax.set_title('FP Category Distribution')
    for bar, count in zip(bars, cat_counts.values()):
        pct = count / len(categories) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='FP missed annotation analysis')
    parser.add_argument('--results-json', required=True,
                        help='Path to multiorg_sam2_results.json')
    parser.add_argument('--data-root', required=True,
                        help='MultiOrg_v2 root (auto-detects test/ subdir)')
    parser.add_argument('--annotator', default='t1_b',
                        help='GT annotator (default: t1_b)')
    parser.add_argument('--output-dir', default='results/fp_missed_analysis',
                        help='Output directory')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for TP/FP (default 0.5, matches mAP50)')
    parser.add_argument('--conf-threshold', type=float, default=0.4,
                        help='Min confidence for missed annotation candidates')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载 results
    print(f"\n{'='*60}")
    print(f"=== Loading results: {args.results_json} ===")
    with open(args.results_json) as f:
        results = json.load(f)
    
    print(f"  Images: {len(results['per_image'])}")
    print(f"  Baseline: TP={results['baseline']['tp']}, FP={results['baseline']['fp']}")
    
    # 2. 分析 FP 到 GT 的距离
    print(f"\n{'='*60}")
    print(f"=== Analyzing FP-to-GT relationships ===")
    print(f"  Data root: {args.data_root}")
    print(f"  Annotator: {args.annotator}")
    print(f"  IoU threshold: {args.iou_threshold}")
    
    all_fp_analysis, missed_candidates = analyze_fp_to_gt(
        results, args.data_root, args.annotator, args.iou_threshold
    )
    
    print(f"\n  Total FP analyzed: {len(all_fp_analysis)}")
    
    # 3. 分类统计
    cat_counts = defaultdict(int)
    for fp in all_fp_analysis:
        cat_counts[fp['category']] += 1
    
    print(f"\n=== FP Category Distribution ===")
    for cat in ['partial_overlap', 'near_miss', 'vicinity', 'isolated']:
        count = cat_counts[cat]
        pct = count / len(all_fp_analysis) * 100 if all_fp_analysis else 0
        print(f"  {cat:>20}: {count:>6} ({pct:.1f}%)")
    
    # 4. 距离统计
    dists = [fp['nearest_gt_dist'] for fp in all_fp_analysis]
    ious = [fp['nearest_gt_iou'] for fp in all_fp_analysis]
    dist_ratios = [fp['dist_ratio'] for fp in all_fp_analysis if fp['dist_ratio'] < 100]
    
    print(f"\n=== FP-to-Nearest-GT Distance ===")
    print(f"  mean={np.mean(dists):.1f}, median={np.median(dists):.1f}")
    print(f"  p10={np.percentile(dists, 10):.1f}, p90={np.percentile(dists, 90):.1f}")
    
    print(f"\n=== FP-to-Nearest-GT IoU ===")
    print(f"  mean={np.mean(ious):.3f}, median={np.median(ious):.3f}")
    iou_above_03 = sum(1 for i in ious if i >= 0.3)
    iou_above_02 = sum(1 for i in ious if i >= 0.2)
    print(f"  IoU >= 0.3: {iou_above_03} ({iou_above_03/len(ious)*100:.1f}%)")
    print(f"  IoU >= 0.2: {iou_above_02} ({iou_above_02/len(ious)*100:.1f}%)")
    
    # 5. 漏标候选
    total_candidates = sum(len(v) for v in missed_candidates.values())
    print(f"\n=== Missed Annotation Candidates (conf > {args.conf_threshold}) ===")
    print(f"  Total candidates: {total_candidates}")
    print(f"  Images with candidates: {len(missed_candidates)}")
    
    # 按 category 统计候选
    cand_cats = defaultdict(int)
    for candidates in missed_candidates.values():
        for c in candidates:
            cand_cats[c['category']] += 1
    for cat, count in sorted(cand_cats.items()):
        print(f"    {cat}: {count}")
    
    # 6. 导出 napari JSON
    napari_dir = output_dir / 'napari_missed'
    total_exported = export_napari_json(missed_candidates, str(napari_dir))
    print(f"\n=== Exported napari JSON ===")
    print(f"  Directory: {napari_dir}")
    print(f"  Files: {len(list(napari_dir.glob('*.json')))}")
    print(f"  Total candidates exported: {total_exported}")
    
    # 7. 保存分析结果
    analysis_path = output_dir / 'fp_gt_distance_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump({
            'config': {
                'results_json': args.results_json,
                'data_root': args.data_root,
                'annotator': args.annotator,
                'iou_threshold': args.iou_threshold,
                'conf_threshold': args.conf_threshold,
            },
            'summary': {
                'total_fp': len(all_fp_analysis),
                'categories': dict(cat_counts),
                'distance_mean': float(np.mean(dists)),
                'distance_median': float(np.median(dists)),
                'iou_mean': float(np.mean(ious)),
                'iou_median': float(np.median(ious)),
                'iou_above_03': iou_above_03,
                'iou_above_02': iou_above_02,
                'missed_candidates_total': total_candidates,
            },
            'fp_analysis': all_fp_analysis,
        }, f, indent=2)
    print(f"\n  Saved: {analysis_path}")
    
    # 8. 保存汇总
    summary_path = output_dir / 'missed_candidates_summary.json'
    per_image_summary = {}
    for img_label, candidates in missed_candidates.items():
        per_image_summary[img_label] = {
            'n_candidates': len(candidates),
            'conf_mean': float(np.mean([c['confidence'] for c in candidates])),
            'categories': dict(Counter(c['category'] for c in candidates)),
        }
    with open(summary_path, 'w') as f:
        json.dump({
            'total_candidates': total_candidates,
            'conf_threshold': args.conf_threshold,
            'per_image': per_image_summary,
        }, f, indent=2)
    print(f"  Saved: {summary_path}")
    
    # 9. 画图
    plot_path = output_dir / 'fp_distance_hist.png'
    plot_distance_hist(all_fp_analysis, str(plot_path))
    
    # 10. 总结
    print(f"\n{'='*60}")
    print(f"=== SUMMARY ===")
    print(f"{'='*60}")
    print(f"  Total FP: {len(all_fp_analysis)}")
    print(f"  Partial overlap (IoU 0.3-0.5): {cat_counts['partial_overlap']} ({cat_counts['partial_overlap']/len(all_fp_analysis)*100:.1f}%)")
    print(f"  Near miss (dist < fp_diag):    {cat_counts['near_miss']} ({cat_counts['near_miss']/len(all_fp_analysis)*100:.1f}%)")
    print(f"  Vicinity (dist < 2*fp_diag):   {cat_counts['vicinity']} ({cat_counts['vicinity']/len(all_fp_analysis)*100:.1f}%)")
    print(f"  Isolated (dist > 2*fp_diag):   {cat_counts['isolated']} ({cat_counts['isolated']/len(all_fp_analysis)*100:.1f}%)")
    print(f"\n  Missed annotation candidates: {total_candidates}")
    print(f"  Napari files: {napari_dir}")
    print(f"\n  Next steps:")
    print(f"    1. Open napari_missed/*.json in napari alongside original GT")
    print(f"    2. Visually confirm which are truly missed organoids")
    print(f"    3. Merge confirmed candidates into GT")
    print(f"    4. Re-evaluate detector with updated GT")


if __name__ == "__main__":
    from collections import Counter
    main()
