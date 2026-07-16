"""
云 VM 跨域评估一站式脚本

流程：
1. 用 YOLOv12 鼠肝 checkpoint 生成 crops（替代 RF-DETR，云 VM 无 RF-DETR checkpoint）
2. 用 MultiOrg crops 训练 supcon slot model
3. 跨域评估（supcon → 鼠肝 crops）

Usage:
    cd /home/z/my-project/organoid-fl
    python scripts/mouse_liver/run_cross_domain_cloud.py
"""
import json
import os
import sys
import time
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image

# Paths
PROJECT_ROOT = Path('/home/z/my-project/organoid-fl')
DATA_ROOT = Path('/home/z/my-project/mouse_liver_data_correct')
MULTIORG_CROPS = PROJECT_ROOT / 'results' / 'phase2_vlm_100'
MULTIORG_VLM_RESULTS = MULTIORG_CROPS / 'vlm_results.json'
CROPS_DIR = PROJECT_ROOT / 'data' / 'mouse_crops'
RESULTS_DIR = PROJECT_ROOT / 'results'

# YOLOv12 鼠肝 checkpoints (cloud VM 上有)
YOLO_CKPTS = {
    'b1': '/home/z/my-project/mouse_liver_organoid/runs/detect/runs/mouse_liver/exp1_b1_train/weights/best.pt',
    'b2': '/home/z/my-project/mouse_liver_organoid/runs/detect/runs/mouse_liver/exp3_unified/weights/best.pt',  # all 40 images
    'b3': '/home/z/my-project/mouse_liver_organoid/runs/detect/runs/mouse_liver/batch3_only/weights/best.pt',
}

BATCH_IMAGES = {
    'b1': DATA_ROOT / 'batch1' / 'images',
    'b2': DATA_ROOT / 'batch2' / 'images',
    'b3': DATA_ROOT / 'batch3' / 'images',
}
BATCH_ANNOT = {
    'b1': DATA_ROOT / 'batch1' / 'annotations.json',
    'b2': DATA_ROOT / 'batch2' / 'annotations.json',
    'b3': DATA_ROOT / 'batch3' / 'annotations.json',
}


def bbox_iou(box1, box2):
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)


def load_gt_bboxes(annotations_path, image_name):
    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for entry in data:
        if entry['image'] == image_name:
            bboxes = []
            for bb in entry['bboxes']:
                x, y, w, h = bb['x'], bb['y'], bb['w'], bb['h']
                bboxes.append([x, y, x + w, y + h])
            return bboxes
    return []


def match_detections_to_gt(dets, gt_bboxes, iou_threshold=0.5):
    matched_flags = [False] * len(dets)
    matched_ious = [0.0] * len(dets)
    gt_used = [False] * len(gt_bboxes)
    det_order = sorted(range(len(dets)), key=lambda i: -dets[i]['confidence'])
    for di in det_order:
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_bboxes):
            if gt_used[gi]:
                continue
            iou = bbox_iou(dets[di]['bbox'], gt)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_threshold and best_gi >= 0:
            matched_flags[di] = True
            matched_ious[di] = best_iou
            gt_used[best_gi] = True
    return matched_flags, matched_ious


def crop_and_resize(img_arr, bbox, target_size=224):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_arr.shape[1], x2)
    y2 = min(img_arr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    crop = img_arr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return crop


def generate_crops_yolo(batch_name, yolo_ckpt, img_dir, annot_path, dst_dir, conf_thr=0.25, crop_size=224):
    """用 YOLOv12 检测生成 crops。"""
    from ultralytics import YOLO
    
    os.makedirs(dst_dir, exist_ok=True)
    crops_dir = Path(dst_dir) / 'crops'
    crops_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating crops: {batch_name}")
    print(f"  YOLO checkpoint: {yolo_ckpt}")
    print(f"  Images: {img_dir}")
    print(f"  Annotations: {annot_path}")
    print(f"{'='*60}")
    
    model = YOLO(yolo_ckpt)
    
    images = sorted(list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png')))
    print(f"Found {len(images)} images")
    
    all_dets = []
    total_tp, total_fp = 0, 0
    
    for img_idx, img_path in enumerate(images):
        # YOLOv12 inference (let YOLO handle image loading to avoid duplicate memory)
        import gc
        results = model(str(img_path), conf=conf_thr, verbose=False, imgsz=640)
        result = results[0]
        
        det_bboxes = result.boxes.xyxy.cpu().numpy().tolist() if len(result.boxes) > 0 else []
        det_confs = result.boxes.conf.cpu().numpy().tolist() if len(result.boxes) > 0 else []
        
        # Load GT
        gt_bboxes = load_gt_bboxes(annot_path, img_path.name)
        
        # Match
        det_list = [{'bbox': bb, 'confidence': cf} for bb, cf in zip(det_bboxes, det_confs)]
        matched_flags, matched_ious = match_detections_to_gt(det_list, gt_bboxes)
        
        n_tp = sum(matched_flags)
        n_fp = len(matched_flags) - n_tp
        total_tp += n_tp
        total_fp += n_fp
        
        print(f"  [{img_idx+1}/{len(images)}] {img_path.name}: "
              f"det={len(det_list)} gt={len(gt_bboxes)} TP={n_tp} FP={n_fp}")
        
        # Load image for cropping (only if there are detections)
        if len(det_list) > 0:
            img_pil = Image.open(img_path)
            img_np = np.array(img_pil.convert('RGB'))
            h, w = img_np.shape[:2]
        else:
            img_np = None
            h, w = 0, 0
        
        # Save crops
        for di, det in enumerate(det_list):
            cache_key = f"{batch_name}_{img_path.stem}_{di}"
            crop = crop_and_resize(img_np, det['bbox'], crop_size)
            crop_path = crops_dir / f"{cache_key}.png"
            success, buf = cv2.imencode('.png', cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            if success:
                buf.tofile(str(crop_path))
            
            entry = {
                'cache_key': cache_key,
                'image': f"{batch_name}/{img_path.stem}",
                'det_idx': di,
                'bbox': det['bbox'],
                'rfdetr_conf': det['confidence'],  # keep field name for compatibility
                'matched': bool(matched_flags[di]),
                'match_iou': float(matched_ious[di]),
                'area': 0, 'circularity': 0, 'solidity': 0, 'aspect_ratio': 0,
                'crop_path': str(crop_path),
                'image_size': [w, h],
                'n_gt': len(gt_bboxes),
            }
            all_dets.append(entry)
        
        # Free memory (large images on 4GB cloud VM)
        del img_np, results, result
        gc.collect()
    
    # Free model memory before supcon training
    del model
    gc.collect()
    
    meta_path = Path(dst_dir) / 'crop_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_dets, f, indent=2, ensure_ascii=False)
    
    print(f"\nSUMMARY {batch_name}: {len(all_dets)} dets, TP={total_tp}, FP={total_fp}")
    print(f"  Metadata: {meta_path}")
    return dst_dir


def train_supcon():
    """用 MultiOrg crops 训练 supcon slot model。"""
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'multiorg'))
    import slot_supcon
    
    # Prepare metadata for slot_supcon (it expects ctm_metadata.json format)
    # vlm_results.json has: image, det_idx, bbox, confidence, matched, cache_key, crop_path, crop_exists, vlm, vlm_raw
    with open(MULTIORG_VLM_RESULTS, 'r', encoding='utf-8') as f:
        vlm_data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Training SupCon on MultiOrg crops")
    print(f"  Crops: {len(vlm_data)} (TP={sum(1 for x in vlm_data if x.get('matched'))}, "
          f"FP={sum(1 for x in vlm_data if not x.get('matched'))})")
    print(f"{'='*60}")
    
    # Create metadata in the format slot_supcon expects
    metadata = []
    for item in vlm_data:
        if not item.get('crop_exists', True):
            continue
        metadata.append({
            'cache_key': item['cache_key'],
            'image': item['image'],
            'det_idx': item.get('det_idx', 0),
            'bbox': item.get('bbox', [0, 0, 0, 0]),
            'rfdetr_conf': item.get('confidence', 0.5),
            'matched': item.get('matched', False),
            'match_iou': item.get('match_iou', 0.0),
            'area': 0, 'circularity': 0, 'solidity': 0, 'aspect_ratio': 0,
            'crop_path': item.get('crop_path', ''),
            'image_size': [0, 0],
            'n_gt': 0,
        })
    
    # Write temporary metadata
    meta_path = MULTIORG_CROPS / 'ctm_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Metadata: {meta_path} ({len(metadata)} entries)")
    
    # Training config (same as冬生本地: 8s_d128_p256_t0.07_b0.1)
    output_dir = str(RESULTS_DIR / 'supcon_8s_d128_p256_t0.07_b0.1_cloud')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run slot_supcon train
    sys.argv = [
        'slot_supcon.py',
        '--metadata', str(meta_path),
        '--crops-dir', str(MULTIORG_CROPS / 'crops'),
        '--output-dir', output_dir,
        '--num-slots', '8',
        '--dim-slots', '128',
        '--proj-dim', '256',
        '--temperature', '0.07',
        '--supcon-weight', '0.1',
        '--epochs', '50',
        '--batch-size', '32',
        '--device', 'cpu',
        '--num-workers', '0',
    ]
    
    print(f"  Output: {output_dir}")
    print(f"  Running slot_supcon.main()...")
    slot_supcon.main()
    
    ckpt_path = os.path.join(output_dir, 'best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SupCon checkpoint not found: {ckpt_path}")
    print(f"  Checkpoint: {ckpt_path}")
    return ckpt_path


def cross_domain_eval(supcon_ckpt, crop_dirs):
    """跨域评估。"""
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'mouse_liver'))
    
    from cross_domain_eval import load_model, SimpleCropDataset
    from sklearn.metrics import roc_auc_score, average_precision_score
    from torch.utils.data import DataLoader
    
    device = torch.device('cpu')
    model = load_model(supcon_ckpt, device)
    
    all_results = {}
    for batch_name, crop_dir in crop_dirs.items():
        metadata_path = Path(crop_dir) / 'crop_metadata.json'
        crops_dir = str(Path(crop_dir) / 'crops')
        
        if not metadata_path.exists():
            print(f"[WARN] Metadata not found: {metadata_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Cross-domain eval: {batch_name}")
        print(f"{'='*60}")
        
        dataset = SimpleCropDataset(str(metadata_path), crops_dir)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        all_scores, all_confs, all_labels = [], [], []
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(device)
                logits, _, _ = model(images, return_embeddings=True)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                all_scores.extend(probs.cpu().numpy())
                all_confs.extend(batch['confidence'].numpy())
                all_labels.extend(batch['label'].numpy())
        
        scores = np.array(all_scores)
        confs = np.array(all_confs)
        labels = np.array(all_labels)
        
        n_tp = int(labels.sum())
        n_fp = len(labels) - n_tp
        
        slot_auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5
        conf_auc = roc_auc_score(labels, confs) if len(set(labels)) > 1 else 0.5
        slot_ap = average_precision_score(labels, scores)
        conf_ap = average_precision_score(labels, confs)
        
        print(f"  Crops: {len(labels)}, TP={n_tp}, FP={n_fp}")
        print(f"  Slot AUC: {slot_auc:.4f}  AP: {slot_ap:.4f}")
        print(f"  Conf AUC: {conf_auc:.4f}  AP: {conf_ap:.4f}")
        print(f"  Δ AUC:    {slot_auc - conf_auc:+.4f}")
        
        output_dir = RESULTS_DIR / f'cross_domain_{batch_name}'
        output_dir.mkdir(exist_ok=True)
        np.save(output_dir / 'embeddings.npy', np.array([]))  # placeholder
        np.save(output_dir / 'labels.npy', labels)
        np.save(output_dir / 'confs.npy', confs)
        
        all_results[batch_name] = {
            'n_crops': len(labels), 'n_tp': n_tp, 'n_fp': n_fp,
            'slot_auc': float(slot_auc), 'conf_auc': float(conf_auc),
            'slot_ap': float(slot_ap), 'conf_ap': float(conf_ap),
            'delta_auc': float(slot_auc - conf_auc),
        }
        
        with open(output_dir / 'cross_domain_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results[batch_name], f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN SUMMARY")
    print(f"{'='*60}")
    print(f"{'Batch':<8} {'Crops':>6} {'TP':>5} {'FP':>5} {'Slot AUC':>10} {'Conf AUC':>10} {'Δ':>8}")
    print("-" * 55)
    for batch_name, r in all_results.items():
        print(f"{batch_name:<8} {r['n_crops']:6d} {r['n_tp']:5d} {r['n_fp']:5d} "
              f"{r['slot_auc']:10.4f} {r['conf_auc']:10.4f} {r['delta_auc']:+8.4f}")
    
    summary_path = RESULTS_DIR / 'cross_domain_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


def main():
    os.chdir(str(PROJECT_ROOT))
    t0 = time.time()
    
    print("\n" + "=" * 60)
    print("  Cross-Domain Evaluation Pipeline (Cloud VM)")
    print("  YOLOv12 crops → SupCon training → Cross-domain eval")
    print("=" * 60)
    
    # Step 1: Generate crops for each batch
    crop_dirs = {}
    for batch_name in ['b1', 'b2', 'b3']:
        dst = str(CROPS_DIR / batch_name)
        yolo_ckpt = YOLO_CKPTS[batch_name]
        img_dir = str(BATCH_IMAGES[batch_name])
        annot_path = str(BATCH_ANNOT[batch_name])
        
        if not os.path.exists(yolo_ckpt):
            print(f"[ERROR] YOLO checkpoint not found: {yolo_ckpt}")
            continue
        
        crop_dirs[batch_name] = generate_crops_yolo(
            batch_name, yolo_ckpt, img_dir, annot_path, dst
        )
    
    # Step 2: Train supcon on MultiOrg crops
    supcon_ckpt = train_supcon()
    
    # Step 3: Cross-domain eval
    cross_domain_eval(supcon_ckpt, crop_dirs)
    
    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
