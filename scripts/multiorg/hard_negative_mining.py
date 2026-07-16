r"""
Hard Negative Mining for RF-DETR on MultiOrg

流程：
  1. 用当前 RF-DETR checkpoint 在训练集上推理
  2. 提取 FP patches（IoU < 0.3 的检测框）
  3. 把 FP patches 作为负样本（空标签 .txt）加入训练集
  4. 用原始 checkpoint 重训 N epochs

用法（Windows，单行命令）：
cd C:\Users\decha\organoid-fl
python scripts\multiorg\hard_negative_mining.py --checkpoint output\checkpoint_best_regular.pth --data-yaml D:\datasets\MultiOrg_v4_640\data.yaml --model-variant small --output-dir runs\rfdetr_hnm --hnm-epochs 50 --imgsz 512

依赖（和现有训练脚本相同）：
    pip install rfdetr supervision tifffile

输出：
    runs\rfdetr_hnm\
        checkpoint_best_regular.pth   - HNM 重训后的 checkpoint
        checkpoint_best_ema.pth
        training_log.json             - 训练日志
        hnm_negatives\                - 负样本 patch + 空标签
            train\images\*.png
            train\labels\*.txt (空文件)
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_data_yaml(yaml_path):
    """加载 YOLO data.yaml"""
    import yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def compute_iou(box1, box2):
    """IoU between two [x1,y1,x2,y2] boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-16)


def load_yolo_labels(label_path, img_w, img_h):
    """YOLO label → [[x1,y1,x2,y2], ...]"""
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            bboxes.append([x1, y1, x2, y2])
    return bboxes


def extract_fp_patches(model, data_yaml, output_dir, iou_threshold=0.3, conf=0.25, max_images=None):
    """在训练集上推理，提取 FP patches 作为负样本
    
    返回负样本数量
    """
    from PIL import Image
    import supervision as sv
    
    data = load_data_yaml(data_yaml)
    dataset_dir = os.path.dirname(os.path.abspath(data_yaml))
    
    # 训练集路径
    train_img_dir = os.path.join(dataset_dir, 'train', 'images')
    train_lbl_dir = os.path.join(dataset_dir, 'train', 'labels')
    
    if not os.path.isdir(train_img_dir):
        print(f"[ERROR] train/images not found: {train_img_dir}")
        return 0
    
    # 负样本输出目录
    neg_img_dir = os.path.join(output_dir, 'hnm_negatives', 'train', 'images')
    neg_lbl_dir = os.path.join(output_dir, 'hnm_negatives', 'train', 'labels')
    os.makedirs(neg_img_dir, exist_ok=True)
    os.makedirs(neg_lbl_dir, exist_ok=True)
    
    # 遍历训练集图片
    img_files = sorted([f for f in os.listdir(train_img_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    if max_images:
        img_files = img_files[:max_images]
    
    print(f"\n{'='*60}")
    print(f"=== Hard Negative Mining ===")
    print(f"{'='*60}")
    print(f"  Train images: {len(img_files)}")
    print(f"  IoU threshold: {iou_threshold}")
    print(f"  Conf threshold: {conf}")
    print(f"  Output: {neg_img_dir}")
    
    total_fps = 0
    total_dets = 0
    total_gts = 0
    n_images_with_fps = 0
    
    for i, img_name in enumerate(img_files):
        img_path = os.path.join(train_img_dir, img_name)
        lbl_path = os.path.join(train_lbl_dir, os.path.splitext(img_name)[0] + '.txt')
        
        # 加载图片
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
            img_np = np.array(img)
            if img_np.ndim == 2:
                img_np = np.stack([img_np]*3, axis=-1)
        except Exception as e:
            print(f"  [SKIP] {img_name}: {e}")
            continue
        
        # 加载 GT
        gt_bboxes = load_yolo_labels(lbl_path, img_w, img_h)
        total_gts += len(gt_bboxes)
        
        # 推理
        try:
            detections_raw = model.predict(img_path, threshold=conf)
        except Exception as e:
            # fallback: predict with PIL
            try:
                detections_raw = model.predict(img, threshold=conf)
            except Exception as e2:
                print(f"  [SKIP] {img_name} predict failed: {e2}")
                continue
        
        # 提取检测框
        det_boxes = []
        if hasattr(detections_raw, 'xyxy'):
            boxes = detections_raw.xyxy
            scores = detections_raw.confidence
            for j in range(len(scores)):
                det_boxes.append((boxes[j], float(scores[j])))
        elif isinstance(detections_raw, dict):
            boxes = detections_raw.get('boxes', [])
            scores = detections_raw.get('scores', [])
            for j in range(len(scores)):
                det_boxes.append((boxes[j], float(scores[j])))
        
        total_dets += len(det_boxes)
        
        # 匹配 FP
        fp_boxes = []
        gt_matched = [False] * len(gt_bboxes)
        for det_box, det_score in det_boxes:
            best_iou = 0
            best_gt = -1
            for j, gt_box in enumerate(gt_bboxes):
                if gt_matched[j]:
                    continue
                iou = compute_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = j
            
            if best_iou >= iou_threshold and best_gt >= 0:
                gt_matched[best_gt] = True
            else:
                fp_boxes.append(det_box)
        
        if fp_boxes:
            n_images_with_fps += 1
            total_fps += len(fp_boxes)
            
            # 把 FP 区域裁剪出来作为负样本 patch
            for j, fp_box in enumerate(fp_boxes):
                x1, y1, x2, y2 = [int(v) for v in fp_box]
                # 加 padding（50% of bbox size）
                w = x2 - x1
                h = y2 - y1
                pad_x = int(w * 0.5)
                pad_y = int(h * 0.5)
                x1_pad = max(0, x1 - pad_x)
                y1_pad = max(0, y1 - pad_y)
                x2_pad = min(img_w, x2 + pad_x)
                y2_pad = min(img_h, y2 + pad_y)
                
                # 裁剪
                patch = img.crop((x1_pad, y1_pad, x2_pad, y2_pad))
                if patch.size[0] < 10 or patch.size[1] < 10:
                    continue
                
                # resize 到 imgsz
                patch_name = f"hnm_neg_{i:05d}_{j:03d}.png"
                patch.save(os.path.join(neg_img_dir, patch_name))
                
                # 空标签文件（负样本 = 无目标）
                lbl_name = os.path.splitext(patch_name)[0] + '.txt'
                open(os.path.join(neg_lbl_dir, lbl_name), 'w', encoding='utf-8').close()
        
        if (i+1) % 100 == 0:
            print(f"  [{i+1}/{len(img_files)}] dets={total_dets} fps={total_fps} "
                  f"({total_fps/max(total_dets,1)*100:.1f}%) imgs_with_fps={n_images_with_fps}")
    
    print(f"\n  === HNM Extraction Summary ===")
    print(f"  Total images: {len(img_files)}")
    print(f"  Images with FP: {n_images_with_fps} ({n_images_with_fps/len(img_files)*100:.1f}%)")
    print(f"  Total detections: {total_dets}")
    print(f"  Total GT: {total_gts}")
    print(f"  Total FP extracted: {total_fps} ({total_fps/max(total_dets,1)*100:.1f}%)")
    print(f"  Negative samples saved: {neg_img_dir}")
    
    return total_fps


def prepare_hnm_dataset(data_yaml, hnm_dir, output_dataset_dir):
    """合并原始训练数据 + HNM 负样本到新数据集目录
    
    用 symlink（Windows 需要管理员权限）或 copy2
    """
    data = load_data_yaml(data_yaml)
    dataset_dir = os.path.dirname(os.path.abspath(data_yaml))
    
    # 输出目录
    out_train_img = os.path.join(output_dataset_dir, 'train', 'images')
    out_train_lbl = os.path.join(output_dataset_dir, 'train', 'labels')
    out_val_img = os.path.join(output_dataset_dir, 'valid', 'images')
    out_val_lbl = os.path.join(output_dataset_dir, 'valid', 'labels')
    
    for d in [out_train_img, out_train_lbl, out_val_img, out_val_lbl]:
        os.makedirs(d, exist_ok=True)
    
    # 1. 复制原始训练数据
    src_train_img = os.path.join(dataset_dir, 'train', 'images')
    src_train_lbl = os.path.join(dataset_dir, 'train', 'labels')
    
    print(f"\n  Copying original train data...")
    n_orig = 0
    for f in os.listdir(src_train_img):
        src = os.path.join(src_train_img, f)
        dst = os.path.join(out_train_img, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            n_orig += 1
    for f in os.listdir(src_train_lbl):
        src = os.path.join(src_train_lbl, f)
        dst = os.path.join(out_train_lbl, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f"  Copied {n_orig} original images")
    
    # 2. 复制 HNM 负样本
    hnm_img_dir = os.path.join(hnm_dir, 'hnm_negatives', 'train', 'images')
    hnm_lbl_dir = os.path.join(hnm_dir, 'hnm_negatives', 'train', 'labels')
    
    print(f"  Copying HNM negatives...")
    n_neg = 0
    for f in os.listdir(hnm_img_dir):
        src = os.path.join(hnm_img_dir, f)
        dst = os.path.join(out_train_img, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            n_neg += 1
    for f in os.listdir(hnm_lbl_dir):
        src = os.path.join(hnm_lbl_dir, f)
        dst = os.path.join(out_train_lbl, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f"  Copied {n_neg} negative samples")
    
    # 3. 复制 val 数据（不加工）
    src_val_img = os.path.join(dataset_dir, 'valid', 'images')
    src_val_lbl = os.path.join(dataset_dir, 'valid', 'labels')
    # fallback: test/ (RF-DETR 要 valid/)
    if not os.path.isdir(src_val_img):
        src_val_img = os.path.join(dataset_dir, 'test', 'images')
        src_val_lbl = os.path.join(dataset_dir, 'test', 'labels')
    
    print(f"  Copying val data...")
    n_val = 0
    for f in os.listdir(src_val_img):
        src = os.path.join(src_val_img, f)
        dst = os.path.join(out_val_img, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            n_val += 1
    for f in os.listdir(src_val_lbl):
        src = os.path.join(src_val_lbl, f)
        dst = os.path.join(out_val_lbl, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f"  Copied {n_val} val images")
    
    # 4. 写 data.yaml
    yaml_path = os.path.join(output_dataset_dir, 'data.yaml')
    yaml_content = f"""path: {output_dataset_dir}
train: train/images
valid: valid/images
test: valid/images
nc: 1
names: ['organoid']
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"  data.yaml: {yaml_path}")
    
    total_train = n_orig + n_neg
    print(f"\n  === HNM Dataset Summary ===")
    print(f"  Original train: {n_orig}")
    print(f"  HNM negatives:  {n_neg}")
    print(f"  Total train:    {total_train}")
    print(f"  Val:            {n_val}")
    
    return yaml_path


def train_with_hnm(data_yaml, checkpoint, model_variant, epochs, imgsz, output_dir):
    """用 HNM 数据集重训 RF-DETR"""
    from rfdetr import RFDETRBase, RFDETRSmall, RFDETRNano, RFDETRMedium, RFDETRLarge
    import time
    
    model_map = {
        'nano': RFDETRNano, 'small': RFDETRSmall, 'base': RFDETRBase,
        'medium': RFDETRMedium, 'large': RFDETRLarge,
    }
    ModelClass = model_map.get(model_variant, RFDETRSmall)
    
    print(f"\n{'='*60}")
    print(f"=== HNM Training ===")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Model: {model_variant}")
    print(f"  Epochs: {epochs}")
    print(f"  ImgSz: {imgsz}")
    print(f"  Output: {output_dir}")
    
    # 用 HNM checkpoint 初始化
    model = ModelClass(pretrain_weights=checkpoint)
    
    dataset_dir = os.path.dirname(os.path.abspath(data_yaml))
    train_kwargs = {
        'dataset_dir': dataset_dir,
        'epochs': epochs,
        'grad_accum_steps': 4,
        'resolution': imgsz,
        'output_dir': output_dir,
    }
    
    print(f"\n  Starting training...")
    start_time = time.time()
    model.train(**train_kwargs)
    elapsed = time.time() - start_time
    
    hours = elapsed / 3600
    print(f"\n  Training completed in {hours:.2f} hours")
    
    # 保存日志
    log = {
        'checkpoint': checkpoint,
        'model_variant': model_variant,
        'epochs': epochs,
        'imgsz': imgsz,
        'training_time_hours': hours,
    }
    log_path = os.path.join(output_dir, 'training_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)
    print(f"  Log: {log_path}")
    
    # 最终 checkpoint 路径
    final_ckpt = os.path.join(output_dir, 'checkpoint_best_regular.pth')
    if os.path.exists(final_ckpt):
        print(f"\n  Final checkpoint: {final_ckpt}")
    else:
        print(f"\n  [WARN] checkpoint not found at {final_ckpt}")


def main():
    parser = argparse.ArgumentParser(description='Hard Negative Mining for RF-DETR')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to current RF-DETR checkpoint')
    parser.add_argument('--data-yaml', required=True,
                        help='Path to original data.yaml')
    parser.add_argument('--model-variant', default='small',
                        choices=['nano', 'small', 'base', 'medium', 'large'])
    parser.add_argument('--output-dir', default='./runs/rfdetr_hnm',
                        help='Output directory for HNM training')
    parser.add_argument('--hnm-epochs', type=int, default=50,
                        help='Epochs for HNM retraining')
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                        help='IoU threshold for FP detection (lower = more FP captured)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detection')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Max train images to process (for debugging)')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip FP extraction (use existing hnm_negatives)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training (only extract FP patches)')
    args = parser.parse_args()
    
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: 提取 FP
    if not args.skip_extraction:
        from rfdetr import RFDETRBase, RFDETRSmall, RFDETRNano, RFDETRMedium, RFDETRLarge
        model_map = {
            'nano': RFDETRNano, 'small': RFDETRSmall, 'base': RFDETRBase,
            'medium': RFDETRMedium, 'large': RFDETRLarge,
        }
        ModelClass = model_map.get(args.model_variant, RFDETRSmall)
        
        print(f"\n  Loading model: {args.checkpoint}")
        model = ModelClass(pretrain_weights=args.checkpoint)
        
        n_fps = extract_fp_patches(
            model, args.data_yaml, output_dir,
            iou_threshold=args.iou_threshold,
            conf=args.conf,
            max_images=args.max_images,
        )
        
        if n_fps == 0:
            print("\n  [WARN] No FP extracted. Check model/data paths.")
            return
    else:
        print("\n  [SKIP] FP extraction")
    
    if args.skip_training:
        print("\n  [SKIP] Training (--skip-training)")
        return
    
    # Step 2: 准备 HNM 数据集
    hnm_dataset_dir = os.path.join(output_dir, 'hnm_dataset')
    hnm_yaml = prepare_hnm_dataset(args.data_yaml, output_dir, hnm_dataset_dir)
    
    # Step 3: 重训
    train_with_hnm(
        hnm_yaml, args.checkpoint, args.model_variant,
        args.hnm_epochs, args.imgsz, output_dir,
    )
    
    print(f"\n{'='*60}")
    print(f"=== HNM Complete ===")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Checkpoint: {output_dir}\\checkpoint_best_regular.pth")
    print(f"\n  Next steps:")
    print(f"    1. Evaluate HNM checkpoint on test set:")
    print(f"       python scripts\\multiorg\\multiorg_sam2.py --weights {output_dir}\\checkpoint_best_regular.pth ...")
    print(f"    2. Compare mAP50 with baseline (77.15%)")


if __name__ == '__main__':
    main()
