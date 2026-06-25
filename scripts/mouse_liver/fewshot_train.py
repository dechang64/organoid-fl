r"""
Few-shot 微调：用 10 张鼠肝图片 8+2 分，RF-DETR small 预训练权重微调

Usage (冬生本地):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate
    python scripts\mouse_liver\fewshot_train.py --pretrained output\checkpoint_best_regular.pth --data scripts\mouse_liver\yolo_format --output runs\mouse_liver_fewshot --epochs 20
"""
import argparse
import os
import sys
import json
import shutil
import random
from pathlib import Path

def split_dataset(data_dir, output_dir, train_ratio=0.8, seed=42):
    """8+2 分：8张训练，2张测试"""
    random.seed(seed)
    img_dir = Path(data_dir) / 'images'
    lbl_dir = Path(data_dir) / 'labels'
    
    images = sorted(img_dir.glob('*.jpg'))
    random.shuffle(images)
    
    n_train = int(len(images) * train_ratio)
    train_imgs = images[:n_train]
    val_imgs = images[n_train:]
    
    # Create split directories
    split_dir = Path(output_dir) / 'dataset'
    for split in ['train', 'valid']:
        (split_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    for img in train_imgs:
        shutil.copy2(img, split_dir / 'train' / 'images' / img.name)
        lbl = lbl_dir / (img.stem + '.txt')
        if lbl.exists():
            shutil.copy2(lbl, split_dir / 'train' / 'labels' / lbl.name)
    
    for img in val_imgs:
        shutil.copy2(img, split_dir / 'valid' / 'images' / img.name)
        lbl = lbl_dir / (img.stem + '.txt')
        if lbl.exists():
            shutil.copy2(lbl, split_dir / 'valid' / 'labels' / lbl.name)
    
    # Create data.yaml
    yaml_content = f"""path: {split_dir.resolve()}
train: train/images
val: valid/images
test: valid/images
nc: 1
names: ['organoid']
"""
    yaml_path = split_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Split: {len(train_imgs)} train + {len(val_imgs)} val")
    print(f"Train: {[f.name for f in train_imgs]}")
    print(f"Val: {[f.name for f in val_imgs]}")
    print(f"Dataset: {split_dir}")
    
    return str(yaml_path.resolve()), [f.name for f in train_imgs], [f.name for f in val_imgs]

def train(pretrained_path, data_yaml, output_dir, epochs=20, resolution=544):
    """RF-DETR few-shot 微调"""
    from rfdetr import RFDETRSmall
    
    # RF-DETR resolution 必须能被 32 整除
    assert resolution % 32 == 0, f"resolution must be divisible by 32, got {resolution}"
    
    model = RFDETRSmall(pretrain_weights=pretrained_path, num_classes=1)
    
    train_kwargs = {
        'dataset_dir': str(Path(data_yaml).parent.resolve()),
        'epochs': epochs,
        'grad_accum_steps': 4,
        'resolution': resolution,
        'output_dir': str(Path(output_dir).resolve()),
    }
    
    print(f"\nStarting few-shot training...")
    print(f"  Pretrained: {pretrained_path}")
    print(f"  Resolution: {resolution}")
    print(f"  Epochs: {epochs}")
    print(f"  Output: {output_dir}")
    print(f"  kwargs: {train_kwargs}")
    
    model.train(**train_kwargs)
    print(f"\nTraining complete! Checkpoint: {Path(output_dir) / 'checkpoint_best_regular.pth'}")

def evaluate(model_path, data_yaml, output_dir, model_variant='small'):
    """评估微调后的模型（手动计算 TP/FP/FN，不依赖 sv.MeanAveragePrecision）"""
    from rfdetr import RFDETRSmall
    from PIL import Image
    import numpy as np
    
    model = RFDETRSmall(pretrain_weights=model_path, num_classes=1)
    
    # Load val images
    data_dir = Path(data_yaml).parent
    val_img_dir = data_dir / 'valid' / 'images'
    val_lbl_dir = data_dir / 'valid' / 'labels'
    
    images = sorted(val_img_dir.glob('*.jpg'))
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for img_path in images:
        img = Image.open(img_path)
        w, h = img.size
        dets = model.predict(img, threshold=0.25)
        
        # Load GT
        lbl_path = val_lbl_dir / (img_path.stem + '.txt')
        gt_boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, bw, bh = parts
                        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
                        x1 = (xc - bw/2) * w
                        y1 = (yc - bh/2) * h
                        x2 = (xc + bw/2) * w
                        y2 = (yc + bh/2) * h
                        gt_boxes.append([x1, y1, x2, y2])
        
        # Match TP/FP/FN
        if len(dets.xyxy) > 0 and gt_boxes:
            matched = set()
            tp = 0
            for di in range(len(dets.xyxy)):
                dx1, dy1, dx2, dy2 = dets.xyxy[di]
                best_iou, best_gi = 0, -1
                for gi, (gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
                    if gi in matched:
                        continue
                    ix1, iy1 = max(dx1, gx1), max(dy1, gy1)
                    ix2, iy2 = min(dx2, gx2), min(dy2, gy2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        da = (dx2 - dx1) * (dy2 - dy1)
                        ga = (gx2 - gx1) * (gy2 - gy1)
                        iou = inter / (da + ga - inter)
                        if iou > best_iou:
                            best_iou, best_gi = iou, gi
                if best_iou > 0.5 and best_gi >= 0:
                    tp += 1
                    matched.add(best_gi)
            total_tp += tp
            total_fp += len(dets.xyxy) - tp
            total_fn += len(gt_boxes) - len(matched)
        else:
            total_fp += len(dets.xyxy)
            total_fn += len(gt_boxes)
    
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"FEW-SHOT EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Val images: {len(images)}")
    print(f"  Precision: {prec:.4f} ({prec*100:.1f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.1f}%)")
    print(f"  F1:        {f1:.4f} ({f1*100:.1f}%)")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")

def main():
    parser = argparse.ArgumentParser(description='Few-shot RF-DETR training on mouse liver organoids')
    parser.add_argument('--pretrained', required=True, help='Pretrained RF-DETR checkpoint (MultiOrg)')
    parser.add_argument('--data', default='scripts/mouse_liver/yolo_format', help='YOLO format data dir')
    parser.add_argument('--output', default='runs/mouse_liver_fewshot', help='Output directory')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--resolution', type=int, default=544, help='Must be divisible by 32')
    parser.add_argument('--eval-only', action='store_true', help='Skip training, eval only')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint for eval-only mode')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Split dataset
    data_yaml, train_files, val_files = split_dataset(args.data, args.output)
    
    if not args.eval_only:
        # Train
        train(args.pretrained, data_yaml, args.output, args.epochs, args.resolution)
    
    # Evaluate
    ckpt = args.checkpoint or str(Path(args.output) / 'checkpoint_best_regular.pth')
    if not Path(ckpt).exists():
        # Try output dir directly
        ckpt = str(Path(args.output) / 'checkpoint_best_regular.pth')
    if Path(ckpt).exists():
        evaluate(ckpt, data_yaml, args.output)
    else:
        print(f"\n[WARN] No checkpoint found at {ckpt}, skipping evaluation")

if __name__ == '__main__':
    main()
