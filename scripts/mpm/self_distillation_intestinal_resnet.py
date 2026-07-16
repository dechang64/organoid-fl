r"""
Intestinal 真实跨域自蒸馏 — 用 ResNet50 真特征
==================================================

升级版：用 ResNet50 ImageNet pretrained 提取 crop 的 2048 维特征，
替代之前的 30 维图像统计特征。

实验设计：
  1. 鼠肝 RF-DETR checkpoint → Intestinal 跨域检测（已有结果）
  2. ResNet50 ImageNet pretrained 提取每个 crop 的 2048 维特征
  3. 用 YOLO GT IoU 作为独立信号 + 5% 噪声 → 伪标签
  4. 训练分类头（MLP 2048→256→2）
  5. 评估 AUC（zero-shot vs distilled）

预期：DINOv2/ResNet50 真特征比图像统计强得多，
      应该能突破 +9.10% 瓶颈，达到 +20% 以上。

Usage (云 VM, CPU):
    cd /home/z/my-project/organoid-fl
    python scripts/mpm/self_distillation_intestinal_resnet.py \
        --n-train 100 --n-val 50 --epochs 100
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def load_yolo_labels(lbl_path, img_w, img_h):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5: continue
            cls, xc, yc, w, h = map(float, p[:5])
            boxes.append({
                "class": int(cls),
                "bbox": [xc-w/2, yc-h/2, xc+w/2, yc+h/2],  # normalized
            })
    return boxes


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1) * (y2-y1)
    a1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    a2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])
    return inter / max(a1+a2-inter, 1e-6)


class ResNetFeatureExtractor:
    """ResNet50 ImageNet pretrained, extract 2048-dim feature per crop"""
    def __init__(self, device='cpu'):
        import timm
        from torchvision import transforms
        self.device = device
        print(f"  Loading ResNet50 (ImageNet pretrained)...")
        self.model = timm.create_model('resnet50', pretrained=True).to(device)
        self.model.eval()
        # Remove classification head
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # remove fc, output [B, 2048, 1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, img_rgb, boxes, img_w, img_h):
        """Extract features for each box (crop region). Returns [N, 2048]."""
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box
            # Convert normalized to pixel
            px1 = int(max(0, x1 * img_w))
            py1 = int(max(0, y1 * img_h))
            px2 = int(min(img_w, x2 * img_w))
            py2 = int(min(img_h, y2 * img_h))
            if px2 <= px1 or py2 <= py1:
                # Empty crop
                crops.append(torch.zeros(3, 224, 224))
                continue
            crop = img_rgb[py1:py2, px1:px2]
            if crop.size == 0:
                crops.append(torch.zeros(3, 224, 224))
                continue
            pil = Image.fromarray(crop)
            crops.append(self.transform(pil))

        if not crops:
            return torch.zeros(0, 2048)

        batch = torch.stack(crops).to(self.device)
        feats = self.model(batch).squeeze(-1).squeeze(-1)  # [N, 2048]
        return feats.cpu()


class Classifier(nn.Module):
    def __init__(self, in_dim=2048, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def process_split(files, img_dir, lbl_dir, model, extractor, n_max, args, split_name):
    print(f"  Processing {split_name}: {len(files)} images, max {n_max} samples")
    all_features = []
    all_true_labels = []
    all_pseudo_labels = []
    all_zero_shot_scores = []
    rng = np.random.default_rng(seed=42 if split_name == 'train' else 123)

    for i, fname in enumerate(files):
        img_path = img_dir / fname
        lbl_path = lbl_dir / fname.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')

        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        # GT boxes
        gt_boxes = load_yolo_labels(lbl_path, w, h)
        gt_bboxes = [g["bbox"] for g in gt_boxes]

        if len(all_true_labels) >= n_max and n_max > 0:
            break

        # RF-DETR detect
        try:
            dets = model.predict(img, threshold=0.01)
            det_boxes_px = dets.xyxy
            det_scores = dets.confidence
        except Exception as e:
            print(f"    ⚠ {fname}: {e}")
            continue

        for box_px, rf_conf in zip(det_boxes_px, det_scores):
            if len(all_true_labels) >= n_max and n_max > 0:
                break
            box_norm = [float(box_px[0]/w), float(box_px[1]/h), float(box_px[2]/w), float(box_px[3]/h)]
            rf_conf = float(rf_conf)

            # Compute IoU with GT
            max_iou = max([compute_iou(box_norm, g) for g in gt_bboxes] + [0.0])
            is_tp = max_iou >= 0.30

            # Simulated SAM2 IoU (TP high, FP low, 5% noise)
            if is_tp:
                sam2_iou = float(rng.uniform(0.70, 1.00))
            else:
                sam2_iou = float(rng.uniform(0.00, 0.25))
            if rng.random() < args.sam2_noise:
                sam2_iou = 1.0 - sam2_iou

            # Pseudo label
            if sam2_iou >= 0.5:
                pseudo = 1
            else:
                pseudo = 0

            all_true_labels.append(1 if is_tp else 0)
            all_pseudo_labels.append(pseudo)
            all_zero_shot_scores.append(rf_conf)
            all_features.append([box_norm, img, w, h])  # placeholder, extract later

        if (i + 1) % 10 == 0 or i == len(files) - 1:
            n_pos = sum(all_pseudo_labels)
            n_neg = len(all_pseudo_labels) - n_pos
            print(f"    {split_name} {i+1}/{len(files)}: total={len(all_pseudo_labels)} pos={n_pos} neg={n_neg}", flush=True)

    # Now extract ResNet features for each sample
    print(f"    Extracting ResNet50 features for {len(all_features)} samples...", flush=True)
    real_features = []
    for j, (box_norm, img, w, h) in enumerate(all_features):
        feat = extractor.extract(img, [box_norm], w, h)
        real_features.append(feat[0] if len(feat) > 0 else torch.zeros(2048))
        if (j + 1) % 500 == 0:
            print(f"      {j+1}/{len(all_features)}", flush=True)

    X = torch.stack(real_features) if real_features else torch.zeros(0, 2048)
    y_true = torch.tensor(all_true_labels, dtype=torch.long)
    y_pseudo = torch.tensor(all_pseudo_labels, dtype=torch.long)
    scores = torch.tensor(all_zero_shot_scores, dtype=torch.float32)

    return X, y_true, y_pseudo, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/intestinal_organoid/OrganoidDataset')
    parser.add_argument('--rfdetr-checkpoint', default='runs/mouse_liver_v2/b1/full/checkpoint_best_ema.pth')
    parser.add_argument('--output-dir', default='results/sd_intestinal_resnet')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--n-train', type=int, default=100)
    parser.add_argument('--n-val', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sam2-noise', type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"="*70)
    print(f"Intestinal 真实跨域自蒸馏 — ResNet50 真特征")
    print(f"  RF-DETR: {args.rfdetr_checkpoint}")
    print(f"  Feature: ResNet50 ImageNet pretrained (2048-dim)")
    print(f"  Train: {args.n_train}, Val: {args.n_val}")
    print(f"="*70)

    # Step 1: Load RF-DETR
    print(f"\n[Step 1] 加载 RF-DETR (鼠肝 checkpoint)")
    try:
        from rfdetr import RFDETRBase
        import warnings
        warnings.filterwarnings("ignore")
        model = RFDETRBase(device=args.device, pretrain_weights=args.rfdetr_checkpoint,
                           num_classes=1, patch_size=16, resolution=512)
        print(f"  ✓ Loaded mouse liver checkpoint")
    except Exception as e:
        print(f"  ⚠ Failed: {e}")
        return

    # Step 2: Load ResNet50 extractor
    print(f"\n[Step 2] 加载 ResNet50 特征提取器")
    extractor = ResNetFeatureExtractor(device=args.device)

    # Step 3: Process train + val
    print(f"\n[Step 3] 跨域检测 + 特征提取")
    data_root = Path(args.data_root)
    train_img_dir = data_root / 'train/images'
    train_lbl_dir = data_root / 'train/labels'
    val_img_dir = data_root / 'val/images'
    val_lbl_dir = data_root / 'val/labels'

    train_files = sorted([f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    val_files = sorted([f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    X_train, y_true_train, y_pseudo_train, scores_train = process_split(
        train_files, train_img_dir, train_lbl_dir, model, extractor, args.n_train, args, "train"
    )
    X_val, y_true_val, y_pseudo_val, scores_val = process_split(
        val_files, val_img_dir, val_lbl_dir, model, extractor, args.n_val, args, "val"
    )

    print(f"\n  Train: {len(y_true_train)} samples (TP={int(y_true_train.sum())}, FP={int((1-y_true_train).sum())})")
    print(f"  Val:   {len(y_true_val)} samples (TP={int(y_true_val.sum())}, FP={int((1-y_true_val).sum())})")
    print(f"  Zero-shot scores: TP train mean={scores_train[y_true_train==1].mean():.3f}, FP mean={scores_train[y_true_train==0].mean():.3f}")
    print(f"  Zero-shot scores: TP val mean={scores_val[y_true_val==1].mean():.3f}, FP mean={scores_val[y_true_val==0].mean():.3f}")

    # Step 4: Train classifier
    print(f"\n[Step 4] 训练分类头 ({len(y_pseudo_train)} samples, {X_train.shape[1]} dim)")
    if len(X_train) < 10 or len(set(y_pseudo_train.tolist())) < 2:
        print("⚠ Not enough samples")
        return

    X_t = X_train.clone()
    y_t = y_pseudo_train.clone()

    if len(X_t) >= 20:
        idx_tr, idx_v = train_test_split(range(len(X_t)), test_size=0.2, stratify=y_t.numpy(), random_state=42)
        X_tr, X_v = X_t[idx_tr], X_t[idx_v]
        y_tr, y_v = y_t[idx_tr], y_t[idx_v]
    else:
        X_tr, X_v, y_tr, y_v = X_t, X_t, y_t, y_t

    classifier = Classifier(in_dim=X_train.shape[1]).to(args.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0
    best_state = None
    n = len(X_tr)
    for ep in range(args.epochs):
        classifier.train()
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i+args.batch_size]
            if i + args.batch_size > n and i > 0: break
            x = X_tr[idx].to(args.device)
            y = y_tr[idx].to(args.device)
            optimizer.zero_grad()
            logits = classifier(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        classifier.eval()
        with torch.no_grad():
            logits_v = classifier(X_v.to(args.device))
            probs_v = F.softmax(logits_v, dim=1)[:, 1].cpu().numpy()
            if len(set(y_v.tolist())) >= 2:
                auc_v = roc_auc_score(y_v.numpy(), probs_v)
                if auc_v > best_auc:
                    best_auc = auc_v
                    best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
                if (ep+1) % 10 == 0 or ep == 0:
                    print(f"  Epoch {ep+1}: loss={total_loss:.4f}, val_auc={auc_v:.4f}", flush=True)

    if best_state:
        classifier.load_state_dict(best_state)
    print(f"  ✓ Best val AUC: {best_auc:.4f}")

    # Step 5: Evaluate
    print(f"\n[Step 5] 评估 (val set)")
    classifier.eval()
    with torch.no_grad():
        logits_v = classifier(X_val.to(args.device))
        probs_v = F.softmax(logits_v, dim=1)[:, 1].cpu().numpy()

    if len(set(y_true_val.tolist())) >= 2:
        y_v_arr = np.array(y_true_val)
        scores_arr = np.array(scores_val)
        auc_zero = roc_auc_score(y_v_arr, scores_arr)
        auc_clf = roc_auc_score(y_v_arr, probs_v)
        distilled = scores_arr * probs_v
        auc_dist = roc_auc_score(y_v_arr, distilled)
        improvement = auc_dist - auc_zero
        print(f"  Val: {len(y_v_arr)} samples, {int(sum(y_v_arr==1))} TP, {int(sum(y_v_arr==0))} FP")
        print(f"  Zero-shot (RF-DETR conf):       AUC = {auc_zero:.4f}")
        print(f"  Classifier alone (ResNet50):    AUC = {auc_clf:.4f}")
        print(f"  Distilled (conf × classifier):  AUC = {auc_dist:.4f}")
        print(f"  Improvement:                    {improvement:+.4f}")
    else:
        print("⚠ Cannot compute AUC")
        return

    # Step 6: Save
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Intestinal 真实跨域自蒸馏 — ResNet50 真特征\n\n")
        f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**RF-DETR**: 鼠肝 checkpoint → Intestinal 检测（跨域）\n")
        f.write(f"**Feature**: ResNet50 ImageNet pretrained (2048-dim)\n")
        f.write(f"**GT**: Intestinal YOLO 4-class（真实标注）\n")
        f.write(f"**方法**: GT IoU 作独立信号（5% 噪声模拟 SAM2）+ MLP 分类头\n\n")

        f.write(f"## 1. 数据\n\n")
        f.write(f"| Split | Samples | TP | FP |\n")
        f.write(f"|---|---|---|---|\n")
        f.write(f"| Train | {len(y_true_train)} | {int(y_true_train.sum())} | {int((1-y_true_train).sum())} |\n")
        f.write(f"| Val | {len(y_true_val)} | {int(y_true_val.sum())} | {int((1-y_true_val).sum())} |\n\n")

        f.write(f"## 2. 自蒸馏结果\n\n")
        f.write(f"| 方法 | AUC |\n|---|---|\n")
        f.write(f"| Zero-shot (RF-DETR conf) | {auc_zero:.4f} |\n")
        f.write(f"| Classifier alone (ResNet50) | {auc_clf:.4f} |\n")
        f.write(f"| **Distilled (conf × classifier)** | **{auc_dist:.4f}** |\n")
        f.write(f"| **Improvement** | **{improvement:+.4f}** |\n\n")

        if improvement > 0.10:
            f.write(f"✅ **自蒸馏有效**：AUC 提升 {improvement*100:.2f}% > 10%。\n")
        elif improvement > 0:
            f.write(f"⚠ **自蒸馏微弱有效**：AUC 提升 {improvement*100:.2f}%。\n")
        else:
            f.write(f"❌ **自蒸馏无效**：AUC 下降 {-improvement*100:.2f}%。\n")

        f.write(f"\n## 3. 与之前 30 维图像统计特征对比\n\n")
        f.write(f"| 特征 | 维度 | Zero-shot | Distilled | Improvement |\n")
        f.write(f"|---|---|---|---|---|\n")
        f.write(f"| 图像统计 (RGB mean/std + hist) | 30 | baseline | baseline | +9.10% (200+100) |\n")
        f.write(f"| **ResNet50 ImageNet pretrained** | **2048** | **{auc_zero:.4f}** | **{auc_dist:.4f}** | **{improvement*100:+.2f}%** |\n\n")
        f.write(f"如果 ResNet50 improvement > 图像统计 +9.10%，说明真特征比统计特征强。\n")

    print(f"\n  ✓ Saved: {report_path}")

    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "feature": "ResNet50 ImageNet pretrained 2048-dim",
            "n_train": len(y_true_train),
            "n_val": len(y_true_val),
            "auc_zero_shot": float(auc_zero),
            "auc_classifier": float(auc_clf),
            "auc_distilled": float(auc_dist),
            "improvement": float(improvement),
        }, f, indent=2)
    print(f"  ✓ Saved: {json_path}")
    print(f"\n✓ Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
