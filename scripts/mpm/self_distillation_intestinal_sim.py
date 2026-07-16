r"""
Intestinal 自蒸馏 — 真实 RF-DETR + SAM2 + YOLO GT 验证
======================================================

简化版：不依赖 RF-DETR backbone 特征，用 crop 的图像统计特征。
目的：用更大的数据集（ intestinal 756+84）验证自蒸馏方法是否真的有效，
      排除 MPM 实验中"10 个正样本太少"的问题。

实验设计：
  1. 用 YOLO GT 作为 TP + 随机生成 FP（模拟 RF-DETR 检测）
  2. 用模拟 SAM2 IoU（TP 高 IoU, FP 低 IoU + 5% 噪声）
  3. 训练分类头在图像统计特征上
  4. 评估真实 TP/FP 标签上的 AUC

关键：如果分类头能学到 TP vs FP 的差异（即使特征简单），
      说明方法本身可行；如果不行，说明需要更强的特征。

Usage (云 VM, CPU):
    cd /home/z/my-project/organoid-fl
    python scripts/mpm/self_distillation_intestinal_sim.py \
        --data-root data/intestinal_organoid/OrganoidDataset \
        --output-dir results/sd_intestinal_sim \
        --n-train 300 --n-val 100 --epochs 50
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

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="results/sd_intestinal_sim")
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pos-iou-thresh", type=float, default=0.50)
    p.add_argument("--neg-iou-thresh", type=float, default=0.10)
    p.add_argument("--sam2-noise", type=float, default=0.05, help="SAM2 error rate")
    return p.parse_args()


def load_yolo_labels(label_path):
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, xc, yc, w, h = map(float, parts[:5])
            boxes.append([xc-w/2, yc-h/2, xc+w/2, yc+h/2])
    return boxes


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1) * (y2-y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def extract_simple_features(img_np, boxes):
    """从图像 crop 中提取简单特征（颜色直方图 + 统计）"""
    h, w = img_np.shape[:2]
    features = []
    for box in boxes:
        x1, y1, x2, y2 = box
        px1, py1 = max(0, int(x1*w)), max(0, int(y1*h))
        px2, py2 = min(w, int(x2*w)), min(h, int(y2*h))
        if px2 <= px1 or py2 <= py1:
            features.append(np.zeros(30, dtype=np.float32))
            continue
        crop = img_np[py1:py2, px1:px2]
        # 30 features: 3 channels × (mean + std + 8-bin hist)
        feats = []
        for c in range(3):
            ch = crop[:, :, c].astype(np.float32) / 255.0
            feats.append(ch.mean())
            feats.append(ch.std())
            hist, _ = np.histogram(ch, bins=8, range=(0, 1), density=True)
            feats.extend(hist.tolist())
        features.append(np.array(feats, dtype=np.float32))
    return np.stack(features) if features else np.zeros((0, 30))


class Classifier(nn.Module):
    def __init__(self, in_dim=30, hidden=64, n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Intestinal 自蒸馏验证（simulation, 不需 RF-DETR/SAM2）")
    print("=" * 70)

    # Step 0: Load data
    print(f"\n[Step 0] 加载数据: {args.data_root}")
    train_img_dir = Path(args.data_root) / "train" / "images"
    train_lbl_dir = Path(args.data_root) / "train" / "labels"
    val_img_dir = Path(args.data_root) / "val" / "images"
    val_lbl_dir = Path(args.data_root) / "val" / "labels"

    train_files = sorted([f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    val_files = sorted([f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  Train: {len(train_files)} images, Val: {len(val_files)} images")

    # Step 1: Build dataset (TP from GT, FP from random boxes)
    print(f"\n[Step 1] 构建 dataset")

    def build_dataset(files, img_dir, lbl_dir, n_max, split_name, seed_base=0):
        rng = np.random.default_rng(seed=seed_base)
        all_features = []
        all_true_labels = []  # 1 = TP (real GT), 0 = FP (random box)
        all_pseudo_labels = []  # from simulated SAM2 IoU
        all_zero_shot_scores = []

        for i, fname in enumerate(files):
            if len(all_true_labels) >= n_max:
                break
            img_path = img_dir / fname
            lbl_path = lbl_dir / fname.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            img = np.array(Image.open(img_path).convert("RGB"))
            h, w = img.shape[:2]

            gt_boxes = load_yolo_labels(lbl_path)

            # TP: GT boxes (with simulated RF-DETR conf)
            for gt in gt_boxes:
                if len(all_true_labels) >= n_max:
                    break
                # HARD scenario: TP conf same distribution as FP (zero-shot AUC ~ 0.5)
                # This tests whether classifier can learn from features when conf is useless
                rf_score = float(rng.uniform(0.01, 0.50))  # same as FP range
                # Simulated SAM2 IoU (TP → high, 0.7-1.0)
                sam2_iou = float(rng.uniform(0.70, 1.00))
                # 5% noise: SAM2 makes mistake
                if rng.random() < args.sam2_noise:
                    sam2_iou = float(rng.uniform(0.0, 0.30))

                if sam2_iou >= args.pos_iou_thresh:
                    pseudo = 1
                elif sam2_iou < args.neg_iou_thresh:
                    pseudo = 0
                else:
                    continue

                all_true_labels.append(1)
                all_pseudo_labels.append(pseudo)
                all_zero_shot_scores.append(rf_score)

            # FP: random boxes (with simulated RF-DETR conf 0.01-0.40)
            n_fp_per_img = max(2, len(gt_boxes))
            for _ in range(n_fp_per_img):
                if len(all_true_labels) >= n_max:
                    break
                bw = float(rng.uniform(0.05, 0.20))
                bh = float(rng.uniform(0.05, 0.20))
                x1 = float(rng.uniform(0, 1-bw))
                y1 = float(rng.uniform(0, 1-bh))
                box = [x1, y1, x1+bw, y1+bh]

                # Check overlap with GT
                max_gt_iou = max([compute_iou(box, g) for g in gt_boxes] + [0.0])
                # Skip if accidentally high GT IoU (would be TP not FP)
                if max_gt_iou > 0.30:
                    continue

                rf_score = float(rng.uniform(0.01, 0.40))
                # Simulated SAM2 IoU (FP → low, 0.0-0.25)
                sam2_iou = float(rng.uniform(0.00, 0.25))
                if rng.random() < args.sam2_noise:
                    sam2_iou = float(rng.uniform(0.70, 1.00))

                if sam2_iou >= args.pos_iou_thresh:
                    pseudo = 1
                elif sam2_iou < args.neg_iou_thresh:
                    pseudo = 0
                else:
                    continue

                all_true_labels.append(0)
                all_pseudo_labels.append(pseudo)
                all_zero_shot_scores.append(rf_score)

            # Extract features for all collected boxes in this image
            # (we collected labels above, now get features)
            # Re-walk GT + FP boxes to extract features
            collected_boxes = []
            for gt in gt_boxes:
                collected_boxes.append(gt)
            for _ in range(n_fp_per_img):
                bw = float(rng.uniform(0.05, 0.20))
                bh = float(rng.uniform(0.05, 0.20))
                x1 = float(rng.uniform(0, 1-bw))
                y1 = float(rng.uniform(0, 1-bh))
                collected_boxes.append([x1, y1, x1+bw, y1+bh])

            # Actually, just extract features for all GT + FP we just added
            # For simplicity, re-extract per-image with all det_boxes
            # but this creates a mismatch. Let me simplify:
            # collect features in the same loop
            # (will fix below)

            if (i+1) % 50 == 0:
                n_pos = sum(all_true_labels)
                print(f"  {split_name} {i+1}/{len(files)}: total={len(all_true_labels)} pos={n_pos}", flush=True)

        # Re-extract features for all collected labels by re-processing images
        # Simpler: extract features during the loop
        # But we already collected labels. Let me just re-do the loop with features.
        # For now: simulate features as random (placeholder)
        # Actually, let me redo this properly:

        # Restart with features
        rng = np.random.default_rng(seed=seed_base)
        all_features = []
        all_true_labels = []
        all_pseudo_labels = []
        all_zero_shot_scores = []

        for i, fname in enumerate(files):
            if len(all_true_labels) >= n_max:
                break
            img_path = img_dir / fname
            lbl_path = lbl_dir / fname.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            img = np.array(Image.open(img_path).convert("RGB"))

            gt_boxes = load_yolo_labels(lbl_path)

            # Generate TP + FP
            tp_boxes = []
            tp_rf_scores = []
            tp_sam2_ious = []
            for gt in gt_boxes:
                if len(all_true_labels) + len(tp_boxes) >= n_max:
                    break
                rf = float(rng.uniform(0.50, 0.95))
                s2 = float(rng.uniform(0.70, 1.00))
                if rng.random() < args.sam2_noise:
                    s2 = float(rng.uniform(0.0, 0.30))
                tp_boxes.append(gt)
                tp_rf_scores.append(rf)
                tp_sam2_ious.append(s2)

            fp_boxes = []
            fp_rf_scores = []
            fp_sam2_ious = []
            n_fp = max(2, len(gt_boxes))
            for _ in range(n_fp):
                if len(all_true_labels) + len(tp_boxes) + len(fp_boxes) >= n_max:
                    break
                bw = float(rng.uniform(0.05, 0.20))
                bh = float(rng.uniform(0.05, 0.20))
                x1 = float(rng.uniform(0, 1-bw))
                y1 = float(rng.uniform(0, 1-bh))
                box = [x1, y1, x1+bw, y1+bh]
                max_gt_iou = max([compute_iou(box, g) for g in gt_boxes] + [0.0])
                if max_gt_iou > 0.30:
                    continue
                rf = float(rng.uniform(0.01, 0.40))
                s2 = float(rng.uniform(0.00, 0.25))
                if rng.random() < args.sam2_noise:
                    s2 = float(rng.uniform(0.70, 1.00))
                fp_boxes.append(box)
                fp_rf_scores.append(rf)
                fp_sam2_ious.append(s2)

            # Extract features
            all_boxes = tp_boxes + fp_boxes
            if not all_boxes:
                continue
            feats = extract_simple_features(img, all_boxes)

            for j in range(len(all_boxes)):
                is_tp = j < len(tp_boxes)
                s2 = tp_sam2_ious[j] if is_tp else fp_sam2_ious[j - len(tp_boxes)]
                rf = tp_rf_scores[j] if is_tp else fp_rf_scores[j - len(tp_boxes)]

                if s2 >= args.pos_iou_thresh:
                    pseudo = 1
                elif s2 < args.neg_iou_thresh:
                    pseudo = 0
                else:
                    continue

                all_features.append(feats[j])
                all_true_labels.append(1 if is_tp else 0)
                all_pseudo_labels.append(pseudo)
                all_zero_shot_scores.append(rf)

            if (i+1) % 50 == 0:
                n_pos = sum(all_true_labels)
                print(f"  {split_name} {i+1}/{len(files)}: total={len(all_true_labels)} pos={n_pos}", flush=True)

        return (
            np.stack(all_features) if all_features else np.zeros((0, 30)),
            np.array(all_true_labels, dtype=np.int64),
            np.array(all_pseudo_labels, dtype=np.int64),
            np.array(all_zero_shot_scores, dtype=np.float32),
        )

    X_train, y_true_train, y_pseudo_train, scores_train = build_dataset(
        train_files, train_img_dir, train_lbl_dir, args.n_train, "train", seed_base=42
    )
    X_val, y_true_val, y_pseudo_val, scores_val = build_dataset(
        val_files, val_img_dir, val_lbl_dir, args.n_val, "val", seed_base=99
    )

    print(f"\n  Train: {len(y_true_train)} samples (true pos={sum(y_true_train)}, neg={sum(1-y_true_train)})")
    print(f"  Val:   {len(y_true_val)} samples (true pos={sum(y_true_val)}, neg={sum(1-y_true_val)})")
    print(f"  Pseudo train: pos={sum(y_pseudo_train)}, neg={sum(1-y_pseudo_train)}")
    print(f"  Pseudo val:   pos={sum(y_pseudo_val)}, neg={sum(1-y_pseudo_val)}")

    if len(X_train) < 10 or len(set(y_pseudo_train.tolist())) < 2:
        print("⚠ Not enough samples")
        return

    # Step 2: Train classifier
    print(f"\n[Step 2] 训练分类头 ({len(y_pseudo_train)} samples, {X_train.shape[1]} dim)")
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_pseudo_train, dtype=torch.long)

    # Use full set as both train and val for small data
    if len(X_t) >= 20:
        from sklearn.model_selection import train_test_split as _split
        idx_tr, idx_v = _split(range(len(X_t)), test_size=0.2, stratify=y_t.numpy(), random_state=42)
        X_tr, X_v = X_t[idx_tr], X_t[idx_v]
        y_tr, y_v = y_t[idx_tr], y_t[idx_v]
    else:
        X_tr, X_v, y_tr, y_v = X_t, X_t, y_t, y_t

    n = len(X_tr)
    print(f"  Train: {n}, Val: {len(X_v)}")

    classifier = Classifier(in_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0
    best_state = None
    for ep in range(args.epochs):
        classifier.train()
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i+args.batch_size]
            if i + args.batch_size > n and i > 0:
                break  # skip last partial batch
            x = X_tr[idx]
            y = y_tr[idx]
            optimizer.zero_grad()
            logits = classifier(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        classifier.eval()
        with torch.no_grad():
            logits_v = classifier(X_v)
            probs_v = F.softmax(logits_v, dim=1)[:, 1].numpy()
            if len(set(y_v.tolist())) >= 2:
                auc = roc_auc_score(y_v.numpy(), probs_v)
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
                if (ep+1) % 10 == 0 or ep == 0:
                    print(f"  Epoch {ep+1}: loss={total_loss:.4f}, val_auc={auc:.4f}", flush=True)

    if best_state:
        classifier.load_state_dict(best_state)
    print(f"  ✓ Best val AUC: {best_auc:.4f}")

    # Save
    ckpt_path = output_dir / "classifier.pt"
    torch.save({"state_dict": classifier.state_dict(), "in_dim": classifier.fc1.in_features}, ckpt_path)

    # Step 3: Evaluate
    print(f"\n[Step 3] 评估 (val set)")
    classifier.eval()
    X_v_all = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        logits = classifier(X_v_all)
        probs = F.softmax(logits, dim=1)[:, 1].numpy()

    if len(set(y_true_val.tolist())) >= 2:
        # Debug: score distributions
        y_v_arr = np.array(y_true_val) if not isinstance(y_true_val, np.ndarray) else y_true_val
        scores_arr = np.array(scores_val) if not isinstance(scores_val, np.ndarray) else scores_val
        tp_mask = y_v_arr == 1
        fp_mask = y_v_arr == 0
        print(f"  Val: {len(y_v_arr)} samples, {int(sum(tp_mask))} TP, {int(sum(fp_mask))} FP")
        if tp_mask.sum() > 0 and fp_mask.sum() > 0:
            print(f"  Zero-shot scores: TP mean={scores_arr[tp_mask].mean():.3f}, FP mean={scores_arr[fp_mask].mean():.3f}")

        auc_zero = roc_auc_score(y_v_arr, scores_arr)
        auc_classifier = roc_auc_score(y_v_arr, probs)
        distilled = scores_arr * probs
        auc_distilled = roc_auc_score(y_v_arr, distilled)
        print(f"  Zero-shot (RF-DETR conf):       AUC = {auc_zero:.4f}")
        print(f"  Classifier alone:               AUC = {auc_classifier:.4f}")
        print(f"  Distilled (conf × classifier):  AUC = {auc_distilled:.4f}")
        print(f"  Improvement (distilled - zero):  {auc_distilled - auc_zero:+.4f}")
    else:
        auc_zero = auc_classifier = auc_distilled = float('nan')
        print(f"  ⚠ Cannot compute AUC")

    # Step 4: Report
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Intestinal 自蒸馏验证实验（模拟 RF-DETR + SAM2）\n\n")
        f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**数据集**: Intestinal organoid (YOLO 4-class)\n")
        f.write(f"**方法**: 模拟 RF-DETR conf + SAM2 IoU，用 YOLO GT 作真值\n")
        f.write(f"**目的**: 验证自蒸馏方法在样本量充足时是否有效\n\n")

        f.write(f"## 1. 数据\n\n")
        f.write(f"| Split | N | True TP | True FP | Pseudo TP | Pseudo FP |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        f.write(f"| Train | {len(y_true_train)} | {sum(y_true_train)} | {sum(1-y_true_train)} | {sum(y_pseudo_train)} | {sum(1-y_pseudo_train)} |\n")
        f.write(f"| Val | {len(y_true_val)} | {sum(y_true_val)} | {sum(1-y_true_val)} | {sum(y_pseudo_val)} | {sum(1-y_pseudo_val)} |\n\n")

        f.write(f"## 2. 结果\n\n")
        f.write(f"| 方法 | AUC |\n")
        f.write(f"|---|---|\n")
        f.write(f"| RF-DETR conf (zero-shot) | {auc_zero:.4f} |\n")
        f.write(f"| Classifier alone | {auc_classifier:.4f} |\n")
        f.write(f"| Distilled (conf × classifier) | {auc_distilled:.4f} |\n")
        f.write(f"| **Improvement** | **{auc_distilled - auc_zero:+.4f}** |\n\n")

        f.write(f"## 3. 结论\n\n")
        imp = auc_distilled - auc_zero
        if imp > 0.05:
            f.write(f"✅ **自蒸馏有效**：AUC 提升 {imp*100:.1f}%。\n")
        elif imp > 0:
            f.write(f"⚠ **自蒸馏微弱有效**：AUC 提升 {imp*100:.1f}%。\n")
        else:
            f.write(f"❌ **自蒸馏无效**：AUC 下降 {-imp*100:.1f}%。\n")

        f.write(f"\n## 4. 对比 MPM 实验\n\n")
        f.write(f"| 实验 | N train | N val | AUC zero | AUC distill | 改进 |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        f.write(f"| MPM (10 pos) | 3131 | - | 0.9779 | 1.0000 | +0.0221 |\n")
        f.write(f"| Intestinal sim | {len(y_true_train)} | {len(y_true_val)} | {auc_zero:.4f} | {auc_distilled:.4f} | {imp:+.4f} |\n\n")

        f.write(f"## 5. 说明\n\n")
        f.write(f"- 此实验用**模拟 RF-DETR conf + 模拟 SAM2 IoU**（非真实检测）\n")
        f.write(f"- 5% SAM2 噪声（5% 概率 SAM2 给出错误 IoU）\n")
        f.write(f"- 特征：30维图像统计（3通道×(mean+std+8-bin hist)）\n")
        f.write(f"- 如果分类头能在这些特征上学习 TP vs FP，说明方法可行\n")

    print(f"  ✓ Saved: {report_path}")

    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_train": len(y_true_train),
            "n_val": len(y_true_val),
            "auc_zero_shot": float(auc_zero),
            "auc_classifier": float(auc_classifier),
            "auc_distilled": float(auc_distilled),
            "improvement": float(auc_distilled - auc_zero),
            "sam2_noise": args.sam2_noise,
        }, f, indent=2)
    print(f"  ✓ Saved: {json_path}")

    print(f"\n✓ Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
