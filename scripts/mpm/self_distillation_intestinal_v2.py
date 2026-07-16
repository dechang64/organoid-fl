r"""
Intestinal 自蒸馏验证（修正版）
=================================

修正设计：
  - 不用 pseudo label 过滤样本（避免过滤后 val 集只剩易分样本）
  - 所有 TP + FP 都进入训练和评估
  - 训练分类头用 pseudo label（来自模拟 SAM2 IoU + 5% 噪声）
  - 评估用 true label（来自 YOLO GT）

  - TP conf: 0.05-0.50（与 FP 重叠，模拟跨域失效）
  - FP conf: 0.01-0.40
  - 真实 SAM2 IoU: TP=0.7-1.0, FP=0.0-0.3（5% 噪声翻转）
  - pseudo label: SAM2 IoU ≥ 0.5 = 1, < 0.5 = 0（不再过滤）

关键期望：
  - Zero-shot AUC ≈ 0.5-0.6（conf 不能区分 TP/FP）
  - Distilled AUC > 0.7（分类头学到 SAM2 信号）
  - 如果 distilled > zero+0.1 → 自蒸馏有效
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


def load_yolo_labels(lbl_path):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5: continue
            cls, xc, yc, w, h = map(float, p[:5])
            boxes.append([xc-w/2, yc-h/2, xc+w/2, yc+h/2])
    return boxes


def compute_iou(box, boxes):
    """IoU of one box vs list."""
    if not boxes: return 0.0
    x1 = np.maximum(box[0], [b[0] for b in boxes])
    y1 = np.maximum(box[1], [b[1] for b in boxes])
    x2 = np.minimum(box[2], [b[2] for b in boxes])
    y2 = np.minimum(box[3], [b[3] for b in boxes])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    a1 = (box[2]-box[0]) * (box[3]-box[1])
    a2 = np.array([(b[2]-b[0])*(b[3]-b[1]) for b in boxes])
    return inter / np.maximum(a1 + a2 - inter, 1e-6)


def extract_features(img, boxes):
    """30-dim features: 3 channels × (mean+std+8-bin hist)."""
    feats = []
    for box in boxes:
        x1, y1, x2, y2 = box
        h, w = img.shape[:2]
        px1 = int(x1 * w); py1 = int(y1 * h)
        px2 = int(x2 * w); py2 = int(y2 * h)
        px1 = max(0, px1); py1 = max(0, py1)
        px2 = min(w, px2); py2 = min(h, py2)
        if px2 <= px1 or py2 <= py1:
            feats.append(np.zeros(30))
            continue
        crop = img[py1:py2, px1:px2]
        f = []
        for c in range(3):
            ch = crop[:,:,c].astype(np.float32)
            f.append(ch.mean() / 255.0)
            f.append(ch.std() / 128.0)
            hist, _ = np.histogram(ch, bins=8, range=(0, 255))
            f.extend(hist / max(crop.size, 1))
        feats.append(np.array(f))
    return np.array(feats)


def build_dataset(data_root, n_max, split, seed):
    """Build TP + FP samples with simulated RF-DETR conf + SAM2 IoU."""
    img_dir = Path(data_root) / split / "images"
    lbl_dir = Path(data_root) / split / "labels"
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg','.jpeg','.png'))])
    rng = np.random.default_rng(seed)

    all_features = []
    all_true_labels = []
    all_rf_scores = []
    all_sam2_ious = []

    for i, fname in enumerate(files):
        if len(all_true_labels) >= n_max: break
        img_path = img_dir / fname
        lbl_path = lbl_dir / fname.replace('.jpg','.txt').replace('.jpeg','.txt').replace('.png','.txt')
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]
        gt_boxes = load_yolo_labels(lbl_path)

        # TP samples (from GT)
        for gt in gt_boxes:
            if len(all_true_labels) >= n_max: break
            # RF-DETR conf for TP: 0.05-0.50 (overlaps with FP)
            rf = float(rng.uniform(0.05, 0.50))
            # SAM2 IoU: TP -> 0.70-1.00 (5% noise: flip to 0.0-0.30)
            s2 = float(rng.uniform(0.70, 1.00))
            if rng.random() < 0.05:
                s2 = float(rng.uniform(0.0, 0.30))
            all_features.append(extract_features(img, [gt])[0])
            all_true_labels.append(1)
            all_rf_scores.append(rf)
            all_sam2_ious.append(s2)

        # FP samples (random boxes, no GT overlap)
        n_fp = max(2, len(gt_boxes))
        attempts = 0
        added = 0
        while added < n_fp and attempts < n_fp * 3 and len(all_true_labels) < n_max:
            attempts += 1
            bw = float(rng.uniform(0.05, 0.20))
            bh = float(rng.uniform(0.05, 0.20))
            x1 = float(rng.uniform(0, 1-bw))
            y1 = float(rng.uniform(0, 1-bh))
            box = [x1, y1, x1+bw, y1+bh]
            # Skip if high GT IoU
            ious = compute_iou(box, gt_boxes)
            if max(ious) > 0.10: continue
            # RF-DETR conf for FP: 0.01-0.40
            rf = float(rng.uniform(0.01, 0.40))
            # SAM2 IoU: FP -> 0.00-0.25 (5% noise: flip to 0.7-1.0)
            s2 = float(rng.uniform(0.0, 0.25))
            if rng.random() < 0.05:
                s2 = float(rng.uniform(0.70, 1.00))
            all_features.append(extract_features(img, [box])[0])
            all_true_labels.append(0)
            all_rf_scores.append(rf)
            all_sam2_ious.append(s2)
            added += 1

        if (i+1) % 50 == 0:
            print(f"    {split} {i+1}/{len(files)}: total={len(all_true_labels)}", flush=True)

    return (
        np.array(all_features),
        np.array(all_true_labels),
        np.array(all_rf_scores),
        np.array(all_sam2_ious),
    )


class Classifier(nn.Module):
    def __init__(self, in_dim=30, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2)
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 70)
    print("Intestinal 自蒸馏验证（修正版：保留所有 TP+FP）")
    print("=" * 70)
    print(f"\n[Step 0] 加载数据: {args.data_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 1] 构建 dataset")
    X_train, y_true_train, scores_train, sam2_train = build_dataset(
        args.data_root, args.n_train, "train", seed=42
    )
    X_val, y_true_val, scores_val, sam2_val = build_dataset(
        args.data_root, args.n_val, "val", seed=99
    )

    # Pseudo labels from SAM2 IoU (no filtering!)
    y_pseudo_train = (sam2_train >= 0.50).astype(np.int64)
    y_pseudo_val = (sam2_val >= 0.50).astype(np.int64)

    n_tp = sum(y_true_train); n_fp = len(y_true_train) - n_tp
    print(f"  Train: {len(y_true_train)} (TP={n_tp}, FP={n_fp})")
    n_tp_v = sum(y_true_val); n_fp_v = len(y_true_val) - n_tp_v
    print(f"  Val:   {len(y_true_val)} (TP={n_tp_v}, FP={n_fp_v})")
    print(f"  Pseudo train: TP={sum(y_pseudo_train)}, FP={sum(1-y_pseudo_train)}")
    print(f"  Pseudo val:   TP={sum(y_pseudo_val)}, FP={sum(1-y_pseudo_val)}")

    # Debug: score distributions on FULL val (no filter)
    print(f"\n  === Val set 分布 ===")
    tp_mask = y_true_val == 1
    fp_mask = y_true_val == 0
    print(f"  RF-DETR conf: TP mean={scores_val[tp_mask].mean():.3f}, FP mean={scores_val[fp_mask].mean():.3f}")
    print(f"  SAM2 IoU:     TP mean={sam2_val[tp_mask].mean():.3f}, FP mean={sam2_val[fp_mask].mean():.3f}")

    # Step 2: Train classifier
    print(f"\n[Step 2] 训练分类头 ({len(y_pseudo_train)} samples, {X_train.shape[1]} dim)")
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_pseudo_train, dtype=torch.long)
    X_v_t = torch.tensor(X_val, dtype=torch.float32)

    if len(X_t) >= 20:
        idx_tr, idx_v = train_test_split(range(len(X_t)), test_size=0.2,
                                          stratify=y_t.numpy(), random_state=42)
        X_tr = X_t[idx_tr]; X_v = X_t[idx_v]
        y_tr = y_t[idx_tr]; y_v = y_t[idx_v]
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
            if len(idx) < 4: break
            x = X_tr[idx]; y = y_tr[idx]
            optimizer.zero_grad()
            logits = classifier(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        classifier.eval()
        with torch.no_grad():
            logits_v = classifier(X_v)
            probs_v = F.softmax(logits_v, dim=1)[:, 1].cpu().numpy()
        if len(set(y_v.numpy())) >= 2:
            auc = roc_auc_score(y_v.numpy(), probs_v)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
            if (ep+1) % 10 == 0 or ep == 0:
                print(f"  Epoch {ep+1}: loss={total_loss:.4f}, val_auc={auc:.4f}")

    if best_state:
        classifier.load_state_dict(best_state)
    print(f"  ✓ Best val AUC: {best_auc:.4f}")

    # Step 3: Evaluate on FULL val (no filter!)
    print(f"\n[Step 3] 评估 (val set, no filter)")
    classifier.eval()
    with torch.no_grad():
        logits = classifier(X_v_t)
        probs = F.softmax(logits, dim=1)[:, 1].numpy()

    # Three metrics
    auc_zero = roc_auc_score(y_true_val, scores_val)
    auc_classifier = roc_auc_score(y_true_val, probs)
    distilled = scores_val * probs
    auc_distilled = roc_auc_score(y_true_val, distilled)

    print(f"  Zero-shot (RF-DETR conf):       AUC = {auc_zero:.4f}")
    print(f"  Classifier alone:               AUC = {auc_classifier:.4f}")
    print(f"  Distilled (conf × classifier):  AUC = {auc_distilled:.4f}")
    print(f"  Improvement (distilled - zero):  {auc_distilled - auc_zero:+.4f}")

    # Step 4: Save
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Intestinal 自蒸馏验证（修正版）\n\n")
        f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 1. 设计\n\n")
        f.write(f"- TP: YOLO GT boxes + RF-DETR conf 0.05-0.50 + SAM2 IoU 0.70-1.00 (5% 噪声)\n")
        f.write(f"- FP: 随机 box + RF-DETR conf 0.01-0.40 + SAM2 IoU 0.00-0.25 (5% 噪声)\n")
        f.write(f"- 保留所有样本（不过滤）\n")
        f.write(f"- 训练分类头用 pseudo label，评估用 true label\n\n")
        f.write(f"## 2. 数据\n\n")
        f.write(f"| Split | Total | TP | FP |\n|---|---|---|---|\n")
        f.write(f"| Train | {len(y_true_train)} | {n_tp} | {n_fp} |\n")
        f.write(f"| Val | {len(y_true_val)} | {n_tp_v} | {n_fp_v} |\n\n")
        f.write(f"## 3. 结果\n\n")
        f.write(f"| 方法 | AUC |\n|---|---|\n")
        f.write(f"| Zero-shot (RF-DETR conf) | {auc_zero:.4f} |\n")
        f.write(f"| Classifier alone | {auc_classifier:.4f} |\n")
        f.write(f"| Distilled (conf × classifier) | {auc_distilled:.4f} |\n")
        f.write(f"| **Improvement** | **{auc_distilled - auc_zero:+.4f}** |\n\n")
        imp = auc_distilled - auc_zero
        if imp > 0.10:
            f.write(f"✅ **自蒸馏有效**：AUC 提升 {imp*100:.1f}% > 10%。\n")
        elif imp > 0:
            f.write(f"⚠ **自蒸馏微弱有效**：AUC 提升 {imp*100:.1f}%。\n")
        else:
            f.write(f"❌ **自蒸馏无效**：AUC 下降 {-imp*100:.1f}%。\n")
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
        }, f, indent=2)
    print(f"  ✓ Saved: {json_path}")
    print(f"\n✓ Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
