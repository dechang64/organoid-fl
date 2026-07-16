r"""
鼠肝 cross-batch 自蒸馏验证（真实 RF-DETR 检测结果）
=====================================================

用已有 crop_metadata.json（含真实 RF-DETR conf + GT matched 标签），
验证自蒸馏方法在真实数据上是否有效。

数据：
  - b1: 23 crops (22 TP + 1 FP), RF-DETR conf 0.964
  - b3: 7 crops (3 TP + 4 FP), RF-DETR conf 0.339
  - 总计 30 样本 (25 TP + 5 FP)

实验设计：
  1. 用 b1+b3 的真实 RF-DETR conf 作 zero-shot score
  2. 用真实 matched (True/False) 作 true label
  3. 用图像统计特征 (RGB mean/std + 8-bin hist × 3 = 30 dim)
  4. 训练分类头，Leave-One-Batch-Out 交叉验证：
     - 训练 b1 → 测试 b3
     - 训练 b3 → 测试 b1
  5. 对比 Zero-shot AUC vs Distilled AUC

关键：样本量小 (30)，但 RF-DETR conf 是真实的（不是模拟的）
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def extract_features(img_path):
    """30-dim image statistics: RGB mean/std + 8-bin hist × 3 channels"""
    img = np.array(Image.open(img_path).convert("RGB"))
    if img.size == 0:
        return np.zeros(30)
    feats = []
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        feats.extend([ch.mean(), ch.std()])
        hist, _ = np.histogram(ch, bins=8, range=(0, 255))
        feats.extend(hist / max(ch.size, 1))
    return np.array(feats, dtype=np.float32)


class Classifier(nn.Module):
    def __init__(self, in_dim=30, hidden=64, n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


def load_batch(batch_dir, crop_meta_path):
    """Load one batch's crops + features + labels"""
    meta = json.load(open(crop_meta_path, encoding="utf-8"))
    features = []
    labels = []      # true: matched = TP
    scores = []      # RF-DETR conf
    paths = []
    for m in meta:
        crop_path = m["crop_path"]
        if not os.path.exists(crop_path):
            print(f"  ⚠ Missing: {crop_path}")
            continue
        feats = extract_features(crop_path)
        features.append(feats)
        labels.append(1 if m["matched"] else 0)
        scores.append(float(m["rfdetr_conf"]))
        paths.append(crop_path)
    return (
        np.array(features),
        np.array(labels, dtype=np.int64),
        np.array(scores, dtype=np.float32),
        paths,
    )


def train_and_eval(X_train, y_train, X_val, y_val, scores_val, epochs=100, lr=1e-3):
    """Train classifier on train set, evaluate on val set"""
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_v = torch.tensor(X_val, dtype=torch.float32)

    clf = Classifier(in_dim=X_train.shape[1])
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(epochs):
        clf.train()
        opt.zero_grad()
        logits = clf(X_t)
        loss = crit(logits, y_t)
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        probs = F.softmax(clf(X_v), dim=1)[:, 1].numpy()

    if len(set(y_val.tolist())) >= 2:
        auc_zero = roc_auc_score(y_val, scores_val)
        auc_clf = roc_auc_score(y_val, probs)
        distilled = scores_val * probs
        auc_dist = roc_auc_score(y_val, distilled)
    else:
        auc_zero = auc_clf = auc_dist = float("nan")

    return auc_zero, auc_clf, auc_dist, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/mouse_crops")
    parser.add_argument("--output-dir", default="results/sd_mouse_liver")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    print("=" * 70)
    print("鼠肝 cross-batch 自蒸馏验证（真实 RF-DETR 检测）")
    print("=" * 70)

    # Load batches
    batches = {}
    for b in ["b1", "b2", "b3"]:
        meta_path = data_root / b / "crop_metadata.json"
        if not meta_path.exists():
            print(f"  ⚠ {b}: no metadata")
            continue
        X, y, s, paths = load_batch(b, meta_path)
        if len(X) == 0:
            print(f"  ⚠ {b}: no crops")
            continue
        batches[b] = {"X": X, "y": y, "scores": s, "paths": paths}
        n_tp = int(y.sum())
        print(f"  {b}: {len(X)} crops (TP={n_tp}, FP={len(X)-n_tp}), conf mean={s.mean():.3f}")

    if len(batches) < 2:
        print("⚠ Need at least 2 batches")
        return

    # Leave-One-Batch-Out
    print(f"\n[Leave-One-Batch-Out 交叉验证]")
    results = {}
    all_true = []
    all_zero = []
    all_dist = []

    for test_batch in batches:
        train_batches = [b for b in batches if b != test_batch]
        X_train = np.concatenate([batches[b]["X"] for b in train_batches])
        y_train = np.concatenate([batches[b]["y"] for b in train_batches])
        X_val = batches[test_batch]["X"]
        y_val = batches[test_batch]["y"]
        scores_val = batches[test_batch]["scores"]

        auc_zero, auc_clf, auc_dist, probs = train_and_eval(
            X_train, y_train, X_val, y_val, scores_val, epochs=args.epochs
        )

        n_tp = int(y_val.sum())
        print(f"\n  Test={test_batch} (Train={'+'+ '+'.join(train_batches)}):")
        print(f"    Val: {len(y_val)} samples (TP={n_tp}, FP={len(y_val)-n_tp})")
        print(f"    Zero-shot AUC:    {auc_zero:.4f}")
        print(f"    Classifier AUC:   {auc_clf:.4f}")
        print(f"    Distilled AUC:    {auc_dist:.4f}")
        print(f"    Improvement:      {auc_dist - auc_zero:+.4f}")

        results[test_batch] = {
            "n_val": len(y_val),
            "n_tp": n_tp,
            "n_fp": len(y_val) - n_tp,
            "auc_zero_shot": float(auc_zero),
            "auc_classifier": float(auc_clf),
            "auc_distilled": float(auc_dist),
            "improvement": float(auc_dist - auc_zero),
        }
        all_true.extend(y_val.tolist())
        all_zero.extend(scores_val.tolist())
        all_dist.extend((scores_val * probs).tolist())

    # Overall (pooled)
    print(f"\n[Pooled 全样本评估]")
    all_true = np.array(all_true)
    all_zero = np.array(all_zero)
    all_dist = np.array(all_dist)
    if len(set(all_true.tolist())) >= 2:
        auc_zero_pooled = roc_auc_score(all_true, all_zero)
        auc_dist_pooled = roc_auc_score(all_true, all_dist)
        print(f"  Pooled Zero-shot AUC: {auc_zero_pooled:.4f}")
        print(f"  Pooled Distilled AUC:  {auc_dist_pooled:.4f}")
        print(f"  Pooled Improvement:    {auc_dist_pooled - auc_zero_pooled:+.4f}")
    else:
        auc_zero_pooled = auc_dist_pooled = float("nan")

    # Save report
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 鼠肝 cross-batch 自蒸馏验证\n\n")
        f.write(f"**日期**: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**数据**: 鼠肝 organoid 3 batch (b1+b3 真实 RF-DETR 检测)\n")
        f.write(f"**方法**: Leave-One-Batch-Out + 图像统计特征 + 轻量分类头\n\n")

        f.write(f"## 1. 数据\n\n")
        f.write(f"| Batch | Samples | TP | FP | RF-DETR conf mean |\n")
        f.write(f"|---|---|---|---|---|\n")
        for b, d in batches.items():
            n_tp = int(d["y"].sum())
            f.write(f"| {b} | {len(d['X'])} | {n_tp} | {len(d['X'])-n_tp} | {d['scores'].mean():.3f} |\n")

        f.write(f"\n## 2. Leave-One-Batch-Out 结果\n\n")
        f.write(f"| Test Batch | Val N | TP | FP | Zero-shot AUC | Classifier AUC | Distilled AUC | Improvement |\n")
        f.write(f"|---|---|---|---|---|---|---|---|\n")
        for b, r in results.items():
            f.write(f"| {b} | {r['n_val']} | {r['n_tp']} | {r['n_fp']} | {r['auc_zero_shot']:.4f} | {r['auc_classifier']:.4f} | {r['auc_distilled']:.4f} | {r['improvement']:+.4f} |\n")

        f.write(f"\n## 3. Pooled 结果\n\n")
        f.write(f"| 方法 | AUC |\n|---|---|\n")
        f.write(f"| Zero-shot (RF-DETR conf) | {auc_zero_pooled:.4f} |\n")
        f.write(f"| Distilled (conf × classifier) | {auc_dist_pooled:.4f} |\n")
        f.write(f"| **Improvement** | **{auc_dist_pooled - auc_zero_pooled:+.4f}** |\n\n")

        imp = auc_dist_pooled - auc_zero_pooled
        if imp > 0.10:
            f.write(f"✅ **自蒸馏有效**：AUC 提升 {imp*100:.1f}% > 10%。\n")
        elif imp > 0:
            f.write(f"⚠ **自蒸馏微弱有效**：AUC 提升 {imp*100:.1f}%。\n")
        else:
            f.write(f"❌ **自蒸馏无效**：AUC 下降 {-imp*100:.1f}%。\n")

        f.write(f"\n## 4. 说明\n\n")
        f.write(f"- 样本量小 (30 total, 5 FP)，统计功效有限\n")
        f.write(f"- RF-DETR conf 是真实的（鼠肝训练 checkpoint 检测鼠肝）\n")
        f.write(f"- b1 conf 高 (0.964)，b3 conf 低 (0.339)——跨 batch domain gap\n")
        f.write(f"- 特征：30 维图像统计（3 通道 mean/std + 8-bin hist）\n")
        f.write(f"- 这个实验验证 pipeline 在真实数据上可跑\n")

    print(f"\n  ✓ Saved: {report_path}")

    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "batches": {b: {"n": len(d["X"]), "tp": int(d["y"].sum())} for b, d in batches.items()},
            "lobo_results": results,
            "pooled": {
                "auc_zero_shot": float(auc_zero_pooled),
                "auc_distilled": float(auc_dist_pooled),
                "improvement": float(auc_dist_pooled - auc_zero_pooled),
            },
        }, f, indent=2)
    print(f"  ✓ Saved: {json_path}")
    print(f"\n✓ Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
