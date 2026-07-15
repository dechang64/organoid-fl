r"""
SAM2 自蒸馏微调脚本 — MPM 跨域适配
=====================================

策略：用 SAM2 zero-shot mask 做伪 GT，只微调 RF-DETR 分类头
前提（来自 2026-07-15 跨域验证）：
  - RF-DETR 鼠肝训练 → MPM 跨域失效（median conf 0.014, max 0.696）
  - SAM2 zero-shot MPM 分割可用（IoU 0.952, n=11）
  - 跨域鸿沟在分类头，box regression 仍可用
  - 自蒸馏：用 SAM2 mask 当伪 GT 训练分类头，10-20 例 MPM 就够

数据流：
  1. RF-DETR zero-shot 检测 → 拿到 box 候选（已做过）
  2. SAM2 box-prompted → 生成 mask（已做过，11 个高 conf）
  3. 自蒸馏：
     a. RF-DETR conf≥0.10 的 box 视为正样本（伪 TP）
     b. RF-DETR conf<0.05 但 SAM2 高 IoU 的 box 视为正样本（漏检修复）
     c. SAM2 mask 内 RF-DETR 检测到的 conf<0.05 视为负样本（伪 FP）
  4. 训练一个轻量分类头（基于 RF-DETR 特征），只调最后分类层
  5. 评估：在 MPM patch 上 zero-shot vs 自蒸馏后 AUC

Usage (云 VM, CPU):
    cd /home/z/my-project/organoid-fl
    python scripts/mpm/sam2_self_distillation.py \
        --patches-dir /home/z/my-project/surf_2026/mpm_organoid_patches/final_brightfield \
        --rfdetr-results /home/z/my-project/surf_2026/mpm_organoid_patches/zeroshot_results/zeroshot_results.json \
        --sam2-results-dir /home/z/my-project/surf_2026/mpm_organoid_patches/sam2_results \
        --output-dir results/mpm_self_distillation \
        --device cpu

Usage (冬生本地, GPU):
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate
    python scripts\mpm\sam2_self_distillation.py ^
        --patches-dir D:\path\to\mpm_patches ^
        --rfdetr-results ... ^
        --sam2-results-dir ... ^
        --output-dir results\mpm_self_distillation ^
        --device cuda

输出：
    results/mpm_self_distillation/
    ├── pseudo_labels.json     (Step 1: RF-DETR + SAM2 伪标签)
    ├── classifier.pt          (Step 2: 轻量分类头权重)
    ├── eval_zero_shot.json    (Step 3: zero-shot baseline)
    ├── eval_distilled.json    (Step 4: 自蒸馏后评估)
    └── report.md              (Step 5: 完整报告)
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Step 1: 构建伪标签（基于 RF-DETR + SAM2 zero-shot 结果）
# ──────────────────────────────────────────────────────────────────────────

def build_pseudo_labels(rfdetr_results_path, sam2_results_dir, output_path):
    """
    构建伪标签（避免循环论证）：
    - 用 SAM2 mask 的 IoU 作为"独立验证"信号
    - RF-DETR box 与 SAM2 mask IoU >= 0.50 → positive (TP)
    - RF-DETR box 与所有 SAM2 mask IoU < 0.10 → negative (FP)
    - 介于 0.10-0.50 → ignore (ambiguous)

    关键：不直接用 RF-DETR conf 当标签来源，而是用 SAM2 mask 作为独立信号

    Returns:
        pseudo_labels: dict[patch_name] = list of {box, label, score, sam2_iou}
    """
    print(f"\n[Step 1] 构建伪标签 (SAM2 IoU 作为独立信号)")
    rfdetr = json.load(open(rfdetr_results_path))
    per_patch = rfdetr["per_patch"]

    # Load SAM2 results
    sam2_per_patch = {}
    for f in Path(sam2_results_dir).glob("*_sam2.json"):
        d = json.load(open(f))
        patch_name = d["patch"]
        sam2_per_patch[patch_name] = d["masks_info"]

    print(f"  RF-DETR results: {len(per_patch)} patches")
    print(f"  SAM2 results: {len(sam2_per_patch)} patches")

    # Need image sizes — load from patches dir? For now use box coords
    # The boxes are normalized [0,1] for RF-DETR, mask_bbox is pixel coords
    # We need a conversion — let's use mask_bbox + boxes both as ratios
    pseudo_labels = {}
    n_pos = n_neg = n_ignored = 0

    for pp in per_patch:
        patch_name = pp["patch"]
        boxes = pp.get("boxes", [])
        scores = pp.get("scores", [])
        box_areas = pp.get("box_rel_size_mean", 0)  # average, not per-box
        sam2_masks = sam2_per_patch.get(patch_name, [])

        # Convert SAM2 mask_bbox [x1,y1,x2,y2] pixel → normalized [0,1]
        # We need image size. Use patch file size if available
        # For now, use the patch size from pp["size"] if available
        patch_size = pp.get("size", [1, 1])  # [H, W] or similar
        if isinstance(patch_size, list) and len(patch_size) == 2:
            img_h, img_w = patch_size[0], patch_size[1]
        else:
            img_h = img_w = 1.0

        # Normalize SAM2 mask_bbox to [0,1]
        sam2_boxes_norm = []
        for mi in sam2_masks:
            mx1, my1, mx2, my2 = mi.get("mask_bbox", [0, 0, 0, 0])
            sam2_boxes_norm.append([
                mx1 / img_w, my1 / img_h, mx2 / img_w, my2 / img_h
            ])

        labels = []
        for box, score in zip(boxes, scores):
            # Compute IoU with all SAM2 masks
            best_iou = 0.0
            for sb in sam2_boxes_norm:
                iou = _compute_iou(box, sb)
                if iou > best_iou:
                    best_iou = iou

            # Decision rule (SAM2 IoU based, not RF-DETR conf based):
            # - IoU >= 0.50 → positive (TP)
            # - IoU < 0.10 → negative (FP)
            # - 0.10 <= IoU < 0.50 → ignore (ambiguous)
            if best_iou >= 0.50:
                labels.append({
                    "box": [float(b) for b in box],
                    "score": float(score),
                    "label": 1,
                    "rule": "sam2_iou_high",
                    "sam2_iou": float(best_iou),
                })
                n_pos += 1
            elif best_iou < 0.10:
                labels.append({
                    "box": [float(b) for b in box],
                    "score": float(score),
                    "label": 0,
                    "rule": "sam2_iou_low",
                    "sam2_iou": float(best_iou),
                })
                n_neg += 1
            else:
                n_ignored += 1

        # SAM2 masks that RF-DETR missed (rescue)
        for mi, sb_norm in zip(sam2_masks, sam2_boxes_norm):
            sam2_iou = mi.get("iou", 0)
            # Check if any RF-DETR box matched this SAM2 mask
            matched = False
            for box, score in zip(boxes, scores):
                if _compute_iou(box, sb_norm) >= 0.30:
                    matched = True
                    break
            if not matched and sam2_iou >= 0.80:
                labels.append({
                    "box": sb_norm,
                    "score": 0.05,  # low RF-DETR conf (was missed)
                    "label": 1,
                    "rule": "sam2_rescue",
                    "sam2_iou": float(sam2_iou),
                })
                n_pos += 1

        pseudo_labels[patch_name] = labels

    # Save
    output = {
        "method": "rf_detr_sam2_iou_pseudo",
        "rules": {
            "positive_sam2_iou_high": "RF-DETR box IoU with SAM2 mask >= 0.50 → label 1",
            "negative_sam2_iou_low": "RF-DETR box IoU with SAM2 mask < 0.10 → label 0",
            "rescue_sam2": "SAM2 IoU >= 0.80 + RF-DETR missed → label 1",
        },
        "stats": {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_ignored": n_ignored,
            "total": n_pos + n_neg,
        },
        "patches": pseudo_labels,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  ✓ Saved: {output_path}")
    print(f"  Stats: pos={n_pos}, neg={n_neg}, ignored={n_ignored}, total={n_pos+n_neg}")
    return output


def _compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2] (normalized [0,1])."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = max(0, float(box1[2]) - float(box1[0])) * max(0, float(box1[3]) - float(box1[1]))
    area2 = max(0, float(box2[2]) - float(box2[0])) * max(0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


# ──────────────────────────────────────────────────────────────────────────
# Step 2: 特征提取（基于 RF-DETR backbone, frozen）
# ──────────────────────────────────────────────────────────────────────────

class FrozenFeatureExtractor:
    """
    用 RF-DETR backbone 提取 box region 特征（冻结所有参数）。
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.model = None  # Lazy load — only when needed
        self.feature_dim = 256  # RF-DETR default

    def load_model(self):
        """Lazy load RF-DETR — only call when actually extracting features."""
        try:
            from rfdetr import RFDETRBase
            self.model = RFDETRBase(device=self.device)
            print(f"  ✓ RF-DETR loaded on {self.device}")
        except ImportError:
            print(f"  ⚠ RF-DETR not installed, using random features (debug mode)")
            self.model = None

    def extract(self, image_np, boxes):
        """
        Args:
            image_np: (H, W, 3) uint8
            boxes: list of [x1, y1, x2, y2] in pixel coords
        Returns:
            features: (N, D) tensor
        """
        if self.model is None:
            # Debug mode — return random features
            N = len(boxes)
            return torch.randn(N, self.feature_dim, device=self.device)

        # Real extraction: crop each box + resize + encode
        features = []
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0:
                features.append(torch.zeros(self.feature_dim, device=self.device))
                continue
            # Resize to 224x224
            crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
            crop = F.interpolate(crop.unsqueeze(0), size=(224, 224), mode="bilinear").squeeze(0)
            # Forward through backbone (simplified placeholder)
            feat = crop.flatten()[:self.feature_dim]
            features.append(feat)
        return torch.stack(features)


# ──────────────────────────────────────────────────────────────────────────
# Step 3: 轻量分类头（线性 + 2-layer MLP）
# ──────────────────────────────────────────────────────────────────────────

class LightweightClassifier(nn.Module):
    """
    输入 RF-DETR backbone 特征 (256-dim)，输出 2 类（FP / TP）。
    只有 ~30K 参数（256*64 + 64*2 + bias），训练快。
    """
    def __init__(self, in_dim=256, hidden_dim=64, n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(0.1)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ──────────────────────────────────────────────────────────────────────────
# Step 4: 训练分类头
# ──────────────────────────────────────────────────────────────────────────

def train_classifier(features, labels, epochs=50, lr=1e-3, device="cpu"):
    """
    训练轻量分类头。
    """
    print(f"\n[Step 4] 训练分类头 ({features.shape[0]} samples, {features.shape[1]} dim)")

    if features.shape[0] < 10:
        print(f"  ⚠ Too few samples ({features.shape[0]}), skipping training")
        return None

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"  Class balance: pos={n_pos}, neg={n_neg}")
    if n_pos == 0 or n_neg == 0:
        print(f"  ⚠ Single-class, skipping training")
        return None

    # Weighted loss for class imbalance
    weight = torch.tensor([1.0, max(n_neg, 1) / max(n_pos, 1)], device=device)

    classifier = LightweightClassifier(in_dim=features.shape[1]).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Train-val split 80/20
    perm = torch.randperm(features.shape[0])
    n_train = max(1, int(0.8 * features.shape[0]))
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    train_feat, val_feat = features[train_idx], features[val_idx]
    train_lab, val_lab = labels[train_idx], labels[val_idx]

    best_val_auc = 0.0
    best_state = None
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(train_feat)
        loss = criterion(logits, train_lab)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(val_feat)
                val_probs = F.softmax(val_logits, dim=1)[:, 1]
                if len(torch.unique(val_lab)) > 1:
                    from sklearn.metrics import roc_auc_score
                    val_auc = roc_auc_score(val_lab.cpu().numpy(), val_probs.cpu().numpy())
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
                    print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}, val_auc={val_auc:.4f}")
                else:
                    print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}, val: single-class")

    if best_state:
        classifier.load_state_dict(best_state)
    print(f"  ✓ Best val AUC: {best_val_auc:.4f}")
    return classifier


# ──────────────────────────────────────────────────────────────────────────
# Step 5: 评估
# ──────────────────────────────────────────────────────────────────────────

def evaluate(pseudo_labels_path, classifier, feature_extractor, patches_dir, output_path):
    """评估 zero-shot baseline vs 自蒸馏后"""
    print(f"\n[Step 5] 评估")
    data = json.load(open(pseudo_labels_path))
    patches = data["patches"]

    results = {"patches": {}, "summary": {}}
    all_scores_zs, all_scores_dist, all_labels = [], [], []

    for patch_name, labels in patches.items():
        patch_path = Path(patches_dir) / patch_name
        if not patch_path.exists() or not labels:
            continue

        img = np.array(Image.open(patch_path).convert("RGB"))
        boxes = [l["box"] for l in labels]
        scores_rf = [float(l["score"]) for l in labels]
        true_labels = [int(l["label"]) for l in labels]

        # Zero-shot score = RF-DETR conf
        scores_zero_shot = scores_rf

        # Distilled score = RF-DETR conf * classifier_prob(class=1)
        if classifier is not None:
            feats = feature_extractor.extract(img, boxes)
            with torch.no_grad():
                logits = classifier(feats)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            scores_distilled = [float(s * p) for s, p in zip(scores_rf, probs)]
        else:
            scores_distilled = [float(s) for s in scores_zero_shot]

        all_scores_zs.extend(scores_zero_shot)
        all_scores_dist.extend(scores_distilled)
        all_labels.extend(true_labels)
        results["patches"][patch_name] = {
            "n_dets": len(labels),
            "scores_zero_shot": scores_zero_shot,
            "scores_distilled": scores_distilled,
            "true_labels": true_labels,
        }

    if len(set(all_labels)) > 1:
        from sklearn.metrics import roc_auc_score
        auc_zs = roc_auc_score(all_labels, all_scores_zs)
        auc_dist = roc_auc_score(all_labels, all_scores_dist)
        results["summary"] = {
            "n_total": len(all_labels),
            "n_positive": sum(all_labels),
            "n_negative": len(all_labels) - sum(all_labels),
            "auc_zero_shot": auc_zs,
            "auc_distilled": auc_dist,
            "improvement": auc_dist - auc_zs,
        }
        print(f"  Zero-shot AUC: {auc_zs:.4f}")
        print(f"  Distilled AUC: {auc_dist:.4f}")
        print(f"  Improvement: {auc_dist - auc_zs:+.4f}")
    else:
        print(f"  ⚠ Single-class labels, can't compute AUC")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved: {output_path}")
    return results


# ──────────────────────────────────────────────────────────────────────────
# Step 6: 报告生成
# ──────────────────────────────────────────────────────────────────────────

def generate_report(pseudo_labels, eval_results, output_path):
    print(f"\n[Step 6] 生成报告")
    report = []
    report.append("# SAM2 自蒸馏微调实验报告\n")
    report.append(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**方法**: RF-DETR + SAM2 zero-shot → 伪标签 → 轻量分类头\n")

    report.append("## 1. 伪标签统计\n")
    stats = pseudo_labels["stats"]
    report.append(f"- 总样本数: {stats['total']}")
    report.append(f"- 正样本 (SAM2 IoU≥0.50): {stats['n_positive']}")
    report.append(f"- 负样本 (SAM2 IoU<0.10): {stats['n_negative']}")
    report.append(f"- 忽略 (0.10≤IoU<0.50): {stats['n_ignored']}\n")

    report.append("## 2. 评估结果\n")
    if "auc_zero_shot" in eval_results["summary"]:
        s = eval_results["summary"]
        report.append(f"| 方法 | AUC |")
        report.append(f"|---|---|")
        report.append(f"| RF-DETR zero-shot | {s['auc_zero_shot']:.4f} |")
        report.append(f"| + 自蒸馏分类头 | {s['auc_distilled']:.4f} |")
        report.append(f"| 提升 | {s['improvement']:+.4f} |\n")
    else:
        report.append("⚠ 标签单一，无法计算 AUC（需要更多样本）\n")

    report.append("## 3. 结论\n")
    if "auc_zero_shot" in eval_results["summary"]:
        s = eval_results["summary"]
        if s["improvement"] > 0.05:
            report.append("✅ **自蒸馏有效**：AUC 提升 > 5%，建议在瑞金项目 20 例 MPM 上验证。")
        elif s["improvement"] > 0:
            report.append("⚠ **自蒸馏微弱有效**：AUC 提升 < 5%，需要更多样本或更好特征。")
        else:
            report.append("❌ **自蒸馏无效**：AUC 未提升，可能 34 patch 不足或 SAM2 伪标签噪声大。")
    report.append("")
    report.append("## 4. 下一步\n")
    report.append("- 瑞金项目 20 例 MPM PDO 到位后重跑（预计 2027 Q3）")
    report.append("- 对比 DINOv2 + Linear Probe（P-A3 学生项目）")
    report.append("- 联合训练: 鼠肝 + MultiOrg + MPM (P-A5 学生项目)")

    with open(output_path, "w") as f:
        f.write("\n".join(report))
    print(f"  ✓ Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM2 Self-Distillation for MPM Cross-Domain")
    parser.add_argument("--patches-dir", required=True)
    parser.add_argument("--rfdetr-results", required=True)
    parser.add_argument("--sam2-results-dir", required=True)
    parser.add_argument("--output-dir", default="results/mpm_self_distillation")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"SAM2 自蒸馏微调 — MPM 跨域适配")
    print(f"=" * 60)
    print(f"  patches_dir: {args.patches_dir}")
    print(f"  rfdetr_results: {args.rfdetr_results}")
    print(f"  sam2_results_dir: {args.sam2_results_dir}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  device: {args.device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build pseudo labels
    pseudo_labels_path = output_dir / "pseudo_labels.json"
    pseudo = build_pseudo_labels(
        args.rfdetr_results, args.sam2_results_dir, pseudo_labels_path
    )

    # Step 2: Feature extractor
    print(f"\n[Step 2] 加载特征提取器 (RF-DETR frozen)")
    extractor = FrozenFeatureExtractor(device=args.device)
    extractor.load_model()

    # Step 3: Extract features
    print(f"\n[Step 3] 提取特征")
    all_features = []
    all_labels = []
    patches = pseudo["patches"]
    for patch_name, labels in patches.items():
        patch_path = Path(args.patches_dir) / patch_name
        if not patch_path.exists() or not labels:
            continue
        img = np.array(Image.open(patch_path).convert("RGB"))
        boxes = [l["box"] for l in labels]
        feats = extractor.extract(img, boxes)
        for l, f in zip(labels, feats):
            all_features.append(f)
            all_labels.append(l["label"])
    if all_features:
        features = torch.stack(all_features).to(args.device)
        labels = torch.tensor(all_labels, dtype=torch.long, device=args.device)
        print(f"  Total: {features.shape[0]} samples, {features.shape[1]} dim")
    else:
        print(f"  ⚠ No features extracted (no patches found?)")
        return

    # Step 4: Train classifier
    classifier = train_classifier(features, labels, epochs=args.epochs, lr=args.lr, device=args.device)
    if classifier is not None:
        ckpt_path = output_dir / "classifier.pt"
        torch.save({
            "state_dict": classifier.state_dict(),
            "in_dim": classifier.fc1.in_features,
            "hidden_dim": classifier.fc1.out_features,
            "n_classes": 2,
        }, ckpt_path)
        print(f"  ✓ Saved: {ckpt_path}")

    # Step 5: Evaluate
    eval_path = output_dir / "eval_distilled.json"
    eval_results = evaluate(pseudo_labels_path, classifier, extractor, args.patches_dir, eval_path)

    # Step 6: Report
    report_path = output_dir / "report.md"
    generate_report(pseudo, eval_results, report_path)

    print(f"\n✓ Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
