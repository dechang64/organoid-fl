r"""
FP 聚类分析 + DINOv2+DPMM 二级验证（单文件版）

用法（在冬生 Win11 + RTX 3060 上跑）：
    cd C:\Users\decha\organoid-fl
    python scripts\multiorg\fp_clustering_analysis.py ^
        --results-json D:\path\to\multiorg_sam2_results.json ^
        --data-root D:\datasets\mutliorg\MultiOrg_v2 ^
        --annotator t1_b ^
        --output-dir results\fp_analysis

依赖（脚本会自动检查并提示安装）：
    pip install timm tifffile scikit-learn matplotlib

输出：
    1. fp_clusters_vis.png       - FP 聚类可视化（每簇代表图片）
    2. tp_fp_tsne.png            - TP vs FP embedding t-SNE
    3. dpmm_pr_curve.png         - PR 曲线
    4. dpmm_loglik_dist.png      - Log-likelihood 分布
    5. dpmm_verification.json    - DPMM 验证指标
    6. fp_cluster_summary.json   - 簇统计
    7. tp_embeddings.npy / fp_embeddings.npy - 原始 embedding
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# ============================================================
# 依赖检查
# ============================================================

MISSING_DEPS = []
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
except ImportError:
    MISSING_DEPS.append("torch torchvision")

try:
    import timm
except ImportError:
    MISSING_DEPS.append("timm")

try:
    import tifffile
except ImportError:
    MISSING_DEPS.append("tifffile")

try:
    from PIL import Image
except ImportError:
    MISSING_DEPS.append("Pillow")

try:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except ImportError:
    MISSING_DEPS.append("scikit-learn")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    MISSING_DEPS.append("matplotlib")

if MISSING_DEPS:
    print("ERROR: Missing dependencies:")
    for dep in MISSING_DEPS:
        print(f"  - {dep}")
    print("\nInstall with:")
    print(f"  pip install {' '.join(MISSING_DEPS)}")
    sys.exit(1)


# ============================================================
# 1. GT 标注加载（模糊匹配，与 sahi_inference.py 一致）
# ============================================================

def find_annotation_file(img_dir, annotator='t1_b'):
    """找到图像对应的标注文件（模糊匹配，与原始 sahi_inference.py 一致）。
    
    annotator: 't0' | 't1_a' | 't1_b' | 'annotator_a' | 'annotator_b' | 'any'
    """
    target = annotator.lower()
    for f in os.listdir(img_dir):
        if not f.lower().endswith('.json'):
            continue
        if target in f.lower():
            return os.path.join(img_dir, f)
    
    # fallback: any json
    if annotator == 'any':
        for f in os.listdir(img_dir):
            if f.lower().endswith('.json'):
                return os.path.join(img_dir, f)
    return None


def load_gt_annotations(data_root, image_path, annotator):
    """加载 GT 标注（napari [row,col] 4点多边形）→ bbox list
    
    Args:
        data_root: MultiOrg_v2 根目录（自动检测 test/ 子目录）
        image_path: 如 "Macros/Plate_15/image_0"
        annotator: "t1_b" 等
    Returns:
        list of [x1,y1,x2,y2]
    """
    parts = image_path.split('/')
    cls_name = parts[0]   # Macros 或 Normal
    plate = parts[1]      # Plate_15
    image = parts[2]      # image_0
    
    # 自动检测 data_root 是否已包含 test/
    root = Path(data_root)
    if (root / "test" / cls_name).exists():
        img_dir = root / "test" / cls_name / plate / image
    elif (root / cls_name).exists():
        img_dir = root / cls_name / plate / image
    else:
        return []
    
    # 模糊匹配标注文件
    json_path = find_annotation_file(str(img_dir), annotator)
    if json_path is None:
        return []
    
    with open(json_path) as f:
        annot = json.load(f)
    
    bboxes = []
    for key, polygon in annot.items():
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        pts = np.array(polygon)
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue
        # napari 坐标系: [row, col] = [y, x]
        xs = pts[:, 1]
        ys = pts[:, 0]
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()), float(ys.max())
        bboxes.append([x1, y1, x2, y2])
    
    return bboxes


# ============================================================
# 2. TP/FP 匹配（与原始 compute_ap_full 一致，IoU=0.5）
# ============================================================

def iou(box1, box2):
    """计算两个 bbox 的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_tp_fp(detections, gt_bboxes, iou_threshold=0.5):
    """贪心匹配检测和 GT（按 confidence 降序），标记 TP/FP。
    
    与原始 compute_ap_full 逻辑一致：
    - 按 confidence 降序排列检测
    - 每个 GT 只能匹配一次
    - IoU >= threshold 才算 TP
    """
    det_sorted = sorted(enumerate(detections),
                        key=lambda x: -x[1]['confidence'])
    
    gt_matched = [False] * len(gt_bboxes)
    labels = [False] * len(detections)
    
    for orig_idx, det in det_sorted:
        det_bbox = det['bbox']
        best_iou = 0
        best_gt_idx = -1
        for gi, gt_bbox in enumerate(gt_bboxes):
            if gt_matched[gi]:
                continue
            cur_iou = iou(det_bbox, gt_bbox)
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_gt_idx = gi
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            labels[orig_idx] = True  # TP
    
    return labels


# ============================================================
# 3. 图像加载与 patch 裁剪
# ============================================================

def load_tiff_image(data_root, image_path):
    """加载 16-bit TIFF → 8-bit RGB（用 tifffile 避免 PIL segfault）"""
    parts = image_path.split('/')
    cls_name = parts[0]
    plate = parts[1]
    image = parts[2]
    
    root = Path(data_root)
    if (root / "test" / cls_name).exists():
        img_dir = root / "test" / cls_name / plate / image
    elif (root / cls_name).exists():
        img_dir = root / cls_name / plate / image
    else:
        raise FileNotFoundError(f"Cannot find {cls_name} under {data_root}")
    
    # 找 TIFF 文件（可能 .tiff 或 .tif）
    tiff_path = None
    for f in os.listdir(img_dir):
        if f.lower().endswith(('.tiff', '.tif')):
            tiff_path = img_dir / f
            break
    if tiff_path is None:
        raise FileNotFoundError(f"No TIFF in {img_dir}")
    
    arr = tifffile.imread(str(tiff_path))
    
    # 16-bit → 8-bit min-max 归一化
    if arr.dtype == np.uint16:
        vmin, vmax = arr.min(), arr.max()
        if vmax > vmin:
            arr = ((arr.astype(np.float64) - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    
    # 灰度 → RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    
    return arr


def crop_patch(img, bbox, pad=10):
    """从图片裁 patch，加 padding"""
    h, w = img.shape[:2]
    x1 = max(0, int(bbox[0]) - pad)
    y1 = max(0, int(bbox[1]) - pad)
    x2 = min(w, int(bbox[2]) + pad)
    y2 = min(h, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


# ============================================================
# 4. DINOv2 特征提取
# ============================================================

class DINOv2FeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        print(f"  Loading DINOv2 ViT-B/14 (downloading ~350MB weights on first run)...")
        self.model = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m',
            pretrained=True,
            num_classes=0,
        ).to(device).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print(f"  DINOv2 loaded on {device}, "
              f"{sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params")
    
    @torch.no_grad()
    def extract(self, patches, batch_size=32):
        """提取一批 patch 的 768维 CLS embedding"""
        embeddings = []
        n = len(patches)
        
        for i in range(0, n, batch_size):
            batch = patches[i:i+batch_size]
            tensors = []
            for p in batch:
                if p is None or p.size == 0:
                    p = np.zeros((64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(p)
                t = self.transform(img)
                tensors.append(t)
            
            batch_tensor = torch.stack(tensors).to(self.device)
            feats = self.model(batch_tensor)
            embeddings.append(feats.cpu().numpy())
            
            done = min(i + batch_size, n)
            if done % 320 < batch_size or done == n:
                print(f"    DINOv2: {done}/{n} patches")
        
        return np.concatenate(embeddings, axis=0)


# ============================================================
# 5. FP 聚类分析
# ============================================================

def cluster_fp_analysis(fp_embeddings, fp_patches, n_clusters=8, output_dir="."):
    """K-means 聚类 FP embedding，每簇展示代表 patch"""
    print(f"\n=== FP Clustering (K={n_clusters}) ===")
    
    scaler = StandardScaler()
    fp_emb_scaled = scaler.fit_transform(fp_embeddings)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(fp_emb_scaled)
    
    # 每簇统计
    cluster_summary = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_summary[f"cluster_{c}"] = {
            "count": int(mask.sum()),
            "percentage": float(mask.sum() / len(cluster_labels) * 100),
        }
    
    # 可视化：每簇 8 个代表（离簇中心最近）
    n_cols = min(8, max(1, len(fp_patches) // n_clusters))
    fig, axes = plt.subplots(n_clusters, n_cols, figsize=(3*n_cols, 3*n_clusters))
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    for c in range(n_clusters):
        mask = cluster_labels == c
        count = int(mask.sum())
        if count == 0:
            for j in range(n_cols):
                axes[c, j].axis('off')
            continue
        
        dists = np.linalg.norm(fp_emb_scaled[mask] - kmeans.cluster_centers_[c], axis=1)
        nearest = np.argsort(dists)[:n_cols]
        orig_indices = np.where(mask)[0][nearest]
        
        for j, idx in enumerate(orig_indices):
            ax = axes[c, j]
            patch = fp_patches[idx]
            if patch is not None and patch.size > 0:
                ax.imshow(patch, cmap='gray' if patch.ndim == 2 else None)
            ax.set_title(f"#{j}", fontsize=8)
            ax.axis('off')
        
        axes[c, 0].set_ylabel(
            f"C{c}\nn={count}\n{count/len(cluster_labels)*100:.1f}%",
            fontsize=10, rotation=0, labelpad=40
        )
    
    plt.suptitle("FP Clustering by DINOv2 Embedding (K-means)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "fp_clusters_vis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fp_clusters_vis.png")
    
    return cluster_labels, cluster_summary


# ============================================================
# 6. t-SNE 可视化
# ============================================================

def tsne_tp_fp(tp_emb, fp_emb, output_dir="."):
    """t-SNE 可视化 TP vs FP"""
    print(f"\n=== t-SNE Visualization (TP={len(tp_emb)}, FP={len(fp_emb)}) ===")
    
    # 采样（t-SNE O(n^2)，>2000 太慢）
    max_samples = 2000
    if len(tp_emb) > max_samples:
        tp_idx = np.random.choice(len(tp_emb), max_samples, replace=False)
        tp_emb_s = tp_emb[tp_idx]
    else:
        tp_emb_s = tp_emb
    
    if len(fp_emb) > max_samples:
        fp_idx = np.random.choice(len(fp_emb), max_samples, replace=False)
        fp_emb_s = fp_emb[fp_idx]
    else:
        fp_emb_s = fp_emb
    
    all_emb = np.concatenate([tp_emb_s, fp_emb_s], axis=0)
    labels = np.array([1]*len(tp_emb_s) + [0]*len(fp_emb_s))
    
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(all_emb)
    
    perplexity = min(30, len(all_emb) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    emb_2d = tsne.fit_transform(emb_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(emb_2d[labels==0, 0], emb_2d[labels==0, 1],
               c='red', alpha=0.3, s=5, label=f'FP (n={len(fp_emb_s)})')
    ax.scatter(emb_2d[labels==1, 0], emb_2d[labels==1, 1],
               c='blue', alpha=0.3, s=5, label=f'TP (n={len(tp_emb_s)})')
    ax.set_title("DINOv2 Embedding: TP vs FP (t-SNE)", fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "tp_fp_tsne.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: tp_fp_tsne.png")


# ============================================================
# 7. DPMM 二级验证（PCA 降维 + Bayesian GMM）
# ============================================================

def dpmm_verification(tp_emb, fp_emb, output_dir=".", pca_dim=50):
    """PCA 降维 → Bayesian GMM（DPMM 近似）→ 似然验证
    
    关键：768维 full covariance GMM 会过拟合且极慢，
    先 PCA 降到 50 维再拟合 GMM。
    """
    print(f"\n=== DPMM Verification ===")
    print(f"  TP samples: {len(tp_emb)}")
    print(f"  FP samples: {len(fp_emb)}")
    
    # PCA 降维
    all_emb = np.concatenate([tp_emb, fp_emb], axis=0)
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_emb)
    
    actual_pca_dim = min(pca_dim, all_scaled.shape[1], all_scaled.shape[0] - 1)
    print(f"  PCA: {all_scaled.shape[1]} → {actual_pca_dim} dims")
    pca = PCA(n_components=actual_pca_dim, random_state=42)
    all_pca = pca.fit_transform(all_scaled)
    
    tp_pca = all_pca[:len(tp_emb)]
    fp_pca = all_pca[len(tp_emb):]
    
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Bayesian GMM（近似 DPMM）
    n_max = min(15, max(2, len(tp_pca) // 100))
    print(f"  Fitting Bayesian GMM (max_components={n_max}, covariance=diag)...")
    
    dpgmm = BayesianGaussianMixture(
        n_components=n_max,
        covariance_type='diag',          # diag 避免 50维 full 协方差过拟合
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1e-2,
        max_iter=200,
        random_state=42,
    )
    dpgmm.fit(tp_pca)
    
    active = np.sum(dpgmm.weights_ > 0.01)
    print(f"  Active components: {active}/{n_max}")
    print(f"  Weights: {dpgmm.weights_.round(4)}")
    
    # 似然
    tp_loglik = dpgmm.score_samples(tp_pca)
    fp_loglik = dpgmm.score_samples(fp_pca)
    
    print(f"\n  TP log-likelihood: mean={tp_loglik.mean():.2f}, std={tp_loglik.std():.2f}")
    print(f"  FP log-likelihood: mean={fp_loglik.mean():.2f}, std={fp_loglik.std():.2f}")
    
    # PR 曲线
    all_scores = np.concatenate([tp_loglik, fp_loglik])
    all_labels = np.array([1]*len(tp_loglik) + [0]*len(fp_loglik))
    
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    pr_auc = auc(recall, precision)
    ap = average_precision_score(all_labels, all_scores)
    
    print(f"\n  PR-AUC: {pr_auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    
    # 最优 F1
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]
    best_p = precision[best_idx]
    best_r = recall[best_idx]
    
    print(f"\n  Best F1 threshold: {best_threshold:.2f}")
    print(f"  Best F1: {best_f1:.4f} (P={best_p:.4f}, R={best_r:.4f})")
    
    # 不同阈值
    threshold_results = []
    for pct in range(5, 100, 5):
        thr = float(np.percentile(all_scores, pct))
        pred_tp = all_scores >= thr
        tp_correct = int(np.sum(pred_tp & (all_labels == 1)))
        fp_rejected = int(np.sum(~pred_tp & (all_labels == 0)))
        fp_kept = int(np.sum(pred_tp & (all_labels == 0)))
        tp_rejected = int(np.sum(~pred_tp & (all_labels == 1)))
        
        p = tp_correct / max(tp_correct + fp_kept, 1)
        r = tp_correct / max(tp_correct + tp_rejected, 1)
        f1 = 2*p*r / max(p+r, 1e-10)
        
        threshold_results.append({
            "percentile": pct,
            "threshold": thr,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "tp_kept": tp_correct,
            "tp_rejected": tp_rejected,
            "fp_kept": fp_kept,
            "fp_rejected": fp_rejected,
            "fp_reduction_rate": float(fp_rejected / max(fp_rejected + fp_kept, 1)),
        })
        print(f"  p{pct}: thr={thr:.1f} P={p:.3f} R={r:.3f} F1={f1:.3f}  "
              f"FP reject {fp_rejected}/{fp_rejected+fp_kept} "
              f"({fp_rejected/max(fp_rejected+fp_kept,1)*100:.0f}%)")
    
    # PR 曲线图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'DPMM (AUC={pr_auc:.4f})')
    ax.axhline(y=0.286, color='r', linestyle='--', label='Baseline (no filter, P=0.286)')
    ax.scatter([best_r], [best_p], c='red', s=100, zorder=5,
               label=f'Best F1={best_f1:.3f}')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('DPMM Verification: Precision-Recall Curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "dpmm_pr_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: dpmm_pr_curve.png")
    
    # Log-likelihood 分布图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(tp_loglik, bins=50, alpha=0.6, color='blue', label=f'TP (n={len(tp_loglik)})', density=True)
    ax.hist(fp_loglik, bins=50, alpha=0.6, color='red', label=f'FP (n={len(fp_loglik)})', density=True)
    ax.axvline(x=best_threshold, color='green', linestyle='--', linewidth=2,
               label=f'Best F1 threshold={best_threshold:.2f}')
    ax.set_xlabel('Log-likelihood (DPMM)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('DPMM Log-likelihood Distribution: TP vs FP', fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "dpmm_loglik_dist.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dpmm_loglik_dist.png")
    
    return {
        "pr_auc": float(pr_auc),
        "average_precision": float(ap),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "best_precision": float(best_p),
        "best_recall": float(best_r),
        "active_components": int(active),
        "pca_dim": int(actual_pca_dim),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "n_tp": len(tp_emb),
        "n_fp": len(fp_emb),
        "tp_loglik_mean": float(tp_loglik.mean()),
        "fp_loglik_mean": float(fp_loglik.mean()),
        "threshold_results": threshold_results,
    }


# ============================================================
# 8. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="FP clustering + DINOv2 DPMM verification for MultiOrg"
    )
    parser.add_argument("--results-json", required=True,
                        help="Path to multiorg_sam2_results.json")
    parser.add_argument("--data-root", required=True,
                        help="MultiOrg_v2 root (auto-detects test/ subdir)")
    parser.add_argument("--annotator", default="t1_b",
                        help="GT annotator: t1_b, t1_a, t0, any (default: t1_b)")
    parser.add_argument("--output-dir", default="results/fp_analysis",
                        help="Output directory")
    parser.add_argument("--n-clusters", type=int, default=8,
                        help="Number of FP clusters for K-means")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda, cuda:0, cpu")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for TP/FP matching (default 0.5, "
                             "matches baseline mAP50)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images to process (for debugging)")
    parser.add_argument("--pca-dim", type=int, default=50,
                        help="PCA dimensions before DPMM (default 50)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- 1. 加载结果 JSON ----
    print(f"{'='*60}")
    print(f"=== Loading results: {args.results_json} ===")
    print(f"{'='*60}")
    with open(args.results_json) as f:
        results = json.load(f)
    
    config = results['config']
    baseline = results['baseline']
    print(f"  Model: {config['model']}/{config.get('model_variant','?')}")
    print(f"  SAM2:  {config.get('sam2_checkpoint', 'none')}")
    print(f"  Images: {config['n_images']}")
    print(f"  Baseline: mAP50={baseline['mAP50']:.4f}, "
          f"P={baseline['precision']:.4f}, R={baseline['recall']:.4f}")
    print(f"  Baseline TP={baseline['tp']}, FP={baseline['fp']}, FN={baseline['fn']}")
    
    per_images = results['per_image']
    if args.max_images:
        per_images = per_images[:args.max_images]
        print(f"  [DEBUG] Using first {len(per_images)} images")
    
    # ---- 2. TP/FP 匹配 + patch 裁剪 ----
    print(f"\n{'='*60}")
    print(f"=== Matching TP/FP (IoU={args.iou_threshold}, annotator={args.annotator}) ===")
    print(f"{'='*60}")
    
    all_tp_patches = []
    all_fp_patches = []
    all_tp_info = []
    all_fp_info = []
    total_tp = 0
    total_fp = 0
    
    for img_idx, img_data in enumerate(per_images):
        image_path = img_data['image']
        detections = img_data['detections']
        
        # GT
        gt_bboxes = load_gt_annotations(args.data_root, image_path, args.annotator)
        if not gt_bboxes:
            print(f"  [{img_idx+1}/{len(per_images)}] WARNING: No GT for {image_path}, skipping")
            continue
        
        # 匹配
        labels = match_tp_fp(detections, gt_bboxes, args.iou_threshold)
        n_tp = sum(labels)
        n_fp = len(labels) - n_tp
        total_tp += n_tp
        total_fp += n_fp
        
        # 加载图片
        try:
            img = load_tiff_image(args.data_root, image_path)
        except Exception as e:
            print(f"  [{img_idx+1}/{len(per_images)}] ERROR loading {image_path}: {e}")
            continue
        
        print(f"  [{img_idx+1}/{len(per_images)}] {image_path}: "
              f"det={len(detections)}, gt={len(gt_bboxes)}, TP={n_tp}, FP={n_fp}")
        
        # 裁 patch
        for i, (det, is_tp) in enumerate(zip(detections, labels)):
            patch = crop_patch(img, det['bbox'], pad=10)
            if patch is None or patch.size == 0:
                continue
            
            info = {
                "image": image_path,
                "bbox": det['bbox'],
                "confidence": det['confidence'],
                "circularity": det.get('circularity', 0),
                "area": det.get('area', 0),
            }
            
            if is_tp:
                all_tp_patches.append(patch)
                all_tp_info.append(info)
            else:
                all_fp_patches.append(patch)
                all_fp_info.append(info)
    
    print(f"\n  Total: TP={len(all_tp_patches)}, FP={len(all_fp_patches)}")
    print(f"  Baseline reference: TP={baseline['tp']}, FP={baseline['fp']}")
    
    # 校验 TP/FP 数量与 baseline 是否一致
    if abs(len(all_tp_patches) - baseline['tp']) > max(10, baseline['tp'] * 0.05):
        print(f"  WARNING: TP count mismatch (got {len(all_tp_patches)}, "
              f"baseline {baseline['tp']}). Check IoU threshold or annotator.")
    if abs(len(all_fp_patches) - baseline['fp']) > max(10, baseline['fp'] * 0.05):
        print(f"  WARNING: FP count mismatch (got {len(all_fp_patches)}, "
              f"baseline {baseline['fp']}). Check IoU threshold or annotator.")
    
    if len(all_tp_patches) == 0 or len(all_fp_patches) == 0:
        print("  ERROR: No TP or FP patches found!")
        sys.exit(1)
    
    # ---- 3. DINOv2 特征提取 ----
    print(f"\n{'='*60}")
    print(f"=== DINOv2 Feature Extraction ===")
    print(f"{'='*60}")
    
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("  CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        device = args.device
    print(f"  Device: {device}")
    
    extractor = DINOv2FeatureExtractor(device=device)
    
    print(f"\n  Extracting TP embeddings ({len(all_tp_patches)} patches)...")
    tp_embeddings = extractor.extract(all_tp_patches, batch_size=32)
    
    print(f"\n  Extracting FP embeddings ({len(all_fp_patches)} patches)...")
    fp_embeddings = extractor.extract(all_fp_patches, batch_size=32)
    
    print(f"\n  TP embeddings: {tp_embeddings.shape}")
    print(f"  FP embeddings: {fp_embeddings.shape}")
    
    np.save(output_dir / "tp_embeddings.npy", tp_embeddings)
    np.save(output_dir / "fp_embeddings.npy", fp_embeddings)
    with open(output_dir / "tp_info.json", 'w') as f:
        json.dump(all_tp_info, f)
    with open(output_dir / "fp_info.json", 'w') as f:
        json.dump(all_fp_info, f)
    print(f"  Saved embeddings and info to {output_dir}")
    
    # ---- 4. FP 聚类 ----
    print(f"\n{'='*60}")
    print(f"=== FP Clustering Analysis ===")
    print(f"{'='*60}")
    
    # FP 可视化用最多 2000 个 patch（避免内存爆）
    fp_vis_limit = 2000
    if len(all_fp_patches) > fp_vis_limit:
        vis_idx = np.random.choice(len(all_fp_patches), fp_vis_limit, replace=False)
        fp_vis_patches = [all_fp_patches[i] for i in vis_idx]
        fp_vis_emb = fp_embeddings[vis_idx]
        print(f"  Sampling {fp_vis_limit}/{len(all_fp_patches)} FPs for visualization")
    else:
        fp_vis_patches = all_fp_patches
        fp_vis_emb = fp_embeddings
    
    cluster_labels, cluster_summary = cluster_fp_analysis(
        fp_vis_emb, fp_vis_patches,
        n_clusters=args.n_clusters, output_dir=output_dir
    )
    
    with open(output_dir / "fp_cluster_summary.json", 'w') as f:
        json.dump(cluster_summary, f, indent=2)
    
    # ---- 5. t-SNE ----
    tsne_tp_fp(tp_embeddings, fp_embeddings, output_dir=output_dir)
    
    # ---- 6. DPMM 验证 ----
    print(f"\n{'='*60}")
    print(f"=== DPMM Verification ===")
    print(f"{'='*60}")
    
    dpmm_results = dpmm_verification(
        tp_embeddings, fp_embeddings,
        output_dir=output_dir, pca_dim=args.pca_dim
    )
    
    with open(output_dir / "dpmm_verification.json", 'w') as f:
        json.dump(dpmm_results, f, indent=2)
    
    # ---- 7. 总结 ----
    print(f"\n{'='*60}")
    print(f"=== SUMMARY ===")
    print(f"{'='*60}")
    print(f"  TP patches:  {len(all_tp_patches)}")
    print(f"  FP patches:  {len(all_fp_patches)}")
    print(f"  FP clusters: {args.n_clusters}")
    print(f"  DPMM PR-AUC: {dpmm_results['pr_auc']:.4f}")
    print(f"  DPMM Best F1: {dpmm_results['best_f1']:.4f} "
          f"(P={dpmm_results['best_precision']:.4f}, R={dpmm_results['best_recall']:.4f})")
    print(f"  DPMM Active components: {dpmm_results['active_components']}")
    print(f"  PCA: {dpmm_results['pca_dim']} dims, "
          f"explained var={dpmm_results['pca_explained_variance']:.3f}")
    print(f"\n  Output: {output_dir}")
    print(f"\n  Next steps:")
    print(f"    1. Check fp_clusters_vis.png — FP 分几类？每类长什么样？")
    print(f"    2. Check tp_fp_tsne.png — TP 和 FP 在 embedding 空间可分吗？")
    print(f"    3. Check dpmm_pr_curve.png — DPMM 验证 AUC 多少？")
    print(f"    4. PR-AUC > 0.85 → 部署 DPMM 为后置过滤器")
    print(f"    5. PR-AUC < 0.7  → 换特征（SAM2 mask, CLIP text-image）")


if __name__ == "__main__":
    main()
