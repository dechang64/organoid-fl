"""
Phase 5: FL Primitive Distribution Aggregation
=============================================
各 Plate 作为 FL client，提取 primitive 向量，聚合全局分布。
评估：全局分布能否提升 TP/FP 判别？跨域检索是否有效？

不共享原图，不共享模型权重——只共享 primitive 向量（768维→6维这里）。
"""
import json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os

# === Load data ===
SAM2_PATH = '/home/z/my-project/organoid-fl/results/multiorg_sam2_zeroshot/multiorg_sam2_results.json'
OUT_DIR = '/home/z/my-project/organoid-fl/results/phase5_fl_primitive'
os.makedirs(OUT_DIR, exist_ok=True)

with open(SAM2_PATH, encoding="utf-8") as f:
    sam2_data = json.load(f)

per_img = sam2_data['per_image']

# === Extract primitives per client (Plate) ===
FEATURES = ['area', 'perimeter', 'circularity', 'solidity', 'aspect_ratio', 'confidence']

clients = {}  # client_name -> {'primitives': X, 'labels': y, 'images': [...]}

for img_data in per_img:
    name = img_data['image']
    parts = name.split('/')
    cls = parts[0]
    plate = parts[1] if len(parts) > 1 else 'unknown'
    client_name = f'{cls}_{plate}'  # e.g. 'Macros_Plate_15'

    if client_name not in clients:
        clients[client_name] = {'primitives': [], 'labels': [], 'images': []}

    for det in img_data['detections']:
        vec = [det.get(f, 0) for f in FEATURES]
        clients[client_name]['primitives'].append(vec)
        clients[client_name]['labels'].append(1 if det.get('matched', False) else 0)
        clients[client_name]['images'].append(name)

# Convert to numpy
for name, data in clients.items():
    data['primitives'] = np.array(data['primitives'], dtype=np.float64)
    data['labels'] = np.array(data['labels'])
    # Handle NaN/Inf
    data['primitives'] = np.nan_to_num(data['primitives'], nan=0.0, posinf=1e6, neginf=-1e6)

print("=" * 70)
print("Phase 5: FL Primitive Distribution Aggregation")
print("=" * 70)
print(f"\nFeatures: {FEATURES}")
print(f"\nClient (Plate) distribution:")
print(f"{'Client':<25} {'N':>7} {'TP':>7} {'FP':>7} {'TP%':>7}")
print("-" * 55)
for name in sorted(clients.keys()):
    d = clients[name]
    n = len(d['labels'])
    tp = sum(d['labels'] == 1)
    fp = sum(d['labels'] == 0)
    print(f"{name:<25} {n:>7} {tp:>7} {fp:>7} {100*tp/n:>6.1f}%")

# === 1. Per-client primitive statistics ===
print(f"\n{'=' * 70}")
print("1. Per-Client Primitive Statistics (TP vs FP)")
print("=" * 70)

print(f"\n{'Client':<25} {'Feature':<15} {'TP med':>8} {'FP med':>8} {'p-val':>10} {'Sig':>5}")
print("-" * 75)

for client_name in sorted(clients.keys()):
    d = clients[client_name]
    tp_mask = d['labels'] == 1
    fp_mask = d['labels'] == 0
    tp_data = d['primitives'][tp_mask]
    fp_data = d['primitives'][fp_mask]

    if len(tp_data) < 3 or len(fp_data) < 3:
        continue

    for i, feat in enumerate(FEATURES):
        tp_vals = tp_data[:, i]
        fp_vals = fp_data[:, i]
        u_stat, p_val = stats.mannwhitneyu(tp_vals, fp_vals, alternative='greater')
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        print(f"{client_name:<25} {feat:<15} {np.median(tp_vals):>8.3f} {np.median(fp_vals):>8.3f} {p_val:>10.6f} {sig:>5}")

# === 2. Cross-domain primitive distribution shift ===
print(f"\n{'=' * 70}")
print("2. Cross-Domain Primitive Distribution Shift (KL Divergence)")
print("=" * 70)

# Normalize primitives per feature
all_primitives = np.vstack([d['primitives'] for d in clients.values()])
scaler = StandardScaler()
scaler.fit(all_primitives)

for name in clients:
    clients[name]['primitives_scaled'] = scaler.transform(clients[name]['primitives'])

# KL divergence between clients (on confidence feature, most informative)
print(f"\nKL Divergence (confidence feature, 20 bins):")
client_names = sorted(clients.keys())
print(f"{'':>25}", end='')
for cn in client_names:
    print(f"{cn.split('_')[-1]:>12}", end='')
print()

kl_matrix = np.zeros((len(client_names), len(client_names)))
for i, ci in enumerate(client_names):
    print(f"{ci.split('_')[-1]:>25}", end='')
    di = clients[ci]['primitives_scaled'][:, 5]  # confidence is index 5
    # Normalize to probability distribution
    hist_i, _ = np.histogram(di, bins=20, range=(-3, 3), density=True)
    hist_i = hist_i / hist_i.sum() + 1e-10
    for j, cj in enumerate(client_names):
        if i == j:
            print(f"{'0.000':>12}", end='')
            continue
        dj = clients[cj]['primitives_scaled'][:, 5]
        hist_j, _ = np.histogram(dj, bins=20, range=(-3, 3), density=True)
        hist_j = hist_j / hist_j.sum() + 1e-10
        kl = np.sum(hist_i * np.log(hist_i / hist_j))
        kl_matrix[i, j] = kl
        print(f"{kl:>12.3f}", end='')
    print()

# === 3. Local vs Global ROC-AUC (TP/FP discrimination) ===
print(f"\n{'=' * 70}")
print("3. Local vs Global: Can Aggregation Improve TP/FP Discrimination?")
print("=" * 70)

def compute_auc(X, y):
    """Compute ROC-AUC for each feature and combined."""
    results = {}
    for i, feat in enumerate(FEATURES):
        try:
            auc = roc_auc_score(y, X[:, i])
            results[feat] = auc
        except:
            results[feat] = 0.5
    return results

def best_f1_score(y, scores):
    best_f1, best_t = 0, 0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (scores >= t).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_f1, best_t

print(f"\n{'Client':<25} {'Scope':<10} {'conf AUC':>10} {'area AUC':>10} {'circ AUC':>10} {'Best F1':>10}")
print("-" * 75)

# Per-client local AUC
for client_name in sorted(clients.keys()):
    d = clients[client_name]
    if sum(d['labels'] == 0) < 3:
        continue
    aucs = compute_auc(d['primitives'], d['labels'])
    f1, t = best_f1_score(d['labels'], d['primitives'][:, 5])  # confidence
    print(f"{client_name:<25} {'local':>10} {aucs['confidence']:>10.4f} {aucs['area']:>10.4f} {aucs['circularity']:>10.4f} {f1:>10.4f}")

# Global AUC (all clients pooled)
all_X = np.vstack([d['primitives'] for d in clients.values()])
all_y = np.concatenate([d['labels'] for d in clients.values()])
global_aucs = compute_auc(all_X, all_y)
f1_global, t_global = best_f1_score(all_y, all_X[:, 5])
print(f"{'GLOBAL (pooled)':<25} {'global':>10} {global_aucs['confidence']:>10.4f} {global_aucs['area']:>10.4f} {global_aucs['circularity']:>10.4f} {f1_global:>10.4f}")

# === 4. k-NN Outlier Detection: Can global distribution flag local FPs? ===
print(f"\n{'=' * 70}")
print("4. k-NN Outlier Detection: Can Global Distribution Flag Local FPs?")
print("=" * 70)

# For each client, build k-NN index from ALL OTHER clients' TP primitives
# Then check: are this client's FPs farther from the global TP centroid?

k = 5  # k-NN
results_knn = {}

for client_name in sorted(clients.keys()):
    d = clients[client_name]
    if sum(d['labels'] == 0) < 3:
        continue

    # Build global TP index from OTHER clients
    other_tp = []
    for other_name in clients:
        if other_name == client_name:
            continue
        other_d = clients[other_name]
        other_tp.append(other_d['primitives_scaled'][other_d['labels'] == 1])

    if not other_tp or sum(len(x) for x in other_tp) < k:
        continue

    other_tp_all = np.vstack(other_tp)

    # k-NN distances for this client's detections
    nbrs = NearestNeighbors(n_neighbors=min(k, len(other_tp_all)), metric='euclidean')
    nbrs.fit(other_tp_all)

    local_scaled = d['primitives_scaled']
    distances, _ = nbrs.kneighbors(local_scaled)
    avg_dist = distances.mean(axis=1)  # Average distance to k nearest TPs

    # Can distance distinguish TP from FP?
    local_labels = d['labels']
    tp_dist = avg_dist[local_labels == 1]
    fp_dist = avg_dist[local_labels == 0]

    try:
        auc_dist = roc_auc_score(local_labels, -avg_dist)  # closer = more likely TP
    except:
        auc_dist = 0.5

    u_stat, p_val = stats.mannwhitneyu(tp_dist, fp_dist, alternative='less')

    results_knn[client_name] = {
        'auc': auc_dist,
        'tp_dist_med': float(np.median(tp_dist)),
        'fp_dist_med': float(np.median(fp_dist)),
        'p_val': float(p_val)
    }

    print(f"\n{client_name}:")
    print(f"  k-NN distance AUC (closer=TP): {auc_dist:.4f}")
    print(f"  TP median dist: {np.median(tp_dist):.4f}, FP median dist: {np.median(fp_dist):.4f}")
    print(f"  Mann-Whitney p: {p_val:.6f} {'***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))}")

    # Combined: confidence + k-NN distance
    combined = 0.5 * d['primitives'][:, 5] + 0.5 * (-avg_dist / (avg_dist.max() + 1e-10))
    try:
        combined_auc = roc_auc_score(local_labels, combined)
    except:
        combined_auc = 0.5
    conf_auc = roc_auc_score(local_labels, d['primitives'][:, 5])
    print(f"  Confidence AUC: {conf_auc:.4f}, Combined AUC: {combined_auc:.4f} (Δ={combined_auc - conf_auc:+.4f})")

# === 5. Cross-domain primitive transfer: "normal range" from global ===
print(f"\n{'=' * 70}")
print("5. Cross-Domain 'Normal Range': Global TP Statistics as Reference")
print("=" * 70)

# Build global TP distribution
global_tp_primitives = np.vstack([d['primitives'][d['labels'] == 1] for d in clients.values()])
print(f"\nGlobal TP count: {len(global_tp_primitives)}")

# Per-feature global statistics
for i, feat in enumerate(FEATURES):
    vals = global_tp_primitives[:, i]
    lo, hi = np.percentile(vals, [5, 95])
    med = np.median(vals)
    print(f"  {feat:<15}: median={med:.3f}, [5%-95%]=[{lo:.3f}, {hi:.3f}]")

# Apply global "normal range" to each client's FPs
print(f"\nFP rate outside global normal range (per feature):")
for client_name in sorted(clients.keys()):
    d = clients[client_name]
    fp_data = d['primitives'][d['labels'] == 0]
    if len(fp_data) < 3:
        continue
    print(f"\n  {client_name} ({len(fp_data)} FPs):")
    for i, feat in enumerate(FEATURES):
        lo, hi = np.percentile(global_tp_primitives[:, i], [5, 95])
        outside = (fp_data[:, i] < lo) | (fp_data[:, i] > hi)
        print(f"    {feat:<15}: {sum(outside)}/{len(fp_data)} ({100*sum(outside)/len(fp_data):.1f}%) outside normal range")

# === 6. Summary ===
print(f"\n{'=' * 70}")
print("Summary & Recommendations")
print("=" * 70)

# Average k-NN AUC across clients
knn_aucs = [v['auc'] for v in results_knn.values()]
print(f"\n1. k-NN outlier detection AUC: mean={np.mean(knn_aucs):.3f}, range=[{min(knn_aucs):.3f}, {max(knn_aucs):.3f}]")
print(f"2. Global confidence AUC: {global_aucs['confidence']:.3f}")
print(f"3. Cross-domain KL divergence range: [{kl_matrix[kl_matrix > 0].min():.3f}, {kl_matrix[kl_matrix > 0].max():.3f}]")

# Save results
summary = {
    'phase': 'Phase 5: FL Primitive Aggregation',
    'date': '2026-07-10',
    'features': FEATURES,
    'clients': {name: {'n': len(d['labels']), 'tp': int(sum(d['labels'] == 1)), 'fp': int(sum(d['labels'] == 0))}
               for name, d in clients.items()},
    'global_aucs': global_aucs,
    'global_f1': float(f1_global),
    'knn_results': {k: {'auc': v['auc'], 'tp_dist_med': v['tp_dist_med'], 'fp_dist_med': v['fp_dist_med'], 'p_val': v['p_val']}
                      for k, v in results_knn.items()},
    'knn_auc_mean': float(np.mean(knn_aucs)),
    'kl_divergence_matrix': kl_matrix.tolist(),
    'client_names': client_names,
}

with open(f'{OUT_DIR}/phase5_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {OUT_DIR}/phase5_summary.json")
print("Done.")
