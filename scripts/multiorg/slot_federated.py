"""
Phase 10: Federated Slot Aggregation (FORLA + FedCtx HNSW)

动机：FORLA (NeurIPS 2025) 证明联邦 slot attention 能学到跨域 object-centric 表征。
我们用"数据不动，知识动"理念——各 client 本地提取 slot embeddings，
上传到 FedCtx HNSW 做分布聚合，不共享原图/权重，只共享 slot 向量。

架构：
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │ Client 1 (P15N) │  │ Client 2 (P15M) │  │ Client 3 (P23M) │
    │                 │  │                 │  │                 │
    │ [SupCon model]  │  │ [SupCon model]  │  │ [SupCon model]  │
    │      ↓          │  │      ↓          │  │      ↓          │
    │ slot embeddings │  │ slot embeddings │  │ slot embeddings │
    │    [N1, 256]    │  │    [N2, 256]    │  │    [N3, 256]    │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  ↓
                    ┌─────────────────────────┐
                    │   FedCtx HNSW (Global)   │
                    │   Union of all slots     │
                    │   [N_total, 256]         │
                    └─────────────────────────┘
                                  ↓
                    New crop → extract slots
                    → k-NN in global HNSW
                    → majority vote TP/FP

评估：
  - Local k-NN per client (只有自己的数据)
  - Global k-NN (union of all clients' slots)
  - 看聚合是否提升各 client 的 TP/FP AUC

Usage:
    python slot_federated.py \
        --embeddings results/supcon_xxx/test_embeddings.npy \
        --labels results/supcon_xxx/test_labels.npy \
        --confs results/supcon_xxx/test_confs.npy \
        --metadata data/ctm_crops/ctm_metadata.json
"""
import os
import sys
import json
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict


def compute_pr_auc(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def knn_vote_score(train_embeddings, train_labels, test_embeddings, k=7):
    """
    k-NN majority vote: for each test sample, find k nearest train samples,
    return fraction of TP neighbors as score.

    Args:
        train_embeddings: [N_train, D]
        train_labels: [N_train] (0=FP, 1=TP)
        test_embeddings: [N_test, D]
        k: number of neighbors

    Returns:
        scores: [N_test] fraction of TP among k-NN
    """
    # Use cosine distance (embeddings are already L2-normalized, but just in case)
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    nn.fit(train_embeddings)
    distances, indices = nn.kneighbors(test_embeddings)

    # For each test sample, compute fraction of TP neighbors
    neighbor_labels = train_labels[indices]  # [N_test, k]
    scores = neighbor_labels.mean(axis=1)  # fraction of TP
    return scores


def split_clients(metadata_path, embeddings, labels, confs, seed=42,
                  train_ratio=0.7, val_ratio=0.15):
    """
    Split test set into clients by Plate×Class.

    Reuses ctm_dataset.py's split logic (seed=42, 70/15/15, by image)
    to ensure test set order matches the saved embeddings.

    Returns:
        clients: dict {client_name: {'embeddings': [N, D], 'labels': [N], 'confs': [N]}}
    """
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build all_dets same as ctm_dataset.py
    all_dets = []
    if isinstance(data, list):
        for entry in data:
            image_name = entry.get('image', '')
            det_idx = entry.get('det_idx', 0)
            cache_key = f"{image_name.replace('/', '_')}_{det_idx}"
            all_dets.append({
                'label': 1 if entry.get('matched', False) else 0,
                'confidence': entry.get('rfdetr_conf',
                                        entry.get('confidence', 0.5)),
                'image_name': image_name,
            })
    else:
        raise ValueError(f"Unexpected metadata format: {type(data)}")

    # Apply same split as ctm_dataset.py (seed=42, by image)
    unique_images = sorted(set(d['image_name'] for d in all_dets))
    n_images = len(unique_images)
    rng = np.random.RandomState(seed)
    img_perm = rng.permutation(n_images)
    n_train_img = int(n_images * train_ratio)
    n_val_img = int(n_images * val_ratio)

    test_images = set(unique_images[i] for i in img_perm[n_train_img + n_val_img:])
    test_indices = [i for i, d in enumerate(all_dets)
                    if d['image_name'] in test_images]

    print(f"Total crops: {len(all_dets)}")
    print(f"Test crops: {len(test_indices)}")
    print(f"Embeddings: {embeddings.shape[0]}")

    if len(test_indices) != embeddings.shape[0]:
        print(f"WARNING: test crops ({len(test_indices)}) != embeddings ({embeddings.shape[0]})")
        print("  Falling back to synthetic client split (7 clients, by position)")
        # Split into 7 synthetic clients by position (simulating Plate×Class)
        n = embeddings.shape[0]
        n_clients = 7
        sizes = [n // n_clients] * n_clients
        for i in range(n % n_clients):
            sizes[i] += 1

        clients = defaultdict(lambda: {'embeddings': [], 'labels': [], 'confs': []})
        start = 0
        for c in range(n_clients):
            end = start + sizes[c]
            client_name = f"Client_{c+1}"
            clients[client_name]['embeddings'] = embeddings[start:end]
            clients[client_name]['labels'] = labels[start:end]
            clients[client_name]['confs'] = confs[start:end]
            start = end
        return clients

    # Group by Plate×Class (parsed from image_name like 'Macros/Plate_23/image_4')
    clients = defaultdict(lambda: {'embeddings': [], 'labels': [], 'confs': []})
    for i, idx in enumerate(test_indices):
        image_name = all_dets[idx]['image_name']
        # Parse 'Macros/Plate_23/image_4' → class='Macros', plate='Plate_23'
        parts = image_name.split('/')
        if len(parts) >= 2:
            cls = parts[0]  # 'Macros' or 'Normal'
            plate = parts[1]  # 'Plate_23'
            client_name = f"{cls}_{plate}"
        else:
            client_name = 'unknown'

        clients[client_name]['embeddings'].append(embeddings[i])
        clients[client_name]['labels'].append(labels[i])
        clients[client_name]['confs'].append(confs[i])

    # Convert lists to arrays
    for name in clients:
        clients[name]['embeddings'] = np.array(clients[name]['embeddings'])
        clients[name]['labels'] = np.array(clients[name]['labels'])
        clients[name]['confs'] = np.array(clients[name]['confs'])

    return clients


def main():
    parser = argparse.ArgumentParser(description='Phase 10: Federated Slot Aggregation')
    parser.add_argument('--embeddings', required=True, help='test_embeddings.npy (from Phase 11)')
    parser.add_argument('--labels', required=True, help='test_labels.npy')
    parser.add_argument('--confs', required=True, help='test_confs.npy')
    parser.add_argument('--metadata', required=True, help='ctm_metadata.json for Plate×Class info')
    parser.add_argument('--output-dir', default='results/federated_slot')
    parser.add_argument('--k', type=int, default=7, help='k-NN neighbors')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    embeddings = np.load(args.embeddings)  # [N, 256]
    labels = np.load(args.labels)  # [N]
    confs = np.load(args.confs)  # [N]

    print(f"Embeddings: {embeddings.shape}")
    print(f"Labels: {labels.shape} (TP={labels.sum()}, FP={len(labels)-labels.sum()})")
    print(f"Confs: {confs.shape}")
    print()

    # Split into clients
    print("=" * 60)
    print("Splitting into clients (Plate×Class)")
    print("=" * 60)
    clients = split_clients(args.metadata, embeddings, labels, confs)

    for name, data in sorted(clients.items()):
        n = len(data['labels'])
        tp = data['labels'].sum()
        print(f"  {name:20s}: {n:5d} samples (TP={tp}, FP={n-tp})")
    print()

    # === 1. Global k-NN (union of all clients) ===
    print("=" * 60)
    print(f"Global k-NN (k={args.k}, all clients pooled)")
    print("=" * 60)

    # Use train split from metadata for k-NN train set
    # But we only have test embeddings... need to also load train embeddings
    # For now, do leave-one-client-out: train on all other clients, test on one

    all_emb = np.vstack([c['embeddings'] for c in clients.values()])
    all_labels = np.concatenate([c['labels'] for c in clients.values()])
    all_confs = np.concatenate([c['confs'] for c in clients.values()])

    # L2 normalize embeddings for cosine distance
    # (StandardScaler would break L2-normalized embeddings from Phase 11)
    norms = np.linalg.norm(all_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    all_emb_scaled = all_emb / norms

    # Global: leave-one-out cross-validation (each sample tested against all others)
    # This simulates "global HNSW" — query in the union of all slots
    print(f"\nGlobal LOO k-NN (each sample vs all others):")
    nn = NearestNeighbors(n_neighbors=args.k + 1, metric='cosine', n_jobs=-1)  # +1 to exclude self
    nn.fit(all_emb_scaled)
    distances, indices = nn.kneighbors(all_emb_scaled)

    # Exclude self (index 0 is always self)
    neighbor_labels = all_labels[indices[:, 1:]]  # [N, k]
    global_scores = neighbor_labels.mean(axis=1)  # fraction of TP

    global_roc = roc_auc_score(all_labels, global_scores)
    global_pr = compute_pr_auc(all_labels, global_scores)
    print(f"  Global k-NN:    ROC-AUC={global_roc:.4f}  PR-AUC={global_pr:.4f}")

    # RF-DETR baseline
    conf_roc = roc_auc_score(all_labels, all_confs)
    conf_pr = compute_pr_auc(all_labels, all_confs)
    print(f"  RF-DETR conf:  ROC-AUC={conf_roc:.4f}  PR-AUC={conf_pr:.4f}")
    print(f"  Δ vs RF-DETR:  ROC {global_roc-conf_roc:+.4f}  PR {global_pr-conf_pr:+.4f}")
    print()

    # === 2. Per-client: local k-NN vs global k-NN ===
    print("=" * 60)
    print(f"Per-client: Local k-NN (within client) vs Global k-NN (union)")
    print("=" * 60)
    print(f"{'Client':<22} {'N':>5} {'TP':>4} {'FP':>4}  "
          f"{'Local ROC':>10} {'Global ROC':>11}  "
          f"{'Local PR':>9} {'Global PR':>10}  {'Conf':>6}")
    print("-" * 85)

    offset = 0
    client_results = {}
    for name in sorted(clients.keys()):
        data = clients[name]
        n = len(data['labels'])
        tp = int(data['labels'].sum())
        fp = n - tp

        # Get global scores for this client's samples
        client_global_scores = global_scores[offset:offset + n]
        client_labels = data['labels']
        client_confs = data['confs']

        # Local k-NN: train on rest of this client's data, test on each sample
        # (leave-one-out within client)
        n_tp = int(data['labels'].sum())
        n_fp = n - n_tp
        if n > args.k + 1 and n_tp > 0 and n_fp > 0:
            # L2 normalize local embeddings (consistent with global)
            local_emb = data['embeddings'].copy()
            local_norms = np.linalg.norm(local_emb, axis=1, keepdims=True)
            local_norms[local_norms == 0] = 1
            local_emb = local_emb / local_norms

            local_nn = NearestNeighbors(n_neighbors=min(args.k + 1, n - 1),
                                        metric='cosine', n_jobs=-1)
            local_nn.fit(local_emb)
            _, local_indices = local_nn.kneighbors(local_emb)
            local_neighbor_labels = data['labels'][local_indices[:, 1:]]
            local_scores = local_neighbor_labels.mean(axis=1)

            local_roc = roc_auc_score(client_labels, local_scores)
            local_pr = compute_pr_auc(client_labels, local_scores)
        else:
            local_scores = np.full(n, 0.5)
            local_roc = 0.5
            local_pr = 0.5

        # Global k-NN for this client's samples
        if n_tp > 0 and n_fp > 0:
            global_roc_client = roc_auc_score(client_labels, client_global_scores)
            global_pr_client = compute_pr_auc(client_labels, client_global_scores)
            conf_roc_client = roc_auc_score(client_labels, client_confs)
            conf_pr_client = compute_pr_auc(client_labels, client_confs)
        else:
            global_roc_client = 0.5
            global_pr_client = 0.5
            conf_roc_client = 0.5
            conf_pr_client = 0.5

        client_results[name] = {
            'n': n, 'tp': tp, 'fp': fp,
            'local_roc': float(local_roc), 'global_roc': float(global_roc_client),
            'local_pr': float(local_pr), 'global_pr': float(global_pr_client),
            'conf_roc': float(conf_roc_client), 'conf_pr': float(conf_pr_client),
        }

        print(f"{name:<22} {n:5d} {tp:4d} {fp:4d}  "
              f"{local_roc:10.4f} {global_roc_client:11.4f}  "
              f"{local_pr:9.4f} {global_pr_client:10.4f}  {conf_roc_client:6.4f}")

        offset += n

    print()

    # === 3. Global k-NN vs local k-NN summary ===
    print("=" * 60)
    print("Summary: Local vs Global vs RF-DETR confidence")
    print("=" * 60)

    local_rocs = [r['local_roc'] for r in client_results.values()]
    global_rocs = [r['global_roc'] for r in client_results.values()]
    conf_rocs = [r['conf_roc'] for r in client_results.values()]

    print(f"  Local k-NN mean ROC:  {np.mean(local_rocs):.4f}")
    print(f"  Global k-NN mean ROC: {np.mean(global_rocs):.4f}  "
          f"(Δ {np.mean(global_rocs) - np.mean(local_rocs):+.4f})")
    print(f"  RF-DETR conf mean:    {np.mean(conf_rocs):.4f}")
    print(f"  Global k-NN (all):    ROC={global_roc:.4f}  PR={global_pr:.4f}")
    print(f"  RF-DETR (all):        ROC={conf_roc:.4f}  PR={conf_pr:.4f}")
    print(f"  Δ Global vs RF-DETR:  ROC {global_roc-conf_roc:+.4f}  "
          f"PR {global_pr-conf_pr:+.4f}")
    print()

    # === 4. Combined: Global k-NN score + RF-DETR confidence ===
    print("=" * 60)
    print("Combined: LR(global_knn_score, conf)")
    print("=" * 60)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    meta_features = np.column_stack([global_scores, all_confs])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_probs = np.zeros(len(all_labels))
    for train_idx, test_idx in skf.split(meta_features, all_labels):
        lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        lr.fit(meta_features[train_idx], all_labels[train_idx])
        meta_probs[test_idx] = lr.predict_proba(meta_features[test_idx])[:, 1]

    combined_roc = roc_auc_score(all_labels, meta_probs)
    combined_pr = compute_pr_auc(all_labels, meta_probs)
    print(f"  Combined (kNN+conf): ROC-AUC={combined_roc:.4f}  PR-AUC={combined_pr:.4f}")
    print(f"  vs RF-DETR alone:    ROC {combined_roc-conf_roc:+.4f}  "
          f"PR {combined_pr-conf_pr:+.4f}")
    print(f"  vs Phase 11 comb:   ROC {combined_roc-0.903:+.4f}  "
          f"(Phase 11 comb=0.903)")
    print()

    # Save results
    results = {
        'global_knn': {
            'roc_auc': float(global_roc), 'pr_auc': float(global_pr),
            'k': args.k,
        },
        'rf_detr': {
            'roc_auc': float(conf_roc), 'pr_auc': float(conf_pr),
        },
        'combined': {
            'roc_auc': float(combined_roc), 'pr_auc': float(combined_pr),
        },
        'per_client': client_results,
        'summary': {
            'local_mean_roc': float(np.mean(local_rocs)),
            'global_mean_roc': float(np.mean(global_rocs)),
            'conf_mean_roc': float(np.mean(conf_rocs)),
            'global_vs_local_delta': float(np.mean(global_rocs) - np.mean(local_rocs)),
            'global_vs_conf_delta': float(global_roc - conf_roc),
        },
    }
    with open(os.path.join(args.output_dir, 'federated_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_dir}/federated_results.json")


if __name__ == '__main__':
    main()
