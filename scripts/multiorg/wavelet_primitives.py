"""
Phase 8: Wavelet Frequency-Domain Primitive Analysis (WIPES-inspired)

Tests whether TP/FP are separable in wavelet frequency domain.
Previous results: spatial features (circ/solidity) PR-AUC < 0.50,
DINOv2 CLS embedding PR-AUC = 0.29. This tests a new feature space.

Experiments:
    W1: Haar 2-level decomposition
    W2: Daubechies-4 3-level decomposition
    W3: Best wavelet + morphological features concatenated
    W4: Cross-domain validation on mouse liver

Usage:
    # MultiOrg (uses ctm_metadata.json + crops from CTM pipeline)
    python wavelet_primitives.py --metadata data/ctm_crops/ctm_metadata.json --crops-dir data/ctm_crops

    # Mouse liver (uses vlm_mask_results.json + crops)
    python wavelet_primitives.py --metadata results/phase2_vlm_100_mask/vlm_mask_results.json --crops-dir results/phase2_vlm_100_mask/crops
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import pywt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract_wavelet_features(image_array, wavelet='haar', levels=2):
    """
    Extract wavelet sub-band statistics from a grayscale image.
    
    Args:
        image_array: [H, W] or [H, W, 3] numpy array (uint8)
        wavelet: wavelet name ('haar', 'db4', etc.)
        levels: decomposition levels
    
    Returns:
        feature vector of sub-band statistics
    """
    # Convert to grayscale
    if image_array.ndim == 3:
        gray = np.mean(image_array, axis=-1)
    else:
        gray = image_array
    gray = gray.astype(np.float64)
    
    # Multi-level 2D discrete wavelet transform
    coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=levels)
    
    features = []
    feature_names = []
    
    # coeffs[0] = LL (approximation), coeffs[1:] = (LH, HL, HH) per level
    for i, c in enumerate(coeffs):
        if i == 0:
            # LL sub-band (lowest frequency)
            name = f'LL'
            subbands = [c]
        else:
            # Detail sub-bands (LH=horizontal, HL=vertical, HH=diagonal)
            name = f'L{i}'
            subbands = c  # tuple of (LH, HL, HH)
        
        for j, sb in enumerate(subbands):
            if i == 0:
                sb_name = name
            else:
                sb_names = ['LH', 'HL', 'HH']
                sb_name = f'{name}_{sb_names[j]}'
            
            # Sub-band statistics
            sb_flat = sb.flatten()
            features.extend([
                np.mean(sb_flat),
                np.std(sb_flat),
                np.mean(sb_flat ** 2),  # energy
                -np.sum(sb_flat * np.log(np.abs(sb_flat) + 1e-10)) / len(sb_flat),  # entropy
            ])
            feature_names.extend([
                f'{sb_name}_mean', f'{sb_name}_std', f'{sb_name}_energy', f'{sb_name}_entropy'
            ])
    
    return np.array(features), feature_names


def load_metadata_and_crops(metadata_path, crops_dir, max_samples=None):
    """Load metadata JSON and return list of (crop_path, label, confidence)."""
    with open(metadata_path, encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    if isinstance(data, list):
        for entry in data:
            cache_key = entry.get('cache_key', '')
            crop_path = os.path.join(crops_dir, f'{cache_key}.png')
            if os.path.exists(crop_path):
                label = 1 if entry.get('matched', False) else 0
                conf = entry.get('rfdetr_conf', entry.get('confidence', 0.5))
                samples.append((crop_path, label, conf))
    elif isinstance(data, dict) and 'per_image' in data:
        for img_info in data['per_image']:
            image_name = img_info['image']
            for det_idx, det in enumerate(img_info['detections']):
                cache_key = f"{image_name.replace('/', '_')}_{det_idx}"
                crop_path = os.path.join(crops_dir, f'{cache_key}.png')
                if os.path.exists(crop_path):
                    label = 1 if det.get('matched', False) else 0
                    conf = det.get('confidence', 0.5)
                    samples.append((crop_path, label, conf))
    
    if max_samples and len(samples) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in indices]
    
    return samples


def compute_pr_auc(labels, scores):
    """Compute PR-AUC (average precision)."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def run_experiment(samples, wavelet, levels, exp_name, output_dir):
    """Run one wavelet experiment."""
    print(f"\n{'='*60}")
    print(f"  {exp_name}: {wavelet}, {levels} levels")
    print(f"{'='*60}")
    
    features_list = []
    labels_list = []
    confs_list = []
    
    n_total = len(samples)
    n_tp = sum(1 for _, l, _ in samples if l == 1)
    n_fp = n_total - n_tp
    print(f"  Samples: {n_total} ({n_tp} TP + {n_fp} FP)")
    
    for i, (crop_path, label, conf) in enumerate(samples):
        try:
            img = np.array(Image.open(crop_path).convert('L'))
            feats, feat_names = extract_wavelet_features(img, wavelet=wavelet, levels=levels)
            features_list.append(feats)
            labels_list.append(label)
            confs_list.append(conf)
        except Exception as e:
            print(f"  [ERROR] {crop_path}: {e}")
        
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1}/{n_total}...")
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    confs = np.array(confs_list)
    
    print(f"  Feature matrix: {features.shape}")
    print(f"  Feature names: {feat_names}")
    
    # Per-feature PR-AUC
    print(f"\n  Per-feature PR-AUC (top 10):")
    per_feat_auc = []
    for j in range(features.shape[1]):
        try:
            score = features[:, j]
            # Handle negative scores (PR-AUC needs positive scores)
            score = score - score.min()
            pa = compute_pr_auc(labels, score)
            per_feat_auc.append((feat_names[j], pa))
        except:
            per_feat_auc.append((feat_names[j], 0.5))
    
    per_feat_auc.sort(key=lambda x: -x[1])
    for name, pa in per_feat_auc[:10]:
        print(f"    {name:30s}: {pa:.4f}")
    
    # Overall: use all features with logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores_cv = cross_val_predict(clf, features_scaled, labels, 
                                   method='decision_function', cv=5)
    
    wavelet_pr_auc = compute_pr_auc(labels, scores_cv)
    wavelet_roc_auc = roc_auc_score(labels, scores_cv)
    rfdetr_pr_auc = compute_pr_auc(labels, confs)
    rfdetr_roc_auc = roc_auc_score(labels, confs)
    
    print(f"\n  Results:")
    print(f"    Wavelet PR-AUC:  {wavelet_pr_auc:.4f}")
    print(f"    Wavelet ROC-AUC: {wavelet_roc_auc:.4f}")
    print(f"    RF-DETR PR-AUC:   {rfdetr_pr_auc:.4f}")
    print(f"    RF-DETR ROC-AUC:  {rfdetr_roc_auc:.4f}")
    
    # t-SNE visualization
    print(f"\n  Generating t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1))
    tsne_features = tsne.fit_transform(features_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    tp_mask = labels == 1
    ax.scatter(tsne_features[tp_mask, 0], tsne_features[tp_mask, 1], 
              c='blue', s=5, alpha=0.3, label=f'TP (n={n_tp})')
    ax.scatter(tsne_features[~tp_mask, 0], tsne_features[~tp_mask, 1], 
              c='red', s=5, alpha=0.3, label=f'FP (n={n_fp})')
    ax.set_title(f'{exp_name}: {wavelet} {levels}L\nPR-AUC={wavelet_pr_auc:.3f}, ROC-AUC={wavelet_roc_auc:.3f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{exp_name}_tsne.png', dpi=150)
    plt.close()
    
    return {
        'exp': exp_name,
        'wavelet': wavelet,
        'levels': levels,
        'n_samples': n_total,
        'n_tp': n_tp,
        'n_fp': n_fp,
        'n_features': features.shape[1],
        'wavelet_pr_auc': float(wavelet_pr_auc),
        'wavelet_roc_auc': float(wavelet_roc_auc),
        'rfdetr_pr_auc': float(rfdetr_pr_auc),
        'rfdetr_roc_auc': float(rfdetr_roc_auc),
        'top_features': [(n, float(a)) for n, a in per_feat_auc[:10]],
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 8: Wavelet Primitive Analysis')
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--crops-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results/wavelet_primitives')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples for quick test')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load samples
    print(f"Loading metadata: {args.metadata}")
    samples = load_metadata_and_crops(args.metadata, args.crops_dir, args.max_samples)
    print(f"Loaded {len(samples)} samples")
    
    if len(samples) == 0:
        print("ERROR: No crops found!")
        sys.exit(1)
    
    results = []
    
    # W1: Haar 2-level
    r = run_experiment(samples, 'haar', 2, 'W1', args.output_dir)
    results.append(r)
    
    # W2: Daubechies-4 3-level
    r = run_experiment(samples, 'db4', 3, 'W2', args.output_dir)
    results.append(r)
    
    # W3: Best wavelet + morphological features
    # Pick best wavelet from W1/W2
    best = max(results, key=lambda x: x['wavelet_pr_auc'])
    print(f"\n  Best wavelet: {best['wavelet']} ({best['wavelet_pr_auc']:.4f})")
    
    # Add morphological features from metadata
    with open(args.metadata, encoding='utf-8') as f:
        data = json.load(f)
    
    morph_features = {}
    if isinstance(data, list):
        for entry in data:
            ck = entry.get('cache_key', '')
            morph_features[ck] = [
                entry.get('area', 0),
                entry.get('circularity', 0),
                entry.get('solidity', 0),
                entry.get('aspect_ratio', 0),
            ]
    
    # Re-run with wavelet + morph
    combined_features = []
    combined_labels = []
    combined_confs = []
    
    for crop_path, label, conf in samples:
        ck = os.path.basename(crop_path).replace('.png', '')
        try:
            img = np.array(Image.open(crop_path).convert('L'))
            w_feats, _ = extract_wavelet_features(img, wavelet=best['wavelet'], levels=best['levels'])
            m_feats = morph_features.get(ck, [0, 0, 0, 0])
            combined_features.append(np.concatenate([w_feats, m_feats]))
            combined_labels.append(label)
            combined_confs.append(conf)
        except:
            pass
    
    combined_features = np.array(combined_features)
    combined_labels = np.array(combined_labels)
    combined_confs = np.array(combined_confs)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_features)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores_cv = cross_val_predict(clf, combined_scaled, combined_labels,
                                   method='decision_function', cv=5)
    
    w3_pr_auc = compute_pr_auc(combined_labels, scores_cv)
    w3_roc_auc = roc_auc_score(combined_labels, scores_cv)
    
    print(f"\n{'='*60}")
    print(f"  W3: {best['wavelet']} {best['levels']}L + morphological")
    print(f"{'='*60}")
    print(f"    Combined PR-AUC:  {w3_pr_auc:.4f}")
    print(f"    Combined ROC-AUC: {w3_roc_auc:.4f}")
    print(f"    (vs wavelet-only: {best['wavelet_pr_auc']:.4f})")
    
    results.append({
        'exp': 'W3',
        'wavelet': f"{best['wavelet']}+morph",
        'levels': best['levels'],
        'n_samples': len(combined_labels),
        'n_tp': int(combined_labels.sum()),
        'n_fp': int(len(combined_labels) - combined_labels.sum()),
        'n_features': combined_features.shape[1],
        'wavelet_pr_auc': float(w3_pr_auc),
        'wavelet_roc_auc': float(w3_roc_auc),
        'rfdetr_pr_auc': float(compute_pr_auc(combined_labels, combined_confs)),
        'rfdetr_roc_auc': float(roc_auc_score(combined_labels, combined_confs)),
    })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  Phase 8 Summary")
    print(f"{'='*60}")
    print(f"  {'Exp':6s} {'Wavelet':15s} {'Levels':6s} {'PR-AUC':8s} {'ROC-AUC':8s} {'n_feat':6s}")
    for r in results:
        print(f"  {r['exp']:6s} {r['wavelet']:15s} {r['levels']:6d} {r['wavelet_pr_auc']:8.4f} {r['wavelet_roc_auc']:8.4f} {r['n_features']:6d}")
    print(f"  {'RF-DETR':6s} {'':15s} {'':6s} {results[0]['rfdetr_pr_auc']:8.4f} {results[0]['rfdetr_roc_auc']:8.4f}")
    
    # Save results
    with open(f'{args.output_dir}/wavelet_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
