"""
CTM Evaluation: Tick-wise analysis + attention visualization

Key metrics:
1. ROC-AUC at each tick (should increase if CTM is "thinking")
2. Certainty distribution (should be long-tail for adaptive computation)
3. Attention map visualization (should show "looking around" behavior)
4. TP vs FP tick patterns (should differ)
5. Calibration curve (predicted prob vs actual accuracy)

Usage:
    python ctm_evaluate.py --checkpoint results/ctm/best.pt --metadata vlm_mask_results.json --crops-dir crops/
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
try:
    import timm
except ImportError:
    print("ERROR: timm not installed. Run: pip install timm")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from ctm_module import CTM
from ctm_dataset import OrganoidCTMDataset

# Import the full model class from train script
from ctm_train import OrganoidCTM, DINOv2FeatureExtractor


def load_model(checkpoint_path: str, device: str = 'cpu', 
               n_ticks=20, d_internal=256, n_heads=8, mem_len=20,
               n_action_pairs=128, n_output_pairs=128, d_hidden=16,
               img_size=224):
    """Load trained CTM model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    config = ckpt.get('config', {})
    
    # Always infer all dims from state_dict (config may be wrong)
    sd = ckpt['model_state_dict']
    if 'ctm.nlms.weights_1' in sd:
        # weights_1: [mem_len, d_hidden, d_internal]
        config['d_hidden'] = sd['ctm.nlms.weights_1'].shape[1]
        config['mem_len'] = sd['ctm.nlms.weights_1'].shape[0]
        config['d_internal'] = sd['ctm.nlms.weights_1'].shape[2]
        print(f"  [inferred] d_hidden={config['d_hidden']}, mem_len={config['mem_len']}, d_internal={config['d_internal']}")
    if 'ctm.attention.in_proj_weight' in sd:
        # attention: in_proj_weight is [3*d_model, d_model]
        full_dim = sd['ctm.attention.in_proj_weight'].shape[0]
        config['n_heads'] = config.get('n_heads', 8)  # can't infer n_heads from weights
    if 'ctm.synch_action.pair_indices' in sd:
        config['n_action_pairs'] = sd['ctm.synch_action.pair_indices'].shape[0]
    if 'ctm.synch_output.pair_indices' in sd:
        config['n_output_pairs'] = sd['ctm.synch_output.pair_indices'].shape[0]
    
    model = OrganoidCTM(
        n_ticks=config.get('n_ticks', n_ticks),
        d_internal=config.get('d_internal', d_internal),
        n_heads=config.get('n_heads', n_heads),
        mem_len=config.get('mem_len', mem_len),
        n_action_pairs=config.get('n_action_pairs', n_action_pairs),
        n_output_pairs=config.get('n_output_pairs', n_output_pairs),
        d_hidden=config.get('d_hidden', d_hidden),
        img_size=img_size,
    )
    # New checkpoints only save trainable params (no backbone.* keys).
    # Need strict=False for backward compat with old full-state checkpoints
    # and for new trainable-only checkpoints alike.
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    
    return model, config


@torch.no_grad()
def evaluate_tick_wise(model, loader, device):
    """
    Evaluate CTM tick by tick.
    
    Returns dict with:
        - auc_per_tick: list of ROC-AUC at each tick
        - acc_per_tick: list of accuracy at each tick
        - certainty_per_tick: list of mean certainty at each tick
        - final_auc: ROC-AUC at final tick
        - best_auc: best ROC-AUC across ticks
        - rfdetr_auc: RF-DETR confidence AUC (baseline)
        - all_scores: [N, T] TP prob per tick
        - all_labels: [N] ground truth
        - all_certainties: [N, T] certainty per tick
        - all_rfdetr_conf: [N] RF-DETR confidence
        - all_attn_weights: [N, T, n_heads, S] attention weights
    """
    all_logits = []
    all_certs = []
    all_labels = []
    all_rfdetr = []
    all_attn = []  # Only collect first 50 samples for visualization
    n_attn_collected = 0
    max_attn_samples = 50
    
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].numpy()
        rfdetr_conf = batch['confidence'].numpy()
        B = len(labels)
        
        logits_hist, cert_hist, attn_hist = model(images)
        
        # TP probability (class 1) per tick
        tp_prob = F.softmax(logits_hist, dim=1)[:, 1, :]  # [B, T]
        
        all_logits.append(tp_prob.cpu().numpy())
        all_certs.append(cert_hist.cpu().numpy())
        all_labels.append(labels)
        all_rfdetr.append(rfdetr_conf)
        
        # Only collect attention for first max_attn_samples (avoid OOM)
        if n_attn_collected < max_attn_samples:
            n_take = min(B, max_attn_samples - n_attn_collected)
            all_attn.append(attn_hist[:n_take].cpu().numpy())
            n_attn_collected += n_take
    
    all_logits = np.concatenate(all_logits, axis=0)  # [N, T]
    all_certs = np.concatenate(all_certs, axis=0)  # [N, T]
    all_labels = np.concatenate(all_labels, axis=0)  # [N]
    all_rfdetr = np.concatenate(all_rfdetr, axis=0)  # [N]
    all_attn = np.concatenate(all_attn, axis=0) if all_attn else np.array([])  # [≤50, T, n_heads, S]
    
    N, T = all_logits.shape
    
    # Per-tick metrics
    auc_per_tick = []
    acc_per_tick = []
    for t in range(T):
        scores = all_logits[:, t]
        preds = (scores > 0.5).astype(int)
        try:
            auc = roc_auc_score(all_labels, scores)
        except:
            auc = 0.5
        acc = (preds == all_labels).mean()
        auc_per_tick.append(auc)
        acc_per_tick.append(acc)
    
    # RF-DETR baseline
    try:
        rfdetr_auc = roc_auc_score(all_labels, all_rfdetr)
    except:
        rfdetr_auc = 0.5
    
    # Certainty stats
    cert_per_tick = all_certs.mean(axis=0)  # [T]
    
    # Best tick AUC
    best_tick = np.argmax(auc_per_tick)
    
    # Adaptive computation: how many ticks needed for 80% certainty?
    cert_threshold = 0.8
    ticks_needed = []
    for i in range(N):
        for t in range(T):
            if all_certs[i, t] >= cert_threshold:
                ticks_needed.append(t + 1)
                break
        else:
            ticks_needed.append(T)
    ticks_needed = np.array(ticks_needed)
    
    # Certain-tick AUC (per-sample argmax certainty tick, not global best tick)
    certain_ticks = all_certs.argmax(axis=1)  # [N]
    certain_scores = all_logits[np.arange(N), certain_ticks]  # [N]
    try:
        certain_auc = roc_auc_score(all_labels, certain_scores)
    except:
        certain_auc = 0.5
    
    return {
        'auc_per_tick': auc_per_tick,
        'acc_per_tick': acc_per_tick,
        'certainty_per_tick': cert_per_tick.tolist(),
        'final_auc': auc_per_tick[-1],
        'best_auc': max(auc_per_tick),
        'best_tick': int(best_tick),
        'certain_auc': float(certain_auc),
        'rfdetr_auc': rfdetr_auc,
        'all_scores': all_logits.tolist(),
        'all_labels': all_labels.tolist(),
        'all_certainties': all_certs.tolist(),
        'all_rfdetr_conf': all_rfdetr.tolist(),
        'n_samples': N,
        'n_ticks': T,
        'ticks_needed': ticks_needed.tolist(),
        'ticks_needed_mean': float(ticks_needed.mean()),
        'ticks_needed_median': float(np.median(ticks_needed)),
        'all_attn_subset': all_attn.tolist() if all_attn.size > 0 else [],
        'n_attn_samples': int(all_attn.shape[0]) if all_attn.size > 0 else 0,
    }


def plot_results(results, output_dir):
    """Generate evaluation plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    T = results['n_ticks']
    
    # 1. Tick-wise AUC
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, T+1), results['auc_per_tick'], 'b-o', label='CTM AUC')
    ax.axhline(y=results['rfdetr_auc'], color='r', linestyle='--', label=f'RF-DETR AUC={results["rfdetr_auc"]:.3f}')
    ax.set_xlabel('Internal Tick')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('CTM Tick-wise AUC (should increase if "thinking")')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tick_wise_auc.png', dpi=150)
    plt.close()
    
    # 2. Certainty distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    cert_arr = np.array(results['all_certainties'])
    ax.plot(range(1, T+1), cert_arr.mean(axis=0), 'g-o', label='Mean certainty')
    ax.fill_between(range(1, T+1), 
                    cert_arr.mean(axis=0) - cert_arr.std(axis=0),
                    cert_arr.mean(axis=0) + cert_arr.std(axis=0),
                    alpha=0.3, color='g')
    ax.axhline(y=0.8, color='r', linestyle='--', label='0.8 certainty threshold')
    ax.set_xlabel('Internal Tick')
    ax.set_ylabel('Certainty (1 - normalized entropy)')
    ax.set_title('Certainty Evolution Across Ticks')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/certainty_evolution.png', dpi=150)
    plt.close()
    
    # 3. Ticks needed distribution (adaptive computation)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(results['ticks_needed'], bins=range(1, T+2), edgecolor='black', alpha=0.7)
    ax.axvline(x=results['ticks_needed_mean'], color='r', linestyle='--', 
               label=f'Mean={results["ticks_needed_mean"]:.1f}')
    ax.set_xlabel('Ticks Needed (certainty > 0.8)')
    ax.set_ylabel('Count')
    ax.set_title('Adaptive Computation: Ticks Needed Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ticks_needed.png', dpi=150)
    plt.close()
    
    # 4. TP vs FP score trajectories
    fig, ax = plt.subplots(figsize=(10, 5))
    scores = np.array(results['all_scores'])
    labels = np.array(results['all_labels'])
    tp_scores = scores[labels == 1]
    fp_scores = scores[labels == 0]
    
    ax.plot(range(1, T+1), tp_scores.mean(axis=0), 'g-o', label=f'TP (n={len(tp_scores)})')
    ax.fill_between(range(1, T+1),
                    tp_scores.mean(axis=0) - tp_scores.std(axis=0),
                    tp_scores.mean(axis=0) + tp_scores.std(axis=0),
                    alpha=0.3, color='g')
    ax.plot(range(1, T+1), fp_scores.mean(axis=0), 'r-o', label=f'FP (n={len(fp_scores)})')
    ax.fill_between(range(1, T+1),
                    fp_scores.mean(axis=0) - fp_scores.std(axis=0),
                    fp_scores.mean(axis=0) + fp_scores.std(axis=0),
                    alpha=0.3, color='r')
    ax.set_xlabel('Internal Tick')
    ax.set_ylabel('TP Probability')
    ax.set_title('TP vs FP Score Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tp_fp_trajectories.png', dpi=150)
    plt.close()
    
    # 5. Calibration curve
    fig, ax = plt.subplots(figsize=(6, 6))
    final_scores = scores[:, -1]
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_acc = []
    bin_counts = []
    for i in range(len(bins)-1):
        mask = (final_scores >= bins[i]) & (final_scores < bins[i+1])
        if mask.sum() > 0:
            bin_acc.append(labels[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_acc.append(np.nan)
            bin_counts.append(0)
    
    ax.plot(bin_centers, bin_acc, 'b-o', label='CTM')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Predicted TP Probability')
    ax.set_ylabel('Actual TP Rate')
    ax.set_title('Calibration Curve (final tick)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/calibration.png', dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Evaluate CTM model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--crops-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory. If None, auto-generates from checkpoint dir + _eval')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    
    # Expand glob patterns in --checkpoint (Windows doesn't auto-expand wildcards)
    if any(c in args.checkpoint for c in '*?[]'):
        matches = sorted(glob.glob(args.checkpoint))
        if not matches:
            print(f"ERROR: No files match pattern: {args.checkpoint}")
            sys.exit(1)
        args.checkpoint = matches[-1]  # Use latest match
        print(f"Resolved checkpoint: {args.checkpoint}")
    
    # Auto-generate output dir from checkpoint path
    if args.output_dir is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        args.output_dir = os.path.join(ckpt_dir, 'eval')
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}")
    
    # Load model
    model, config = load_model(args.checkpoint, args.device, img_size=args.img_size)
    print(f"Loaded model from {args.checkpoint}")
    print(f"Config: {config}")
    
    # Load test dataset
    test_ds = OrganoidCTMDataset(
        args.metadata, args.crops_dir, 'test',
        img_size=args.img_size, augment=False, balance=False
    )
    print(f"Test: {len(test_ds)} samples")
    
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Evaluate
    print(f"\nEvaluating...")
    results = evaluate_tick_wise(model, test_loader, args.device)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"CTM Evaluation Results")
    print(f"{'='*60}")
    print(f"  Samples: {results['n_samples']}")
    print(f"  Ticks: {results['n_ticks']}")
    print(f"")
    print(f"  RF-DETR AUC (baseline):  {results['rfdetr_auc']:.4f}")
    print(f"  CTM Final-tick AUC:      {results['final_auc']:.4f}")
    print(f"  CTM Best-tick AUC:       {results['best_auc']:.4f} (tick {results['best_tick']+1})")
    print(f"  CTM Certain-tick AUC:    {results['certain_auc']:.4f} (per-sample argmax certainty)")
    print(f"")
    print(f"  Tick-wise AUC:")
    for t in range(results['n_ticks']):
        marker = ' ← best' if t == results['best_tick'] else ''
        print(f"    Tick {t+1:2d}: AUC={results['auc_per_tick'][t]:.4f}, acc={results['acc_per_tick'][t]:.4f}, cert={results['certainty_per_tick'][t]:.4f}{marker}")
    print(f"")
    print(f"  Adaptive computation (certainty > 0.8):")
    print(f"    Mean ticks needed: {results['ticks_needed_mean']:.1f}")
    print(f"    Median ticks needed: {results['ticks_needed_median']:.1f}")
    
    # Key diagnostic: does AUC increase across ticks?
    auc_trend = results['auc_per_tick']
    if auc_trend[-1] > auc_trend[0]:
        print(f"\n  ✓ AUC increases across ticks ({auc_trend[0]:.3f} → {auc_trend[-1]:.3f}) — CTM is 'thinking'")
    else:
        print(f"\n  ✗ AUC does not increase ({auc_trend[0]:.3f} → {auc_trend[-1]:.3f}) — no iterative refinement")
    
    # Generate plots
    plot_results(results, args.output_dir)
    
    # Save results JSON (exclude attention weights — too large for JSON)
    results_for_json = {k: v for k, v in results.items() if k != 'all_attn_subset'}
    with open(f'{args.output_dir}/eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_for_json, f, indent=2)
    
    # Save attention weights separately as .npy (much smaller than JSON)
    if results.get('n_attn_samples', 0) > 0:
        attn_arr = np.array(results['all_attn_subset'])
        np.save(f'{args.output_dir}/attn_weights.npy', attn_arr)
        print(f"Attention weights saved: {attn_arr.shape} → attn_weights.npy")
    
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
