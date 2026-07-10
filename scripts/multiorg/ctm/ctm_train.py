"""
CTM Training Pipeline: DINOv2 backbone (frozen) + CTM head (trained)

Usage on cloud VM (CPU test, 100 crops):
    python ctm_train.py --metadata vlm_mask_results.json --crops-dir crops/ --epochs 5

Usage on 冬生's 3060 (full training, 16198 crops):
    python ctm_train.py --metadata multiorg_sam2_results.json --crops-dir all_crops/ --epochs 50 --device cuda:0 --batch-size 32
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import timm

# Add CTM module path
sys.path.insert(0, str(Path(__file__).parent))
from ctm_module import CTM, CTMLoss
from ctm_dataset import OrganoidCTMDataset


class DINOv2FeatureExtractor(nn.Module):
    """DINOv2 ViT-B/14 feature extractor (frozen)."""
    
    def __init__(self, model_name='vit_base_patch14_dinov2.lvd142m', img_size=224):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
        
        # Freeze ALL backbone parameters (eval + requires_grad=False)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get number of tokens
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            tokens = self.backbone.forward_features(dummy)
            if tokens.ndim == 3:  # [B, N, C]
                self.n_tokens = tokens.shape[1]
                self.embed_dim = tokens.shape[2]
            else:  # [B, C, H, W]
                self.n_tokens = tokens.numel() // tokens.shape[0]
                self.embed_dim = tokens.shape[1]
        
        n_total = sum(p.numel() for p in self.backbone.parameters())
        n_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"[DINOv2] {model_name}: {self.n_tokens} tokens × {self.embed_dim} dim, "
              f"{n_total/1e6:.1f}M params ({n_trainable/1e6:.1f}M trainable)")
    
    def forward(self, images):
        """Extract spatial features (KV for CTM cross-attention)."""
        with torch.no_grad():
            features = self.backbone.forward_features(images)  # [B, N, C]
            if features.ndim == 4:
                B, C, H, W = features.shape
                features = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        return features  # [B, N, C]


class OrganoidCTM(nn.Module):
    """Full model: DINOv2 backbone + CTM head for TP/FP discrimination."""
    
    def __init__(self, n_ticks=20, d_internal=256, n_heads=8, n_classes=2,
                 mem_len=20, n_action_pairs=128, n_output_pairs=128,
                 d_hidden=32, img_size=224):
        super().__init__()
        
        # Frozen feature extractor
        self.backbone = DINOv2FeatureExtractor(img_size=img_size)
        d_model = self.backbone.embed_dim  # 768 for ViT-B/14
        
        # CTM head
        self.ctm = CTM(
            d_model=d_model,
            d_internal=d_internal,
            n_ticks=n_ticks,
            n_heads=n_heads,
            n_classes=n_classes,
            mem_len=mem_len,
            n_action_pairs=n_action_pairs,
            n_output_pairs=n_output_pairs,
            d_hidden_nlm=d_hidden,
        )
        
        # Project KV to lower dim for efficiency (optional)
        self.kv_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.kv_proj.weight)  # Identity init
    
    def forward(self, images):
        """
        Args:
            images: [B, 3, H, W]
        Returns:
            logits_history: [B, C, T]
            certainties_history: [B, T]
            attention_weights_history: list of [B, n_heads, 1, N] per tick
        """
        # Extract features (frozen)
        kv_features = self.backbone(images)  # [B, N, C]
        kv_features = self.kv_proj(kv_features)
        
        # CTM forward
        logits_hist, cert_hist, attn_hist = self.ctm(kv_features)
        
        return logits_hist, cert_hist, attn_hist


def evaluate(model, loader, device, n_classes=2):
    """Evaluate model on a dataloader."""
    model.eval()
    all_logits = []
    all_certs = []
    all_labels = []
    all_confs = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            confs = batch['confidence'].numpy()
            
            logits_hist, cert_hist, _ = model(images)
            
            # Use last tick for evaluation
            final_logits = logits_hist[:, :, -1]  # [B, C]
            final_probs = F.softmax(final_logits, dim=-1)
            
            # Also get most certain tick
            cert_per_sample = cert_hist.max(dim=1)[0]  # [B]
            best_tick = cert_hist.argmax(dim=1)  # [B]
            certain_logits = logits_hist[torch.arange(len(best_tick)), :, best_tick]
            certain_probs = F.softmax(certain_logits, dim=-1)
            
            all_logits.append(final_probs.cpu().numpy())
            all_certs.append(cert_hist.cpu().numpy())
            all_labels.append(labels)
            all_confs.append(confs)
    
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    all_confs = np.concatenate(all_confs)
    
    # Metrics
    results = {}
    
    # Final tick AUC
    try:
        results['auc_final_tick'] = roc_auc_score(all_labels, all_logits[:, 1])
    except:
        results['auc_final_tick'] = 0.5
    
    # Most certain tick AUC
    try:
        results['auc_certain_tick'] = roc_auc_score(all_labels, all_logits[:, 1])
    except:
        results['auc_certain_tick'] = 0.5
    
    # RF-DETR confidence baseline
    try:
        results['auc_rfdetr'] = roc_auc_score(all_labels, all_confs)
    except:
        results['auc_rfdetr'] = 0.5
    
    # F1 at threshold 0.5
    preds = (all_logits[:, 1] > 0.5).astype(int)
    results['f1'] = f1_score(all_labels, preds, zero_division=0)
    results['precision'] = precision_score(all_labels, preds, zero_division=0)
    results['recall'] = recall_score(all_labels, preds, zero_division=0)
    
    # Accuracy at each tick (tick-wise analysis)
    # This requires re-running to get per-tick predictions
    # For now, just report final tick
    
    return results


def train_epoch(model, loader, optimizer, loss_fn, device, n_classes=2):
    """Train for one epoch."""
    model.train()
    # Keep backbone frozen
    model.backbone.eval()
    
    total_loss = 0
    n_batches = 0
    correct_best = 0
    correct_certain = 0
    total = 0
    
    for batch in loader:
        images = batch['image'].to(device)
        targets = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits_hist, cert_hist, _ = model(images)
        
        loss, info = loss_fn(logits_hist, cert_hist, targets)
        loss.backward()
        
        # Gradient clipping (only CTM params, backbone is frozen)
        torch.nn.utils.clip_grad_norm_(model.ctm.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        # Accuracy (use final tick predictions + info from loss)
        with torch.no_grad():
            final_preds = logits_hist[:, :, -1].argmax(dim=-1)  # final tick
            correct_best += (final_preds == targets).sum().item()
            correct_certain += (final_preds == targets).sum().item()
            total += targets.size(0)
    
    return {
        'loss': total_loss / max(n_batches, 1),
        'acc_best': correct_best / max(total, 1),
        'acc_certain': correct_certain / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='Train CTM for organoid TP/FP discrimination')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to metadata JSON (vlm_mask_results.json or SAM2 results)')
    parser.add_argument('--crops-dir', type=str, required=True,
                        help='Directory containing crop PNGs')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory. If None, auto-generates: results/ctm_{n_ticks}ticks_d{d_internal}_{date}')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)  # Paper: no weight decay
    parser.add_argument('--n-ticks', type=int, default=20)
    parser.add_argument('--d-internal', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--mem-len', type=int, default=20)
    parser.add_argument('--n-action-pairs', type=int, default=128)
    parser.add_argument('--n-output-pairs', type=int, default=128)
    parser.add_argument('--d-hidden', type=int, default=16, help='NLM hidden dim')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if 'cuda' in str(device):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory (auto-generate if not specified)
    if args.output_dir is None:
        from datetime import datetime
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'results/ctm_{args.n_ticks}ticks_d{args.d_internal}_{date_str}'
    
    # Safety check: don't overwrite existing results
    if os.path.exists(args.output_dir):
        existing = os.listdir(args.output_dir)
        if any(f.endswith('.pt') for f in existing):
            print(f"⚠️  WARNING: {args.output_dir} already has checkpoints!")
            print(f"   Existing files: {[f for f in existing if f.endswith('.pt')]}")
            # Auto-append _v2, _v3, etc.
            v = 2
            while os.path.exists(f'{args.output_dir}_v{v}'):
                v += 1
            args.output_dir = f'{args.output_dir}_v{v}'
            print(f"   Using {args.output_dir} instead to avoid overwrite")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}")
    
    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)
    
    # Data
    print(f"\n{'='*60}")
    print(f"Loading data")
    print(f"{'='*60}")
    train_ds = OrganoidCTMDataset(
        args.metadata, args.crops_dir, 'train',
        img_size=args.img_size, augment=True, balance=True, seed=args.seed
    )
    val_ds = OrganoidCTMDataset(
        args.metadata, args.crops_dir, 'val',
        img_size=args.img_size, augment=False, balance=False, seed=args.seed
    )
    test_ds = OrganoidCTMDataset(
        args.metadata, args.crops_dir, 'test',
        img_size=args.img_size, augment=False, balance=False, seed=args.seed
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory='cuda' in str(device))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory='cuda' in str(device))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory='cuda' in str(device))
    
    # Model
    print(f"\n{'='*60}")
    print(f"Building model")
    print(f"{'='*60}")
    model = OrganoidCTM(
        n_ticks=args.n_ticks,
        d_internal=args.d_internal,
        n_heads=args.n_heads,
        mem_len=args.mem_len,
        img_size=args.img_size,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {n_params/1e6:.2f}M")
    print(f"Trainable params: {n_trainable/1e6:.2f}M")
    
    # Optimizer (only CTM parameters, no backbone)
    ctm_params = list(model.ctm.parameters()) + list(model.kv_proj.parameters())
    optimizer = torch.optim.Adam(ctm_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Loss
    loss_fn = CTMLoss(n_classes=2)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training ({args.epochs} epochs)")
    print(f"{'='*60}")
    
    best_val_auc = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        t1 = time.time()
        epoch_time = t1 - t0
        
        # Log
        print(f"Epoch {epoch:3d}/{args.epochs} | {epoch_time:.1f}s | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['acc_best']:.3f}/{train_metrics['acc_certain']:.3f} | "
              f"Val AUC: final={val_metrics['auc_final_tick']:.3f}, "
              f"rfdetr={val_metrics['auc_rfdetr']:.3f} | "
              f"Val F1: {val_metrics['f1']:.3f}")
        
        history.append({
            'epoch': epoch,
            'time_s': epoch_time,
            'train_loss': train_metrics['loss'],
            'train_acc_best': train_metrics['acc_best'],
            'train_acc_certain': train_metrics['acc_certain'],
            'val_auc_final': val_metrics['auc_final_tick'],
            'val_auc_rfdetr': val_metrics['auc_rfdetr'],
            'val_f1': val_metrics['f1'],
        })
        
        # Save best
        if val_metrics['auc_final_tick'] > best_val_auc:
            best_val_auc = val_metrics['auc_final_tick']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'args': vars(args),
                'config': {
                    'n_ticks': args.n_ticks,
                    'd_internal': args.d_internal,
                    'n_heads': args.n_heads,
                    'mem_len': args.mem_len,
                    'n_action_pairs': args.n_action_pairs,
                    'n_output_pairs': args.n_output_pairs,
                    'd_hidden': args.d_hidden,
                    'img_size': args.img_size,
                },
            }, os.path.join(args.output_dir, 'best.pt'))
            print(f"  → New best! Val AUC: {best_val_auc:.3f}")
        else:
            patience_counter += 1
        
        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, os.path.join(args.output_dir, f'checkpoint_ep{epoch}.pt'))
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break
    
    # Final test evaluation
    print(f"\n{'='*60}")
    print(f"Test evaluation")
    print(f"{'='*60}")
    
    # Load best model
    ckpt = torch.load(os.path.join(args.output_dir, 'best.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded best model from epoch {ckpt['epoch']} (val AUC={ckpt['val_auc']:.3f})")
    
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest Results:")
    print(f"  CTM AUC (final tick): {test_metrics['auc_final_tick']:.4f}")
    print(f"  RF-DETR AUC:          {test_metrics['auc_rfdetr']:.4f}")
    print(f"  F1:                   {test_metrics['f1']:.4f}")
    print(f"  Precision:            {test_metrics['precision']:.4f}")
    print(f"  Recall:               {test_metrics['recall']:.4f}")
    
    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'args': vars(args),
            'history': history,
            'test_metrics': test_metrics,
        }, f, indent=2)
    
    print(f"\nDone. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
