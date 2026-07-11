"""
Phase 9: Slot Attention Primitive Extraction (FORLA-inspired)

Tests whether TP/FP are separable in slot attention space.
Previous results: DINOv2 CLS PR-AUC=0.29, wavelet PR-AUC=0.45.
Slot attention learns object-centric decomposition which may capture
structure that global embeddings miss.

Architecture:
    Input crop → DINOv2 (frozen) → spatial tokens [257, 768]
                                        ↓
                              Slot Attention (K slots)
                                        ↓
                              K slot vectors [K, 128]
                                        ↓
                              Classifier (TP/FP)

Usage (cloud VM quick test, 100 crops):
    python slot_primitives.py --metadata results/phase2_vlm_100_mask/vlm_mask_results.json \
        --crops-dir results/phase2_vlm_100_mask/crops --epochs 20 --device cpu

Usage (冬生's 3060, full 16198 crops):
    python slot_primitives.py --metadata data/ctm_crops/ctm_metadata.json \
        --crops-dir data/ctm_crops --device cuda:0 --epochs 50 --batch-size 32
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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score

try:
    import timm
except ImportError:
    print("ERROR: timm not installed. Run: pip install timm")
    sys.exit(1)

# Add CTM module path for shared dataset
sys.path.insert(0, str(Path(__file__).parent / 'ctm'))
from ctm_dataset import OrganoidCTMDataset


class SlotAttention(nn.Module):
    """
    Slot Attention module (Locatello et al., NeurIPS 2020).

    Learns K object-centric slots from spatial features.
    Each slot attends to different parts of the image,
    decomposing the scene into object-level representations.

    Key difference from standard attention:
    - Q comes from slots (learned), not from input
    - Softmax is over slots (competition), not over tokens
    - GRU update + MLP residual for slot refinement
    """
    def __init__(self, num_slots=4, dim_input=768, dim_slots=128,
                 num_iters=3, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.dim_input = dim_input
        self.dim_slots = dim_slots
        self.num_iters = num_iters
        self.eps = eps

        # Projections
        self.norm_input = nn.LayerNorm(dim_input)
        self.norm_slots = nn.LayerNorm(dim_slots)
        self.to_q = nn.Linear(dim_slots, dim_slots, bias=False)
        self.to_k = nn.Linear(dim_input, dim_slots, bias=False)
        self.to_v = nn.Linear(dim_input, dim_slots, bias=False)

        # GRU for slot update
        self.gru = nn.GRUCell(dim_slots, dim_slots)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(dim_slots, dim_slots * 4),
            nn.GELU(),
            nn.Linear(dim_slots * 4, dim_slots),
        )

        # Learnable slot initialization
        self.slots_init = nn.Parameter(torch.randn(1, num_slots, dim_slots))
        nn.init.normal_(self.slots_init, std=0.02)

    def forward(self, kv_features, return_attn=False):
        """
        Args:
            kv_features: [B, N, D] spatial features from backbone
            return_attn: if True, return attention maps

        Returns:
            slots: [B, K, dim_slots]
            attns: [B, K, N] (if return_attn)
        """
        B = kv_features.shape[0]

        # Normalize input
        kv_features = self.norm_input(kv_features)

        # Initialize slots
        slots = self.slots_init.expand(B, -1, -1)  # [B, K, D]

        attns_all = []
        for _ in range(self.num_iters):
            # Normalize slots
            slots_norm = self.norm_slots(slots)

            # Q from slots, K/V from input
            q = self.to_q(slots_norm)  # [B, K, D]
            k = self.to_k(kv_features)  # [B, N, D]
            v = self.to_v(kv_features)  # [B, N, D]

            # Attention: [B, K, N] — softmax over SLOTS (competition)
            scale = q.shape[-1] ** -0.5
            attn_logits = torch.einsum('bkd,bnd->bkn', q * scale, k)  # [B, K, N]
            attn = F.softmax(attn_logits, dim=1)  # over K (slots compete)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)  # normalize over N

            if return_attn:
                attns_all.append(attn.detach())

            # Aggregate values
            updates = torch.einsum('bkn,bnd->bkd', attn, v)  # [B, K, D]

            # GRU update
            slots = self.gru(
                updates.reshape(B * self.num_slots, -1),
                slots.reshape(B * self.num_slots, -1),
            ).reshape(B, self.num_slots, -1)

            # MLP residual
            slots = slots + self.mlp(self.norm_slots(slots))

        if return_attn:
            return slots, torch.stack(attns_all, dim=1)  # [B, iters, K, N]
        return slots


class SlotModel(nn.Module):
    """
    Full model: DINOv2 backbone (frozen) + Slot Attention + Classifier.

    Outputs:
    - logits: [B, 2] for TP/FP classification
    - slots: [B, K, dim_slots] for analysis
    - attns: [B, iters, K, N] for visualization
    """
    def __init__(self, num_slots=4, dim_slots=128, num_iters=3,
                 n_classes=2, img_size=224,
                 backbone_name='vit_base_patch14_dinov2.lvd142m'):
        super().__init__()

        # Frozen DINOv2 backbone
        self.backbone = timm.create_model(
            backbone_name, pretrained=True,
            num_classes=0, dynamic_img_size=True,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        d_model = self.backbone.num_features  # 768

        # Slot attention
        self.slot_attn = SlotAttention(
            num_slots=num_slots,
            dim_input=d_model,
            dim_slots=dim_slots,
            num_iters=num_iters,
        )

        # Classifier on slot features
        # Pool slots: mean + max + individual slot logits
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim_slots),
            nn.Linear(dim_slots, dim_slots * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_slots * 2, n_classes),
        )

    def forward(self, images, return_attn=False):
        """
        Args:
            images: [B, 3, H, W]
            return_attn: if True, return attention maps

        Returns:
            logits: [B, n_classes]
            slots: [B, K, dim_slots]
            attns: [B, iters, K, N] (if return_attn)
        """
        # Extract DINOv2 spatial features
        with torch.no_grad():
            features = self.backbone.forward_features(images)
            if features.dim() == 3:
                spatial_tokens = features  # [B, N, D]
            else:
                # Handle models that return [B, D, H, W]
                spatial_tokens = features.flatten(2).transpose(1, 2)

        # Slot attention
        if return_attn:
            slots, attns = self.slot_attn(spatial_tokens, return_attn=True)
        else:
            slots = self.slot_attn(spatial_tokens)

        # Pool slots for classification (mean pooling)
        pooled = slots.mean(dim=1)  # [B, dim_slots]

        # Classify
        logits = self.classifier(pooled)

        if return_attn:
            return logits, slots, attns
        return logits, slots, None


def compute_pr_auc(labels, scores):
    """Compute PR-AUC (average precision)."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    model.backbone.eval()  # Keep frozen backbone in eval mode

    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        images = batch['image'].to(device)
        targets = batch['label'].to(device)

        optimizer.zero_grad()
        logits, _, _ = model(images)
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return {
        'loss': total_loss / max(len(loader), 1),
        'acc': correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    all_probs = []
    all_labels = []
    all_confs = []
    all_slots = []

    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].numpy()
        confs = batch['confidence'].numpy()

        logits, slots, _ = model(images)
        probs = F.softmax(logits, dim=-1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels)
        all_confs.append(confs)
        all_slots.append(slots.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_confs = np.concatenate(all_confs)
    all_slots = np.concatenate(all_slots)

    results = {}
    try:
        results['auc_slot'] = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        results['auc_slot'] = 0.5
    try:
        results['pr_auc_slot'] = compute_pr_auc(all_labels, all_probs[:, 1])
    except:
        results['pr_auc_slot'] = 0.5
    try:
        results['auc_rfdetr'] = roc_auc_score(all_labels, all_confs)
    except:
        results['auc_rfdetr'] = 0.5
    try:
        results['pr_auc_rfdetr'] = compute_pr_auc(all_labels, all_confs)
    except:
        results['pr_auc_rfdetr'] = 0.5

    preds = (all_probs[:, 1] > 0.5).astype(int)
    results['f1'] = f1_score(all_labels, preds, zero_division=0)
    results['n_samples'] = len(all_labels)
    results['slots'] = all_slots  # [N, K, dim_slots]
    results['labels'] = all_labels
    results['confs'] = all_confs

    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 9: Slot Attention Primitives')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to metadata JSON')
    parser.add_argument('--crops-dir', type=str, required=True,
                        help='Directory containing crop PNGs')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory. If None, auto-generates.')
    parser.add_argument('--num-slots', type=int, default=4,
                        help='Number of slots K (experiment S1: 4, S2: 8)')
    parser.add_argument('--dim-slots', type=int, default=128)
    parser.add_argument('--num-iters', type=int, default=3,
                        help='Slot attention iterations')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    # Auto-generate output dir
    if args.output_dir is None:
        date_str = time.strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'results/slot_{args.num_slots}slots_d{args.dim_slots}_{date_str}'
        # Avoid overwrite
        v = 2
        while os.path.exists(args.output_dir):
            args.output_dir = f'results/slot_{args.num_slots}slots_d{args.dim_slots}_{date_str}_v{v}'
            v += 1
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}")

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)

    # Build model
    print(f"\nBuilding model (K={args.num_slots}, dim={args.dim_slots}, iters={args.num_iters})...")
    model = SlotModel(
        num_slots=args.num_slots,
        dim_slots=args.dim_slots,
        num_iters=args.num_iters,
        img_size=args.img_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Frozen (DINOv2): {(total_params-trainable_params)/1e6:.1f}M")

    # Data
    print(f"\nLoading data...")
    train_ds = OrganoidCTMDataset(
        args.metadata, args.crops_dir, 'train',
        img_size=args.img_size, augment=True, balance=True, seed=42
    )
    val_ds = OrganoidCTMDataset(
        args.metadata, args.crops_dir, 'val',
        img_size=args.img_size, augment=False, balance=False, seed=42
    )
    test_ds = OrganoidCTMDataset(
        args.metadata, args.crops_dir, 'test',
        img_size=args.img_size, augment=False, balance=False, seed=42
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0
    )

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training ({args.epochs} epochs)")
    print(f"{'='*60}")

    best_val_auc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | {epoch_time:.1f}s | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['acc']:.3f} | "
              f"Val AUC: slot={val_metrics['auc_slot']:.3f}, "
              f"rfdetr={val_metrics['auc_rfdetr']:.3f} | "
              f"Val PR-AUC: slot={val_metrics['pr_auc_slot']:.3f}")

        history.append({
            'epoch': epoch,
            'time_s': epoch_time,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['acc'],
            'val_auc_slot': val_metrics['auc_slot'],
            'val_pr_auc_slot': val_metrics['pr_auc_slot'],
            'val_auc_rfdetr': val_metrics['auc_rfdetr'],
            'val_pr_auc_rfdetr': val_metrics['pr_auc_rfdetr'],
        })

        # Save best
        if val_metrics['auc_slot'] > best_val_auc:
            best_val_auc = val_metrics['auc_slot']
            patience_counter = 0
            # Save only trainable params (skip frozen DINOv2)
            trainable_sd = {
                k: v for k, v in model.state_dict().items()
                if not k.startswith('backbone.')
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainable_sd,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'args': vars(args),
                'config': {
                    'num_slots': args.num_slots,
                    'dim_slots': args.dim_slots,
                    'num_iters': args.num_iters,
                    'img_size': args.img_size,
                },
            }, os.path.join(args.output_dir, 'best.pt'))
            print(f"  → New best! Val AUC: {best_val_auc:.3f}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # Final test evaluation
    print(f"\n{'='*60}")
    print(f"Test evaluation")
    print(f"{'='*60}")

    ckpt = torch.load(os.path.join(args.output_dir, 'best.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"Loaded best model from epoch {ckpt['epoch']} (val AUC={ckpt['val_auc']:.3f})")

    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest Results:")
    print(f"  Slot AUC:       {test_metrics['auc_slot']:.4f}")
    print(f"  Slot PR-AUC:    {test_metrics['pr_auc_slot']:.4f}")
    print(f"  RF-DETR AUC:    {test_metrics['auc_rfdetr']:.4f}")
    print(f"  RF-DETR PR-AUC: {test_metrics['pr_auc_rfdetr']:.4f}")
    print(f"  F1:             {test_metrics['f1']:.4f}")

    # Save slot embeddings for downstream analysis (Phase 10/11)
    np.save(os.path.join(args.output_dir, 'test_slots.npy'),
            test_metrics['slots'])
    np.save(os.path.join(args.output_dir, 'test_labels.npy'),
            test_metrics['labels'])
    np.save(os.path.join(args.output_dir, 'test_confs.npy'),
            test_metrics['confs'])
    print(f"\n  Slot embeddings saved: {test_metrics['slots'].shape}")

    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'args': vars(args),
            'history': history,
            'test_metrics': {k: v for k, v in test_metrics.items()
                             if k not in ('slots', 'labels', 'confs')},
        }, f, indent=2)

    print(f"\nDone. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
