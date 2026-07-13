"""
Phase 11: Supervised Contrastive Learning for Slot Attention (SupCon)

动机：Phase 9 无监督 slot attention PR-AUC=0.788，组合 confidence 后 0.854。
目标：用 SupCon loss 强制拉开 TP/FP slot embedding，让 slot alone 也能到 0.85+。

架构：
    Input crop → DINOv2 (frozen) → spatial tokens [257, 768]
                                        ↓
                              Slot Attention (K=8, d=128, 3 iters)
                                        ↓
                              slots [8, 128] → mean pool → [128]
                                        ↓
                              Projection Head (128 → 256 → L2 norm) → [256]
                                        ↓
                              SupCon Loss (TP-TP 拉近, TP-FP 推远)
                              + CE Loss (classification)

Loss = α * CE + β * SupCon
    α = 1.0 (classification head)
    β = 0.5 (contrastive, 防止过强 contrastive 破坏分类)

Usage (冬生's 3060, full 16198 crops):
    python slot_supcon.py --metadata data/ctm_crops/ctm_metadata.json \
        --crops-dir data/ctm_crops --device cuda:0 --epochs 50 \
        --num-slots 8 --dim-slots 128 --num-iters 3 \
        --batch-size 32 --temperature 0.07 --supcon-weight 0.5

Note: SupCon ideally wants B>=64 for more positives, but 3060 12GB
needs B=32 (same as Phase 9). With balanced 50/50, B=32 gives ~16
TP + ~16 FP per batch, sufficient for SupCon.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

try:
    import timm
except ImportError:
    print("ERROR: timm not installed. Run: pip install timm")
    sys.exit(1)

# Add CTM module path for shared dataset
sys.path.insert(0, str(Path(__file__).parent / 'ctm'))
from ctm_dataset import OrganoidCTMDataset

# Import SlotAttention from Phase 9
sys.path.insert(0, str(Path(__file__).parent))
from slot_primitives import SlotAttention


class SlotSupConModel(nn.Module):
    """
    Phase 11 model: DINOv2 (frozen) + Slot Attention + Projection Head + Classifier.

    Outputs:
    - logits: [B, 2] for TP/FP classification (CE loss)
    - embeddings: [B, 256] L2-normalized (SupCon loss)
    - slots: [B, K, dim_slots] for analysis
    """
    def __init__(self, num_slots=8, dim_slots=128, num_iters=3,
                 n_classes=2, img_size=224, proj_dim=256,
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

        # Slot attention (same as Phase 9)
        self.slot_attn = SlotAttention(
            num_slots=num_slots,
            dim_input=d_model,
            dim_slots=dim_slots,
            num_iters=num_iters,
        )

        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.LayerNorm(dim_slots),
            nn.Linear(dim_slots, dim_slots * 2),
            nn.GELU(),
            nn.Linear(dim_slots * 2, proj_dim),
        )

        # Classifier (for CE loss + evaluation)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim_slots),
            nn.Linear(dim_slots, dim_slots * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_slots * 2, n_classes),
        )

    def forward(self, images, return_embeddings=True):
        """
        Args:
            images: [B, 3, H, W]
            return_embeddings: if True, also return L2-normalized embeddings

        Returns:
            logits: [B, n_classes]
            slots: [B, K, dim_slots]
            embeddings: [B, proj_dim] (L2-normalized) or None
        """
        with torch.no_grad():
            features = self.backbone.forward_features(images)
            if features.dim() == 3:
                spatial_tokens = features
            else:
                spatial_tokens = features.flatten(2).transpose(1, 2)

        slots = self.slot_attn(spatial_tokens)
        pooled = slots.mean(dim=1)  # [B, dim_slots]

        logits = self.classifier(pooled)

        if return_embeddings:
            embeddings = F.normalize(self.projector(pooled), dim=-1)
            return logits, slots, embeddings
        return logits, slots, None


def supcon_loss(embeddings, labels, temperature=0.07):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    For each anchor, positives = same-class samples in batch,
    negatives = different-class samples.

    L = -1/N * sum_i sum_{p in P(i)} log(
        exp(z_i · z_p / τ) / sum_{a in A(i)} exp(z_i · z_a / τ)
    )

    Args:
        embeddings: [B, D] L2-normalized
        labels: [B] class labels (0=FP, 1=TP)
        temperature: τ

    Returns:
        loss: scalar
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    if batch_size <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Similarity matrix [B, B]
    sim = torch.matmul(embeddings, embeddings.T) / temperature  # [B, B]

    # Mask: exclude self
    mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    sim = sim.masked_fill(mask, -1e9)

    # Log probability: log_softmax over dim 1
    log_prob = F.log_softmax(sim, dim=1)  # [B, B]

    # Positive mask: same class, not self
    labels = labels.view(-1, 1)  # [B, 1]
    pos_mask = (labels == labels.T) & ~mask  # [B, B] same class, not self

    # Count positives per anchor
    n_pos = pos_mask.sum(dim=1)  # [B]

    # Mean of log-prob over positives
    # For anchors with no positives, loss = 0
    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1)  # [B]

    # Avoid division by zero
    n_pos = n_pos.clamp(min=1).float()
    loss = -mean_log_prob_pos / n_pos

    # Only count anchors that have at least one positive
    has_pos = (pos_mask.sum(dim=1) > 0).float()
    loss = (loss * has_pos).sum() / has_pos.sum().clamp(min=1)

    return loss


def compute_pr_auc(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def train_epoch(model, loader, optimizer, device, temperature, supcon_weight):
    """Train one epoch with combined CE + SupCon loss."""
    model.train()
    model.backbone.eval()

    total_loss = 0
    total_ce = 0
    total_supcon = 0
    correct = 0
    total = 0

    for batch in loader:
        images = batch['image'].to(device)
        targets = batch['label'].to(device)

        optimizer.zero_grad()
        logits, _, embeddings = model(images, return_embeddings=True)

        # CE loss
        ce_loss = F.cross_entropy(logits, targets)

        # SupCon loss
        sc_loss = supcon_loss(embeddings, targets, temperature)

        # Combined
        loss = ce_loss + supcon_weight * sc_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_supcon += sc_loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    n = max(len(loader), 1)
    return {
        'loss': total_loss / n,
        'ce_loss': total_ce / n,
        'supcon_loss': total_supcon / n,
        'acc': correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model. Returns metrics + embeddings for downstream analysis."""
    model.eval()
    all_probs = []
    all_labels = []
    all_confs = []
    all_slots = []
    all_embeddings = []

    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].numpy()
        confs = batch['confidence'].numpy()

        logits, slots, embeddings = model(images, return_embeddings=True)
        probs = F.softmax(logits, dim=-1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels)
        all_confs.append(confs)
        all_slots.append(slots.cpu().numpy())
        all_embeddings.append(embeddings.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_confs = np.concatenate(all_confs)
    all_slots = np.concatenate(all_slots)
    all_embeddings = np.concatenate(all_embeddings)

    results = {}
    # Classifier AUC
    try:
        results['auc_slot'] = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        results['auc_slot'] = 0.5
    try:
        results['pr_auc_slot'] = compute_pr_auc(all_labels, all_probs[:, 1])
    except:
        results['pr_auc_slot'] = 0.5
    # RF-DETR baseline
    try:
        results['auc_rfdetr'] = roc_auc_score(all_labels, all_confs)
    except:
        results['auc_rfdetr'] = 0.5
    try:
        results['pr_auc_rfdetr'] = compute_pr_auc(all_labels, all_confs)
    except:
        results['pr_auc_rfdetr'] = 0.5

    # Embedding-space AUC (5-fold LR on L2-normalized embeddings)
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(all_embeddings)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    emb_probs = np.zeros(len(all_labels))
    for train_idx, test_idx in skf.split(emb_scaled, all_labels):
        lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        lr.fit(emb_scaled[train_idx], all_labels[train_idx])
        emb_probs[test_idx] = lr.predict_proba(emb_scaled[test_idx])[:, 1]

    try:
        results['auc_embedding'] = roc_auc_score(all_labels, emb_probs)
    except:
        results['auc_embedding'] = 0.5
    try:
        results['pr_auc_embedding'] = compute_pr_auc(all_labels, emb_probs)
    except:
        results['pr_auc_embedding'] = 0.5

    # Combined: LR(slot_prob, conf)
    meta_features = np.column_stack([emb_probs, all_confs])
    meta_scaler = StandardScaler()
    meta_scaled = meta_scaler.fit_transform(meta_features)
    meta_probs = np.zeros(len(all_labels))
    for train_idx, test_idx in skf.split(meta_scaled, all_labels):
        lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        lr.fit(meta_scaled[train_idx], all_labels[train_idx])
        meta_probs[test_idx] = lr.predict_proba(meta_scaled[test_idx])[:, 1]

    try:
        results['auc_combined'] = roc_auc_score(all_labels, meta_probs)
    except:
        results['auc_combined'] = 0.5
    try:
        results['pr_auc_combined'] = compute_pr_auc(all_labels, meta_probs)
    except:
        results['pr_auc_combined'] = 0.5

    preds = (all_probs[:, 1] > 0.5).astype(int)
    results['f1'] = f1_score(all_labels, preds, zero_division=0)
    results['n_samples'] = len(all_labels)
    results['slots'] = all_slots
    results['labels'] = all_labels
    results['confs'] = all_confs
    results['embeddings'] = all_embeddings

    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 11: SupCon Slot Attention')
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--crops-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--num-slots', type=int, default=8)
    parser.add_argument('--dim-slots', type=int, default=128)
    parser.add_argument('--num-iters', type=int, default=3)
    parser.add_argument('--proj-dim', type=int, default=256,
                        help='Projection head output dimension')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='SupCon temperature τ')
    parser.add_argument('--supcon-weight', type=float, default=0.5,
                        help='β in Loss = CE + β*SupCon')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='B=32 for 12GB GPU (same as Phase 9). SupCon ideally B>=64')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    # Auto-generate output dir
    if args.output_dir is None:
        date_str = time.strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'results/supcon_{args.num_slots}s_d{args.dim_slots}_p{args.proj_dim}_t{args.temperature}_b{args.supcon_weight}_{date_str}'
        v = 2
        while os.path.exists(args.output_dir):
            args.output_dir = f'results/supcon_{args.num_slots}s_d{args.dim_slots}_p{args.proj_dim}_t{args.temperature}_b{args.supcon_weight}_{date_str}_v{v}'
            v += 1
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}")

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)

    # Build model
    print(f"\nBuilding SupCon model (K={args.num_slots}, dim={args.dim_slots}, "
          f"proj={args.proj_dim}, τ={args.temperature}, β={args.supcon_weight})...")
    model = SlotSupConModel(
        num_slots=args.num_slots,
        dim_slots=args.dim_slots,
        num_iters=args.num_iters,
        n_classes=2,  # TP/FP binary classification
        img_size=args.img_size,
        proj_dim=args.proj_dim,
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
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)  # drop_last for consistent batch size
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
    print(f"Training ({args.epochs} epochs, CE + {args.supcon_weight}×SupCon)")
    print(f"  Batch size: {args.batch_size} (need >1 TP and >1 FP per batch for SupCon)")
    print(f"{'='*60}")

    best_val_auc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            args.temperature, args.supcon_weight
        )
        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | {epoch_time:.1f}s | "
              f"Loss: {train_metrics['loss']:.4f} "
              f"(CE={train_metrics['ce_loss']:.4f}, "
              f"SupCon={train_metrics['supcon_loss']:.4f}) | "
              f"Train Acc: {train_metrics['acc']:.3f} | "
              f"Val: slot={val_metrics['auc_slot']:.3f}, "
              f"emb={val_metrics['auc_embedding']:.3f}, "
              f"comb={val_metrics['auc_combined']:.3f}")

        history.append({
            'epoch': epoch,
            'time_s': epoch_time,
            'train_loss': train_metrics['loss'],
            'train_ce_loss': train_metrics['ce_loss'],
            'train_supcon_loss': train_metrics['supcon_loss'],
            'train_acc': train_metrics['acc'],
            'val_auc_slot': val_metrics['auc_slot'],
            'val_auc_embedding': val_metrics['auc_embedding'],
            'val_auc_combined': val_metrics['auc_combined'],
            'val_pr_auc_slot': val_metrics['pr_auc_slot'],
            'val_pr_auc_embedding': val_metrics['pr_auc_embedding'],
            'val_pr_auc_combined': val_metrics['pr_auc_combined'],
            'val_auc_rfdetr': val_metrics['auc_rfdetr'],
        })

        # Save best (by combined AUC)
        if val_metrics['auc_combined'] > best_val_auc:
            best_val_auc = val_metrics['auc_combined']
            patience_counter = 0
            trainable_sd = {
                k: v for k, v in model.state_dict().items()
                if not k.startswith('backbone.')
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainable_sd,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc_combined': best_val_auc,
                'args': vars(args),
                'config': {
                    'num_slots': args.num_slots,
                    'dim_slots': args.dim_slots,
                    'num_iters': args.num_iters,
                    'proj_dim': args.proj_dim,
                    'img_size': args.img_size,
                },
            }, os.path.join(args.output_dir, 'best.pt'))
            print(f"  → New best! Val combined AUC: {best_val_auc:.3f}")
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
    print(f"Loaded best model from epoch {ckpt['epoch']} (val combined AUC={ckpt['val_auc_combined']:.3f})")

    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest Results:")
    print(f"  Slot AUC (classifier):  {test_metrics['auc_slot']:.4f}")
    print(f"  Embedding AUC (5-fold): {test_metrics['auc_embedding']:.4f}")
    print(f"  Combined AUC (emb+conf):{test_metrics['auc_combined']:.4f}")
    print(f"  RF-DETR AUC:             {test_metrics['auc_rfdetr']:.4f}")
    print(f"  ---")
    print(f"  Slot PR-AUC:             {test_metrics['pr_auc_slot']:.4f}")
    print(f"  Embedding PR-AUC:        {test_metrics['pr_auc_embedding']:.4f}")
    print(f"  Combined PR-AUC:         {test_metrics['pr_auc_combined']:.4f}")
    print(f"  RF-DETR PR-AUC:          {test_metrics['pr_auc_rfdetr']:.4f}")
    print(f"  F1:                      {test_metrics['f1']:.4f}")

    # Compare with Phase 9
    print(f"\n  --- Phase 9 vs Phase 11 ---")
    print(f"  Phase 9  slot:     ROC=0.868, PR=0.788")
    print(f"  Phase 9  combined:  ROC=0.903, PR=0.853")
    print(f"  Phase 11 slot:     ROC={test_metrics['auc_slot']:.3f}, PR={test_metrics['pr_auc_slot']:.3f}")
    print(f"  Phase 11 emb:      ROC={test_metrics['auc_embedding']:.3f}, PR={test_metrics['pr_auc_embedding']:.3f}")
    print(f"  Phase 11 combined: ROC={test_metrics['auc_combined']:.3f}, PR={test_metrics['pr_auc_combined']:.3f}")

    # Save embeddings for Phase 10
    np.save(os.path.join(args.output_dir, 'test_slots.npy'), test_metrics['slots'])
    np.save(os.path.join(args.output_dir, 'test_labels.npy'), test_metrics['labels'])
    np.save(os.path.join(args.output_dir, 'test_confs.npy'), test_metrics['confs'])
    np.save(os.path.join(args.output_dir, 'test_embeddings.npy'), test_metrics['embeddings'])
    print(f"\n  Embeddings saved: {test_metrics['embeddings'].shape}")
    print(f"  Slots saved: {test_metrics['slots'].shape}")

    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'args': vars(args),
            'history': history,
            'test_metrics': {k: v for k, v in test_metrics.items()
                             if k not in ('slots', 'labels', 'confs', 'embeddings')},
        }, f, indent=2)

    print(f"\nDone. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
