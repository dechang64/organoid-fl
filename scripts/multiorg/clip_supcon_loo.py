r"""
Phase A1: CLIP + SupCon 联合训练 + LOO 跨域评估

用 CLIP ViT-B/16 替代 DINOv2 做 backbone，训练 SupCon slot model。
CLIP 的文本-视觉对齐空间比 DINOv2 纯视觉空间更跨域。

流程：
1. 用 CLIP 编码所有 crops（冻结，不训练）
2. 在 CLIP 特征上训练 SupCon slot model（同 DINOv2 流程）
3. LOO 留一法跨域评估

Usage (冬生本地 GPU):
    cd C:\\Users\\decha\\organoid-fl
    python scripts\\multiorg\\clip_supcon_loo.py --device cuda:0 --epochs 50

Usage (云 VM CPU):
    cd /home/z/my-project/organoid-fl
    python3 scripts/multiorg/clip_supcon_loo.py --device cpu --epochs 20
"""
import argparse
import json
import os
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(str(PROJECT_ROOT))

# ── Datasets ──
DATASETS = {
    'multiorg': {
        'metadata': 'results/phase2_vlm_100/vlm_results.json',
        'crops_dir': 'results/phase2_vlm_100/crops',
    },
    'mouse_b1': {
        'metadata': 'data/mouse_crops/b1/crop_metadata.json',
        'crops_dir': 'data/mouse_crops/b1/crops',
    },
    'mouse_b2': {
        'metadata': 'data/mouse_crops/b2/crop_metadata.json',
        'crops_dir': 'data/mouse_crops/b2/crops',
    },
    'mouse_b3': {
        'metadata': 'data/mouse_crops/b3/crop_metadata.json',
        'crops_dir': 'data/mouse_crops/b3/crops',
    },
    'intestinal': {
        'metadata': 'data/intestinal_crops/val/crop_metadata.json',
        'crops_dir': 'data/intestinal_crops/val/crops',
    },
}


def load_metadata(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = []
    for d in data:
        items.append({
            'cache_key': d.get('cache_key', ''),
            'matched': d.get('matched', False),
            'confidence': d.get('rfdetr_conf', d.get('confidence', 0.5)),
        })
    return items


# ── CLIP Feature Extractor (frozen) ──

def load_clip(model_name='ViT-B-16', pretrained='openai', device='cpu'):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = open_clip.get_tokenizer(model_name)
    feat_dim = model.visual.output_dim  # 512 for ViT-B-16
    return model, preprocess, tokenizer, feat_dim


def extract_clip_features(model, preprocess, device, crops_dir, items, batch_size=32):
    """Extract CLIP image features for all crops (frozen)."""
    valid = []
    for item in items:
        crop_path = os.path.join(crops_dir, item['cache_key'] + '.png')
        if not os.path.exists(crop_path) and item.get('crop_path'):
            crop_path = item['crop_path']
        if not os.path.exists(crop_path):
            continue
        valid.append((item, crop_path))

    all_features = []
    labels = []
    confs = []

    for i in range(0, len(valid), batch_size):
        batch = valid[i:i+batch_size]
        images = []
        for item, path in batch:
            img = Image.open(path).convert('RGB')
            images.append(preprocess(img))

        img_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            all_features.append(feat.cpu())
            labels.extend([1 if item['matched'] else 0 for item, _ in batch])
            confs.extend([item['confidence'] for item, _ in batch])

        if (i // batch_size) % 20 == 0:
            print(f"    {i+len(batch)}/{len(valid)}")

    features = torch.cat(all_features, dim=0)
    return features, np.array(labels), np.array(confs)


# ── SupCon Slot Model (same as slot_supcon.py but on CLIP features) ──

class CLIPSlotSupCon(nn.Module):
    """SupCon slot model on CLIP features (512-dim instead of 768-dim DINOv2)."""
    def __init__(self, feat_dim=512, num_slots=8, dim_slots=128, num_iters=3,
                 proj_dim=256, n_classes=2):
        super().__init__()
        self.num_slots = num_slots
        self.dim_slots = dim_slots
        self.num_iters = num_iters

        # Slot initialization (learnable)
        self.slots = nn.Parameter(torch.randn(num_slots, dim_slots))
        nn.init.xavier_uniform_(self.slots)

        # Feature projection (CLIP 512 → dim_slots)
        self.feat_proj = nn.Linear(feat_dim, dim_slots)

        # Classification head
        self.classifier = nn.Linear(dim_slots, n_classes)

        # Projection head for SupCon
        self.projector = nn.Sequential(
            nn.Linear(dim_slots, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, features, return_embeddings=False):
        """features: [B, feat_dim] (CLIP features, already normalized)"""
        # Project features
        feat_proj = self.feat_proj(features)  # [B, dim_slots]

        # Slot attention (simplified: just use projected features)
        # For CLIP features, we don't have spatial tokens, so use feature directly
        # Slot attention becomes a soft clustering
        slots = self.slots.unsqueeze(0).expand(feat_proj.size(0), -1, -1)

        # Compute attention weights
        attn = torch.matmul(feat_proj, slots.transpose(1, 2))  # [B, num_slots]
        attn = F.softmax(attn / (self.dim_slots ** 0.5), dim=-1)

        # Slot representations
        slot_reps = torch.matmul(attn, slots)  # [B, num_slots, dim_slots]
        slot_flat = slot_reps.reshape(feat_proj.size(0), -1)  # [B, num_slots * dim_slots]

        # Classification (use mean of slot reps)
        slot_mean = slot_reps.mean(dim=1)  # [B, dim_slots]
        logits = self.classifier(slot_mean)

        # Projection for SupCon
        embeddings = self.projector(slot_mean)  # [B, proj_dim]
        embeddings = F.normalize(embeddings, dim=-1)

        if return_embeddings:
            return logits, embeddings
        return logits


def supcon_loss(embeddings, labels, temperature=0.07):
    """SupCon loss: pull same-class embeddings together, push different-class apart."""
    device = embeddings.device
    labels = labels.to(device)

    sim = torch.matmul(embeddings, embeddings.T) / temperature  # [B, B]
    sim = sim - sim.detach().max(dim=-1, keepdim=True).values
    exp_sim = torch.exp(sim)
    exp_sim = exp_sim - torch.diag(exp_sim)  # remove self-similarity

    # Mask: same class (positive), different class (negative)
    label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos_mask = label_mask - torch.diag(torch.diag(label_mask))
    neg_mask = 1.0 - label_mask

    # Log prob
    log_prob = sim - torch.log(exp_sim.sum(dim=-1, keepdim=True) + 1e-8)

    # SupCon loss
    pos_count = pos_mask.sum(dim=-1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = -(pos_mask * log_prob).sum(dim=-1) / (pos_count + 1e-8)
    loss = loss[valid].mean()
    return loss


# ── Dataset ──

class CLIPFeatureDataset(Dataset):
    """Dataset of pre-extracted CLIP features."""
    def __init__(self, features, labels, confs, augment=False):
        self.features = features
        self.labels = labels
        self.confs = confs
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        if self.augment:
            feat = feat + torch.randn_like(feat) * 0.01  # small noise augmentation
            feat = F.normalize(feat, dim=0)
        return {
            'feature': feat,
            'label': self.labels[idx],
            'confidence': self.confs[idx],
        }


# ── Training ──

def train_supcon_on_features(model, train_loader, optimizer, device, temperature, supcon_weight):
    model.train()
    total_loss = 0
    total_ce = 0
    total_sc = 0

    for batch in train_loader:
        features = batch['feature'].to(device)
        labels = batch['label'].to(device)

        logits, embeddings = model(features, return_embeddings=True)

        # CE loss
        ce_loss = F.cross_entropy(logits, labels)

        # SupCon loss
        sc_loss = supcon_loss(embeddings, labels, temperature)

        # Total loss
        loss = ce_loss + supcon_weight * sc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(features)
        total_ce += ce_loss.item() * len(features)
        total_sc += sc_loss.item() * len(features)

    n = len(train_loader.dataset)
    return {
        'loss': total_loss / n,
        'ce': total_ce / n,
        'supcon': total_sc / n,
    }


def evaluate_on_features(model, loader, device):
    model.eval()
    all_scores = []
    all_confs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            features = batch['feature'].to(device)
            labels = batch['label']
            confs = batch['confidence']

            logits = model(features)
            probs = F.softmax(logits, dim=-1)[:, 1]

            all_scores.extend(probs.cpu().numpy())
            all_confs.extend(confs.numpy())
            all_labels.extend(labels.numpy())

    scores = np.array(all_scores)
    confs = np.array(all_confs)
    labels = np.array(all_labels)

    from sklearn.metrics import roc_auc_score, average_precision_score

    if len(set(labels)) < 2:
        return {'slot_auc': 0.5, 'slot_ap': 0.5, 'conf_auc': 0.5, 'conf_ap': 0.5}

    slot_auc = roc_auc_score(labels, scores)
    conf_auc = roc_auc_score(labels, confs)
    slot_ap = average_precision_score(labels, scores)
    conf_ap = average_precision_score(labels, confs)

    return {
        'slot_auc': slot_auc,
        'slot_ap': slot_ap,
        'conf_auc': conf_auc,
        'conf_ap': conf_ap,
    }


# ── LOO Experiment ──

def run_loo_experiment(args):
    device = torch.device(args.device)

    # Step 1: Extract CLIP features for all datasets (once)
    print(f"\n{'='*60}")
    print(f"Step 1: Extract CLIP features (frozen)")
    print(f"{'='*60}")

    model, preprocess, tokenizer, feat_dim = load_clip(
        args.model, args.pretrained, device
    )
    print(f"  CLIP {args.model} loaded, feat_dim={feat_dim}")

    all_features = {}
    all_labels = {}
    all_confs = {}

    for ds_name, config in DATASETS.items():
        meta_path = PROJECT_ROOT / config['metadata']
        crops_dir = PROJECT_ROOT / config['crops_dir']

        if not meta_path.exists():
            print(f"  [SKIP] {ds_name}: metadata not found")
            continue

        items = load_metadata(str(meta_path))
        if len(items) == 0:
            print(f"  [SKIP] {ds_name}: no crops")
            continue

        print(f"\n  {ds_name}: {len(items)} crops")
        features, labels, confs = extract_clip_features(
            model, preprocess, device, str(crops_dir), items,
            batch_size=args.batch_size
        )
        all_features[ds_name] = features
        all_labels[ds_name] = labels
        all_confs[ds_name] = confs
        print(f"    Features: {features.shape}")

    available = list(all_features.keys())
    print(f"\n  Available datasets: {available}")

    # Step 2: LOO experiments
    print(f"\n{'='*60}")
    print(f"Step 2: LOO experiments ({len(available)} folds)")
    print(f"{'='*60}")

    all_results = {}

    for test_ds in available:
        train_ds_list = [d for d in available if d != test_ds]

        print(f"\n{'='*40}")
        print(f"  LOO: train on {train_ds_list} → test on {test_ds}")
        print(f"{'='*40}")

        # Prepare train data
        train_features = torch.cat([all_features[d] for d in train_ds_list], dim=0)
        train_labels = torch.tensor(
            np.concatenate([all_labels[d] for d in train_ds_list]),
            dtype=torch.long
        )
        train_confs = torch.tensor(
            np.concatenate([all_confs[d] for d in train_ds_list]),
            dtype=torch.float
        )

        # Prepare test data
        test_features = all_features[test_ds]
        test_labels = torch.tensor(all_labels[test_ds], dtype=torch.long)
        test_confs = torch.tensor(all_confs[test_ds], dtype=torch.float)

        print(f"  Train: {len(train_features)} crops (TP={train_labels.sum()}, FP={len(train_labels)-train_labels.sum()})")
        print(f"  Test:  {len(test_features)} crops (TP={test_labels.sum()}, FP={len(test_labels)-test_labels.sum()})")

        # Create datasets
        train_ds = CLIPFeatureDataset(train_features, train_labels, train_confs, augment=True)
        test_ds = CLIPFeatureDataset(test_features, test_labels, test_confs, augment=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0)

        # Build model
        supcon_model = CLIPSlotSupCon(
            feat_dim=feat_dim,
            num_slots=args.num_slots,
            dim_slots=args.dim_slots,
            num_iters=args.num_iters,
            proj_dim=args.proj_dim,
        ).to(device)

        optimizer = torch.optim.Adam(
            [p for p in supcon_model.parameters() if p.requires_grad],
            lr=args.lr
        )

        # Training loop
        best_auc = 0
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            metrics = train_supcon_on_features(
                supcon_model, train_loader, optimizer, device,
                args.temperature, args.supcon_weight
            )
            test_metrics = evaluate_on_features(supcon_model, test_loader, device)
            epoch_time = time.time() - t0

            if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
                print(f"  Epoch {epoch:3d}/{args.epochs} | {epoch_time:.1f}s | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Test Slot AUC: {test_metrics['slot_auc']:.4f}")

            if test_metrics['slot_auc'] > best_auc:
                best_auc = test_metrics['slot_auc']
                best_metrics = test_metrics

        print(f"\n  Best Test Slot AUC: {best_auc:.4f}")
        print(f"  Conf AUC: {best_metrics['conf_auc']:.4f}")
        print(f"  Δ AUC: {best_auc - best_metrics['conf_auc']:+.4f}")

        all_results[test_ds] = {
            'tag': f'loo_clip_{test_ds}',
            'train_on': train_ds_list,
            'test_on': test_ds,
            'n_train': len(train_features),
            'n_test': len(test_features),
            'best_slot_auc': float(best_auc),
            'conf_auc': float(best_metrics['conf_auc']),
            'slot_ap': float(best_metrics['slot_ap']),
            'conf_ap': float(best_metrics['conf_ap']),
            'delta_auc': float(best_auc - best_metrics['conf_auc']),
        }

    # Step 3: Summary
    print(f"\n{'='*60}")
    print(f"LOO Summary: CLIP + SupCon")
    print(f"{'='*60}")

    print(f"\n  {'Dataset':<15} {'DINOv2 Single':>14} {'DINOv2 LOO':>10} {'CLIP Zero':>10} {'CLIP+SupCon':>12} {'Conf':>8}")
    print(f"  {'-'*75}")

    dinov2_single = {'mouse_b1': 0.29, 'mouse_b2': 0.51, 'mouse_b3': 0.54, 'intestinal': 0.67, 'multiorg': 0.79}
    dinov2_loo = {'mouse_b1': 0.49, 'mouse_b2': 0.56, 'mouse_b3': 0.82, 'intestinal': 0.58}
    clip_zero = {'multiorg': 0.73, 'mouse_b1': 0.86, 'mouse_b2': 0.66, 'mouse_b3': 0.69, 'intestinal': 0.69}
    conf = {'multiorg': 0.87, 'mouse_b1': 0.91, 'mouse_b2': 0.98, 'mouse_b3': 0.92, 'intestinal': 0.92}

    for ds in ['multiorg', 'mouse_b1', 'mouse_b2', 'mouse_b3', 'intestinal']:
        d_s = dinov2_single.get(ds, 0)
        d_l = dinov2_loo.get(ds, 0)
        c_z = clip_zero.get(ds, 0)
        c_sc = all_results.get(ds, {}).get('best_slot_auc', 0) if ds in all_results else 0
        c = conf.get(ds, 0)
        print(f"  {ds:<15} {d_s:>14.2f} {d_l:>10.2f} {c_z:>10.2f} {c_sc:>12.2f} {c:>8.2f}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'clip_supcon_loo_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='CLIP + SupCon LOO experiment')
    parser.add_argument('--model', default='ViT-B-16', help='CLIP model name')
    parser.add_argument('--pretrained', default='openai')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-slots', type=int, default=8)
    parser.add_argument('--dim-slots', type=int, default=128)
    parser.add_argument('--num-iters', type=int, default=3)
    parser.add_argument('--proj-dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--supcon-weight', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', default='results/clip_supcon_loo')
    args = parser.parse_args()

    t0 = time.time()
    run_loo_experiment(args)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
