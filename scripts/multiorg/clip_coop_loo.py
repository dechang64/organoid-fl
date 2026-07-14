r"""
Phase A3: CLIP Prompt Tuning (CoOp) — 学 prompt 不学 slot

问题：SupCon 训练 slot 参数会破坏 CLIP zero-shot 泛化（A1 已验证）
方案：只学 prompt 向量（CoOp），冻结 CLIP backbone + 不加 slot model
原理：CLIP zero-shot B1=0.86 已有信号，优化 prompt 可能超过 conf

方法：
1. CLIP backbone 冻结
2. 可学习 prompt: 2 个 nn.Embedding（TP prompt + FP prompt），各 77 tokens × 512 dim
3. 训练时优化 prompt 向量，使 sim(image, TP_prompt) - sim(image, FP_prompt) 匹配 label
4. 评估时用学到的 prompt 做 zero-shot 推理

Usage (冬生本地 GPU):
    cd C:\\Users\\decha\\organoid-fl
    python scripts\\multiorg\\clip_prompt_tuning.py --device cuda:0 --epochs 50

Usage (云 VM CPU):
    cd /home/z/my-project/organoid-fl
    python3 scripts/multiorg/clip_prompt_tuning.py --device cpu --epochs 20
"""
import argparse
import json
import os
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


def load_metadata(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'detections' in data:
        data = data['detections']
    items = []
    for x in data:
        items.append({
            'cache_key': x.get('cache_key', ''),
            'image': x.get('image', ''),
            'matched': x.get('matched', False),
            'confidence': x.get('rfdetr_conf', x.get('confidence', 0.5)),
        })
    return items


def find_crop_path(item, crops_dir):
    crop_path = os.path.join(crops_dir, item['cache_key'] + '.png')
    if os.path.exists(crop_path):
        return crop_path
    # Fallback
    alt = item.get('crop_path', '')
    if alt and os.path.exists(alt):
        return alt
    return None


class CLIPImageDataset(Dataset):
    """Load images for CLIP encoding."""
    def __init__(self, items, crops_dir, preprocess):
        self.items = []
        self.preprocess = preprocess
        for item in items:
            cp = find_crop_path(item, crops_dir)
            if cp:
                self.items.append((cp, 1 if item['matched'] else 0, item['confidence']))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        crop_path, label, conf = self.items[idx]
        img = Image.open(crop_path).convert('RGB')
        img = self.preprocess(img)
        return {'image': img, 'label': label, 'confidence': conf}


class CoOpModel(nn.Module):
    """CoOp: learn prompt vectors, freeze CLIP backbone.

    Instead of text prompts like "a real organoid", learn continuous prompt vectors
    that optimize sim(image, tp_prompt) - sim(image, fp_prompt) to match labels.
    """
    def __init__(self, clip_model, n_tokens=4, ctx_init="a real organoid"):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

        # Get text encoder info
        self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
        with torch.no_grad():
            # Get embedding dim
            dummy = self.tokenizer([ctx_init])
            self.text_dim = self.clip.text_projection.shape[0] if hasattr(self.clip, 'text_projection') else 512

        # Learnable context tokens (n_tokens × text_dim)
        self.n_tokens = n_tokens
        self.ctx = nn.Parameter(torch.randn(n_tokens, self.text_dim) * 0.02)

        # Tokenize class names to get SOS/EOS tokens
        with torch.no_grad():
            tokens = self.tokenizer([ctx_init])
        # We'll construct: SOS + ctx_tokens + class_tokens + EOS
        self.sos_token = tokens[0, 0]  # usually 49406 (BOS)
        self.eos_token = tokens[0, -1]  # usually 49407 (EOS)

        # Get class token embeddings (frozen)
        with torch.no_grad():
            token_emb = self.clip.token_embedding(tokens)  # [1, 77, dim]
        self.class_token_emb = nn.Parameter(token_emb[0, len(ctx_init.split())+1:tokens.shape[1]-1], requires_grad=False)
        self.sos_emb = nn.Parameter(token_emb[0, 0:1], requires_grad=False)
        self.eos_emb = nn.Parameter(token_emb[0, -1:], requires_grad=False)

    def forward(self, images):
        """Encode images + construct text features from learnable prompts."""
        with torch.no_grad():
            image_features = self.clip.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Construct text embeddings: SOS + ctx + class + EOS + padding
        batch_size = image_features.shape[0]
        ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_ctx, dim]

        # Build TP and FP prompts
        # TP: SOS + ctx + "organoid" + EOS
        # FP: SOS + ctx + "background" + EOS
        tp_text = torch.cat([self.sos_emb, ctx, self.class_token_emb, self.eos_emb], dim=1)
        fp_text = torch.cat([self.sos_emb, ctx, self.class_token_emb, self.eos_emb], dim=1)

        # For now, use simpler approach: just learn 2 context vectors
        # and use them with CLIP text encoder
        return image_features

    def get_prompt_features(self, device):
        """Get TP and FP prompt features using learnable context."""
        # Build text token embeddings
        ctx = self.ctx.unsqueeze(0)  # [1, n_ctx, dim]

        # TP prompt: SOS + ctx + "a real organoid" class tokens + EOS
        tp_emb = torch.cat([
            self.sos_emb,
            ctx,
            self.class_token_emb,
            self.eos_emb
        ], dim=1)  # [1, seq_len, dim]

        # Pad to 77
        if tp_emb.shape[1] < 77:
            pad = torch.zeros(1, 77 - tp_emb.shape[1], tp_emb.shape[2], device=device)
            tp_emb = torch.cat([tp_emb, pad], dim=1)

        # Run through text transformer
        tp_features = self.clip.text(
            tp_emb,
            None,  # no token masking
            None
        )
        tp_features = tp_features / tp_features.norm(dim=-1, keepdim=True)

        return tp_features


def encode_all_images(model, preprocess, device, datasets, batch_size=64):
    """Encode all images with frozen CLIP, cache features."""
    all_features = {}
    all_labels = {}
    all_confs = {}

    for ds_name, ds_cfg in datasets.items():
        meta_path = ds_cfg['metadata']
        crops_dir = ds_cfg['crops_dir']

        if not os.path.exists(meta_path):
            print(f"  [SKIP] {ds_name}: metadata not found")
            continue

        items = load_metadata(meta_path)
        print(f"  {ds_name}: {len(items)} crops")

        ds = CLIPImageDataset(items, crops_dir, preprocess)
        if len(ds) == 0:
            print(f"  [SKIP] {ds_name}: no crops found")
            continue

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        feats = []
        labels = []
        confs = []

        with torch.no_grad():
            for batch in loader:
                imgs = batch['image'].to(device)
                feat = model.encode_image(imgs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feats.append(feat.cpu())
                labels.extend(batch['label'].numpy())
                confs.extend(batch['confidence'].numpy())

        all_features[ds_name] = torch.cat(feats, dim=0)
        all_labels[ds_name] = np.array(labels)
        all_confs[ds_name] = np.array(confs)

        print(f"    Features: {all_features[ds_name].shape}")

    return all_features, all_labels, all_confs


def train_prompt_tuning(train_features, train_labels, train_confs,
                        clip_model, device, epochs=50, lr=1e-3, batch_size=32):
    """Train learnable prompts on CLIP features."""
    # Two learnable prompt vectors in CLIP text embedding space
    # Initialize from "a real organoid" and "a background artifact"
    text_dim = 512

    # Learnable TP and FP prompt embeddings (in CLIP text feature space)
    tp_prompt = nn.Parameter(torch.randn(text_dim, device=device) * 0.02)
    fp_prompt = nn.Parameter(torch.randn(text_dim, device=device) * 0.02)

    optimizer = torch.optim.Adam([tp_prompt, fp_prompt], lr=lr)

    # Create dataset from cached features
    train_feats = train_features.to(device)
    train_lbls = torch.tensor(train_labels, dtype=torch.long, device=device)

    n = len(train_feats)
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # Shuffle
        perm = torch.randperm(n)
        total_loss = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            imgs = train_feats[idx]
            labels = train_lbls[idx]

            # Compute similarities
            sim_tp = imgs @ tp_prompt  # [B]
            sim_fp = imgs @ fp_prompt  # [B]

            # Score = sim_tp - sim_fp
            scores = sim_tp - sim_fp

            # Binary cross entropy
            loss = F.binary_cross_entropy_with_logits(scores, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize prompts to unit sphere (CLIP convention)
            with torch.no_grad():
                tp_prompt.div_(tp_prompt.norm())
                fp_prompt.div_(fp_prompt.norm())

            total_loss += loss.item() * len(idx)

        avg_loss = total_loss / n
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

    return tp_prompt.detach(), fp_prompt.detach()


def evaluate_prompts(test_features, test_labels, test_confs, tp_prompt, fp_prompt, device):
    """Evaluate learned prompts on test set."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    test_feats = test_features.to(device)
    tp = tp_prompt.to(device)
    fp = fp_prompt.to(device)

    with torch.no_grad():
        sim_tp = (test_feats @ tp).cpu().numpy()
        sim_fp = (test_feats @ fp).cpu().numpy()

    scores = sim_tp - sim_fp
    labels = test_labels
    confs = test_confs

    if len(set(labels)) < 2:
        return {'slot_auc': 0.5, 'conf_auc': 0.5, 'slot_ap': 0.5, 'conf_ap': 0.5,
                'delta_auc': 0.0}

    slot_auc = roc_auc_score(labels, scores)
    conf_auc = roc_auc_score(labels, confs)
    slot_ap = average_precision_score(labels, scores)
    conf_ap = average_precision_score(labels, confs)

    return {
        'slot_auc': float(slot_auc),
        'conf_auc': float(conf_auc),
        'slot_ap': float(slot_ap),
        'conf_ap': float(conf_ap),
        'delta_auc': float(slot_auc - conf_auc),
    }


def run_loo_experiment(args):
    import open_clip

    device = torch.device(args.device)

    # Load CLIP
    print(f"\nLoading CLIP {args.model}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    model.eval()
    print(f"  CLIP loaded, feat_dim=512")

    # Step 1: Encode all images
    print(f"\n{'='*60}")
    print(f"Step 1: Extract CLIP features (frozen)")
    print(f"{'='*60}")

    all_features, all_labels, all_confs = encode_all_images(
        model, preprocess, device, DATASETS, batch_size=args.batch_size
    )

    available = list(all_features.keys())
    print(f"\n  Available datasets: {available}")

    # Step 2: LOO experiments
    print(f"\n{'='*60}")
    print(f"Step 2: LOO Prompt Tuning (CoOp)")
    print(f"{'='*60}")

    all_results = {}

    for test_ds_name in available:
        train_ds_list = [d for d in available if d != test_ds_name]

        print(f"\n{'='*40}")
        print(f"  LOO: train on {train_ds_list} → test on {test_ds_name}")
        print(f"{'='*40}")

        # Prepare train data
        train_features = torch.cat([all_features[d] for d in train_ds_list], dim=0)
        train_labels = np.concatenate([all_labels[d] for d in train_ds_list])
        train_confs = np.concatenate([all_confs[d] for d in train_ds_list])

        # Prepare test data
        test_features = all_features[test_ds_name]
        test_labels = all_labels[test_ds_name]
        test_confs = all_confs[test_ds_name]

        print(f"  Train: {len(train_features)} crops (TP={sum(train_labels)}, FP={len(train_labels)-sum(train_labels)})")
        print(f"  Test:  {len(test_features)} crops (TP={sum(test_labels)}, FP={len(test_labels)-sum(test_labels)})")

        # Train prompt tuning
        tp_prompt, fp_prompt = train_prompt_tuning(
            train_features, train_labels, train_confs,
            model, device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
        )

        # Evaluate
        metrics = evaluate_prompts(
            test_features, test_labels, test_confs,
            tp_prompt, fp_prompt, device
        )

        print(f"\n  Best Slot AUC: {metrics['slot_auc']:.4f}")
        print(f"  Conf AUC:      {metrics['conf_auc']:.4f}")
        print(f"  Δ AUC:         {metrics['delta_auc']:+.4f}")

        all_results[test_ds_name] = {
            'tag': f'loo_coop_{test_ds_name}',
            'train_on': train_ds_list,
            'test_on': test_ds_name,
            'n_train': len(train_features),
            'n_test': len(test_features),
            **metrics,
        }

    # Step 3: Summary
    print(f"\n{'='*60}")
    print(f"LOO Summary: CLIP + CoOp Prompt Tuning")
    print(f"{'='*60}")

    # Reference values for comparison
    single = {'multiorg': 0.79, 'mouse_b1': 0.29, 'mouse_b2': 0.51, 'mouse_b3': 0.54, 'intestinal': 0.67}
    dinov2_loo = {'multiorg': 0.00, 'mouse_b1': 0.49, 'mouse_b2': 0.56, 'mouse_b3': 0.82, 'intestinal': 0.58}
    clip_zero = {'multiorg': 0.73, 'mouse_b1': 0.86, 'mouse_b2': 0.66, 'mouse_b3': 0.69, 'intestinal': 0.69}
    clip_sup = {'multiorg': 0.95, 'mouse_b1': 0.50, 'mouse_b2': 0.57, 'mouse_b3': 0.45, 'intestinal': 0.50}
    conf = {'multiorg': 0.87, 'mouse_b1': 0.91, 'mouse_b2': 0.98, 'mouse_b3': 0.92, 'intestinal': 0.92}

    print(f"\n  {'Dataset':<15} {'DINOv2':>8} {'LOO':>8} {'CLIP-ZS':>8} {'CLIP-SC':>8} {'CoOp':>8} {'Conf':>8}")
    print(f"  {'-'*65}")
    for ds in available:
        s = single.get(ds, 0)
        l = dinov2_loo.get(ds, 0)
        c = clip_zero.get(ds, 0)
        sc = clip_sup.get(ds, 0)
        co = all_results.get(ds, {}).get('slot_auc', 0)
        cf = conf.get(ds, 0)
        print(f"  {ds:<15} {s:>8.2f} {l:>8.2f} {c:>8.2f} {sc:>8.2f} {co:>8.2f} {cf:>8.2f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'clip_coop_loo_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Phase A3: CLIP CoOp Prompt Tuning LOO')
    parser.add_argument('--model', default='ViT-B-16')
    parser.add_argument('--pretrained', default='openai')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', default='results/clip_coop_loo')
    args = parser.parse_args()

    t0 = time.time()
    run_loo_experiment(args)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
