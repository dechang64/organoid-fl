r"""
SAM2 Mask Decoder 微调 for MultiOrg（重写版）

策略：
- 冻结 image_encoder + memory_attention + memory_encoder
- 只解冻 sam_mask_decoder + sam_prompt_encoder
- 逐 sample forward（image_encoder no_grad，mask_decoder 有 grad）
- Loss: BCE + Dice
- 3060 12GB: batch_size=1（逐 sample 累积梯度），image_size=1024

数据格式（由 prepare_sam2_data.py 生成）:
    data/
      train/
        images/xxx.png
        masks/xxx.png  (uint16 instance map)
        manifest.json
      val/
        ...

Usage:
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate
    scripts\multiorg\run_sam2_finetune.bat
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Dataset
# ============================================================

class MultiOrgSAM2Dataset(Dataset):
    """MultiOrg SAM2 微调数据集（按 image 组织，每张图一次 image_encoder）
    
    每个 sample = 一张图的所有 instances
    __getitem__ 返回 (img_tensor, list of boxes, list of gt_masks)
    """

    def __init__(self, manifest_path, image_size=1024):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.image_size = image_size

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]

        # 加载图片
        img = cv2.imread(item['image'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 加载 instance mask
        inst_map = cv2.imread(item['mask'], cv2.IMREAD_UNCHANGED)

        # Resize 到 image_size
        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # 所有 instances 的 box 和 mask
        scale_x = self.image_size / w
        scale_y = self.image_size / h
        boxes = []
        masks = []
        for i, inst in enumerate(item['instances']):
            x1, y1, x2, y2 = inst['bbox']
            box = torch.tensor([
                x1 * scale_x, y1 * scale_y,
                x2 * scale_x, y2 * scale_y
            ], dtype=torch.float32)
            mask = (inst_map == (i + 1)).astype(np.float32)
            mask_resized = cv2.resize(mask, (self.image_size, self.image_size),
                                      interpolation=cv2.INTER_NEAREST)
            boxes.append(box)
            masks.append(torch.from_numpy(mask_resized))

        return img_tensor, boxes, masks


# ============================================================
# SAM2 Loader
# ============================================================

def freeze_backbone(model):
    """冻结 image_encoder + memory，只解冻 mask_decoder + prompt_encoder"""
    # 冻结 image_encoder
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # 冻结 memory_attention + memory_encoder
    if hasattr(model, 'memory_attention'):
        for param in model.memory_attention.parameters():
            param.requires_grad = False
    if hasattr(model, 'memory_encoder'):
        for param in model.memory_encoder.parameters():
            param.requires_grad = False

    # 解冻 mask_decoder
    for param in model.sam_mask_decoder.parameters():
        param.requires_grad = True

    # 解冻 prompt_encoder
    for param in model.sam_prompt_encoder.parameters():
        param.requires_grad = True

    # 统计
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total/1e6:.1f}M")
    print(f"  Trainable:    {trainable/1e6:.1f}M ({trainable/total*100:.1f}%)")
    print(f"  Frozen:       {(total-trainable)/1e6:.1f}M")


# ============================================================
# Forward（绕过 predictor 的 @torch.no_grad()，直接调 model 组件）
# ============================================================

def forward_sam2_image(model, img_tensor, image_size, device):
    """跑一次 image_encoder，返回 features（可被多个 box prompt 复用）
    
    Args:
        model: SAM2Base
        img_tensor: (1, 3, H, W) on device
        image_size: int
        device: torch device
    
    Returns:
        image_embed: (1, C, H', W')
        high_res_feats: list of (1, C, H'', W'')
    """
    with torch.no_grad():
        backbone_out = model.forward_image(img_tensor)
        _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)
        if model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]
        image_embed = feats[-1]
        high_res_feats = feats[:-1]
    return image_embed, high_res_feats


def forward_mask_decoder(model, image_embed, high_res_feats, box_tensor, device):
    """用 image features + box prompt 跑 mask_decoder（有梯度）
    
    Args:
        model: SAM2Base
        image_embed: (1, C, H', W') from forward_sam2_image
        high_res_feats: list from forward_sam2_image
        box_tensor: (4,) [x1,y1,x2,y2] in image_size coords
        device: torch device
    
    Returns:
        pred_masks: (1, 1, H, W) logits
    """
    box = box_tensor.reshape(1, 4)
    box_coords = box.reshape(-1, 2, 2)  # (1, 2, 2)
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)
    
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=(box_coords, box_labels),
        boxes=None,
        masks=None,
    )
    
    low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_feats,
    )
    
    pred_masks = F.interpolate(
        low_res_masks,
        size=(image_embed.shape[-2] * 16, image_embed.shape[-1] * 16),
        mode='bilinear',
        align_corners=False,
    )
    return pred_masks  # (1, 1, H, W) logits



# ============================================================
# Loss & Metrics
# ============================================================

def compute_loss(pred_logits, gt_mask):
    """BCE + Dice loss
    
    Args:
        pred_logits: (1, 1, H, W) or (H, W)
        gt_mask: (1, H, W) or (H, W) — 0/1
    """
    pred_logits = pred_logits.squeeze()
    gt = gt_mask.squeeze()
    
    # BCE
    bce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt)
    
    # Dice
    pred_prob = torch.sigmoid(pred_logits)
    intersection = (pred_prob * gt).sum()
    dice_loss = 1 - 2 * intersection / (pred_prob.sum() + gt.sum() + 1e-8)
    
    return bce_loss + dice_loss, bce_loss.item(), dice_loss.item()


def compute_iou(pred_logits, gt_mask, threshold=0.0):
    """计算 IoU"""
    pred_bin = (pred_logits > threshold).float()
    gt_bin = (gt_mask > 0.5).float()
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, dataloader, optimizer, device, epoch, image_size, accumulation_steps=4):
    """训练一个 epoch — 按 image 迭代，每张图一次 image_encoder + 多次 mask_decoder"""
    model.train()
    model.image_encoder.eval()
    
    total_loss = 0
    total_bce = 0
    total_dice = 0
    total_iou = 0
    n_instances = 0
    n_images = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (img_tensor, boxes, masks) in enumerate(dataloader):
        # DataLoader batch_size=1, 但 collate 会把 list 变成 nested structure
        # img_tensor: (1, 3, H, W), boxes: list of [list of tensors], masks: list of [list of tensors]
        img = img_tensor.to(device)  # (1, 3, H, W)
        # 取出第一个 sample 的 boxes 和 masks
        if isinstance(boxes[0], list):
            box_list = boxes[0]  # list of (4,) tensors
            mask_list = masks[0]  # list of (H, W) tensors
        else:
            box_list = [boxes[i] for i in range(len(boxes))]
            mask_list = [masks[i] for i in range(len(masks))]
        
        # Step 1: image_encoder 一次（no_grad）
        image_embed, high_res_feats = forward_sam2_image(model, img, image_size, device)
        
        # Step 2: 每个 instance 跑 mask_decoder（有梯度）
        for box, gt_mask in zip(box_list, mask_list):
            box = box.to(device)
            gt = gt_mask.to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            pred_masks = forward_mask_decoder(model, image_embed, high_res_feats, box, device)
            
            # 确保尺寸一致
            if pred_masks.shape[-2:] != gt.shape[-2:]:
                pred_masks = F.interpolate(pred_masks, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            
            loss, bce_val, dice_val = compute_loss(pred_masks, gt)
            loss = loss / accumulation_steps
            loss.backward()
            
            with torch.no_grad():
                iou = compute_iou(pred_masks.detach(), gt)
            
            total_loss += loss.item() * accumulation_steps
            total_bce += bce_val
            total_dice += dice_val
            total_iou += iou
            n_instances += 1
            
            if n_instances % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
        
        n_images += 1
        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"imgs={n_images} insts={n_instances} "
                  f"loss={loss.item()*accumulation_steps:.4f} iou={iou:.4f}", flush=True)
    
    if n_instances % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0
        )
        optimizer.step()
        optimizer.zero_grad()
    
    return {
        'loss': total_loss / max(n_instances, 1),
        'bce': total_bce / max(n_instances, 1),
        'dice': total_dice / max(n_instances, 1),
        'iou': total_iou / max(n_instances, 1),
        'n_images': n_images,
        'n_instances': n_instances,
    }


@torch.no_grad()
def validate(model, dataloader, device, image_size):
    """验证 — 按 image 迭代"""
    model.eval()
    
    total_iou = 0
    total_dice = 0
    n_instances = 0
    
    for img_tensor, boxes, masks in dataloader:
        img = img_tensor.to(device)
        if isinstance(boxes[0], list):
            box_list = boxes[0]
            mask_list = masks[0]
        else:
            box_list = [boxes[i] for i in range(len(boxes))]
            mask_list = [masks[i] for i in range(len(masks))]
        
        image_embed, high_res_feats = forward_sam2_image(model, img, image_size, device)
        
        for box, gt_mask in zip(box_list, mask_list):
            box = box.to(device)
            gt = gt_mask.to(device).unsqueeze(0).unsqueeze(0)
            
            pred_masks = forward_mask_decoder(model, image_embed, high_res_feats, box, device)
            if pred_masks.shape[-2:] != gt.shape[-2:]:
                pred_masks = F.interpolate(pred_masks, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            
            iou = compute_iou(pred_masks, gt)
            pred_bin = (pred_masks > 0).float()
            gt_bin = (gt > 0.5).float()
            dice = 2 * (pred_bin * gt_bin).sum() / (pred_bin.sum() + gt_bin.sum() + 1e-8)
            
            total_iou += iou
            total_dice += dice.item()
            n_instances += 1
    
    return {
        'iou': total_iou / max(n_instances, 1),
        'dice': total_dice / max(n_instances, 1),
        'n_instances': n_instances,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Finetune SAM2 mask decoder for MultiOrg')
    parser.add_argument('--data', required=True, help='Data directory (with train/ val/)')
    parser.add_argument('--checkpoint', required=True, help='SAM2 checkpoint path')
    parser.add_argument('--dst', required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=1, help='Must be 1 (per-sample forward)')
    parser.add_argument('--accum-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--max-train-samples', type=int, default=None, help='Limit train samples for testing')
    args = parser.parse_args()

    print("=" * 60)
    print("SAM2 Mask Decoder Finetune for MultiOrg")
    print("=" * 60)
    print(f"  Data: {args.data}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {args.dst}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")
    print(f"  Accum steps: {args.accum_steps}")
    print(f"  Image size: {args.image_size}")
    print(f"  Device: {args.device}")
    print("=" * 60)

    os.makedirs(args.dst, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print("\n[1/4] Loading SAM2...")
    from sam2.build_sam import build_sam2
    
    last_err = None
    model = None
    for cfg_name in ['sam2_hiera_s', 'sam2_hiera_small']:
        try:
            model = build_sam2(cfg_name, args.checkpoint, device=args.device)
            print(f"  Loaded with config '{cfg_name}'")
            break
        except Exception as e:
            last_err = e
            print(f"  config '{cfg_name}' failed: {e}")
    if model is None:
        raise RuntimeError(f"Failed to load SAM2: {last_err}")
    
    model = model.to(device)

    # 冻结 backbone
    print("\n[2/4] Freezing backbone...")
    freeze_backbone(model)

    # 数据
    print("\n[3/4] Loading data...")
    train_ds = MultiOrgSAM2Dataset(
        Path(args.data) / 'train' / 'manifest.json',
        image_size=args.image_size
    )
    val_ds = MultiOrgSAM2Dataset(
        Path(args.data) / 'val' / 'manifest.json',
        image_size=args.image_size
    )
    
    # 限制样本数（测试用）
    if args.max_train_samples and len(train_ds) > args.max_train_samples:
        train_ds.manifest = train_ds.manifest[:args.max_train_samples]

    print(f"  Train: {len(train_ds)} images")
    print(f"  Val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    print(f"\n[4/4] Training...")
    best_iou = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            args.image_size, args.accum_steps
        )
        val_metrics = validate(model, val_loader, device, args.image_size)

        scheduler.step()

        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch}/{args.epochs} ({elapsed:.0f}s)")
        print(f"    Train: loss={train_metrics['loss']:.4f} bce={train_metrics['bce']:.4f} "
              f"dice={train_metrics['dice']:.4f} iou={train_metrics['iou']:.4f}")
        print(f"    Val:   iou={val_metrics['iou']:.4f} dice={val_metrics['dice']:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_bce': train_metrics['bce'],
            'train_dice': train_metrics['dice'],
            'train_iou': train_metrics['iou'],
            'val_iou': val_metrics['iou'],
            'val_dice': val_metrics['dice'],
            'lr': optimizer.param_groups[0]['lr'],
            'time_s': elapsed,
        })

        # Save checkpoint（只存 trainable params）
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            ckpt_path = Path(args.dst) / 'sam2_finetuned.pt'
            torch.save({
                'trainable_state_dict': {k: v for k, v in model.state_dict().items() if 'sam_mask_decoder' in k or 'sam_prompt_encoder' in k},
                'model_state_dict': model.state_dict(),  # 全量，用于 build_sam2 加载
                'epoch': epoch,
                'val_iou': val_metrics['iou'],
            }, ckpt_path)
            print(f"    ★ Best! Saved to {ckpt_path}")

    # Save history
    with open(Path(args.dst) / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Best val IoU: {best_iou:.4f}")
    print(f"  Finetuned model: {Path(args.dst) / 'sam2_finetuned.pt'}")
    print(f"  History: {Path(args.dst) / 'training_history.json'}")
    print(f"\nNext: run multiorg_sam2.py with --sam2-checkpoint {Path(args.dst) / 'sam2_finetuned.pt'}")


if __name__ == '__main__':
    main()
