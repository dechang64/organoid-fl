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
    """MultiOrg SAM2 微调数据集
    
    每个 sample = (image_tensor, box_prompt, gt_mask)
    一个 image 可能有多个 instances，每个 instance 一个 sample
    """

    def __init__(self, manifest_path, image_size=1024):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.image_size = image_size
        self.samples = []

        for item in self.manifest:
            n = item['n_instances']
            for i in range(n):
                self.samples.append({
                    'image': item['image'],
                    'mask': item['mask'],
                    'inst_idx': i,  # 0-based
                    'bbox': item['instances'][i]['bbox'],  # [x1,y1,x2,y2] in original coords
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 加载图片
        img = cv2.imread(s['image'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 加载 instance mask
        inst_map = cv2.imread(s['mask'], cv2.IMREAD_UNCHANGED)
        mask = (inst_map == (s['inst_idx'] + 1)).astype(np.float32)

        # box prompt
        x1, y1, x2, y2 = s['bbox']
        box = np.array([x1, y1, x2, y2], dtype=np.float32)

        # Resize 到 image_size（SAM2 默认 1024）
        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        mask_resized = cv2.resize(mask, (self.image_size, self.image_size),
                                  interpolation=cv2.INTER_NEAREST)

        # box 坐标缩放
        scale_x = self.image_size / w
        scale_y = self.image_size / h
        box_resized = np.array([
            x1 * scale_x, y1 * scale_y,
            x2 * scale_x, y2 * scale_y
        ], dtype=np.float32)

        # To tensor（不 normalize，SAM2 transforms 会处理）
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_resized).float()  # (H, W)
        box_tensor = torch.from_numpy(box_resized)  # (4,)

        return img_tensor, box_tensor, mask_tensor


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
# Forward（用 SAM2ImagePredictor 接口，保持 mask_decoder 梯度）
# ============================================================

def forward_sam2(predictor, img_np, box_xyxy, orig_hw, device):
    """单 sample forward: image_encoder(no_grad) → mask_decoder(grad)
    
    用 predictor.set_image 跑 image_encoder（冻结，无梯度），
    然后直接调 predictor._predict 保持 mask_decoder 梯度。
    
    Args:
        predictor: SAM2ImagePredictor（已加载模型）
        img_np: (H, W, 3) uint8 RGB numpy array
        box_xyxy: (4,) tensor [x1,y1,x2,y2] in orig coords
        orig_hw: (H, W) 原始尺寸
        device: torch device
    
    Returns:
        pred_masks: (1, 1, H, W) logits on device（未阈值化）
    """
    # Step 1: set_image 跑 image_encoder（冻结参数，自动无梯度）
    predictor.set_image(img_np)
    
    # Step 2: 准备 box prompt（SAM2 格式: (B, 4) tensor）
    box = box_xyxy.reshape(1, 4).to(device).float()
    
    # Step 3: 直接调 _predict（不调 predict，避免 numpy 转换断梯度）
    # _predict 期望 box 已 transform 到 SAM2 内部坐标
    # predictor._transforms 会 normalize box 到 [0,1] 再缩放到 image_size
    # 但 _predict 假设输入已经是 transformed 的，所以我们手动 transform
    # box normalize: / orig_hw
    h, w = orig_hw
    box_normalized = box.clone()
    box_normalized[:, 0] = box[:, 0] / w  # x1
    box_normalized[:, 1] = box[:, 1] / h  # y1
    box_normalized[:, 2] = box[:, 2] / w  # x2
    box_normalized[:, 3] = box[:, 3] / h  # y2
    # 缩放到 SAM2 内部 resolution
    sam_size = predictor.model.image_size
    box_transformed = box_normalized * sam_size
    
    # 调 _predict（return_logits=True 保持梯度）
    masks, iou_predictions, low_res_masks = predictor._predict(
        point_coords=None,
        point_labels=None,
        boxes=box_transformed,
        mask_input=None,
        multimask_output=False,
        return_logits=True,
    )
    
    return masks  # (1, 1, H, W) logits



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

def train_one_epoch(predictor, model, dataloader, optimizer, device, epoch, accumulation_steps=4):
    """训练一个 epoch — 逐 sample forward，梯度累积"""
    model.train()
    model.image_encoder.eval()  # image_encoder 保持 eval
    
    total_loss = 0
    total_bce = 0
    total_dice = 0
    total_iou = 0
    n_samples = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (images, boxes, masks) in enumerate(dataloader):
        # batch_size=1，直接取第一个
        img_tensor = images[0]  # (3, H, W)
        box = boxes[0].to(device)  # (4,)
        gt_mask = masks.to(device)  # (1, H, W)
        orig_hw = img_tensor.shape[-2:]
        
        # Forward：用 predictor 接口
        # set_image 需要 numpy RGB HWC
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pred_masks = forward_sam2(predictor, img_np, box, orig_hw, device)
        # pred_masks: (1, 1, H, W) logits
        
        # Loss（除以 accumulation_steps 做梯度累积）
        loss, bce_val, dice_val = compute_loss(pred_masks, gt_mask)
        loss = loss / accumulation_steps
        
        loss.backward()
        
        # IoU（不参与梯度）
        with torch.no_grad():
            iou = compute_iou(pred_masks.detach(), gt_mask)
        
        total_loss += loss.item() * accumulation_steps
        total_bce += bce_val
        total_dice += dice_val
        total_iou += iou
        n_samples += 1
        
        # 梯度累积
        if n_samples % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()
        
        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"loss={loss.item()*accumulation_steps:.4f} iou={iou:.4f}", flush=True)
    
    # 处理剩余的梯度
    if n_samples % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0
        )
        optimizer.step()
        optimizer.zero_grad()
    
    return {
        'loss': total_loss / n_samples,
        'bce': total_bce / n_samples,
        'dice': total_dice / n_samples,
        'iou': total_iou / n_samples,
    }


@torch.no_grad()
def validate(predictor, model, dataloader, device):
    """验证"""
    model.eval()
    
    total_iou = 0
    total_dice = 0
    n = 0
    
    for images, boxes, masks in dataloader:
        img_tensor = images[0]
        box = boxes[0].to(device)
        gt_mask = masks.to(device)
        orig_hw = img_tensor.shape[-2:]
        
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pred_masks = forward_sam2(predictor, img_np, box, orig_hw, device)
        
        iou = compute_iou(pred_masks, gt_mask)
        pred_bin = (pred_masks > 0).float()
        gt = (gt_mask > 0.5).float()
        dice = 2 * (pred_bin * gt).sum() / (pred_bin.sum() + gt.sum() + 1e-8)
        
        total_iou += iou
        total_dice += dice.item()
        n += 1
    
    return {
        'iou': total_iou / max(n, 1),
        'dice': total_dice / max(n, 1),
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
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
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
    
    predictor = SAM2ImagePredictor(model)
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
        train_ds.samples = train_ds.samples[:args.max_train_samples]
    
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")

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
            predictor, model, train_loader, optimizer, device, epoch,
            args.accum_steps
        )
        val_metrics = validate(predictor, model, val_loader, device)

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
