r"""
SAM2 Mask Decoder 微调 for MultiOrg

策略：
- 冻结 image_encoder + memory_attention + memory_encoder
- 只解冻 sam_mask_decoder + sam_prompt_encoder
- Forward: image → image_encoder (no_grad) → features; box prompt → mask_decoder → mask
- Loss: BCE + Dice
- 3060 12GB: batch_size=4, image_size=1024 (SAM2 默认)

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

    python scripts\multiorg\finetune_sam2.py ^
        --data data\multiorg_sam2 ^
        --checkpoint sam2_checkpoints\sam2_hiera_small.pt ^
        --dst runs\sam2_finetune ^
        --epochs 5 --lr 1e-5 --batch-size 2

    # 多行在 PowerShell 不行，用 .bat:
    scripts\multiorg\run_sam2_finetune.bat
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
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
        
        # To tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        # Normalize (ImageNet stats, SAM2 内部会处理，但这里简单 normalize)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        mask_tensor = torch.from_numpy(mask_resized).float()  # (H, W)
        box_tensor = torch.from_numpy(box_resized)  # (4,)
        
        return img_tensor, box_tensor, mask_tensor


# ============================================================
# SAM2 Loader
# ============================================================

def load_sam2_model(checkpoint_path, device='cuda'):
    """加载 SAM2 模型，返回 model 和 predictor"""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    last_err = None
    for cfg in ['sam2_hiera_s', 'sam2_hiera_small']:
        try:
            model = build_sam2(cfg, checkpoint_path, device=device)
            predictor = SAM2ImagePredictor(model)
            return model, predictor
        except Exception as e:
            last_err = e
            print(f"  [SAM2] config '{cfg}' failed: {e}")
            continue
    raise RuntimeError(f"Could not load SAM2: {last_err}")


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
# Training
# ============================================================

def compute_iou(pred_mask, gt_mask, threshold=0.0):
    """计算 IoU"""
    pred_bin = (pred_mask > threshold).float()
    gt_bin = (gt_mask > 0.5).float()
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def train_one_epoch(model, predictor, dataloader, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    # image_encoder 保持 eval 模式（no_grad）
    model.image_encoder.eval()
    
    total_loss = 0
    total_bce = 0
    total_dice = 0
    total_iou = 0
    n_batches = 0
    
    for batch_idx, (images, boxes, masks) in enumerate(dataloader):
        images = images.to(device)  # (B, 3, H, W)
        boxes = boxes.to(device)    # (B, 4)
        masks = masks.to(device)    # (B, H, W)
        
        optimizer.zero_grad()
        
        batch_loss = 0
        batch_iou = 0
        
        for i in range(images.size(0)):
            img = images[i:i+1]  # (1, 3, H, W)
            box = boxes[i:i+1]   # (1, 4)
            gt_mask = masks[i:i+1]  # (1, H, W)
            
            # Forward: image_encoder (no_grad)
            with torch.no_grad():
                predictor.set_image(img.cpu().numpy()[0])  # SAM2 内部处理
                # 实际上 predictor.set_image 会调 image_encoder
                # 我们需要确保它用 no_grad
            
            # Box prompt → mask_decoder
            # predictor._predict 内部调 mask_decoder
            # 我们需要 return_logits=True 来拿 logits 做 loss
            with torch.enable_grad():
                # 准备 box prompt
                box_t = box.reshape(-1, 2, 2)  # (1, 2, 2) - two points
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)
                box_labels = box_labels.repeat(box.size(0), 1)
                
                concat_points = (box_t, box_labels)
                
                sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                    points=concat_points,
                    boxes=None,
                    masks=None,
                )
                
                high_res_features = [
                    feat_level[i].unsqueeze(0)
                    for feat_level in predictor._features["high_res_feats"]
                ]
                
                low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][i].unsqueeze(0),
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,  # 单 mask
                    repeat_image=False,
                    high_res_features=high_res_features,
                )
                
                # Upscale to original resolution
                pred_masks = predictor._transforms.postprocess_masks(
                    low_res_masks, predictor._orig_hw[i]
                )  # (1, 1, H, W)
                
                # Loss: BCE + Dice
                pred_logits = pred_masks.squeeze()  # (H, W)
                gt = gt_mask.squeeze()  # (H, W)
                
                # BCE
                bce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt)
                
                # Dice
                pred_prob = torch.sigmoid(pred_logits)
                intersection = (pred_prob * gt).sum()
                dice_loss = 1 - 2 * intersection / (pred_prob.sum() + gt.sum() + 1e-8)
                
                loss = bce_loss + dice_loss
                
                batch_loss += loss
                batch_iou += compute_iou(pred_logits.detach(), gt)
            
        batch_loss = batch_loss / images.size(0)
        batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 
            max_norm=1.0
        )
        
        optimizer.step()
        
        total_loss += batch_loss.item()
        total_iou += batch_iou / images.size(0)
        n_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"loss={batch_loss.item():.4f} iou={batch_iou/images.size(0):.4f}")
    
    return {
        'loss': total_loss / n_batches,
        'iou': total_iou / n_batches,
    }


@torch.no_grad()
def validate(model, predictor, dataloader, device):
    """验证"""
    model.eval()
    
    total_iou = 0
    total_dice = 0
    n = 0
    
    for images, boxes, masks in dataloader:
        for i in range(images.size(0)):
            img = images[i:i+1].to(device)
            box = boxes[i:i+1].to(device)
            gt_mask = masks[i:i+1].to(device)
            
            predictor.set_image(img.cpu().numpy()[0])
            
            box_t = box.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)
            box_labels = box_labels.repeat(box.size(0), 1)
            concat_points = (box_t, box_labels)
            
            sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                points=concat_points, boxes=None, masks=None,
            )
            high_res_features = [
                feat_level[i].unsqueeze(0)
                for feat_level in predictor._features["high_res_feats"]
            ]
            low_res_masks, _, _, _ = model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][i].unsqueeze(0),
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            pred_masks = predictor._transforms.postprocess_masks(
                low_res_masks, predictor._orig_hw[i]
            )
            
            pred_bin = (pred_masks.squeeze() > 0).float()
            gt = gt_mask.squeeze()
            
            iou = compute_iou(pred_bin, gt)
            dice = 2 * (pred_bin * gt).sum() / (pred_bin.sum() + gt.sum() + 1e-8)
            
            total_iou += iou
            total_dice += dice.item()
            n += 1
    
    return {
        'iou': total_iou / n,
        'dice': total_dice / n,
    }


def main():
    parser = argparse.ArgumentParser(description='Finetune SAM2 mask decoder for MultiOrg')
    parser.add_argument('--data', required=True, help='Data directory (with train/ val/)')
    parser.add_argument('--checkpoint', required=True, help='SAM2 checkpoint path')
    parser.add_argument('--dst', required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--save-every', type=int, default=1, help='Save checkpoint every N epochs')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAM2 Mask Decoder Finetune for MultiOrg")
    print("=" * 60)
    print(f"  Data: {args.data}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {args.dst}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"  Image size: {args.image_size}")
    print(f"  Device: {args.device}")
    print("=" * 60)
    
    os.makedirs(args.dst, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print("\n[1/4] Loading SAM2...")
    model, predictor = load_sam2_model(args.checkpoint, device=device.device.type)
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
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
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
        
        train_metrics = train_one_epoch(model, predictor, train_loader, optimizer, device, epoch)
        val_metrics = validate(model, predictor, val_loader, device)
        
        scheduler.step()
        
        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch}/{args.epochs} ({elapsed:.0f}s)")
        print(f"    Train: loss={train_metrics['loss']:.4f} iou={train_metrics['iou']:.4f}")
        print(f"    Val:   iou={val_metrics['iou']:.4f} dice={val_metrics['dice']:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_iou': train_metrics['iou'],
            'val_iou': val_metrics['iou'],
            'val_dice': val_metrics['dice'],
            'lr': optimizer.param_groups[0]['lr'],
        })
        
        # Save checkpoint
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            ckpt_path = Path(args.dst) / 'sam2_finetuned.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_iou': val_metrics['iou'],
            }, ckpt_path)
            print(f"    ★ Best! Saved to {ckpt_path}")
        
        if epoch % args.save_every == 0:
            ckpt_path = Path(args.dst) / f'sam2_epoch{epoch}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
            }, ckpt_path)
    
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
