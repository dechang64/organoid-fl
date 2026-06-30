#!/usr/bin/env python3
"""
Phase 3: EMASlideLoss 训练集成

通过自定义 DetectionModel 和 DetectionTrainer，
将 EMASlideLoss 替换标准 BCE cls loss。

用法:
    from train_with_ema_slide import train_with_ema_slide
    train_with_ema_slide(
        model_yaml='orga_dete_yolo11n.yaml',
        data_yaml='D:\\datasets\\MultiOrg_v4_640\\data.yaml',
        epochs=300, imgsz=640, batch=8
    )
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from orga_dete_modules import MPCA, EMASlideLoss
import ultralytics.nn.tasks as ul_tasks
import ultralytics.nn.modules as ul_modules
import ultralytics.utils.loss as ul_loss
from ultralytics.nn.tasks import v8DetectionLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG, RANK
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 注册 MPCA
# ============================================================
ul_tasks.MPCA = MPCA
ul_modules.MPCA = MPCA


# ============================================================
# 2. EMASlideLoss Detection Loss
# ============================================================

class EMASlideDetectionLoss(v8DetectionLoss):
    """替换 v8DetectionLoss 的 BCE cls loss 为 EMASlideLoss
    
    保留 box loss (CIoU) 和 dfl loss 不变，
    只替换 classification loss。
    """
    
    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.tal_topk = tal_topk
        self.tal_topk2 = tal_topk2
        self.ema_slide_loss = EMASlideLoss(alpha=0.9, reduction='none')
    
    def get_assigned_targets_and_loss(self, preds, batch):
        """重写 loss 计算，用 EMASlideLoss 替换 BCE cls loss
        
        复用 v8DetectionLoss 的原始逻辑，只替换 cls loss 部分。
        """
        from ultralytics.utils.tal import make_anchors, dist2bbox
        from ultralytics.utils.metrics import bbox_iou
        
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        
        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        
        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        
        # Assigner (复用 self.assigner)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        # === EMASlideLoss 替换 BCE ===
        # 计算每个 anchor 的 IoU（用于 EMASlideLoss 权重）
        iou_per_anchor = torch.zeros(batch_size, pred_scores.shape[1], device=self.device)
        if fg_mask.sum():
            fg_pred = pred_bboxes[fg_mask]  # (N_fg, 4) — normalized
            # stride_tensor (A,1) → broadcast to (B,A) → index by fg_mask (B,A)
            stride_broadcast = stride_tensor.squeeze(-1).unsqueeze(0).expand_as(fg_mask)
            fg_stride = stride_broadcast[fg_mask]  # (N_fg,)
            fg_target = target_bboxes[fg_mask] / fg_stride.unsqueeze(-1)  # (N_fg, 4)
            if fg_pred.numel() > 0:
                ious = bbox_iou(fg_pred, fg_target, xywh=False, CIoU=False).squeeze(-1)
                iou_per_anchor[fg_mask] = ious
        
        # EMASlideLoss
        cls_loss = self.ema_slide_loss(
            pred_scores, target_scores.to(dtype),
            iou=iou_per_anchor, fg_mask=fg_mask
        )
        loss[1] = cls_loss.sum() / target_scores_sum
        
        # Bbox loss (保持不变)
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor, target_scores, target_scores_sum,
                fg_mask, imgsz, stride_tensor
            )
        
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


# ============================================================
# 3. 自定义 DetectionModel
# ============================================================

class OrgaDeteModel(ul_tasks.DetectionModel):
    """使用 EMASlideLoss 的 DetectionModel"""
    
    def init_criterion(self):
        return EMASlideDetectionLoss(self)


# ============================================================
# 4. 自定义 Trainer
# ============================================================

class OrgaDeteTrainer(DetectionTrainer):
    """使用 OrgaDeteModel + EMASlideLoss 的 Trainer"""
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = OrgaDeteModel(
            cfg or self.args.model,
            ch=3,
            nc=self.data.get("nc", 1),
            verbose=verbose and RANK in {-1, 0},
        )
        if weights:
            model.load(weights)
        return model


# ============================================================
# 5. 训练入口
# ============================================================

def train_with_ema_slide(
    model_yaml='scripts/multiorg/orga_dete/orga_dete_yolo11n.yaml',
    data_yaml='D:\\datasets\\MultiOrg_v4_640\\data.yaml',
    epochs=300, imgsz=640, batch=8, device='0',
    project='runs/orga_dete', name='phase3_ema_slide',
    pretrained='yolo11n.pt'
):
    """用 EMASlideLoss 训练 Orga-Dete 模型
    
    Args:
        model_yaml: 模型 YAML 路径
        data_yaml: 数据集 YAML 路径
        epochs: 训练轮数
        imgsz: 输入尺寸
        batch: batch size
        device: GPU 设备
        project: 输出项目目录
        name: 实验名
        pretrained: 预训练权重
    """
    overrides = {
        'model': model_yaml,
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'name': name,
        'pretrained': pretrained,
        'cos_lr': True,
        'close_mosaic': 15,
        'patience': 50,
        'label_smoothing': 0.1,
        'copy_paste': 0.1,
        'mixup': 0.1,
    }
    
    trainer = OrgaDeteTrainer(overrides=overrides)
    trainer.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Orga-Dete with EMASlideLoss')
    parser.add_argument('--model', default='scripts/multiorg/orga_dete/orga_dete_yolo11n.yaml')
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--project', default='runs/orga_dete')
    parser.add_argument('--name', default='phase3_ema_slide')
    parser.add_argument('--pretrained', default='yolo11n.pt')
    
    args = parser.parse_args()
    train_with_ema_slide(
        model_yaml=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
    )
