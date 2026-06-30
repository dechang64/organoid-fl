#!/usr/bin/env python3
r"""
Orga-Dete 训练脚本 — YOLOv11n + BiFPN + MPCA + EMASlideLoss

基于 Orga-Dete (Huang et al., Applied Sciences 2025) 的三模块迁移。
在标准 YOLOv11n 上加：
1. MPCA — 加在 C2PSA 后（backbone 末端）
2. BiFPN — 替换 head 的 FPN+PAN
3. EMASlideLoss — 替换 BCE cls loss

用法（Windows PowerShell）:
    cd C:\Users\decha\organoid-fl
    python scripts\multiorg\orga_dete\train_orga_dete.py --data D:\datasets\MultiOrg_v4_640\data.yaml --epochs 300 --imgsz 640 --batch 8

依赖：
    pip install ultralytics
"""

import os
import sys
import argparse
import copy

# 添加模块路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from orga_dete_modules import MPCA, BiFPN, EMASlideLoss


# ============================================================
# 1. Monkey-patch Ultralytics 识别自定义模块
# ============================================================

def patch_ultralytics():
    """让 Ultralytics 的 parse_model 识别 MPCA 和 BiFPN"""
    import ultralytics.nn.modules as ul_modules
    import ultralytics.nn.tasks as ul_tasks
    
    # 注册模块
    ul_modules.MPCA = MPCA
    ul_modules.BiFPN = BiFPN
    
    # 保存原始 parse_model
    original_parse_model = ul_tasks.parse_model
    
    def patched_parse_model(d, ch, verbose=True):
        """补丁版 parse_model，在 base_modules 中添加 MPCA"""
        # MPCA: 单次调用模块（类似 Conv）
        # BiFPN: 接收多输入（类似 Concat）
        
        # 临时修改 modules 命名空间
        # parse_model 内部通过 getattr(modules, m_name) 查找模块
        # 只要 MPCA 和 BiFPN 在 modules 命名空间里就能被找到
        
        # 关键：parse_model 用 base_modules 判断是否需要 ch[0] 参数
        # MPCA 需要 in_channels 参数，BiFPN 需要 list 输入
        # 我们通过修改 args 传递方式解决
        
        return original_parse_model(d, ch, verbose)
    
    # 不替换 parse_model——它在内部直接调用，patch 无效
    # 正确方式：MPCA 的 __init__ 接收和 C2PSA 类似的参数
    # BiFPN 的 __init__ 接收 list 输入
    
    # 已经通过 ul_modules.MPCA = MPCA 注册
    # parse_model 会通过 getattr(modules, 'MPCA') 找到它
    
    return ul_modules


# ============================================================
# 2. 自定义 YAML 配置
# ============================================================

ORGA_DETE_YAML = {
    'nc': 1,
    'scales': {
        'n': [0.50, 0.25, 1024],
    },
    'backbone': [
        [-1, 1, 'Conv', [64, 3, 2]],       # 0-P1/2
        [-1, 1, 'Conv', [128, 3, 2]],      # 1-P2/4
        [-1, 2, 'C3k2', [256, False]],      # 2
        [-1, 1, 'Conv', [256, 3, 2]],       # 3-P3/8
        [-1, 2, 'C3k2', [256, False]],      # 4 — P3 out
        [-1, 1, 'Conv', [512, 3, 2]],       # 5-P4/16
        [-1, 2, 'C3k2', [512, True]],       # 6 — P4 out
        [-1, 1, 'Conv', [1024, 3, 2]],      # 7-P5/32
        [-1, 2, 'C3k2', [1024, True]],      # 8
        [-1, 1, 'SPPF', [1024, 5]],         # 9
        [-1, 2, 'C2PSA', [1024]],           # 10
        [-1, 1, 'MPCA', [1024]],            # 11 — Orga-Dete MPCA
    ],
    'head': [
        # BiFPN 接收 P3(4), P4(6), P5(11) 三个特征图
        [[4, 6, 11], 1, 'BiFPN', [256]],    # 12 — BiFPN
        # BiFPN 返回 [P3', P4', P5']，但 Detect 需要 3 个独立输入
        # 用 Split 模块或直接传 list
        # Ultralytics Detect 接收 list 输入: [[12, 12, 12], 1, Detect, [nc]]
        # 但需要 BiFPN 的输出能被正确解析
        # 临时方案：BiFPN 输出只有 P3'，P4'/P5' 通过 Conv 下采样得到
        # 更好方案：修改 BiFPN 让它输出 tuple，Detect 直接用
        
        # 简化方案：用标准 FPN+PAN 结构，只在 backbone 加 MPCA
        # BiFPN 的实现较复杂，先验证 MPCA + EMASlideLoss 的效果
    ]
}


# ============================================================
# 3. 简化版 YAML — 只加 MPCA + EMASlideLoss（不加 BiFPN）
# ============================================================

SIMPLE_YAML = """\
# Orga-Dete Simple: YOLO11n + MPCA + EMASlideLoss (no BiFPN)
# 先验证 MPCA + EMASlideLoss 效果，BiFPN 后续添加

nc: 1

scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]   # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]    # 9
  - [-1, 2, C2PSA, [1024]]      # 10
  - [-1, 1, MPCA, [1024]]       # 11 — Orga-Dete MPCA

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5 (注意：用 10 而非 11，因为 MPCA 在 11)
  # 修正：MPCA 在 11，应该 cat MPCA 输出
  # 但 MPCA 不改变通道数，直接 cat [-1, 11]
  # 等等，这里 -1 是第 20 层（Conv），不是 10
  # 需要重新计算索引

  # 重新设计 head：
  # 12: Upsample
  # 13: Concat [-1, 6]
  # 14: C3k2
  # 15: Upsample
  # 16: Concat [-1, 4]
  # 17: C3k2 (P3/8-small)
  # 18: Conv
  # 19: Concat [-1, 14]
  # 20: C3k2 (P4/16-medium)
  # 21: Conv
  # 22: Concat [-1, 11]  # cat MPCA output
  # 23: C3k2 (P5/32-large)
  # 24: Detect [17, 20, 23]
"""


def write_simple_yaml(yaml_path):
    """写简化版 YAML（只加 MPCA）"""
    yaml_content = """\
# Orga-Dete Simple: YOLO11n + MPCA + EMASlideLoss
# Ref: Huang et al., Applied Sciences 2025

nc: 1

scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]   # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [256, False]] # 4 — P3 out
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]  # 6 — P4 out
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]] # 8
  - [-1, 1, SPPF, [1024, 5]]    # 9
  - [-1, 2, C2PSA, [1024]]      # 10
  - [-1, 1, MPCA, [1024]]       # 11 — Orga-Dete MPCA

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]           # 12
  - [[-1, 6], 1, Concat, [1]]                              # 13 cat P4
  - [-1, 2, C3k2, [512, False]]                            # 14

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]            # 15
  - [[-1, 4], 1, Concat, [1]]                              # 16 cat P3
  - [-1, 2, C3k2, [256, False]]                            # 17 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]                             # 18
  - [[-1, 14], 1, Concat, [1]]                             # 19 cat head P4
  - [-1, 2, C3k2, [512, False]]                            # 20 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]                             # 21
  - [[-1, 11], 1, Concat, [1]]                             # 22 cat MPCA output
  - [-1, 2, C3k2, [1024, True]]                            # 23 (P5/32-large)

  - [[17, 20, 23], 1, Detect, [nc]]                        # 24 Detect(P3, P4, P5)
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"✅ YAML written: {yaml_path}")


# ============================================================
# 4. 自定义 Loss — 替换 v8DetectionLoss 的 BCE 为 EMASlideLoss
# ============================================================

def patch_loss_with_ema_slide():
    """用 EMASlideLoss 替换 v8DetectionLoss 中的 BCE cls loss"""
    import ultralytics.utils.loss as ul_loss
    import torch
    
    original_get_loss = ul_loss.v8DetectionLoss.get_assigned_targets_and_loss
    
    # 给 v8DetectionLoss 实例添加 EMASlideLoss
    original_init = ul_loss.v8DetectionLoss.__init__
    
    def new_init(self, model, tal_topk=10, tal_topk2=None):
        original_init(self, model, tal_topk, tal_topk2)
        self.ema_slide_loss = EMASlideLoss(alpha=0.9, reduction='none')
    
    def new_get_loss(self, preds, batch):
        """替换 cls loss 计算为 EMASlideLoss"""
        import torch
        from ultralytics.utils.torch_utils import make_anchors, dist2bbox
        from ultralytics.utils.ops import xywh2xyxy
        
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        targets, pred_scores, pred_distri = (
            targets.to(dtype),
            pred_scores.to(dtype),
            pred_distri.to(dtype),
        )
        
        # Task-Aligned Assigner
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_distri.detach().exp() * self.proj.type(dtype)[None]).sum(-1),
            anchor_points * stride_tensor,
            targets,
            {"imgsz": imgsz, "device": self.device},
        )
        target_scores_sum = max(target_scores.sum(), 1)
        
        # === EMASlideLoss 替换 BCE ===
        # 计算 IoU 用于 EMASlideLoss
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        if fg_mask.sum():
            # 计算匹配 anchor 的 IoU
            assigned_pred = pred_bboxes[fg_mask]  # (N_fg, 4)
            assigned_gt = target_bboxes[fg_mask]  # (N_fg, 4)
            iou = self._compute_iou(assigned_pred, assigned_gt)  # (N_fg,)
            
            # 构建完整 iou tensor
            full_iou = torch.zeros(fg_mask.shape, device=self.device, dtype=dtype)
            full_iou[fg_mask] = iou
            
            # EMASlideLoss
            cls_loss = self.ema_slide_loss(pred_scores, target_scores.to(dtype), full_iou, fg_mask)
        else:
            cls_loss = self.bce(pred_scores, target_scores.to(dtype))
        
        loss[1] = cls_loss.sum() / target_scores_sum
        
        # Bbox loss (不变)
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor, target_scores, target_scores_sum,
                fg_mask, imgsz, stride_tensor,
            )
        
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )
    
    @staticmethod
    def _compute_iou(box1, box2):
        """计算 IoU (xyxy format)"""
        import torch
        inter = (torch.min(box1[:, 2], box2[:, 2]) - torch.max(box1[:, 0], box2[:, 0])).clamp(0) * \
                (torch.min(box1[:, 3], box2[:, 3]) - torch.max(box1[:, 1], box2[:, 1])).clamp(0)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = area1 + area2 - inter
        return inter / union.clamp(min=1e-6)
    
    # 应用 patch
    ul_loss.v8DetectionLoss.__init__ = new_init
    ul_loss.v8DetectionLoss.get_assigned_targets_and_loss = new_get_loss
    ul_loss.v8DetectionLoss._compute_iou = _compute_iou
    
    print("✅ v8DetectionLoss patched with EMASlideLoss")


# ============================================================
# 5. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Orga-Dete training: YOLOv11n + MPCA + EMASlideLoss')
    parser.add_argument('--data', required=True, help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--project', default='runs/orga_dete')
    parser.add_argument('--name', default='orga_dete_v1')
    parser.add_argument('--pretrained', default='yolo11n.pt', help='Pretrained weights')
    
    args = parser.parse_args()
    
    # 1. Patch Ultralytics
    patch_ultralytics()
    patch_loss_with_ema_slide()
    
    # 2. 写 YAML
    yaml_path = os.path.join(args.project, 'orga_dete_yolo11n.yaml')
    os.makedirs(args.project, exist_ok=True)
    write_simple_yaml(yaml_path)
    
    # 3. 训练
    from ultralytics import YOLO
    
    print(f"\n{'='*60}")
    print(f"Orga-Dete Training")
    print(f"{'='*60}")
    print(f"  Model: {yaml_path}")
    print(f"  Data: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Imgsz: {args.imgsz}")
    print(f"  Batch: {args.batch}")
    print(f"  Device: {args.device}")
    print(f"  Project: {args.project}/{args.name}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"{'='*60}\n")
    
    model = YOLO(yaml_path)
    model.load(args.pretrained)  # 加载预训练权重
    
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        cos_lr=True,
        close_mosaic=15,
        patience=50,
        label_smoothing=0.1,
        copy_paste=0.1,
        mixup=0.1,
    )


if __name__ == '__main__':
    main()
