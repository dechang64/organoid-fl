"""
Orga-Dete 三模块实现 —迁移自 Huang et al., Applied Sciences 2025

三个模块:
1. EMASlideLoss — 动态阈值校准分类 loss（零额外参数）
2. BiFPN — 双向特征金字塔网络（替换 FPN+PAN neck）
3. MPCA — 多路径坐标注意力（加在 C2PSA 之后）

参考文献:
- Huang et al., "Orga-Dete: An Improved Lightweight Deep Learning Model for Lung Organoid Detection", Applied Sciences 2025
- Tan et al., "EfficientDet: Scalable and Efficient Object Detection", CVPR 2020 (BiFPN)
- Hou et al., "Coordinate Attention", CVPR 2021 (CA → MPCA 基础)

用法:
    from orga_dete_modules import EMASlideLoss, BiFPN, MPCA
    
    # EMASlideLoss: 替换 v8DetectionLoss 中的 BCE cls loss
    # BiFPN: 在 YAML 中替换 head 的 FPN+PAN 结构
    # MPCA: 在 YAML 中加在 C2PSA 之后
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ============================================================================
# 1. EMASlideLoss — 动态阈值校准分类 loss
# ============================================================================

class EMASlideLoss(nn.Module):
    """EMASlideLoss — Orga-Dete 论文 §2.2.3
    
    用 EMA 平滑的 IoU 均值作为动态阈值 μ，对样本加权:
    - IoU < μ-0.1: hard sample, weight=1.0
    - μ-0.1 ≤ IoU < μ: transitional, weight=e^(1-μ)
    - IoU ≥ μ: easy sample, weight=e^(1-IoU)（IoU越高权重越低）
    
    公式 (Orga-Dete 论文 Eq.4):
        f(x) = 1,                    x ≤ μ - 0.1
        f(x) = e^(1-μ),              μ - 0.1 < x < μ  
        f(x) = e^(1-x),              x ≥ μ
    
    其中 μ = α·μ_prev + (1-α)·mean(IoU_batch)，α=0.9（平滑因子）
    
    实现: 替换 v8DetectionLoss 中的 BCE cls loss，对每个 anchor 的 cls loss
    按 IoU 加权。
    """
    
    def __init__(self, alpha: float = 0.9, reduction: str = 'none'):
        super().__init__()
        self.alpha = alpha  # EMA 平滑因子
        self.reduction = reduction
        # EMA 状态，初始 0.5（中等难度）
        self.register_buffer('ema_mu', torch.tensor(0.5))
    
    def update_mu(self, iou_mean: torch.Tensor):
        """更新 EMA 阈值
        
        Args:
            iou_mean: 当前 batch 的平均 IoU（标量 tensor）
        """
        with torch.no_grad():
            self.ema_mu = self.alpha * self.ema_mu + (1 - self.alpha) * float(iou_mean)
    
    def compute_weights(self, iou: torch.Tensor) -> torch.Tensor:
        """根据 IoU 和动态阈值 μ 计算每个样本的权重
        
        Args:
            iou: (N,) 每个样本的 IoU 值
            
        Returns:
            weights: (N,) 每个样本的权重
        """
        mu = self.ema_mu
        weights = torch.where(
            iou <= mu - 0.1,
            torch.ones_like(iou),  # hard: weight=1
            torch.where(
                iou < mu,
                torch.exp(torch.ones_like(iou) * (1 - mu)),  # transitional
                torch.exp(1 - iou)  # easy: weight=e^(1-IoU)
            )
        )
        return weights
    
    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor,
                iou: Optional[torch.Tensor] = None,
                fg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算 EMASlideLoss
        
        Args:
            pred_scores: (B, A, NC) 预测分数 (logits)
            target_scores: (B, A, NC) 目标分数 (soft labels from assigner)
            iou: (B, A) 每个 anchor 的 IoU（预测框 vs 匹配的 GT）
            fg_mask: (B, A) 前景 mask（匹配到 GT 的 anchor）
            
        Returns:
            loss: (B, A, NC) 逐元素 weighted loss
        """
        # BCE loss per element: (B, A, NC)
        bce = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction='none')
        
        if iou is not None and fg_mask is not None and fg_mask.any():
            # 更新 EMA
            fg_iou = iou[fg_mask]  # (N_fg,)
            if fg_iou.numel() > 0:
                self.update_mu(fg_iou.mean())
            
            # 计算每个 anchor 的权重: (B, A)
            mu = float(self.ema_mu)
            # 默认权重=1（背景样本和未匹配 anchor）
            anchor_weights = torch.ones_like(iou)
            # 前景样本按 EMASlideLoss 公式加权
            anchor_weights = torch.where(
                fg_mask,
                torch.where(
                    iou <= mu - 0.1,
                    torch.ones_like(iou),
                    torch.where(
                        iou < mu,
                        torch.exp(torch.full_like(iou, 1.0 - mu)),
                        torch.exp(1.0 - iou)
                    )
                ),
                torch.ones_like(iou)
            )
            # 扩展到 (B, A, NC) 用于 element-wise 乘法
            weights = anchor_weights.unsqueeze(-1).expand_as(bce)
            weighted_loss = bce * weights
        else:
            # 没有 IoU 信息时退化为普通 BCE
            weighted_loss = bce
        
        if self.reduction == 'sum':
            return weighted_loss.sum()
        elif self.reduction == 'mean':
            return weighted_loss.mean()
        else:
            return weighted_loss


# ============================================================================
# 2. BiFPN — 双向特征金字塔网络
# ============================================================================

class BiFPNNode(nn.Module):
    """BiFPN 单个融合节点
    
    加权融合多个输入特征，权重可学习且归一化
    
    Tan et al., "EfficientDet", CVPR 2020
    """
    
    def __init__(self, in_channels_list: list, out_channels: int):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # 每个输入一个可学习权重（用 ReLU 保证非负）
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 1.0 / len(in_channels_list))
            for _ in in_channels_list
        ])
        
        # 每个输入一个 1x1 conv + BN 对齐通道数
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ) for c in in_channels_list
        ])
        
        # 融合后的 depthwise separable conv
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
    
    def forward(self, inputs: list) -> torch.Tensor:
        """加权融合多个特征图
        
        Args:
            inputs: list of (B, C, H, W) tensors
            
        Returns:
            fused: (B, out_channels, H, W)
        """
        # 确保所有输入尺寸一致（用最小尺寸）
        min_h = min(x.shape[2] for x in inputs)
        min_w = min(x.shape[3] for x in inputs)
        
        # 归一化权重: w_i = w_i / (sum(w_j) + eps)
        eps = 1e-4
        w_sum = sum(w.relu() for w in self.weights) + eps
        
        fused = 0
        for i, (x, conv, w) in enumerate(zip(inputs, self.convs, self.weights)):
            x = conv(x)
            if x.shape[2] != min_h or x.shape[3] != min_w:
                x = F.interpolate(x, size=(min_h, min_w), mode='nearest')
            fused = fused + w.relu() / w_sum * x
        
        return self.fuse_conv(fused)


class BiFPN(nn.Module):
    """BiFPN — 双向特征金字塔网络
    
    替换 YOLO 默认的 FPN+PAN neck
    
    输入: 3 个尺度的特征图 (P3, P4, P5)
    输出: 3 个融合后的特征图 (P3', P4', P5')
    
    结构 (3-scale, 2-repeat):
      Round 1:
        P4' = BiFPNNode([P4, P5_down])  # top-down
        P3' = BiFPNNode([P3, P4'_down])  # top-down
        P4'' = BiFPNNode([P4', P3'_up])  # bottom-up
        P5' = BiFPNNode([P5, P4''_up])   # bottom-up
      Round 2: 同上，输入用 Round 1 的输出
    """
    
    def __init__(self, in_channels: list, out_channels: int, repeats: int = 2):
        super().__init__()
        self.in_channels = in_channels  # [C3, C4, C5]
        self.out_channels = out_channels
        self.repeats = repeats
        
        # 初始通道对齐
        self.init_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
            ) for c in in_channels
        ])
        
        # 堆叠的 BiFPN rounds
        self.bifpn_rounds = nn.ModuleList()
        for _ in range(repeats):
            # 每个 round 有 4 个融合节点
            # top-down: P4_td = fuse(P4, P5_down), P3_td = fuse(P3, P4_td_down)
            # bottom-up: P4_bu = fuse(P4_td, P3_td_up), P5_bu = fuse(P5, P4_bu_up)
            round_nodes = nn.ModuleList([
                BiFPNNode([out_channels, out_channels], out_channels),  # P4_td
                BiFPNNode([out_channels, out_channels], out_channels),  # P3_td
                BiFPNNode([out_channels, out_channels], out_channels),  # P4_bu
                BiFPNNode([out_channels, out_channels], out_channels),  # P5_bu
            ])
            self.bifpn_rounds.append(round_nodes)
    
    def forward(self, features: list) -> list:
        """前向传播
        
        Args:
            features: [P3, P4, P5] from backbone, 尺寸递减
            
        Returns:
            [P3', P4', P5'] 融合后的特征
        """
        # 初始通道对齐
        p3, p4, p5 = [conv(f) for conv, f in zip(self.init_convs, features)]
        
        for round_nodes in self.bifpn_rounds:
            # Top-down: P5 → P4 → P3
            p5_down = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
            p4_td = round_nodes[0]([p4, p5_down])
            
            p4_td_down = F.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
            p3_td = round_nodes[1]([p3, p4_td_down])
            
            # Bottom-up: P3 → P4 → P5
            p3_td_up = F.interpolate(p3_td, size=p4_td.shape[2:], mode='nearest')
            p4_bu = round_nodes[2]([p4_td, p3_td_up])
            
            p4_bu_up = F.interpolate(p4_bu, size=p5.shape[2:], mode='nearest')
            p5_bu = round_nodes[3]([p5, p4_bu_up])
            
            # 更新特征供下一 round 使用
            p3, p4, p5 = p3_td, p4_bu, p5_bu
        
        return [p3, p4, p5]


# ============================================================================
# 3. MPCA — 多路径坐标注意力
# ============================================================================

class MPCA(nn.Module):
    """MPCA — Multi-Path Coordinate Attention
    
    Orga-Dete 论文 §2.2.2
    
    输入特征分 4 条路径，每条路径在不同维度做 coordinate attention:
    - Path 1: (H, 1) 沿高度编码
    - Path 2: (1, W) 沿宽度编码  
    - Path 3: (H, 1) + (1, W) 拼接
    - Path 4: 全局池化 → 1x1
    
    4 条路径的编码向量拼接 → 1x1 conv → MLP → channel weights
    weights 和原始特征 element-wise 相乘
    
    Hou et al., "Coordinate Attention", CVPR 2021 的增强版
    """
    
    def __init__(self, c1: int = None, c2: int = None, reduction: int = 32):
        """Ultralytics 兼容构造函数 — Lazy init
        
        parse_model 对非 base_modules 模块传入 args=[YAML值]
        MPCA 不在 base_modules 里，所以 args[0]=1024（YAML原始值，未 scaled）
        但实际输入通道是 256（scaled 后）。
        
        解法：__init__ 只存参数，第一次 forward 时根据实际 x.shape 创建 conv
        """
        super().__init__()
        self._reduction = reduction
        self._channels = None  # lazy init
        self._initialized = False
    
    def _build_layers(self, channels: int):
        """根据实际通道数构建层（lazy init）"""
        mid = max(channels // self._reduction, 16)
        self._channels = channels
        self._mid = mid
        
        # Path 1: X 方向
        self.path1_pool = nn.AdaptiveAvgPool2d((1, None))
        self.path1_conv = nn.Conv2d(channels, mid, 1, bias=False)
        
        # Path 2: Y 方向
        self.path2_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.path2_conv = nn.Conv2d(channels, mid, 1, bias=False)
        
        # Path 3: 双向
        self.path3_x_conv = nn.Conv2d(channels, mid, 1, bias=False)
        self.path3_y_conv = nn.Conv2d(channels, mid, 1, bias=False)
        
        # Path 4: 全局
        self.path4_pool = nn.AdaptiveAvgPool2d(1)
        self.path4_conv = nn.Conv2d(channels, mid, 1, bias=False)
        
        # 融合
        self.fuse_conv = nn.Conv2d(mid * 4, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()
        
        # MLP — 最后一层 conv 带 bias，初始化为 identity
        # sigmoid(5) ≈ 0.993 ≈ 1.0 → output ≈ x（不破坏预训练特征）
        mid_mlp = max(channels // 4, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid_mlp, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid_mlp, channels, 1, bias=True),  # bias=True for identity init
            nn.Sigmoid(),
        )
        # Identity init: weight=0, bias=5 → sigmoid(5)≈1 → output≈x
        _last_conv = self.mlp[-2]
        nn.init.zeros_(_last_conv.weight)
        nn.init.constant_(_last_conv.bias, 5.0)
        
        # 移到和输入相同的 device
        self._initialized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: (B, C, H, W) 输入特征图
            
        Returns:
            out: (B, C, H, W) 加权后的特征图
        """
        # Lazy init: 第一次 forward 时根据实际通道数构建层
        if not self._initialized:
            self._build_layers(x.shape[1])
            self.to(x.device)
        
        B, C, H, W = x.shape
        
        # Path 1: X 方向 (B, C, 1, W) → (B, mid, 1, W)
        p1 = self.path1_pool(x)
        p1 = self.path1_conv(p1)
        
        # Path 2: Y 方向 (B, C, H, 1) → (B, mid, H, 1)
        p2 = self.path2_pool(x)
        p2 = self.path2_conv(p2)
        
        # Path 3: 双向
        p3_x = self.path3_x_conv(self.path1_pool(x))  # (B, mid, 1, W)
        p3_y = self.path3_y_conv(self.path2_pool(x))  # (B, mid, H, 1)
        # 广播相加: (B, mid, H, W)
        p3 = p3_x + p3_y
        
        # Path 4: 全局 (B, C, 1, 1) → (B, mid, 1, 1)
        p4 = self.path4_pool(x)
        p4 = self.path4_conv(p4)
        
        # 对齐尺寸到 (H, W) 用于拼接
        p1_full = p1.expand(B, -1, H, W)
        p2_full = p2.expand(B, -1, H, W)
        p3_full = p3  # 已经是 (B, mid, H, W)
        p4_full = p4.expand(B, -1, H, W)
        
        # 拼接 → 1x1 conv → BN → SiLU
        fused = torch.cat([p1_full, p2_full, p3_full, p4_full], dim=1)
        fused = self.act(self.bn(self.fuse_conv(fused)))
        
        # MLP 生成 channel weights
        weights = self.mlp(fused)  # (B, C, H, W) — 每个 channel 每个位置一个权重
        
        # Element-wise multiply
        return x * weights


# ============================================================================
# 4. BiFPNHead — BiFPN + Detect 包装层
# ============================================================================

class BiFPNHead(nn.Module):
    """BiFPN + Detect 包装层
    
    Ultralytics 的 Detect head 期望接收多个特征图，
    但 parse_model 只能给它一个 "from" 列表。
    BiFPNHead 作为一个整体模块，内部先跑 BiFPN 再跑 Detect。
    
    YAML 用法:
        - [[4, 6, 11], 1, BiFPNHead, [256, nc]]  # 替换 FPN+PAN+Detect
    """
    
    def __init__(self, out_channels: int, nc: int = 1, reg_max: int = 16,
                 in_channels: list = None):
        """BiFPNHead 构造函数
        
        Args:
            out_channels: BiFPN 输出通道数
            nc: 类别数
            reg_max: Detect head 的 reg_max
            in_channels: [C3, C4, C5] 输入通道数（lazy init if None）
        """
        super().__init__()
        self.nc = nc
        self.out_channels = out_channels
        self._in_channels = in_channels
        self._initialized = False
        
        # Detect head 延迟构建（需要 in_channels）
        self._detect = None
    
    def _build_layers(self, in_channels: list, device=None):
        """Lazy init: 根据实际输入通道构建 BiFPN + Detect"""
        from ultralytics.nn.modules import Detect
        
        self.bifpn = BiFPN(in_channels, self.out_channels, repeats=2)
        
        # Detect head 接收 3 个特征图，通道都是 out_channels
        self._detect = Detect(
            nc=self.nc,
            ch=[self.out_channels, self.out_channels, self.out_channels]
        )
        
        if device:
            self.bifpn.to(device)
            self._detect.to(device)
        
        self._initialized = True
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: list [P3, P4, P5] from backbone
            
        Returns:
            Detect head 输出
        """
        if not self._initialized:
            # 从输入推断通道数
            in_chs = [xi.shape[1] for xi in x]
            self._build_layers(in_chs, device=x[0].device)
        
        # BiFPN 融合
        features = self.bifpn(x)  # [P3', P4', P5']
        
        # Detect head
        return self._detect(features)


# ============================================================================
# 测试
# ============================================================================

if __name__ == '__main__':
    print("Testing EMASlideLoss...")
    loss_fn = EMASlideLoss(alpha=0.9)
    
    # 模拟数据
    pred_scores = torch.randn(2, 100, 1, requires_grad=True)  # (B, A, NC)
    target_scores = torch.zeros(2, 100, 1)
    target_scores[0, :10, 0] = 1.0  # 10 个正样本
    
    iou = torch.rand(2, 100) * 0.8 + 0.1  # 0.1 - 0.9
    fg_mask = torch.zeros(2, 100, dtype=torch.bool)
    fg_mask[0, :10] = True
    
    loss = loss_fn(pred_scores, target_scores, iou, fg_mask)
    print(f"  Loss: {loss.sum().item():.4f}")
    print(f"  EMA μ: {loss_fn.ema_mu.item():.4f}")
    print("  ✅ EMASlideLoss OK")
    
    print("\nTesting BiFPN...")
    bifpn = BiFPN(in_channels=[256, 512, 1024], out_channels=256, repeats=2)
    
    # 模拟 backbone 输出
    p3 = torch.randn(1, 256, 80, 80)
    p4 = torch.randn(1, 512, 40, 40)
    p5 = torch.randn(1, 1024, 20, 20)
    
    out = bifpn([p3, p4, p5])
    print(f"  Input: P3{list(p3.shape)}, P4{list(p4.shape)}, P5{list(p5.shape)}")
    print(f"  Output: P3'{list(out[0].shape)}, P4'{list(out[1].shape)}, P5'{list(out[2].shape)}")
    params = sum(p.numel() for p in bifpn.parameters()) / 1e6
    print(f"  Parameters: {params:.2f}M")
    print("  ✅ BiFPN OK")
    
    print("\nTesting MPCA...")
    mpca = MPCA(channels=1024, reduction=32)
    
    x = torch.randn(1, 1024, 20, 20)
    out = mpca(x)
    print(f"  Input: {list(x.shape)}")
    print(f"  Output: {list(out.shape)}")
    params = sum(p.numel() for p in mpca.parameters()) / 1e3
    print(f"  Parameters: {params:.1f}K")
    print("  ✅ MPCA OK")
    
    print("\n✅ All three modules verified!")
