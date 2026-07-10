# Continuous Thought Machine: 完整技术拆解

> 2026-07-10 曼卿 | CTM 论文逐节精读 + PyTorch 实现

---

## 1. 为什么 CTM 不是"多跑几次"

标准 RNN/LSTM 也有循环，为什么 CTM 不同？关键在三个字：**从哪来**。

| | LSTM | Transformer | CTM |
|---|---|---|---|
| 隐状态来源 | 上一步的 h 和 c | 无（每层独立） | 神经同步化矩阵 |
| Attention Q 来源 | 不适用 | 从输入投影 | **从同步化投影** |
| 表示 | 当前时刻的 h | 当前层的特征 | **时间相关性矩阵** |
| 自适应 | 无 | 无 | **损失函数内建** |

LSTM 的 h_t 包含的是"到此刻为止的所有信息压缩"，但它是**快照**——一个时刻的截面。CTM 的同步化矩阵 St 包含的是"整个历史中每对神经元如何相关"——是**模式**，不是快照。

这个区别是根本性的：快照告诉你"现在状态是什么"，模式告诉你"经过怎样的动态到达现在"。

---

## 2. 逐组件拆解

### 2.1 突触模型（Synapse Model）

**功能**：融合 attention 输出和当前状态，产生预激活

```python
# f_syn 可以是：
# - 简单 MLP
# - U-Net 结构（论文中 ImageNet 用这个）
a_t = f_syn(concat(o_t, z_t))  # ∈ R^D
```

U-Net 变体：bottleneck 最窄 16 维，skip connection + layer-norm 保持信息。31.9M 参数（含 backbone ResNet-152）。

**为什么用 U-Net**：多尺度处理——bottleneck 做信息压缩，skip connection 做信息保留。这对图像任务有效，因为不同神经元需要关注不同尺度的信息。

### 2.2 预激活历史（Pre-activation History）

**功能**：保留最近 M 步的预激活，作为 NLM 的输入

```python
A_t = [a_{t-M+1}, a_{t-M+2}, ..., a_t]  # ∈ R^{D×M}
# M ≈ 10-100
# FIFO 结构：新进旧出
pre_acts_history = concat((pre_acts_history[:, :, :-1], pre_acts), dim=-1)
```

**关键**：M 不是全文历史，是滑动窗口。M=10 意味着每个神经元只看最近 10 步的"刺激历史"。这模拟了生物神经元的短期可塑性。

### 2.3 神经元级模型（NLMs）⭐核心创新1

**功能**：每个神经元用**私有权重**处理自己的预激活历史

```python
# 每个神经元 d 有独立的 g_{θ_d}（depth-1 MLP）
# Listing 2 伪代码：
weights_1 = Parameter(shape=(M, d_hidden, d_model))  # 私有！
bias_1 = zeros(shape=(1, d_hidden, d_model))
weights_2 = Parameter(shape=(d_hidden, d_model))
bias_2 = zeros(shape=(1, d_model))

def nlm(pre_acts_history):
    # pre_acts_history: (B, D, M) — 每个神经元自己的 M 步历史
    inputs = pre_acts_history[-M:]  # (B, D, M)
    out = einsum('bdM, Mhd -> bdh', inputs, weights_1) + bias_1  # 私有投影
    out = einsum('bdh, hd -> bd', out, weights_2) + bias_2       # 压缩
    return out  # (B, D) — 每个神经元一个值

z_{t+1} = nlm(A_t)  # ∈ R^D
```

**为什么这是革命性的**：
- 标准 NN：所有神经元共享同一组权重（一个 Linear 层）
- CTM：每个神经元有私有权重，处理自己的历史
- 这意味着 D=512 的 CTM 有 512 个不同的"个体处理器"

**计算技巧**：用 `einsum('bdM, Mhd -> bdh')` 而非 for 循环，GPU 友好。

**生物对应**：生物大脑中每个神经元都是不同的——有不同的形态、突触位置、离子通道分布。CTM 的 NLM 捕捉了这种"个体性"。

### 2.4 神经同步化矩阵（Synchronization Matrix）⭐核心创新2

**功能**：从后激活历史计算神经元间的时间相关性，作为"表示"

**数学定义**：

$$Z_t = [z_1, z_2, ..., z_t] \in \mathbb{R}^{D \times t}$$

$$S_t = Z_t \cdot Z_t^T \in \mathbb{R}^{D \times D}$$

每个元素 $S_t[i,j]$ = 神经元 i 和神经元 j 在整个历史中的内积 = **它们有多同步**。

**时间衰减（可学习）**：

$$R_{ij}^t = [e^{-r_{ij}(t-1)}, e^{-r_{ij}(t-2)}, ..., e^{0}]^T$$

$$S_{ij}^t = \frac{(Z_i^t)^T \cdot \text{diag}(R_{ij}^t) \cdot Z_j^t}{\sqrt{\sum_\tau (R_{ij}^t)^\tau}}$$

- $r_{ij} = 0$：无衰减，全部历史等权 → 长期记忆
- $r_{ij}$ 大：偏向近期 → 短期记忆
- **ImageNet 上 CTM 几乎没用到衰减**（$r_{ij} \approx 0$），maze 上用更多
- 这说明不同任务需要不同的时间尺度

**为什么用同步化而不是直接用 $z_t$**：
> 论文原文："We found that 'snapshot' representations were too constraining: projecting from z_t strongly ties it to the downstream task and thereby limits the types of dynamics it can produce, whereas synchronization decouples it."
>
> 翻译：快照表示太约束了——直接从 z_t 投影到输出会把内部状态和下游任务绑死，限制了能产生的动态模式。同步化解耦了表示和任务。

**核心洞察**：同步化矩阵捕获的不是"当前状态"，而是"状态变化的模式"。两个不同的 z_t 序列可以有相同的同步化模式（如果它们的协方差结构相同），这意味着 CTM 的表示是**动态不变量**，不是静态快照。

### 2.5 神经配对策略（Neuron Pairing）

**问题**：$S_t \in \mathbb{R}^{D \times D}$，D=512 → 262144 个元素，太大

**三种配对策略**：

1. **Dense pairing**：选 J 个神经元，算所有 $\frac{J(J+1)}{2}$ 对
   - 强瓶颈：所有梯度必须通过选中的 J 个神经元
   - 适合需要强正则化的任务

2. **Semi-dense**：选两组 $J_1, J_2$，左来自 $J_1$，右来自 $J_2$
   - 瓶颈宽度 2x
   - 允许更多信息流

3. **Random pairing**：随机选 $D_{out}$ 和 $D_{action}$ 对
   - 瓶颈最宽
   - 允许重叠
   - 还计算 $(i,i)$ 自内积 → 可以恢复快照表示如果需要

**两组同步化**：
- $S_t^{out}$ → 用于输出 $y_t = W_{out} \cdot S_t^{out}$
- $S_t^{action}$ → 用于 attention query $q_t = W_{in} \cdot S_t^{action}$
- **不共享神经元** → 输出和观察解耦

### 2.6 Cross-Attention（Q 从内部来）⭐核心创新3

```python
# 标准 cross-attention:
q = W_q(input)           # Q 从输入来
k, v = W_k(input2), W_v(input2)
attn_out = Attention(q, k, v)

# CTM cross-attention:
q = W_action(S_t^action)  # Q 从神经同步化来！
k, v = backbone(image)     # KV 从图像来
attn_out = Attention(q, k, v)
```

**这个区别是 CTM 的灵魂**：
- 标准 attention："从输入中找到相关的部分" → 被动
- CTM attention："基于内部状态，主动决定看哪里" → 主动

**涌现的扫描行为**（ImageNet 实验）：
- 没有任何训练信号告诉 CTM "要扫视图像"
- 但训练后，CTM 的 attention 路径自然呈现出"先看中心 → 再扫边缘 → 回到关键区域"的模式
- 16 个 attention head 呈现出复杂的路径（论文 Figure 2b）
- 论文用"not quite entirely unlike how humans might look around images"描述

**为什么 Q 从内部来能产生扫描**：
- tick 1：z_init → 同步化 ≈ 初始模式 → Q 指向某个区域
- tick 2：z 更新 → 同步化变化 → Q 指向另一个区域
- 每个 tick 的 attention 输出改变内部状态 → 下一个 Q 变化 → 形成"扫描轨迹"

**对比我们的 VLM**：
- VLM 的 Q 从输入(crop)来 → 被动看 → 单次打分
- CTM 的 Q 从内部来 → 主动看 → 多步精炼

### 2.7 损失函数（Loss Function）⭐核心创新4

```python
def ctm_loss(logits, targets):
    B, C, T = logits.shape  # B=batch, C=classes, T=ticks
    
    # 计算每个 tick 的确定性
    p = F.softmax(logits, 1)
    entropy = -torch.sum(p * torch.log_softmax(logits, 1), dim=1)
    max_entropy = torch.log(C)
    certainties = 1 - (entropy / max_entropy)  # (B, T)
    
    # 计算每个 tick 的损失
    losses = F.cross_entropy(predictions, targets_exp, reduction='none')  # (B, T)
    
    # t1: 损失最小的 tick
    lowest_idx = losses.argmin(-1)  # (B,)
    
    # t2: 确定性最高的 tick
    certain_idx = certainties.argmax(-1)  # (B,)
    
    # 最终损失 = 两个 tick 的平均
    loss = (losses[:, lowest_idx] + losses[:, certain_idx]) / 2
    return loss
```

**为什么这实现了自适应计算**：
- 简单样本：tick 3 就高确定性、低损失 → $t_1 = t_2 = 3$ → 只需 3 步
- 困难样本：到 tick 40 才收敛 → $t_1 = t_2 = 40$ → 需要 40 步
- **不需要显式的 halting module** → 自适应是损失函数的自然结果

**推理时的自适应**：
```python
# 训练后推理
for t in range(T_max):
    output_t, certainty_t = ctm_tick(t)
    if certainty_t > threshold:  # e.g., 0.8
        return output_t  # 早停
# ImageNet: 设定 0.8 阈值，大多数样本 <10 ticks 即可停止
```

**校准也是涌现的**：
- 损失函数要求"确定性高的 tick 也要准确"
- 这迫使模型学会：不确定时给低确定性 → 概率自然校准
- ImageNet 上 CTM 的校准曲线接近对角线（Figure 5b）

---

## 3. 完整 PyTorch 实现骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

class CTM(nn.Module):
    def __init__(self, backbone, d_model=512, n_neurons=512, 
                 memory_len=10, n_heads=16, 
                 n_action_pairs=256, n_output_pairs=256,
                 n_ticks=50, n_classes=2):
        super().__init__()
        self.backbone = backbone  # e.g., ResNet, DINOv2
        self.n_ticks = n_ticks
        
        # 突触模型 (U-Net or MLP)
        self.synapses = nn.Sequential(
            nn.Linear(d_model + d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # NLMs: 每个神经元私有权重
        self.nlm_weights_1 = nn.Parameter(torch.randn(memory_len, 64, n_neurons) * 0.02)
        self.nlm_bias_1 = nn.Parameter(torch.zeros(1, 64, n_neurons))
        self.nlm_weights_2 = nn.Parameter(torch.randn(64, n_neurons) * 0.02)
        self.nlm_bias_2 = nn.Parameter(torch.zeros(1, n_neurons))
        
        # 可学习初始状态
        self.z_init = nn.Parameter(torch.randn(n_neurons) * 0.02)
        self.pre_acts_init = nn.Parameter(torch.randn(n_neurons, memory_len) * 0.02)
        
        # 可学习衰减率 (每对神经元)
        self.decay_rates_action = nn.Parameter(torch.zeros(n_action_pairs))  # r_ij
        self.decay_rates_output = nn.Parameter(torch.zeros(n_output_pairs))
        
        # 随机配对 (固定，训练时不更新)
        self.action_pairs = torch.randint(0, n_neurons, (n_action_pairs, 2))
        self.output_pairs = torch.randint(0, n_neurons, (n_output_pairs, 2))
        
        # 投影
        self.q_projector = nn.Linear(n_action_pairs, d_model)
        self.kv_projector = nn.Linear(backbone.dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.output_proj = nn.Linear(n_output_pairs, n_classes)
    
    def compute_synch(self, post_acts_history, pairs, decay_rates):
        """计算神经同步化
        post_acts_history: list of (B, D) tensors, length = t
        pairs: (n_pairs, 2) — 神经元对索引
        decay_rates: (n_pairs,) — 每对的衰减率
        """
        Z = torch.stack(post_acts_history, dim=-1)  # (B, D, t)
        t = Z.shape[-1]
        
        # 获取每对神经元的活动
        i_idx = pairs[:, 0]  # (n_pairs,)
        j_idx = pairs[:, 1]
        Z_i = Z[:, i_idx, :]  # (B, n_pairs, t)
        Z_j = Z[:, j_idx, :]  # (B, n_pairs, t)
        
        # 时间衰减
        tau = torch.arange(t, device=Z.device).float()  # (t,)
        R = torch.exp(-decay_rates.unsqueeze(1) * (t - 1 - tau).unsqueeze(0))  # (n_pairs, t)
        R = R.unsqueeze(0)  # (1, n_pairs, t)
        
        # 同步化 = 加权内积
        synch = (Z_i * R * Z_j).sum(-1) / (R.abs().sum(-1) + 1e-8).sqrt()  # (B, n_pairs)
        return synch
    
    def forward(self, x):
        B = x.shape[0]
        
        # 1. 提取图像特征 → KV
        features = self.backbone(x)  # (B, H, W, C) or (B, N, C)
        if features.dim() == 4:
            features = features.flatten(1, 2)  # (B, N, C)
        kv = self.kv_projector(features)  # (B, N, d_model)
        
        # 2. 初始化
        pre_acts_hist = self.pre_acts_init.unsqueeze(0).repeat(B, 1, 1)  # (B, D, M)
        z = self.z_init.unsqueeze(0).repeat(B, 1)  # (B, D)
        post_acts_hist = [z]
        outputs = []
        
        # 3. 核心循环
        for t in range(self.n_ticks):
            # Q 从同步化来
            synch_a = self.compute_synch(post_acts_hist, 
                                         self.action_pairs.to(x.device),
                                         self.decay_rates_action)
            q = self.q_projector(synch_a)  # (B, d_model)
            q = q.unsqueeze(1)  # (B, 1, d_model) — 作为单 token query
            
            # Cross-attention
            attn_out, _ = self.attn(q, kv, kv)  # (B, 1, d_model)
            attn_out = attn_out.squeeze(1)  # (B, d_model)
            
            # 突触模型
            pre_acts = self.synapses(torch.cat([attn_out, z], dim=-1))  # (B, D)
            
            # 更新历史 (FIFO)
            pre_acts_hist = torch.cat([pre_acts_hist[:, :, :-1], pre_acts.unsqueeze(-1)], dim=-1)
            
            # NLMs: 每个神经元用私有权重处理历史
            # einsum: (B, D, M) × (M, H, D) → (B, H, D)
            hidden = torch.einsum('bdm,mhd->bhd', pre_acts_hist, 
                                   self.nlm_weights_1) + self.nlm_bias_1
            hidden = F.gelu(hidden)
            z = torch.einsum('bhd,hd->bd', hidden, self.nlm_weights_2) + self.nlm_bias_2
            
            post_acts_hist.append(z)
            
            # 输出从同步化来
            synch_o = self.compute_synch(post_acts_hist,
                                          self.output_pairs.to(x.device),
                                          self.decay_rates_output)
            y = self.output_proj(synch_o)  # (B, n_classes)
            outputs.append(y)
        
        return torch.stack(outputs, dim=-1)  # (B, n_classes, T)


def ctm_loss(logits, targets):
    """logits: (B, C, T), targets: (B,)"""
    B, C, T = logits.shape
    
    # 每个tick的确定性
    p = F.softmax(logits, 1)
    entropy = -torch.sum(p * F.log_softmax(logits, 1), dim=1)  # (B, T)
    certainty = 1 - entropy / torch.log(torch.tensor(C, dtype=torch.float))
    
    # 每个tick的损失
    targets_exp = targets.unsqueeze(-1).repeat(1, T)
    losses = F.cross_entropy(logits.permute(0, 2, 1), targets_exp, reduction='none')  # (B, T)
    
    # t1: 最小损失, t2: 最大确定性
    t1 = losses.argmin(-1)  # (B,)
    t2 = certainty.argmax(-1)  # (B,)
    
    loss = (losses[torch.arange(B), t1] + losses[torch.arange(B), t2]) / 2
    return loss
```

---

## 4. CTM 的三个实验揭示了什么

### 4.1 2D Maze（Section 4）—— 内部世界模型

**任务**：39×39 迷宫，输出 100 步路径（上下左右等）

**关键设置**：
- 无位置编码 → 模型必须自己构建空间表示
- 输出是动作序列 → 不是逐位置分类

**结果**：
- CTM 大幅超越 LSTM（LSTM 不稳定，50 ticks 后崩溃）
- CTM 能泛化到 99×99 更大迷宫（用滑动窗口重复应用）
- attention 轨迹显示 CTM 沿路径"看"——像人走迷宫

**启示**：CTM 能构建**内部世界模型**。这对 organoid 检测的意义是：CTM 可能构建一个"organoid 应该长什么样"的内部模型，而不只是模式匹配。

### 4.2 ImageNet（Section 5）—— 涌现的扫视和校准

**任务**：标准 ImageNet-1K 分类

**结果**：
- 50 ticks + ResNet-152 → 72.47% top-1
- 不是 SOTA（论文说不追求），但展示了独特能力

**涌现行为**：
1. **扫视**：16 个 attention head 呈现复杂路径，先看中心→扫边缘→回到关键区域
2. **自适应**：0.8 确定性阈值下，大多数样本 <10 ticks 停止
3. **校准**：预测概率和实际准确率接近对角线
4. **旅行波**：UMAP 投影的神经元活动出现低频旅行波——类似大脑皮层的神经活动模式

**启示**：CTM 不是被输入驱动的，而是**主动观察**。对 organoid 检测：CTM 可能会"先看整个视野 → 关注某个 organoid → 看边缘确认形态 → 给出判断"。

### 4.3 Parity（Section 6）—— 可解释的算法学习

**任务**：64 位二进制序列，预测每个位置的累积奇偶性

**结果**：
- CTM 75-100 ticks 能达到完美准确率
- LSTM 即使 100 ticks 也远不如 CTM
- attention 策略可解释：至少一个 head 从头到尾扫描序列

**启示**：CTM 能学习**算法策略**。对 organoid 检测：CTM 可能学会"扫描图像→找到候选→验证形态→确认"的多步算法。

---

## 5. CTM vs 我们当前方法的本质对比

| 维度 | 当前方法 | CTM |
|---|---|---|
| **观察方式** | RF-DETR 单次扫描全图 | 逐 tick 注意不同区域 |
| **状态表示** | 无状态（每个检测独立） | 同步化矩阵（累积动态） |
| **决策过程** | 单次分类 | 多 tick 逐步精炼 |
| **置信度** | RF-DETR 分数（未校准） | 涌现校准（1-归一化熵） |
| **困难样本** | 和简单样本同等处理 | 自动分配更多 ticks |
| **内部模型** | 无 | 有（世界模型） |
| **神经元个体性** | 无（权重共享） | 有（NLM 私有参数） |

**最核心的差异**：我们的 pipeline 把检测、分割、特征提取、评估做成独立步骤。CTM 把它们融为一体——同一个循环里同时完成观察、思考、输出。这不是"加迭代"——是范式的根本转变。

---

## 6. 应用到 Organoid 检测的具体设计

### 6.1 任务定义

**输入**：MultiOrg 图像（或 crop）
**输出**：每个检测的 TP/FP 判定 + 置信度
**训练数据**：16198 检测（4639 TP + 11559 FP），有 bbox + mask + matched 标签

### 6.2 架构设计

```
输入 crop → DINOv2 ViT-B/14 → KV tokens (37×37×768)
                                        ↑
              cross-attention (Q from sync) |
              ↑                             |
z_init → [tick: sync → Q → attn → synapse → NLM → z → output] × T
              ↓
         TP/FP score + certainty per tick
              ↓
         argmin(loss) + argmax(certainty) → final output
```

**参数**：
- DINOv2 ViT-B/14：86M 参数（预训练，冻结）
- CTM 头：~2M 参数
- D=256, M=20, T=20, n_heads=8
- n_action_pairs=128, n_output_pairs=128

### 6.3 训练计划

1. **数据**：16198 检测 → 7:2:1 train/val/test
2. **损失**：binary cross-entropy + ctm_loss（选最优 tick）
3. **优化器**：Adam, lr=1e-4, weight_decay=0（论文推荐）
4. **训练轮次**：100 epochs, batch=32
5. **GPU**：冬生 3060 12GB，预估 ~8h
6. **评估**：ROC-AUC, F1, 以及**每 tick 的 accuracy 曲线**（看是否真的迭代精炼）

### 6.4 关键验证指标

不只是看 AUC，要看 CTM 是否真的在做"连续思考"：

1. **Tick-wise accuracy**：accuracy 应随 tick 增加（如果只是噪声波动说明没学到迭代）
2. **Attention 轨迹**：可视化每 tick 的 attention map，看是否有"扫视"行为
3. **自适应分布**：统计每个样本用了多少 ticks（0.8 确定性阈值），应呈长尾分布
4. **校准曲线**：预测概率 vs 实际准确率，应接近对角线
5. **TP vs FP 的 tick pattern**：TP 是否在前几个 tick 就高确定性？FP 是否需要更多 ticks？

如果这些指标都不好，说明 CTM 对这个任务不work；如果 accuracy 随 tick 上升 + attention 有规律，说明迭代推理确实在发生。
