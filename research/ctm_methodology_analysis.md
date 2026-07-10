# CTM 连续思维链方法论深度拆解

> 2026-07-10 曼卿 | 基于冬生"我们现在都是一锤子买卖"的洞察

## 核心问题

冬生说得对——我们整个 pipeline 是**五次一锤子买卖**：

```
RF-DETR(单次) → SAM2(单次) → cv2 contour(单次) → VLM(单次) → k-NN(单次)
```

每一步都是 feed-forward，没有迭代，没有自适应计算，没有内部推理。CTM 的方法论完全不同。

---

## 1. CTM 的架构（逐步拆解）

### 1.1 核心循环（Listing 1 伪代码还原）

```python
# 初始化
kv = kv_projector(backbone(inputs))      # 图像 → KV tokens
z = z_init                                 # 可学习初始状态 (D维)
pre_acts_history = init_history            # 可学习预激活历史 (D×M)
post_acts_history = [z_init]               # 后激活历史

# 核心循环：T 个 internal ticks
for t in range(T):                         # T=50(ImageNet) 或 75(maze)
    # 1️⃣ 从神经同步化生成 attention query
    synch_a = compute_synch(post_acts_history, type="action")
    q = q_projector(synch_a)              # Q 来自内部动力学, NOT 来自输入
    
    # 2️⃣ Cross-attention: 用内部状态去"看"图像
    attn_out = attn(q, kv, kv)             # Q=内部, KV=图像
    
    # 3️⃣ 突触模型: 融合 attention 输出和当前状态
    pre_acts = synapses(concat(attn_out, z))
    
    # 4️⃣ 更新历史 (FIFO)
    pre_acts_history = roll(pre_acts_history, pre_acts)
    
    # 5️⃣ NLMs: 每个神经元用私有权重处理自己的历史
    z = neuron_level_models(pre_acts_history)  # D个独立MLP
    post_acts_history.append(z)
    
    # 6️⃣ 从同步化生成输出
    synch_o = compute_synch(post_acts_history, type="output")
    output = output_proj(synch_o)          # 预测
    
    # 7️⃣ 计算确定性（1 - 归一化熵）
    certainty = 1 - entropy(output) / max_entropy
```

### 1.2 三个核心创新

| 创新 | 标准模型 | CTM | 差异本质 |
|------|----------|-----|----------|
| **Attention Q 来源** | 从输入投影 | 从**神经同步化**投影 | Q 来自内部动力学，不是输入 |
| **表示方式** | 静态隐状态 | **神经同步化矩阵** St=Zt·Zt^T | 时间相关性作为表示 |
| **计算自适应** | 固定层数 | 基于确定性的**自适应停止** | 简单样本早停，困难样本多想 |

### 1.3 神经同步化——CTM 的"表示"

$$S_t = Z_t \cdot Z_t^\top \in \mathbb{R}^{D \times D}$$

其中 $Z_t = [z_1, z_2, \ldots, z_t]$ 是后激活历史。$S_t^{ij}$ 捕获神经元 $i$ 和 $j$ 在时间上的相关性。

**可学习时间衰减**（公式10）：
$$S_t^{ij} = (Z_t^i)^\top \cdot \text{diag}(R_t^{ij}) \cdot Z_t^j, \quad R_t^{ij}[\tau] = \exp(-r_{ij}(t-\tau))$$

- $r_{ij}=0$：全程同步（长程记忆）
- $r_{ij}$ 大：偏向近期（短程记忆）
- 每个神经元对有独立的 $r_{ij}$ → 多时间尺度

### 1.4 损失函数——选择最优 tick

```python
def ctm_loss(logits, targets):
    # logits: [B, C, T] — 每个tick都有预测
    certainties = 1 - normalized_entropy(logits)  # [B, T]
    
    # 选两个tick：loss最小 + certainty最高
    lowest_idx = losses.argmin(-1)    # 每个样本各自选
    certain_idx = certainties.argmax(-1)
    
    loss = (losses[:, lowest_idx] + losses[:, certain_idx]) / 2
    return loss.mean()
```

**关键**：不是取最后一个 tick 的预测，而是让模型自己决定哪个 tick 最好。这使模型能学到"什么时候该输出"。

---

## 2. CTM 在 ImageNet 上的"看图"行为

### 2.1 涌现的注意力扫描

Section 5.2 原文：
> "The CTM learns to 'look around' an image in order to gather information and make a prediction. It does this **entirely without prompting or any guide**, implementing computationally beneficial adaptive compute in an intuitive fashion."

- 16 个 attention heads 各自形成不同的扫描路径
- Attention 权重在 ticks 间演化，形成"目光轨迹"
- 没有任何训练信号告诉它"要扫视"——这是涌现的

### 2.2 自适应计算

Section 5.1：
- 确定性阈值 0.8 → 大多数样本在 <10/50 ticks 就可以停
- 困难样本继续思考到 50 ticks
- 这是 loss 函数的自然结果，不是额外机制

### 2.3 校准

- 迭代精炼带来优秀的模型校准
- 预测概率与实际准确率对齐——"说0.8就是0.8"

---

## 3. 我们的方法 vs CTM：差距在哪

| 维度 | 我们的方法 | CTM | 差距 |
|------|-----------|-----|------|
| **推理模式** | 单次前传 | T 步迭代精炼 | 无迭代 |
| **Attention Q** | 不存在（或从输入） | 从神经同步化 | 无内部动力学 |
| **表示** | 静态特征向量 | 时间相关同步矩阵 | 无时序 |
| **自适应** | 所有样本同等计算 | 基于确定性自适应停止 | 无早停 |
| **可解释性** | 黑箱打分 | 注意力轨迹可视化 | 无轨迹 |
| **校准** | VLM打分(0.8≠80%准确) | 涌现校准 | 无校准 |

**"一锤子买卖"的本质问题**：
- RF-DETR 检测一次就定死 bbox，不再修正
- VLM 评估一次就给分数，不会"多看几眼再决定"
- k-NN 检索一次就定距离，不会迭代逼近

---

## 4. 应用到 Organoid 检测的三个路径

### 路径 A：轻量 CTM 替换 VLM 评估（P0，可快速验证）

**思路**：用预训练 backbone（DINOv2）做特征提取，加一个轻量 CTM 循环做 TP/FP 判别。

```
crop → DINOv2 → KV tokens
                   ↓
    z_init → [tick 1: attend → synapse → NLM → sync → output]
           → [tick 2: attend → synapse → NLM → sync → output]
           → ...
           → [tick T: attend → synapse → NLM → sync → output]
                   ↓
            选最优tick输出 (loss最小 + certainty最高)
```

**优势**：
- 不需要训练大模型，只需训练轻量 CTM 头（~2M 参数）
- Attention Q 从内部同步化来，不直接用输入 → 涌现扫描行为
- 自适应停止：简单检测早停，困难检测多看几眼
- 输出校准：确定性能反映真实准确率

**训练数据**：MultiOrg 16198 检测（4639 TP + 11559 FP），已标注

**实现量**：~500 行 PyTorch，1 H100 训练 ~4h

### 路径 B：CTM 式多步 VLM Prompt（P1，无需训练）

**思路**：用 GLM-4.6V 做多步推理，每步输出坐标 + 理由，后步基于前步修正。

```python
# Tick 1: 初始观察
prompt_1 = "观察这张类器官图像，用 [[x1,y1,x2,y2]] 标注你看到的 organoid，说明理由"
response_1 = vlm(prompt_1, image)  # 输出坐标 + 理由

# Tick 2: 基于tick 1的观察，重新审视
prompt_2 = f"你之前说 organoid 在 {response_1.bbox}。重新审视这个区域和周围，确认或修正"
response_2 = vlm(prompt_2, image)  # 修正或确认

# Tick 3: 如果不确定，换视角
prompt_3 = f"从形态学角度评估 {response_2.bbox} 的圆度和边界清晰度"
response_3 = vlm(prompt_3, image)

# 自适应停止：如果response一致性高，停止
if consistency(response_1, response_2, response_3) > threshold:
    return response_3
else:
    # 继续 tick 4...
```

**优势**：
- 零训练，纯 prompt engineering
- 每步输出坐标（visual primitive），锚定到空间位置
- 自适应停止（一致性高就停）
- 涌现"多看几眼"的行为

**局限**：
- VLM 不是真正的 CTM，没有神经同步化
- API 调用成本随 ticks 线性增长
- 但比单次打分好——至少有迭代修正

### 路径 C：完整 CTM 检测器（P2，长期方向）

**思路**：用 CTM 替换 RF-DETR + SAM2 + VLM 整个 pipeline。

```
原图 → ResNet/ViT backbone → KV tokens
                              ↓
    z_init → [tick 1: attend → NLM → output: bbox + mask + is_organoid]
           → [tick 2: attend(修正) → NLM → output: refined bbox + mask]
           → ...
           → [tick T: final output]
```

**优势**：
- 端到端连续推理，无 pipeline 断裂
- 涌现扫描行为：模型自己决定先看哪里
- 自适应：简单图像 5 ticks，复杂图像 50 ticks
- 统一框架：检测 + 分割 + 评估一体化

**实现量**：~2000 行 PyTorch，1 H100 训练 ~100h

---

## 5. 建议优先级

| 路径 | 工作量 | 预期收益 | 风险 | 建议 |
|------|--------|----------|------|------|
| **B: 多步 VLM Prompt** | 2天 | 中 | 低 | **立即开始** |
| **A: 轻量 CTM** | 1周 | 高 | 中 | B验证后启动 |
| **C: 完整 CTM** | 1月+ | 最高 | 高 | 论文方向，不急 |

**路径 B 的价值**：即使最终不用 VLM 多步 prompt，也能验证"迭代推理是否比单次打分好"这个核心假设。如果 B 有效，说明 CTM 思路对；如果 B 无效，需要路径 A 的真正神经同步化。

---

## 6. CTM 方法论的核心启发

1. **Q 从内部来**：Attention 的 query 不是从输入投影的，而是从内部神经同步化生成的。这意味着模型"自己决定看哪里"，不是被输入牵着走。

2. **同步化作为表示**：不是用某个 tick 的隐状态，而是用整个历史的同步化模式。时间相关性 IS the representation。

3. **损失选择最优 tick**：不是取最后一步，而是让模型学到"哪个 tick 最好"。这使训练更灵活——不同样本可以在不同 tick 收敛。

4. **涌现的扫描**：没有任何训练信号告诉 CTM "要扫视图像"，但它自己学会了。这说明迭代 attention + 神经动力学天然产生"主动观察"行为。

5. **自适应是副产品**：不是显式的 halting module，而是 loss 函数 + 确定性计算的自然结果。简单样本在前几个 tick 就高确定性 → 可以早停。

**对我们最大的启发**：不是"加个 attention 层"就行，而是**整个推理范式要从 feed-forward 变成 iterative**。"一锤子买卖"的问题不是锤子不够大，而是只锤了一次。
