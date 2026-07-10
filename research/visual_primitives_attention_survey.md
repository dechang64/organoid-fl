# 视觉原语、注意力机制与连续思维链：文献深度调研与方法反思

> 2026-07-10 曼卿 | 基于冬生指出的方法问题

## 核心结论（先说）

冬生的直觉是对的——**我们的方法有三个根本性偏差**：

1. **"视觉原语"定义错了**：我们把 visual primitives 理解成形态学特征（area/circularity/solidity），但 DeepSeek 论文中的 visual primitives 是**空间标记（点、bbox）作为"思维最小单元"直接交织到推理链中**——是 reasoning units，不是 feature statistics。
2. **完全没有注意力机制**：CTM 用 cross-attention（Q 来自神经同步化），DeepSeek 用 Compressed Sparse Attention，Argus (CVPR 2025) 用 goal-conditioned visual attention，DeShiftNet 用 deformable-shifted cross-attention 做类器官分割——面上项目正文也明确写了"跨模态注意力机制"。我们的 pipeline 是 hand-crafted features + k-NN，零注意力。
3. **没有迭代推理**：CTM 有 internal ticks（多步思考），DeepSeek 有 "point while reasoning"（推理时输出坐标）——我们做的是单次 VLM 评估。

---

## 1. 三篇论文的核心创新

### 1.1 DeepSeek "Thinking with Visual Primitives"

**论文**：DeepSeek-AI + 北大 + 清华，arXiv 2505.05522v4，2025-10-03
**模型**：DeepSeek-V4-Flash（MoE 284B total / 13B active）+ DeepSeek-ViT + CSA

#### 核心洞察：Reference Gap

论文识别了两个层面的鸿沟：

| 鸿沟 | 定义 | 现有解法 | 局限 |
|------|------|----------|------|
| **Perception Gap** | 模型"看到"的 vs 图像实际内容 | 高分辨率裁剪、动态 patching | "seeing" ≠ "reasoning" |
| **Reference Gap** | 自然语言无法精确指代空间位置 | 无 | 即使感知完美，语言指代仍模糊 → 逻辑崩溃 |

> "The inherent ambiguity of natural language often fails to provide precise, unambiguous pointers to complex spatial layouts, leading to logical collapse in tasks requiring rigorous grounding."

#### 解决方案：Visual Primitives as "Minimal Units of Thought"

**不是**把 bbox/point 作为后处理验证（post-hoc verification），而是把它们**提升为推理链中的最小思维单元**：

```
传统 CoT: "图中有一只熊在树上，所以排除..."
Visual Primitive CoT: "图中有一只熊在 [[452,23,804,411]]，它紧贴树干，所以排除..."
```

关键区别：
- **传统 grounding**：检测 → 输出 bbox → 作为结果
- **Visual Primitive reasoning**：推理过程中**随时输出**坐标 → 作为**思维步骤** → 后续推理基于这些坐标

#### "Point-to-Reason" Synergy

灵感来自人类认知：走迷宫或数密集物体时，人会自然用手指（deictic pointers）来减轻认知负荷、维持逻辑一致性。模型通过交织 visual primitives 模拟这种"边指边推理"的协同。

#### 架构：极致视觉 Token 效率

- **DeepSeek-ViT**：图像 756×756 → 571,536 pixels → patch embedding → 9:1 channel compression
- **Compressed Sparse Attention (CSA)**：KV cache 再压缩 4x
- **总压缩率**：7056x（从原始像素到 KV cache entries）
- 结果：284B 模型用 ~90 entries in KV cache（vs GPT-5.4 ~660, Gemini-3-Flash ~740）

#### 训练管线

```
Pretraining (40M+ grounding data) 
  → Specialized SFT (counting, maze, path tracing)
  → Specialized RL 
  → Unified RFT (Reinforced Fine-Tuning)
  → On-Policy Distillation
```

#### 关键任务

1. **Counting**：粗粒度（区域分组）+ 细粒度（逐个标注 bbox）
2. **Spatial Reasoning**：空间关系判断
3. **Maze Navigation**：路径规划
4. **Path Tracing**：Bézier 曲线追踪——曲线交叉处用 local geometric-continuity primitive 判断走向

### 1.2 Continuous Thought Machines (CTM, Sakana AI)

**论文**：Luke Darlow et al., NeurIPS 2025 Spotlight, arXiv 2505.05522v4

#### 两个核心创新

1. **Neuron-Level Models (NLMs)**：每个神经元有**独立的权重参数**处理 incoming pre-activation 历史——不是简单的 `Wx+b`，而是每个神经元是一个微型时序处理器
2. **Neural Synchronization as Representation**：用神经元间的**时间相关性**（synchronization）作为核心表示，而非静态激活值

#### CTM 的注意力机制（关键！）

```python
# CTM 的核心循环
for step in range(n_thought_steps):
    # 1. 从神经同步化计算 attention query
    synch_a = compute_synch(post_acts_history, type="action")
    q = q_projector(synch_a)  # Q 来自内部神经动力学！
    
    # 2. Cross-attention: Q(内部) → K,V(外部数据)
    attn_out = attn(q, kv, kv)  # kv = FeatureExtractor(data)
    
    # 3. 拼接注意力输出，通过突触模型处理
    pre_acts = synapses(concat((attn_out, z)))
    
    # 4. NLM 处理 pre-activation 历史
    post_acts = nlms(pre_acts_history)  # 每个神经元独立处理
```

**关键洞察**：CTM 的 attention query `qt` **不来自输入数据**，而来自**内部神经同步化**。这意味着：
- 模型可以"想象"要找什么（episodic future thinking）
- 在 ImageNet 上，CTM 的注意力路径**自然涌现**——无需训练信号，模型会"扫视"图像不同区域
- 在迷宫任务中，CTM 通过 attention "想象"未来路径

#### 涌现特性

- **ImageNet**：16 个 attention head 形成复杂扫视路径，无需位置编码
- **Maze**：CTM 能泛化到比训练长 4x 的路径（100→400 步）
- **Adaptive computation**：可以在任何 internal tick 输出预测，不确定时多想几步

### 1.3 Argus: Vision-Centric Reasoning with Grounded CoT (CVPR 2025)

**论文**：Man et al., CVPR 2025, arXiv 2505.23766

- **Object-centric grounding as visual CoT signals**：物体中心的 grounding 作为 CoT 信号
- **Goal-conditioned visual attention**：注意力由推理目标引导
- 与 DeepSeek 类似，但更强调 attention 的目标导向性

### 1.4 JD Real-Time Visual Search

**论文**：JD.com, KDD 2017

- **分布式分层架构**：Blender(特征提取) → Broker(路由) → Searcher(检索)
- **倒排索引 + k-means 聚类**：图像聚类到最近的簇，只在簇内检索
- **层级索引**：coarse-to-fine，从粗到细
- **商品特征**：CNN 提取 → hash 编码 → 倒排索引
- **百亿级实时检索**：sub-second timescales

与我们的关系：JD 是工程实现，核心是**大规模检索效率**。我们的 FedCtx HNSW 已经解决了检索效率问题，但 JD 的层级索引（coarse-to-fine）可以作为优化参考。

### 1.5 DeShiftNet: 类器官分割的 deformable cross-attention

**论文**：BMC Bioinformatics 2026, Springer

- **Deformable-Shifted Encoder**：内容感知的空间建模
- **Cross-attention-guided decoder**：精确边界定位
- **在 multi-type OrganoID 数据集上验证**——和我们的领域完全重合
- 轻量级 + 鲁棒

**这条直接相关**——有人在同一数据集上用 cross-attention 做分割了。

---

## 2. 我们的方法问题（逐条对照）

### 2.1 "Visual Primitives" 概念混淆

| | DeepSeek 的 Visual Primitives | 我们的 "Primitives" |
|---|---|---|
| **定义** | 空间标记（点、bbox）作为思维最小单元 | 形态学特征（area, circularity, solidity） |
| **角色** | 推理链中的**思维步骤** | 检测后的**特征统计** |
| **使用方式** | 推理过程中随时输出坐标 | 一次性提取后用于 k-NN |
| **与推理的关系** | 交织（interleave）→ 后续推理基于坐标 | 分离（detached）→ 不参与推理过程 |
| **类比** | 人类用手指指向 → 边指边想 | 人类测量物体 → 量完再想 |

**冬生的直觉完全正确**：我们叫"视觉原语"但做的是"形态学特征提取"，名词对不上内容。

### 2.2 完全没有注意力机制

| 论文 | 注意力机制 | 我们的对应 |
|---|---|---|
| **CTM** | Cross-attention, Q 来自神经同步化 | ❌ 无 |
| **DeepSeek** | Compressed Sparse Attention (7056x 压缩) | ❌ 无 |
| **Argus** | Goal-conditioned visual attention | ❌ 无 |
| **DeShiftNet** | Deformable-shifted cross-attention | ❌ 无 |
| **面上项目** | "跨模态注意力机制" (研究内容1) | ❌ 无 |

我们的 pipeline：RF-DETR → SAM2 → cv2.contourFeatures → k-NN

**零注意力。** 全程是 hand-crafted features + distance matching。

DINOv2 我们试过（768d embedding），但用来做 k-NN 检索，PR-AUC=0.29 失败。原因正是**没有 attention-based reasoning**——只取了 CLS token 做 embedding，没有用 cross-attention 来"查询"图像中与 organoid 相关的区域。

### 2.3 没有迭代推理

| 论文 | 迭代方式 | 我们的对应 |
|---|---|---|
| **CTM** | Internal ticks (多步思考, adaptive) | ❌ 单次 VLM 评估 |
| **DeepSeek** | Point while reasoning (推理时输出坐标) | ❌ VLM 只输出分数 |
| **Argus** | Grounded CoT (多步推理) | ❌ 无 CoT |

我们的 Phase 2 VLM 评估流程：
1. 裁剪 bbox → 发给 GLM-4.6V → 得到 "is_organoid: 0.8"
2. 单次，无迭代，无推理链

**Reference Gap 在我们这里的具体表现**：VLM 输出 "is_organoid: 0.8"，但无法说出"因为在 [x1,y1] 处有圆形轮廓，在 [x2,y2] 处有致密结构"——它只能给一个分数，不能给出空间定位的推理依据。

### 2.4 特征提取 vs 学习表示

| 面上项目方案 | 我们的实现 |
|---|---|
| ResNet-50 + ViT-L/14 → 768d 学习特征 | cv2 contour features (6d hand-crafted) |
| 跨模态注意力融合 → 1536d | 无融合，直接拼接 |
| 对抗学习+对比学习对齐向量空间 | 无对齐 |

---

## 3. 方向建议

### 3.1 短期（不需训练大模型）

**A. VLM Prompt 重构：从"打分"到"指认"**

当前 prompt：`"这是类器官吗？给出 0-1 分数"`
改为 DeepSeek 式：`"在图像中找到所有类器官，用 [[x1,y1,x2,y2]] 坐标标注，并解释为什么"`

这不需要训练新模型——只需要改 prompt。VLM 被迫输出坐标时，它的"推理"会被锚定到空间位置，减少幻觉。

**B. Attention-based Feature Extraction（替换 hand-crafted）**

用 DINOv2/ViT 提取**空间注意力图**而非全局 embedding：
- 给定一个 bbox（organoid 检测）
- 用 bbox 中心作为 query
- 对图像特征做 cross-attention
- 输出 attention-weighted feature（而非 CLS token）

这是 CTM 的简化版：Q = detection query, K/V = image features。

**C. 多尺度 Attention（替代 SAM2 mask 的形态学）**

不提取 area/circularity，而是：
1. 在 organoid bbox 内做 multi-scale attention
2. 每个 scale 关注不同特征（整体形状/内部结构/边界）
3. 拼接成 attention feature vector

### 3.2 中期（需要轻量训练）

**D. 简化 CTM：Organoid-CTM**

在 RF-DETR + SAM2 基础上叠加一个轻量 CTM：
1. RF-DETR 检测 → bbox
2. SAM2 分割 → mask
3. **CTM 循环**（3-5 ticks）：
   - Tick 1: 关注 mask 整体形状
   - Tick 2: 关注内部结构
   - Tick 3: 关注边界
   - 每个 tick 用 cross-attention 从图像特征中提取信息
4. 输出：confidence + uncertainty

**关键**：不需要 284B MoE，只需要一个小型 attention module（几 MB）在 RF-DETR + SAM2 之上。

**E. DeShiftNet 迁移**

DeShiftNet 在 multi-type OrganoID 上做了 deformable-shifted cross-attention 分割。直接用/复现 DeShiftNet 替换我们的 SAM2，可以：
- 获得 attention-based 的 organoid mask
- 内置 content-aware spatial modeling
- 同一数据集验证过

### 3.3 长期（需要 GPU + 训练）

**F. 面上项目原方案**

按面上项目正文执行：
- ResNet-50 + ViT-L/14 → 768d
- 跨模态注意力融合 → 1536d
- 对抗学习+对比学习对齐
- Diffusion 流形扩展

**G. Visual Primitive Reasoning（DeepSeek 式）**

如果资源允许：
- Fine-tune GLM-4.6V 或类似 VLM
- 训练数据：organoid 图像 + bbox 标注 → "point while reasoning" 格式
- 推理链：`"在 [[x1,y1,x2,y2]] 处有一个类器官，因为..."`

---

## 4. 优先级建议

| 优先级 | 方案 | 难度 | 预期收益 | 时间 |
|--------|------|------|----------|------|
| **P0** | A. VLM Prompt 重构 | 低 | 中 | 1天 |
| **P0** | B. Attention-based feature | 中 | 高 | 2-3天 |
| **P1** | E. DeShiftNet 迁移 | 中 | 高 | 3-5天 |
| **P1** | C. 多尺度 attention | 中 | 中 | 2天 |
| **P2** | D. 简化 CTM | 高 | 高 | 1-2周 |
| **P3** | G. Visual primitive reasoning | 极高 | 极高 | 需GPU |

**P0 可以立即做**——改 VLM prompt + 加 attention-based feature extraction，不需要训练新模型。

---

## 5. 与面上项目研究内容的对齐

面上项目三个研究内容：

| 研究内容 | 面上方案 | 我们当前 | 论文指导 |
|----------|----------|-----------|----------|
| **1. 多模态向量化表征** | ResNet+ViT → 768d + 跨模态注意力 → 1536d | cv2 features (6d) | CTM cross-attention + DeepSeek CSA |
| **2. LLM 质量评估** | LLM-as-Judge + 领域知识图谱 | GLM-4.6V 单次打分 | DeepSeek "point while reasoning" |
| **3. 联邦隐私检索** | HNSW + 对抗学习对齐 | FedCtx HNSW (已完成) | JD hierarchical indexing |

**差距最大的是研究内容 1 和 2**——完全没有注意力，没有学习表示，没有迭代推理。FedCtx（研究内容3）反而最接近完成。

---

## 6. 参考文献清单

1. **DeepSeek-AI**. "Thinking with Visual Primitives." arXiv:2505.05522v4, 2025-10-03.
   - 核心：Visual primitives = spatial markers as reasoning units
   - 创新：Reference Gap, Point-to-Reason synergy, CSA
   
2. **Darlow, L. et al.** "Continuous Thought Machines." NeurIPS 2025 Spotlight. arXiv:2505.05522v4.
   - 核心：Neural synchronization as representation
   - 创新：NLMs, cross-attention with Q from synchronization, emergent attention
   
3. **Man, Q. et al.** "Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought." CVPR 2025.
   - 核心：Object-centric grounding as CoT signals
   - 创新：Goal-conditioned visual attention
   
4. **JD.com**. "The Design and Implementation of a Real Time Visual Search System on JD E-commerce Platform." KDD 2017.
   - 核心：Distributed hierarchical indexing
   - 创新：Blender-Broker-Searcher, k-means inverted index
   
5. **DeShiftNet**. "A deformable-shifted cross-attention network for lightweight and robust organoid image segmentation." BMC Bioinformatics 2026.
   - 核心：Deformable-shifted cross-attention for organoid
   - 创新：Content-aware spatial modeling, same dataset as ours
   
6. **Vaswani, A. et al.** "Attention is All You Need." NeurIPS 2017.
   - 基础：Self-attention, cross-attention
   
7. **DeepSeek-AI**. "DeepSeek-V3 Technical Report." (CSA origin)
   - Compressed Sparse Attention 的原始来源
