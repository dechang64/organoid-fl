# 视觉原语 × 大模型 × 类器官图像分析：深度文献调研

> 2026-07-14 | 基于跨域评估失效的反思 + 冬生"借鉴 NLP 语义分析成功经验"的思路

---

## 一、核心问题

### 1.1 跨域失效的诊断

SupCon slot model 在 MultiOrg 同域 AP=0.910，但跨域到鼠肝（AUC=0.29-0.54）和 Intestinal（AUC=0.67）全部失效。根因：**100 crops 上学到的 TP/FP 区分是域特定的，不是通用 organoid 原语**。

### 1.2 NLP 的成功经验

NLP 中 BERT/GPT 的预训练-微调范式之所以成功，核心是：
1. **大规模语料预训练** → 通用语义表示（不是单个任务数据集）
2. **token 作为最小语义单元** → 跨任务复用
3. **语义对齐** → 词向量空间中相似语义聚集
4. **上下文理解** → 同一词在不同语境有不同表示

视觉原语的类比：将图像中的"objects/regions"作为最小语义单元，用大规模预训练学到通用视觉语义，再迁移到下游任务。

---

## 二、大模型 + 类器官图像分析：前沿工作

### 2.1 TransOrga-plus（BMC Biology 2025）⭐

**论文**: "A knowledge-driven deep learning framework for organoid morphological segmentation and characterization"
- 链接: https://link.springer.com/article/10.1186/s12915-025-02411-8
- **核心思路**: 知识驱动的深度学习，结合领域知识和深度学习
- **大规模数据集**: 收集了多种组织类型的类器官数据，比 MultiOrg 大得多
- **时序追踪**: "Using the sequence of organoid images and segmentation results, the tracking module consistently tracks organoids over time"
- **关键启示**: 知识驱动 + 大规模数据 + 时序追踪 = 超越纯数据驱动的性能

### 2.2 Digitalized Organoids（Nature Methods 2025）⭐⭐

**论文**: "Digitalized organoids: integrated pipeline for high-speed 3D analysis of organoid structures"
- 链接: https://www.nature.com/articles/s41592-025-02685-4
- **核心思路**: AI-based multilevel segmentation + cellular topology pipeline
- **3D 分析**: 3D 类器官结构的数字化重建，不只是 2D 切片
- **高通量筛选**: fast and precise plug-and-play image analysis of diverse 3D cellular structures
- **关键启示**: 多层级分割（cell → organoid → topology）比单层级检测更有效

### 2.3 OrganoID（PLOS ONE 2022）

**论文**: "OrganoID: A versatile deep learning platform for tracking and segmentation of organoids in time-lapse microscopy"
- 链接: https://www.biorxiv.org/content/10.1101/2022.01.13.476248.full
- **时序追踪**: Single-organoid tracking accuracy 89% over a 4-day time-lapse study
- **泛化性**: 在多种 organoid 类型的 bright-field 和 fluorescent images 上工作
- **关键启示**: 4 天时序追踪，同一 organoid 的形态变化可以提供远超单帧的信息

### 2.4 LabelFreeTracker（Cell Reports Physical Science 2025）

**论文**: "Label-free cell imaging and tracking in 3D organoids"
- 链接: https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864(25)00121-3
- **3D label-free**: Intestinal organoids 的 3D label-free image analysis
- **细胞膜预测**: Predicts 3D cell membrane from label-free images
- **关键启示**: 不需要荧光标记就能做 3D 分割和追踪

### 2.5 OrganoidTracker 2.0（Nature Methods 2025）

**论文**: "Cell tracking with accurate error prediction"
- 链接: https://www.nature.com/articles/s41592-025-02845-6
- **低错误率**: <0.5% error per cell per frame
- **误差预测**: 能预测自己的追踪误差，知道何时不可信
- **关键启示**: 追踪 + 不确定性量化 = 可靠的下游分析

### 2.6 SAM4Organoid（PMC 2025）

**论文**: "IDCC-SAM: A Zero-Shot Approach for Cell Counting in Organoid Microscopy"
- 链接: https://pmc.ncbi.nlm.nih.gov/articles/PMC11851800
- **Zero-shot**: SAM 专为 organoid 显微镜图像设计的 zero-shot 分割
- **关键启示**: SAM 的 prompt-based 架构天然适合 zero-shot 跨域

### 2.7 AI-Enhanced Patient-Derived Cancer Organoids（MDPI 2025）

**论文**: "AI-Enhanced Patient-Derived Cancer Organoids: Integrating Deep Learning with Phenotypic Profiling"
- 链接: https://www.mdpi.com/2674-1172/4/4/30
- **时序动态表型**: High-content, time-lapse imaging + deep learning → dynamic phenotypic profiling
- **关键启示**: 时序信息 + 深度学习 = 动态表型分析，远超单帧形态学

---

## 三、视觉原语 × 大模型：关键概念

### 3.1 Thinking with Visual Primitives（DeepSeek 2025）⭐⭐⭐

**论文**: DeepSeek-AI, "Thinking with Visual Primitives"
- GitHub: https://github.com/mitkox/Thinking-with-Visual-Primitives
- **核心概念**: 将视觉原语（points, bounding boxes）提升为"思维的最小单元"
- **Reference Gap**: 传统 VLM 只能"看"图像，不能在推理过程中"指向"图像中的特定区域
- **解决方案**: 在 CoT 推理中插入视觉原语操作（crop, zoom in, point to bbox），让模型边推理边看
- **关键启示**: **视觉原语不是特征，而是推理操作的单元**。这和我们的 slot attention 完全不同——slot 是被动特征，TwVP 是主动操作

### 3.2 Insight-V（CVPR 2025）

**论文**: "Insight-V: Exploring Long-Chain Visual Reasoning"
- **CoT for vision**: Chain-of-thought 推理扩展到视觉领域
- **+7.5% 提升**: 在 7 个多模态 benchmark 上平均提升 7.5%
- **关键启示**: 长链推理 + 视觉操作 = 更强的视觉理解

### 3.3 Latent Chain-of-Thought（NeurIPS 2025）

**论文**: "Latent Chain-of-Thought for Visual Reasoning"
- **隐式推理**: 不需要显式文本 CoT，在 latent space 做多步推理
- **关键启示**: 和 CTM（连续思维链）思路一致——不一定要用文本做推理

### 3.4 OpenAI "Thinking with Images"

- 链接: https://openai.com/index/thinking-with-images
- **多模态 CoT**: GPT-4o 能在推理过程中 crop, zoom in, rotate 图像
- **关键启示**: 工业界已验证"视觉操作 + 语言推理"的范式

---

## 四、大模型跨域：CLIP/DINOv2 方向

### 4.1 CLIP-DINOv2 Multimodal Fusion（2025）

**论文**: "Zero-Shot Industrial Anomaly Detection via CLIP-DINOv2 Multimodal Fusion"
- **多模态融合**: CLIP 的文本-视觉对齐 + DINOv2 的视觉特征 → zero-shot 异常检测
- **Stabilized Attention Pooling**: 解决 attention pooling 不稳定的问题
- **关键启示**: CLIP 文本对齐能力 + DINOv2 视觉特征 = 跨域 zero-shot

### 4.2 DINOv2 Meets Text（CVPR 2025）

**论文**: "DINOv2 Meets Text: A Unified Framework for Image-Text Pretraining"
- **统一框架**: DINOv2 视觉编码器 + 文本对齐 → zero-shot 分类 SOTA
- **关键启示**: DINOv2 本身不含文本对齐，加上文本对齐后可以做 zero-shot

### 4.3 Prompt as Knowledge Bank（ICLR 2025）

**论文**: "StructuralGLIP: Zero-Shot Medical Detection via Prompt as Knowledge Bank"
- **零样本医学检测**: 用 VLM 作为知识库，zero-shot 检测医学图像中的目标
- **细粒度对齐**: 目标文本描述和图像区域的细粒度对齐
- **关键启示**: VLM 的领域知识可以作为 zero-shot 检测的先验

### 4.4 Foundation Model Cascades（2025）

**论文**: "Foundation model cascades enable zero-shot microscopy image analysis"
- 链接: https://www.sciencedirect.com/science/article/pii/S146532492600040X
- **级联基础模型**: 多个基础模型级联 → zero-shot 显微镜图像分析
- **最简 prompt**: "count live and dead cells in the image and output results"
- **关键启示**: 基础模型级联（SAM → VLM → counter）可以做 zero-shot 细胞分析

---

## 五、类器官 3D 时序数据集

### 5.1 已发现的 3D/时序数据集

| 数据集 | 类型 | 规模 | 来源 | 特点 |
|--------|------|------|------|------|
| OrganoID | bright-field + fluorescent | 4-day time-lapse | bioRxiv 2022 | 89% tracking accuracy |
| LabelFreeTracker | intestinal 3D label-free | 3D + time | Cell Reports 2025 | 3D 细胞膜预测 |
| TransOrga-plus | 多组织类型 | 大规模 | BMC Biology 2025 | 知识驱动 + 时序追踪 |
| Digitalized Organoids | 3D 多种 organoid | 3D | Nature Methods 2025 | multilevel segmentation + topology |
| OrganoidTracker 2.0 | 3D cell tracking | 3D + time | Nature Methods 2025 | <0.5% error rate |
| Phase contrast time-lapse | label-free | time-lapse | Nature 2024 | 自动 + 手动标注 |

### 5.2 公共数据资源

- **IDR (Image Data Resource)**: https://idr.openmicroscopy.org — 公共生物图像数据仓库
- **MultiOrg (NeurIPS 2024)**: 400+ 张 2D 显微镜图像，60K+ organoid 标注（我们已用）

### 5.3 关键发现

1. **2D → 3D/时序是趋势**: 2025 年的 Nature Methods/BMC Biology 顶级工作都在做 3D 和时序，不只是 2D 单帧
2. **Tracking > Detection**: OrganoidTracker 2.0、OrganoID 都强调追踪（同一 organoid 的时序变化），而非单帧检测
3. **Label-free 是方向**: LabelFreeTracker 证明不需要荧光标记就能做 3D 分割
4. **知识驱动 > 纯数据驱动**: TransOrga-plus 用领域知识引导深度学习，比纯数据驱动更泛化

---

## 六、对我们的启示：新方向

### 6.1 为什么跨域失败——重新诊断

| 假设 | 证据 | 结论 |
|------|------|------|
| DINOv2 特征不够区分 | CTM 全量 16K crops AUC=0.80 < RF-DETR 0.893 | DINOv2 全局特征有上限 |
| 100 crops 太少 | Intestinal 2744 crops AUC=0.67 > 鼠肝 26-60 crops AUC=0.29-0.54 | 数据量正相关 |
| Slot 学到域特定模式 | 跨域 AUC 全部 < 0.70，B1 反预测 | **确认** |
| 缺乏语义对齐 | 没有 text-aligned 表示 | **新发现** |

### 6.2 三条新路径

#### 路径 A：VLM 语义对齐（借鉴 NLP 成功经验）

**核心思想**: NLP 成功是因为预训练模型有通用语义表示。视觉也可以通过 VLM（CLIP/LLaVA）获得文本-视觉对齐的语义空间。

```
DINOv2 features (域特定) → CLIP text-aligned features (通用语义)
```

- 用 CLIP/DINOv2 提取 crop features
- 用 VLM 生成 crop 的文本描述（"a bright round organoid with smooth boundary"）
- 文本描述作为语义特征，跨域更稳健
- **参考**: StructuralGLIP（ICLR 2025）、CLIP-DINOv2 Fusion

#### 路径 B：视觉原语作为推理操作（TwVP 范式）

**核心思想**: 不是把视觉原语当特征，而是当推理操作的单元。

```
传统: image → features → classify TP/FP
TwVP:  image → VLM "is this a real organoid?" → crop/zoom → VLM "check boundary" → 决策
```

- VLM 在推理过程中主动 crop/zoom/point，像人类专家一样检查
- 每一步的 VLM 输出（文本+视觉操作）作为 slot 的输入
- **参考**: DeepSeek TwVP、OpenAI Thinking with Images、Insight-V

#### 路径 C：3D 时序信息利用

**核心思想**: 单帧信息有限，时序信息（同一 organoid 的形态变化）能提供更稳健的特征。

```
单帧: [frame_t] → features → classify
时序: [frame_t-2, t-1, t, t+1, t+2] → temporal features → classify
```

- 同一 organoid 的时序变化（生长、分裂、死亡）是跨域不变的
- 追踪 + 时序特征 → 比单帧检测更泛化
- **参考**: OrganoID（4-day tracking）、OrganoidTracker 2.0、TransOrga-plus

### 6.3 推荐优先级

| 路径 | 可行性 | 数据需求 | 预期效果 | 参考工作 |
|------|--------|----------|----------|----------|
| A: VLM 语义对齐 | 高（已有 VLM） | 现有 crops | 中等 | StructuralGLIP, CLIP-DINOv2 |
| B: TwVP 推理操作 | 中（需 VLM pipeline） | 现有 crops | 高 | DeepSeek TwVP, Insight-V |
| C: 3D 时序 | 低（需视频数据） | 新数据集 | 最高但最慢 | OrganoID, OrganoidTracker 2.0 |

**短期（1-2周）**: 路径 A — 用 CLIP 替换 DINOv2 做 crop features，验证跨域是否改善
**中期（1-2月）**: 路径 B — 构建 VLM 推理 pipeline，让 VLM 主动检查 crops
**长期（3-6月）**: 路径 C — 收集/寻找 3D 时序 organoid 数据集

---

## 七、参考文献

### 类器官 + AI

1. TransOrga-plus, BMC Biology 2025. https://link.springer.com/article/10.1186/s12915-025-02411-8
2. Digitalized Organoids, Nature Methods 2025. https://www.nature.com/articles/s41592-025-02685-4
3. OrganoID, PLOS ONE 2022. https://www.biorxiv.org/content/10.1101/2022.01.13.476248
4. LabelFreeTracker, Cell Reports Physical Science 2025. https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864(25)00121-3
5. OrganoidTracker 2.0, Nature Methods 2025. https://www.nature.com/articles/s41592-025-02845-6
6. SAM4Organoid, PMC 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC11851800
7. AI-Enhanced Cancer Organoids, MDPI 2025. https://www.mdpi.com/2674-1172/4/4/30
8. MultiOrg, NeurIPS 2024. https://neurips.cc/virtual/2024/poster/97645

### 视觉原语 + LLM

9. Thinking with Visual Primitives, DeepSeek 2025. https://github.com/mitkox/Thinking-with-Visual-Primitives
10. Insight-V, CVPR 2025. https://cvpr.thecvf.com/virtual/2025/poster/34306
11. Latent CoT for Visual Reasoning, NeurIPS 2025. https://neurips.cc/virtual/2025/poster/120293
12. OpenAI Thinking with Images. https://openai.com/index/thinking-with-images
13. Tool-augmented Thinking with Images for Medical VLM, arXiv 2025. https://arxiv.org/html/2512.14157v1

### CLIP/DINOv2 跨域

14. CLIP-DINOv2 Multimodal Fusion, 2025. https://www.researchgate.net/publication/398367800
15. DINOv2 Meets Text, CVPR 2025. https://cvpr.thecvf.com/virtual/2025/poster/33482
16. StructuralGLIP (Prompt as Knowledge Bank), ICLR 2025. https://openreview.net/forum?id=l0t2rumAvR
17. Foundation Model Cascades for Microscopy, 2025. https://www.sciencedirect.com/science/article/pii/S146532492600040X
18. TV-SAM, 2025. https://www.sciopen.com/article/10.26599/BDMA.2024.9020058

### 3D/时序数据集

19. IDR (Image Data Resource). https://idr.openmicroscopy.org
20. Phase contrast time-lapse, Nature 2024. https://www.nature.com/articles/s44303-024-00046-y
21. Deep learning predicts retinal organoid differentiation, PLOS Biology 2025. https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003597

---

## 七、补充调研（2026-07-14 下午）

### 7.1 训练破坏 zero-shot 泛化的理论解释

**我们的发现**：SupCon 训练 slot 和 CoOp 训练 prompt 都破坏 CLIP zero-shot 跨域能力。

**文献支撑**：
- **"Fine Tuning without Catastrophic Forgetting"** (arXiv 2501.15377, 2025)：微调导致灾难性遗忘是已知问题
- **"Quantified Task Misalignment to Inform PEFT"** (ICLR 2024)：CLIP 的 task difficulty 和 PEFT 性能有系统关系
- **"Revisiting Catastrophic Forgetting in LLM"** (EMNLP 2024)：大模型微调后的遗忘模式
- **"Trade-offs in Cross-Domain Generalization of Foundation Model"** (arXiv 2509.14921, 2025)：
  - **"finetuned models suffer from over-specialization, especially when finetuned for complex tasks"**
  - Zero-shot 和 linear-probe 评估显示 task-specific adaptation 影响跨域
- **"Linearization Explains Fine-Tuning in LLM"** (NeurIPS 2025)：微调的线性化分析

**结论**：我们的发现和文献一致——**监督训练破坏 foundation model 的 zero-shot 泛化**。这不是 bug，是 foundation model 的固有特性。

### 7.2 Prompt 工程方向（不训练参数）

**文献**：
- **"Cluster-Aware Prompt Ensemble Learning for Few-Shot Vision"** (arXiv 2510.09867, 2025)：
  - 多 prompt 集成 + 聚类感知 → 提升 CLIP zero-shot
- **"A Simple Zero-shot Prompt Weighting Technique"** (PMLR 2023)：
  - 给 prompt pool 里每个 prompt 自动打分，加权聚合
- **"Effective Zero-shot Classification with Hundreds of Multi-modal CLIPs"** (ACM 2025)：
  - 多 CLIP 模型集成
- **"Prompt Ensemble in Zero-shot Classification using CLIP"** (Medium 2025)：
  - 实践指南 + PyTorch 实现
  - GitHub: https://github.com/satojkovic/clip-pytorch

**MaPLe** (CVPR 2023)：
- Multi-modal Prompt Learning：同时学视觉和语言 prompt
- 比单模态 prompt 更好
- 但仍然是训练——可能也破坏 zero-shot

**方案**：不训练的 prompt 集成——用 5-20 个不同 prompt 做 ensemble voting，类似我们 A4 的 5 组 prompt，但加权聚合而非选最优。

### 7.3 Test-Time Training (TTT)

**文献**：
- **"The Surprising Effectiveness of TTT for Few-Shot"** (PMLR 2025, arXiv 2411.07279)：
  - 推理时临时更新模型参数，用测试样本自监督
  - **不需要标注数据，在测试样本上自适应**
  - "TTT drastically improves LM's few-shot learning on out-of-distribution tasks"
- **"Forget, Anticipate and Adapt: TTT for Long Videos"** (arXiv 2606.26515, 2025)：
  - 视频时序的 TTT
- **"Learning to Learn at Test Time"** (TTT-LM, 2025)：
  - TTT 作为 RNN 的隐藏状态

**方案**：TTT 在每个测试 crop 上自适应 CLIP prompt——不需要标注，用 crop 自身的自监督信号（如旋转预测）。可能避免过拟合到源域。

### 7.4 3D 时序数据集补充

**新发现**：
- **OrgLine (Cell Reports Methods, 2026)**：4 个独立数据集的时序追踪评估
  - https://www.sciencedirect.com/science/article/pii/S2667237526001827
- **3D Organoid Time-Lapse Nuclei Segmentation** (image.sc forum)：
  - FRET-based protein activity 时序追踪
  - https://forum.image.sc/t/3d-organoid-time-lapse-nuclei-segmentation-and-tracking-workflow/120475
- **OrganoidTracker 2.0** (GitHub hrlblab)：
  - **SAM2-powered 3D organoid tracking**，zero-shot 分割 + 时序追踪
  - https://github.com/hrlblab/OrganoidTracker
- **OrganoidTracker** (PMC, PLOS ONE)：
  - 3D cell tracking + machine learning + manual correction
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC7580893
  - Website: https://organoidtracker.org
- **LabelFreeTracker** (Patterns 2025)：
  - 3D label-free intestinal organoid tracking
  - https://www.sciencedirect.com/science/article/pii/S2666386425001213

**3D 时序数据集清单**（更新版）：

| 数据集 | 类型 | 规模 | 特点 | 来源 |
|--------|------|------|------|------|
| OrganoID | 2D 时序 | 4-day tracking | 肠道 organoid 生长追踪 | PMC9645660 |
| OrganoidTracker 2.0 | 3D 时序 | SAM2-powered | zero-shot 分割+追踪 | GitHub hrlblab |
| OrganoidTracker | 3D 时序 | intestinal | ML + manual correction | PLOS ONE |
| LabelFreeTracker | 3D label-free | intestinal | FRET protein activity | Patterns 2025 |
| OrgLine | 4 数据集 | 时序追踪 | detector-guided SAM2 | Cell Reports Methods |
| Digitalized Organoids | 3D | multilevel | 拓扑+筛查 | Nature Methods 2025 |
| Phase contrast time-lapse | 2D | cell tracking | 标注完整 | Nature 2024 |
| Retinal organoid | 时序 | differentiation | 预测分化 | PLOS Biology 2025 |

### 7.5 下一步实验方案（更新）

基于"训练破坏泛化"的发现：

| 方案 | 方法 | 训练？ | 预期 |
|------|------|--------|------|
| A5 | Prompt 集成（20 prompt 加权） | 否 | zero-shot 0.73 → 0.78+ |
| A6 | Prompt 自动权重（PMLR 2023） | 否 | 自动选最优 prompt 权重 |
| B2 | VLM 推理（API 待恢复） | 否 | VLM 推理操作 |
| C1 | 3D 时序 + OrganoID 4-day | — | 时序变化跨域不变 |
| TTT | Test-Time Training | 自监督 | 可能避免过拟合 |

### 7.6 补充参考文献

22. Fine Tuning without Catastrophic Forgetting. arXiv 2501.15377, 2025.
23. Quantified Task Misalignment to Inform PEFT. ICLR 2024.
24. Trade-offs in Cross-Domain Generalization of FM. arXiv 2509.14921, 2025.
25. Cluster-Aware Prompt Ensemble Learning. arXiv 2510.09867, 2025.
26. A Simple Zero-shot Prompt Weighting. PMLR 2023.
27. The Surprising Effectiveness of TTT. PMLR 2025.
28. MaPLe: Multi-Modal Prompt Learning. CVPR 2023.
29. OrganoidTracker 2.0 (SAM2-powered). GitHub hrlblab, 2025.
30. OrgLine: detector-guided SAM2. Cell Reports Methods, 2026.
31. LabelFreeTracker. Patterns, 2025.
