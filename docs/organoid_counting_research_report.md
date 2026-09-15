# 类器官计数（Organoid Counting）应用场景调研报告

> 为论文 "SAM2-Guided Self-Distillation for Cross-Domain Organoid Detection: A Federated Learning Platform" 补充明确应用场景的调研

---

## 一、论文现状分析

### 1.1 当前论文的强项
- **技术深度**：SAM2-guided self-distillation + RF-DETR + CLIP zero-shot + Hard-gate FL
- **实验严谨**：三层证据链（simulation +28.22% → real cross-domain +9.10% → MPM pilot +2.21%）
- **负面结果诚实**：SAM2 fine-tuning 有害、FP suppression 失败、VLM 不如 detector confidence
- **HCMI 真实证据**：400 张图、4 个国际中心、25 种癌症类型，zero detection failure

### 1.2 核心短板：缺少明确的应用场景

论文目前的问题：

| 问题 | 具体表现 |
|------|---------|
| **应用目标模糊** | 论文标题说 "organoid detection"，但 detection 本身不是应用——它服务于什么？药物筛选？质控？精准医疗？ |
| **动机不够锐利** | 三个挑战（隐私、跨域、标注稀缺）虽然成立，但读者会问："为什么需要在多个实验室之间做 organoid detection？" |
| **与现有工具的对比缺失** | 没有提到 OSCAR、CellDrop、Corning Cell Counter 等已有计数工具，无法体现差异化 |
| **FL 的必要性论证偏理论** | HCMI 实验证明了跨中心差异，但没有回答"计数场景下 FL 到底带来多少实际价值" |

### 1.3 为什么"类器官计数"是最佳应用场景

**计数是 organoid 研究中最基础、最高频、最商业化的操作**：

1. **药物筛选**：每个 384-well plate 有 384 个孔，每孔需要计数 organoid 数量来评估药物效果
2. **质控（QC）**：培养过程中需要计数来评估 organoid 生长状态和批次一致性
3. **精准医疗**：PDO 药敏测试需要计数来量化肿瘤细胞对药物的响应
4. **跨实验室一致性**：不同实验室的计数结果不可比是行业痛点

**计数正好命中论文的三个核心贡献**：
- 跨域问题 → 不同实验室的成像条件导致计数模型迁移失败
- 隐私问题 → 患者 PDO 图像不能跨院共享
- 标注稀缺 → 罕见癌症 organoid 样本太少，无法训练专用计数模型

---

## 二、文献调研

### 2.1 现有类器官计数方法

| 工具/方法 | 来源 | 技术路线 | 局限性 |
|-----------|------|----------|--------|
| **OSCAR** | Burnell et al., *Cell Rep Methods* 2025 | Instance segmentation + brightfield | 单实验室训练，未验证跨域能力 |
| **CellDrop FLi** | DeNovix (2025) | ML-based Organoid App，商用细胞计数仪 | 硬件绑定，无法跨平台部署 |
| **Corning Cell Counter** | Corning | 商用软件扩展模块 | 闭源，仅适配 Corning 硬件 |
| **GelCount** | Oxford Optronix | 3D object counting | ±10% size 估计误差 → +90%/−45% 计数误差 |
| **Incucyte Organoid Analysis** | Sartorius | Label-free 自动成像+分析 | 仅适配 Incucyte 平台 |
| **OrganoSeg2** | Nature 2026 | Learning-free, 多窗口自适应阈值 | 无跨域验证，同域 only |
| **LGBP-Net** | ScienceDirect 2026 | CNN + 频域特征融合 | 同域评估，无跨域 |
| **Deliod** | Nature Sci Rep 2025 | YOLOv8s 轻量检测 | 单一实验室数据 |
| **Orga-Dete** | Appl. Sci. 2025 | YOLOv11 + BiFPN + MPCA | 仅 lung organoid |
| **Enhanced U-Net** | Bioengineering 2025 | Attention gates | 分割而非计数，无跨域 |

### 2.2 关键文献发现

#### 发现 1：计数误差的量级问题
> "An error of only ±10% in the visual estimate of organoid size across samples can result in +90% to −45% counting error."
> — Oxford Optronix

这意味着跨实验室的计数差异不是小问题——10% 的尺寸感知差异会导致近 2 倍的计数误差。

#### 发现 2：现有工具全部是单域的
- OSCAR（2025 年最新）：使用 instance segmentation，但**仅在训练数据的成像条件下验证**
- CellDrop / Corning：商用硬件绑定，**无法在不同显微镜之间迁移**
- OrganoSeg2 / LGBP-Net / Deliod：学术论文**均未做跨域评估**

**这是论文的核心空白点**：没有任何现有工作解决 organoid counting 的跨域问题。

#### 发现 3：MIOR 标准化倡议
2025 年 11 月发布的 *Minimum Information about Organoid Research (MIOR)* 标准呼吁：
- 统一报告 organoid 培养和分析条件
- 强调跨实验室可重复性

**这为论文的 FL + 跨域 counting 提供了政策层面的支撑**。

#### 发现 4：高通量药物筛选的需求
- 384-well plate 药物筛选需要**每孔计数 organoid**
- 高通量筛选（HTS）要求**自动化、标准化、跨批次可比**
- 目前依赖人工计数或单实验室模型，**无法跨中心协作**

### 2.3 计数 vs 检测：论文需要做的调整

论文目前定位为 "organoid detection"，但：

| 维度 | Detection（当前） | Counting（建议） |
|------|-------------------|------------------|
| 输出 | bounding box + confidence | organoid 数量 + 位置 |
| 评估指标 | mAP@0.5 | Counting Accuracy / MAE / R² |
| 应用驱动 | 弱 | 强（药物筛选、QC、药敏测试） |
| 商业价值 | 间接 | 直接（计数是付费功能） |
| 跨域需求 | 有但不够锐利 | 极其锐利（计数不可比 = 实验不可重复） |

**建议**：不改变技术核心，但将应用场景从 "detection" 重新定位为 "cross-domain organoid counting for multi-center drug screening"。

---

## 三、市场调研

### 3.1 类器官市场规模

| 报告来源 | 2024/2025 市场规模 | 2030/2034/2035 预测 | CAGR |
|----------|---------------------|----------------------|------|
| Grand View Research | $804M (2024) | $2,716M (2030) | 22.5% |
| Fortune Business Insights | $1.18B (2025) | $5.71B (2034) | ~19.4% |
| Towards Healthcare | $1.22B (2025) | $9.77B (2035) | 23.13% |
| Mordor Intelligence | $1.20B (2025) | $3.29B (2031) | 18.4% |
| Strategic Market Research | $3.7B (2024) | $10.94B (2030) | 19.8% |
| Research and Markets | $1.28B (2023) | $4.22B (2029) | ~21.9% |

**关键结论**：
- 全球类器官市场 2025 年约 **$1-4B**，2030 年约 **$3-11B**，CAGR **19-23%**
- 亚太地区增速最快（CAGR 23.89%，Grand View Research）
- 自动化 organoid 培养市场：$421M (2025) → $1,258M (2034)

### 3.2 类器官药物筛选市场

| 细分市场 | 规模 | 增速 |
|----------|------|------|
| Organoid Drug Discovery Platforms | $1.12B (2025) → $4.29B (2033) | ~18.4% CAGR |
| Precision Medicine | $118B (2025) → $134B (2026) | 7.5% CAGR |
| High-throughput Screening | 快速增长中 | — |

### 3.3 类器官计数工具竞争格局

#### 商业产品

| 产品 | 公司 | 模式 | 价格区间 | 跨域能力 |
|------|------|------|----------|----------|
| CellDrop FLi | DeNovix | 硬件+ML App | $15K-30K | ❌ 硬件绑定 |
| Corning Cell Counter | Corning | 硬件+软件 | $10K-20K | ❌ 闭源 |
| GelCount | Oxford Optronix | 硬件 | $20K-40K | ❌ 硬件绑定 |
| Incucyte | Sartorius | 活细胞成像系统 | $50K-100K+ | ❌ 平台绑定 |
| ImageXpress | Molecular Devices | HCS 系统 | $100K-300K+ | ❌ 平台绑定 |
| AI Decision Making | Molecular Devices | AI+自动化 | 企业级 | ❌ 单平台 |

#### 学术工具

| 工具 | 论文 | 开源 | 跨域能力 |
|------|------|------|----------|
| OSCAR | Burnell 2025, *Cell Rep Methods* | 在线工具 | ❌ 单域 |
| OrganoSeg2 | Nature 2026 | 开源 | ❌ 同域 |
| LGBP-Net | ScienceDirect 2026 | 未知 | ❌ 同域 |
| Deliod | Sci Rep 2025 | 开源 | ❌ 单一实验室 |
| Orga-Dete | Appl. Sci. 2025 | 未知 | ❌ 仅 lung |

#### 关键发现

**没有任何现有产品或学术工具解决了跨实验室计数问题。**

- 所有商业产品都是**硬件绑定**的——买了一套设备就只能在那套设备上计数
- 所有学术工具都是**单域训练**的——没有跨域验证
- 这正是论文 Organoid-FL 的**独特价值主张（UVP）**

### 3.4 市场需求验证

#### 需求 1：高通量药物筛选
- 384-well plate 每孔需要计数 → 一个实验需要 384 次计数
- 大型药企每年进行数千次筛选 → **百万级计数需求**
- 人工计数：每孔 5-15 分钟 → 384 孔 = 32-96 小时（不可行）

#### 需求 2：多中心临床试验
- PDO 药敏测试正在进入临床试验阶段
- 多中心试验需要**计数结果可比**
- 目前不同中心的计数结果**不可比** → MIOR 标准化倡议

#### 需求 3：QC 标准化
- 类器官培养 QC 需要计数来评估批次一致性
- 跨批次、跨实验室的 QC 计数**目前无标准方法**
- MIOR (2025) 明确要求标准化计数报告

---

## 四、论文修改建议

### 4.1 重新定位应用场景

**当前**：
> "We present Organoid-FL, a federated learning platform that integrates a novel SAM2-guided self-distillation strategy for cross-domain organoid detection without requiring additional annotations."

**建议修改为**：
> "We present Organoid-FL, a federated learning platform for **multi-center organoid counting** that integrates a novel SAM2-guided self-distillation strategy for cross-domain organoid detection without requiring additional annotations. **Organoid counting is the foundational operation in drug screening, quality control, and precision medicine, yet current automated counting tools are trained on single-lab data and fail to generalize across institutions—a gap directly addressed by federated learning.**"

### 4.2 新增 Section：Application Scenario — Multi-Center Organoid Counting

建议在 Introduction 之后、Related Work 之前新增一个 Section（约 0.5-1 页）：

```markdown
## 2. Application Scenario: Multi-Center Organoid Counting

Organoid counting—determining the number of organoids in a well—is the most
fundamental quantitative operation in organoid-based research, directly supporting
three high-impact applications:

1. **Drug screening**: 384-well plate-based drug screening requires per-well
   organoid counts to quantify drug response (IC50, GR50). A single screening
   campaign involves 10⁴–10⁶ counting operations.

2. **Quality control (QC)**: Organoid culture QC requires counting to assess
   batch consistency, growth rate, and structural integrity. The MIOR standard
   (2025) explicitly calls for standardized counting across laboratories.

3. **Precision medicine**: Patient-derived organoid (PDO) drug sensitivity
   tests require counting to measure tumor cell response to treatment—
   currently limited by inter-lab counting variability.

**The cross-lab counting problem.** Despite the availability of automated
counting tools (OSCAR, CellDrop, Corning Cell Counter), all existing solutions
are trained and validated on single-lab data. Our HCMI evaluation (Section 4.8)
demonstrates that models trained at one institution produce zero detections at
another—making cross-lab counting results non-comparable. This is not merely
a technical limitation but a practical barrier to multi-center clinical trials
and collaborative drug screening programs.

**Why federated learning?** FL enables institutions to collaboratively train
a counting model without sharing patient-derived organoid images, directly
addressing HIPAA/GDPR constraints while improving cross-domain generalization.
Our hard-gate aggregation strategy (Section 3.4) achieves 0.92 mAP@0.5 across
three heterogeneous mouse liver batches, demonstrating that FL can produce
a model that generalizes across imaging conditions—a capability no existing
counting tool provides.
```

### 4.3 在 Discussion 中增加与现有计数工具的对比

```markdown
### 5.X Comparison with Existing Organoid Counting Tools

Table X: Comparison with existing organoid counting tools.

| Tool | Type | Cross-domain | Multi-center | Privacy | Open-source |
|------|------|-------------|-------------|---------|-------------|
| OSCAR [Burnell 2025] | ML (instance seg) | ✗ | ✗ | ✗ | Partial |
| CellDrop [DeNovix] | Hardware+ML | ✗ | ✗ | ✗ | ✗ |
| Corning Counter | Hardware | ✗ | ✗ | ✗ | ✗ |
| OrganoSeg2 [Nature 2026] | Thresholding | ✗ | ✗ | ✗ | ✓ |
| Deliod [Sci Rep 2025] | YOLOv8s | ✗ | ✗ | ✗ | ✓ |
| **Organoid-FL (ours)** | **FL+SAM2+RF-DETR** | **✓** | **✓** | **✓** | **✓** |

Organoid-FL is the first organoid counting platform that addresses cross-domain
generalization through federated learning. Unlike hardware-bound commercial
solutions, our platform operates on any brightfield microscopy image. Unlike
single-lab academic tools, our FL framework enables multi-institutional
collaboration without data sharing.
```

### 4.4 在 Introduction 中强化动机

建议在 Introduction 的三个挑战之前，加一段应用背景：

```markdown
Organoid counting is the foundational quantitative operation in organoid-based
research, supporting drug screening (10⁴–10⁶ counts per campaign), quality
control (MIOR standardization, 2025), and precision medicine (PDO drug
sensitivity tests). However, current automated counting tools are trained on
single-lab data and fail to generalize across institutions—our HCMI evaluation
shows zero detections when a model trained at one center is deployed at another
(Section 4.8). This cross-lab counting gap directly motivates our federated
learning approach.
```

### 4.5 补充计数评估指标

当前论文用 mAP@0.5 评估检测，建议增加计数特定的指标：

| 指标 | 公式 | 意义 |
|------|------|------|
| Counting Accuracy | 1 - \|N_pred - N_gt\| / N_gt | 计数准确率 |
| MAE | mean(\|N_pred - N_gt\|) | 平均绝对误差 |
| R² | — | 预测计数与真实计数的相关性 |
| Per-well F1 | — | 每孔级别的 F1（药物筛选场景） |

在 MultiOrg 数据集上，可以计算：
- 每张图的 ground truth organoid 数量
- 模型预测的 organoid 数量
- Counting Accuracy / MAE / R²

### 4.6 HCMI 实验的计数视角重新解读

当前 HCMI 实验的解读是"detection failure"，建议增加 counting 视角：

```markdown
**Counting implication.** SAM2 zero-shot finds ~20 organoids per image across
four HCMI centers, while RF-DETR finds 0. This means a counting model trained
at one institution would report 0 organoids for all 400 HCMI images—
a complete counting failure. Even if a model detects some organoids, the 35%
density difference and 92% size difference across centers would produce
non-comparable counts. This demonstrates that cross-center organoid counting
is not a detection problem but a domain adaptation problem, directly
motivating our FL + self-distillation approach.
```

---

## 五、总结与行动项

### 5.1 核心判断

**"类器官计数"是论文最理想的应用场景**，因为：

1. **技术对齐**：论文的 detection → counting 只需加一个计数层（NMS 后的 box 数量）
2. **市场对齐**：计数是 $1-4B organoid 市场的基础操作
3. **差异化**：没有任何现有工具解决跨实验室计数问题
4. **FL 必要性**：计数不可比 = 多中心实验不可重复 → FL 是刚需
5. **政策支撑**：MIOR (2025) 标准化倡议明确要求跨实验室计数一致性

### 5.2 建议的修改清单

| 优先级 | 修改项 | 工作量 |
|--------|--------|--------|
| P0 | Introduction 加一段应用背景（计数场景） | 0.5h |
| P0 | 新增 Section 2: Application Scenario | 2h |
| P0 | Discussion 加与现有计数工具对比表 | 1h |
| P1 | 补充计数评估指标（Counting Accuracy, MAE, R²） | 3h（需跑实验） |
| P1 | HCMI 实验增加 counting 视角解读 | 1h |
| P2 | Abstract 和 Title 微调（detection → counting） | 0.5h |
| P2 | Related Work 加 organoid counting 工具综述 | 1h |

### 5.3 一句话总结

> 论文的技术核心（SAM2 self-distillation + FL + RF-DETR）不需要改，但需要一个锐利的应用场景来锚定读者——**"跨实验室类器官计数"**正好是这个锚点：它让三个技术贡献（跨域、隐私、标注稀缺）从"技术问题"变成"实际问题"，让审稿人不再问 "why organoid detection" 而是问 "how fast can you count"。

---

## 参考来源

### 文献
1. Burnell SEA. "OSCAR is an online ML-powered tool for organoid cell counting using bright-field images." *Cell Rep Methods* 2025;5(12):101251.
2. "OrganoSeg2: Learning-free organoid segmentation and quantification." *Nature* 2026.
3. "LGBP-Net: Learnable Gaussian band pass fusion for organoid segmentation." *ScienceDirect* 2026.
4. "Deliod: a lightweight detection model for intestinal organoids." *Sci Rep* 2025.
5. "Orga-Dete: YOLOv11 with BiFPN and MPCA for lung organoid detection." *Appl Sci* 2025.
6. "Enhanced U-Net-based deep learning model for automated organoid segmentation." *Bioengineering* 2025.
7. "MIOR: Minimum Information about Organoid Research." *SciOpen* 2025.
8. Castiglione H. "Towards a quality control framework for cerebral cortical organoids." *Nature* 2025.
9. ElHarouni D. "A compendium of next-generation patient-derived models for diverse cancers." *Nature* 2026. (HCMI)
10. Choi JS. "Protocol for high-content drug screening using tumor organoids." *Cell* 2025.

### 市场报告
1. Grand View Research — Human Organoids Market Report 2024-2030
2. Fortune Business Insights — Human Organoids Market 2025-2034
3. Towards Healthcare — Organoids and Spheroids Market 2025-2035
4. Mordor Intelligence — Organoids Market 2025-2031
5. DelveInsight — Automated Organoid Culturing Market 2025-2034
6. DataBridge — Organoid Drug Discovery Platforms Market 2025-2033

### 商业产品
1. DeNovix CellDrop FLi — https://www.denovix.com/tn-248-automated-organoid-counting
2. Corning Cell Counter Organoid Software — https://www.corning.com
3. Oxford Optronix GelCount — https://www.oxford-optronix.com
4. Sartorius Incucyte Organoid Analysis — https://www.sartorius.com
5. Molecular Devices AI Decision Making — https://www.moleculardevices.com
6. Axion BioSystems Organoid Counting Module — https://axionbiosystems.com
