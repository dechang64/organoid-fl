# 视觉原语实验方案 v1.0

> 2026-07-08 曼卿起草，冬生确认后逐步落实

## 1. 背景与动机

### 1.1 bbox 范式到天花板

鼠肝 v2 实验确认了 bbox 检测范式的局限：
- B1 (2592×1944, 大目标): F1=1.00 — 完美
- B2 (4000×3000, 小目标): F1=0.80 — 有 FP
- B3 (4000×3000, 小目标): F1=0.75 — 有 FP
- Central 合并反而不如 B1 单独 (0.71 < 1.00) — 域差异拖累

SAHI/NMS/Soft-NMS/Ensemble/Orga-Dete 迁移均无法突破。**bbox 对不规则类器官只能参考定位，不能做形态学判断。**

### 1.2 范式转换方向

从"bbox 检测 + 权重聚合"转向"**视觉原语 + 分布聚合**"：
- 检测：bbox → SAM2 mask（像素级精确轮廓）
- 特征：bbox 坐标 → 形态学 primitive 向量（面积/圆度/solidity/长宽比/离心率）
- 推理：单次前传 → CTM 持续思考（多步推理，后步修正前步）
- FL：聚合模型权重 → 聚合 primitive 分布（不共享权重，共享知识）

### 1.3 与面上项目的关系

本方案对应面上项目研究内容 1（类器官微观形态的多模态向量化表征与增强），并为研究内容 2（LLM 质量评估）和 3（联邦隐私检索）提供数据基础。

## 2. 总体架构

```
原图
 │
 ├── Step 1: 检测 (RF-DETR) → bbox
 │
 ├── Step 2: 分割 (SAM2 zero-shot) → 无类别 mask
 │
 ├── Step 3: 形态学特征提取 → primitive 向量 (768 维)
 │           面积/周长/圆度/solidity/extent/长宽比/离心率/纹理特征
 │
 ├── Step 4: VLM 语义确认 (GLM-4V) → 质量评估
 │           语义一致性 + 特征完整性 + 分布匹配度
 │
 ├── Step 5: Diffusion 生成增强 (条件引导) → 补充流形稀疏区域
 │
 └── Step 6: FL 聚合 primitive 分布 (FedCtx HNSW)
             向量空间对齐 → 神经同步 → 全局分布
```

**CTM 持续思考**：Step 1-4 不是串行管线，而是 internal ticks 循环。Step 4 (VLM) 不确定时，回到 Step 1 重新检测，形成"持续思考"闭环。

## 3. 各模块实验设计

### 3.1 Step 1-2: 检测 + 分割（已完成 ✓）

**现状**：
- RF-DETR Small: 鼠肝 B1 F1=1.00, B2=0.80, B3=0.75
- SAM2 zero-shot: mask F1 = bbox F1（不增不减检测）
- MultiOrg: RF-DETR 77.8% 超 SOTA

**结论**：Step 1-2 已验证，直接复用。SAM2 zero-shot 不需要微调（4 点粗糙 GT 微调 = 负优化）。

### 3.2 Step 3: 形态学特征提取

**输入**：SAM2 mask（二值图）
**输出**：768 维 primitive 向量

**特征清单**（每个 organoid 一个向量）：

| 类别 | 特征 | 维度 | 说明 |
|------|------|------|------|
| 形状 | 面积 | 1 | mask 像素数 |
| 形状 | 周长 | 1 | 轮廓长度 |
| 形状 | 圆度 | 1 | 4πA/P² |
| 形状 | Solidity | 1 | 凸包面积比 |
| 形状 | Extent | 1 | bbox 面积比 |
| 形状 | 长宽比 | 1 | 最小外接矩形 |
| 形状 | 离心率 | 1 | 椭圆拟合 |
| 纹理 | 均值 | 1 | mask 区域像素均值 |
| 纹理 | 标准差 | 1 | mask 区域像素 std |
| 纹理 | 熵 | 1 | 灰度直方图熵 |
| 纹理 | GLCM 对比度 | 1 | 灰度共生矩阵 |
| 纹理 | GLCM 同质性 | 1 | |
| 纹理 | GLCM 能量 | 1 | |
| 纹理 | LBP 直方图 | 26 | 局部二值模式 (256→26 bin) |
| 深度 | DINOv2 embedding | 768 | ViT-B/14 CLS token |
| — | padding | — | 补齐到 768 维 |

**实验**：
1. 对鼠肝 B1/B2/B3 + MultiOrg 所有 mask 提取 primitive 向量
2. UMAP 可视化：不同 batch/数据集的 primitive 分布
3. 统计检验：primitive 分布是否能区分 TP/FP（之前 DINOv2 失败了，但形态学特征不同）
4. 跨域稳定性：B1 的 primitive 分布 vs B2/B3 的，看域偏移程度

**评估**：TP vs FP 的 primitive 分布可分性（PR-AUC > 0.5 = 有效）

### 3.3 Step 4: VLM 语义确认

**输入**：mask + 原图裁剪
**输出**：质量评估三元组 (语义一致性, 特征完整性, 分布匹配度)

**方案**：
- 模型：GLM-4V（通过 z-ai-web-dev-sdk 调用）
- Prompt: "这是显微镜下的类器官图像，绿色区域是模型分割的 mask。请判断：(1) 这个 mask 是否准确覆盖了一个类器官？(2) mask 的边界是否精确？(3) 这个形态是否典型？"
- 输出：三维评分 (0-1)，保留不确定性（不硬删，只降权）

**实验**：
1. 对鼠肝 B2 的 FP（3 个）和 TP（6 个）调用 VLM
2. 看 VLM 评分能否区分 TP/FP
3. 对比 VLM 评分 vs DINOv2 embedding 的区分度（之前 DINOv2 PR-AUC=0.29）

**CTM 持续思考**：VLM 不确定 (评分 0.4-0.6) 时，触发重新检测——调整 RF-DETR threshold 或 SAM2 prompt，重新生成 mask，再评估。循环直到评分 > 0.6 或达到最大 ticks（5 次）。

### 3.4 Step 5: Diffusion 生成增强

**动机**：鼠肝 B2 只有 6 张训练图，FL 有效区间需要适度异质但不能太极端。Diffusion 生成可以填充流形稀疏区域。

**方案**：
1. 对现有 organoid mask 提取 DINOv2 embedding (768 维)
2. UMAP 降维到 2D，识别稀疏区域
3. 在稀疏区域采样目标点，用条件引导 Diffusion 生成合成 organoid 图像
4. 生成条件：面积/圆度/solidity 目标值 → 控制 mask 形态
5. 合成图像 → SAM2 分割 → 验证 mask 形态是否符合目标

**实验**：
1. 鼠肝 B2 (6 张) → 生成到 12 张/24 张 → 重训 RF-DETR → 看精度提升
2. 对比：真实 6 张 vs 合成 6 张 vs 真实+合成 12 张
3. 流形覆盖度：UMAP 上生成前后分布的变化

**注意**：
- 合成数据必须标注 GT（用合成条件作为 GT）
- 合成数据质量用 Step 4 VLM 评估
- 真实:合成比例从 1:0 到 1:3 扫描最优比

### 3.5 Step 6: FL 聚合 primitive 分布

**输入**：各 client 的 primitive 向量集合
**输出**：全局 primitive 分布（HNSW 索引）

**方案**：
1. 各 client 本地提取所有 organoid 的 primitive 向量
2. 向量空间对齐：用对抗学习（domain adaptation）对齐各 client 的 primitive 空间
3. 聚合：各 client 的 primitive 向量上传到 FedCtx HNSW（不共享原图，只共享向量）
4. 全局分布：HNSW 支持范围查询和 k-NN 检索

**FL 策略对比**：
- FedAvg (权重聚合) — 基线
- EWA (熵加权) — 信号区分度依赖
- **Distribution aggregation (新)** — 聚合 primitive 分布，不聚合权重

**实验**：
1. 鼠肝 B1/B2/B3 作为 3 个 client，各自有 primitive 分布
2. 聚合后全局分布 vs 各 client 本地分布
3. 新 organoid 检索：在全局分布中 k-NN 检索，看召回率
4. 隐私评估：从 primitive 向量能否反推原图（反演攻击）

**增量索引**（JD 启发）：
- 新实验室加入时，实时插入 primitive 向量到 HNSW
- PQ 量化：768 维 → 64 字节，适合联邦传输
- 有效性 bitmap：管理离线 client

## 4. 数据集

| 数据集 | 用途 | 张数 | 状态 |
|--------|------|------|------|
| 鼠肝 B1/B2/B3 | Step 1-4 验证 | 40 | ✓ 已有 |
| MultiOrg | Step 1-4 大规模验证 | 411 | ✓ 已有 |
| Intestinal (Deliod) | 跨器官验证 | — | ✓ 已有 |
| Diffusion 合成 | Step 5 数据增强 | 按需 | 待生成 |
| 未来数据集 | 跨域泛化验证 | — | 待收集 |

## 5. 评估指标

| 层面 | 指标 | 目标 |
|------|------|------|
| 检测 | mAP50, F1 | B1=1.00 (已达), B2>0.85 |
| 分割 | mask IoU, mask F1 | >0.85 |
| 形态学 | TP/FP PR-AUC | >0.5 (DINOv2=0.29 失败) |
| VLM | TP/FP 区分 AUC | >0.7 |
| Diffusion | 流形覆盖度 | UMAP 稀疏区域减少 |
| FL | 全局 k-NN 召回率 | >0.9 |
| 隐私 | 反演攻击成功率 | <5% |

## 6. 时间线

| 阶段 | 内容 | 预计时间 | 依赖 |
|------|------|---------|------|
| Phase 1 | Step 3 形态学特征提取 + UMAP 可视化 | 1 周 | Step 1-2 ✓ |
| Phase 2 | Step 4 VLM 语义确认 + TP/FP 区分 | 1 周 | Phase 1 |
| Phase 3 | CTM 持续思考闭环（Step 1-4 循环） | 2 周 | Phase 2 |
| Phase 4 | Step 5 Diffusion 生成增强 | 2 周 | Phase 1 |
| Phase 5 | Step 6 FL 聚合 primitive 分布 | 2 周 | Phase 1 |
| Phase 6 | 端到端集成 + 跨域验证 | 2 周 | Phase 3-5 |
| Phase 7 | 论文撰写 + 专利申请 | 2 周 | Phase 6 |
| **Phase 8** | **小波频域原语分析（WIPES 路线）** | **1 周** | **Phase 1** |
| **Phase 9** | **Slot Attention 原语提取（FORLA 路线）** | **2 周** | **Phase 8** |
| **Phase 10** | **联邦原语聚合（FORLA + FedCtx）** | **2 周** | **Phase 9** |
| **Phase 11** | **对比原语学习（TP/FP contrastive）** | **1 周** | **Phase 9** |

---

## 7. 新增：基于文献调研的原语实验设计（Phase 8-11）

> 2026-07-11 新增，基于视觉原语文献调研
> 背景：Phase 1 形态学特征 + DINOv2 embedding 均无法区分 TP/FP（PR-AUC 0.29-0.50），需要换一个特征空间

### 7.1 调研发现的关键论文

| 论文 | 会议 | 核心贡献 | 与我们的关联 |
|------|------|---------|-------------|
| **FORLA** | NeurIPS 2025 | 联邦 slot attention，无监督 object-centric 跨域表征 | 直接命中：FL + object-centric + slot attention |
| **TwVP (DeepSeek)** | 2025 (撤稿) | 点/bbox 作为推理链中的空间原语 | CTM 的 cross-attention 已接近此理念 |
| **WIPES** | ICCV 2025 | 小波频域视觉原语，低频=全局结构+高频=局部细节 | 零成本验证：TP/FP 频域特征是否可分 |
| **Neurosymbolic Ambiguity Resolution** | AAAI 2025 | 混合 CV + 逻辑推理消歧 | TP/FP 视觉消歧的神经符号方法 |
| **CZSL Survey** | 2025 | 属性×物体原语组合，seen→unseen 泛化 | organoid 类型 = 属性原语组合 |

### 7.2 核心问题

当前所有 TP/FP 区分方案在**全局特征空间**失败：
- 形态学特征（circ/solidity/ar）：TP/FP 都 0.8-0.9，不可分
- DINOv2 768维 CLS token：cosine distance TP/FP = TP/TP = 0.150，完全重叠
- HNM（FP 空标签重训）：灾难性遗忘 -40pp

**新假设**：TP/FP 在全局特征空间不可分，但在**原语分解空间**可能可分——因为原语是 object-centric 的分解，不是 global embedding

### 7.3 Phase 8: 小波频域原语分析（WIPES 路线）

**动机**：WIPES (ICCV 2025) 证明小波分解能同时捕获低频（全局结构）和高频（局部细节），而 TP/FP 可能在频域有差异——TP 有 organoid 内部纹理（高频周期性），FP 是背景碎片（高频随机噪声）

**输入**：SAM2 mask 裁剪的 crop（TP + FP，MultiOrg 全量 16198 个）
**输出**：每个 crop 的小波系数原语向量

**方法**：
1. 对每个 crop 做 2D 离散小波变换（Haar / Daubechies-4，2-3 层分解）
2. 提取子带统计量：LL/LH/HL/HH 每层的均值、方差、能量、熵
3. 拼成原语向量（~24-36 维）
4. 评估 TP/FP 可分性：PR-AUC + t-SNE 可视化

**实验**：

| 编号 | 实验 | 数据 | 评估 | 预期 |
|------|------|------|------|------|
| W1 | Haar 2层分解 | MultiOrg 全量 TP/FP | PR-AUC | >0.50? |
| W2 | Daubechies-4 3层 | 同上 | PR-AUC | >0.50? |
| W3 | 最优小波 + 形态学特征拼接 | 同上 | PR-AUC | >0.60? |
| W4 | 鼠肝 B2/B3 TP/FP 交叉验证 | 鼠肝 | PR-AUC | 跨域泛化? |

**成本**：零（纯 numpy + pywt，CPU 可跑，30 分钟内完成）
**决策点**：如果 PR-AUC > 0.60，小波原语有效，进入 Phase 9 用 slot attention 进一步提升；如果 < 0.50，频域也不可分，直接进 Phase 9

### 7.4 Phase 9: Slot Attention 原语提取（FORLA 路线）

**动机**：FORLA (NeurIPS 2025) 证明 slot attention 在联邦跨域场景学到 object-centric 表征，比 DINOv2 全局 embedding 更有区分力——因为 slot 学的是"图片里有什么物体"（分解），不是"整张图长什么样"（全局）

**输入**：SAM2 mask 裁剪的 crop（TP + FP）
**输出**：每个 crop 的 K 个 slot 向量（K=4-8，每个 64-128 维）

**方法**：
1. 在 DINOv2 ViT-B/14 backbone 上接 slot attention 模块（不冻结 backbone，端到端训练）
2. 输入 224×224 crop → DINOv2 spatial tokens (256×768) → slot attention → K 个 slot (K×128)
3. 训练目标：重建 + 对比（TP slots 正样本对，FP slots 负样本对）
4. 评估：slot 空间的 TP/FP PR-AUC

**实验**：

| 编号 | 实验 | Slot 数 | 训练 | 评估 | 预期 |
|------|------|---------|------|------|------|
| S1 | Slot attention (K=4) | 4 | 62 crops 快速验证 | PR-AUC | >0.50? |
| S2 | Slot attention (K=8) | 8 | 62 crops | PR-AUC | >S1? |
| S3 | 最优 slot + 全量训练 | — | 16198 crops | PR-AUC | >0.70? |
| S4 | Slot vs DINOv2 CLS 对比 | — | 同上 | PR-AUC | S3 vs 0.29 |

**关键对比**：S4 直接对比 slot attention vs DINOv2 CLS token。如果 slot PR-AUC > 0.50 而 DINOv2 CLS = 0.29，证明 object-centric 分解是关键

**成本**：需要训练 slot attention 模块（~2M 参数），云 VM 可跑，预计 4-8h
**风险**：16198 crops 的 slot attention 可能仍不够——如果 TP/FP 在所有可观测空间都不可分，需要 Phase 11 对比学习强制拉开

### 7.5 Phase 10: 联邦原语聚合（FORLA + FedCtx）

**动机**：FORLA 的核心贡献是联邦 slot attention——各 client 本地训练 slot 模块，通过共享 adapter + slot attention 对齐 object-centric 表征跨域。这和我们的"数据不动，知识动"理念完全一致

**输入**：各 client（鼠肝 B1/B2/B3 或 MultiOrg 不同 Plate）的 slot 向量集合
**输出**：全局 slot 分布（FedCtx HNSW 索引）

**方法**：
1. 各 client 本地跑 Phase 9 的 slot attention 提取 slot 向量
2. 上传 slot 向量到 FedCtx HNSW（不共享原图/权重，只共享 slot）
3. 全局分布：HNSW k-NN 检索 + 范围查询
4. 下发全局 slot 分布 → 各 client 用全局分布校准本地检测

**实验**：

| 编号 | 实验 | Client | 聚合 | 评估 | 对比 |
|------|------|--------|------|------|------|
| F1 | 各 client 独立 slot | B1/B2/B3 | 无 | 本地 mask F1 | baseline |
| F2 | 联邦 slot (FedAvg adapter) | B1/B2/B3 | adapter 参数 | mask F1 | vs F1 |
| F3 | 联邦 slot (FedCtx HNSW) | B1/B2/B3 | slot 向量分布 | mask F1 | vs F2 |
| F4 | FORLA teacher-student | B1/B2/B3 | 双分支 | mask F1 | vs F3 |

**关键对比**：
- F2 vs F1：联邦 adapter 是否提升各 client
- F3 vs F2：分布聚合（FedCtx HNSW）vs 参数聚合（FedAvg）
- F4 vs F3：FORLA teacher-student 架构是否优于直接 HNSW 聚合

**成本**：需要 FedCtx 集成 slot attention（2-3 天代码），训练 3-5 轮联邦
**论文价值**：FORLA 是 NeurIPS 2025 新方法，我们是首个将其应用到 organoid 检测 + 医学图像的

### 7.6 Phase 11: 对比原语学习（TP/FP Contrastive）

**动机**：如果 Phase 9 的 slot attention 无监督训练仍然无法区分 TP/FP（slot 空间仍重叠），需要用**监督对比学习**强制拉开——TP slots 和 FP slots 在对比 loss 下被推向空间两侧

**输入**：TP/FP 标注的 crops（matched=True/False）
**输出**：对比训练后的 slot encoder

**方法**：
1. 用 Phase 9 的 slot attention encoder
2. InfoNCE loss：TP-TP 正样本对拉近，TP-FP 负样本对推远
3. 训练 50-100 epochs
4. 评估：slot 空间 TP/FP PR-AUC + 下游检测 mAP 提升

**实验**：

| 编号 | 实验 | Loss | 评估 | 预期 |
|------|------|------|------|------|
| C1 | 无监督 slot (Phase 9 baseline) | 重建 | PR-AUC | ~0.50 |
| C2 | InfoNCE 对比 | 对比 | PR-AUC | >0.70 |
| C3 | 合成负样本增强 | 对比 | PR-AUC | >0.80? |
| C4 | 对比 slot → 检测 mAP | — | mAP50 | >77.8%? |

**C4 关键验证**：如果对比训练的 slot encoder 能提升 MultiOrg 检测 mAP 从 77.8% 到 80%+，突破门槛

**与 HNM 的区别**：HNM 失败因为用 FP 空标签重训检测器 → 灾难性遗忘。对比学习不改检测器，只训 slot encoder 做后置过滤——检测器和 encoder 解耦

### 7.7 新增评估指标

| 层面 | 指标 | 目标 | 现有 baseline |
|------|------|------|--------------|
| 小波原语 | TP/FP PR-AUC | >0.60 | 无 |
| Slot 原语 | TP/FP PR-AUC | >0.50 | DINOv2 CLS=0.29 |
| 对比原语 | TP/FP PR-AUC | >0.70 | 无 |
| 联邦 slot | 全局 k-NN 召回率 | >0.9 | 无 |
| 端到端 | MultiOrg mAP50 | >80% | RF-DETR 77.8% |

### 7.8 决策树

```
Phase 8 (小波)
  ├── PR-AUC > 0.60 → 有效，进 Phase 9 提升
  └── PR-AUC < 0.50 → 无效，直接进 Phase 9

Phase 9 (Slot Attention)
  ├── PR-AUC > 0.50 → object-centric 分解有效
  │   ├── 无监督足够 → 进 Phase 10 联邦
  │   └── 无监督不够 → 进 Phase 11 对比学习
  └── PR-AUC < 0.50 → slot 也不可分，进 Phase 11 对比学习

Phase 10 (联邦 slot)
  └── F3 (FedCtx HNSW) vs F2 (FedAvg) → 分布聚合 vs 参数聚合

Phase 11 (对比学习)
  └── C4: 对比 slot → 检测 mAP > 80%? → 突破门槛
```

### 7.9 与现有 Phase 1-7 的关系

| 现有 Phase | 内容 | 新 Phase 关系 |
|-----------|------|-------------|
| Phase 1 | 形态学特征提取 | Phase 8 小波是频域版，Phase 1 是空间域版，两者互补 |
| Phase 2 | VLM 语义确认 | 不变，VLM 仍是可选的语义层 |
| Phase 3 | CTM 持续思考 | CTM 可用 Phase 9 的 slot 替代 DINOv2 token 做输入 |
| Phase 4 | Diffusion 生成 | 不变 |
| Phase 5 | FL 聚合 primitive | Phase 10 是 Phase 5 的 FORLA 实现版 |
| Phase 6-7 | 集成 + 论文 | 不变，最终都汇入 |

**核心改变**：Phase 1-5 用**形态学特征 + DINOv2 CLS**（全局特征，已证明无效），Phase 8-11 用**小波 + Slot Attention + 对比学习**（object-centric 分解特征，新方向）

## 8. 已有技术基础

| 组件 | 项目 | 状态 |
|------|------|------|
| RF-DETR 检测 | organoid-fl | ✓ B1 F1=1.00 |
| SAM2 分割 | organoid-fl | ✓ mask F1=93.9% |
| FedCtx HNSW | unified-fl-backend | ✓ v0.8.0, 60/60 测试 |
| EWA 熵加权 | ewa-fed | ✓ Phase 2 完成 |
| 五层幻觉防御 | NeuroSync | ✓ 可复用为 VLM 质量评估 |
| z-ai LLM/VLM SDK | z-ai-web-dev-sdk | ✓ 可调用 GLM-4V |

## 9. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 形态学特征 TP/FP 不可分（同 DINOv2） | 中 | 先做小规模验证（Phase 1），不行就转向 Diffusion 路线 |
| VLM 推理速度慢 | 高 | 批量调用 + 缓存 + fallback 到形态学规则 |
| Diffusion 生成质量差 | 中 | 用 VLM 评估筛选，真实:合成比例从低开始 |
| 向量空间对齐失败 | 低 | 退化到各 client 独立分布，不做全局聚合 |
| CTM 循环不收敛 | 中 | 设最大 ticks=5，不收敛用最后一次结果 |

## 10. 下一步

1. **立即可做**：Phase 1 形态学特征提取（不需要新模型，只需要 cv2 + numpy）
2. **立即可做**：Phase 8 小波频域分析（pywt + numpy，CPU 可跑，30 分钟）
3. **需要 API**：Phase 2 VLM 调用（z-ai-web-dev-sdk）
4. **需要 GPU**：Phase 4 Diffusion 生成（冬生 3060 或云 VM）
5. **需要 GPU**：Phase 9 Slot Attention 训练（云 VM，4-8h）
6. **已有基础设施**：Phase 5/10 FedCtx HNSW（直接复用）

先从 Phase 1 + Phase 8 并行开始，形态学（空间域）+ 小波（频域）同时验证 TP/FP 可分性。Phase 8 成本几乎为零，如果小波有效则直接跳到 Phase 9。
