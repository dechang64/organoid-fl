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

## 7. 已有技术基础

| 组件 | 项目 | 状态 |
|------|------|------|
| RF-DETR 检测 | organoid-fl | ✓ B1 F1=1.00 |
| SAM2 分割 | organoid-fl | ✓ mask F1=93.9% |
| FedCtx HNSW | unified-fl-backend | ✓ v0.8.0, 60/60 测试 |
| EWA 熵加权 | ewa-fed | ✓ Phase 2 完成 |
| 五层幻觉防御 | NeuroSync | ✓ 可复用为 VLM 质量评估 |
| z-ai LLM/VLM SDK | z-ai-web-dev-sdk | ✓ 可调用 GLM-4V |

## 8. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 形态学特征 TP/FP 不可分（同 DINOv2） | 中 | 先做小规模验证（Phase 1），不行就转向 Diffusion 路线 |
| VLM 推理速度慢 | 高 | 批量调用 + 缓存 + fallback 到形态学规则 |
| Diffusion 生成质量差 | 中 | 用 VLM 评估筛选，真实:合成比例从低开始 |
| 向量空间对齐失败 | 低 | 退化到各 client 独立分布，不做全局聚合 |
| CTM 循环不收敛 | 中 | 设最大 ticks=5，不收敛用最后一次结果 |

## 9. 下一步

1. **立即可做**：Phase 1 形态学特征提取（不需要新模型，只需要 cv2 + numpy）
2. **需要 API**：Phase 2 VLM 调用（z-ai-web-dev-sdk）
3. **需要 GPU**：Phase 4 Diffusion 生成（冬生 3060 或云 VM）
4. **已有基础设施**：Phase 5 FedCtx HNSW（直接复用）

先从 Phase 1 开始，用鼠肝 B1/B2/B3 的 SAM2 mask 提取 primitive 向量，UMAP 可视化看分布。
