# 视觉原语 × 大模型 × 类器官：实验方案 v2.0

> 2026-07-14 曼卿起草
> 基于：Phase 1-11 完整结果 + 跨域评估 + 多数据集联合训练 + 深度文献调研
> 冬生要求："想得全面一些"——借鉴 NLP 语义分析成功经验，利用大模型提升类器官图像分析，探索 3D 时序数据

---

## 〇、核心问题与已验证的结论

### 0.1 跨域评估完整结果

| 数据集 | 单域 Slot AUC | 联合 Slot AUC | Conf AUC | 联合提升 | vs Conf |
|--------|---------------|---------------|----------|----------|---------|
| 鼠肝 B1 | 0.29（反预测） | 0.74 | 0.91 | +0.45 | -0.17 |
| 鼠肝 B2 | 0.51（随机） | 0.70 | 0.98 | +0.19 | -0.28 |
| 鼠肝 B3 | 0.54（随机） | 0.85 | 0.92 | +0.31 | -0.07 |
| Intestinal | 0.67 | 0.74 | 0.92 | +0.07 | -0.18 |

### 0.2 已确认的事实

1. **DINOv2 全局特征有天花板**：CTM 全量 16K crops AUC=0.80 < RF-DETR 0.893
2. **联合训练有效但不够**：跨域 AUC 从 0.29-0.67 提升到 0.70-0.85，但仍低于 conf 0.91-0.98
3. **单帧信息有限**：所有方案都基于单帧 crop，没有时序信息
4. **形态学/小波/CTM 均无法超越 RF-DETR confidence**
5. **联合训练的 B3 soft_penalize 首次有实质正 delta**（+0.0125）——方向对，但特征空间有上限

### 0.3 关键诊断：为什么跨域失败？

| 假设 | 证据 | 验证状态 |
|------|------|----------|
| DINOv2 特征域特定 | CTM 0.80 < RF-DETR 0.893，跨域 AUC 0.29-0.67 | ✅ 确认 |
| 数据量不足 | Intestinal 2744 crops AUC=0.67 > 鼠肝 26-60 crops 0.29-0.54 | ✅ 确认 |
| 缺乏语义对齐 | 没有 text-aligned 表示，DINOv2 纯视觉 | ✅ 新发现 |
| 单帧信息不足 | 无时序变化信息 | ⚠️ 未验证 |

### 0.4 NLP 成功经验的启示

| NLP 成功要素 | 视觉对应 | 我们现状 | 差距 |
|-------------|---------|---------|------|
| 大规模语料预训练 | 大规模图像预训练 | DINOv2 有（1B 图像） | ✓ 有 |
| Token 作为语义单元 | Object/region 作为视觉单元 | Slot attention（K=8） | ⚠️ 被动分解 |
| 语义对齐（文本-视觉） | CLIP 文本-视觉对齐 | 无 | ❌ 缺失 |
| 上下文理解 | 多帧/多尺度上下文 | 单帧 224×224 | ❌ 缺失 |
| 推理链（CoT） | 视觉推理操作 | 单次前传 | ❌ 缺失 |

**核心差距**：我们用了预训练 backbone（DINOv2），但没有语义对齐、没有上下文、没有推理链。NLP 跨域成功不只靠预训练，靠的是预训练 + 微调 + 语义对齐的组合。

---

## 一、总体架构：三层递进

```
Layer 1: 语义对齐（CLIP 替代 DINOv2）
  └── 视觉特征 → 文本-视觉对齐空间 → 跨域通用语义
       │
Layer 2: 推理操作（VLM 主动检查）
  └── 单次打分 → 多步推理（crop/zoom/compare）→ 专家级判断
       │
Layer 3: 时序泛化（3D 视频数据）
  └── 单帧检测 → 时序追踪 → 跨域不变的动态特征
```

**递进逻辑**：
- Layer 1 是**特征空间升级**（成本最低，1-2 周）
- Layer 2 是**范式转换**（成本中等，1-2 月）
- Layer 3 是**数据维度扩展**（成本最高，3-6 月）
- 三层不互斥，可以叠加：CLIP 特征 + VLM 推理 + 时序信息

---

## 二、Layer 1：CLIP 语义对齐（短期）

### 2.1 动机

DINOv2 是纯视觉自监督模型，学到的是视觉特征聚类，没有语义对齐。CLIP 有文本-视觉对齐，"a round bright organoid with smooth boundary" 和对应图像在同一个语义空间——这个空间跨域更稳健，因为文本语义不变。

### 2.2 方案

**Phase A: CLIP 特征替代 DINOv2**

```
当前: crop → DINOv2 ViT-B/14 → 768d CLS token → slot attention → TP/FP
改为: crop → CLIP ViT-B/16 → 512d text-aligned embedding → slot attention → TP/FP
```

**模型选择**：
- `openai/clip-vit-base-patch16`（512d，最经典）
- `apple/DFN5B-CLIP-ViT-H-14`（1024d，更强）
- `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`（1280d，最强开源）

**实验矩阵**：

| 编号 | 实验 | Backbone | 维度 | 训练数据 | 评估 |
|------|------|----------|------|----------|------|
| A1 | CLIP ViT-B/16 | OpenAI CLIP | 512 | MultiOrg 100 | Slot AUC + 跨域 |
| A2 | CLIP ViT-H/14 | DFN5B | 1024 | MultiOrg 100 | Slot AUC + 跨域 |
| A3 | CLIP + 联合训练 | 最优 CLIP | — | MultiOrg + 鼠肝 + Intestinal | 跨域 AUC |
| A4 | CLIP text prompt | CLIP 文本编码器 | 512 | 手工 prompt | Zero-shot AUC |

**A4 关键实验——文本 prompt zero-shot**：
- Prompt: "a microscopy image of a real organoid"（TP 描述）
- Prompt: "a background artifact or debris"（FP 描述）
- CLIP 图像编码 × 文本编码 → cosine similarity
- 不需要训练，直接 zero-shot 评估
- 如果 zero-shot AUC > 0.60，说明 CLIP 语义空间确实能区分 TP/FP

### 2.3 预期与决策

| 结果 | 含义 | 下一步 |
|------|------|--------|
| A4 zero-shot AUC > 0.70 | CLIP 语义对齐有效 | 进入 A3 联合训练 |
| A4 AUC 0.50-0.70 | 有信号但不够 | 联合训练 + 更好 prompt |
| A4 AUC < 0.50 | CLIP 语义也不够 | 转向 Layer 2 VLM 推理 |
| A3 跨域 AUC > 0.85 | 突破！ | 叠加 Layer 2/3 |
| A3 跨域 AUC 0.70-0.85 | 改善但不够 | 叠加 Layer 2 |

### 2.4 成本

- 模型下载：CLIP ViT-B/16 ~600MB，ViT-H/14 ~3GB
- 训练：和 SupCon 一样的 pipeline，只换 backbone
- 时间：云 VM CPU 可跑（CLIP 比 DINOv2 小），冬生 3060 更快
- 代码改动：`slot_supcon.py` 的 backbone 从 `timm.create_model('vit_base_patch14_dinov2')` 改为 `clip.load()`

### 2.5 风险

| 风险 | 概率 | 应对 |
|------|------|------|
| CLIP 在显微镜图像上表现差（域外） | 中 | 用 BiomedCLIP（医学预训练 CLIP）替代 |
| 文本 prompt 不准确 | 中 | A4 用多个 prompt 模板取平均 |
| 联合训练仍不够 | 低 | 叠加 Layer 2 |

---

## 三、Layer 2：VLM 推理操作（中期）

### 3.1 动机

DeepSeek "Thinking with Visual Primitives" 的核心洞察：**视觉原语不是特征，而是推理操作的单元**。当前我们做的是 `image → features → classify`，应该改为 `image → VLM "is this real?" → crop/zoom → VLM "check boundary" → 决策`。

### 3.2 关键概念：Reference Gap

DeepSeek 指出 VLM 的 **Reference Gap**：自然语言无法精确指代空间位置。VLM 输出 "is_organoid: 0.8" 但无法说 "因为在 [x1,y1] 处有圆形轮廓"——只有分数没有空间定位。

**解决思路**：让 VLM 在推理过程中**指向**图像中的区域（crop/zoom/point），而不是只给一个全局分数。

### 3.3 方案

**Phase B: VLM 多步推理 pipeline**

```
Step 1: VLM 全局评估
  Input: 原图 + 检测框
  VLM: "图中标注了 N 个检测框，哪些是真正的类器官？逐个分析"
  Output: 每个框的初步判断 + 不确定项

Step 2: VLM 局部放大（对不确定项）
  Input: 不确定框的 crop（放大到 224×224）
  VLM: "这是显微镜下的一个 crop，判断是否为类器官：
        1. 边界是否清晰？
        2. 内部是否有细胞结构？
        3. 形态是否典型？
        给出 0-1 分和理由"
  Output: 精细判断 + 理由

Step 3: VLM 对比推理（对仍不确定项）
  Input: 不确定 crop + 同图 TP crop（参考）
  VLM: "左图是确定的类器官，右图待判断。比较两者：
        形态差异？纹理差异？大小差异？"
  Output: 对比判断 + 最终决策
```

**VLM 选择**：
- GLM-4V（z-ai-web-dev-sdk，已集成）
- Qwen-VL-Max（阿里通义千问 VL）
- GPT-4o（OpenAI，支持 thinking with images）

### 3.4 实验矩阵

| 编号 | 实验 | VLM | Steps | 数据 | 评估 |
|------|------|-----|-------|------|------|
| B1 | 单步 VLM（baseline） | GLM-4V | 1 | MultiOrg 100 | AUC（已有：0.713） |
| B2 | 两步 VLM（全局+局部） | GLM-4V | 2 | MultiOrg 100 | AUC |
| B3 | 三步 VLM（+对比） | GLM-4V | 3 | MultiOrg 100 | AUC |
| B4 | 跨域 VLM | GLM-4V | 2 | 鼠肝 + Intestinal | 跨域 AUC |
| B5 | VLM + CLIP 融合 | GLM-4V | 2 | 联合 | AUC（Layer 1+2 叠加） |

### 3.5 成本

- VLM 调用：每次 ~1-3 秒，100 crops ≈ 5 分钟
- 三步 pipeline：100 crops × 3 steps ≈ 15 分钟
- 跨域评估：3 个数据集 × 3 steps ≈ 1 小时
- **不需要 GPU**，纯 API 调用
- 代码改动：新建 `vlm_reasoning_pipeline.py`

### 3.6 风险

| 风险 | 概率 | 应对 |
|------|------|------|
| VLM 速度慢 | 高 | 批量调用 + 缓存 + 只对不确定项做多步 |
| VLM 幻觉 | 中 | 多步对比 + 置信度阈值 + fallback 到 CLIP |
| 成本高 | 低 | GLM-4V 免费额度足够 |

---

## 四、Layer 3：3D 时序数据（长期）

### 4.1 动机

单帧信息有限——同一 organoid 在不同时间点的形态变化（生长、分裂、死亡）是跨域不变的。OrganoID 实现 4 天时序追踪 89% 精度，TransOrga-plus 用时序追踪提升分割一致性。

### 4.2 已发现的数据集

| 数据集 | 类型 | 规模 | 来源 | 获取难度 |
|--------|------|------|------|----------|
| OrganoID | bright-field + fluorescent, 4-day time-lapse | 多 lab | bioRxiv 2022 | 低（开源） |
| LabelFreeTracker | intestinal 3D label-free | 3D + time | Cell Reports 2025 | 中 |
| TransOrga-plus | 多组织类型 | 大规模 | BMC Biology 2025 | 中 |
| Digitalized Organoids | 3D 多种 organoid | 3D | Nature Methods 2025 | 中 |
| OrganoidTracker 2.0 | 3D cell tracking | 3D + time | Nature Methods 2025 | 中 |
| Phase contrast time-lapse | label-free | time-lapse | Nature 2024 | 低（开源） |

### 4.3 方案

**Phase C: 时序特征提取**

```
单帧: [frame_t] → DINOv2/CLIP → features → classify
时序: [frame_t-2, t-1, t, t+1, t+2] → temporal encoder → features → classify
```

**两种路径**：
1. **追踪 + 时序特征**：用 OrganoidTracker 追踪同一 organoid → 提取时序特征（面积变化率、形态变化率）
2. **3D 体积特征**：3D 堆栈 → 3D 分割 → 体积/表面积/3D 形态

### 4.4 实验矩阵

| 编号 | 实验 | 数据 | 方法 | 评估 |
|------|------|------|------|------|
| C1 | 下载 OrganoID 时序数据 | OrganoID | 追踪 + 时序特征 | 跨域 AUC |
| C2 | 时序 vs 单帧对比 | 同上 | 时序特征 + DINOv2 | Δ AUC |
| C3 | 3D 体积特征 | Digitalized Organoids | 3D 分割 + 体积 | AUC |
| C4 | 时序 + CLIP 联合 | OrganoID | CLIP + temporal | 跨域 AUC |

### 4.5 成本

- 数据下载：OrganoID 开源，1-2 天
- 追踪：需要跑 OrganoidTracker 2.0（GPU 训练）
- 时序特征：CPU 可跑
- 周期：3-6 月

### 4.6 风险

| 风险 | 概率 | 应对 |
|------|------|------|
| 数据获取慢 | 中 | 先做 OrganoID（开源） |
| 追踪不准 | 中 | 用 OrganoidTracker 2.0（<0.5% error） |
| 时序特征不够 | 低 | 叠加 3D 体积特征 |

---

## 五、交叉实验矩阵（三层叠加）

| 实验 | Layer 1 (CLIP) | Layer 2 (VLM) | Layer 3 (时序) | 预期 AUC |
|------|-----------------|---------------|----------------|----------|
| 现有 baseline | DINOv2 | — | — | 0.29-0.67 |
| 联合训练 | DINOv2 | — | — | 0.70-0.85 |
| A3 | CLIP + 联合 | — | — | 0.80-0.90? |
| B5 | DINOv2 | VLM 三步 | — | 0.75-0.85? |
| A+B | CLIP + 联合 | VLM 三步 | — | 0.85-0.95? |
| A+B+C | CLIP + 联合 | VLM 三步 | 时序 | >0.95? |

**终极目标**：Layer 1 + 2 + 3 叠加，跨域 AUC > 0.90，超过 conf baseline。

---

## 六、已有技术基础复用

| 组件 | 项目 | 状态 | 用于 |
|------|------|------|------|
| RF-DETR 检测 | organoid-fl | ✓ 77.8% MultiOrg | 所有 Phase 的检测器 |
| YOLOv12s 检测 | organoid-fl | ✓ 88.1% Intestinal | Intestinal 评估 |
| SAM2 分割 | organoid-fl | ✓ zero-shot | 可选 mask crop |
| SupCon 训练 | organoid-fl | ✓ slot_supcon.py | Layer 1 backbone 替换 |
| 跨域评估 | organoid-fl | ✓ cross_domain_eval.py | 所有跨域实验 |
| 多数据集合并 | organoid-fl | ✓ merge_datasets.py | 联合训练 |
| z-ai VLM SDK | z-ai-web-dev-sdk | ✓ GLM-4V | Layer 2 |
| FedCtx HNSW | unified-fl-backend | ✓ v0.8.0 | FL 聚合（可选） |
| Streamlit 展示 | organoid-fl | ✓ app.py | 结果可视化 |

---

## 七、时间线与优先级

| 优先级 | Phase | 内容 | 周期 | 依赖 | 预期跨域 AUC |
|--------|-------|------|------|------|-------------|
| P0 | A4 | CLIP zero-shot text prompt | 1 天 | 无 | 0.50-0.70 |
| P0 | A1 | CLIP ViT-B/16 + SupCon | 3 天 | A4 有信号 | 0.70-0.80 |
| P1 | A3 | CLIP + 联合训练 | 1 周 | A1 | 0.80-0.90 |
| P1 | B2 | 两步 VLM pipeline | 1 周 | 无 | 0.75-0.85 |
| P2 | B5 | CLIP + VLM 融合 | 2 周 | A3 + B2 | 0.85-0.95 |
| P2 | B3 | 三步 VLM + 跨域 | 2 周 | B2 | 0.80-0.90 |
| P3 | C1 | 下载 OrganoID + 追踪 | 2 周 | 无 | — |
| P3 | C2 | 时序 vs 单帧对比 | 2 周 | C1 | +0.05-0.15 |
| P4 | A+B+C | 全部叠加 | 1 月 | A3+B5+C2 | >0.95? |

---

## 八、决策树

```
A4 (CLIP zero-shot)
  ├── AUC > 0.70 → CLIP 语义对齐有效
  │   └── A1 (CLIP + SupCon) → A3 (联合训练)
  │       ├── 跨域 AUC > 0.85 → 突破！叠加 B5 (VLM)
  │       └── 跨域 AUC 0.70-0.85 → 叠加 B2 (VLM 两步)
  │
  ├── AUC 0.50-0.70 → 有信号但不够
  │   └── B2 (VLM 两步) → B5 (CLIP + VLM 融合)
  │
  └── AUC < 0.50 → CLIP 语义也不够
      └── B3 (VLM 三步推理) → 转向 Layer 3 时序

B5 (CLIP + VLM 融合)
  ├── 跨域 AUC > 0.90 → 接近 conf，可以发论文
  └── 跨域 AUC 0.80-0.90 → 叠加 C2 (时序)

C2 (时序 vs 单帧)
  ├── Δ AUC > 0.10 → 时序有效，深入 C4
  └── Δ AUC < 0.05 → 时序不关键，聚焦 Layer 1+2
```

---

## 九、与面上项目的关系

| 研究内容 | 对应 Phase |
|---------|-----------|
| RC1: 类器官微观形态的多模态向量化表征与增强 | Layer 1 (CLIP) + Layer 3 (3D) |
| RC2: LLM 质量评估 | Layer 2 (VLM 推理) |
| RC3: 联邦隐私检索 | 现有 Phase 10 (FedCtx HNSW) |

---

## 十、论文叙事

### 10.1 问题

类器官检测的跨域泛化是未解决的难题——同一模型在不同实验室/不同器官类型上精度骤降。现有方法（SAHI/NMS/Ensemble/Orga-Dete）都是单帧单域优化，无法跨域。

### 10.2 贡献

1. **系统验证跨域失效**：SupCon slot model 在 4 个跨域场景全部失效（AUC 0.29-0.67），揭示 DINOv2 全局特征的域特定性
2. **多数据集联合训练**：跨域 AUC 提升 +7~45pp，验证数据多样性是关键
3. **CLIP 语义对齐**：文本-视觉对齐空间比纯视觉空间更跨域（假设）
4. **VLM 推理操作**：从单次打分到多步推理，借鉴 NLP CoT 成功经验（假设）
5. **3D 时序泛化**：时序变化跨域不变（假设）

### 10.3 目标期刊

- **Nature Methods**（如 Digitalized Organoids）
- **Nature Machine Intelligence**（VLM + 医学）
- **NeurIPS 2026**（视觉原语 + 联邦学习）
- **MICCAI 2026**（医学图像分析）

---

## 十一、下一步

**立即执行**：Phase A4（CLIP zero-shot text prompt），1 天完成，决策后续方向。

```bash
# 安装 CLIP
pip install open_clip_torch

# A4 zero-shot 评估
python scripts/multiorg/clip_zeroshot_eval.py \
  --crops-dir results/phase2_vlm_100/crops \
  --metadata results/phase2_vlm_100/vlm_results.json \
  --model ViT-B-16 --pretrained openai
```

如果 A4 有信号（AUC > 0.50），立即推进 A1（CLIP + SupCon）。如果 A4 无信号，转向 B2（VLM 两步推理）。
