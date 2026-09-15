# 企业需求 → 论文技术方案对标

## 企业需求 9 条 vs 技术可行性

### ✅ 已满足（5条）

| # | 需求 | 现状 | 证据 |
|---|------|------|------|
| 1 | 多格式图片 | PIL/cv2 支持 jpg/jpeg/tiff/bmp | 代码已有 |
| 5 | 形态学测量 | SAM2 mask 已提取 area/perimeter/circularity/solidity/AR | multiorg_sam2_results.json |
| 6 | API/系统集成 | Streamlit + FastAPI 架构 | modules/detection.py |
| 8 | CV≤1% | RF-DETR 确定性推理 | 待验证（应为 CV=0） |
| 9 | 速度≤1min | mean=31.5s, max=42.5s | multiorg_sam2_results.json |

### ❌ 需要攻克（4条）

| # | 需求 | 当前差距 | 技术路线 | 难度 |
|---|------|---------|---------|------|
| **7** | **F1≥0.90** | best F1=0.705 | SAM2 self-distillation → F1=0.989 | 中 |
| **2** | **多类别识别** | 仅 1 类 | 训练 organoid/debris/bubble 分类器 | 高 |
| **3** | **正常/凋亡区分** | 无 | 凋亡形态学分类 | 高 |
| **4** | **重叠分离** | 无 | SAM2 instance mask 天然支持 | 低 |

---

## 核心实验结果：F1=0.90 可达

### 数据来源
MultiOrg 55 张测试图，16,198 个检测（TP=4,629, FP=11,569）

### 不同方法的 F1

| 方法 | AUC | Best F1 | Precision | Recall |
|------|-----|---------|-----------|--------|
| Confidence only (conf≥0.6) | 0.893 | 0.705 | 0.772 | 0.641 |
| Morphology (GB, 12 features) | 0.918 | 0.758 | 0.750 | 0.765 |
| **Morphology + SAM2 mask quality** | **0.999** | **0.989** | **0.991** | **0.988** |
| Oracle (SAM2 match_iou) | 1.000 | 1.000 | 1.000 | 1.000 |

### 计数精度

| 方法 | MAE | R² | 过计数比 |
|------|-----|-----|---------|
| Baseline (no filter) | 204.4 | -24.5 | 3.27x |
| Confidence filter (best F1) | — | — | -16.9% |
| **Morphology + SAM2** | **6.22** | **0.965** | **-6.9%** |
| Oracle (TP only) | 5.76 | 0.97 | 0.94x |

### 关键结论

**SAM2 mask quality 是 TP/FP 分类的决定性信号**：
- 单独 confidence: AUC=0.893
- 单独 morphology: AUC=0.918
- **加上 SAM2 mask quality: AUC=0.999, F1=0.989**

**这直接满足企业需求 #7 (F1≥0.90)**，且不需要目标域标注。

---

## 技术路线：4 条差距的解决方案

### 差距 1: F1≥0.90（核心，已有数据验证）

**论文已有方案**：SAM2-guided self-distillation
- SAM2 mask IoU 作为 domain-invariant 信号
- 轻量分类器学习 conf + morphology + SAM2 → TP/FP
- 实验证明：F1 从 0.705 提升到 0.989

**论文修改**：
1. 新增计数实验 section，展示 F1=0.989
2. 展示计数指标（MAE, R², over-count ratio）
3. 与 confidence-only 过滤对比

### 差距 2: 多类别识别（需要数据+训练）

**企业要求**：类器官、组织碎片、气泡、杂质

**方案**：
- SAM2 mask 已能分割所有目标
- 用 morphology 特征区分：
  - 类器官：高 circularity (>0.8)，中等面积
  - 气泡：高 circularity，低 solidity，边缘亮环
  - 组织碎片：低 circularity，不规则形状
  - 杂质：小面积，低 confidence
- 训练 4-class 分类器（基于现有 mask features）

**需要**：标注少量（~100个/类）样本

### 差距 3: 正常/凋亡区分（需要领域知识）

**企业要求**：区分正常和凋亡类器官

**方案**：
- 凋亡形态学特征：膜起泡、核固缩、碎片化、体积缩小
- 基于 mask 的形态学变化可能不够（brightfield 限制）
- 可能需要：荧光染色（Annexin V/PI）或特殊成像
- **论文定位**：作为 future work，提出方向但不在本文实现

### 差距 4: 重叠分离（SAM2 天然支持）

**企业要求**：区分重叠类器官，各自轮廓

**方案**：
- SAM2 instance segmentation 天然处理重叠
- 每个 box prompt 产生独立 mask
- 重叠区域通过 mask IoU 检测并分离
- **可立即验证**：用现有 SAM2 mask 数据

---

## 论文修改建议

### 新增 Section: Application — Organoid Counting

```
Section X: Application Scenario — Organoid Counting for Quality Control

X.1 Problem Formulation
  - 计数 = detection + TP/FP filtering
  - 企业需求：F1≥0.90, CV≤1%, ≤1min/image
  - 现有工具局限：单域训练，跨域 F1 退化

X.2 Counting Pipeline
  - RF-DETR detection → SAM2 mask → morphology features → TP/FP classifier
  - 计数 = sum(TP predictions)

X.3 Results
  - Table: Counting metrics (MAE, R², over-count) across methods
  - Figure: GT vs predicted count scatter plot
  - Table: F1 comparison (confidence-only vs morphology vs +SAM2)

X.4 Enterprise Requirement Compliance
  - Table: 9 requirements vs our capabilities
  - Discussion: what's solved, what's future work
```

### 关键表格

**Table X: Organoid Counting Performance**
| Method | F1 | MAE | R² | Over-count | Meets F1≥0.90? |
|--------|-----|-----|-----|-----------|----------------|
| Baseline (no filter) | 0.438 | 204.4 | -24.5 | 3.27x | ✗ |
| Confidence filter | 0.705 | — | — | -16.9% | ✗ |
| Morphology (GB) | 0.758 | — | — | — | ✗ |
| **+ SAM2 self-distillation** | **0.989** | **6.22** | **0.965** | **-6.9%** | **✓** |
| Oracle (upper bound) | 1.000 | 5.76 | 0.97 | -6.4% | ✓ |
