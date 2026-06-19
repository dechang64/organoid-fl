# MultiOrg 数据集超 80% 方案

## 目标

超越 MultiOrg SOTA (SSD 68.1% mAP@0.5) → 达到 80%+

数据集：MultiOrg (NeurIPS 2024, 411张肺类器官 6K×5.7K, 63K bbox, 单类)

---

## 当前问题分析

### 我们之前的错误
| 错误 | 正确做法（论文） |
|------|----------------|
| 2类（Normal/Macros） | 单类 organoid |
| 640 patch, 裁剪边界 bbox | 512 patch, 丢弃边界 bbox |
| patch 上直接评估 | 全图滑动窗口推理 |
| 100 epoch | 400 epoch |
| yolo12n | 论文用 SSD/YOLOv3/RTMDet |

### SOTA 基线（论文 Table 3）

| 模型 | mAP@0.5 | mAP@0.75 | Precision | Recall |
|------|---------|----------|-----------|--------|
| Faster R-CNN | 60.7% | 19.2% | 0.19 | 0.31 |
| **SSD** | **68.1%** | **25.3%** | 0.52 | 0.61 |
| YOLOv3 | 64.6% | 21.6% | 0.66 | 0.62 |
| RTMDet | 61.6% | 24.3% | 0.58 | 0.60 |

### 瓶颈分析

1. **小目标居多**：6K 图缩到 512 patch，小 organoid 在 patch 里仍很小
2. **背景噪声**：defocused organoid 产生大量假阳性
3. **标注噪声**：多标注者 t0/t1 不一致，inter-rater 差异大
4. **单类 vs 多类**：论文承认 Normal/Macros 未标注为不同类，多类可能更好
5. **训练量不足**：我们只跑 100 epoch，论文跑 400 epoch

---

## SOTA 文献关键发现

### 1. 论文作者自己的建议（Discussion 节）

> "although two organoid types are present in the images, they have not been annotated as different classes. Treating this as a multi-class detection problem may boost the overall performance"

> "it would be interesting to observe how the label sets change by using DL models"

**作者明确指出两个改进方向：多类检测 + 标签噪声学习。**

### 2. 滑动窗口推理细节（论文 Section 4.1）

- 双尺度滑动：window=512（downsample=2）+ window=2048（downsample=8）
- overlap=0.5
- NMS threshold=0.5
- 目的：小 organoid 用 512 窗口，大 organoid 用 2048 窗口

### 3. 标注噪声量化（论文 Section 3.3）

- t0：初始标注，噪声较高
- t1_A / t1_B：二次标注，噪声较低
- **模型在 t1 上评估比 t0 高 ~8pp**——说明 t0 标注噪声拉低了表观性能
- 如果用 t1 标签训练，潜在 +8pp 空间

### 4. napari-organoid-counter 插件

- 论文发布了最佳模型权重（SSD），集成在 napari 插件中
- 可直接下载使用，作为 teacher 或 baseline

---

## 六条超越路径

### 路径 1：正确复现论文方法（最确定，68% → 72-75%）

**核心：按论文方法正确做一遍。**

1. **单类 organoid**：合并 Normal+Macros 为 class 0
2. **512×512 patch，stride=256（50% overlap）**
3. **丢弃跨边界 bbox**（不裁剪）
4. **400 epoch, A100 级训练**（我们 3060 可以跑，时间换精度）
5. **全图滑动窗口推理**（512+2048 双尺度, overlap=0.5, NMS=0.5）

**预期：72-75%**。我们之前 59.9% 是因为方法全错，不是模型不行。

### 路径 2：模型代差（68% → 75-78%）

论文用 SSD/YOLOv3（2018-2020），我们用 YOLOv12（2025）。

| 论文模型 | 年份 | COCO mAP | YOLOv12n | 提升 |
|---------|------|----------|----------|------|
| SSD300 | 2016 | 25.1% | - | - |
| YOLOv3 | 2018 | 33.0% | - | - |
| RTMDet | 2022 | 52.5% | - | - |
| **YOLOv12n** | **2025** | **40.6%** | vs SSD +15.5pp | - |
| **YOLOv12s** | **2025** | **48.0%** | vs YOLOv3 +15pp | - |

**但 COCO mAP 不直接迁移到 organoid**——关键是小目标能力。YOLOv12 的 A2C2 区域注意力对密集小目标有天然优势。

**预期：+5-8pp → 73-76%。**

### 路径 3：多类检测（68% → 72-75%）

**论文作者建议但未实现的方向。**

- Normal 和 Macros 形态不同，单类检测混淆
- 多类检测让模型学习形态差异，减少背景假阳性
- 推理时合并两类结果即可

**风险**：需要重新标注 Normal/Macros 标签。数据集原始 JSON 里是否有 study type 信息？需要检查。

**预期：+3-5pp → 71-73%。**

### 路径 4：标签噪声学习（68% → 75-80%）

**MultiOrg 独特优势：3 套标注（t0, t1_A, t1_B）。**

1. **用 t1 标签训练**：t1 噪声比 t0 低 ~8pp，换标签直接提升
2. **多标注者共识**：取 t1_A ∩ t1_B 交集做 high-confidence 标签
3. **Co-teaching**：两个模型互相过滤噪声标签
4. **Label smoothing**：在 t0/t1 不一致的区域降低 loss 权重

**预期：+5-8pp → 73-76%。**

### 路径 5：SAHI 推理增强（训练不变，+3-5pp）

**论文用滑动窗口，我们用更现代的 SAHI。**

1. 训练：512 patch 单类
2. 推理：SAHI 切片（512×512, overlap=0.3）+ 全图合并
3. WBF 融合多模型预测
4. TTA（水平翻转 + 多尺度）

**论文的滑动窗口是 SAHI 的前身，SAHI 更成熟。**

**预期：+3-5pp → 75-78%。**

### 路径 6：数据增强 + freebies（+2-3pp）

- copy_paste=1.0：复制小 organoid 实例
- Albumentations：RandomBrightnessContrast + GaussNoise（处理 defocus 噪声）
- mixup=0.15, label_smoothing=0.1, cos_lr=True
- close_mosaic=15

**预期：+2-3pp → 75-77%。**

---

## 综合方案：三阶段到 80%

### 阶段 1：正确复现 + 模型代差（目标 72-75%）

**数据准备**：
1. 重新 tile：512×512, stride=256, 丢弃边界 bbox, 单类
2. 用 t1_A 标签（噪声更低）

**训练**：
```powershell
yolo train model=yolo12s.pt data=multiorg_correct.yaml imgsz=512 epochs=400 patience=80 batch=16 close_mosaic=15 copy_paste=1.0 mixup=0.15 cos_lr=True device=0 name=multiorg_12s_correct
```

**推理**：
- SAHI 全图滑动窗口（512, overlap=0.3）
- NMS=0.5

### 阶段 2：标签噪声 + 多类（目标 76-78%）

1. **t1_A + t1_B 共识标签**：交集做 high-conf，差集做 soft label
2. **多类检测**：Normal=0, Macros=1（需确认 study type 可获取）
3. **Co-teaching**：训练两个模型，互相过滤噪声样本

### 阶段 3：推理优化（目标 80%+）

1. **SAHI + 双尺度**（512 + 1024）
2. **WBF 融合**：yolo12n + yolo12s + yolo12m 三模型
3. **TTA**：水平翻转 + 多尺度
4. **Soft-NMS** 替代 NMS

---

## 关键验证点

### 1. study type 是否可获取？
MultiOrg 原始数据按 study 组织目录，study type（Normal/Macros）在文件名或目录名中。需检查：
```
MultiOrg_v2/{train,test}/{Normal,Macros}/Plate_X/image_Y/
```
→ **已确认可获取**（我们之前 tiling 时就用了 Normal/Macros 路径）

### 2. t1 标签是否在数据集中？
论文提供 3 套标签：t0, t1_A, t1_B。需检查 Kaggle 数据集是否包含全部 3 套。

### 3. 512 patch 的训练量？
411 张 6K 图 → 512 patch stride=256 → 约 12,000-15,000 patches。足够训练。

### 4. 3060 能否跑 400 epoch？
512 分辨率 batch=16，约 1-2 min/epoch。400 epoch ≈ 7-13 小时。可以。

---

## 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 512 patch 还是漏检小 organoid | 中 | 加 SAHI 双尺度推理 |
| t1 标签不在数据集中 | 低 | 用 t0 + label smoothing |
| 多类反而降精度 | 中 | 先做单类 baseline，再试多类 |
| 400 epoch 过拟合 | 低 | patience=80 + early stopping |

---

## 论文/专利布局

### 论文方向
1. **"YOLOv12 + SAHI for Multi-Rater Organoid Detection"**
   - 首个 YOLOv12 + MultiOrg
   - SAHI 替代滑动窗口推理
   - 多标注者共识标签

2. **"Label Noise-Aware Federated Learning for Organoid Detection"**
   - MultiOrg 3 套标签 + FL
   - 多标注者不确定性建模

### 专利方向
1. **多标注者共识 + organoid 检测方法** — 无专利
2. **SAHI + 双尺度 + organoid 推理** — 无专利
3. **FL + 多标注者 + organoid** — 完全空白

---

## 与 Intestinal 方案的协同

| 维度 | Intestinal | MultiOrg |
|------|-----------|----------|
| 数据量 | 840 张 | 411 张（→ 12K patches） |
| 分辨率 | 1280×960 | 6K×5.7K |
| 类别 | 4 类形态 | 单类（论文）/ 多类（我们试） |
| 标注噪声 | 无 | 3 套多标注者 |
| SOTA | 87.5% (Deliod) | 68.1% (SSD) |
| 目标 | 90%+ | 80%+ |
| 核心方法 | YOLOv12s + freebies | YOLOv12s + SAHI + 标签噪声 |
| 共享技术 | copy_paste, cos_lr, close_mosaic | 同左 |
| FL 潜力 | 4 类 non-IID | study-based non-IID |

**两个数据集的技术栈可以复用**：相同的 YOLOv12 架构、相同的 freebies、相同的 SAHI 推理框架。

---

## 版本历史

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-06-19 | v1.0 | 初始方案，基于 MultiOrg 论文 + SOTA 调研 |
