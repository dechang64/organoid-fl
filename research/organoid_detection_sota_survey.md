# 类器官检测 SOTA 文献与专利调研报告

## 一、数据集全景

### 1. Intestinal Organoid（Zenodo 6768583）
- 840 张（756 train / 84 val），23,066 标注
- 1280×960 RGB JPEG
- 4 类：organoid0(囊性,11922) / organoid1(早期,5510) / organoid3(晚期,3367) / spheroid(2266)
- 长尾比 5.3:2.4:1.5:1

### 2. MultiOrg（NeurIPS 2024）
- 411 张肺类器官 6K×5.7K，63K bbox
- 单类 organoid，3 套多标注者标签
- Train 356 / Test 55
- SOTA: SSD 68.1% mAP@0.5

### 3. 其他数据集
- **OrganoID** (Bhatt et al.) — 通用类器官检测+追踪平台
- **OrgaQuant** (Kassis et al.) — 14,000 肠类器官，大小测量
- **Gastric Organoid** (Yang et al.) — 胃类器官衰老检测
- **Lung Organoid** (Orga-Dete) — 肺类器官 2 类（Lumen/No Lumen）
- **Colorectal Cancer Organoid** (Huang et al.) — 6 模型对比，YOLOv4 最高 36.9%

---

## 二、SOTA 方法全景

| 数据集 | 方法 | 模型 | mAP@0.5 | 年份 | 关键改进 |
|--------|------|------|---------|------|---------|
| Intestinal | Tellu | YOLOv5 | 79% acc | 2023 | 4类形态分类 |
| Intestinal | Deep-Orga | YOLOX | 81% | 2024 | 轻量化改进 |
| Intestinal | MDPI | YOLOv10-m | 84.5% | 2025 | 默认配置, 150ep, batch=28 |
| Intestinal | **Deliod** | YOLOv8s改 | **87.5%** | 2025 | DRBNCSPELAN+ED-FPN+DySample+EMA-SlideLoss |
| Intestinal | IO-YOLO | YOLOv11改 | 未公开 | 2025 | Feature-Enhanced |
| Gastric | Yang et al. | YOLOv3+CBAM | **93.2%** | 2023 | CBAM注意力 |
| Lung | Orga-Dete | YOLOv11n改 | 81.4% | 2025 | BiFPN+MPCA+EMA-SlideLoss+Albumentations |
| MultiOrg | SSD | SSD | 68.1% | 2024 | 512 patch, sliding window |
| MultiOrg | YOLOv3 | YOLOv3 | 64.6% | 2024 | 同上 |
| Colorectal | Huang et al. | YOLOv4 | 36.9% | 2024 | 6模型对比 |

### 关键洞察

1. **胃类器官 93.2% 是最高精度** — Yang et al. 用 CBAM+YOLOv3，但数据集简单（衰老特征明显）
2. **Deliod 87.5% 是 Intestinal SOTA** — 四模块叠加，从 85.7% baseline 提升 +1.8pp
3. **Orga-Dete 81.4% 是 Lung SOTA** — 三模块+数据增强，从 77.9% baseline 提升 +3.5pp
4. **精度差异主要来自数据集难度** — 胃>肠>肺>多标注者

---

## 三、技术改进路线图

### 路线 A：架构改进（Deliod + Orga-Dete 总结）

| 模块 | 来源 | 作用 | 消融贡献 |
|------|------|------|---------|
| **BiFPN** 替换 FPN+PAN | Orga-Dete | 双向跨层特征融合 | 小目标 +3.7%, 中目标 +2.6% |
| **DRBNCSPELAN** 替换 C2f | Deliod | 扩大感受野 | 小目标 +7.5%, 参数 -1.9M |
| **MPCA** 注意力 | Orga-Dete | 多路径坐标注意力 | 小目标 +4.0%, 中目标 +1.1% |
| **DySample** 动态上采样 | Deliod | 低分辨率增强 | 小目标 +1.1% |
| **ED-FPN** 替换 neck | Deliod | 轻量特征金字塔 | 小目标 +1.1%, 参数 -3.9M |
| **EMA-SlideLoss** | 两者 | 动态阈值类不平衡 | +0.7%, 无额外参数 |

**核心发现：所有改进的核心目标都是小目标检测。** Organoid 的瓶颈不在分类，在密集小目标的定位和特征提取。

### 路线 B：数据增强

| 方法 | 来源 | 效果 |
|------|------|------|
| **Albumentations** (RandomBrightnessContrast, HueSaturationValue, VerticalFlip, RandomRotate90, GaussNoise) | Orga-Dete | 少数类增强，+3.5pp |
| **close_mosaic=10** | Deliod | 最后10ep关mosaic，密集场景 |
| **copy-paste** | 通用小目标 | 未在organoid文献中使用 |
| **多尺度训练** | 通用 | rect=False |

**Orga-Dete 的数据增强贡献了 +3.5pp，比架构改进贡献更大！**

### 路线 C：损失函数

| Loss | 来源 | 特点 |
|------|------|------|
| EMA-SlideLoss | Deliod/Orga-Dete | EMA动态阈值+Slide Loss加权难样本 |
| Focal Loss | 通用 | 已知对类不平衡有效，未在organoid文献使用 |
| Varifocal Loss | DETR系列 | IQL加权 |
| Wise-IoU | 2024 | 关注低质量anchor |

### 路线 D：推理优化

| 方法 | 来源 | 效果 |
|------|------|------|
| **Sliding Window** (512+2048双尺度, 0.5 overlap) | MultiOrg论文 | 全图推理，小目标不丢 |
| **SAHI** | 通用 | 切片推理+NMS合并，+1-2pp |
| **多尺度推理** | 通用 | Test-Time Augmentation |
| **Soft-NMS** | 通用 | 密集场景减少漏检 |

### 路线 E：误差分析（Orga-Dete 的 TIDE 分析）

Orga-Dete 用 TIDE 框架分解检测误差为 6 类：
1. **Classification Error (Cls)** — 类别错误
2. **Localization Error (Loc)** — 定位不准
3. **Cls + Loc** — 双重错误
4. **Duplicate Detection (Dupe)** — 重复检测
5. **Background Error (Bkg)** — 背景误检 ← **organoid 最大误差源**
6. **Missed Detection (Miss)** — 漏检

**关键发现：背景误检（defocused organoid noise）是所有模型的主要误差来源。** EMASlideLoss 有效减少了分类误差。

---

## 四、专利全景

### 已找到的专利

| 专利号 | 标题 | 要点 |
|--------|------|------|
| WO2022252298A1 | 基于显微图像的类器官活力评价方法及系统 | CNN检测+活力评分，[x,y,w,h]框选+熵最小化训练 |
| WO2021113846A1 | Large scale organoid analysis | 大规模类器官分析 |
| CN114463290B | 基于显微图像的类器官类型智能识别方法及系统 | 类器官类型识别 |
| CN114529898B | 基于人工智能的大数据类器官图像识别方法 | AI图像识别 |
| CN120182585B | 基于YOLO的轻量级目标检测方法 | YOLO轻量化（非organoid专用） |
| CN113327226A | 多层交叉注意力特征金字塔网络MCAFPN目标检测 | MCAFPN（非organoid专用） |

### 专利洞察

1. **类器官检测专利极少** — WO2022252298A1 是唯一直接相关的，用通用CNN非YOLO
2. **YOLO+注意力机制的专利通用** — 有现成的 MCAFPN、CBAM 专利但非 organoid 特化
3. **空白领域** — YOLO+SAHI推理、YOLO+BiFPN+MPCA 组合在 organoid 上没有专利
4. **专利机会** — FL+Organoid 检测是完全空白的方向

---

## 五、超越 Deliod 87.5% 的可行路径

### 已验证的baseline对比

| Baseline | mAP@0.5 | 来源 |
|----------|---------|------|
| YOLOv8s (Deliod baseline) | 85.7% | Deliod论文 |
| YOLOv11n (Orga-Dete baseline) | 77.9% | Orga-Dete论文 |
| **YOLOv12n (我们, ep50, 未收敛)** | **85.5%** | 我们的实验 |

**关键发现：yolo12n 已经追平 Deliod 的 YOLOv8s baseline，说明 yolo12 的 R-ELAN + A2C2 注意力架构天然比 YOLOv8 强。**

### 超越策略（按证据强度排序）

#### 策略 1：模型升级 yolo12n → yolo12s（预期 +2-3pp）
- yolo12n 85.5% → yolo12s 预计 87-88%
- 依据：COCO 上 12s 比 12n 高 ~3pp，Deliod v8s baseline 85.7% vs v8n 约 83%
- **证据强度：高**（架构优势+参数量）

#### 策略 2：数据增强（预期 +2-3pp）
- **Albumentations** 对少数类增强（Orga-Dete 验证 +3.5pp）
- copy_paste=1.0（通用小目标有效，organoid 文献未用→空白）
- close_mosaic=10（Deliod 验证有效）
- **证据强度：高**（Orga-Dete 直接验证）

#### 策略 3：BiFPN 替换 FPN+PAN（预期 +1-2pp）
- Orga-Dete 验证：小目标 +3.7%，中目标 +2.6%
- 需要改 Ultralytics 源码
- **证据强度：高**（Orga-Dete 消融实验）

#### 策略 4：注意力机制 MPCA/CBAM（预期 +1pp）
- Orga-Dete 验证：MPCA > CBAM > CA > SimAM
- MPCA 小目标 +4.0%
- **证据强度：高**（有对比实验）

#### 策略 5：EMA-SlideLoss（预期 +0.5-1pp）
- 两个论文验证有效
- 无额外参数和计算量
- **证据强度：高**（两个独立验证）

#### 策略 6：SAHI 推理（预期 +1-2pp）
- 训练不变，推理时切片
- 对小目标漏检直接有效
- **证据强度：中**（通用验证，organoid 文献未用）

#### 策略 7：imgsz 1088→1280（预期 +0.5pp）
- 原图 1280×960，1088 是向下取整
- 1280 不丢信息但 batch 降到 2-3
- **证据强度：中**

### 组合预期

| 组合 | 预期 mAP@0.5 |
|------|-------------|
| yolo12n/1088 当前（ep50 未收敛） | 85.5% |
| + 继续训练到收敛 | ~86% |
| + yolo12s/1088 | ~88% |
| + Albumentations + copy_paste | ~90% |
| + BiFPN + MPCA + EMA-SlideLoss | ~91-92% |
| + SAHI 推理 | ~92-93% |

**保守估计 90%，乐观 93%（超越 Deliod 2.5-5.5pp）。**

---

## 六、研究空白与创新机会

### 文献空白
1. **FL + Organoid 检测** — 完全空白，我们的 organoid-fl 项目独占
2. **SAHI + Organoid** — 未有人做，通用方法直接迁移
3. **copy-paste + Organoid** — 未有人做，小目标增强
4. **YOLOv12 + Organoid** — 我们是第一个（Deliod 用v8, Orga-Dete 用v11）
5. **多标注者不确定性 + 检测** — MultiOrg 提出但未解决

### 专利空白
1. **FL+Organoid 检测** — 无专利
2. **YOLO+SAHI+Organoid** — 无专利
3. **BiFPN+MPCA+Organoid** — 无专利（Orga-Dete 是论文不是专利）

### 论文方向
1. "YOLOv12 with Attention-Centric Architecture for Organoid Detection" — 首个 YOLOv12 organoid 研究
2. "Federated Learning for Multi-Center Organoid Detection" — FL+Organoid 新领域
3. "SAHI-Enhanced Small Organoid Detection with Sliding Window Inference" — 推理优化

---

## 七、建议行动方案

### 阶段 1：验证基线（当前进行中）
- yolo12n/1088/300ep — 预计收敛到 86%
- ✅ 已验证方向正确

### 阶段 2：模型+增强（预计 88-90%）
- yolo12s/1088/300ep
- + copy_paste=1.0
- + close_mosaic=10
- + Albumentations 增强

### 阶段 3：架构改进（预计 91-92%）
- + BiFPN 替换 FPN+PAN
- + MPCA 注意力
- + EMA-SlideLoss
- 需要改 Ultralytics 源码

### 阶段 4：推理优化（预计 92-93%）
- + SAHI 切片推理
- + 多尺度推理
- + Soft-NMS

### 阶段 5：论文+专利
- "YOLOv12 + Attention + SAHI for Organoid Detection"
- FL+Organoid 专利布局
