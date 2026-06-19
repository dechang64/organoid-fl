# 类器官检测超 SOTA 方案

## 目标

超越 Deliod (87.5% mAP@0.5) → 达到 90%+

数据集：Intestinal Organoid (Zenodo 6768583, 840张/23K标注/4类/1280×960)

---

## 当前进展

| 配置 | mAP@0.5 | 状态 |
|------|---------|------|
| yolo12n/640/100ep（旧 baseline） | 0.885 | v12s 旧实验 |
| yolo12n/1088/50ep（当前） | 0.855 | 训练中 |
| Deliod YOLOv8s baseline | 0.857 | SOTA 参考 |
| Deliod YOLOv8s + 全部改进 | 0.875 | SOTA |
| MDPI YOLOv10-m | 0.845 | SOTA 参考 |

---

## SOTA 调研摘要

### 论文全景

| 方法 | 模型 | mAP@0.5 | 年份 | 关键改进 |
|------|------|---------|------|---------|
| Tellu | YOLOv5 | 79% acc | 2023 | 4类形态分类 |
| Deep-Orga | YOLOX | 81% | 2024 | 轻量化 |
| MDPI | YOLOv10-m | 84.5% | 2025 | 默认配置 |
| **Deliod** | YOLOv8s 改 | **87.5%** | 2025 | DRBNCSPELAN+ED-FPN+DySample+EMA-SlideLoss |
| **Orga-Dete** | YOLOv11n 改 | **81.4%** | 2025 | BiFPN+MPCA+EMA-SlideLoss+Albumentations |
| CBAM-YOLOv3 | YOLOv3 改 | 93.2% | 2023 | CBAM（胃类器官，数据集简单） |
| YOLOv4 | YOLOv4 | 36.9% | - | 结直肠类器官（数据集极难） |
| MultiOrg SSD | SSD | 68.1% | 2024 | NeurIPS 2024 benchmark |

### Deliod 消融（85.7→87.5%）

| 改进 | mAP 变化 | 小目标 mAP 变化 | 参数量 |
|------|---------|---------------|--------|
| A: YOLOv8s baseline | 85.7% | 51.1% | 11.3M |
| B: +DRBNCSPELAN 替换 C2f | 86.3% (+0.6) | 58.6% (+7.5) | 9.3M |
| C: +ED-FPN 替换 neck | 86.8% (+0.5) | 59.7% (+1.1) | 5.4M |
| D: +EMA-SlideLoss | 87.5% (+0.7) | 62.7% (+3.0) | 5.4M |

**核心发现：小目标 mAP 从 51.1%→62.7%（+11.6pp），是整体提升的关键。**

### Orga-Dete 消融（77.9→81.4%）

| 改进 | 增益 |
|------|------|
| BiFPN 替换 FPN+PAN | 小目标 +3.7pp |
| MPCA 注意力 | 小目标 +4.0pp |
| EMA-SlideLoss | +0.5pp |
| **Albumentations 数据增强** | **+3.5pp（最大贡献！）** |

**核心发现：数据增强贡献超过架构改进。**

### TIDE 误差分解（Orga-Dete）

主要误差来源：
1. **背景误检（Bkg FP）** — 最大，defocused organoid 噪声
2. **分类错误（Cls）** — EMA-SlideLoss 可缓解
3. **定位错误（Loc）** — 小目标 bbox 不准
4. **漏检（Miss）** — 密集遮挡导致

### 专利调研

| 专利号 | 内容 | 相关性 |
|--------|------|--------|
| WO2021113846A1 | 大规模类器官分析 | 通用框架 |
| WO2022252298A1 | 类器官活力评价（中文） | CNN+活力评分 |
| CN114463290B | 类器官类型智能识别 | 检测+分类 |
| CN114529898B | AI 类器官图像识别 | 通用 |

**专利空白**：FL+Organoid、YOLO+SAHI+Organoid、BiFPN+MPCA+Organoid 均无专利。

---

## 超越路径：三阶段方案

### 阶段 A：模型升级 + 训练 Freebies（目标 89%）

**理论依据**：
- YOLOv12 的 R-ELAN + A2C2 注意力天然比 YOLOv8 强（代差红利 +2-3pp）
- Bag of Freebies 论文证明纯训练技巧可叠加 +3.55pp
- copy_paste 对小目标密集场景特别有效（Deliod/Orga-Dete 都没用）
- Orga-Dete 的 Albumentations 贡献 +3.5pp，说明数据增强是关键杠杆

**配置**：
```powershell
yolo train model=yolo12s.pt data="C:\Users\decha\organoid-fl\data\intestinal_organoid\OrganoidDataset\data.yaml" imgsz=1088 epochs=300 patience=50 batch=4 close_mosaic=15 copy_paste=1.0 mixup=0.15 label_smoothing=0.1 cos_lr=True device=0 name=intestinal_12s_1088_freebies
```

**改动说明**：
| 参数 | 值 | 旧值 | 理由 |
|------|-----|------|------|
| model | yolo12s | yolo12n | 参数量 2.4M→9.3M，COCO +3pp |
| copy_paste | 1.0 | 0 | 小目标增强，Deliod 没用 |
| mixup | 0.15 | 0 | +0.89pp (Bag of Freebies) |
| label_smoothing | 0.1 | 0 | +0.64pp |
| cos_lr | True | False | +0.5pp |
| close_mosaic | 15 | 10 | 更长的稳定期 |

**预期增益**：
- yolo12n→yolo12s：+2-3pp（代差红利）
- copy_paste：+1-2pp（小目标）
- mixup+label_smoothing+cos_lr：+1.5pp
- 总计：+4.5-6.5pp → **88-92%**

**风险**：12GB 显存可能撑不住 yolo12s/1088/batch=4。如果 OOM，降 batch=2 或 imgsz=960。

**验证标准**：
- 50 epoch mAP50 ≥ 0.87 → 方向正确，继续跑
- 50 epoch mAP50 < 0.85 → 检查是否过拟合，考虑关 mixup

---

### 阶段 B：推理增强（目标 91-92%）

**理论依据**：
- SAHI 文献报道 +1.7-3pp 小目标提升
- WBF 比 NMS 更优，+1-2pp
- TTA 内置 augment=True，+0.5-1pp

**三个推理优化叠加**：

#### B1: SAHI 切片推理
```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="runs/detect/intestinal_12s_1088_freebies/weights/best.pt",
    confidence_threshold=0.25,
    device="cuda:0",
)
result = get_sliced_prediction(
    image_path,
    model,
    slice_height=512, slice_width=512,
    overlap_height_ratio=0.2, overlap_width_ratio=0.2,
    postprocess_type="NMS",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.5,
)
```

#### B2: TTA
```powershell
yolo predict model=best.pt source=test images augment=True
```

#### B3: WBF 融合（可选，需训练第二个模型）
```python
from ensemble_boxes import weighted_boxes_fusion
# 融合 yolo12n + yolo12s 预测框
```

**预期增益**：
- SAHI：+1.5-2pp
- TTA：+0.5-1pp
- WBF（可选）：+1-2pp
- 总计：+2-4pp → **91-94%**

**验证标准**：
- SAHI 后 mAP50 ≥ 0.90 → 超越 SOTA
- SAHI 后 mAP50 < 0.89 → 检查切片参数

---

### 阶段 C：架构改进 + 知识蒸馏（目标 93%+）

**理论依据**：
- Deliod 和 Orga-Dete 独立验证了 BiFPN + 注意力 + EMA-SlideLoss 的有效性
- 知识蒸馏在检测任务中稳定 +1-2pp（MGD/CWD/FGD）
- YOLOv12 已有 A2C2 注意力，BiFPN 可直接替换 FPN

#### C1: 架构改进（改 Ultralytics 源码）

**BiFPN 替换 FPN+PAN**：
- 跨层双向连接 + 可学习权重
- Orga-Dete 验证小目标 +3.7pp
- 需要修改 yolo12s.yaml 的 neck 配置

**MPCA 注意力**：
- 多路径坐标注意力，放在 backbone 末端
- Orga-Dete 验证小目标 +4.0pp
- 需要自定义模块

**EMA-SlideLoss**：
- 动态阈值校准，解决类不平衡
- 两个 SOTA 都用了，+0.5-0.7pp
- 需要修改 loss 函数

#### C2: 知识蒸馏

**方案**：
1. 先训练 yolo12x（大模型）作为 teacher
2. 用 MGD（Masked Generative Distillation）蒸馏 yolo12s
3. Student 保持 yolo12s 参数量，推理速度不变

**文献参考**：
- MGD：+1.9pp mAP50
- CWD：+2.5pp mAP50
- FGD：+2-3pp mAP50

**预期增益**：
- BiFPN+MPCA：+2-3pp（但 YOLOv12 部分覆盖）
- EMA-SlideLoss：+0.5pp
- 知识蒸馏：+1-2pp
- 总计：+2-4pp → **93-95%**

**工程成本**：高，需要改 Ultralytics 源码 + 训练 teacher 模型。

---

## 时间表

| 阶段 | 耗时 | 累计 mAP50 | 关键里程碑 |
|------|------|-----------|-----------|
| 当前 yolo12n/1088 | 2h（跑完） | ~86% | 验证基线 |
| 阶段 A: yolo12s+freebies | 3h | **89%** | 超越 Deliod baseline |
| 阶段 B: SAHI+TTA | 1h | **91%** | 超越 Deliod 最终 |
| 阶段 C: 架构+蒸馏 | 8h | **93%** | 远超 SOTA |

---

## 风险与应对

### 风险 1：显存不足
- yolo12s/1088/batch=4 可能 OOM
- **应对**：降 batch=2 或 imgsz=960，或用梯度累积

### 风险 2：copy_paste 反效果
- organoid 密集场景下 copy_paste 可能造成重叠
- **应对**：先跑 copy_paste=0.5，如果 mAP 下降则关闭

### 风险 3：SAHI 推理速度慢
- 切片推理比直接推理慢 5-10 倍
- **应对**：只对大图用 SAHI，小图直接推理

### 风险 4：知识蒸馏 teacher 不够强
- yolo12x 在 840 张图上可能过拟合
- **应对**：用 COCO 预训练权重做 teacher，不重新训练

---

## 评估方法

### 主指标
- **mAP@0.5**（与 SOTA 对标）
- **mAP@0.5:0.95**（综合精度）

### 辅助指标
- **Per-class AP**（监测类不平衡）
- **AP small/medium/large**（监测小目标改进）
- **TIDE 误差分解**（定位主要误差源）

### 对比基线
| 基线 | mAP@0.5 |
|------|---------|
| Deliod baseline (YOLOv8s) | 85.7% |
| Deliod 最终 | 87.5% |
| MDPI (YOLOv10-m) | 84.5% |
| 我们 yolo12n/1088 | ~86% |

---

## 论文/专利布局

### 论文方向
1. **"YOLOv12 with Attention-Centric Architecture for Intestinal Organoid Detection"**
   - 首个 YOLOv12 organoid 检测研究
   - 证明 YOLOv12 R-ELAN + A2C2 天然适合 organoid 密集小目标
   - 超越 Deliod 87.5% 无需架构修改

2. **"SAHI-Enhanced Small Organoid Detection: Training-Free Inference Optimization"**
   - SAHI + TTA + WBF 推理优化框架
   - 不改训练，推理增强 +2-4pp

3. **"Federated Learning for Multi-Center Organoid Detection"**
   - FL+Organoid 新领域（专利空白）

### 专利方向
1. **FL+Organoid 检测系统** — 完全空白
2. **YOLO+SAHI+Organoid 推理方法** — 无专利
3. **BiFPN+MPCA+Organoid 架构** — 无专利

---

## 文件结构

```
organoid-fl/research/
├── organoid_detection_sota_survey.md    # SOTA 全景调研
├── organoid_detection_beyond_sota.md    # 本方案
├── training_logs/                       # 训练记录
│   ├── yolo12n_1088_baseline.csv
│   ├── yolo12s_1088_freebies.csv
│   └── ...
└── inference_results/                   # 推理优化结果
    ├── sahi_results.json
    ├── tta_results.json
    └── wbf_results.json
```

---

## 版本历史

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-06-19 | v1.0 | 初始方案，基于 SOTA 调研 |
