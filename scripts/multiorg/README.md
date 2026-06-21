# MultiOrg 突破 SOTA 操作指南

## 概览

阶段2（RF-DETR）+ 阶段3（标签噪声处理）并行推进。

```
 scripts/multiorg/
 ├── explore_multiorg.py     ← Step 0: 探查数据结构
 ├── multiorg_tiling_v3.py   ← Step 1: 数据预处理
 ├── train_rfdetr.py         ← Step 2: RF-DETR 训练
 ├── label_consensus.py      ← Step 3: 多标注者共识标签
 └── sahi_inference.py       ← Step 4: 全图滑动窗口推理评估
```

---

## Step 0: 探查数据（必须先跑）

确认每个 image 目录下有哪些标注文件（t0/Annotator_A/B/C？）。

```powershell
cd D:\path\to\organoid-fl\scripts\multiorg
python explore_multiorg.py --src D:\datasets\mutliorg\MultiOrg_v2
```

**输出**：`multiorg_structure_report.json` + `.txt`

**把 report.txt 发给我**，我确认标注文件命名后再跑后续步骤。

---

## Step 1: 数据预处理（单类 + 512px + 丢弃边界 bbox）

```powershell
# 单标注者版（仅 Annotator A）
python multiorg_tiling_v3.py --src D:\datasets\mutliorg\MultiOrg_v2 --dst D:\datasets\MultiOrg_v3_512

# 多标注者版（生成 t0/t1_A/t1_B 三套标签）
python multiorg_tiling_v3.py --src D:\datasets\mutliorg\MultiOrg_v2 --dst D:\datasets\MultiOrg_v3_512_multi --multi-rater
```

**关键修正 vs 旧版**：
- ✅ 单类 organoid（不是 Normal/Macros 两类）
- ✅ 512×512 patch（不是 640）
- ✅ 丢弃跨 patch 边界的 bbox（只保留中心在 patch 内的）
- ✅ 16-bit TIFF → 8-bit RGB

---

## Step 2: RF-DETR 训练（阶段2）

### 安装

```powershell
pip install rfdetr
```

### 训练

```powershell
# RF-DETR Small（推荐 12GB 3060）
python train_rfdetr.py --data D:\datasets\MultiOrg_v3_512\data.yaml --model small --epochs 200

# RF-DETR Base（12GB 刚好，560px auto batch）
python train_rfdetr.py --data D:\datasets\MultiOrg_v3_512\data.yaml --model base --epochs 200
```

### 对比实验（YOLOv12s baseline）

```powershell
yolo detect train model=yolo12s.pt data=D:\datasets\MultiOrg_v3_512\data.yaml epochs=400 imgsz=512 batch=8
```

---

## Step 3: 多标注者共识标签（阶段3）

```powershell
# 生成共识标签
python label_consensus.py --data D:\datasets\MultiOrg_v3_512_multi
```

**输出**：
- `train/labels_consensus/` — 清洗后的共识标签
- `test/labels_consensus/`
- `consensus_report.json` — 噪声统计报告
- `data_consensus.yaml` — 用共识标签训练的配置

### 用共识标签训练

```powershell
# YOLOv12s + 共识标签
yolo detect train model=yolo12s.pt data=D:\datasets\MultiOrg_v3_512_multi\data_consensus.yaml epochs=400 imgsz=512 batch=8

# RF-DETR + 共识标签
python train_rfdetr.py --data D:\datasets\MultiOrg_v3_512_multi\data_consensus.yaml --model small --epochs 200
```

---

## Step 4: 全图滑动窗口推理评估

```powershell
# YOLOv12s + SAHI 推理
python sahi_inference.py --model yolo --weights path\to\best.pt --src D:\datasets\mutliorg\MultiOrg_v2\test --dst .\results\sahi_yolo

# RF-DETR + SAHI 推理
python sahi_inference.py --model rfdetr --weights path\to\checkpoint.pt --src D:\datasets\mutliorg\MultiOrg_v2\test --dst .\results\sahi_rfdetr
```

**关键设计**：
- 全图滑动窗口（不在 patch 上评估）
- 双尺度: 512 + 2048
- WBF 融合重叠检测
- 自动对 ground truth 评估 mAP@0.5

---

## 实验矩阵

| # | 模型 | 标签 | 推理 | 预期 mAP@0.5 |
|---|------|------|------|------------|
| 1 | YOLOv12s | Annotator A | patch | 72-75% (baseline) |
| 2 | RF-DETR Small | Annotator A | patch | 74-77% |
| 3 | YOLOv12s | Consensus | patch | 74-78% |
| 4 | RF-DETR Small | Consensus | patch | 76-80% |
| 5 | YOLOv12s | Consensus | SAHI 双尺度 | 78-82% |
| 6 | **RF-DETR Small** | **Consensus** | **SAHI 双尺度** | **80-83%** |

---

## 时间线

| 步骤 | 时间 | 依赖 |
|------|------|------|
| Step 0 探查 | 2 min | 无 |
| Step 1 tiling | 30-60 min | Step 0 确认 |
| Step 2 RF-DETR 训练 | 3-5h | Step 1 |
| Step 2 YOLOv12s 训练（对照） | 2-3h | Step 1 |
| Step 3 共识标签 | 5 min | Step 1 (--multi-rater) |
| Step 3 共识训练 | 3-5h | Step 1 + Step 3 |
| Step 4 SAHI 评估 | 30-60 min | Step 2/3 模型 |

**总时间**：约 8-12 小时（可并行 Step 2 和 Step 3）

---

## 注意事项

1. **先跑 explore_multiorg.py**，确认标注文件命名后再跑 tiling
2. **RF-DETR 需要 `pip install rfdetr`**，依赖 PyTorch>=2.0
3. **12GB VRAM**：RF-DETR Base at 560px 可跑（auto batch），Small 更稳
4. **多标注者标签**：如果数据中只有 Annotator_A，`--multi-rater` 会自动退化为单标注者
5. **SAHI 推理**：全图推理不在 patch 上评估，这是论文的标准做法

---

## 对比论文 SOTA

| 方法 | mAP@0.5 | 来源 |
|------|---------|------|
| Faster R-CNN | 60.7% | MultiOrg 论文 |
| SSD | 68.1% | MultiOrg 论文（当前 SOTA） |
| YOLOv3 | 64.6% | MultiOrg 论文 |
| RTMDet | 61.6% | MultiOrg 论文 |
| **我们的目标** | **80%+** | RF-DETR + 共识标签 + SAHI |
