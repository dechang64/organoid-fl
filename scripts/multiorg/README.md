# MultiOrg 突破 SOTA 操作指南

## 概览

阶段2（RF-DETR）+ 阶段3（标签噪声处理）并行推进。

```
scripts/multiorg/
 ├── explore_multiorg.py     ← Step 0: 探查数据结构 ✅ 已完成
 ├── multiorg_tiling_v3.py   ← Step 1: 数据预处理
 ├── train_rfdetr.py         ← Step 2: RF-DETR 训练
 ├── label_consensus.py      ← Step 3: 多标注者共识标签
 └── sahi_inference.py       ← Step 4: 全图滑动窗口推理评估
```

---

## 数据结构（explore_multiorg.py 确认）

| Split | 图片数 | 标注文件 | 多标注者？ |
|-------|--------|---------|-----------|
| Train | 356 | `image_N_Annotator_A.json` 或 `image_N_Annotator_B.json` | ❌ 每图1个标注者 |
| Test  | 55  | `image_N_t0.json` + `image_N_t1_a.json` + `image_N_t1_b.json` | ✅ 每图3套标注 |

**关键**：多标注者共识只在 test set 可用。Train set 每张图只有一个标注者（A 或 B），共识 = 合并。

---

## Step 1: 数据预处理

### 单标注者版（简单，用于 YOLOv12 baseline）
```powershell
python multiorg_tiling_v3.py --src D:\datasets\mutliorg\MultiOrg_v2 --dst D:\datasets\MultiOrg_v3_512
```
- Train: 合并 A+B 标注 → `train/labels/`
- Test: 只用 t0 → `test/labels/`

### 多标注者版（用于共识标签实验）
```powershell
python multiorg_tiling_v3.py --src D:\datasets\mutliorg\MultiOrg_v2 --dst D:\datasets\MultiOrg_v3_512_multi --multi-rater
```
- Train: A → `train/labels_annotator_a/`，B → `train/labels_annotator_b/`
- Test: t0 → `test/labels_t0/`，t1_a → `test/labels_t1_a/`，t1_b → `test/labels_t1_b/`

**关键修正 vs 旧版**：
- ✅ 单类 organoid（不是 Normal/Macros 两类）
- ✅ 512 patch（不是 640）
- ✅ 丢弃边界 bbox（中心不在 patch 内的不计）
- ✅ 正确处理 train 的单标注者情况（A 或 B，不是每图都有 A）

---

## Step 2: RF-DETR 训练（阶段2）

```powershell
# 安装
pip install rfdetr

# 训练 RF-DETR Small（12GB 更稳）
python train_rfdetr.py --data D:\datasets\MultiOrg_v3_512\data.yaml --model small --epochs 200

# 或 RF-DETR Base（12GB 可跑, auto batch）
python train_rfdetr.py --data D:\datasets\MultiOrg_v3_512\data.yaml --model base --epochs 200
```

**YOLOv12s 对照组**（同时跑）：
```powershell
yolo detect train model=yolo12s.pt data="D:\datasets\MultiOrg_v3_512\data.yaml" epochs=400 imgsz=512 batch=8
```

---

## Step 3: 共识标签（阶段3）

```powershell
# 先用多标注者版 tiling
python multiorg_tiling_v3.py --src D:\datasets\mutliorg\MultiOrg_v2 --dst D:\datasets\MultiOrg_v3_512_multi --multi-rater

# 生成共识标签
python label_consensus.py --data D:\datasets\MultiOrg_v3_512_multi
```

**处理逻辑**：
- Train: 合并 `labels_annotator_a/` + `labels_annotator_b/` → `labels/`（同一 patch 只有一个标注者，直接复制）
- Test: IoU 投票 `labels_t0/` + `labels_t1_a/` + `labels_t1_b/` → `labels/`（≥2 标注者同意 = 高置信度共识）

**然后用共识标签训练**：
```powershell
yolo detect train model=yolo12s.pt data="D:\datasets\MultiOrg_v3_512_multi\data_consensus.yaml" epochs=400 imgsz=512 batch=8
```

---

## Step 4: SAHI 全图推理评估

```powershell
# YOLOv12 推理
python sahi_inference.py --model yolo --weights path\to\best.pt --src D:\datasets\mutliorg\MultiOrg_v2\test --dst .\results\sahi_yolo

# RF-DETR 推理
python sahi_inference.py --model rfdetr --weights path\to\checkpoint.pt --src D:\datasets\mutliorg\MultiOrg_v2\test --dst .\results\sahi_rfdetr
```

---

## 实验矩阵

| 实验 | 模型 | 训练标签 | 推理 | 预期 mAP50 |
|------|------|---------|------|-----------|
| A: YOLO baseline | YOLOv12s | 单标注者合并 | patch 评估 | 72-75% |
| B: RF-DETR | RF-DETR Small | 单标注者合并 | patch 评估 | 74-77% |
| C: YOLO + 共识 | YOLOv12s | 共识标签 | patch 评估 | 74-78% |
| D: RF-DETR + 共识 | RF-DETR Small | 共识标签 | patch 评估 | 76-80% |
| E: 最佳模型 + SAHI | A/B/C/D 最佳 | 共识标签 | SAHI 双尺度 | 80-83% |

---

## 时间线

| 步骤 | 时间 | 依赖 |
|------|------|------|
| Step 0 探查 | ✅ 完成 | — |
| Step 1 tiling（两种） | 60-90 min | 无 |
| Step 2 RF-DETR 训练 | 3-5h | Step 1 单标注者版 |
| Step 2 YOLOv12s 训练（对照） | 2-3h | Step 1 单标注者版 |
| Step 3 共识标签生成 | 5 min | Step 1 多标注者版 |
| Step 3 共识训练 | 3-5h | Step 1 多标注者 + Step 3 |
| Step 4 SAHI 评估 | 30-60 min | Step 2/3 模型 |

**总时间**：约 8-12 小时（Step 2 和 Step 3 可并行）

---

## 注意事项

1. **先跑 explore_multiorg.py** ✅ 已完成
2. **RF-DETR 需要 `pip install rfdetr`**
3. **12GB VRAM**：RF-DETR Base at 560px 可跑（auto batch），Small 更稳
4. **Train 每张图只有一个标注者**（A 或 B），不能做 IoU 共识，只能合并
5. **Test 每张图有三套标注**（t0/t1_a/t1_b），可以做 IoU 共识
6. **SAHI 推理**：全图推理不在 patch 上评估，这是论文的标准做法
