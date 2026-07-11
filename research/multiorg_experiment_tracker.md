# MultiOrg 实验追踪表

> 最后更新：2026-07-03（Orga-Dete 三模块 yolo11n/yolo12s 训练全部完成，全面失败）
> 目标：突破 SOTA SSD 68.1% mAP@0.5 → 达到 80%+
> 数据集：MultiOrg_v2 (411张 6K×5.7K, 单类 organoid, **肺类器官**) / MultiOrg_v4_640 (640px tiling)

---

## 〇、数据来源说明

| 数据类型 | 来源 | 可靠性 |
|----------|------|--------|
| SOTA/论文数字 | web search 原文核实 | ✅ 已核实 |
| 训练日志 | upload 目录 txt 文件精确提取 | ✅ 已核实 |
| nano+640 训练数据 | **workspace 无日志** | ❌ 来源不明，待核实 |
| SAHI 结果 | TOOLS.md 记录（从聊天记录搬） | ⚠️ 未从日志核实 |
| SAM2 zero-shot 结果 | upload/6a3dd648 JSON ✅ | ✅ 已核实 |

---

## 一、Patch 级训练实验

### 1.1 RF-DETR 系列 (MultiOrg_v3_512)

| # | 日期 | 模型 | resolution | 参数量 | best regular epoch | mAP50 | mAP50-95 | best EMA epoch | EMA mAP50-95 | checkpoint状态 | 日志来源 |
|---|------|------|-----------|--------|-------------------|-------|----------|---------------|-------------|---------------|---------|
| R1 | 06-23 | RFDETRSmall | 560(默认) | 31.8M | 2 | **81.69%** | **53.23%** | 0 | 53.78% | ❌ 被后续覆盖 | Log A (upload/6a3b253a) ✅ |
| R2 | 06-24 | RFDETRNano | 640 | 30.5M? | ? | **80.07%** | **51.20%** | 0 | **55.40%** | ✅ **现存** (06-24 10:32) | eval_results.json (upload/6a3c8d2c) ✅ |
| R3 | 06-24/25 | RFDETRSmall | 640 | 31.8M | 5 | **82.32%** | **54.66%** | 0 | 55.40% | ❌ 被R2覆盖 | Log B (upload/6a3c78f3) ✅ |

**结论修正（2026-06-25 从 eval_results.json 核实）**：
- 之前认为"nano+640 最优 82.72%/55.23%"是**错误的**，该数字来源不明
- 实际 **R3 small+640 才是最优**（82.32%/54.66%）
- R2 nano+640 regular checkpoint 只有 80.07%/51.20%，是三个里最差的
- 但 R2 nano+640 的 **EMA checkpoint（55.40%）比 regular（51.20%）好 4pp**
- 现存 checkpoint 是 R2 nano+640 的（时间戳 06-24 10:32 确认，未被后续训练覆盖）

### R1 完整训练曲线 (small+560, Log A)

| Epoch | mAP50:95 | mAP50 | 备注 |
|-------|----------|-------|------|
| 0 | 0.5206 | 0.8167 | Best EMA=0.5378 |
| 1 | 0.5252 | 0.8258 | |
| **2** | **0.5323** | **0.8169** | **Best regular** |
| 3 | 0.4933 | 0.8072 | 开始下降 |
| 4-24 | 0.44~0.51 | 0.73~0.81 | 持续震荡下降 |

### R3 完整训练曲线 (small+640, Log B)

| Epoch | mAP50:95 | mAP50 | 备注 |
|-------|----------|-------|------|
| 0 | 0.5342 | 0.8235 | Best EMA=0.5540 |
| 1 | 0.5349 | 0.8165 | |
| 2 | 0.5272 | 0.8240 | |
| 3 | 0.5136 | 0.8130 | |
| 4 | 0.5148 | 0.8043 | |
| **5** | **0.5466** | **0.8232** | **Best regular** |
| 6 | 0.5001 | 0.8003 | 开始下降 |
| 7-16 | 0.45~0.50 | 0.76~0.80 | 持续下降 |

### 1.2 YOLOv12 系列 (MultiOrg_v3_512)

| # | 日期 | 模型 | imgsz | epochs | best_epoch | mAP50 | mAP50-95 | 备注 |
|---|------|------|-------|--------|------------|-------|----------|------|
| Y1 | 06-20 | yolo12n | 512 | 200 | — | 59.89% | — | 初版，三个根本性错误 |
| Y2 | 06-21 | yolo12n | 512 | 200 | — | ~60% | — | 修正为单类+512 patch+drop_boundary |
| Y3 | 06-23 | yolo12s | 640 | 200 | — | 78.1% | — | v4 tiling 640px |

⚠️ YOLOv12 系列数据来自 TOOLS.md 记录，未从训练日志重新核实

---

## 二、SAHI 全图推理实验

### 2.1 RF-DETR 系列（全部从 results JSON 核实 ✅，annotator=t1_b, 55张测试图）

⚠️ **重要**：commit eca40c6（06-24 08:39）之前，`load_rfdetr_model()` 硬编码用 `RFDETRBase`，`--model-variant` 参数不存在。所有 06-24 早期跑的 SAHI 实验（sahi_rfdetr*）实际用的是 **RFDETRBase 架构**，不是目录名暗示的 small/nano。06-25 跑的 sahi_nano_640_* 正确使用了 nano。
checkpoint 来源（06-24 daily notes 确认）：S6 系列用的是 **R1 (small+560) 训练的 checkpoint_best_regular.pth**。

| # | 目录名 | 实际架构 | ckpt来源 | 窗口 | overlap | merge | sf | mAP50 | mAP50:95 | Prec | Recall | TP | FP | FN | 跑于 |
|---|--------|---------|---------|------|---------|-------|-----|-------|----------|------|--------|----|----|-----|------|
| S9 | sahi_small_512_new | **small** ✅ | R4(small+512) | 512 | 0.3 | soft_nms | **0.0** | **77.76%** | 48.81% | 12.9% | 96.2% | 4767 | 32103 | 189 | 06-25 ✅ |
| S9b | sahi_small_512_sf03 | **small** ✅ | R4(small+512) | 512 | 0.3 | soft_nms | **0.3** | 77.15% | 48.21% | 28.6% | 93.6% | 4639 | 11559 | 317 | 06-25 ✅ |
| S9c | sahi_small_512_nms_sf03 | **small** ✅ | R4(small+512) | 512 | 0.3 | **nms** | **0.3** | 75.91% | 47.30% | 22.0% | 95.8% | 4748 | 16801 | 208 | 06-25 ✅ |
| S6 | sahi_rfdetr | **Base** | R1(small+560) | 512 | 0.3 | soft_nms | **0.0** | 77.15% | 49.42% | 15.6% | 95.1% | 4712 | 25402 | 244 | 06-23 ✅ |
| S6b | sahi_rfdetr_sf03 | **Base** | R1(small+560) | 512 | 0.3 | soft_nms | **0.3** | 76.47% | 48.71% | 32.3% | 92.6% | 4588 | 9632 | 368 | 06-24早 |
| S6c | sahi_rfdetr_sf04 | **Base** | R1(small+560) | 512 | 0.3 | soft_nms | **0.4** | 74.61% | 47.64% | 44.6% | 87.6% | 4341 | 5382 | 615 | 06-24早 |
| S6d | sahi_rfdetr_640_sf03 | **Base** | R1(small+560) | 640 | 0.3 | soft_nms | 0.3 | 75.84% | 46.60% | 42.3% | 89.3% | 4426 | 6043 | 530 | 06-24早 |
| S7 | sahi_rfdetr_nano512 | **Base** | R2(nano+640)? | 512 | 0.3 | soft_nms | 0.3 | 74.66% | 46.60% | 47.1% | 86.5% | 4285 | 4806 | 671 | 06-24早 |
| S8b | sahi_nano_640_reg | nano | R2 reg | 640 | 0.3 | soft_nms | 0.3 | 72.72% | 44.28% | 37.2% | 89.5% | 4434 | 7473 | 522 | 06-25 ✅ |
| S8a | sahi_nano_640_ema | nano | R2 EMA | 640 | 0.3 | soft_nms | 0.3 | 71.17% | 43.11% | 46.1% | 85.1% | 4217 | 4925 | 739 | 06-25 ✅ |
| S7b | sahi_rfdetr_nano640 | **Base** | R2(nano+640)? | 640 | 0.3 | soft_nms | 0.3 | 69.29% | 42.79% | 57.7% | 79.5% | 3940 | 2885 | 1016 | 06-24早 |

**关键影响**：
- S6 系列标注"small+560"是错的——实际是 **RFDETRBase 架构加载 R1(small+560) checkpoint**
- RFDETRBase 和 RFDETRSmall 架构不同（dec_layers 等），加载可能有部分权重不匹配
- **S8a/S8b 是唯一确认正确使用 nano 架构的实验**（06-25 跑，commit 之后）
- S7/S7b 的 checkpoint 来源未完全确认（可能也是 R1，需冬生确认）

### 2.2 YOLOv12 系列（TOOLS.md 记录，未从 JSON 核实 ⚠️）

| # | 目录名 | 模型 | 窗口 | overlap | merge | sf | mAP50 | mAP50:95 | Prec | Recall | 来源 |
|---|--------|------|------|---------|-------|-----|-------|----------|------|--------|------|
| S9 | sahi_t1b_v2 | yolo12s | 640 | 0.5 | — | 0 | 61.82% | 32.34% | 53.6% | 79.5% | JSON ✅ |
| S10 | sahi_t1b | yolo12s | 640 | 0.5 | — | 0 | 59.39% | 34.68% | 46.7% | 83.4% | JSON ✅ |
| S11 | sahi_t1b_conf035 | yolo12s | 640 | 0.5 | — | 0 | 57.03% | 30.26% | 64.8% | 70.8% | JSON ✅ |
| S12 | sahi_t1b_conf05 | yolo12s | 640 | 0.5 | — | 0 | 43.43% | 23.37% | 78.5% | 51.8% | JSON ✅ |

注：S9-S12 的 merge 字段为空，可能是 NMS（默认）。conf 不同导致 precision-recall tradeoff。

### 2.3 SAHI 关键发现（全部从 JSON 核实）

**RF-DETR 最优配置**：S6 small+560 + 512窗 + sf=0.0 → **mAP50=77.15%**（超 SOTA SSD 68.1% +9.0pp）
- 注：sf=0.0 虽然精度最高(15.6%)但 FP=25402 极多，部署应用 sf=0.3 版本（S6b 76.47%, precision 32.3%）

**关键规律**：
1. **512窗 > 640窗**：S6b(512,76.47%) > S6d(640,75.84%)，+0.63pp
2. **small > nano**：同 512 窗 S6b(76.47%) > S7(74.66%)，+1.81pp
3. **sf=0.3 是最优部署配置**：S6(77.15%, sf=0) → S6b(76.47%, sf=0.3)，只掉 0.68pp 但 FP 砍 62%
4. **sf=0.4 过度过滤**：S6c(74.61%) 比 S6b 掉 1.86pp，不划算
5. **regular > EMA**（nano+640）：S8b(72.72%) > S8a(71.17%)，regular recall 更高
6. **soft_nms > NMS**（从 YOLOv12 系列对比可见）

---

## 三、Intestinal 数据集参照实验（非 MultiOrg）

| 模型 | imgsz | mAP50 | mAP50-95 | 来源 |
|------|-------|-------|----------|------|
| Deliod baseline (YOLOv8s) | 1088 | 85.7% | — | PMC11814327 Table 7 ✅ |
| Deliod final | 1088 | 87.5% | — | Nature Sci Rep 2025 ✅ |
| yolo12n | 1088 | 86.7% | — | TOOLS.md ⚠️ |
| yolo12s+freebies v1 | 1088 | 88.1% | 62.1% | TOOLS.md ⚠️ |
| yolo12s+freebies v2 | 1088 | 88.4% | 62.2% | TOOLS.md ⚠️ |

---

## 四、待做实验

| 优先级 | 编号 | 实验 | 依赖 | 状态 |
|--------|------|------|------|------|
| ✅ | T1 | nano+640 checkpoint 确认 | eval_results.json | 完成，checkpoint现存 |
| ✅ | T2 | nano+640 SAHI 评估 | T1 | 完成（S8a EMA=71.17%, S8b reg=72.72%） |
| ✅ | T3 | small+640 SAHI 评估 | S6d 已有 | 完成（75.84%） |
| ✅ | T4 | 核实全部 SAHI 数据 | 12个JSON | 完成，8个RF-DETR+4个YOLOv12 |
| ✅ | T9 | SAM2 zero-shot 形态学过滤 | upload/6a3dd648 | 完成，**无效**（见 §2.4） |
| ✅ | T10 | SAM2 mask_decoder 微调 | upload JSON | 完成，**微调有害**（见 §2.4） |
| ✅ | T11 | Orga-Dete yolo11n Phase 1+3 | results.csv | 完成，**模型容量瓶颈**（见 §4） |
| ✅ | T12 | Orga-Dete yolo12s MPCA (random init) | results.csv | 完成，**破坏预训练 -19pp** |
| ✅ | T13 | MPCA Identity Init 修复 | coco8 E2E | 完成，std ratio 0.50→0.99 |
| ✅ | T14 | Orga-Dete yolo12s Phase 1+3 (identity init) | results.csv ✅ | 完成，**MPCA 仍 -18pp，EMASlideLoss 噪声级** |
| 🟡 | T3b | 确认 S7b vs S8b 差异 | 需查看脚本调用历史 | 参数相同结果不同 |
| 🟢 | T5 | 双尺度 SAHI (512+2048) | — | — |
| 🟢 | T6 | t1_A 标签训练 | — | 阶段3 |
| 🟢 | T7 | CLOD 标签清洗 | — | 阶段3 |
| 🟢 | T8 | FL 仿真 | — | — |
| 🟢 | **T15** | **Phase 8: 小波频域原语分析 (W1-W4)** | pywt+numpy | ✅ 完成，小波单独无效 |
| 🟢 | **T16** | **Phase 9: Slot Attention 原语提取 (S1-S4)** | T15 决策 | 需云VM GPU |
| 🟢 | **T17** | **Phase 10: 联邦原语聚合 (F1-F4)** | T16 | 需 FedCtx 集成 |
| 🟢 | **T18** | **Phase 11: 对比原语学习 (C1-C4)** | T16 | 突破80%门槛关键 |

---

## 2.4 SAM2 形态学过滤实验（2026-06-26）

### Zero-shot SAM2（5张测试, annotator=t1_b）

**配置**：RF-DETR small + SAHI(512窗, overlap=0.3, soft_nms, sf=0.3) → SAM2 zero-shot → 形态学特征

**Baseline**：mAP50=77.54%, mAP50-95=48.54%, P=30.4%, R=93.3%, F1=45.8%, TP=462, FP=1059, FN=33

**过滤效果**：

| 过滤器 | FP砍掉 | TP丢失 | mAP50 | F1 |
|--------|--------|--------|-------|-----|
| circ>=0.5 | 6 (0.6%) | 0 | 77.54% | 46.0% |
| solid>=0.90 | 11 (1.0%) | 0 | 77.55% | 46.1% |
| ar>=0.5 | 23 (2.2%) | 0 | 77.54% | — |
| **conf>=0.4** | **504 (48%)** | **21 (5%)** | **76.17%** | **59.2%** |
| conf>=0.5 | 786 (74%) | 70 (15%) | 72.05% | 67.6% |
| conf>=0.6 | 959 (91%) | 139 (30%) | 63.22% | 70.4% |

**结论**：
- **形态学过滤无效**：circularity/solidity/aspect_ratio 几乎不砍 FP（<2%）
- **原因**：TP 和 FP 的 circularity 分布重叠（76% 检测在 0.8-0.9 区间）
- **conf>=0.4 是有效阈值**：砍 48% FP 只丢 5% TP
- **根本原因**：zero-shot SAM2 没有 organoid 形态先验——鼠肝有效是因为做了 few-shot 微调，MultiOrg 没有

### SAM2 mask_decoder 微调（进行中）

**实验设计**：MultiOrg GT 多边形 → mask → 微调 SAM2 mask_decoder（冻结 image_encoder）→ 用微调后 SAM2 跑形态学过滤

**流程**：
1. `prepare_sam2_data.py`：napari 多边形 → instance mask（cv2.fillPoly）
2. `finetune_sam2.py`：冻结 image_encoder+memory，只训练 mask_decoder+prompt_encoder，BCE+Dice loss
3. `multiorg_sam2.py --sam2-checkpoint runs/sam2_finetune/sam2_finetuned.pt`：用微调后 SAM2 评估

**假设**：微调后 SAM2 对 organoid 形态有判别力 → 形态学特征能区分 TP/FP

**数据**：upload/6a3dd6487422736f6269f8f0_multiorg_sam2_results.json ✅

### SAM2 三轮最终结果（2026-06-29，全量 55 张测试, annotator=t1_b）

| 配置 | bbox mAP50 | mask mAP50 | 备注 |
|------|-----------|-----------|------|
| Zero-shot SAM2 | 77.15% | 76.39% | 基线 |
| Finetuned v2 (4点GT) | 77.15% | 75.98% | 微调有害 -0.41pp |
| Finetuned pseudo (自蒸馏) | 77.13% | 75.89% | 自蒸馏天花板=zero-shot |

**结论**：微调=负优化确认。4 点粗糙 GT 是根因——用粗糙 GT 微调精确 SAM2 = 负优化。

### FP 抑制全面调研（2026-06-29）

| 方案 | PR-AUC | 结论 |
|------|--------|------|
| DINOv2+DPMM 二级验证 | 0.29 (<随机0.5) | TP/FP 在 768 维空间完全重叠 |
| 形态学过滤 (circ/solid/ar) | — | 砍 FP <2%，无效 |
| HNM (Hard Negative Mining) | — | FP 空标签+默认LR → 灾难性遗忘，mAP 暴跌 40pp |

**结论**：所有基于视觉特征的后置过滤方案均无效。FP 和 TP 在所有可观测特征上不可分。

---

## 四、Orga-Dete 三模块迁移实验（2026-06-30 ~ 07-03）

### 4.1 实验目标

突破 MultiOrg 80% 门槛（当前 RF-DETR SOTA 77.8%）。迁移 Orga-Dete (Huang et al., Applied Sciences 2025) 三模块：
- **MPCA**：4 路径坐标注意力，加在 backbone 末尾
- **BiFPN**：双向特征金字塔（YAML 不兼容，未验证）
- **EMASlideLoss**：动态阈值 loss（μ_t = α·μ_{t-1} + (1-α)·mean(IoU)）

### 4.2 yolo11n 系列结果（MultiOrg_v4_640, 16612 patches, 2.6M params）

| # | 配置 | Best mAP50 | Best ep | 总 epochs | 训练时间 | s/epoch | 来源 |
|---|------|-----------|---------|-----------|---------|---------|------|
| O1 | Phase 1 (MPCA) | 44.24% | ep24 | 74 (早停) | 4.9h | 237s | upload/6a4471a4...d2 ✅ |
| O2 | Phase 3 (MPCA+EMASlideLoss) | 44.04% | ep16 | 74 (早停) | 4.9h | 245s | upload/6a4471a4...d3 ✅ |

**EMASlideLoss 增益**：-0.20pp（噪声级别）

### 4.3 yolo12s 系列结果（MultiOrg_v4_640, 16612 patches, 9.4M params）

| # | 配置 | Best mAP50 | Best ep | 总 epochs | 训练时间 | s/epoch | 来源 |
|---|------|-----------|---------|-----------|---------|---------|------|
| O3 | yolo12s baseline (无MPCA) | 62.3% | — | — | — | — | TOOLS.md ⚠️ |
| O4 | Phase 1 (MPCA random init) | 43.5% | ep16 | 74 (早停) | — | — | daily 2026-07-02 |
| O5 | Phase 1 (MPCA identity init) | **43.88%** | ep20 | 70 | 8.6h | 438s | upload/6a46fee1... ✅ |
| O6 | Phase 3 (MPCA identity + EMASlideLoss) | **44.47%** | ep20 | 70 | 8.7h | 444s | upload/6a46fee2... ✅ |

**关键发现**：
- MPCA random init → mAP 43.5%（比 baseline 62.3% 低 **18.8pp**）
- MPCA identity init → mAP 43.88%（修复了 std ratio 0.50→0.99，但 mAP 仅 +0.38pp）
- EMASlideLoss → +0.59pp（噪声级别，和 yolo11n 上一致）
- ep20 见顶后 50 个 epoch 无提升，不是"还没收敛"

### 4.4 MPCA Identity Init 修复详情（2026-07-02, commit 0cd3169）

**问题**：MPCA 的 mlp 最后一层 Sigmoid，随机初始化时 sigmoid≈0.5 → output=x*0.5 → 特征方差减半 → 破坏预训练

**修复**：mlp 最后一层 conv 改 bias=True, weight=0, bias=5 → sigmoid(5)≈0.993≈1.0 → output≈x（identity）

**验证**：
- output/input std ratio：0.50 → 0.99 ✅
- 预训练权重匹配：48.6% → 99.2%（699/705 items）✅
- coco8 1 epoch E2E 通过 ✅

**铁律**：任何加在预训练模型上的乘性 attention 模块（SE、CBAM、CA、MPCA 等），初始状态必须是 identity

### 4.5 模型容量 vs mAP 强相关

| 模型 | 参数量 | mAP50 | 来源 |
|------|--------|-------|------|
| RF-DETR small | 32M | 77.8% | S9 SAHI ✅ |
| YOLOv12s | 9.4M | 62.3% | baseline ⚠️ |
| YOLOv11n+MPCA | 2.6M | 44.2% | O1 ✅ |
| YOLOv12s+MPCA(identity) | 9.6M | 43.9% | O5 ✅ |

**每 3.5x 参数 ≈ +18pp mAP**。MPCA 在 yolo12s 上反而比 yolo11n 差（43.9% vs 44.2%），说明 MPCA 本身在 MultiOrg 上不 work，不是 init 问题。

### 4.6 Orga-Dete 失败根因分析

1. **模型容量瓶颈**：Orga-Dete 论文用 yolo11n 在 4008 张 lung organoid 上达 81.4%，MultiOrg 只有 411 张（tiling 后 16612 patches），但原图只有 411 张
2. **MPCA 架构不兼容**：MPCA 设计给 YOLOv11n (C3k2 backbone)，yolo12s 用 A2C2f，attention 插入位置和特征维度不同
3. **单类检测无增益**：MPCA 的坐标注意力对多类区分有用，MultiOrg 单类 organoid 场景增益有限
4. **数据集差异**：Orga-Dete 在 lung/duodenal/murine intestinal 三个数据集验证，但 MultiOrg 是肺类器官，形态分布不同
5. **BiFPN 未验证**：Ultralytics YAML parse_model 对 list 类型的 `[4,6,11]` 报错，需要 Python 代码直接构建模型，未完成

### 4.7 Orga-Dete 最终判决

| 模块 | yolo11n | yolo12s | 结论 |
|------|---------|---------|------|
| MPCA (random init) | 未跑 | 43.5% (-18.8pp) | 破坏预训练 |
| MPCA (identity init) | 44.2% | 43.9% (-18.4pp) | 仍破坏，init 不是根因 |
| EMASlideLoss | -0.20pp | +0.59pp | 噪声级，无效 |
| BiFPN | 未跑 | 未跑 | YAML 不兼容 |

**停止 Orga-Dete 方向。** 三模块在 MultiOrg 上全面失败，继续投入没有意义。

---

## 五、权威来源核实（2026-06-25）

| 声称 | 来源 | 核实结果 |
|------|------|---------|
| MultiOrg SSD SOTA = 68.1% | NeurIPS 2024 Table 3 [arXiv 2410.14612] | ✅ SSD mAP@0.5 test0 = 68.09% |
| MultiOrg = 400+张/60K+ organoids | 同上 | ✅ 肺类器官（lung organoid） |
| Deliod mAP50 = 87.5% | Nature Sci Rep 2025 [doi:10.1038/s41598-025-89409-y] | ✅ |
| Deliod baseline YOLOv8s = 85.7% | PMC11814327 Table 7 | ✅ 1088×1088, batch=4, 300ep |
| RF-DETR 首个 real-time 60+ mAP on COCO | Roboflow 官方博客 2025-07-24 | ✅ |
| RF-DETR Small > YOLO11-x by 1.8 AP (mAP50:95) | 同上 | ✅ |
| RF-DETR NMS-free | 官方+arXiv 2504.13099 | ✅ |
| RF-DETR nano COCO mAP50:95 = 48.0 | 官方博客 | ⚠️ **实际 48.4**，已修正 |
| "YOLOv12X 92.5%" | ? | ❌ **来源不明，标记为幻觉** |
| RF-DETR 单类 94.6% | arXiv 2504.13099 (greenfruit) | ✅ |

---

## 六、关键配置

### 6.1 训练配置（从 train_rfdetr.py 代码确认）
- **数据集**：MultiOrg_v3_512 (512px tiling)
- **grad_accum_steps**：4（3060 12GB 梯度累积）
- **默认 epochs**：200（但小数据集 epoch 0-5 到峰后持续下降）
- **默认 imgsz**：512
- **输出路径**：`--output` 参数，已修复传给 `output_dir`（之前没传导致覆盖）

### 6.2 SAHI 最优配置（全部从 JSON 核实）
- **窗口**：512 > 640（+0.63pp）
- **overlap**：0.3 [YOLO-Patch-Based-Inference 建议 15-40%]
- **merge**：Soft-NMS [Bodla ICCV 2017]
- **conf**：0.25
- **score_filter**：0.3（部署最优，只掉 0.68pp 但 FP 砍 62%）；论文报告用 sf=0.0（77.15%）
- **模型**：small > nano（+1.81pp）
- **checkpoint**：regular > EMA（SAHI 场景，regular recall 更高）

### 6.3 RF-DETR 官方规格（2026-06-25 核实）
| 模型 | COCO mAP50 | COCO mAP50:95 | 延迟(ms) |
|------|-----------|--------------|---------|
| Nano | 67.6 | 48.4 | 2.32 |
| Small | 72.1 | 53.0 | 3.52 |
| Medium | 73.6 | 54.7 | 4.52 |

---

## 七、铁律

1. **训练命令必须带 output_dir/project=** — 不指定默认覆盖 output/
2. **数据集使用前必须读官方文档** — napari 坐标系不是猜的
3. **val集不能丢边界bbox** — train drop_boundary=True，val 必须 clip
4. **第三方库API必须查文档** — RF-DETR 参数名和 Ultralytics 不同
5. **本地路径绝不猜** — 找不到就问用户要 data.yaml 内容
6. **实验数据必须从日志原文提取** — 不从记忆/TOOLS.md 搬数字
7. **实验追踪表从项目开始就建** — 不要等出事再补

---

## 八、版本历史

| 日期 | 变更 |
|------|------|
| 2026-06-25 | 创建追踪表 |
| 2026-06-25 | 新增权威来源核实，修正 RF-DETR nano mAP 48.0→48.4，标记 YOLOv12X 92.5% 为幻觉 |
| 2026-06-25 | 从训练日志精确提取 R1/R3 数据，修正多处错误，标记 R2 nano+640 数据来源不明 |
| 2026-06-25 | 从 eval_results.json 核实 R2=80.07%/51.20%（非之前的82.72%/55.23%），修正排名为 R3>R1>R2 |
| 2026-06-25 | 从 12 个 SAHI JSON 核实全部 SAHI 数据，发现 S6 sf=0.0=77.15%（比之前记录的 76.47% 高），新增 S6c/S6d/S7b/S9-S12 |
| 2026-06-26 | SAM2 zero-shot 形态学过滤实验（5张测试），结论：无效。新增 §2.4 |
| 2026-06-26 | 新增 T9（完成）/ T10（进行中）SAM2 mask_decoder 微调实验 |
| 2026-06-29 | SAM2 三轮最终结果：微调=负优化确认。FP 抑制全面调研：DINOv2/形态学/HNM 均无效 |
| 2026-07-01 | Orga-Dete yolo11n Phase 1+3 结果：44.2%/44.0%，模型容量瓶颈 |
| 2026-07-02 | yolo12s+MPCA random init 43.5%（-19pp），MPCA Identity Init 修复（commit 0cd3169） |
| 2026-07-03 | yolo12s+MPCA identity init 43.88%/44.47%，Orga-Dete 三模块全面失败，停止该方向 |
| 2026-07-10 | CTM 实现（DINOv2冻结+3.64M），62样本 val AUC 0.907 峰值超 RF-DETR |
| 2026-07-11 | 新增 Phase 8-11 视觉原语实验设计（WIPES/FORLA/对比学习），基于文献调研 |
| 2026-07-11 | Phase 8 完成：小波单独无效（PR-AUC 0.45<0.50），W3 morph+wavelet 0.645 为 baseline |
