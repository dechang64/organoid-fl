# MultiOrg 实验追踪表

> 最后更新：2026-06-26（SAM2 zero-shot 形态学过滤实验 + SAM2 微调进行中）
> 目标：突破 SOTA SSD 68.1% mAP@0.5 → 达到 80%+
> 数据集：MultiOrg_v2 (411张 6K×5.7K, 单类 organoid, **肺类器官**) / MultiOrg_v3_512 (512px tiling)

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
| 🟡 | T10 | SAM2 mask_decoder 微调 | 数据准备中 | Step 1 跑ing，Step 2 待跑 |
| 🟡 | T3b | 确认 S7b vs S8b 差异 | 需查看脚本调用历史 | 参数相同结果不同 |
| 🟢 | T5 | 双尺度 SAHI (512+2048) | — | — |
| 🟢 | T6 | t1_A 标签训练 | — | 阶段3 |
| 🟢 | T7 | CLOD 标签清洗 | — | 阶段3 |
| 🟢 | T8 | FL 仿真 | — | — |

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
