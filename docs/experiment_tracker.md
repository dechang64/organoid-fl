# 鼠肝 Organoid 实验追踪表

## v2 实验矩阵（2026-07-08, B2 图片反序 Bug 修复后）

### 数据分配
| 批次 | 张数 | 分辨率 | Train | Val | Test | Fewshot |
|------|------|--------|-------|-----|------|---------|
| B1 | 10 | 2592×1944 | 6 | 2 | 2 | 3 |
| B2 | 10 | 4000×3000 | 6 | 2 | 2 | 3 |
| B3 | 20 | 4000×3000 | 12 | 4 | 4 | 3 |
| Central | 40 | 混合 | 24 | 8 | — | — |

### Step 1: Baseline & Ceiling (RF-DETR Small, COCO 预训练)
| # | 实验 | 训练量 | Resolution | Batch | Epochs | Best Epoch | val mAP50 | val mAP50-95 | test F1 |
|---|------|--------|-----------|-------|--------|-----------|-----------|-------------|---------|
| 1 | B1 full | 6 | 544 | 4 | 20 | ep2 | 1.000 | 0.600 | **1.00** |
| 2 | B2 full | 6 | 768 | 1 | 20 | ep11 | 0.818 | 0.684 | **0.80** |
| 3 | B3 full | 12 | 768 | 1 | 14 | ep6 | 0.881 | 0.724 | **0.75** |
| 4 | Central | 24 | 640 | 2 | 16 | ep14 | 0.872 | 0.672 | **0.71** |

### Step 2: 跨域迁移 (B1 checkpoint → B2/B3)
| # | 实验 | 目标 | 模式 | val mAP50 | test F1 |
|---|------|------|------|-----------|---------|
| 5 | B1→B2 zeroshot | B2 | 直接推理 | — | 0.22 |
| 6 | B1→B2 fewshot | B2 | 3-shot 微调 | 0.814 | **0.92** |
| 7 | B1→B3 zeroshot | B3 | 直接推理 | — | 0.00 |
| 8 | B1→B3 fewshot | B3 | 3-shot 微调 | 0.881 | 0.67 |

### Step 3: 传统 CV & SAM2
| # | 实验 | 方法 | B1 F1 | B2 F1 | B3 F1 |
|---|------|------|-------|-------|-------|
| — | Traditional CV | Otsu 阈值分割 | 0.00 | 0.00 | 0.00 |
| — | SAM2 (full) | bbox-guided 分割 | 1.00 | 0.80 | 0.75 |
| — | SAM2 (fewshot) | bbox-guided 分割 | — | 0.92 | 0.67 |

### Step 4: 联邦学习 (YOLOv12n, 10 rounds × 10 local epochs)
| # | Tag | Gate | Order | Final mAP50 | Final mAP50-95 | Best mAP50-95 | Best@Round |
|---|-----|------|-------|------------|---------------|--------------|------------|
| 9 | F1 | none | b1→b2→b3 | 0.30 | 0.24 | 0.26 | R3 |
| 10 | F2 | soft (EWA) | b1→b2→b3 | 0.75 | 0.47 | 0.47 | R10 |
| 11 | F3 | hard | b1→b2→b3 | **0.92** | **0.62** | 0.62 | R5 |
| 12 | F4 | hard | b3→b2→b1 | 0.91 | 0.51 | 0.51 | R10 |

### FL 收敛曲线 (逐轮 mAP50)
| Round | F1 (none) | F2 (soft) | F3 (hard) | F4 (hard rev) |
|-------|-----------|-----------|-----------|---------------|
| 1 | 0.28 | 0.28 | 0.37 | 0.31 |
| 2 | 0.32 | 0.32 | 0.31 | 0.54 |
| 3 | 0.33 | 0.35 | 0.74 | 0.71 |
| 4 | 0.31 | 0.35 | **0.92** | 0.84 |
| 5 | 0.31 | 0.35 | 0.92 | 0.84 |
| 6 | 0.28 | 0.58 | 0.92 | 0.84 |
| 7 | 0.28 | 0.65 | 0.92 | 0.84 |
| 8 | 0.30 | 0.67 | 0.92 | 0.86 |
| 9 | 0.30 | 0.72 | 0.92 | 0.86 |
| 10 | 0.30 | 0.75 | 0.92 | 0.91 |

---

## B2 图片反序 Bug 修复记录

### 问题
- B2 RF-DETR 训练 loss=15.4 (B1/B3=6.9), mAP=0
- B2 test n_det=0, 模型学到"全抑制"

### 根因
- 图片重命名时排序方向和标注图相反
- 标注图: 微信图片_xxx.jpg, 原图: image_00.jpg
- 按排序索引配对 → image_00 实际是 image_09 的图
- 正确标签配在错误图片上

### 诊断脚本链
1. check_pairing.py — MD5 对比, 图片本身没问题
2. verify_labels.py — 标签和 annotations.json 全匹配
3. check_image_naming.py — 像素相似度对比 → B2 全部反序 (0/10)

### 修复
- fix_b2_reverse.py 反转 B2 images
- extract_labels_v2.py 从 annotations.json 重新生成 labels
- B2 重训: ep0 mAP50=0.555, ep8 best=0.803/0.675, ep20 final=0.812/0.645

### B2 修复前后对比
| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 初始 loss | 15.4 | 正常 |
| best val mAP50 | 0.011 | 0.818 |
| best val mAP50-95 | 0.003 | 0.684 |
| test F1 | 0.00 | 0.80 |

---

## v1 历史实验（参考, 数据可能有误）
- v1 使用 YOLOv12n + CPU + 旧数据分配 (无独立 test set)
- E4-E11 顺序链式 FL: E6(soft) 最优 0.506
- B2 旧数据 2592×1944 (搞混), 修复后 4000×3000
- v1 结论 "bbox 到天花板" 需要重新评估

---

## MultiOrg 视觉原语实验追踪（2026-07-08 ~ 07-13）

### 数据基础
- 数据集：MultiOrg v3_512 tiling，16198 crops（4629 TP + 11569 FP）
- TP/FP 定义：IoU≥0.5 = TP（matched=True），和 mAP50 一致
- Train/Val/Test 按图像分割（0 泄露）：6316/2409/2445
- RF-DETR confidence baseline：ROC-AUC=0.888, PR-AUC=0.828

### Phase 1: 形态学特征提取（2026-07-08）
| 数据集 | 样本 | 形态学 PR-AUC | confidence PR-AUC | LOO LR (all) | LOO LR (morph) |
|--------|------|-------------|-------------------|-------------|-----------------|
| 鼠肝 | 39 (27TP+12FP) | 全 p>0.05 ns | p<0.0001*** | 0.821 | 0.744 (+5pp) |
| MultiOrg | 16198 | 全 p<0.0001*** 但 effect 极小 | — | 0.862 | — |
| 结论 | 形态学只贡献 +5pp，主要靠 confidence | |

### Phase 2: VLM 语义确认（2026-07-09, 100 crops = 50TP+50FP）
| 指标 | VLM is_organoid | RF-DETR confidence | Combined (avg) |
|------|----------------|-------------------|-----------------|
| ROC-AUC | 0.713 | 0.871 | 0.834（VLM 拉低）|
| p-value | 0.000141*** | — | — |
| F1 | 0.69 | 0.80 | — |
| CROWN | 91 AGREE, 9 HALLUCINATION, 0 MISSED | — | — |
| 结论 | VLM 有信号但不如 RF-DETR，组合无增益 | |

### Phase 3: CTM 持续思考（2026-07-11, 16198 crops, 50 epochs）
| 指标 | CTM | RF-DETR | 差距 |
|------|-----|---------|------|
| Test ROC-AUC | 0.800 | 0.893 | -9.3pp |
| Best tick AUC | 0.801 (tick 17) | — | — |
| Certain tick AUC | 0.800 | — | — |
| Train acc ep2 | 1.0（过拟合）| — | — |
| 参数量 | 4.68M trainable | — | — |
| Tick 1-2 AUC | 0.20（反预测）| — | — |
| Certainty 峰值 | <0.37（未启动自适应）| — | — |
| Bug 修复 | 5个（数据泄露+同步化零值+3个评估bug）| — | — |
| 结论 | CTM 没超过 RF-DETR，过拟合严重，thinking 效果极弱 | |

### Phase 5: FL 聚合 primitive 分布（2026-07-10, 7 clients = Plate×Class）
| Client | n_det | TP | FP | k-NN AUC | p-value |
|--------|-------|----|----|----------|---------|
| Macros_Plate_15 | 2085 | 594 | 1491 | 0.700 | 1.05e-46 |
| Macros_Plate_23 | 2103 | 384 | 1719 | 0.714 | 9.59e-40 |
| Macros_Plate_4 | 3472 | 1143 | 2329 | 0.774 | 1.67e-152 |
| Normal_Plate_15 | 3651 | 1312 | 2339 | 0.685 | 1.67e-77 |
| Normal_Plate_31 | 2566 | 621 | 1945 | 0.787 | 1.81e-103 |
| Normal_Plate_37 | 1818 | 409 | 1409 | 0.703 | 3.79e-36 |
| Normal_Plate_4 | 503 | 166 | 337 | 0.809 | 8.92e-30 |
| **Global** | **16198** | **4629** | **11569** | **0.739** | **<0.001** |

Per-feature global AUC：confidence=0.893 > area=0.750 > perimeter=0.740 > circularity=0.686 > aspect_ratio=0.611 > solidity=0.272

### Phase 8: 小波频域原语（2026-07-11, 16198 crops, CPU 30min）
| 实验 | 特征 | PR-AUC | ROC-AUC | 维度 |
|------|------|--------|---------|------|
| W1 Haar 2L | 小波统计 | 0.457 | 0.681 | 28 |
| W2 db4 3L | 小波统计 | 0.446 | 0.687 | 40 |
| W3 haar+morph | 拼接 | 0.645 | 0.791 | 32 |
| RF-DETR baseline | confidence | 0.819 | 0.893 | 1 |
| 结论 | 小波单独无效（<0.50），morph 补救了（+18.8pp），但仍远低于 RF-DETR | |

### Phase 9: Slot Attention（2026-07-13, 16198 crops, 36 epochs, 冬生 3060）
| 指标 | Slot Attention | RF-DETR | 差距 |
|------|---------------|---------|------|
| Test ROC-AUC | 0.868 | 0.888 | -2.0pp |
| Test PR-AUC | 0.788 | 0.828 | -4.0pp |
| Val best AUC | 0.890 (ep21) | 0.883 | +0.7pp（val 追平）|
| Train acc ep36 | 0.878 | — | 过拟合轻 |
| 参数量 | 480K trainable | — | CTM 的 1/10 |
| 128s/epoch × 36ep | ~77min | — | — |
| Early stop | ep36 (patience=15) | — | — |
| 结论 | 最强替代方案！object-centric 分解有效（PR-AUC > 0.50）| |

### 全方案对比总表
| 方法 | Test ROC-AUC | Test PR-AUC | vs RF-DETR |
|------|-------------|-------------|------------|
| RF-DETR confidence | 0.888 | 0.828 | baseline |
| **Slot Attention (Phase 9)** | **0.868** | **0.788** | -2.0pp |
| CTM (Phase 3) | 0.800 | — | -8.8pp |
| W3 haar+morph (Phase 8) | 0.791 | 0.645 | -9.7pp |
| W1 haar 2L (Phase 8) | 0.681 | 0.457 | -20.7pp |
| Phase 5 k-NN (形态学) | 0.739 | — | — |
| DINOv2 CLS (之前) | — | 0.29 | -53.8pp |
| VLM (Phase 2, 100 crops) | 0.713 | — | -17.5pp |

### Slot + confidence 组合分析（2026-07-13, 云 VM, CPU 30s）
| 方法 | ROC-AUC | PR-AUC | vs RF-DETR (ROC) |
|------|---------|--------|-----------------|
| RF-DETR confidence | 0.888 | 0.828 | baseline |
| Slot (5-fold CV LR) | 0.842 | 0.739 | -4.5pp |
| Avg (slot+conf)/2 | 0.887 | 0.823 | -0.0pp（无效）|
| **LR(slot_prob, conf)** | **0.903** | **0.853** | **+1.5pp** ✅ |
| LR(slots_1024 + conf) | 0.883 | 0.810 | -0.5pp（过拟合）|
| **Best weighted (w=0.20)** | **0.903** | **0.854** | **+1.6pp** ✅ |

- Pearson r(slot, conf) = 0.626 → 中等相关，有互补性
- 最优权重 w_slot=0.20：80% confidence + 20% slot_prob
- 简单平均 50/50 无效（slot 太弱拉低），需要学习权重
- 1024 维全特征 LR 过拟合，2 维元学习最优

### 缺失/未做到位清单
| # | 缺失项 | 优先级 | 状态 |
|---|--------|--------|------|
| 1 | Slot + confidence 组合 AUC | 高 | ✅ 已做 |
| 2 | Phase 3 CTM 没用 mask crop（只用 bbox）| 高 | 待做 |
| 3 | Phase 3 CTM + confidence 组合 | 高 | 待做 |
| 4 | Phase 11 对比学习（InfoNCE）| 高 | 待做 |
| 5 | Phase 4 Diffusion 生成增强 | 中 | 完全跳过 |
| 6 | Phase 10 联邦 slot 聚合 | 中 | 待做 |
| 7 | Phase 5 用形态学不是原语 | 中 | 待改 |
| 8 | Phase 8 W4 鼠肝交叉验证 | 低 | 待做 |
| 9 | MultiOrg Phase 1 per-feature PR-AUC | 低 | 待做 |
| 10 | Phase 2 VLM 全量扩展 | 低 | 只 100 crops |
