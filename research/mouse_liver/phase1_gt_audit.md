# Phase 1 修订：GT mask 可用性分析

> 2026-07-06 审计发现

## GT 标注可用性

| 批次 | bbox 标注 | 红色折线标注图 | 标注图目录 |
|------|-----------|---------------|-----------|
| B1 (10张, 2592×1944) | ✅ annotations.json | ❌ 无 | — |
| B2 (10张, 2592×1944) | ✅ annotations.json | ❌ 无 | — |
| B3 (20张, 4000×3000) | ✅ annotations.json + source_annotated | ✅ 20张 | datasets/mouse_liver_annotated_20260702/ |

## mouse_liver_new 的 10 张标注图

- 4000×3000, 有红色折线
- 不是 B1 的 (坐标不匹配)
- 不是 B2 的 (坐标不匹配)
- 可能是未使用的额外数据，或 B1/B2 原始高分辨率版本

## M14 实验回顾

- F1=93.9% 是 **bbox 级 F1**（IoU>0.5 bbox 匹配），不是 mask 级
- GT 来自 YOLO label 的 bbox 填充，不是真实轮廓
- `load_gt_mask_from_annot` 函数存在但 M14 没用到

## Phase 1 修订方案

### 可做的评估
1. **bbox 级 F1**（所有 batch 都有 bbox GT）
2. **mask 级 F1**（仅 B3，有红色折线→真实轮廓 mask）
3. **形态学特征分布**（所有 batch，从 SAM2 输出 mask 计算）

### 修订实验
| 实验 | 训练 | 测试 | 评估指标 | 说明 |
|------|------|------|----------|------|
| P1-A | B1 8张 RF-DETR | B1 2张 | bbox F1 | 复现 M14 (93.9%) |
| P1-B | B1 8张 RF-DETR | B2 10张 | bbox F1 | 同分辨率跨域 |
| P1-C | B1 8张 RF-DETR | B3 20张 | bbox F1 + **mask F1** | 跨分辨率，B3 有红色折线 |
| P1-D | B1+B2 16张 | B3 20张 | bbox F1 + **mask F1** | 多源训练 |
| P1-E | B1+B2+B3 31张 | val 9张 | bbox F1 | 集中式上界 |

### 关键变化
- mask 级 F1 只在 B3 上评估（唯一有红色折线 GT 的）
- P1-C/D 是核心：bbox F1 + mask F1 + 形态学特征分布
- 形态学特征分布对比：B1/B2 的 SAM2 mask vs B3 的 SAM2 mask
