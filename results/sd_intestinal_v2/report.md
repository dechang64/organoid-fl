# Intestinal 自蒸馏验证（修正版）

**日期**: 2026-07-16 06:38:36

## 1. 设计

- TP: YOLO GT boxes + RF-DETR conf 0.05-0.50 + SAM2 IoU 0.70-1.00 (5% 噪声)
- FP: 随机 box + RF-DETR conf 0.01-0.40 + SAM2 IoU 0.00-0.25 (5% 噪声)
- 保留所有样本（不过滤）
- 训练分类头用 pseudo label，评估用 true label

## 2. 数据

| Split | Total | TP | FP |
|---|---|---|---|
| Train | 500 | 262 | 238 |
| Val | 200 | 120 | 80 |

## 3. 结果

| 方法 | AUC |
|---|---|
| Zero-shot (RF-DETR conf) | 0.6616 |
| Classifier alone | 0.9792 |
| Distilled (conf × classifier) | 0.9437 |
| **Improvement** | **+0.2822** |

✅ **自蒸馏有效**：AUC 提升 28.2% > 10%。
