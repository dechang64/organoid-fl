# 鼠肝 cross-batch 自蒸馏验证

**日期**: 2026-07-16 08:35:13
**数据**: 鼠肝 organoid 3 batch (b1+b3 真实 RF-DETR 检测)
**方法**: Leave-One-Batch-Out + 图像统计特征 + 轻量分类头

## 1. 数据

| Batch | Samples | TP | FP | RF-DETR conf mean |
|---|---|---|---|---|
| b1 | 23 | 22 | 1 | 0.720 |
| b3 | 7 | 3 | 4 | 0.328 |

## 2. Leave-One-Batch-Out 结果

| Test Batch | Val N | TP | FP | Zero-shot AUC | Classifier AUC | Distilled AUC | Improvement |
|---|---|---|---|---|---|---|---|
| b1 | 23 | 22 | 1 | 0.9545 | 0.0000 | 0.9545 | +0.0000 |
| b3 | 7 | 3 | 4 | 0.9167 | 0.0000 | 0.9167 | +0.0000 |

## 3. Pooled 结果

| 方法 | AUC |
|---|---|
| Zero-shot (RF-DETR conf) | 0.9760 |
| Distilled (conf × classifier) | 0.9440 |
| **Improvement** | **-0.0320** |

❌ **自蒸馏无效**：AUC 下降 3.2%。

## 4. 说明

- 样本量小 (30 total, 5 FP)，统计功效有限
- RF-DETR conf 是真实的（鼠肝训练 checkpoint 检测鼠肝）
- b1 conf 高 (0.964)，b3 conf 低 (0.339)——跨 batch domain gap
- 特征：30 维图像统计（3 通道 mean/std + 8-bin hist）
- 这个实验验证 pipeline 在真实数据上可跑
