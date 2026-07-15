# Intestinal 自蒸馏验证实验（模拟 RF-DETR + SAM2）

**日期**: 2026-07-16 06:37:38
**数据集**: Intestinal organoid (YOLO 4-class)
**方法**: 模拟 RF-DETR conf + SAM2 IoU，用 YOLO GT 作真值
**目的**: 验证自蒸馏方法在样本量充足时是否有效

## 1. 数据

| Split | N | True TP | True FP | Pseudo TP | Pseudo FP |
|---|---|---|---|---|---|
| Train | 500 | 351 | 149 | 355 | 145 |
| Val | 200 | 154 | 46 | 156 | 44 |

## 2. 结果

| 方法 | AUC |
|---|---|
| RF-DETR conf (zero-shot) | 1.0000 |
| Classifier alone | 0.9888 |
| Distilled (conf × classifier) | 1.0000 |
| **Improvement** | **+0.0000** |

## 3. 结论

❌ **自蒸馏无效**：AUC 下降 -0.0%。

## 4. 对比 MPM 实验

| 实验 | N train | N val | AUC zero | AUC distill | 改进 |
|---|---|---|---|---|---|
| MPM (10 pos) | 3131 | - | 0.9779 | 1.0000 | +0.0221 |
| Intestinal sim | 500 | 200 | 1.0000 | 1.0000 | +0.0000 |

## 5. 说明

- 此实验用**模拟 RF-DETR conf + 模拟 SAM2 IoU**（非真实检测）
- 5% SAM2 噪声（5% 概率 SAM2 给出错误 IoU）
- 特征：30维图像统计（3通道×(mean+std+8-bin hist)）
- 如果分类头能在这些特征上学习 TP vs FP，说明方法可行
