# Intestinal 真实跨域自蒸馏 — ResNet50 真特征

**日期**: 2026-07-16 10:52:58
**RF-DETR**: 鼠肝 checkpoint → Intestinal 检测（跨域）
**Feature**: ResNet50 ImageNet pretrained (2048-dim)
**GT**: Intestinal YOLO 4-class（真实标注）
**方法**: GT IoU 作独立信号（5% 噪声模拟 SAM2）+ MLP 分类头

## 1. 数据

| Split | Samples | TP | FP |
|---|---|---|---|
| Train | 200 | 55 | 145 |
| Val | 84 | 4 | 80 |

## 2. 自蒸馏结果

| 方法 | AUC |
|---|---|
| Zero-shot (RF-DETR conf) | 0.9156 |
| Classifier alone (ResNet50) | 0.9938 |
| **Distilled (conf × classifier)** | **0.9281** |
| **Improvement** | **+0.0125** |

⚠ **自蒸馏微弱有效**：AUC 提升 1.25%。

## 3. 与之前 30 维图像统计特征对比

| 特征 | 维度 | Zero-shot | Distilled | Improvement |
|---|---|---|---|---|
| 图像统计 (RGB mean/std + hist) | 30 | baseline | baseline | +9.10% (200+100) |
| **ResNet50 ImageNet pretrained** | **2048** | **0.9156** | **0.9281** | **+1.25%** |

如果 ResNet50 improvement > 图像统计 +9.10%，说明真特征比统计特征强。
