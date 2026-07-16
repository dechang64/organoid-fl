# 自蒸馏方法综合分析

**日期**: 2026-07-16
**基于**: 4 个实验（simulation + 真实跨域 + ResNet50 + MPM pilot）

## 1. 实验列表

| # | 实验 | 特征 | N_val |
|---|---|---|---|
| 1 | Simulation v2 (Intestinal 500+200) | Image statistics 30-dim | 200 |
| 2 | Real cross-domain (Intestinal 200+100) | Image statistics 30-dim | 11578 |
| 3 | Real cross-domain ResNet50 (Intestinal 200+84) | ResNet50 2048-dim | 84 |
| 4 | MPM pilot (34 patches) | RF-DETR backbone 256-dim | 3131 |

## 2. AUC 对比

| 实验 | Zero-shot | Classifier alone | Distilled (conf×cls) | Best |
|---|---|---|---|---|
| Simulation v2 (Intestinal 500+200) | 0.6616 | 0.9792 | 0.9437 | **Class (0.9792)** |
| Real cross-domain (Intestinal 200+100) | 0.7679 | 0.8202 | 0.8589 | **Distill (0.8589)** |
| Real cross-domain ResNet50 (Intestinal 200+84) | 0.9156 | 0.9938 | 0.9281 | **Class (0.9938)** |
| MPM pilot (34 patches) | 0.9779 | nan | 1.0000 | **Distill (1.0000)** |

## 3. 关键观察

### 3.1 Classifier alone vs Distilled

| 实验 | Classifier | Distilled | 差距 |
|---|---|---|---|
| Simulation v2 (Intestinal 500+200) | 0.9792 | 0.9437 | distilled < classifier by 3.54pp |
| Real cross-domain (Intestinal 200+100) | 0.8202 | 0.8589 | distilled > classifier by 3.87pp |
| Real cross-domain ResNet50 (Intestinal 200+84) | 0.9938 | 0.9281 | distilled < classifier by 6.56pp |

### 3.2 蒸馏 vs Zero-shot

| 实验 | Zero-shot | Distilled | Improvement |
|---|---|---|---|
| Simulation v2 (Intestinal 500+200) | 0.6616 | 0.9437 | +28.22pp |
| Real cross-domain (Intestinal 200+100) | 0.7679 | 0.8589 | +9.10pp |
| Real cross-domain ResNet50 (Intestinal 200+84) | 0.9156 | 0.9281 | +1.25pp |
| MPM pilot (34 patches) | 0.9779 | 1.0000 | +2.21pp |

## 4. 结论与建议


**A. 当 RF-DETR conf 跨域失效（zero-shot AUC < 0.8）时**：
- Classifier alone > Distilled (conf × classifier)
- 因为 conf 是噪声，乘 classifier 反而拉低高置信度样本
- 例：ResNet50 跨域 200+84，classifier 0.9938 vs distilled 0.9281

**B. 当 RF-DETR conf 部分有效（zero-shot AUC 0.75-0.85）时**：
- Distilled > Classifier alone > Zero-shot
- conf 和 classifier 互补
- 例：真实跨域 200+100，distilled 0.8589 > classifier 0.8202 > zero 0.7679

**C. 当 RF-DETR conf 完全有效（zero-shot AUC > 0.9）时**：
- Distilled ≈ Zero-shot
- classifier 帮助有限
- 例：MPM pilot zero 0.9779, distilled 1.0 (ceiling)

**自适应蒸馏策略**：
```python
if zero_shot_auc < 0.70:
    distilled = classifier_prob   # conf 失效，用 classifier alone
elif zero_shot_auc < 0.85:
    distilled = conf * classifier_prob  # 互补
else:
    distilled = conf  # classifier 帮助有限，保持 zero-shot
```

这个策略可以根据每个 patch 的 zero-shot AUC 动态选择公式。
