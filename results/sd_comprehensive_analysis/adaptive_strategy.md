# 自适应蒸馏策略

**日期**: 2026-07-16
**基于**: 4 个实验的 AUC 对比分析

## 策略表

| Zero-shot AUC | 策略 | 公式 | 理由 |
|---|---|---|---|
| < 0.70 | Classifier alone | `distilled = classifier_prob` | conf 失效，classifier 更可靠 |
| 0.70 - 0.85 | Distilled | `distilled = conf × classifier_prob` | 互补 |
| > 0.85 | Zero-shot | `distilled = conf` | classifier 帮助有限 |

## 四实验分布

- Simulation v2 (zero=0.6616): **Classifier alone**
- Real 200+100 (zero=0.7679): **Distilled**
- Real ResNet50 (zero=0.9156): **Zero-shot**
- MPM pilot (zero=0.9779): **Zero-shot**
