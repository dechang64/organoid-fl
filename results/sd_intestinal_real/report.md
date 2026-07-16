# Intestinal 真实跨域自蒸馏验证

**日期**: 2026-07-16 10:36:22
**RF-DETR**: 鼠肝 checkpoint → Intestinal 检测（跨域）
**GT**: Intestinal YOLO 4-class（真实标注）
**方法**: GT IoU 作独立信号（5% 噪声模拟 SAM2）+ 轻量分类头

## 1. 数据

| Split | Images | Samples | TP | FP |
|---|---|---|---|---|
| Train | 200 | 28925 | 3973 | 24952 |
| Val | 100 | 11578 | 2501 | 9077 |

## 2. Zero-shot 跨域失效检查

⚠ **跨域部分有效**: Zero-shot AUC=0.7679
   RF-DETR conf 仍有一定区分能力（可能 checkpoint 迁移性好）。

## 3. 自蒸馏结果

| 方法 | AUC |
|---|---|
| Zero-shot (RF-DETR conf) | 0.7679 |
| Classifier alone | 0.8202 |
| Distilled (conf × classifier) | 0.8589 |
| **Improvement** | **+0.0910** |

⚠ **自蒸馏微弱有效**：AUC 提升 9.1%。

## 4. 说明

- RF-DETR 真实跨域检测（鼠肝训练 → Intestinal 检测）
- GT IoU 作为独立信号（5% 噪声模拟 SAM2 错误）
- 特征：30 维图像统计
- 如果 zero-shot AUC 高（>0.8），说明 RF-DETR 跨域迁移性好，自蒸馏提升空间小
- 如果 zero-shot AUC 低（<0.7），说明跨域失效，自蒸馏应能提升
