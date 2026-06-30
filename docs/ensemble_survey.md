# Ensemble Inference 调研报告 — 2026-06-30

## 1. SOTA 方法：Weighted Boxes Fusion (WBF)

**论文**：Solovyev et al., "Weighted boxes fusion: Ensembling boxes from different object detection models", Image and Vision Computing 2021
- arXiv: https://arxiv.org/abs/1910.13302
- 代码: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
- pip: `ensemble-boxes`

### WBF 算法
1. 所有模型的 box 放入列表 B，按 score 降序排列
2. 遍历 B 中每个 box，在 fused list F 中找 IoU > iou_thr 的匹配
   - 匹配上 → 加入对应 cluster L[pos]
   - 没匹配上 → 在 F 和 L 中新建 cluster
3. 每个 cluster 的 fused box = **score 加权平均坐标**（高置信度 box 贡献更大）
4. fused score = `sum(scores_in_cluster) * T / N`
   - T = cluster 中 box 数
   - N = 模型总数
5. **即使只有一个模型检测到的框也保留**（score × 1/N），不直接丢弃

### WBF vs NMS/Soft-NMS/NMW 实验数据（COCO, EffDetB7）

| Method | mAP(0.5:0.95) | mAP(0.5) | mAP(0.75) |
|---|---|---|---|
| 单模型 | 0.521 | 0.710 | 0.562 |
| NMS | 0.5233 | 0.7129 | 0.5656 |
| Soft-NMS | 0.5210 | 0.7092 | 0.5633 |
| NMW | 0.5250 | 0.7138 | 0.5691 |
| **WBF** | **0.5262** | **0.7144** | **0.5717** |

**WBF 全面最优**。10 个模型 ensemble 在 COCO 上达 56.4 mAP（当年 SOTA 第三）。

## 2. 我们当前方案 vs WBF

| 特性 | 当前 intersection | 当前 union | WBF（SOTA） |
|---|---|---|---|
| 两模型都检测到的 | ✅ 保留, score=avg | ✅ 保留, score=avg | ✅ 保留, score=avg×(T/N) |
| 只有一个模型检测到 | ❌ **丢弃** | ✅ 保留, score×0.7 | ✅ 保留, score×(1/N) |
| box 坐标融合 | avg | avg | **score 加权平均** |
| 后置 NMS | 不需要 | 需要(已加) | 不需要(cluster 机制) |

### 关键差异
1. **intersection 太激进**：丢弃所有"只有一个模型检测到"的框 → recall 必然下降
2. **union 的 score 惩罚(×0.7)是拍脑袋的**，WBF 的 ×(1/N=0.5) 有理论依据
3. **WBF 的坐标融合是 score 加权**，不是简单平均——高置信度的 box 贡献更大
4. **WBF 的 cluster 机制天然避免重复框**，不需要后置 NMS

## 3. 结论与建议

### 当前方案的问题
- intersection 策略本质是"两模型共识才保留"，这会损失 recall
- union 策略的 ×0.7 penalty 和后置 NMS 是 hack，不是最优解
- 两者都没有考虑 score 加权坐标融合

### 建议方案
**直接用 `ensemble-boxes` 库的 WBF 实现**，不要自己造轮子。

```python
from ensemble_boxes import weighted_boxes_fusion
# boxes_list = [[model1_boxes], [model2_boxes]]  # normalized [0,1]
# scores_list = [[model1_scores], [model2_scores]]
# labels_list = [[0,0,...], [0,0,...]]  # 单类=全0
boxes, scores, labels = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    weights=[1, 1],  # 两模型等权
    iou_thr=0.5
)
```

### 为什么 WBF 适合 MultiOrg
- **密集目标场景**：organoid 密集，NMS 会误删 → WBF 保留所有框只调 score
- **单类检测**：不需要考虑类别不匹配
- **两模型互补**：RF-DETR (NMS-free) + YOLO (anchor-based) 架构差异大 → ensemble 收益高
- **论文有据**：WBF 在 COCO 10 模型 ensemble 提升 ~1% mAP，2 模型可能提升 0.5-1%

### 实施步骤
1. `pip install ensemble-boxes`
2. 在 ensemble_inference.py 中加 `--strategy wbf`
3. WBF 要求 box 坐标归一化到 [0,1]，需要加 normalize/denormalize
4. 保留 intersection/union 作为对比 baseline

## 4. 其他发现

### NMW (Non-Maximum Weighted)
- 类似 WBF 但更简单：重叠框直接加权平均，不做 cluster
- 效果略好于 NMS 但不如 WBF

### 多模型 ensemble 的一般规律
- 模型多样性 > 模型数量：不同架构(RF-DETR vs YOLO)比同架构不同尺度(YOLOn vs YOLOs)收益更大
- 2-3 个模型是性价比最高的区间
- WBF 在 5+ 模型时优势最明显，2 模型也有提升

### 医学图像 ensemble
- Weighted Circle Fusion (WCF) 是 WBF 的变体，专门针对圆形目标
- organoid 不是完美圆形，WBF 比 WCF 更通用
