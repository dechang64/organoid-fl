# MultiOrg 检测 SOTA 全面调研 — 2026-06-30

## 1. MultiOrg 数据集 SOTA (mAP@0.5, t1_b 标注)

| 模型 | mAP50 | 来源 |
|---|---|---|
| Faster R-CNN | 68.36% | MultiOrg 论文 (Bukas et al., NeurIPS 2024) |
| SSD | 73.88% | MultiOrg 论文 SOTA |
| YOLOv3 | 70.25% | MultiOrg 论文 |
| RTMDet | 63.23% | MultiOrg 论文 |
| **RF-DETR small+640 (我们)** | **77.76%** | **超 SOTA +3.88pp** |
| RF-DETR SAHI+Soft-NMS (我们) | 77.15% | 之前最优 |
| YOLOv12s+freebies (我们) | 62.33% | 训练配置有 bug |
| WBF ensemble (我们) | 63.22% | ensemble 负优化 |
| Union ensemble (我们) | 69.96% | ensemble 负优化 |

**结论：我们已是 MultiOrg 上的 SOTA，距 80% 门槛差 2.24pp。无后续工作超过我们。**

## 2. 其他 organoid 数据集 SOTA

| 方法 | mAP50 | 数据集 | 数据量 | 关键技术 |
|---|---|---|---|---|
| Deliod | 87.5% | Intestinal organoid | N/A | YOLOv8s + DRBNCSPELAN |
| Orga-Dete | 81.4% | Lung organoid (非 MultiOrg) | 4008 张 | YOLOv11n + BiFPN + MPCA + EMASlideLoss |
| OrgLine | N/A (PQ 指标) | 多数据集聚合 | N/A | detector-guided SAM2 |

## 3. 80% 门槛的可行性分析

### 有利因素
- **RF-DETR 已到 77.76%**，距 80% 仅 2.24pp
- **Orga-Dete 在 lung organoid 达 81.4%**，证明 organoid 检测 80%+ 可行
- **MultiOrg 训练集仅 356 张**，Orga-Dete 用 4008 张 → 数据量是关键瓶颈

### 不利因素
- MultiOrg 数据量小（356 train），难以单纯靠训练策略突破
- 论文 SOTA 仅 73.88%，我们已超 +3.88pp，继续提升难度递增
- ensemble 方向已证伪（YOLOv12s 太弱，无法互补）

## 4. 可能突破 80% 的方向（按可行性排序）

### 方向 A：修复 YOLOv12s 训练配置（预计 77-80%）
- 当前 YOLOv12s 仅 62.33%，远低于预期
- TOOLS.md 记录 args.yaml data 指向 v3_512（训练配置 bug）
- 修好后重训，如果到 77%+，ensemble 才有意义
- **成本：低（修配置+重训 5h）**

### 方向 B：应用 Orga-Dete 的技术到 RF-DETR（预计 +1-3pp）
- BiFPN：多尺度特征融合，对小目标有效
- MPCA：微 organoid 特征增强
- EMASlideLoss：解决类别不平衡
- 在 YOLOv11n 上 +3.5pp，RF-DETR 上可能也有提升
- **成本：中（需要改 RF-DETR 架构，但 RF-DETR 不像 YOLO 那么容易改）**

### 方向 C：用 OrgLine Benchmark 数据补充训练（预计 +2-5pp）
- OrgLine 聚合了多个 organoid 数据集，修正了标注
- 用 OrgLine 的 Object Detection 包预训练，再在 MultiOrg 上 fine-tune
- 数据量从 356 → 4000+，可能突破数据瓶颈
- **成本：中（下载 1.1GB + 预训练 + fine-tune）**

### 方向 D：换更大模型 yolo12m（预计 +1-2pp）
- yolo12m 20.2M 参数（3.2× yolo12s）
- TOOLS.md 记录已在调研中，3060 12GB 可跑 batch=2
- 但 yolo12s 在 MultiOrg 上只有 62.33%，换更大模型未必解决根本问题
- **成本：低（已有 checkpoint），但收益不确定**

### 方向 E：纯推理优化（预计 +0.5-1pp）
- SAHI + Soft-NMS + score_filter 已优化到 76.47%
- 进一步调参：overlap 0.2-0.4, conf 0.2-0.3, sf 0.35-0.4
- **成本：低，但天花板低**

## 5. ensemble 失败的根因总结

- YOLOv12s (62.33%) 是 RF-DETR (77.76%) 的**严格劣势模型**
- YOLO 没有任何独有 TP（ensemble TP=4745 < RF-DETR TP=4767）
- WBF 的 T/N 机制把 RF-DETR 的高分 TP score 拉低了
- **ensemble 需要两个精度接近的模型才能受益**——这是文献共识

## 6. 参考文献

1. Bukas et al., "MultiOrg: A Multi-rater Organoid-detection Dataset", NeurIPS 2024
2. Solovyev et al., "Weighted boxes fusion", Image and Vision Computing 2021
3. Huang et al., "Orga-Dete: An Improved Lightweight Deep Learning Model for Lung Organoid Detection", Applied Sciences 2025
4. Deng et al., "OrgLine: A versatile pipeline for organoid morphometry", Cell Reports Methods 2026
5. Deliod: "a lightweight detection model for intestinal organoids based on YOLOv8", 2024
