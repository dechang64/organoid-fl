# MultiOrg 突破 SOTA 方案 v2.0

## 更新日期：2026-06-21

## 目标

超越 MultiOrg SOTA (SSD 68.1% mAP@0.5) → 达到 80%+

---

## 一、v1.0 → v2.0 新增调研发现

### v1.0 已覆盖（2026-06-19）
- Deliod (YOLOv8s 改, 87.5%), Orga-Dete (YOLOv11n 改, 81.4%)
- YOLOv12 代差红利 + freebies (copy_paste, mixup, cos_lr)
- SAHI 推理增强 + WBF 融合 + TTA
- 标签噪声：t1 标签 + co-teaching + label smoothing
- 专利空白：FL+Organoid, YOLO+SAHI+Organoid

### v2.0 新增（2026-06-21 搜索）

#### 1. RF-DETR (Roboflow, 2025-03) — ⭐ 重大新 SOTA
- **第一个 real-time 60+ mAP on COCO** 的检测器
- Transformer 架构，**NMS-free**（对密集 organoid 场景是巨大优势——NMS 容易误删密集目标）
- RF-DETR-nano: 48.0 AP on COCO（比 D-FINE nano 高 5.3 AP）
- RF-DETR-small: 比 YOLO11-x 高 1.8 AP，速度快 7.77ms
- Apache 2.0 开源，支持 custom dataset fine-tune
- **已有医学图像应用**：urine sediment 检测 (Nature Sci Rep 2025)，小目标改进显著
- **RF-DETR vs YOLOv12 对比** (arXiv 2504.13099)：单类检测 RF-DETR mAP50=94.6% > YOLOv12X 92.5%

**对 MultiOrg 的意义**：
- MultiOrg 是单类检测，RF-DETR 在单类场景优势最明显
- NMS-free 对密集小 organoid 场景可能是 +3-5pp 的关键
- 6K×5.7K 大图切片后密集目标，NMS 误删是 SSD 68% 的潜在瓶颈之一

#### 2. OrgLine (Cell Reports Methods, 2026) — ⭐ 新 organoid SOTA pipeline
- Deng et al. 发布的 bright-field organoid 分析 pipeline
- **预训练在 cell images → fine-tune 到 organoid 数据集**
- 组合检测 + prompt-guided 分割（SAM 类）
- 三个新标注的 organoid 数据集（尚未公开评估）

**对 MultiOrg 的意义**：
- 提供了 cell images 预训练权重作为 transfer learning 起点
- 预训练 + fine-tune 范式已被验证在 organoid 上有效
- 我们可以先用 COCO/cell 预训练，再 fine-tune MultiOrg

#### 3. YOLO26 (Ultralytics, 2025-2026)
- **dual-head 架构**（one-to-one + one-to-many）
- edge-first 设计
- **我们其实已经在用**：Ultralytics 8.4.67 把 yolo12n 自动解析为 yolo26n
- YOLO26 是 YOLO 系列最新版

#### 4. NeurIPS 2025: Early-Learning Distillation (ELD) for Noisy Labels
- **专门为 object detection + noisy labels 设计**
- 蒸馏方法，同时处理分类和定位噪声
- 比 co-teaching 更现代，理论保障更强

**对 MultiOrg 的意义**：
- MultiOrg 3 套标注（t0, t1_A, t1_B）是天然的噪声标签场景
- ELD 可以直接应用，比我们 v1.0 提的 co-teaching 更强

#### 5. CLOD: Confident Learning for Object Detection (arXiv v3, 2024)
- 专门为检测任务做标签噪声清洗
- 可识别 MultiOrg 中的错误标注（defocused organoid 误标、漏标）
- 在清洗后的标签上训练 = 直接提升

#### 6. TransOrga-plus (BMC Biology, 2025-10)
- Knowledge-driven organoid 分割+追踪框架
- 多模态（视觉+频域），bright-field 显微镜
- 做的是分割不是检测，但多模态思路可借鉴

#### 7. RT-DETR 小目标改进 (MDPI 2025)
- Strengthening Small Object Detection in Adapted RT-DETR
- 小目标 mAP 从 0.389 → 0.513（+12.4pp）
- 证明 DETR 系列在小目标上有改进空间

---

## 二、v2.0 综合突破路径（更新）

### 阶段 1：正确复现 + 模型代差（目标 72-75%）

**v1.0 方案（保留）**：
1. 单类 organoid，512 patch，stride=256，丢弃边界 bbox
2. t1_A 标签（噪声更低）
3. YOLOv12s/512/400ep + freebies (copy_paste=1.0, mixup=0.15, cos_lr)
4. SAHI 全图滑动窗口推理

**v2.0 增补**：
- **先用 cell images 预训练**（借鉴 OrgLine 思路），再 fine-tune MultiOrg
- 或者直接用 COCO 预训练 + 更长 warmup

### 阶段 2：RF-DETR 替代 YOLO（目标 76-80%）⭐ 新增

**核心假设**：RF-DETR 的 NMS-free 架构在密集 organoid 场景比 YOLO 有结构性优势。

**方案**：
```python
# RF-DETR fine-tune on MultiOrg
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain=True)  # COCO pretrained
model.train(
    dataset_dir="multiorg_yolo_format/",
    epochs=400,
    batch_size=8,
    grad_accumulate=4,  # 3060 12GB 可能需要梯度累积
    img_size=512,
)
```

**对比实验**：
| 模型 | 架构 | NMS | 密集场景优势 | 预期 mAP50 |
|------|------|-----|-------------|-----------|
| YOLOv12s | CNN + attention | 需要 | 中 | 73-76% |
| RF-DETR-base | Transformer | **不需要** | **高** | 76-80% |

**风险**：
- RF-DETR 在 512 分辨率上可能不如 YOLO（DETR 系列对低分辨率敏感）
- 12GB 显存可能不够（RF-DETR-base 比 YOLOv12s 大）
- 解法：用 RF-DETR-nano，或降 batch + 梯度累积

### 阶段 3：标签噪声处理（目标 78-82%）⭐ 升级

**v1.0 方案**：t1 标签 + co-teaching + label smoothing

**v2.0 升级**：
1. **CLOD 清洗**：先清洗 t0 标签中的明确错误标注
2. **ELD 蒸馏**：NeurIPS 2025 方法，用 t1_A+t1_B 共识标签做 teacher，t0 做 student
3. **多标注者共识**：t1_A ∩ t1_B 交集 = high-confidence label，差集 = soft label

**流水线**：
```
t0 (noisy) → CLOD 清洗 → clean_t0
t1_A + t1_B → 共识 → high_conf_labels
clean_t0 + high_conf_labels → ELD 蒸馏训练 → final model
```

### 阶段 4：推理优化（目标 80-83%）

**v1.0 方案（保留）**：SAHI + WBF + TTA

**v2.0 增补**：
- RF-DETR 不需要 NMS，但 SAHI 切片后合并仍需 postprocess
- 对 RF-DETR 用 WBF 替代 NMS 合并切片
- 双尺度推理：512（小 organoid）+ 1024（大 organoid）

---

## 三、技术路线对比

| 路线 | 核心方法 | 预期 mAP50 | 风险 | 工程量 |
|------|---------|-----------|------|-------|
| A: v1.0 路线 | YOLOv12s + freebies + SAHI | 75-78% | 低 | 低 |
| B: RF-DETR 路线 | RF-DETR + SAHI + WBF | 76-80% | 中（新框架） | 中 |
| C: 标签噪声路线 | CLOD + ELD + 共识标签 | 78-82% | 中（新方法） | 中 |
| D: 全组合 | RF-DETR + CLOD/ELD + SAHI | 80-83% | 高（多新组件） | 高 |

**推荐：B + C 并行，取最优组合。**

---

## 四、新数据集机会

### 已知数据集
| 数据集 | 规模 | 类别 | SOTA | 适用性 |
|--------|------|------|------|-------|
| MultiOrg | 411 张 6K | 1 类 | SSD 68.1% | 主目标 |
| Intestinal | 840 张 1280 | 4 类 | Deliod 87.5% | 已超 SOTA |
| OrgaQuant | 14K 张 | 肠类器官 | - | 大数据量 |
| OrganoID | 通用 | 通用 | 平台 | 工具 |
| OrgLine 3 数据集 | 未公开 | - | - | 待跟踪 |

### 新机会
1. **OrgLine 三个新 organoid 数据集** — Deng et al. 2026 发布，需跟踪是否公开
2. **Brain Organoid 1400 张** (Sci Data 2024) — cross-laboratory brain organoid，可用 OrganoSeg 分析
3. **Digitalized Organoids 3D** (Nat Methods 2025) — 3D organoid 拓扑，新方向

---

## 五、论文/专利布局更新

### 论文方向（v2.0）
1. **"RF-DETR for Multi-Rater Organoid Detection: NMS-Free Architecture for Dense Small Object"**
   - 首个 RF-DETR + MultiOrg
   - NMS-free 对密集场景的结构性优势
   - 多标注者共识标签 + ELD

2. **"Label Noise-Aware Detection with Confident Learning: From MultiOrg to Production"**
   - CLOD + ELD 在 MultiOrg 上的系统应用
   - 多标注者不确定性建模

3. **"Federated Learning for Multi-Center Organoid Detection"**（保留）
   - FL+Organoid 仍是完全空白

### 专利方向（v2.0）
1. **RF-DETR + Organoid 推理方法** — 无专利（RF-DETR 2025 才发布）
2. **CLOD + Multi-rater Organoid 标签清洗** — 无专利
3. **FL+Organoid 检测系统** — 完全空白（保留）

---

## 六、行动建议

### 立即可做（3060 12GB）
1. **下载 RF-DETR nano**，在 MultiOrg 512 patch 上 fine-tune
2. **CLOD 清洗 t0 标签**，生成 clean_t0
3. **t1_A + t1_B 共识标签**，生成 high_conf_labels
4. **对比实验**：YOLOv12s vs RF-DETR-nano，相同数据/推理设置

### 需要 GPU 资源
- RF-DETR-base fine-tune（可能需要 >12GB）
- ELD 蒸馏（需要训练 teacher + student 两个模型）

### 需要跟踪
- OrgLine 数据集是否公开（2026 发表，可能 2026 下半年释放）
- RF-DETR 医学图像领域论文（2026 预计会有更多）

---

## 七、版本历史

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-06-19 | v1.0 | 初始方案：六条路径，三阶段到 80% |
| 2026-06-21 | v2.0 | 新增 RF-DETR/OrgLine/ELD/CLOD，升级四阶段到 83% |
