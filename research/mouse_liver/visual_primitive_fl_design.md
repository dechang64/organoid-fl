# 鼠肝视觉原语 FL 实验方案

> 创建：2026-07-06
> 前置：bbox 范式实验全部完成（E4-E11），确认 bbox 对不规则类器官到天花板
> 技术栈：organoid-fl sam2_segment.py + ewa-fed primitives.py + PCB-Defect-FL hallucination_defense.py

---

## 一、实验动机

### bbox 范式的三个结构性瓶颈

| 瓶颈 | 证据 | 根因 |
|------|------|------|
| 小目标分类失败 | B3@1280 R=0.61 但 P=0.004 | bbox 里 organoid 只占小部分，背景噪声稀释分类信号 |
| FL 聚合权重≠聚合知识 | E6(soft)=0.506 ≈ B2独立=0.520 | round 级参数平均，B3 偏掉的参数拖低全局 |
| 不规则形状无法表达 | bbox F1=80% vs mask F1=93.9% (M14) | bbox 外接矩形丢失轮廓信息 |

### 视觉原语框架的三个对应解法

| 解法 | 机制 | 为什么有效 |
|------|------|-----------|
| bbox→SAM2 mask | bbox 只做 prompt，mask 做精确分割 | mask 覆盖 organoid 真实形状，排除背景 |
| 形态学特征做分类 | circularity/solidity/eccentricity | 比值不受绝对像素尺度影响 — B3 200px 和 B2 800px organoid circularity 一样 |
| FL 聚合 primitive 分布 | 不聚合权重，聚合形态统计 | "数据不动，知识动" — 各中心上传形态分布，不上传像素/权重 |

---

## 二、三层架构设计

### Layer 1: Perception（感知层）— 各中心本地运行

```
图像 → RF-DETR/YOLO 检测 → bbox(prompt) → SAM2 分割 → mask → 形态学特征
```

**输入**：原始图像（各中心本地，不出域）
**输出**：visual primitives 列表

每个 primitive 包含：
- `ref`: "organoid"（语义标签）
- `type`: PATH（轮廓点序列）
- `coords`: 归一化到 0-999 的轮廓坐标
- `entropy`: 检测置信度的负对数（-log(conf)）
- `aux`: 形态学特征字典
  - `area`: 面积（归一化）
  - `circularity`: 4π·area/perimeter²（1.0=完美圆）
  - `solidity`: area/convex_hull_area（1.0=凸形）
  - `eccentricity`: 椭圆离心率（0=圆，1=线）
  - `aspect_ratio`: 长轴/短轴

**关键设计**：
- bbox 只做 SAM2 的 prompt，不做最终输出
- 形态学特征是比值/无量纲量，跨分辨率可比
- 鼠肝已有红色折线 GT → 可做 mask 级评估（非 bbox mAP）

**复用已有代码**：
- `sam2_segment.py`: RF-DETR → SAM2 → 形态学特征（✅ 已验证 F1=93.9%）
- `primitives.py`: PrimitiveCodec.encode_detections + auxiliary 字段（✅ 已实现）

### Layer 2: Reasoning（推理层）— 各中心本地运行

```
visual primitives + 全图上下文 → VLM 推理 → 分类标签 + 置信度 + 推理链
```

**输入**：Layer 1 的 primitives + 原图（本地）
**输出**：过滤后的 primitives（去掉非 organoid 的 FP）

**VLM Prompt 模板**（设计）：
```
图像中有以下检测候选：
1. 位置(120,340)-(450,670), 圆度=0.87, 实度=0.92, 离心率=0.34
2. 位置(200,100)-(250,150), 圆度=0.45, 实度=0.61, 离心率=0.78
...

请判断每个候选是否为类器官（organoid），并给出理由：
- 形态学特征是否符合类器官（圆形/椭圆形，边界清晰，内部均匀）
- 位置是否合理（培养板内，非边缘碎片）

输出 JSON: [{"id": 1, "is_organoid": true, "confidence": 0.9, "reason": "..."}, ...]
```

**关键设计**：
- VLM 只在 Layer 1 输出上做推理，不看原图像素（隐私）
- 推理可解释（CoT）：不只是 yes/no，还有为什么
- 幻觉防御：5 层校验（PCB-Defect-FL hallucination_defense.py）

**注意**：Layer 2 是可选的。Phase 1 先验证 Layer 1+3（不 VLM），Phase 2 加 VLM。

### Layer 3: Collaboration（协作层）— FL 服务器

```
各中心 primitive 分布 → EWA 熵加权聚合 → 全局 primitive 分布 → 下发
```

**输入**：各中心的 PrimitiveBatch（JSON，不含像素）
**输出**：全局 primitive 分布（形态学特征统计）

**聚合内容**（不是权重！）：
1. **形态学特征分布**：各中心的 circularity/solidity/eccentricity 直方图
2. **检测置信度分布**：各中心的 conf 直方图
3. **primitive 数量统计**：各中心每图平均 organoid 数

**聚合策略**：
- EWA 熵加权：低熵（高置信）的中心权重高
- Conformity 从众检测：少数中心的异常 primitive 被压制

**下发内容**：
- 全局形态学特征分布（直方图）
- 各中心的置信度校准参数
- 不下发模型权重（Layer 1 检测器各中心独立训练/保持）

**复用已有代码**：
- `aggregator.py`: EWA 熵加权聚合（✅ 已实现）
- `conformity.py`: 从众检测（✅ 已实现）
- `primitives.py`: PrimitiveBatch JSON 传输（✅ 已实现）

---

## 三、实验设计

### Phase 1: Perception 层验证（无 FL，无 VLM）

**目标**：验证 bbox→SAM2→形态学特征 pipeline 在三批数据上的效果

| 实验 | 训练 | 测试 | 评估指标 | 预期 |
|------|------|------|----------|------|
| P1-A | B1 8张 RF-DETR | B1 2张 | mask F1 | ~93.9% (复现 M14) |
| P1-B | B1 8张 RF-DETR | B2 10张 | mask F1 | ~90% (同分辨率跨域) |
| P1-C | B1 8张 RF-DETR | B3 20张 | mask F1 | >50%? (跨分辨率) |
| P1-D | B1+B2 16张 RF-DETR | B3 20张 | mask F1 | >60%? (多源训练) |
| P1-E | B1+B2+B3 31张 RF-DETR | 统一 val 9张 | mask F1 | 集中式上界 |

**评估方式**：
- mask 级 F1（IoU>0.5 判定 TP）
- 形态学特征对比：B2 vs B3 的 circularity/solidity 分布
- 可视化：SAM2 mask 叠加 + 红线 GT 对比

**关键验证**：P1-C 的 mask F1 是否显著高于 bbox F1（7.1%）
- 如果 mask F1 > 30%：bbox→mask 范式转换有效，继续 Phase 2
- 如果 mask F1 < 10%：SAM2 在 B3 上也失败，需要 VLM 层

**GT mask 来源**：
- B1/B2: 红色折线标注图 → OpenCV 提取轮廓 → fillPoly 生成 mask
- B3: 红色折线标注图 → 同上

### Phase 2: FL Primitive 聚合验证（无 VLM）

**目标**：验证 FL 聚合 primitive 分布是否优于聚合权重（E6）

| 实验 | 聚合对象 | 通信内容 | 评估 | 对比 |
|------|----------|----------|------|------|
| P2-A | 检测器权重 (FedAvg) | state_dict | bbox mAP50 | = E4 (baseline) |
| P2-B | 检测器权重 (EWA soft) | state_dict | bbox mAP50 | = E6 (baseline) |
| P2-C | primitive 分布 | PrimitiveBatch JSON | mask F1 | vs P2-A/B |
| P2-D | primitive 分布 + 形态学 | PrimitiveBatch + 直方图 | mask F1 | vs P2-C |

**P2-C/D 的 FL 流程**：
1. 各中心本地跑 RF-DETR + SAM2（Layer 1）
2. 各中心提取 primitive 分布（形态学直方图）
3. 上传 PrimitiveBatch JSON 到服务器（不含像素/权重）
4. 服务器 EWA 熵加权聚合 → 全局分布
5. 下发全局分布 → 各中心用全局分布校准本地检测

**校准方式**：
- 全局形态学分布作为先验 → 调整本地检测 conf 阈值
- 全局 circularity 直方图 → 过滤本地异常 primitive（circularity 在全局分布外的判为 FP）

**关键验证**：P2-C/D 的 mask F1 是否优于 P2-A/B 的 bbox mAP50
- 如果 P2-C > P2-A：primitive 聚合优于权重聚合
- 如果 P2-C ≈ P2-A：聚合方式不是瓶颈，检测器本身才是

### Phase 3: VLM 推理层验证（可选）

**目标**：验证 VLM 推理是否能进一步提升分类精度

| 实验 | Layer 1 | Layer 2 | Layer 3 | 评估 |
|------|---------|---------|---------|------|
| P3-A | RF-DETR+SAM2 | 无 | 无 | mask F1 (baseline) |
| P3-B | RF-DETR+SAM2 | VLM 过滤 | 无 | mask F1 (VLM 增益) |
| P3-C | RF-DETR+SAM2 | VLM 过滤 | primitive FL | mask F1 (完整框架) |

**VLM 调用**：
- GLM-4V / Qwen-VL
- 输入：形态学特征文本 + mask 轮廓图（不含原图）
- 输出：JSON 分类结果 + CoT 推理链

---

## 四、评估指标体系

### bbox 级（和之前实验可比）
- mAP50 / mAP50-95（Ultralytics 标准评估）
- Precision / Recall

### mask 级（新范式核心指标）
- mask F1（IoU>0.5 判定 TP）
- mask Precision / Recall
- mask IoU 分布（boxplot）

### 形态学级（跨分辨率可比）
- circularity 分布 KL 散度（B2 vs B3 vs 全局）
- solidity 分布 KL 散度
- 检测数量分布（每图 organoid 数）

### FL 级
- 通信效率：PrimitiveBatch JSON 大小 vs state_dict 大小
- 收敛速度：达到目标 F1 所需 FL 轮数
- 隐私性：上传内容不含像素/权重

---

## 五、技术路线图

```
Phase 1: Perception 层验证 (1-2周)
  ├── P1-A: B1→B1 复现 M14 (F1=93.9%)
  ├── P1-B: B1→B2 同分辨率跨域
  ├── P1-C: B1→B3 跨分辨率 ← 关键决策点
  ├── P1-D: B1+B2→B3 多源训练
  └── P1-E: 集中式上界

Phase 2: FL Primitive 聚合 (2-3周)
  ├── P2-A/B: 复用 E4/E6 bbox 结果作 baseline
  ├── P2-C: primitive FL (无形态学)
  ├── P2-D: primitive FL + 形态学 ← 核心实验
  └── 对比分析: primitive FL vs weight FL

Phase 3: VLM 推理层 (2周, 可选)
  ├── P3-A: 无 VLM baseline
  ├── P3-B: VLM 过滤
  └── P3-C: 完整三层框架
```

---

## 六、与 bbox FL 实验的可比性

| 维度 | bbox FL (E4-E11) | primitive FL (P2) |
|------|------------------|-------------------|
| 检测器 | YOLOv12n @640 | RF-DETR small @640 |
| 训练数据 | 同 B1/B2/B3 split | 同 |
| val_set | 同 9 张混合 | 同 |
| 评估 | bbox mAP50 (imgsz=640) | mask F1 (IoU>0.5) |
| FL 聚合 | state_dict 权重 | PrimitiveBatch JSON |
| 通信 | ~5.5MB (state_dict) | ~10KB (primitives JSON) |
| 隐私 | 权重可能泄露训练数据 | primitives 不含像素 |

**注意**：bbox mAP50 和 mask F1 不可直接比较。P1-E 集中式会同时评估 bbox mAP50 和 mask F1，作为两个范式的共同上界。

---

## 七、代码复用清单

| 组件 | 仓库 | 文件 | 状态 |
|------|------|------|------|
| RF-DETR 检测 | organoid-fl | sam2_segment.py | ✅ 已验证 |
| SAM2 分割 | organoid-fl | sam2_segment.py | ✅ F1=93.9% |
| 形态学特征 | organoid-fl | sam2_segment.py compute_morphology() | ✅ 10维 |
| Primitive 编解码 | ewa-fed | primitives.py PrimitiveCodec | ✅ 已实现 |
| EWA 熵加权聚合 | ewa-fed | aggregator.py | ✅ 已实现 |
| Conformity 检测 | ewa-fed | conformity.py | ✅ 已实现 |
| 幻觉防御 | PCB-Defect-FL | hallucination_defense.py | ✅ 5层 |
| 红线→mask 提取 | organoid-fl | sam2_segment.py load_gt_mask_from_annot() | ✅ 已实现 |

**7/8 组件已有现成代码，只需写 FL 编排层。**

---

## 八、决策点

1. **Phase 1 先跑哪个？** P1-C（B1→B3 跨分辨率）是关键决策点。如果 mask F1 > 30%，bbox→mask 转换有效。
2. **RF-DETR vs YOLOv12n？** Phase 1 用 RF-DETR（M14 已验证 F1=93.9%）。如果 RF-DETR 在 B3 上检测不到（M7: F1=7.1%），需要先用 B3 fine-tune RF-DETR。
3. **VLM 是否必须？** Phase 1 结果决定。如果形态学特征分布 B2/B3 重叠度高，VLM 可能不需要。如果重叠度低（B3 的 FP 有不同形态学特征），VLM 能提供额外增益。
4. **FL 轮数？** bbox FL 用 10 轮。primitive FL 每轮通信量小（10KB vs 5.5MB），可以跑更多轮（50-100 轮）。
5. **GT mask 怎么来？** 鼠肝有红色折线标注 → OpenCV 提取 → fillPoly。B1/B2/B3 都有红色折线原图。

---

## 九、论文叙事

### 问题
类器官检测面临三重挑战：小数据 + 不规则形状 + 数据分散（多实验室）

### 现有范式的瓶颈
1. bbox 检测：不规则形状信息丢失，背景噪声稀释信号
2. FL 权重聚合：round 级参数平均 ≠ 知识共享，non-IID 下异质性消解
3. 后处理治标：SAHI/NMS/Soft-NMS/HNM 均无法突破结构性瓶颈

### 我们的方案：三层视觉原语框架
1. **Perception**：bbox→SAM2 mask→形态学特征（跨域通用，不受像素尺度影响）
2. **Reasoning**：VLM 语义推理（泛化来源是理解，不是像素匹配）
3. **Collaboration**：FL 聚合 primitive 分布（数据不动，知识动）

### 实验
- 鼠肝 3 中心（B1/B2/B3，2 种分辨率）
- 对比：bbox FL (E4-E11) vs primitive FL (P2-C/D)
- 指标：bbox mAP50 + mask F1 + 形态分布 KL 散度 + 通信效率

### 贡献
1. 范式转换：从"聚合权重"到"聚合知识"
2. 视觉原语 + FL + 分割 = 完全空白领域
3. 形态学特征跨分辨率可比 — 解决 B3 小目标问题
