# 视觉原语分割推理框架：研究计划

> 创建：2026-06-26
> 前置实验：organoid-fl SAM2 mask_decoder 微调（进行中）
> 技术栈：ewa-fed primitives.py + organoid-fl segmentor.py + PCB-Defect-FL hallucination_defense.py

---

## 一、动机：为什么需要范式转换

### 当前范式的瓶颈

| 范式 | 代表方法 | 瓶颈 |
|------|----------|------|
| 检测优化 | YOLO/RF-DETR + SAHI + NMS | 小数据集天花板（MultiOrg 600张→77.76%），FP 靠后处理治标 |
| 分割微调 | SAM2 mask_decoder finetune | 形态学过滤 zero-shot 无效（MultiOrg 验证），微调后待验证 |
| 联邦学习 | FedAvg/FedProx/EWA | 聚合权重 ≠ 聚合知识，non-IID 下异质性消解 |

### 核心洞察

1. **Soft-NMS 赢 6pp** → 保留指代不确定性比强制决策更优（Reference Gap）
2. **多标注者 = 多提案** → 不需要选"正确答案"，保留分布
3. **FL 泛化 = 聚合视觉原语分布**，不是聚合权重 → 数据不动，知识动
4. **小数据 + 不规则形状** → 像素匹配泛化不了，语义理解可以

---

## 二、三层框架设计

### Layer 1: Perception（感知层）— 跨域通用

**输入**：原始图像
**输出**：无类别 visual primitives（mask + 位置 + 形态学特征）

**组件**：
- RF-DETR 检测 → bbox（已有）
- SAM2 分割（微调后）→ mask（organoid-fl SAM2 微调实验验证中）
- 形态学特征提取 → 10维向量（area/circularity/solidity/eccentricity/...）

**关键设计**：
- 不学习"这是什么"，只学习"这里有什么"
- 微调 SAM2 = 给 Perception 层领域形态先验
- 跨域通用：同一 Perception 层可用于 organoid / PCB / 病理

**当前状态**：
- ✅ segmentor.py 已实现 pipeline
- ⏳ SAM2 微调验证中（organoid）
- ❌ PCB 微调未开始

### Layer 2: Reasoning（推理层）— 语义泛化

**输入**：visual primitives + 全图上下文
**输出**：分类标签 + 置信度 + 推理链（CoT）

**组件**：
- VLM 推理（GLM-4V / CLIP / Qwen-VL）
- 视觉原语编码（DeepSeek 格式，primitives.py 已实现）
- 幻觉防御（5层，hallucination_defense.py 已实现）

**关键设计**：
- 泛化来源：语义理解，不是像素匹配
- VLM 看 mask 轮廓 + 形态学指标 → 判断"这是 organoid 还是碎片"
- 推理可解释（CoT）：不只是 yes/no，还有为什么

**待实现**：
- VLM 推理层（框架中唯一缺失的组件）
- Prompt 模板设计
- 推理结果 → 后验过滤

### Layer 3: Collaboration（协作层）— FL 聚合

**输入**：各中心的 visual primitives 分布
**输出**：全局 primitive 分布 + 中心置信度

**组件**：
- EWA 熵加权聚合（aggregator.py 已实现）
- Conformity 从众检测（conformity.py 已实现）
- Primitive 分布传输（primitives.py PrimitiveBatch 已实现）

**关键设计**：
- FL 不聚合权重，聚合 primitive 分布
- 各中心上传"形态统计"，不上传像素/权重
- EWA 有效区间：适度 non-IID 下优势最明显

---

## 三、实验路线图

### Phase 0: SAM2 微调验证（当前，organoid-fl）

| 实验 | 目的 | 状态 |
|------|------|------|
| SAM2 zero-shot 形态学过滤 | baseline | ✅ 完成，无效 |
| SAM2 mask_decoder 微调 | 给 Perception 层形态先验 | ⏳ 数据准备中 |
| 微调后 SAM2 形态学过滤 | 验证微调有效性 | 待做 |

**决策点**：
- 如果微调后形态学过滤有效 → Perception 层就绪，进入 Phase 1
- 如果微调后仍无效 → 说明 mask 级形态学不够，需要 VLM Reasoning 层补充

### Phase 1: VLM 推理层实现

| 任务 | 输入 | 输出 |
|------|------|------|
| Prompt 模板设计 | visual primitive + 图像 | 结构化文本 |
| VLM 调用 | GLM-4V / Qwen-VL | 分类 + CoT |
| 后验过滤 | VLM 判断 | 过滤 FP |
| 幻觉防御 | 5层校验 | 可信度 |

**验证**：
- 在 MultiOrg SAHI 结果上，用 VLM 过滤 FP
- 对比：conf 阈值过滤 vs VLM 推理过滤
- 鼠肝：VLM 判断"这是类器官吗" → F1 从 80% 提升到？

### Phase 2: FL Primitive 聚合

| 任务 | 输入 | 输出 |
|------|------|------|
| Primitive 分布提取 | 各中心检测结果 | 形态统计 |
| EWA 聚合 | 多中心 primitive | 全局分布 |
| Conformity 检测 | 少数专家知识 | 压制检测 |

**验证**：
- 3-center non-IID organoid FL
- 对比：FedAvg 权重聚合 vs Primitive 分布聚合
- 指标：mAP50-95 + 形态分布 KL 散度

### Phase 3: 跨域验证

| 领域 | 数据 | 目标 |
|------|------|------|
| Organoid | MultiOrg + 鼠肝 | 跨器官泛化 |
| PCB | DeepPCB + SolDef_AI | 跨工业场景泛化 |
| 病理 | (合作方) | 跨医学场景泛化 |

---

## 四、技术栈映射

```
[图像]
  ↓
RF-DETR 检测 → bbox                    ← organoid-fl 已有
  ↓
SAM2 分割（微调后）→ mask              ← organoid-fl SAM2 微调验证中
  ↓
形态学特征 → 10维向量                  ← segmentor.py 已有
  ↓
视觉原语编码 → DeepSeek格式            ← ewa-fed primitives.py 已有
  ↓
VLM 推理 → 分类+CoT                    ← 待实现（Phase 1）
  ↓
幻觉防御 → 5层校验                     ← PCB-Defect-FL hallucination_defense.py 已有
  ↓
FL 聚合 → primitive 分布               ← ewa-fed aggregator.py + conformity.py 已有
  ↓
输出：检测+分割+分类+可信度+推理链
```

**6/7 步已有现成代码，只需实现 VLM 推理层。**

---

## 五、与现有论文的关系

### 可直接引用的论文
- DeepSeek "Thinking with Visual Primitives" (2025) — primitives.py 已实现其格式
- Lu et al. VisDoT (arXiv 2603.11631) — point primitive 推理
- Kirillov et al. SAM (ICCV 2023) + SAM2 (2024) — Perception 层
- Oquab et al. DINOv2 (TMLR 2024) — 跨域特征

### 空白领域（我们的贡献）
- **视觉原语 + FL + 分割 = 完全空白** — primitives.py 是首创
- **SAM2 微调 + 形态学过滤 + VLM 推理** = 没有论文做过完整链路
- **FL 聚合 primitive 分布** = 从"聚合权重"到"聚合知识"的范式跳跃

### 论文叙事
1. 问题：小数据 + 不规则形状 + 数据分散 → 传统检测优化有天花板
2. 洞察：保留指代不确定性（Soft-NMS）+ 语义理解（VLM）+ 知识聚合（FL）
3. 方法：三层视觉原语框架（Perception / Reasoning / Collaboration）
4. 实验：organoid（MultiOrg + 鼠肝）+ PCB（DeepPCB + SolDef_AI）跨域验证
5. 贡献：范式转换，不是渐进优化

---

## 六、与冬生理念的呼应

- "数据不动，模型动" → FL 聚合权重（已有）
- "数据不动，知识动" → FL 聚合 primitive 分布（本框架）
- "数据不动，推理动" → VLM 推理层（本框架）

三层递进，视觉原语框架是这三个层次的完整实现。
