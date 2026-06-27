# SAM2 弱监督微调文献调研

## 问题背景

MultiOrg GT 是 napari 4点多边形（位置标注），不是精确轮廓。用粗糙 GT 做 mask supervision 微调 SAM2 → 负优化（mask mAP -4pp）。

需要找到正确的微调方式：利用位置标注（point/box prompt）而不是 mask GT。

---

## 一、PointSAM：点标注微调 SAM（IEEE TGRS 2025）

**论文**: "PointSAM: Pointly-Supervised Segment Anything Model for Remote Sensing Images"

**核心方法**:
- 用**点标注**（最弱的标注形式）微调 SAM
- **自训练框架**：利用 SAM zero-shot 能力迭代生成伪标签
- 伪标签噪声控制：从不同视角提取目标原型（prototype），过滤噪声伪标签
- PointSAM 可作为 point-to-box converter

**对 MultiOrg 的启示**:
- MultiOrg 的 4点多边形可以退化为中心点 → point prompt
- 用 SAM zero-shot 生成伪 mask（而不是用粗糙 GT mask）
- 伪 mask 做微调 supervision → 避免粗糙 GT 拉低质量
- 关键挑战：伪标签噪声控制

---

## 二、All-in-SAM：弱标注到像素级分割（PMC 2025）

**论文**: "All-in-SAM: from Weak Annotation to Pixel-wise Nuclei Segmentation"

**核心方法**:
- SAM 全流程应用：从标注生成到模型微调
- **第一步**：SAM 用弱 prompt（point/bbox）生成像素级标注
- **第二步**：用生成的像素级标注微调 SAM
- 即 SAM 自己生成伪 GT → 再用伪 GT 微调自己

**对 MultiOrg 的启示**:
- 4点多边形 → bbox prompt → SAM zero-shot 生成精确 mask
- 用这个精确 mask 作为伪 GT 微调 SAM
- 比 PointSAM 更简单：不需要点标注，直接用 bbox
- 本质是**自蒸馏**：teacher=SAM zero-shot, student=finetuned SAM

---

## 三、SAM-2 自蒸馏/半监督方法

**论文**: "Training a Student Expert via Semi-Supervised Foundation Model Distillation" (CVPRW 2026)
- 三阶段框架：(i) 域适应 (ii) 知识蒸馏 (iii) 半监督
- teacher（frozen）→ student（compact）

**论文**: "SAM-Guided Federated Semi-Supervised Learning" (arXiv 2025)
- 双知识蒸馏 + 自适应一致性机制
- SAM 作为 teacher 引导像素级监督

**对 MultiOrg 的启示**:
- SAM zero-shot 作为 frozen teacher
- 微调 student（mask_decoder）学习 teacher 的输出
- 不需要 GT mask，只需要 teacher 的伪标签

---

## 四、推荐方案

### 方案 A：SAM 自蒸馏（最简单，直接可用）

1. **Teacher**: SAM2 zero-shot，用 RF-DETR bbox 作为 prompt 生成精确 mask
2. **Pseudo GT**: Teacher 的 mask 输出（像素级精确）
3. **Student**: SAM2 mask_decoder 微调，用 pseudo GT 做 supervision
4. **微调目标**: 让 student 在 MultiOrg 域上的 mask 更好（而非拟合粗糙 4点多边形）

**优势**: 不需要额外标注，直接利用 SAM2 zero-shot 的精确输出
**风险**: 自蒸馏可能无法引入新信息（teacher 和 student 同构）

### 方案 B：PointSAM 式自训练（更复杂，可能更有效）

1. 4点多边形 → 中心点 → point prompt
2. SAM2 zero-shot 用 point prompt 生成伪 mask
3. 原型过滤：从多个实例提取特征原型，过滤低质量伪 mask
4. 用过滤后的伪 mask 微调 SAM2
5. 迭代：微调后的 SAM2 重新生成伪 mask → 再微调

**优势**: 迭代优化，伪标签质量逐步提升
**风险**: 误差累积，需要好的噪声过滤机制

### 方案 C：混合 supervision（折中）

1. **位置 supervision**: 4点多边形的中心点/bbox 作为 prompt
2. **形状 supervision**: SAM2 zero-shot mask 作为 soft label（而非 hard label）
3. **置信度加权**: 高置信度区域用 zero-shot mask，低置信度区域用 GT 4点多边形

**优势**: 结合两种信息源
**风险**: 实现复杂度较高

---

## 五、参考文献

1. Liu et al. "PointSAM: Pointly-Supervised Segment Anything Model for Remote Sensing Images." IEEE TGRS 2025.
2. "All-in-SAM: from Weak Annotation to Pixel-wise Nuclei Segmentation." PMC 2025.
3. Taghavi et al. "Training a Student Expert via Semi-Supervised Foundation Model Distillation." CVPRW 2026.
4. "SAM-Guided Federated Semi-Supervised Learning for Medical Image Segmentation." arXiv 2025.
5. "SAM4MIS: Segment Anything Model for Medical Image Segmentation." GitHub repo, 2024-2025.
