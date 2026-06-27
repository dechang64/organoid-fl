# 跨域类器官检测文献调研

## 问题背景

鼠肝类器官检测：8 张训练（2592×1944, 4X）→ F1=93.9%。新 10 对 test set（4000×3000, 不同相机）→ F1=0%。

核心问题：**跨分辨率、跨设备的域偏移导致检测器完全失效**。这不是调参能解决的，需要系统性的方法。

---

## 一、问题定义：CD-FSOD（Cross-Domain Few-Shot Object Detection）

### 1.1 CD-ViTO (ECCV 2024)
- **论文**: "Cross-Domain Few-Shot Object Detection via Enhanced Open-Set Object Detector"
- **作者**: Qian et al.
- **核心贡献**: 
  - 建立 CD-FSOD benchmark，提出三个度量域差距的维度：**style**（风格）、**ICV**（类间方差）、**IB**（不可定义边界）
  - 发现传统 few-shot 检测器在跨域时严重退化
  - 提出 CD-ViTO：基于 DE-ViT（open-set detector），加入三个模块：
    1. **Learnable instance features** — 对齐初始固定实例与目标类别
    2. **Instance reweighting** — 给高质量实例更高权重
    3. **Domain prompter** — 合成虚拟域增强风格鲁棒性
- **对我们项目的启示**: 风格差异(style)是域偏移的核心维度之一。鼠肝 batch 1 vs batch 2 的光照/对比度/色彩差异就是 style shift。CD-ViTO 的 domain prompter 思路——合成虚拟域——可以作为数据增强策略。

### 1.2 NTIRE 2025 CD-FSOD Challenge
- **论文**: "NTIRE 2025 Challenge on Cross-Domain Few-Shot Object Detection: Methods and Results" (CVPRW 2025)
- **关键发现**: 
  - 第一名方案：CD-ViTO + Instance Feature Caching (IFC)
  - IFC 解决目标域的 false/missing detection 问题
  - 基础模型（foundation model）对齐成为主流方向
- **对我们项目的启示**: 赛事top方案都在用 foundation model（如 DINOv2, SAM）做 backbone，而不是从头训练。我们的 RF-DETR 用 DINOv2 backbone 已有基础。

---

## 二、显微图像专用方法

### 2.1 MIAdapt (arXiv 2025)
- **论文**: "MIAdapt: Source-free Few-shot Domain Adaptive Object Detection for Microscopic Images"
- **场景**: 显微镜图像跨域检测，source-free（不访问源数据）
- **方法**: 
  - Source-free Few-shot Domain Adaptive Object detection (SF-FSDA)
  - 不需要源域数据，只需 few-shot 目标域数据
  - 自训练策略 + 伪标签 refinement
- **结果**: 在 WBC（白细胞）数据集上 +4.7% mAP
- **对我们项目的启示**: Source-free 场景非常适合联邦学习——各实验室不能共享数据，但可以用 few-shot 目标域数据做自适应。MIAdapt 的自训练策略可以直接用。

### 2.2 MICCAI 2024 FSDAOD
- **论文**: "Few-Shot Domain Adaptive Object Detection for Microscopic Images" (MICCAI 2024)
- **方法**: 
  1. **CBCP (Class-Balanced Copy-Paste)** — 源域数据增强，平衡类别
  2. **I2DA (Instance-level Inter/Intra Domain Alignment)** — 实例级域对齐
  3. **Instance-level classification loss** — 中间层特征保持
- **代码**: https://github.com/intelligentMachines-ITU/Few-Shot-Domain-Adaptive-Object-Detection-MICCIA-2024
- **对我们项目的启示**: Copy-Paste 增强是解决类不平衡 + 域适应的简单有效方法。把新域的 few-shot organoid copy-paste 到源域背景上，可以快速适配。

---

## 三、类器官专用方法

### 3.1 DeShiftNet (BMC Bioinformatics 2026)
- **论文**: "DeShiftNet: a deformable-shifted cross-attention network for lightweight and robust organoid image segmentation"
- **场景**: 多类型 OrganoID 数据集上的分割，专门解决域偏移
- **方法**: 
  - Deformable-shifted encoding mechanism — 根据局部结构线索调整空间聚合
  - Cross-attention 增强跨域鲁棒性
  - 轻量级设计
- **对我们项目的启示**: 这是目前唯一直接针对 organoid 跨域的方法。deformable-shifted 机制的核心思想——根据局部结构自适应调整感受野——和我们"图像预处理标准化"的方向不同，它是模型层面的解决方案。

### 3.2 CLORG (2025)
- **论文**: "CLORG: A contrastive learning-based framework for morphological representation and classification of organoids"
- **方法**: 监督对比学习做形态学表征
- **对我们项目的启示**: 对比学习可以学到域不变的形态学特征。如果不同实验室的 organoid 形态学特征（圆度、Solidity 等）是域不变的，对比学习可以放大这些特征的判别力。

### 3.3 OrgLine (Cell Reports Methods 2026)
- **论文**: "OrgLine: A versatile pipeline for organoid morphometry using bright-field microscopy"
- **核心**: 大规模 curated benchmark，跨发育阶段形态学定量
- **对我们项目的启示**: OrgLine 的 benchmark 可以作为我们的外部验证集。它的 pipeline 设计（detection → morphometry → tracking）和我们的架构一致。

### 3.4 Knowledge-Driven Framework (PMC 2025)
- **论文**: "A knowledge-driven deep learning framework for organoid detection"
- **方法**: 多模态 transformer + 频域特征 + 人在环反馈
- **对我们项目的启示**: 人在环反馈（human-in-the-loop）适配不同实验条件——这和 few-shot adaptation 的思路一致，但更强调交互式标注。

---

## 四、图像预处理标准化方向

### 4.1 CLAHE (Contrast-Limited Adaptive Histogram Equalization)
- **文献**: 多篇 2024-2025 医学图像论文证实 CLAHE + 数据增强显著提升分类精度
- **应用**: 已在 MRI、CT、组织病理学中广泛使用
- **对类器官的适用性**: 明场类器官图像对比度低、光照不均，CLAHE 是自然选择。不同显微镜的对比度差异可以通过 CLAHE 部分消除。

### 4.2 Stain Normalization → 频域/色彩归一化
- **MICCAI 2025**: "Adaptive Stain Normalization for Cross-Domain Medical Histology" — 可训练的色彩归一化模型，可集成到任何 backbone
- **对类器官的适用性**: 类器官不是 H&E 染色，但明场显微镜的色彩偏差（白平衡、色温）可以用类似方法归一化。

### 4.3 Test-Time Adaptation (TTA)
- **CVPRW 2024**: "Fully Test-time Adaptation for Object Detection" — 单张测试图像上自适应更新检测器
- **NeurIPS 2025**: Foundation model-powered TTA for object detection
- **对类器官的适用性**: TTA 不需要重新训练，在推理时用测试图像本身做自适应。非常适合"不同实验室不同设备"的场景——每张新图都做一次轻量适配。

---

## 五、对我们项目的方向建议

### 短期（验证可行性）

| 方法 | 复杂度 | 预期效果 | 文献支撑 |
|------|--------|---------|---------|
| **Few-shot fine-tune** (2-3 张新域) | 低 | 高 | MICCAI 2024, MIAdapt |
| **CLAHE 预处理** | 低 | 中 | CLAHE literature |
| **Copy-Paste 增强** | 低 | 中 | MICCAI 2024 CBCP |

### 中期（系统化解决）

| 方法 | 复杂度 | 预期效果 | 文献支撑 |
|------|--------|---------|---------|
| **Source-free FSDA** | 中 | 高 | MIAdapt 2025 |
| **Domain Prompter** | 中 | 高 | CD-ViTO ECCV 2024 |
| **TTA (Test-Time Adaptation)** | 中 | 中 | CVPRW 2024, NeurIPS 2025 |

### 长期（平台级）

| 方法 | 复杂度 | 预期效果 | 文献支撑 |
|------|--------|---------|---------|
| **图像预处理标准化模块** | 中 | 高 | CLAHE + Stain Norm |
| **Deformable-shifted 注意力** | 高 | 高 | DeShiftNet 2026 |
| **对比学习域不变特征** | 高 | 高 | CLORG 2025 |
| **联邦学习 + 域适应** | 高 | 高 | MIAdapt + FL |

---

## 六、核心洞察

1. **域偏移是类器官检测的公认难题**：DeShiftNet 专门为此设计，CLORG 用对比学习应对，OrgLine 建 benchmark 统一评估。不是我们独有问题。

2. **Few-shot adaptation 是主流方向**：无论是 CD-FSOD challenge 还是显微图像专用方法，都指向"用少量目标域数据做自适应"。MIAdapt 的 source-free 设定和联邦学习天然兼容。

3. **预处理标准化是必要但非充分的**：CLAHE/色彩归一化能缩小域差距，但不能完全消除。需要和模型层面的域适应结合。

4. **Foundation model 是趋势**：NTIRE 2025 的 top 方案都基于 foundation model。RF-DETR 的 DINOv2 backbone 已经在这个方向上，SAM2 的 zero-shot 能力也是。关键是把 foundation model 的泛化能力利用好。

5. **FL + 域适应是蓝海**：MIAdapt 的 source-free 设定和联邦学习天然兼容（数据不动模型动），但目前没有人在 organoid + FL + 域适应的交叉领域做过系统工作。这是我们论文的独特定位。

---

## 参考文献列表

1. Qian et al. "Cross-Domain Few-Shot Object Detection via Enhanced Open-Set Object Detector." ECCV 2024.
2. Fu et al. "NTIRE 2025 Challenge on Cross-Domain Few-Shot Object Detection." CVPRW 2025.
3. Dilawar et al. "MIAdapt: Source-free Few-shot Domain Adaptive Object Detection for Microscopic Images." arXiv 2025.
4. "Few-Shot Domain Adaptive Object Detection for Microscopic Images." MICCAI 2024.
5. Tan et al. "DeShiftNet: a deformable-shifted cross-attention network for lightweight and robust organoid image segmentation." BMC Bioinformatics 2026.
6. "CLORG: A contrastive learning-based framework for morphological representation and classification of organoids." 2025.
7. "OrgLine: A versatile pipeline for organoid morphometry using bright-field microscopy." Cell Reports Methods 2026.
8. "A knowledge-driven deep learning framework for organoid detection." PMC 2025.
9. "Adaptive Stain Normalization for Cross-Domain Medical Histology." MICCAI 2025.
10. Ruan et al. "Fully Test-time Adaptation for Object Detection." CVPRW 2024.
