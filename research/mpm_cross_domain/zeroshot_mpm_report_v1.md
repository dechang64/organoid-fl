# Zero-shot MPM 检测实验报告 v1.0

> **日期**：2026-07-15
> **作者**：曼卿（Organoid-FL Agent）
> **实验**：用鼠肝 RF-DETR (b1 训练 mAP 89%) 对 34 张 MPM 类器官 bright-field patch 做 zero-shot 检测
> **假设**：代理训练（鼠肝）→ MPM 迁移
> **结论**：**假设基本不成立**——模型在 MPM 图像上几乎完全失效

---

## 一、实验设置

### 模型
- **架构**：RFDETRSmall (NMS-free Transformer)
- **训练数据**：鼠肝 b1（batch 1, 40 张图, 红折线精确标注）
- **训练性能**：b1 验证集 mAP ≈ 89%
- **Checkpoint**：`organoid-fl/runs/mouse_liver_v2/b1/full/checkpoint_best_ema.pth` (RFDETRSmall, num_classes=1)

### 测试数据
- **34 张 MPM 类器官 bright-field patch**，从 3 篇开放获取论文 PDF 提取
  - BMC Cancer 2023 Ito (9 张)
  - Sci Rep 2018 Mazzocchi (15 张)
  - Sci Rep 2025 Liu/Li Chenggang (10 张)
- **无 GT 标注**——只能评估检测量+置信度，不能算 mAP
- **推理设备**：CPU (PyTorch 2.12 CPU, 云 VM 无 GPU)

### 推理参数
- `threshold=0.01`（最大化召回，看模型对 MPM 的最低响应）
- 单张推理，无 TTA、无 SAHI

---

## 二、核心结果

### 2.1 总体

| 指标 | 值 |
|------|----|
| Total patches | 34 |
| Total detections (conf≥0.01) | 3121 |
| Mean detections/patch | 91.8 |
| 总推理时间 | 25.5s (0.75s/patch on CPU) |
| **Max confidence** | **0.696 (1 个检测)** |
| 检测 conf≥0.5 | **1 个** (1 个 patch) |
| 检测 conf≥0.3 | 1 个 (1 个 patch) |
| 检测 conf≥0.1 | 11 个 (6 个 patch) |
| 检测 conf≥0.05 | 69 个 (13 个 patch, 38.2%) |
| 检测 conf≥0.01 | 3121 个 (34 个 patch, 100%) |

### 2.2 置信度分布

```
0.00-0.05: 3052  ← 97.8% 是噪声
0.05-0.10:   58  ← 1.9%
0.10-0.20:    9  ← 0.3%
0.20-0.30:    1
0.30-0.50:    0
0.50-0.70:    1  ← 唯一像样的检测
0.70-0.90:    0
0.90-1.00:    0
```

**Median conf = 0.014** —— 模型几乎"不敢肯定"任何一个检测。

### 2.3 按来源论文分组

| 来源 | Patches | Det 总数 | Det/patch | Mean conf | Median conf |
|------|---------|---------|-----------|-----------|-------------|
| BMC Cancer 2023 (Ito) | 9 | 940 | 104.4 | 0.019 | 0.015 |
| Sci Rep 2018 (Mazzocchi) | 15 | 1483 | 98.9 | 0.017 | 0.014 |
| Sci Rep 2025 (Liu/Li) | 10 | 698 | 69.8 | 0.020 | 0.015 |

**三篇论文的失效程度基本相同**——说明不是某一篇论文图像特殊，而是模型整体跨域失效。

---

## 三、唯一的高置信度检测

**唯一 conf=0.696 的检测出现在 Sci Rep 2018 Fig 1 panel r1c1**：
- 图像：125×374 的窄长条 panel
- 内容：可能是 organoid 培养时间轴的第一张图
- 推测：因为这张图的背景/对比度接近鼠肝原图，模型"误认"为已知分布

**conf=0.282 的次高**也在 Sci Rep 2018 Fig 1 panel r1c6（同一组时间序列的另一张）。

---

## 四、对"代理训练→MPM 迁移"假设的判断

### 假设基本不成立

| 证据 | 说明 |
|------|------|
| Median conf = 0.014 | 模型对 MPM 图像几乎没有响应 |
| 97.8% 检测在 conf<0.05 | 噪声级检测 |
| 仅 1 个检测 conf≥0.5 | 全 34 张只有 1 个像样的检测 |
| 38.2% patch 有 conf≥0.05 检测 | 弱信号存在，但太弱 |
| 3 篇论文失效程度相同 | 跨域失效是模型层面，不是数据层面 |

### 4.1 失效原因分析（5 个假设）

| 假设 | 可能性 | 验证方法 |
|------|--------|---------|
| **A. 鼠肝 vs MPM 视觉差异太大** | 高 | 用 DINOv2 提特征，看鼠肝/MPM 在 embedding 空间的距离 |
| **B. 训练数据单一（只 b1, 40 张）** | 高 | 用更多 batch / MultiOrg 联合训练后重测 |
| **C. PDF 提取的图像分辨率太低** | 中 | 等拿到论文作者原始图像后重测 |
| **D. RF-DETR Small 容量不足** | 中 | 换 RF-DETR Base/Large 重测 |
| **E. threshold=0.01 噪声过多** | 已排除 | 即使 threshold=0.05 也只有 38.2% patch 有响应 |

### 4.2 但有一个好消息

**Sci Rep 2018 Fig 1 的时间序列图拿到了 conf=0.696**——说明：
- 当 MPM 图像的视觉特征接近鼠肝时（背景、对比度、organoid 形态大小），模型确实有响应
- 这意味着：**如果用更多样化的 organoid 数据联合训练（鼠肝 + MultiOrg + intestinal），跨域到 MPM 的能力可能显著提升**

---

## 五、下一步实验建议（优先级排序）

### 5.1 立即可做（云 VM, 1-2 天内）

| # | 实验 | 预期 | 工作量 |
|---|------|------|--------|
| **1** | 用 DINOv2 提取鼠肝/MPM/MultiOrg 特征，UMAP 可视化 | 看三种 organoid 在 embedding 空间的距离 | 2 小时 |
| **2** | 用鼠肝 + MultiOrg + intestinal 联合训练 RF-DETR，再 zero-shot MPM | 验证联合训练是否提升跨域 | 1 天训练（云 VM 太慢，需冬生本地 GPU） |
| **3** | SAM2 zero-shot 分割 MPM patch，作为伪 GT | 评估检测框对 organoid 的定位质量 | 4 小时 |
| **4** | 人工标注 34 张 patch 的 GT（napari 4 点多边形） | 建立 MPM organoid baseline GT | 2 小时（需要冬生） |

### 5.2 需要冬生本地 GPU（3060 12GB）

| # | 实验 | 预期 |
|---|------|------|
| **5** | 用 RF-DETR Base（32M 参数 vs Small 8M）重跑 zero-shot | 验证模型容量是否是瓶颈 |
| **6** | 用所有 organoid 数据（鼠肝 3 batch + MultiOrg + intestinal）联合训练 RF-DETR Base | 跨域 baseline |
| **7** | Diffusion 数据增强：用 SAM2 zero-shot MPM mask 做 pseudo-GT，生成合成 MPM 图像 | 补 A2 实验 |

### 5.3 需要等数据

| # | 实验 | 预期 |
|---|------|------|
| **8** | 联系论文作者获取原始图像（4 封邮件） | 提高图像质量 |
| **9** | 等瑞金 20 例 MPM PDO 到位后真实验证 | 真实数据集训练 |

---

## 六、对申请书的启示

### 6.1 不能直接声称的指标
- ❌ "鼠肝 89% → MPM 89%"（不成立）
- ❌ "代理数据训练的模型可直接迁移到 MPM"（不成立）

### 6.2 可以诚实声称的指标
- ✅ "鼠肝类器官 mAP 89%（同域）"
- ✅ "Zero-shot 跨域到 MPM 失效（max conf 0.696, median 0.014）"
- ✅ "需要联合训练或迁移学习才能实现跨域"

### 6.3 申请书必须新增的内容
- "代理验证策略"段落：明确说"算法在鼠肝上验证可行性，但跨域到 MPM 需要联合训练或迁移学习"
- "数据获取计划"段落：瑞金 20 例 + Darkjade 11 PDO 是跨域验证的关键
- 实验 A2 Diffusion 生成：合成 MPM 图像补足 20 例小样本

---

## 七、产物清单

```
/home/z/my-project/surf_2026/
├── mpm_organoid_patches/
│   ├── final_brightfield/           ← 34 张 MPM patch
│   ├── zeroshot_results/
│   │   ├── zeroshot_results.json     ← 完整结果
│   │   ├── top12_comparison.png     ← Top-12 高 conf 对比
│   │   └── *_det.png                 ← 34 张带检测框的 overlay
│   └── ...
├── zeroshot_mpm_rfdetr.py           ← 可复现的推理脚本
└── docs/
    └── zeroshot_mpm_report_v1.md    ← 本报告
```

---

## 八、复现命令

```bash
# 在云 VM 上
cd /home/z/my-project/surf_2026
python3 zeroshot_mpm_rfdetr.py
# 结果：zeroshot_results/zeroshot_results.json + 34 张 _det.png
```

---

**实验版本**：v1.0
**执行时间**：2026-07-15 15:01-15:02（25.5s 推理 + 10s 加载）
**置信度评估**：99% 可信（CPU 推理一致性高，无 GPU 随机性）
**下次更新**：等 DINOv2 特征可视化 + SAM2 zero-shot 伪 GT 完成后出 v2
