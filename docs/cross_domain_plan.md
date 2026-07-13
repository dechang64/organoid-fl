# 跨域泛化实验方案：MultiOrg Slot Model → 鼠肝

> 2026-07-13 曼卿起草

## 1. 背景与动机

### 1.1 传统方法在鼠肝上的失败

鼠肝 v2 实验确认 bbox 检测范式的跨域失败：
- B1→B2 zeroshot: F1=0.22（大目标→小目标，尺度差异导致崩溃）
- B1→B3 zeroshot: F1=0.00（完全失效）
- 传统 CV (Otsu): B1 F1=0.69, B2=0.80, B3=0.80（不如 RF-DETR）
- SAM2 fewshot: B2 F1=0.92, B3=0.67（需要目标域标注）

**根因**：bbox 检测器在 B1（2592×1944, 大目标）训练，到 B2/B3（4000×3000, 小目标）时 organoid 在画面中相对大小变了，检测器学到的尺度特征不适用。

### 1.2 Slot Attention 的跨域潜力

MultiOrg 实验证明 Slot Attention + SupCon 学到了 object-centric 表征：
- Combined AP 0.861（超过 RF-DETR 0.829, +3.3pp）
- DINOv2 backbone 冻结 → 不依赖目标域微调
- Slot embeddings 捕获 organoid 内部结构，不是 bbox 尺度

**假设**：Slot model 在 MultiOrg 上学到的是"organoid 长什么样"（object-centric 原语），不是"MultiOrg 的 bbox 分布"。如果假设成立，直接迁移到鼠肝应该有效。

### 1.3 关键差异

| 维度 | MultiOrg | 鼠肝 |
|------|---------|------|
| 图像类型 | 16-bit TIFF 明场 | JPG 明场 |
| 分辨率 | 6385×5720 | 2592×1944 / 4000×3000 |
| 染色 | 相差/明场 | 明场 |
| 目标 | 类器官（圆形/椭圆形） | 类器官（圆形/椭圆形） |
| 标注 | napari 4点多边形 | 红色折线精确轮廓 |
| 检测器 | RF-DETR Small | RF-DETR Small |
| 之前 SupCon | ✅ 训练过 | ❌ 未训练 |

**相似性**：都是明场类器官图像，目标形态相似（圆形/椭圆形细胞团）。
**差异**：分辨率/位深/染色方法不同 → DINOv2 作为通用视觉编码器应能处理。

## 2. 实验设计

### 2.1 总体流程

```
MultiOrg 训练的 Slot Model (best.pt)
         │
         ├── Step 1: 鼠肝 RF-DETR 检测 → bbox + confidence
         │
         ├── Step 2: 鼠肝 bbox crops → DINOv2 → Slot Attention → slot score
         │   （模型直接从 MultiOrg best.pt 加载，不微调）
         │
         ├── Step 3: Combined score = conf^0.8 × slot^0.2
         │
         └── Step 4: 用鼠肝 GT 标注评估 AP/F1
```

### 2.2 实验矩阵

| # | 实验 | 模型来源 | 目标域 | 评估方式 | 预期 |
|---|------|---------|--------|---------|------|
| G1 | RF-DETR baseline | 鼠肝 B1 full | B1/B2/B3 | bbox F1 | B1=1.00, B2=0.80, B3=0.75 |
| G2 | **Slot zero-shot 跨域** | **MultiOrg best.pt** | B1/B2/B3 | slot AUC + combined AP | 核心实验 |
| G3 | Slot fine-tune | MultiOrg → 鼠肝微调 | B1/B2/B3 | slot AUC + combined AP | 上界 |
| G4 | Slot 鼠肝 native | 鼠肝自训练 | B1/B2/B3 | slot AUC + combined AP | 对比基线 |

### 2.3 关键对比

**G2 vs G1**：Slot 跨域过滤是否提升鼠肝检测？
- 如果 G2 combined AP > G1 baseline AP → 跨域有效
- 如果 G2 slot AUC > 0.5 → slot 学到了通用 organoid 表征

**G2 vs G4**：跨域 vs 同域差距多大？
- 如果 G2 ≈ G4 → 跨域迁移成功（不需鼠肝标注）
- 如果 G2 << G4 → 需要目标域适应

**G2 在 B1 vs B2 vs B3**：跨域是否解决尺度问题？
- 传统方法 B1→B2 F1 从 1.00 降到 0.22
- 如果 G2 在 B2/B3 上 slot AUC ≈ B1 → 尺度无关

## 3. 数据准备

### 3.1 鼠肝 crops 生成

需要为鼠肝 B1/B2/B3 各生成类似 MultiOrg 的 `ctm_metadata.json`：

```json
[
  {
    "cache_key": "b1_image_00_det0",
    "image": "b1/image_00.jpg",
    "det_idx": 0,
    "bbox": [x1, y1, x2, y2],
    "rfdetr_conf": 0.93,
    "matched": true,
    "match_iou": 0.78,
    "crop_path": "crops/b1_image_00_det0.png"
  }
]
```

**已有数据**：
- 鼠肝 RF-DETR 检测结果（`sam2_results.json` 里有 bbox + confidence）
- 鼠肝 GT 标注（`annotations.json` 里有 bboxes）
- 鼠肝原始图片（`images/` 目录）

**需要做**：
1. 对每张鼠肝图片跑 RF-DETR 检测（如果还没有全量结果）
2. 对每个检测框裁剪 crop（224×224，DINOv2 输入尺寸）
3. 匹配检测框和 GT（IoU≥0.5 = TP）
4. 生成 metadata JSON

### 3.2 关键参数

- crop 尺寸：224×224（DINOv2 输入）
- RF-DETR 阈值：conf≥0.1（保留足够 FP 做评估）
- IoU 匹配阈值：0.5（和 MultiOrg 一致）

## 4. 实现步骤

### Step 1: 生成鼠肝 crops + metadata（冬生 3060）

```powershell
python scripts\mouse_liver\generate_mouse_crops.py \
    --data-dir mouse_liver_data_correct \
    --output-dir data\mouse_crops \
    --batch b1 --conf-threshold 0.1
```

对 B1/B2/B3 各跑一次。

### Step 2: 用 MultiOrg best.pt 跨域评估（冬生 3060）

```powershell
python scripts\multiorg\slot_c4_eval.py \
    --checkpoint results\supcon_8s_d128_p256_t0.07_b0.1_20260713_003826\best.pt \
    --metadata data\mouse_crops\mouse_metadata_b1.json \
    --crops-dir data\mouse_crops \
    --device cuda:0
```

对 B1/B2/B3 各跑一次。

### Step 3: 对比分析（云 VM）

汇总结果，对比：
- RF-DETR baseline F1（已有）
- Slot 跨域 combined AP（Step 2 输出）
- Slot 跨域 slot AUC（Step 2 输出）

## 5. 预期结果

### 5.1 乐观情况（跨域有效）

| 指标 | B1 | B2 | B3 |
|------|----|----|----|
| RF-DETR F1 | 1.00 | 0.80 | 0.75 |
| Slot AUC (跨域) | 0.85 | 0.83 | 0.80 |
| Combined AP (跨域) | 0.90 | 0.85 | 0.82 |

→ Slot 学到了通用 organoid 原语，跨域有效

### 5.2 保守情况（部分有效）

| 指标 | B1 | B2 | B3 |
|------|----|----|----|
| Slot AUC (跨域) | 0.75 | 0.65 | 0.60 |
| Combined AP (跨域) | 0.85 | 0.78 | 0.75 |

→ Slot 有信号但不如 MultiOrg，需要少量目标域微调

### 5.3 失败情况（跨域无效）

| 指标 | B1 | B2 | B3 |
|------|----|----|----|
| Slot AUC (跨域) | <0.55 | <0.55 | <0.55 |

→ MultiOrg 的 slot 不通用，需要鼠肝训练数据

## 6. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| DINOv2 对 JPG vs TIFF 特征不同 | 低 | DINOv2 是通用视觉编码器，JPG/TIFF 都能处理 |
| 鼠肝 organoid 形态和 MultiOrg 差异大 | 中 | 先看 G2 slot AUC，如果 >0.7 说明有信号 |
| RF-DETR 鼠肝 checkpoint 和 MultiOrg 不同 | 低 | 用各自 batch 训练的 checkpoint |
| 鼠肝检测框太少（每张图只有几个） | 中 | 对所有检测框评估，不做下采样 |

## 7. 论文价值

如果 G2 成功（跨域有效），论文叙事：
1. MultiOrg 上训练 slot model → AP 0.861（超过 RF-DETR +3.3pp）
2. **直接迁移到鼠肝（零标注）** → 仍然有效
3. 联邦 slot 聚合 → 小 client 获益最大

这是一个完整的"通用 organoid 原语"故事：
- 训练一次（MultiOrg），部署多次（MultiOrg + 鼠肝 + 未来数据集）
- 不需要目标域标注
- 联邦聚合天然支持新实验室加入

如果 G2 失败，论文叙事退回到：
- Slot + SupCon 在 MultiOrg 上有效（+3.3pp）
- 联邦聚合有效（+1.1pp）
- 跨域需要少量微调（G3）

## 8. 下一步

1. **立即可做**：写 `generate_mouse_crops.py`（从鼠肝 RF-DETR 结果 + GT 生成 crops + metadata）
2. **需要冬生跑**：在 3060 上跑 `slot_c4_eval.py`（用 MultiOrg best.pt 评估鼠肝 crops）
3. **云 VM 分析**：汇总 G1-G4 结果，生成对比表
