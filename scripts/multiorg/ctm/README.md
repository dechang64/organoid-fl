# CTM for Organoid TP/FP Discrimination

Continuous Thought Machine (CTM) implementation based on Darlow et al. (Sakana AI, NeurIPS 2025).

## 文件说明

| 文件 | 功能 |
|------|------|
| `ctm_module.py` | CTM 核心模块：NLMs + 神经同步化 + Cross-attention + CTM Loss |
| `ctm_dataset.py` | 数据集：从 metadata JSON + crops 目录加载 |
| `ctm_train.py` | 训练脚本：DINOv2(冻结) + CTM head |
| `ctm_evaluate.py` | 评估脚本：tick-wise AUC + attention 可视化 + 校准曲线 |
| `ctm_generate_crops.py` | Crop 生成：从 MultiOrg 原图 + SAM2 结果裁剪 bbox 区域 |

## 快速开始（冬生的 3060）

### 第1步：安装依赖
```powershell
pip install timm tifffile scikit-learn
```

### 第2步：生成 crops（从 MultiOrg 原图）
```powershell
python scripts\multiorg\ctm\ctm_generate_crops.py ^
    --sam2-results results\multiorg_sam2_zeroshot\multiorg_sam2_results.json ^
    --images-root "D:\datasets\mutliorg\MultiOrg_v2" ^
    --output-dir data\ctm_crops ^
    --pad-ratio 0.2
```
预期产出：~16198 个 PNG crop + `ctm_metadata.json`

### 第3步：训练 CTM
```powershell
python scripts\multiorg\ctm\ctm_train.py ^
    --metadata data\ctm_crops\ctm_metadata.json ^
    --crops-dir data\ctm_crops ^
    --output-dir results\ctm ^
    --epochs 50 ^
    --batch-size 32 ^
    --device cuda:0 ^
    --n-ticks 20 ^
    --d-internal 256 ^
    --mem-len 20 ^
    --n-heads 8 ^
    --lr 1e-4
```
预期：~2M 参数 CTM head + 86M DINOv2(冻结)，3060 上 ~8h

### 第4步：评估
```powershell
python scripts\multiorg\ctm\ctm_evaluate.py ^
    --checkpoint results\ctm\best.pt ^
    --metadata data\ctm_crops\ctm_metadata.json ^
    --crops-dir data\ctm_crops ^
    --output-dir results\ctm_eval ^
    --device cuda:0
```
产出：
- `tick_wise_auc.png` — AUC 随 tick 变化（应上升=在思考）
- `certainty_evolution.png` — 确定性演化
- `ticks_needed.png` — 自适应计算分布（应长尾）
- `tp_fp_trajectories.png` — TP vs FP 分数轨迹
- `calibration.png` — 校准曲线

## CTM 架构

```
crop(224×224) → DINOv2 ViT-B/14(冻结) → KV[257, 768]
                                            ↑
z_init(D=256) → [tick: sync→Q→cross-attn→synapse→NLM→z→output+certainty] × 20
                ↓
        argmin(loss) + argmax(certainty) → TP/FP 判定
```

### 关键创新（vs 标准 Transformer）
1. **NLMs**：每个神经元有私有权重（einsum 'bdM,Mhd->bdh'）
2. **神经同步化**：St = Zt · diag(R) · Zt^T（时间相关性，非快照）
3. **Q 从同步化来**：attention query 来自内部动力学，不是输入
4. **CTM Loss**：选 argmin(loss) + argmax(certainty) 两个 tick → 自适应计算涌现

## 评估关键指标

| 指标 | 期望 | 含义 |
|------|------|------|
| tick-wise AUC | 上升 | CTM 在迭代精炼（"思考"） |
| best AUC > RF-DETR | 是 | CTM 超过 baseline |
| ticks_needed 长尾 | 是 | 自适应计算（简单早停，困难多想） |
| 校准曲线 | 接近对角 | 预测概率 = 实际频率 |
| TP/FP 分离 | TP 分数上升, FP 不升 | 有判别力 |

## 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--n-ticks` | 20 | 内部思考步数 |
| `--d-internal` | 256 | 内部维度 D |
| `--mem-len` | 20 | 预激活历史长度 M |
| `--n-heads` | 8 | attention 头数 |
| `--n-action-pairs` | 128 | action 同步化神经元对数 |
| `--n-output-pairs` | 128 | output 同步化神经元对数 |
| `--d-hidden` | 16 | NLM 隐藏维度 |
| `--lr` | 1e-4 | 学习率 |
| `--epochs` | 50 | 训练轮次 |
| `--patience` | 15 | 早停 patience |

## 已验证（云 VM CPU）

- ✓ CTM module forward + backward
- ✓ Dataset 加载 100 crops (Phase 2 mode)
- ✓ E2E: DINOv2→CTM→loss→backward 通过
- ✓ Loss 下降（0.77→0.62 in 2 batches）

## 注意事项

1. **DINOv2 img_size=224**：224/14=16 patches → 257 tokens（含 CLS）
2. **backbone 冻结**：`requires_grad=False` + `model.eval()`
3. **类不平衡**：TP=28.6%, FP=71.4% → 训练集自动平衡采样
4. **Windows 编码**：所有 `open()` 加 `encoding='utf-8'`（ctm_dataset.py 已处理）
5. **16-bit TIFF**：用 `tifffile.imread()` 不用 PIL（PIL 在 Windows 上 segfault）
