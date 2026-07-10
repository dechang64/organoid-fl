# 三项系统审计：文献调研 + 代码审计 + Workplace 回顾

> 2026-07-10 曼卿 | 冬生提醒"文献调研，代码审计，workplace回顾，都不要忘记"

---

## 1. Workplace 回顾

### 1.1 发现重复造轮子

| 我的文件 | 已有文件 | 状态 |
|---------|---------|------|
| `ctm_generate_crops.py` | `scripts/multiorg/generate_crops.py` | **重复！** 已有版本功能完整：tiff读取+cache+pad+Windows中文路径 |
| `ctm_module.py` DINOv2 wrapper | `analysis/feature_extractor_v2.py` | 部分重复，但已有版本用 transformers 库且只返回 CLS token，我需要 spatial tokens（forward_features），不能直接复用 |
| `ctm_evaluate.py` attention 可视化 | `analysis/gradcam.py` 有 AttentionRollout | 可参考但 CTM 的 attention 是 per-tick 的，格式不同 |
| Phase 5 k-NN | `analysis/vector_engine.py` | 可复用 FedCtx HNSW，但当前用 sklearn NearestNeighbors 够用 |

**行动**：`ctm_generate_crops.py` 应标记为已弃用，README 指向已有的 `generate_crops.py`。但已有的版本采样 50TP+50FP，全量需要改 `--max-tp 0 --max-fp 0`。

### 1.2 已有可复用资产

| 组件 | 路径 | 用途 |
|------|------|------|
| DINOv2 特征提取 | `analysis/feature_extractor_v2.py` | CLS token 768d，可参考 normalize/transform |
| SAM2 分割 | `scripts/multiorg/multiorg_sam2.py` | 已有 SAM2 推理代码 |
| FL 框架 | `fl/federated_learning.py` | FL 训练循环，EWA 聚合 |
| 向量引擎 | `analysis/vector_engine.py` | Phase 5 k-NN 可复用 |
| GradCAM | `analysis/gradcam.py` | attention 可视化参考 |
| SAM2 结果 JSON | `results/multiorg_sam2_zeroshot/multiorg_sam2_results.json` | 16198 检测 + bbox + TP/FP label |
| Phase 2 crops | `results/phase2_vlm_100_mask/crops/` (100张) | 云 VM 测试用 |
| 实验追踪表 | `docs/experiment_tracker.md` | 需更新 CTM 实验 |

### 1.3 数据一致性检查

- `generate_crops.py` cache_key 格式: `{Class}_{Plate}_{image}_{det_idx}` — 和我的 `ctm_dataset.py` 一致 ✓
- DINOv2 normalize: `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]` — 和 `feature_extractor_v2.py` 一致 ✓
- SAM2 results bbox 格式: `[x1,y1,x2,y2]` 原图坐标 — 和 `crop_bbox()` 一致 ✓

---

## 2. 文献调研（补充遗漏）

### 2.1 DeShiftNet — 同数据集，最直接的竞品

**论文**：Zhuang et al., "DeShiftNet: a deformable-shifted cross-attention network for lightweight and robust organoid image segmentation." BMC Bioinformatics 2026.

**关键信息**：
- **数据集**：multi-type OrganoID（和我们的 MultiOrg 同源！）
- **任务**：类器官图像分割
- **核心创新**：
  1. **Deformable-shifted encoder** — 根据局部结构线索调整空间聚合
  2. **Cross-attention-guided decoder** — 精确定位
  3. 轻量级设计（lightweight）
- **结果**：在 multi-type OrganoID 上 Dice/mIoU/Precision/Recall 均最佳，特别在 PDAC organoid 上

**对我们的意义**：
- 这是和我们**同一数据集**上的 cross-attention 方法——不是泛泛的注意力，是针对类器官的
- 如果 DeShiftNet 的 cross-attention 在 OrganoID 上有效，说明 attention 机制对类器官形态确实有用
- 应该对比：CTM 的 cross-attention（Q从同步化来）vs DeShiftNet 的 deformable-shifted cross-attention（Q从局部结构来）
- **CTM 的优势**：迭代推理 + 自适应计算；**DeShiftNet 的优势**：deformable 对不规则形状的适应性

### 2.2 CTM 后续工作

- **NeurIPS 2025 正式发表**：论文已收录 proceedings
- **OpenReview**: forum?id=y0wDflmpLk — 有社区讨论
- **Sakana AI 官网**: pub.sakana.ai/ctm — 有视频演示
- **暂未发现直接后续论文**（2026年7月，论文很新）
- **关键 limitation**：论文自己说"not aiming for SOTA accuracy"——CTM 是概念验证，不是刷分工具

### 2.3 Argus (CVPR 2025) — 补充

**论文**：Man et al., "Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought." CVPR 2025.

**核心**：
- Object-centric grounding 作为视觉 CoT 信号
- Goal-conditioned visual attention — attention 被推理目标引导
- 和 DeepSeek Visual Primitives 互补：DeepSeek 用坐标作为推理单元，Argus 用 object grounding 作为 CoT 信号

### 2.4 文献调研总结——三条注意力路线

| 路线 | 论文 | Attention Q 来源 | 适合 organoid 的理由 |
|------|------|-----------------|---------------------|
| **神经同步化** | CTM | 内部动力学 | 迭代推理，自适应计算 |
| **Deformable shift** | DeShiftNet | 局部结构 | 不规则形状适应 |
| **Goal-conditioned** | Argus | 推理目标 | 目标引导的注意力 |

我们选了 CTM 路线，但 DeShiftNet 在同一数据集上已验证有效，值得作为 baseline 对比。

---

## 3. 代码审计（方法论层面）

### 3.1 CTM 实现忠实度审计

| 论文组件 | 我的实现 | 忠实度 | 备注 |
|---------|---------|--------|------|
| NLMs（per-neuron 私有权重） | ✓ einsum `bdM,Mhd->bdh` | ✓ | 正确 |
| 可学习衰减率 rij | ✓ `nn.Parameter(torch.zeros(n_pairs))` | ✓ | 初始化为 0（无衰减） |
| **同步化归一化** | ~~`sqrt(sum(decay²))`~~ → `sqrt(sum(decay))` | **Bug → 修复** | 论文 Eq 15: `βt_ij = sum(e^{-rij(t-τ)})`，不是平方 |
| Cross-attention Q 来源 | ✓ 从同步化投影 | ✓ | 正确 |
| Loss: argmin+argmax | ✓ | ✓ | 正确 |
| Synapse model | MLP | ⚠️ | 论文用 U-Net（ImageNet），MLP 是论文提到的替代方案 |
| 神经元对采样 | 随机采样 | ⚠️ | 论文有 Dense/Semi-dense/Random 三种策略，我用 Random |
| Q token 数量 | 1 个 | ⚠️ | 论文也是单 query token [B, 1, d_model] |

### 3.2 新发现 Bug

**同步化归一化公式错误（Critical）**：
- 论文 Eq 13-15：`St_ij = αt_ij / sqrt(βt_ij)` where `βt_ij = sum(e^{-rij(t-τ)})`
- 我的实现：`sqrt(sum(decay²))` = `sqrt(sum(e^{-2rij(t-τ)}))` ← **错！**
- 修复：`sqrt(decay.sum())` = `sqrt(sum(e^{-rij(t-τ)}))` ← **对！**
- 影响：归一化系数不对 → synchronization 值的尺度不对 → 可能影响训练稳定性

### 3.3 设计问题（非 bug，但需要注意）

1. **Synapse model 用 MLP 而非 U-Net**：
   - 论文说 ImageNet 上 U-Net 效果最好
   - 但 U-Net 更重，需要更多参数
   - 建议：先用 MLP 跑通，如果效果不好再换 U-Net

2. **DINOv2 img_size=224 → 257 tokens**：
   - 224/14=16 patches → 256 spatial + 1 CLS = 257
   - 每个 token 覆盖 14×14 像素
   - 对于 organoid crop（典型 100-200px），一个 organoid 只占 7-14 个 token
   - 可能太粗 → 可以考虑 518×518（37×37=1369 tokens）但 CPU/GPU 更慢

3. **CLS token 在 KV 中**：
   - DINOv2 的 forward_features 返回 [CLS, spatial_tokens...]
   - CLS token 的 attention 权重可能和 spatial tokens 不同
   - 建议：训练后检查 attention 是否过度关注 CLS

4. **post_acts_history in-place 修改**：
   - `post_acts_history[:, :, t] = z` 是 in-place 操作
   - PyTorch autograd 在这个场景下能工作（E2E 测试验证），但可能在不同配置下出问题
   - 更安全的做法是用 list + stack，但当前实现通过了测试

### 3.4 之前 bug 审计回顾（已修复）

| # | Bug | 严重性 | 状态 |
|---|-----|--------|------|
| 1 | backbone 未冻结（eval≠requires_grad=False） | Critical | ✅ 已修复 |
| 2 | NLM bias 维度反了 | Critical | ✅ 已修复 |
| 3 | 5个 open() 缺 encoding | High | ✅ 已修复 |
| 4 | CTM test 代码 unpack 2 值 | Medium | ✅ 已修复 |
| 5 | docstring 返回类型不匹配 | Low | ✅ 已修复 |
| 6 | 冗余 grad zero 循环 | Low | ✅ 已修复 |
| **7** | **同步化归一化公式错误** | **Critical** | **✅ 本次发现并修复** |

---

## 4. 下一步行动建议

### 立即
1. ✅ 同步化归一化已修复
2. 删除 `ctm_generate_crops.py`，README 指向已有的 `generate_crops.py`
3. 更新 `experiment_tracker.md` 加入 CTM 实验条目

### 文献
1. 下载 DeShiftNet 全文，仔细对比 cross-attention 架构差异
2. 下载 Argus (CVPR 2025) 全文，研究 goal-conditioned attention
3. 在 CTM 训练后，如果效果不好，考虑换 U-Net synapse model

### 代码
1. CTM 训练后检查 attention 是否过度关注 CLS token
2. 如果 224×224 效果不好，尝试 518×518（更多 spatial tokens）
3. 考虑实现论文 Appendix H 的高效递归同步化计算（当前是 O(T²) → 可以 O(T)）
