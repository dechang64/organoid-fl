# multiorg_sam2.py 代码审计报告
**日期**: 2026-06-28  
**文件**: `scripts/multiorg/multiorg_sam2.py` (799行)  
**最后 commit**: c17b381 (已 push)

---

## 🔴 Critical Bug

### 1. mask_iou_cropped() offset x/y 交换（line 316-321）

**GT offset** 格式（sahi_inference.py `load_ground_truth_masks`）:
```python
results.append((bbox, mask, (x0, y0)))  # offset = (x, y) = (col, row)
```

**Detection offset** 格式（line 265）:
```python
morph['mask_offset'] = (cx1, cy1)  # (x, y) = (col, row)
```

两者都是 (x, y)。

但 `mask_iou_cropped` 解包:
```python
y1_1, x1_1 = offset1  # ← 把 offset[0]=x 读成 y，offset[1]=y 读成 x
```

后续全部计算都 x/y 互换：
- `y2_1 = y1_1 + h1` → x + height（col + 行数）❌
- `x2_1 = x1_1 + w1` → y + width（row + 列数）❌
- 交集区域索引 `mask1[iy1-y1_1:iy2-y1_1, ...]` 行列混用

**影响**: 所有 mask IoU 计算结果错误 → mask AP50 数值不可信。  
**修复**: 改解包顺序为 `x1_1, y1_1 = offset1`。

---

## 🟡 Medium Issues

### 2. mask_iou() 是死代码（line 306-310）
`mask_iou()` 定义了但从未被调用。所有调用走 `mask_iou_cropped()`。  
**处理**: 删除。

### 3. AP 计算是 per-image 平均，不是 COCO 风格 dataset AP
`evaluate_detections()` 和 `evaluate_detections_mask()` 都是单图 AP，然后 `main()` 里做 `mean(per_image_ap)`。  
COCO 风格是全数据集累积 TP/FP 后算一个 AP。  
**影响**: 图间 GT 数量差异大时，per-image mAP 会偏离 COCO mAP。  
**处理**: 保持现状（和 sahi_inference.py 一致），但报告里标注 "per-image mean AP"。

### 4. try_morphology_filters() 只评 bbox IoU，不评 mask IoU
filter analysis 调用 `evaluate_detections`（bbox），不调用 `evaluate_detections_mask`。  
**影响**: 无法看形态学过滤对 mask 质量的影响。  
**处理**: 加 mask IoU 评估列。

### 5. load_sam2() 临时文件不清理（line 82-84）
```python
tmp_dir = tempfile.mkdtemp()
tmp_ckpt = os.path.join(tmp_dir, 'model_finetuned.pt')
# ← 从不清理 tmp_dir
```
**处理**: 加载后 `shutil.rmtree(tmp_dir)`。

---

## 🟢 Minor

### 6. circularity 没有上限
`4πA/P²` 理论上限 1.0，但数值噪声可能 >1。不影响功能。

### 7. 可视化 vis = img_np.copy() 对 6K 图占 ~100MB
只在 `--save-vis` 且前 10 张时触发，可接受。

---

## 修复计划
1. 修 mask_iou_cropped offset 解包（Critical）
2. 删死代码 mask_iou()
3. try_morphology_filters 加 mask IoU 评估
4. load_sam2 清理临时文件
