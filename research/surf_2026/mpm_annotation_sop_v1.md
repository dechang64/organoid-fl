# MPM 类器官图像标注规范 v1.0

> **日期**：2026-07-15
> **作者**：曼卿（Organoid-FL Agent）
> **目的**：为瑞金医工交叉项目 20 例 MPM PDO 图像标注提供操作规范
> **适用范围**：MPM 类器官明场图像 + H&E + IHC，亦适用于其他类器官跨域迁移
> **状态**：v1.0 试用版，首批 5 例标注后根据反馈修订

---

## 一、设计原则

1. **两阶段标注**：先 4 点位置标注（快速、所有样本）→ 后红折线轮廓（精细、子集）
2. **多人交叉标注**：每张图至少 2 人独立标注，计算 kappa 一致性
3. **质量分级**：A/B/C 三级，与 MultiOrg Annotator_A/B 体系兼容
4. **坐标系统一**：napari (row, col) = (y, x)，与 MultiOrg 兼容
5. **JSON 格式标准化**：napari-organoid-counter 兼容输出

---

## 二、标注工具

### 2.1 主工具：napari + napari-organoid-counter

**安装**：
```bash
conda create -n organoid-anno python=3.10
conda activate organoid-anno
pip install napari[all] napari-organoid-counter
```

**理由**：
- napari 是 Helmholtz AI 开发的专业生物图像标注工具
- napari-organoid-counter 插件专门为类器官计数/标注设计
- 已在 MultiOrg 数据集上验证可用（我们团队 2026-06 使用过）
- 输出 JSON 格式标准化，与 organoid-fl 平台兼容

### 2.2 辅助工具

- **ImageJ/FIJI**：用于查看 16-bit TIFF（napari 也可，但 ImageJ 更稳）
- **QuPath**：用于 H&E WSI 标注（如果需要全切片标注）
- **CVAT**：备选，多人协作更友好但需要服务器部署

---

## 三、坐标系约定（重要！）

### 3.1 napari 坐标系

napari 使用 **(row, col) = (y, x)** 坐标系，**不是** (x, y)！

这是 2026-06 我们团队犯过的错误：把 MultiOrg 的 (row, col) 当成 (x, y)，导致 100 epoch 训练 mAP=0，浪费 3 小时 GPU 时间。

### 3.2 JSON 输出格式

```json
{
  "image_id": "MPM_P001_d7_brightfield",
  "image_size": [5720, 6385],  // [H, W]
  "image_dtype": "uint8",
  "annotations": {
    "1": {
      "polygon": [[y1, x1], [y2, x2], [y3, x3], [y4, x4]],  // 4 点多边形
      "subtype": "epithelioid",  // epithelioid/sarcomatoid/biphasic
      "quality": "A",            // A/B/C
      "annotator": "annotator_a",
      "confidence": 0.95
    },
    "2": {
      "polygon": [[y1, x1], ...],
      "subtype": "biphasic",
      "quality": "B",
      "annotator": "annotator_b",
      "confidence": 0.85
    }
  },
  "metadata": {
    "patient_id": "P001",
    "specimen_type": "pleural_effusion",  // pleural_effusion/ascites/biopsy/surgical
    "day": 7,                              // 培养第几天
    "modality": "brightfield",             // brightfield/H&E/IHC/IF
    "magnification": "10x",
    "scale_bar_um": 100,
    "staining": null,                      // brightfield 为 null
    "drug_treatment": null                 // 对照组为 null
  }
}
```

### 3.3 验证坐标系正确性

```python
import json
import numpy as np
from PIL import Image

# 加载标注
with open('MPM_P001_d7_brightfield.json') as f:
    ann = json.load(f)

# 加载图像
img = Image.open('MPM_P001_d7_brightfield.png')
W, H = img.size  # PIL 返回 (W, H)

# 验证：所有 x (col) < W, 所有 y (row) < H
for k, v in ann['annotations'].items():
    poly = np.array(v['polygon'])
    ys = poly[:, 0]  # row = y
    xs = poly[:, 1]  # col = x
    assert ys.max() < H, f"y={ys.max()} 超出图像高度 H={H}"
    assert xs.max() < W, f"x={xs.max()} 超出图像宽度 W={W}"
    # 如果 y > H 或 x > W，说明坐标系错了（当成了 (x,y)）
print("✓ 坐标系正确：(row, col) = (y, x)")
```

---

## 四、标注协议

### 4.1 第一阶段：4 点多边形位置标注（所有样本）

**适用**：所有 20 例 MPM PDO 图像，每个时间点（day 0/3/7/14）

**操作**：
1. 在 napari 中打开图像
2. 用 napari-organoid-counter 插件
3. 对每个 organoid 点击 4 个角点（左上、右上、右下、左下）
4. 4 点形成最小外接矩形（旋转对齐）
5. 自动保存 JSON

**时间预算**：
- 每张图 5-15 个 organoid
- 每 organoid 约 10 秒
- 每张图 1-3 分钟
- 20 例 × 4 时间点 × 3 视野 = 240 张图，约 6-12 小时

**质量要求**：
- 4 点必须紧密贴合 organoid 边界
- 漏标率 < 5%（用 SAM2 zero-shot 辅助检查漏标）
- 误标率 < 10%（背景碎片不算 organoid）

### 4.2 第二阶段：红折线精确轮廓（子集）

**适用**：从 20 例中选 5 例（每个亚型 + 不同天数）做精细标注

**操作**：
1. 在 napari 中加载已有 4 点标注作为参考
2. 用 polygon tool 沿 organoid 真实边界画折线
3. 折线点数 20-50 个（取决于 organoid 复杂度）
4. 标注每个 organoid 的精确边界

**时间预算**：
- 每 organoid 3-5 分钟
- 每张图 15-60 分钟
- 5 例 × 4 时间点 × 3 视野 = 60 张图，约 15-60 小时

**质量要求**：
- 需要 1 位以上的临床医生参与（理解 MPM 形态学）
- 双人独立标注 + 第三人仲裁
- Dice 系数 > 0.85（双人一致性）

### 4.3 亚型标注

**适用**：所有 organoid（4 点和红折线阶段都标注）

**类别**：

| 亚型 | 形态特征 | 比例 |
|------|---------|------|
| **epithelioid** | 实心球状、边界清楚、细胞紧密 | ~60% |
| **sarcomatoid** | 梭形、弥散生长、边界模糊 | ~15% |
| **biphasic** | 混合上皮+肉瘤特征 | ~25% |

**标注要求**：
- 如果不确定，标 "biphasic"（最保守）
- H&E 切片可作为参考（如果同步采集）
- IHC 标志物（CK5/6+, D2-40+, WT-1+）可辅助确认

### 4.4 质量分级

| 等级 | 标准 | 用途 |
|------|------|------|
| **A** | 图像清晰、organoid 边界明显、无重叠 | 训练 + 测试 |
| **B** | 图像略模糊、organoid 部分边界不清、少量重叠 | 训练 |
| **C** | 图像模糊、organoid 严重重叠、难以辨认 | 仅做弱监督 |

**每张图整体评级**，不是每个 organoid 评级。

---

## 五、多标注者一致性评估

### 5.1 必须双人独立标注

每张图至少 2 人独立标注（annotator_a, annotator_b），不交流。

### 5.2 一致性指标

**位置标注（4 点）**：
- IoU > 0.5 视为同一 organoid
- 计算 F1-score（precision, recall）
- F1 > 0.85 = 一致性好
- F1 0.7-0.85 = 一致性中等
- F1 < 0.7 = 需要仲裁

**轮廓标注（红折线）**：
- 计算 Dice 系数
- Dice > 0.85 = 一致性好
- 第三人仲裁后取平均值

**亚型标注**：
- Cohen's kappa
- kappa > 0.8 = 一致性极好
- kappa 0.6-0.8 = 一致性较好
- kappa < 0.6 = 需要重新培训

### 5.3 仲裁流程

1. 两标注者独立完成
2. 系统自动计算 F1/Dice/kappa
3. F1 < 0.85 的图交给第三人（资深者）仲裁
4. 仲裁结果作为 ground truth
5. 仲裁过程记录争议点（用于培训）

### 5.4 期望水平

申请书声称 kappa=0.62（中等一致），我们的目标是提升到 **kappa > 0.8**（较好一致）。

---

## 六、漏标检测（SAM2 辅助）

### 6.1 SAM2 zero-shot 漏标检测

对每张标注完的图：
1. 用 SAM2 自动分割所有 organoid-like 区域
2. 对比人工标注，找 SAM2 分割但人工漏标的 organoid
3. 输出候选漏标列表
4. 人工复核确认

### 6.2 实施代码模板

```python
import json
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 加载 SAM2
predictor = SAM2ImagePredictor(build_sam2('sam2_hiera_s', 'sam2_hiera_small.pt'))

# 加载图像和标注
img = np.array(Image.open('MPM_P001_d7_brightfield.png').convert('RGB'))
with open('MPM_P001_d7_brightfield.json') as f:
    ann = json.load(f)

# SAM2 自动分割（automatic mode）
predictor.set_image(img)
masks, scores, _ = predictor.predict()  # 全图自动分割

# 找漏标
manual_polys = [v['polygon'] for v in ann['annotations'].values()]
missed = []
for mask, score in zip(masks, scores):
    if score < 0.5: continue
    # 计算这个 mask 和已有标注的 IoU
    matched = False
    for poly in manual_polys:
        # ... 计算 IoU
        if iou > 0.5:
            matched = True
            break
    if not matched:
        missed.append(mask)

print(f"潜在漏标: {len(missed)} 个 organoid，请人工复核")
```

### 6.3 误标检测

对每个标注：
1. 用 SAM2 在该 4 点位置做分割
2. 看 SAM2 mask 是否在标注内
3. SAM2 mask 跑出去 = 可能误标（背景被当成 organoid）

---

## 七、与 organoid-fl 平台的集成

### 7.1 JSON 转换为 YOLO 格式

标注完成后，转 YOLO 格式训练：

```python
import json
import numpy as np
from PIL import Image

def napari_to_yolo(json_path, image_path, output_label_path):
    """napari JSON → YOLO 单类检测标签"""
    with open(json_path) as f:
        ann = json.load(f)
    
    img = Image.open(image_path)
    W, H = img.size
    
    yolo_lines = []
    for k, v in ann['annotations'].items():
        poly = np.array(v['polygon'])  # (4, 2) in (y, x)
        ys = poly[:, 0]  # row = y
        xs = poly[:, 1]  # col = x
        # Bounding box (axis-aligned)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        # YOLO format: class x_center y_center width height (normalized)
        x_center = (x_min + x_max) / 2 / W
        y_center = (y_min + y_max) / 2 / H
        width = (x_max - x_min) / W
        height = (y_max - y_min) / H
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

# 用法
napari_to_yolo(
    'MPM_P001_d7_brightfield.json',
    'MPM_P001_d7_brightfield.png',
    'MPM_P001_d7_brightfield.txt'
)
```

### 7.2 数据集组织

```
mpm_organoid_dataset/
├── images/
│   ├── train/   (16 例 × 4 时间点 = 64 张)
│   └── val/     (4 例 × 4 时间点 = 16 张)
├── labels/
│   ├── train/   (对应 .txt)
│   └── val/
├── annotations_json/   (napari 原始输出，保留用于精标注升级)
├── metadata.csv        (patient_id, subtype, day, drug, clinical_info)
└── data.yaml           (YOLO 训练配置)
```

### 7.3 data.yaml 模板

```yaml
path: ./mpm_organoid_dataset
train: images/train
val: images/val
test: images/val  # 20 例太少，val = test
names:
  0: organoid  # 单类检测，所有亚型作为同一类
```

---

## 八、与申请书指标的对应

| 申请书声称 | 标注规范对应 | 验证方法 |
|-----------|------------|---------|
| 灵敏度 ≥89% | 4 点位置标注 + SAM2 漏标检查 | val 集 recall |
| 特异性 ≥91% | 4 点位置标注 + 误标检查 | val 集 precision |
| 标注 kappa > 0.8 | 双人独立 + 第三人仲裁 | Cohen's kappa |
| 准确率 ≥80% | YOLO 训练 + val 评估 | val 集 mAP50 |
| 20 例真实 MPM | 20 例 × 4 时间点 × 3 视野 = 240 张 | — |

---

## 九、时间线（与瑞金项目对齐）

| 时间 | 任务 | 产出 |
|------|------|------|
| 2026.Q4 | 项目批准 + 标注工具培训 | 标注 SOP v1.0（本文档） |
| 2027.Q1 | 首批 5 例 MPM PDO 4 点标注 | 60 张标注 JSON + kappa 评估 |
| 2027.Q1 | 5 例红折线精标注（医生参与） | 60 张精标注 + Dice 评估 |
| 2027.Q2 | 20 例全部 4 点标注完成 | 240 张 + 完整 kappa |
| 2027.Q2 | SAM2 漏标/误标检测 | 标注质量报告 |
| 2027.Q3 | YOLO 格式转换 + 训练 RF-DETR | val mAP50 评估 |
| 2027.Q4 | 数据集 v1 发布（仅内部） | mpm_organoid_dataset v1.0 |

---

## 十、风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 谢梦燕团队没时间标注 | 中 | 提供 SAM2 辅助 + 我们团队远程协助 |
| 标注 kappa < 0.7 | 中 | 第三人仲裁 + 重新培训 |
| 20 例太少训不出模型 | 高 | 自蒸馏 + 联合训练（鼠肝 + MultiOrg + MPM） |
| MPM 亚型难以辨认 | 中 | H&E/IHC 辅助 + 临床医生培训 |
| 16-bit TIFF 读取崩溃 | 低 | 使用 tifffile 代替 PIL（2026-06 教训） |

---

## 十一、附录：napari-organoid-counter 使用指南

### 11.1 安装

```bash
pip install napari[all] napari-organoid-counter
```

### 11.2 启动

```bash
napari
```

### 11.3 加载图像

1. File → Open → 选择图像
2. 在右侧 Plugins 找到 "Organoid Counter"

### 11.4 标注流程

1. 点击 "Start Annotating"
2. 对每个 organoid 点击 4 个角点
3. 自动保存到 JSON

### 11.5 输出 JSON 位置

默认输出在图像同目录下，文件名 `[image_name]_Annotator_[A/B].json`

### 11.6 常见问题

- 16-bit TIFF 读取问题：先用 `tifffile.imread()` 转换为 8-bit PNG
- 大图（>10000×10000）卡顿：用 napari 的 multiresolution loading
- 多人协作：用 napari-organoid-counter 的 multi-annotator mode

---

## 十二、参考文献

1. Helmholtz AI. napari-organoid-counter. GitHub.
2. MultiOrg dataset. NeurIPS 2024.
3. Sci Rep 2025. Liu Y et al. MM PDO-T cell co-culture.
4. BMC Cancer 2023. Ito et al. Matrigel-based MM organoid.
5. Sci Rep 2018. Mazzocchi et al. 3D mesothelioma tumor organoids.
6. 冬生团队 2026-06 MultiOrg 标注实战经验（napari 坐标系教训）

---

**文档版本**：v1.0
**下一步**：等瑞金项目批准后，组织 1-2 次线上培训（2 小时/次），让谢梦燕团队试用 SOP
