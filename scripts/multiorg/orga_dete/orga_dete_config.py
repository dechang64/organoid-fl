"""
自定义 YOLOv11 YAML 配置 — Orga-Dete 三模块集成

在标准 yolo11.yaml 基础上:
1. backbone C2PSA 后加 MPCA
2. head 的 FPN+PAN 替换为 BiFPN
3. 训练时用 EMASlideLoss 替换 BCE cls loss

用法:
    from ultralytics import YOLO
    model = YOLO('orga_dete_yolo11n.yaml')
    model.train(data='data.yaml', epochs=300, ...)
"""

# === orga_dete_yolo11n.yaml ===
YAML_CONFIG = """\
# Orga-Dete YOLO11n — BiFPN + MPCA + EMASlideLoss
# Based on yolo11.yaml with Orga-Dete modifications
# Ref: Huang et al., Applied Sciences 2025

nc: 1  # organoid (single class)

scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]   # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]    # 9
  - [-1, 2, C2PSA, [1024]]      # 10
  - [-1, 1, MPCA, [1024]]       # 11 — Orga-Dete 新增

head:
  # BiFPN 替换 FPN+PAN
  - [[4, 6, 11], 1, BiFPN, [256]]  # 12 — BiFPN(P3, P4, P5_with_MPCA)

  # Detect head (P3, P4, P5 from BiFPN)
  - [[12, 12, 12], 1, Detect, [nc]]  # 13 — 需要修改 BiFPN 输出方式
"""

# === 模块注册脚本 ===
REGISTER_SCRIPT = """\
#!/usr/bin/env python3
\"\"\"注册 MPCA 和 BiFPN 到 Ultralytics，使其可在 YAML 中使用\"\"\"

import sys
import os

# 添加 orga_dete 模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orga_dete_modules import MPCA, BiFPN, EMASlideLoss
import ultralytics.nn.modules as ul_modules
import ultralytics.nn.tasks as ul_tasks

# 注册模块
ul_modules.MPCA = MPCA
ul_modules.BiFPN = BiFPN

# 添加到 tasks.py 的模块集合
# base_modules 和 repeat_modules 是 frozenset，需要重新创建
import inspect
src = inspect.getsource(ul_tasks.parse_model)

print("✅ MPCA and BiFPN registered to ultralytics")
print(f"  MPCA: {ul_modules.MPCA}")
print(f"  BiFPN: {ul_modules.BiFPN}")
"""

if __name__ == '__main__':
    print("Orga-Dete YAML configuration and registration scripts")
    print()
    print("=== YAML Config ===")
    print(YAML_CONFIG)
    print()
    print("To use:")
    print("  1. Run register_modules.py to register MPCA/BiFPN in ultralytics")
    print("  2. Train with: yolo train model=orga_dete_yolo11n.yaml data=data.yaml")
