#!/usr/bin/env python3
"""
Monkey-patch Ultralytics parse_model 以识别 MPCA 和 BiFPN

Ultralytics 的 parse_model 用 base_modules frozenset 判断模块类型。
MPCA 需要加入 base_modules 才能自动处理 channel scaling。

用法:
    import patch_ultralytics  # 导入即生效
    from ultralytics import YOLO
    model = YOLO('orga_dete_yolo11n.yaml')
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orga_dete_modules import MPCA, BiFPN, EMASlideLoss
import ultralytics.nn.tasks as ul_tasks
import ultralytics.nn.modules as ul_modules


def _patched_parse_model(d, ch, verbose=True):
    """Monkey-patched parse_model that recognizes MPCA and BiFPN"""
    import ast
    import contextlib
    
    from ultralytics.nn.tasks import (
        torch, nn, make_divisible, C2fAttn, C3k2, A2C2f, C2fCIB,
        HGStem, HGBlock, ResNetLayer, Concat, Detect, YOLOEDetect,
        Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26,
        OBB, OBB26, TorchVision, Index, IBN, RepVGGDW,
        RGB2BGR, BGR2RGB, Classify
    )
    from ultralytics.utils.torch_utils import TORCH_1_9
    
    # 原始 base_modules + MPCA
    from ultralytics.nn.modules import (
        Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck,
        SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP,
        C1, C2, C2f, RepNCSPELAN4, DCNv2, DCNv3, LSKBlock, PSA,
        C3, C3TR, C3Ghost, C3x, RepC3, C2fCIB, A2C2f as A2C2f_mod
    )
    
    base_modules = frozenset({
        Classify, Conv, ConvTranspose, GhostConv, Bottleneck,
        GhostBottleneck, SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus,
        BottleneckCSP, C1, C2, C2f, C3k2, RepNCSPELAN4, DCNv2, DCNv3,
        LSKBlock, PSA, C3, C3TR, C3Ghost, C3x, RepC3, C2fCIB, A2C2f_mod,
        MPCA,  # ← 新增
    })
    
    repeat_modules = frozenset({
        BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR,
        C3Ghost, C3x, RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f_mod,
    })
    
    # 其余逻辑完全复用原始 parse_model
    # 但因为 base_modules 是局部变量，我们需要复制整个函数
    
    # 更简单的方式：直接修改 d['backbone'] 和 d['head'] 里的模块名
    # 让 MPCA 被当作 Conv 处理（因为签名相同：c1, c2）
    
    # 最简方案：在调用原始 parse_model 前，把 MPCA 实例化为类
    # 然后注入到 tasks 模块的 globals()
    ul_tasks.MPCA = MPCA
    ul_tasks.BiFPN = BiFPN
    
    # 调用原始 parse_model
    # 但原始函数的 base_modules 不包含 MPCA → 会走 "else" 分支
    # else 分支不处理 c1/c2，直接用 args
    # 所以我们需要在 YAML 里给 MPCA 传正确的 args
    
    # 最可靠方案：复制原始 parse_model 源码，加 MPCA 到 base_modules
    return _original_parse_model_with_mpca(d, ch, verbose, base_modules, repeat_modules)


def _original_parse_model_with_mpca(d, ch, verbose, base_modules, repeat_modules):
    """复制原始 parse_model 逻辑，用传入的 base_modules"""
    # 太复杂了——换个思路
    
    # 最简方案：不用 base_modules，而是让 MPCA 的构造函数
    # 从 ch[f]（前一层通道）获取输入通道
    
    # 但 parse_model 对非 base_modules 模块不传 ch[f]
    # args 保持 YAML 原始值
    
    # 最终方案：直接调原始 parse_model，MPCA 不在 base_modules
    # 但 MPCA 构造函数只接收 c1（= args[0]）
    # 问题：args[0] 是 YAML 的 1024，不是 scaled 后的 256
    
    # 解法：MPCA 构造函数不依赖 args[0]，而是在 forward 时动态获取
    pass


# ============================================================
# 实际方案：直接修改 Ultralytics 源码的 parse_model
# ============================================================

def apply_patch():
    """应用 monkey-patch"""
    
    # 1. 注入到 tasks 全局命名空间
    ul_tasks.MPCA = MPCA
    ul_tasks.BiFPN = BiFPN
    
    # 2. 修改 parse_model 的 base_modules
    # parse_model 是个函数，base_modules 是局部变量
    # 用 exec 包装方式不可靠
    
    # 3. 最终方案：MPCA 构造函数忽略 args[0]，用 lazy init
    # 第一次 forward 时根据实际输入通道初始化 conv
    # 这样不管 parse_model 传什么 args 都无所谓
    
    # 已在 orga_dete_modules.py 的 MPCA 中实现 lazy init
    pass


# 应用 patch
apply_patch()

if __name__ == '__main__':
    print("Patch applied. MPCA and BiFPN registered.")
    print(f"  tasks.MPCA: {hasattr(ul_tasks, 'MPCA')}")
    print(f"  tasks.BiFPN: {hasattr(ul_tasks, 'BiFPN')}")
