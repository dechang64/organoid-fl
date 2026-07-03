r"""
诊断脚本: 检查 YOLO model.save() 后 ckpt 是 fused 还是 unfused
在冬生本地运行, 输出结果给我看

Usage:
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate
    python scripts\mouse_liver\diagnose_fuse.py
"""
import os, sys, json, torch
from ultralytics import YOLO

print(f"Ultralytics version: {__import__('ultralytics').__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")
print()

# 1. 加载 yolo12n.pt (80类 COCO)
print("="*50)
print("1. YOLO('yolo12n.pt')")
m = YOLO('yolo12n.pt')
sd = m.model.state_dict()
has_bn = any('bn.weight' in k for k in sd)
has_conv_bias = any('.conv.bias' in k for k in sd)
print(f"   keys: {len(sd)}")
print(f"   has bn.weight: {has_bn}")
print(f"   has conv.bias: {has_conv_bias}")
print(f"   nc: {m.model.nc}")
print(f"   is_fused: {m.model.is_fused()}")
print(f"   → {'FUSED' if has_conv_bias and not has_bn else 'UNFUSED'}")

# 2. 训练 1 epoch
print("\n" + "="*50)
print("2. Training 1 epoch (B1 data)...")
data_yaml = r"D:\datasets\mouse_liver_data\batch1\data.yaml"
if not os.path.exists(data_yaml):
    print(f"   ERROR: {data_yaml} not found!")
    sys.exit(1)

# 清除 labels.cache
cache = os.path.join(os.path.dirname(data_yaml), 'labels.cache')
if os.path.exists(cache):
    os.remove(cache)

m.train(data=data_yaml, epochs=1, imgsz=640, batch=4,
        device='cuda', workers=0, cache=False,
        project='runs/diag', name='fuse_test', exist_ok=True, verbose=False)

# 3. 训练后 state_dict
print("\n" + "="*50)
print("3. After train — model.model.state_dict()")
sd2 = m.model.state_dict()
has_bn2 = any('bn.weight' in k for k in sd2)
has_conv_bias2 = any('.conv.bias' in k for k in sd2)
print(f"   keys: {len(sd2)}")
print(f"   has bn.weight: {has_bn2}")
print(f"   has conv.bias: {has_conv_bias2}")
print(f"   is_fused: {m.model.is_fused()}")
print(f"   → {'FUSED' if has_conv_bias2 and not has_bn2 else 'UNFUSED'}")

# 4. model.save()
print("\n" + "="*50)
print("4. model.save('test_save.pt')")
m.save('test_save.pt')

# 5. YOLO('test_save.pt') 加载
print("\n" + "="*50)
print("5. YOLO('test_save.pt') reload")
m2 = YOLO('test_save.pt')
sd3 = m2.model.state_dict()
has_bn3 = any('bn.weight' in k for k in sd3)
has_conv_bias3 = any('.conv.bias' in k for k in sd3)
print(f"   keys: {len(sd3)}")
print(f"   has bn.weight: {has_bn3}")
print(f"   has conv.bias: {has_conv_bias3}")
print(f"   is_fused: {m2.model.is_fused()}")
print(f"   → {'FUSED' if has_conv_bias3 and not has_bn3 else 'UNFUSED'}")

# 6. torch.load 直接看 ckpt
print("\n" + "="*50)
print("6. torch.load('test_save.pt') raw")
ckpt = torch.load('test_save.pt', map_location='cpu', weights_only=False)
print(f"   ckpt keys: {list(ckpt.keys())}")
ckpt_model = ckpt['model']
ckpt_sd = ckpt_model.state_dict() if hasattr(ckpt_model, 'state_dict') else ckpt_model
has_bn4 = any('bn.weight' in k for k in ckpt_sd)
has_conv_bias4 = any('.conv.bias' in k for k in ckpt_sd)
print(f"   ckpt['model'] type: {type(ckpt_model)}")
print(f"   has bn.weight: {has_bn4}")
print(f"   has conv.bias: {has_conv_bias4}")
print(f"   → {'FUSED' if has_conv_bias4 and not has_bn4 else 'UNFUSED'}")

# 7. 关键对比
print("\n" + "="*50)
print("7. KEY COMPARISON")
print(f"   After train (sd2):  {'FUSED' if has_conv_bias2 and not has_bn2 else 'UNFUSED'} ({len(sd2)} keys)")
print(f"   After save+load (sd3): {'FUSED' if has_conv_bias3 and not has_bn3 else 'UNFUSED'} ({len(sd3)} keys)")
print(f"   ckpt raw (ckpt_sd): {'FUSED' if has_conv_bias4 and not has_bn4 else 'UNFUSED'} ({len(ckpt_sd)} keys)")
print()
if has_bn2 != has_bn3:
    print("   ⚠️  model.save() 改变了 fused 状态!")
    print("   → FL 脚本不能用 model.model.state_dict() 取权重")
    print("   → 必须用 model.save() + torch.load 取权重保持一致")
elif has_bn2 == has_bn3:
    print("   ✅ model.save() 保持 fused 状态一致")
    print("   → FL 脚本可以用 model.model.state_dict() 取权重")

# 清理
del m, m2
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("\nDone! 请把以上输出发给我。")
