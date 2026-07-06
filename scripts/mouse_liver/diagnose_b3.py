r"""
B3 数据诊断脚本 — 检查 B3 数据/标注质量

B3 100ep train mAP50=9%, 完全没学到。检查:
  1. 图片尺寸是否和 B1/B2 一致
  2. 标注格式是否正确 (YOLO: class cx cy w h, 归一化)
  3. 标注坐标范围 [0,1]
  4. 标注框数量/大小分布
  5. 图片和标签是否对齐
  6. 标注是否全 0 或异常

Usage:
  cd C:\Users\decha\organoid-fl
  python scripts\mouse_liver\diagnose_b3.py
"""
import os, sys, json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fl_sequential import BATCH_DIRS, VAL_INDICES


def check_batch(batch_name):
    """检查一个 batch 的数据质量"""
    data_dir = BATCH_DIRS[batch_name]
    img_dir = os.path.join(data_dir, 'images')
    lbl_dir = os.path.join(data_dir, 'labels')

    print(f"\n{'='*60}")
    print(f"检查 {batch_name}: {data_dir}")
    print(f"{'='*60}")

    # 收集图片和标签
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    lbl_files = sorted([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])

    print(f"图片数: {len(img_files)}")
    print(f"标签数: {len(lbl_files)}")

    # 检查图片和标签是否对齐
    img_names = set(os.path.splitext(f)[0] for f in img_files)
    lbl_names = set(os.path.splitext(f)[0] for f in lbl_files)
    missing_lbl = img_names - lbl_names
    missing_img = lbl_names - img_names
    if missing_lbl:
        print(f"⚠️  图片无标签: {sorted(missing_lbl)}")
    if missing_img:
        print(f"⚠️  标签无图片: {sorted(missing_img)}")
    if not missing_lbl and not missing_img:
        print("✅ 图片和标签完全对齐")

    # 检查图片尺寸
    try:
        from PIL import Image
        sizes = defaultdict(int)
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    sizes[img.size] += 1
            except Exception as e:
                print(f"⚠️  无法打开 {img_file}: {e}")
        print(f"\n图片尺寸分布:")
        for size, count in sorted(sizes.items(), key=lambda x: -x[1]):
            print(f"  {size[0]}×{size[1]}: {count} 张")
    except ImportError:
        print("PIL 不可用，跳过图片尺寸检查")

    # 检查标签格式和坐标范围
    total_boxes = 0
    empty_labels = 0
    out_of_range = 0
    class_ids = defaultdict(int)
    box_sizes = []
    bad_lines = []

    for lbl_file in lbl_files:
        lbl_path = os.path.join(lbl_dir, lbl_file)
        with open(lbl_path, 'r') as f:
            lines = f.readlines()

        if not lines or all(l.strip() == '' for l in lines):
            empty_labels += 1
            continue

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                bad_lines.append(f"  {lbl_file} line {i}: '{line.strip()}' (字段<5)")
                continue

            try:
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                bad_lines.append(f"  {lbl_file} line {i}: '{line.strip()}' (数值解析失败)")
                continue

            class_ids[cls] += 1
            total_boxes += 1

            # 检查坐标范围
            coords = [cx, cy, w, h]
            if any(c < 0 or c > 1 for c in coords):
                out_of_range += 1
                if out_of_range <= 5:
                    print(f"⚠️  超范围: {lbl_file} line {i}: cls={cls} cx={cx} cy={cy} w={w} h={h}")

            # 检查框大小
            if w > 0 and h > 0:
                box_sizes.append((w, h, w * h))
            elif w == 0 or h == 0:
                print(f"⚠️  零面积框: {lbl_file} line {i}: w={w} h={h}")

    print(f"\n标签统计:")
    print(f"  总框数: {total_boxes}")
    print(f"  空标签文件: {empty_labels}")
    print(f"  超范围坐标: {out_of_range}")
    print(f"  类别分布: {dict(class_ids)}")
    if bad_lines:
        print(f"  格式错误行 ({len(bad_lines)}):")
        for b in bad_lines[:10]:
            print(f"    {b}")
        if len(bad_lines) > 10:
            print(f"    ... 还有 {len(bad_lines)-10} 行")

    if box_sizes:
        ws = [s[0] for s in box_sizes]
        hs = [s[1] for s in box_sizes]
        areas = [s[2] for s in box_sizes]
        print(f"\n框大小分布 (归一化):")
        print(f"  W: min={min(ws):.4f} max={max(ws):.4f} mean={sum(ws)/len(ws):.4f}")
        print(f"  H: min={min(hs):.4f} max={max(hs):.4f} mean={sum(hs)/len(hs):.4f}")
        print(f"  Area: min={min(areas):.6f} max={max(areas):.6f} mean={sum(areas)/len(areas):.6f}")

        # 检查是否有异常小的框
        tiny = [a for a in areas if a < 0.001]  # <0.1% of image
        huge = [a for a in areas if a > 0.5]    # >50% of image
        if tiny:
            print(f"  ⚠️  极小框 (<0.1%): {len(tiny)} 个")
        if huge:
            print(f"  ⚠️  极大框 (>50%): {len(huge)} 个")

    # 每张图的框数分布
    boxes_per_img = []
    for lbl_file in lbl_files:
        lbl_path = os.path.join(lbl_dir, lbl_file)
        with open(lbl_path, 'r') as f:
            n = sum(1 for line in f if line.strip())
        boxes_per_img.append(n)

    if boxes_per_img:
        print(f"\n每图框数分布:")
        print(f"  min={min(boxes_per_img)} max={max(boxes_per_img)} mean={sum(boxes_per_img)/len(boxes_per_img):.1f}")
        # 直方图
        from collections import Counter
        hist = Counter()
        for n in boxes_per_img:
            if n == 0: hist['0'] += 1
            elif n <= 2: hist['1-2'] += 1
            elif n <= 5: hist['3-5'] += 1
            elif n <= 10: hist['6-10'] += 1
            else: hist['10+'] += 1
        for k in ['0', '1-2', '3-5', '6-10', '10+']:
            if hist[k]:
                print(f"    {k} 框: {hist[k]} 张")

    return total_boxes, empty_labels, out_of_range


def compare_batches():
    """对比三个 batch 的关键指标"""
    print(f"\n{'='*60}")
    print("三批数据对比")
    print(f"{'='*60}")

    results = {}
    for batch in ['b1', 'b2', 'b3']:
        total, empty, oor = check_batch(batch)
        results[batch] = {'total': total, 'empty': empty, 'oor': oor}

    print(f"\n{'='*60}")
    print("汇总对比")
    print(f"{'='*60}")
    print(f"{'Batch':<8} {'总框数':<10} {'空标签':<10} {'超范围':<10}")
    for batch in ['b1', 'b2', 'b3']:
        r = results[batch]
        flag = ' ⚠️' if r['oor'] > 0 or r['empty'] > 0 else ''
        print(f"{batch:<8} {r['total']:<10} {r['empty']:<10} {r['oor']:<10}{flag}")


if __name__ == '__main__':
    compare_batches()
