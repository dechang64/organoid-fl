"""
验证 extract_labels_new.py 的标注是否和 annotations.json 一致
在冬生本地运行:
    cd C:\\Users\\decha\\organoid-fl
    python scripts\\mouse_liver\\v2\\verify_labels.py
"""
import json
import os
from pathlib import Path

DATA_ROOT = Path(r"D:\datasets\mouse_liver_correct")


def yolo_from_bbox(bbox, img_w, img_h):
    """annotations.json bbox → YOLO format"""
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return (cx, cy, nw, nh)


def load_yolo_label(path):
    """读取 YOLO label 文件"""
    boxes = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return boxes


def boxes_match(b1, b2, tol=0.01):
    """两个 YOLO box 是否匹配 (容差 0.01)"""
    return all(abs(a - b) < tol for a, b in zip(b1, b2))


def check_batch(batch_name):
    """检查单个 batch 的 annotations.json vs YOLO labels"""
    batch_dir = DATA_ROOT / batch_name
    annot_json = batch_dir / 'annotations.json'
    label_dir = batch_dir / 'labels'

    if not annot_json.exists():
        print(f"  [ERROR] {annot_json} not found")
        return

    with open(annot_json, encoding='utf-8') as f:
        annotations = json.load(f)

    print(f"\n{'='*70}")
    print(f"  {batch_name}: {len(annotations)} images in annotations.json")
    print(f"  Labels dir: {label_dir} (exists={label_dir.exists()})")
    print(f"{'='*70}")

    n_match = 0
    n_mismatch = 0
    n_missing = 0

    for item in annotations:
        img_name = item['image']
        img_w, img_h = item['image_size']

        # annotations.json bboxes → YOLO
        expected_boxes = [yolo_from_bbox(b, img_w, img_h) for b in item['bboxes']]

        # 实际 YOLO label
        label_path = label_dir / (Path(img_name).stem + '.txt')
        if not label_path.exists():
            print(f"  [MISSING] {img_name} → {label_path.name}")
            n_missing += 1
            continue

        actual_boxes = load_yolo_label(label_path)

        # 匹配
        matched = 0
        unmatched_expected = []
        for eb in expected_boxes:
            found = False
            for ab in actual_boxes:
                if boxes_match(eb, ab):
                    found = True
                    matched += 1
                    break
            if not found:
                unmatched_expected.append(eb)

        unmatched_actual = []
        for ab in actual_boxes:
            found = False
            for eb in expected_boxes:
                if boxes_match(eb, ab):
                    found = True
                    break
            if not found:
                unmatched_actual.append(ab)

        status = "OK" if matched == len(expected_boxes) and not unmatched_actual else "MISMATCH"
        if status == "OK":
            n_match += 1
            print(f"  [OK]      {img_name}: {matched}/{len(expected_boxes)} boxes match")
        else:
            n_mismatch += 1
            print(f"  [MISMATCH] {img_name}: matched={matched}, expected={len(expected_boxes)}, actual={len(actual_boxes)}")
            if unmatched_expected:
                print(f"    Expected but not in label:")
                for b in unmatched_expected:
                    print(f"      {b}")
            if unmatched_actual:
                print(f"    In label but not expected:")
                for b in unmatched_actual:
                    print(f"      {b}")

    print(f"\n  Summary: {n_match} match, {n_mismatch} mismatch, {n_missing} missing")


def check_extract_pairing():
    """检查 extract_labels_new.py 的标注图-原图配对"""
    print(f"\n{'='*70}")
    print(f"  Check extract_labels_new.py pairing (annotated ↔ original)")
    print(f"{'='*70}")

    for batch_name in ['batch1', 'batch2', 'batch3']:
        batch_dir = DATA_ROOT / batch_name
        annot_dir = batch_dir / 'annotated'
        orig_dir = batch_dir / 'images'
        annot_json = batch_dir / 'annotations.json'

        if not all(d.exists() for d in [annot_dir, orig_dir, annot_json]):
            print(f"  [SKIP] {batch_name}: missing directories")
            continue

        with open(annot_json, encoding='utf-8') as f:
            annotations = json.load(f)

        # annotations.json 的正确对应
        correct_mapping = {}
        for item in annotations:
            correct_mapping[item['source_annotated']] = item['source_original']

        # extract_labels_new.py 的 sorted 索引配对
        annot_files = sorted(f for f in os.listdir(annot_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg')))
        orig_files = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg')))

        print(f"\n  {batch_name}: {len(annot_files)} annotated, {len(orig_files)} original")
        print(f"  Correct mapping (from annotations.json):")
        for i, (annot, orig) in enumerate(correct_mapping.items()):
            print(f"    {annot} ↔ {orig}")

        print(f"\n  extract_labels_new.py pairing (sorted index):")
        n = min(len(annot_files), len(orig_files))
        n_correct = 0
        for i in range(n):
            annot = annot_files[i]
            orig = orig_files[i]
            correct_orig = correct_mapping.get(annot, "???")
            match = "✓" if orig == correct_orig else "✗ WRONG!"
            if orig == correct_orig:
                n_correct += 1
            print(f"    [{i}] {annot} ↔ {orig} (correct: {correct_orig}) {match}")

        print(f"\n  Result: {n_correct}/{n} correct pairing")
        if n_correct < n:
            print(f"  ⚠️  {n - n_correct} pairs are WRONG! Labels are assigned to wrong images!")


def main():
    print(f"Data root: {DATA_ROOT}")

    # 1. Check annotations.json vs YOLO labels
    for batch in ['batch1', 'batch2', 'batch3']:
        check_batch(batch)

    # 2. Check extract_labels_new.py pairing
    check_extract_pairing()

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
