#!/usr/bin/env python3
"""
MultiOrg 数据探查脚本
— 列出每个 image_Y 目录下的文件，确认标注文件命名
— 统计 t0/t1_A/t1_B 标注可用性
— 统计 bbox 数量、大小分布

Usage (冬生本地 PowerShell):
    python explore_multiorg.py --src D:\\datasets\\mutliorg\\MultiOrg_v2

输出:
    multiorg_structure_report.json  — 完整结构报告
    multiorg_structure_report.txt   — 人类可读摘要
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter


def explore_image_dir(img_dir):
    """列出单个 image_Y 目录下的所有文件，分类统计。"""
    files = os.listdir(img_dir)
    tiff_files = [f for f in files if f.lower().endswith(('.tiff', '.tif'))]
    json_files = [f for f in files if f.lower().endswith('.json')]
    other_files = [f for f in files if not f.lower().endswith(('.tiff', '.tif', '.json'))]

    return {
        'tiff': tiff_files,
        'json': json_files,
        'other': other_files,
    }


def explore_split(src_dir, split):
    """探查 train 或 test split。"""
    split_dir = os.path.join(src_dir, split)
    if not os.path.isdir(split_dir):
        return None

    report = {
        'split': split,
        'classes': {},
        'total_images': 0,
        'annotation_patterns': Counter(),
        'sample_files': [],
    }

    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_info = {
            'plates': 0,
            'images': 0,
            'annotation_types': Counter(),
            'sample_dirs': [],
        }

        for plate_name in sorted(os.listdir(class_dir)):
            plate_dir = os.path.join(class_dir, plate_name)
            if not os.path.isdir(plate_dir):
                continue
            class_info['plates'] += 1

            for img_dir_name in sorted(os.listdir(plate_dir)):
                img_dir = os.path.join(plate_dir, img_dir_name)
                if not os.path.isdir(img_dir):
                    continue
                class_info['images'] += 1
                report['total_images'] += 1

                file_info = explore_image_dir(img_dir)

                # 统计 JSON 文件命名模式
                for jf in file_info['json']:
                    # 提取标注者信息
                    lower = jf.lower()
                    if 'annotator_a' in lower:
                        class_info['annotation_types']['annotator_a'] += 1
                        report['annotation_patterns']['annotator_a'] += 1
                    elif 'annotator_b' in lower:
                        class_info['annotation_types']['annotator_b'] += 1
                        report['annotation_patterns']['annotator_b'] += 1
                    elif 'annotator_c' in lower:
                        class_info['annotation_types']['annotator_c'] += 1
                        report['annotation_patterns']['annotator_c'] += 1
                    elif 't0' in lower:
                        class_info['annotation_types']['t0'] += 1
                        report['annotation_patterns']['t0'] += 1
                    elif 't1' in lower:
                        if '_a' in lower or 'annotator_a' in lower:
                            class_info['annotation_types']['t1_a'] += 1
                            report['annotation_patterns']['t1_a'] += 1
                        elif '_b' in lower or 'annotator_b' in lower:
                            class_info['annotation_types']['t1_b'] += 1
                            report['annotation_patterns']['t1_b'] += 1
                        else:
                            class_info['annotation_types']['t1_unknown'] += 1
                            report['annotation_patterns']['t1_unknown'] += 1
                    else:
                        class_info['annotation_types']['other_json'] += 1
                        report['annotation_patterns']['other_json'] += 1

                # 前 5 个目录的完整文件列表作为样本
                if len(class_info['sample_dirs']) < 5:
                    class_info['sample_dirs'].append({
                        'dir': img_dir_name,
                        'plate': plate_name,
                        'files': file_info,
                    })
                if len(report['sample_files']) < 10:
                    report['sample_files'].append({
                        'split': split,
                        'class': class_name,
                        'plate': plate_name,
                        'dir': img_dir_name,
                        'files': file_info,
                    })

        report['classes'][class_name] = class_info

    return report


def main():
    parser = argparse.ArgumentParser(description='MultiOrg Data Explorer')
    parser.add_argument(
        '--src',
        default=r'D:\datasets\mutliorg\MultiOrg_v2',
        help='Source MultiOrg_v2 directory'
    )
    args = parser.parse_args()

    print(f"Source: {args.src}")
    print("=" * 70)

    full_report = {
        'source': args.src,
        'splits': {},
    }

    for split in ['train', 'test']:
        print(f"\n--- Exploring {split} ---")
        report = explore_split(args.src, split)
        if report is None:
            print(f"  [WARN] {split} directory not found")
            continue

        full_report['splits'][split] = report

        print(f"  Total images: {report['total_images']}")
        for cls, info in report['classes'].items():
            print(f"  {cls}: {info['plates']} plates, {info['images']} images")
            print(f"    Annotation types: {dict(info['annotation_types'])}")

        print(f"\n  Annotation patterns across {split}:")
        for pattern, count in report['annotation_patterns'].most_common():
            print(f"    {pattern}: {count}")

    # 打印样本文件列表
    print("\n" + "=" * 70)
    print("SAMPLE DIRECTORIES (first 10):")
    print("=" * 70)
    for s in full_report.get('sample_files', full_report['splits'].get('train', {}).get('sample_files', [])):
        print(f"\n  {s['split']}/{s['class']}/{s['plate']}/{s['dir']}/")
        for ftype, files in s['files'].items():
            for f in files:
                print(f"    [{ftype}] {f}")

    # 保存 JSON 报告
    json_path = 'multiorg_structure_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n\nFull report saved: {json_path}")

    # 保存人类可读摘要
    txt_path = 'multiorg_structure_report.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"MultiOrg Structure Report\n{'='*70}\n")
        f.write(f"Source: {args.src}\n\n")
        for split, report in full_report['splits'].items():
            f.write(f"\n--- {split} ---\n")
            f.write(f"Total images: {report['total_images']}\n")
            for cls, info in report['classes'].items():
                f.write(f"  {cls}: {info['plates']} plates, {info['images']} images\n")
                f.write(f"    Annotations: {dict(info['annotation_types'])}\n")
            f.write(f"  Patterns: {dict(report['annotation_patterns'])}\n")

            f.write(f"\n  Sample directories:\n")
            for s in report['sample_files'][:5]:
                f.write(f"    {s['split']}/{s['class']}/{s['plate']}/{s['dir']}/\n")
                for ftype, files in s['files'].items():
                    for fn in files:
                        f.write(f"      [{ftype}] {fn}\n")
    print(f"Readable report saved: {txt_path}")


if __name__ == '__main__':
    main()
