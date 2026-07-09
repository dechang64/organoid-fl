r"""
Phase 2: VLM 语义确认 — GLM-4V 评估 detection 质量

 Usage (MultiOrg):
     cd C:\Users\decha\organoid-fl
     python scripts\phase2_vlm_verification.py ^
         --json results\multiorg_sam2_zeroshot\multiorg_sam2_results.json ^
         --src D:\datasets\mutliorg\MultiOrg_v2\test ^
         --dst results\phase2_vlm ^
         --dataset multiorg ^
         --max-tp 10 --max-fp 10

 Usage (Mouse Liver B2):
     python scripts\phase2_vlm_verification.py ^
         --json runs\mouse_liver_phase1\phase1_results.json ^
         --src C:\Users\decha\mouse_liver_correct\B2\原始 ^
         --dst results\phase2_vlm_mouse ^
         --dataset mouse_liver ^
         --max-tp 6 --max-fp 3

 Output:
     results/phase2_vlm/vlm_results.json  — per-detection VLM scores
     results/phase2_vlm/vlm_summary.json  — TP/FP separation analysis
     results/phase2_vlm/crops/            — 临时裁剪图 (可删除)
     results/phase2_vlm/vlm_cache.json    — VLM 响应缓存

 Literature:
     - CTM (Sakana AI, NeurIPS 2025 Spotlight): internal ticks = iterative refinement
     - Generate, but Verify (arXiv 2504.13169): VLM 生成后验证
     - VL-GenRM (arXiv 2506.13888): Vision Expert Filtering
     - VLM hallucination (arXiv 2503.23573): 需要防御机制
"""

import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import cv2
except ImportError:
    cv2 = None


# ============================================================
# 1. 数据加载
# ============================================================

def load_sam2_results(json_path):
    """加载 SAM2 results JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def resolve_image_path(src_dir, image_field, dataset='multiorg'):
    """从 SAM2 JSON image 字段解析原图路径

    MultiOrg: image_field = "Macros/Plate_15/image_0"
              → {src}/Macros/Plate_15/image_0/image_0.tiff

    Mouse Liver: image_field = "微信图片_xxx.jpg"
                 → {src}/微信图片_xxx.jpg
    """
    if dataset == 'multiorg':
        image_name = os.path.basename(image_field)
        return os.path.join(src_dir, image_field, f"{image_name}.tiff")
    elif dataset == 'mouse_liver':
        return os.path.join(src_dir, image_field)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_image(path, dataset='multiorg'):
    """加载图片为 numpy array (RGB uint8)

    MultiOrg: 16-bit TIFF → tifffile.imread → min-max 归一化 → RGB
    Mouse Liver: JPG/PNG → PIL.Image.open
    """
    if dataset == 'multiorg' and path.lower().endswith('.tiff'):
        if tifffile is None:
            raise ImportError("tifffile required for MultiOrg 16-bit TIFF: pip install tifffile")
        arr = tifffile.imread(path)
        # 16-bit to 8-bit min-max normalization
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax > vmin:
            arr = ((arr.astype(np.float64) - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        # grayscale to RGB
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr
    else:
        # PIL for regular images (JPG, PNG, BMP)
        # PIL.Image.open supports Chinese paths on Windows
        img = Image.open(path).convert('RGB')
        return np.array(img)


# ============================================================
# 2. 图像裁剪
# ============================================================

def crop_detection(img_arr, bbox, padding=0.1):
    """从原图裁剪 bbox 区域 (带 padding)

    Args:
        img_arr: numpy array (H, W, 3) uint8 RGB
        bbox: [x1, y1, x2, y2] 原图坐标
        padding: 边缘扩展比例 (0.1 = 10%)

    Returns:
        crop: numpy array (H', W', 3) uint8 RGB
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    px, py = int(w * padding), int(h * padding)
    x1 = max(0, int(x1 - px))
    y1 = max(0, int(y1 - py))
    x2 = min(img_arr.shape[1], int(x2 + px))
    y2 = min(img_arr.shape[0], int(y2 + py))
    return img_arr[y1:y2, x1:x2].copy()


def draw_bbox_on_crop(crop, bbox, padding=0.1):
    """在裁剪图上画 bbox 框 (红色)

    因为裁剪图已经是 bbox 区域+padding，需要计算 bbox 在裁剪图中的位置
    """
    if cv2 is None:
        return crop  # cv2 not available, skip drawing

    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    px, py = int(w * padding), int(h * padding)

    # bbox 在裁剪图中的坐标 (padding 区域后)
    bx1, by1 = px, py
    bx2, by2 = px + int(w), py + int(h)

    # clip to crop bounds
    bx1 = max(0, min(bx1, crop.shape[1] - 1))
    by1 = max(0, min(by1, crop.shape[0] - 1))
    bx2 = max(0, min(bx2, crop.shape[1] - 1))
    by2 = max(0, min(by2, crop.shape[0] - 1))

    # draw red rectangle (BGR for cv2)
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    cv2.rectangle(crop_bgr, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def save_crop(crop, path):
    """保存裁剪图为 PNG

    使用 PIL 而非 cv2.imwrite，避免中文路径问题
    """
    img = Image.fromarray(crop)
    img.save(path, format='PNG')


# ============================================================
# 3. VLM 调用
# ============================================================

VLM_PROMPT = (
    "这是显微镜下的类器官图像。图中红色框是模型检测到的区域。请判断：\n"
    "(1) 这个区域是否包含一个类器官？(0-1分，1=肯定是)\n"
    "(2) 如果有，形态是否典型？(0-1分，1=非常典型)\n"
    "(3) 你的置信度是多少？(0-1分)\n\n"
    "请以JSON格式输出，不要输出其他内容：\n"
    '{"is_organoid": 0.x, "morphology_typical": 0.x, "confidence": 0.x, "reason": "一句话理由"}'
)


def call_vlm(crop_path, prompt, output_path, timeout=60):
    """调用 z-ai vision CLI，返回 VLM 响应文本

    Args:
        crop_path: 裁剪图 PNG 路径
        prompt: VLM 提示词
        output_path: VLM 响应 JSON 保存路径
        timeout: 超时秒数

    Returns:
        (content_str, error_str): 成功返回 (content, None)，失败返回 (None, error)
    """
    # 直接用 'z-ai'，靠 PATH 查找
    # Windows 上 z-ai 可能是 .cmd 脚本，subprocess.run 会自动找到
    cmd = [
        'z-ai',
        'vision',
        '-p', prompt,
        '-i', str(crop_path),
        '-o', str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            shell=False,
        )
    except FileNotFoundError:
        # z-ai not in PATH, try npx z-ai
        cmd = ['npx', 'z-ai', 'vision', '-p', prompt, '-i', str(crop_path), '-o', str(output_path)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return None, f"z-ai not found in PATH. Install with: npm install -g z-ai-web-dev-sdk. Error: {e}"
    except subprocess.TimeoutExpired:
        return None, f"VLM call timed out after {timeout}s"

    if result.returncode != 0:
        return None, f"z-ai vision failed (code {result.returncode}): {result.stderr}"

    # 读取输出 JSON
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            response = json.load(f)
        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        return content, None
    except (FileNotFoundError, json.JSONDecodeError, IndexError) as e:
        return None, f"Failed to parse VLM response: {e}"


def parse_vlm_response(text):
    """解析 VLM 响应文本为评分字典

    Expected format:
        {"is_organoid": 0.8, "morphology_typical": 0.7, "confidence": 0.9, "reason": "..."}

    Returns:
        dict or None (if parsing fails)
    """
    if text is None:
        return None

    # 尝试提取 JSON 块
    try:
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            raw = json.loads(text[json_start:json_end])
            return {
                'is_organoid': float(raw.get('is_organoid', 0.5)),
                'morphology_typical': float(raw.get('morphology_typical', 0.5)),
                'confidence': float(raw.get('confidence', 0.5)),
                'reason': str(raw.get('reason', '')),
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 如果 JSON 解析失败，尝试从文本中提取数值
    # 例如 "is_organoid: 0.8"
    import re
    scores = {}
    for key in ['is_organoid', 'morphology_typical', 'confidence']:
        match = re.search(rf'{key}["\']?\s*[:=]\s*([0-9.]+)', text, re.IGNORECASE)
        if match:
            try:
                scores[key] = float(match.group(1))
            except ValueError:
                scores[key] = 0.5
        else:
            scores[key] = 0.5
    scores['reason'] = text[:200]  # 保留前 200 字符
    return scores


# ============================================================
# 4. CROWN 幻觉防御 (简化版)
# ============================================================

def crown_defense(vlm_scores, rfdetr_conf):
    """CROWN 幻觉防御 — 检测 VLM 与 RF-DETR 的冲突

    Args:
        vlm_scores: parse_vlm_response 的输出
        rfdetr_conf: RF-DETR confidence (0-1)

    Returns:
        dict with defense flags:
            - vlm_hallucination: VLM 说 yes 但 RF-DETR 不确定
            - vlm_missed: VLM 说 no 但 RF-DETR 很确定
            - adjusted_score: 调整后的 is_organoid 分数
    """
    vlm_organoid = vlm_scores.get('is_organoid', 0.5)
    vlm_conf = vlm_scores.get('confidence', 0.5)

    vlm_hallucination = vlm_organoid > 0.7 and rfdetr_conf < 0.4
    vlm_missed = vlm_organoid < 0.3 and rfdetr_conf > 0.7

    # 简单调整：冲突时取两者的平均
    if vlm_hallucination:
        adjusted = (vlm_organoid + rfdetr_conf) / 2
    elif vlm_missed:
        adjusted = (vlm_organoid + rfdetr_conf) / 2
    else:
        adjusted = vlm_organoid

    return {
        'vlm_hallucination': vlm_hallucination,
        'vlm_missed': vlm_missed,
        'adjusted_score': float(adjusted),
    }


# ============================================================
# 5. 评估
# ============================================================

def evaluate(results, output_dir):
    """评估 VLM 评分的 TP/FP 区分能力

    Metrics:
        - Mann-Whitney U test (TP vs FP scores)
        - ROC-AUC
        - Precision-Recall at different thresholds
        - Compare with RF-DETR confidence
    """
    from scipy.stats import mannwhitneyu

    tp_scores = [r['vlm']['is_organoid'] for r in results if r['matched'] and r.get('vlm')]
    fp_scores = [r['vlm']['is_organoid'] for r in results if not r['matched'] and r.get('vlm')]
    tp_conf = [r['rfdetr_conf'] for r in results if r['matched']]
    fp_conf = [r['rfdetr_conf'] for r in results if not r['matched']]

    # Mann-Whitney U
    if len(tp_scores) > 0 and len(fp_scores) > 0:
        u_vlm, p_vlm = mannwhitneyu(tp_scores, fp_scores, alternative='two-sided')
        u_conf, p_conf = mannwhitneyu(tp_conf, fp_conf, alternative='two-sided')
    else:
        p_vlm, p_conf = 1.0, 1.0

    # ROC-AUC (简单计算)
    from sklearn.metrics import roc_auc_score
    y_true = [1] * len(tp_scores) + [0] * len(fp_scores)
    y_vlm = tp_scores + fp_scores
    y_conf = tp_conf + fp_conf

    auc_vlm = roc_auc_score(y_true, y_vlm) if len(set(y_true)) > 1 else 0.5
    auc_conf = roc_auc_score(y_true, y_conf) if len(set(y_true)) > 1 else 0.5

    # CROWN defense stats
    n_hallucination = sum(1 for r in results if r.get('crown', {}).get('vlm_hallucination', False))
    n_missed = sum(1 for r in results if r.get('crown', {}).get('vlm_missed', False))

    summary = {
        "phase": "Phase 2: VLM 语义确认 — GLM-4V TP/FP 验证",
        "total_detections": len(results),
        "tp_count": len(tp_scores),
        "fp_count": len(fp_scores),
        "vlm_is_organoid": {
            "tp_median": float(np.median(tp_scores)) if tp_scores else 0,
            "fp_median": float(np.median(fp_scores)) if fp_scores else 0,
            "p_value": float(p_vlm),
            "roc_auc": float(auc_vlm),
        },
        "rfdetr_confidence": {
            "tp_median": float(np.median(tp_conf)) if tp_conf else 0,
            "fp_median": float(np.median(fp_conf)) if fp_conf else 0,
            "p_value": float(p_conf),
            "roc_auc": float(auc_conf),
        },
        "crown_defense": {
            "vlm_hallucination_count": n_hallucination,
            "vlm_missed_count": n_missed,
        },
        "literature": {
            "ctm": "Sakana AI, NeurIPS 2025 Spotlight — internal ticks for iterative refinement",
            "generate_but_verify": "arXiv 2504.13169 — VLM 生成后验证减少幻觉",
            "vlm_hallucination": "arXiv 2503.23573 — VLM 幻觉是系统性的",
        },
    }

    summary_path = output_dir / 'vlm_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nSaved: {summary_path}")

    # 打印关键结果
    print(f"\n{'='*60}")
    print(f"Phase 2 VLM Verification Results")
    print(f"{'='*60}")
    print(f"  Detections: {len(results)} ({len(tp_scores)} TP, {len(fp_scores)} FP)")
    print(f"\n  VLM is_organoid:")
    print(f"    TP median: {summary['vlm_is_organoid']['tp_median']:.3f}")
    print(f"    FP median: {summary['vlm_is_organoid']['fp_median']:.3f}")
    sig = '***' if p_vlm < 0.001 else '**' if p_vlm < 0.01 else '*' if p_vlm < 0.05 else 'ns'
    print(f"    p-value:   {p_vlm:.6f} {sig}")
    print(f"    ROC-AUC:   {auc_vlm:.3f}")
    print(f"\n  RF-DETR confidence:")
    print(f"    TP median: {summary['rfdetr_confidence']['tp_median']:.3f}")
    print(f"    FP median: {summary['rfdetr_confidence']['fp_median']:.3f}")
    sig = '***' if p_conf < 0.001 else '**' if p_conf < 0.01 else '*' if p_conf < 0.05 else 'ns'
    print(f"    p-value:   {p_conf:.6f} {sig}")
    print(f"    ROC-AUC:   {auc_conf:.3f}")
    print(f"\n  CROWN defense:")
    print(f"    VLM hallucination: {n_hallucination}")
    print(f"    VLM missed:        {n_missed}")


# ============================================================
# 6. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 2: VLM verification')
    parser.add_argument('--json', required=True, help='SAM2 results JSON path')
    parser.add_argument('--src', required=True, help='Original image directory')
    parser.add_argument('--dst', default='results/phase2_vlm', help='Output directory')
    parser.add_argument('--dataset', default='multiorg', choices=['multiorg', 'mouse_liver'])
    parser.add_argument('--max-tp', type=int, default=10, help='Max TP detections to evaluate')
    parser.add_argument('--max-fp', type=int, default=10, help='Max FP detections to evaluate')
    args = parser.parse_args()

    output_dir = Path(args.dst)
    crops_dir = output_dir / 'crops'
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # 加载 SAM2 results
    print(f"Loading: {args.json}")
    sam2_data = load_sam2_results(args.json)

    # 收集 TP 和 FP detections
    # 兼容 MultiOrg (per_image list) 和 Mouse Liver (results dict) 格式
    per_image = sam2_data.get('per_image', [])
    if isinstance(per_image, dict):
        per_image = list(per_image.values())
    if not per_image:
        results_dict = sam2_data.get('results', {})
        if isinstance(results_dict, dict):
            per_image = list(results_dict.values())

    tp_list = []
    fp_list = []

    for img_data in per_image:
        if not isinstance(img_data, dict):
            continue
        if 'detections' not in img_data:
            continue
        image_field = img_data.get('image', img_data.get('image_name', ''))
        for i, det in enumerate(img_data['detections']):
            entry = {
                'image': image_field,
                'det_idx': i,
                'bbox': det.get('bbox', det.get('bbox_xyxy', [])),
                'confidence': float(det.get('confidence', 0)),
                'matched': det.get('matched', det.get('is_tp', False)),
            }
            if entry['matched']:
                tp_list.append(entry)
            else:
                fp_list.append(entry)

    print(f"Found: {len(tp_list)} TP, {len(fp_list)} FP")

    # 采样
    import random
    random.seed(42)
    if len(tp_list) > args.max_tp:
        tp_list = random.sample(tp_list, args.max_tp)
    if len(fp_list) > args.max_fp:
        fp_list = random.sample(fp_list, args.max_fp)

    all_detections = tp_list + fp_list
    print(f"Sampled: {len(tp_list)} TP, {len(fp_list)} FP")

    # 缓存
    cache_path = output_dir / 'vlm_cache.json'
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}

    # 逐个处理
    results = []
    image_cache = {}  # 缓存已加载的图片

    for idx, det in enumerate(all_detections):
        cache_key = f"{det['image']}_{det['det_idx']}".replace('/', '_').replace('\\', '_')
        print(f"\n[{idx+1}/{len(all_detections)}] {cache_key}")

        # 检查缓存
        if cache_key in cache:
            print(f"  (cached)")
            vlm_scores = cache[cache_key]
        else:
            # 解析图片路径
            img_path = resolve_image_path(args.src, det['image'], args.dataset)
            if not os.path.exists(img_path):
                print(f"  [ERROR] Image not found: {img_path}")
                continue

            # 加载图片 (缓存)
            if det['image'] not in image_cache:
                try:
                    image_cache[det['image']] = load_image(img_path, args.dataset)
                    print(f"  Loaded: {img_path}")
                except Exception as e:
                    print(f"  [ERROR] Failed to load image: {e}")
                    continue
            img_arr = image_cache[det['image']]

            # 裁剪
            bbox = det['bbox']
            if len(bbox) != 4:
                print(f"  [ERROR] Invalid bbox: {bbox}")
                continue

            crop = crop_detection(img_arr, bbox, padding=0.1)
            crop = draw_bbox_on_crop(crop, bbox, padding=0.1)

            # 保存裁剪图
            crop_path = crops_dir / f"{cache_key}.png"
            save_crop(crop, crop_path)
            print(f"  Crop saved: {crop_path}")

            # 调用 VLM
            vlm_output_path = output_dir / f"vlm_response_{cache_key}.json"
            content, error = call_vlm(crop_path, VLM_PROMPT, vlm_output_path)

            if error:
                print(f"  [VLM ERROR] {error}")
                vlm_scores = None
            else:
                vlm_scores = parse_vlm_response(content)
                print(f"  VLM: is_organoid={vlm_scores['is_organoid']:.2f}, "
                      f"typical={vlm_scores['morphology_typical']:.2f}, "
                      f"conf={vlm_scores['confidence']:.2f}")

            # 缓存
            cache[cache_key] = vlm_scores
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

            # 清理临时文件
            if vlm_output_path.exists():
                vlm_output_path.unlink()

            # 限速 (避免 API rate limit)
            time.sleep(1)

        # CROWN 防御
        crown = None
        if vlm_scores:
            crown = crown_defense(vlm_scores, det['confidence'])

        results.append({
            'image': det['image'],
            'det_idx': det['det_idx'],
            'bbox': det['bbox'],
            'rfdetr_conf': det['confidence'],
            'matched': det['matched'],
            'vlm': vlm_scores,
            'crown': crown,
        })

    # 保存完整结果
    results_path = output_dir / 'vlm_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nSaved: {results_path}")

    # 评估
    if len(results) > 0 and any(r.get('vlm') for r in results):
        evaluate(results, output_dir)
    else:
        print("\n[WARN] No VLM results to evaluate")

    print(f"\nDone. Crops in: {crops_dir} (can be deleted)")


if __name__ == '__main__':
    main()
