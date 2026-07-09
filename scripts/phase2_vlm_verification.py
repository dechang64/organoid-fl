r"""
Phase 2: VLM 语义确认 — GLM-4V 评估 detection 质量

 Usage (MultiOrg):
     cd C:\Users\decha\organoid-fl
     python scripts\phase2_vlm_verification.py --json results\multiorg_sam2_zeroshot\multiorg_sam2_results.json --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\phase2_vlm --dataset multiorg --max-tp 10 --max-fp 10

 Output:
     results/phase2_vlm/vlm_results.json  — per-detection VLM scores
     results/phase2_vlm/vlm_summary.json  — TP/FP separation analysis
     results/phase2_vlm/crops/            — 临时裁剪图 (可删除)
     results/phase2_vlm/vlm_cache.json    — VLM 响应缓存

 Requires:
     .z-ai-config file in CWD, HOME, or /etc/ with:
     {"baseUrl": "...", "apiKey": "...", "token": "..."}

 Literature:
     - CTM (Sakana AI, NeurIPS 2025 Spotlight): internal ticks = iterative refinement
     - Generate, but Verify (arXiv 2504.13169): VLM 生成后验证
     - VL-GenRM (arXiv 2506.13888): Vision Expert Filtering
     - VLM hallucination (arXiv 2503.23573): 需要防御机制
"""

import argparse
import base64
import json
import os
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

try:
    import requests
except ImportError:
    requests = None


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
        img = Image.open(path).convert('RGB')
        return np.array(img)


# ============================================================
# 2. 图像裁剪
# ============================================================

def crop_detection(img_arr, bbox, padding=0.1):
    """从原图裁剪 bbox 区域 (带 padding)"""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    px, py = int(w * padding), int(h * padding)
    x1 = max(0, int(x1 - px))
    y1 = max(0, int(y1 - py))
    x2 = min(img_arr.shape[1], int(x2 + px))
    y2 = min(img_arr.shape[0], int(y2 + py))
    return img_arr[y1:y2, x1:x2].copy()


def draw_bbox_on_crop(crop, bbox, padding=0.1):
    """在裁剪图上画 bbox 框 (红色)"""
    if cv2 is None:
        return crop

    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    px, py = int(w * padding), int(h * padding)

    bx1, by1 = px, py
    bx2, by2 = px + int(w), py + int(h)

    bx1 = max(0, min(bx1, crop.shape[1] - 1))
    by1 = max(0, min(by1, crop.shape[0] - 1))
    bx2 = max(0, min(bx2, crop.shape[1] - 1))
    by2 = max(0, min(by2, crop.shape[0] - 1))

    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    cv2.rectangle(crop_bgr, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def save_crop(crop, path):
    """保存裁剪图为 PNG (用 PIL 避免中文路径问题)"""
    img = Image.fromarray(crop)
    img.save(path, format='PNG')


# ============================================================
# 3. VLM 调用 (Python requests 直调 GLM-4V API)
# ============================================================

VLM_PROMPT = (
    "这是显微镜下的类器官图像。图中红色框是模型检测到的区域。请判断：\n"
    "(1) 这个区域是否包含一个类器官？(0-1分，1=肯定是)\n"
    "(2) 如果有，形态是否典型？(0-1分，1=非常典型)\n"
    "(3) 你的置信度是多少？(0-1分)\n\n"
    "请以JSON格式输出，不要输出其他内容：\n"
    '{"is_organoid": 0.x, "morphology_typical": 0.x, "confidence": 0.x, "reason": "一句话理由"}'
)


def load_zai_config():
    """加载 z-ai 配置文件

    查找顺序: CWD/.z-ai-config → HOME/.z-ai-config → /etc/.z-ai-config
    格式: {"baseUrl": "...", "apiKey": "...", "token": "...", "chatId": "...", "userId": "..."}
    """
    config_paths = [
        os.path.join(os.getcwd(), '.z-ai-config'),
        os.path.join(os.path.expanduser('~'), '.z-ai-config'),
        '/etc/.z-ai-config',
    ]
    for p in config_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                config = json.load(f)
            if config.get('baseUrl') and config.get('apiKey'):
                return config
    raise FileNotFoundError(
        ".z-ai-config not found or invalid. Create it in CWD, HOME, or /etc/ "
        "with: {\"baseUrl\": \"...\", \"apiKey\": \"...\", \"token\": \"...\"}"
    )


def call_vlm(crop_path, prompt, output_path, timeout=60, zai_config=None):
    """调用 GLM-4V API (Python requests，不依赖 z-ai CLI)

    Args:
        crop_path: 裁剪图 PNG 路径
        prompt: VLM 提示词
        output_path: VLM 响应 JSON 保存路径
        timeout: 超时秒数
        zai_config: 配置 dict (baseUrl, apiKey, token, chatId, userId)

    Returns:
        (content_str, error_str): 成功返回 (content, None)，失败返回 (None, error)
    """
    if requests is None:
        return None, "requests library required: pip install requests"
    if zai_config is None:
        try:
            zai_config = load_zai_config()
        except FileNotFoundError as e:
            return None, f"VLM skipped: {e}"

    base_url = zai_config.get('baseUrl', '')
    api_key = zai_config.get('apiKey', '')
    token = zai_config.get('token', '')
    chat_id = zai_config.get('chatId', '')
    user_id = zai_config.get('userId', '')

    # 读取图片 → base64 data URL
    with open(crop_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    data_url = f"data:image/png;base64,{img_b64}"

    url = f"{base_url}/chat/completions/vision"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'X-Z-AI-From': 'Z',
    }
    if chat_id:
        headers['X-Chat-Id'] = chat_id
    if user_id:
        headers['X-User-Id'] = user_id
    if token:
        headers['X-Token'] = token

    payload = {
        'model': 'glm-4.6v',
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': data_url}},
            ]
        }],
        'thinking': {'type': 'disabled'},
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        return None, f'VLM call timed out after {timeout}s'
    except Exception as e:
        return None, f'Connection error: {e}'

    if resp.status_code != 200:
        return None, f'API returned {resp.status_code}: {resp.text[:200]}'

    resp_json = resp.json()

    # 保存完整响应 (调试用)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(resp_json, f, indent=2, ensure_ascii=False)

    content = resp_json.get('choices', [{}])[0].get('message', {}).get('content', '')
    return content, None


def parse_vlm_response(text):
    """解析 VLM 响应文本为评分字典

    Expected: {"is_organoid": 0.8, "morphology_typical": 0.7, "confidence": 0.9, "reason": "..."}
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

    # 降级：从文本中提取数值
    import re
    scores = {}
    for key in ['is_organoid', 'morphology_typical', 'confidence']:
        match = re.search(rf'{key}["\']?\s*[:=]\s*([0-9.]+)', text, re.IGNORECASE)
        scores[key] = float(match.group(1)) if match else 0.5
    scores['reason'] = text[:200]
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
        dict: vlm_hallucination, vlm_missed, adjusted_score
    """
    vlm_organoid = vlm_scores.get('is_organoid', 0.5)

    vlm_hallucination = vlm_organoid > 0.7 and rfdetr_conf < 0.4
    vlm_missed = vlm_organoid < 0.3 and rfdetr_conf > 0.7

    if vlm_hallucination or vlm_missed:
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
    """评估 VLM 评分的 TP/FP 区分能力"""
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score

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

    # ROC-AUC
    y_true = [1] * len(tp_scores) + [0] * len(fp_scores)
    y_vlm = tp_scores + fp_scores
    y_conf = tp_conf + fp_conf

    auc_vlm = roc_auc_score(y_true, y_vlm) if len(set(y_true)) > 1 else 0.5
    auc_conf = roc_auc_score(y_true, y_conf) if len(set(y_true)) > 1 else 0.5

    # CROWN stats
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
            "ctm": "Sakana AI, NeurIPS 2025 Spotlight",
            "generate_but_verify": "arXiv 2504.13169",
            "vlm_hallucination": "arXiv 2503.23573",
        },
    }

    summary_path = output_dir / 'vlm_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
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
    parser.add_argument('--skip-vlm', action='store_true', help='Only generate crops, skip VLM calls')
    args = parser.parse_args()

    # 路径检查
    if not os.path.exists(args.json):
        print(f"[ERROR] SAM2 JSON not found: {args.json}")
        sys.exit(1)
    if not os.path.isdir(args.src):
        print(f"[ERROR] Source image directory not found: {args.src}")
        sys.exit(1)

    output_dir = Path(args.dst)
    crops_dir = output_dir / 'crops'
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # 加载 SAM2 results
    print(f"Loading: {args.json}")
    sam2_data = load_sam2_results(args.json)

    # 收集 TP 和 FP detections
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
    image_cache = {}

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
            if args.skip_vlm:
                vlm_scores = None
            else:
                content, error = call_vlm(crop_path, VLM_PROMPT, vlm_output_path)

                if error:
                    print(f"  [VLM] {error}")
                    vlm_scores = None
                else:
                    vlm_scores = parse_vlm_response(content)
                    if vlm_scores:
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
        json.dump(results, f, indent=2, ensure_ascii=False,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nSaved: {results_path}")

    # 评估
    if len(results) > 0 and any(r.get('vlm') for r in results):
        evaluate(results, output_dir)
    else:
        print("\n[WARN] No VLM results to evaluate")

    print(f"\nDone. Crops in: {crops_dir} (can be deleted)")


if __name__ == '__main__':
    main()
