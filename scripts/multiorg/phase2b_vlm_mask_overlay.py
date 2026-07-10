"""
Phase 2b: VLM with mask overlay — 按方案用 SAM2 mask + 原图裁剪

Flow:
1. Load each crop (from Phase 2 bbox-only crops)
2. Run SAM2 with center-point prompt → get mask
3. Overlay mask in green semi-transparent on crop
4. Call GLM-4.6V with mask-overlay image
5. Compare with Phase 2 bbox-only results

Usage (cloud VM):
    python scripts/multiorg/phase2b_vlm_mask_overlay.py \
        --crops-dir results/phase2_vlm_100/crops \
        --metadata results/phase2_vlm_100/vlm_results.json \
        --dst results/phase2_vlm_100_mask \
        --sam2-checkpoint sam2_hiera_small.pt
"""
import json, os, sys, base64, time, requests, numpy as np
from pathlib import Path
from PIL import Image
import cv2

# SAM2
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.cfg.default import load_config

VLM_PROMPT = '这是显微镜下的类器官图像。绿色区域是模型分割的mask。请判断：(1) 这个mask是否准确覆盖了一个类器官？(0-1分) (2) mask的边界是否精确？(0-1分) (3) 这个形态是否典型？(0-1分) 请以JSON格式输出：{"is_organoid": 0.x, "morphology_typical": 0.x, "confidence": 0.x, "reason": "一句话"}'


def load_sam2(checkpoint_path):
    """Load SAM2 model — 复用 multiorg_sam2.py 的加载逻辑"""
    import torch
    
    ckpt_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt_data, dict) and 'model_state_dict' in ckpt_data:
        import tempfile, shutil, atexit
        tmp_dir = tempfile.mkdtemp()
        tmp_ckpt = os.path.join(tmp_dir, 'model_finetuned.pt')
        torch.save({'model': ckpt_data['model_state_dict']}, tmp_ckpt)
        checkpoint_path = tmp_ckpt
        atexit.register(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))
    
    last_err = None
    for cfg in ['sam2_hiera_s', 'sam2_hiera_small']:
        try:
            model = build_sam2(cfg, checkpoint_path, device="cpu")
            return SAM2ImagePredictor(model)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load SAM2: {last_err}")


def run_sam2_on_crop(predictor, crop_arr):
    """Run SAM2 on a crop with center-point prompt"""
    # Ensure RGB
    if crop_arr.ndim == 2:
        crop_arr = np.stack([crop_arr]*3, axis=-1)
    elif crop_arr.shape[2] == 4:
        crop_arr = crop_arr[:, :, :3]
    
    h, w = crop_arr.shape[:2]
    
    # Use center point as prompt
    center_x, center_y = w // 2, h // 2
    
    # SAM2 expects RGB
    predictor.set_image(crop_arr)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[center_x, center_y]]),
        point_labels=np.array([1]),
        multimask_output=False,
    )
    
    if len(masks) == 0:
        return None
    
    return masks[0]  # (H, W) boolean


def overlay_mask(crop_arr, mask, alpha=0.4, color=[0, 255, 0]):
    """Overlay mask in green semi-transparent"""
    overlay = crop_arr.copy()
    overlay[mask] = color
    result = cv2.addWeighted(overlay, alpha, crop_arr, 1 - alpha, 0)
    # Draw mask boundary
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result


def load_zai_config():
    config_paths = [
        os.path.join(os.getcwd(), '.z-ai-config'),
        os.path.join(os.path.expanduser('~'), '.z-ai-config'),
        '/etc/.z-ai-config',
    ]
    for p in config_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError('.z-ai-config not found')


def call_vlm(image_path, prompt, config, timeout=30):
    """Call GLM-4.6V API"""
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    data_url = f"data:image/png;base64,{img_b64}"
    
    url = f"{config['baseUrl']}/chat/completions/vision"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {config['apiKey']}",
        'X-Z-AI-From': 'Z',
        'X-Chat-Id': config.get('chatId', ''),
        'X-User-Id': config.get('userId', ''),
        'X-Token': config.get('token', ''),
    }
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
    
    for retry in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content']
                vlm = None
                try:
                    s, e = content.find('{'), content.rfind('}') + 1
                    if s >= 0 and e > s:
                        vlm = {k: float(v) if isinstance(v, (int, float)) else v 
                               for k, v in json.loads(content[s:e]).items()}
                except: pass
                return vlm, content, None
            elif resp.status_code == 429:
                wait = 60 * (retry + 1)
                print(f"  [429] waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                return None, None, f"API {resp.status_code}"
        except Exception as e:
            print(f"  [ERR] {e}", flush=True)
            time.sleep(10)
    return None, None, "Rate limited after 3 retries"


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 2b: VLM with mask overlay')
    parser.add_argument('--crops-dir', required=True, help='Directory with bbox-only crops')
    parser.add_argument('--metadata', required=True, help='vlm_results.json from Phase 2')
    parser.add_argument('--dst', required=True, help='Output directory')
    parser.add_argument('--sam2-checkpoint', default='sam2_hiera_small.pt')
    args = parser.parse_args()
    
    crops_dir = Path(args.crops_dir)
    output_dir = Path(args.dst)
    mask_crops_dir = output_dir / 'crops'
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_crops_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    with open(args.metadata, 'r', encoding='utf-8') as f:
        detections = json.load(f)
    print(f"Loaded {len(detections)} detections")
    
    # Load SAM2
    print("Loading SAM2...")
    predictor = load_sam2(args.sam2_checkpoint)
    print("SAM2 loaded")
    
    # Load VLM config
    config = load_zai_config()
    print(f"VLM config: {config['baseUrl']}")
    
    # Process each detection
    results_path = output_dir / 'vlm_results_mask.json'
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded existing results: {len(results)}")
    else:
        results = []
    
    for idx, det in enumerate(detections):
        cache_key = det.get('cache_key', f"{det['image']}_{det['det_idx']}".replace('/', '_'))
        
        # Skip if already processed
        existing = [r for r in results if r.get('cache_key') == cache_key]
        if existing and existing[0].get('vlm_mask'):
            continue
        
        print(f"\n[{idx+1}/{len(detections)}] {cache_key} ({'TP' if det['matched'] else 'FP'})", flush=True)
        
        crop_path = crops_dir / f"{cache_key}.png"
        if not crop_path.exists():
            print(f"  [SKIP] Crop not found", flush=True)
            continue
        
        # Load crop
        crop_arr = np.array(Image.open(crop_path).convert('RGB'))
        
        # Run SAM2
        try:
            mask = run_sam2_on_crop(predictor, crop_arr)
        except Exception as e:
            print(f"  [SAM2 ERROR] {e}", flush=True)
            mask = None
        
        if mask is None:
            print(f"  [WARN] No mask, using bbox fill", flush=True)
            h, w = crop_arr.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            mask[h//4:3*h//4, w//4:3*w//4] = True
        
        # Overlay mask
        mask_crop = overlay_mask(crop_arr, mask)
        
        # Save mask-overlay crop
        mask_crop_path = mask_crops_dir / f"{cache_key}.png"
        result, encoded = cv2.imencode('.png', cv2.cvtColor(mask_crop, cv2.COLOR_RGB2BGR))
        encoded.tofile(str(mask_crop_path))
        print(f"  Mask crop saved", flush=True)
        
        # Call VLM
        vlm_scores, vlm_raw, error = call_vlm(str(mask_crop_path), VLM_PROMPT, config)
        
        if error:
            print(f"  [VLM] {error}", flush=True)
            vlm_scores = None
        
        if vlm_scores:
            print(f"  -> is_org={vlm_scores.get('is_organoid','?')}, "
                  f"typical={vlm_scores.get('morphology_typical','?')}", flush=True)
        
        # Save result
        result_entry = {
            'cache_key': cache_key,
            'image': det.get('image', ''),
            'det_idx': det.get('det_idx', 0),
            'bbox': det.get('bbox', []),
            'rfdetr_conf': det.get('confidence', det.get('rfdetr_conf', 0)),
            'matched': det.get('matched', False),
            'vlm_bbox': det.get('vlm', None),  # Phase 2 bbox-only result
            'vlm_mask': vlm_scores,
            'vlm_mask_raw': vlm_raw,
        }
        
        # Update or append
        found = False
        for i, r in enumerate(results):
            if r.get('cache_key') == cache_key:
                results[i] = result_entry
                found = True
                break
        if not found:
            results.append(result_entry)
        
        # Save after each call
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)
        
        time.sleep(5)
    
    done = sum(1 for r in results if r.get('vlm_mask'))
    print(f"\nDone: {done}/{len(detections)}")


if __name__ == '__main__':
    main()
