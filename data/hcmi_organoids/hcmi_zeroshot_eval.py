#!/usr/bin/env python3
"""
HCMI Organoid Zero-shot Cross-domain Evaluation
=================================================
Evaluate organoid detection models on HCMI catalog images across 4 centers.

Pipeline:
1. RF-DETR (mouse liver trained) → direct inference (expect poor: domain gap)
2. YOLOv12 (mouse liver trained) → direct inference
3. SAM2 zero-shot (grid point prompt) → pseudo masks as pseudo-GT
4. Use SAM2 masks as pseudo-GT to evaluate RF-DETR/YOLO detection quality
5. Compare 4 centers: CSHL / SANG / BROD / WCMC

Output:
- per_image.json: detailed results per image
- per_center.json: aggregated stats per center
- summary.json: overall comparison
- overlays/: visual overlays (SAM2 mask + detection boxes)

Usage (Windows PowerShell):
    cd C:/Users/decha/organoid-fl

    # Quick test (8 images)
    python data/hcmi_organoids/hcmi_zeroshot_eval.py --max-images 8 --save-overlays

    # Full run (477 images, ~5min on 3060 GPU)
    python data/hcmi_organoids/hcmi_zeroshot_eval.py --save-overlays

    # SAM2 only (skip RF-DETR/YOLO)
    python data/hcmi_organoids/hcmi_zeroshot_eval.py --models sam2 --save-overlays
"""

import argparse
import os
import json
import time
import warnings
import csv
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════
# Configuration — EDIT THESE PATHS FOR YOUR MACHINE
# ═══════════════════════════════════════════════════
# HCMI images directory (downloaded from cloud VM)
DATA_DIR = Path("data/hcmi_organoids/images")
# Manifest CSV (downloaded with images)
MANIFEST_CSV = Path("data/hcmi_organoids/hcmi_organoid_images.csv")
# Output directory (auto-created)
OUTPUT_DIR = Path("data/hcmi_organoids/analysis")

# Model checkpoints
RFDETR_CKPT = "runs/mouse_liver_v2/b1/full/checkpoint_best_ema.pth"
YOLO_CKPT = "runs/detect/runs/mouse_liver/exp3_unified/weights/best.pt"
SAM2_CKPT = "sam2_hiera_small.pt"

# Device: "cuda:0" for GPU, "cpu" for CPU
DEVICE = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"

# SAM2 grid prompt parameters
SAM2_GRID_STEP = 200       # pixel step for grid points (larger=faster)
SAM2_SCORE_THRESH = 0.8    # only keep high-confidence masks
SAM2_MIN_AREA = 1000       # min mask area (px)
SAM2_MAX_AREA = 500000     # max mask area (px)
SAM2_MERGE_IOU = 0.5       # NMS IoU for merging overlapping masks

# RF-DETR thresholds
RFDETR_CONF_THRESHES = [0.5, 0.3, 0.1, 0.05]

# YOLO thresholds
YOLO_CONF_THRESHES = [0.5, 0.3, 0.1, 0.05]


# ═══════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════
def iou_box_mask(box, mask):
    """IoU between a bounding box and a binary mask.
    box: [x1, y1, x2, y2]
    mask: HxW bool array
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    H, W = mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    mask_area = mask[y1:y2, x1:x2].sum()
    box_area = (x2 - x1) * (y2 - y1)
    if box_area == 0:
        return 0.0
    return float(mask_area) / box_area


def mask_to_bbox(mask):
    """Convert binary mask to bounding box [x1, y1, x2, y2]."""
    rows = mask.any(axis=1)
    if not rows.any():
        return None
    cols = mask.any(axis=0)
    y1, y2 = rows.argmax(), len(rows) - rows[::-1].argmax()
    x1, x2 = cols.argmax(), len(cols) - cols[::-1].argmax()
    return [int(x1), int(y1), int(x2), int(y2)]


def box_iou(b1, b2):
    """IoU between two boxes [x1,y1,x2,y2]."""
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def merge_masks_nms(masks, scores, iou_thresh=0.5):
    """NMS-style merge: sort by score, suppress overlapping masks."""
    if not masks:
        return [], []
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    kept = []
    kept_scores = []
    suppressed = set()
    for idx in order:
        if idx in suppressed:
            continue
        kept.append(masks[idx])
        kept_scores.append(scores[idx])
        for jdx in order:
            if jdx == idx or jdx in suppressed:
                continue
            bb1 = mask_to_bbox(masks[idx])
            bb2 = mask_to_bbox(masks[jdx])
            if bb1 and bb2 and box_iou(bb1, bb2) > iou_thresh:
                suppressed.add(jdx)
    return kept, kept_scores


def load_manifest():
    """Load image manifest CSV. Auto-download from HCMI API if missing."""
    if not MANIFEST_CSV.exists():
        print(f"Manifest not found at {MANIFEST_CSV}, downloading from HCMI API...")
        _download_manifest()

    entries = []
    with open(MANIFEST_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    return entries


def _download_manifest():
    """Query HCMI GraphQL API and save manifest CSV."""
    import urllib.request
    import json as _json

    GRAPHQL_URL = "https://hcmi-searchable-catalog.nci.nih.gov/api/graphql"
    query = """
    query($sqon: JSON) {
      model {
        hits(first: 700, filters: $sqon) {
          total
          edges {
            node {
              name
              primary_site
              type
              disease_status
              clinical_diagnosis { clinical_tumor_diagnosis }
              files {
                hits(first: 10) {
                  total
                  edges {
                    node {
                      file_name
                      file_type
                      file_url
                      magnification
                      passage_number
                      scale_bar_length
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    """
    variables = {
        "sqon": {"op": "and", "content": [{"op": "in", "content": {"field": "type", "value": ["3-D: Organoid"]}}]}
    }

    payload = _json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(GRAPHQL_URL, data=payload, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = _json.loads(resp.read())

    edges = data["data"]["model"]["hits"]["edges"]

    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "center", "primary_site", "diagnosis", "disease_status",
            "file_name", "magnification", "passage_number", "scale_bar_length",
            "file_url"
        ])
        writer.writeheader()
        count = 0
        for edge in edges:
            node = edge["node"]
            if node["type"] != "3-D: Organoid":
                continue
            model_name = node["name"]
            center = model_name.split("-")[1]
            for fe in node.get("files", {}).get("hits", {}).get("edges", []):
                fn = fe["node"]
                writer.writerow({
                    "model": model_name,
                    "center": center,
                    "primary_site": node.get("primary_site", ""),
                    "diagnosis": node.get("clinical_diagnosis", {}).get("clinical_tumor_diagnosis", ""),
                    "disease_status": node.get("disease_status", ""),
                    "file_name": fn.get("file_name", ""),
                    "magnification": fn.get("magnification", ""),
                    "passage_number": fn.get("passage_number", ""),
                    "scale_bar_length": fn.get("scale_bar_length", ""),
                    "file_url": fn.get("file_url", ""),
                })
                count += 1
    print(f"  Manifest saved: {count} entries -> {MANIFEST_CSV}")


def get_image_path(entry):
    """Get local image path for a manifest entry."""
    return os.path.join(DATA_DIR, entry["center"], entry["file_name"])


# ═══════════════════════════════════════════════════
# Model Loaders
# ═══════════════════════════════════════════════════
_rfdetr_model = None
_yolo_model = None
_sam2_predictor = None


def get_rfdetr():
    global _rfdetr_model
    if _rfdetr_model is None:
        from rfdetr import RFDETRSmall
        _rfdetr_model = RFDETRSmall(pretrain_weights=RFDETR_CKPT)
    return _rfdetr_model


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_CKPT)
        if DEVICE != "cpu":
            _yolo_model.to(DEVICE)
    return _yolo_model


def get_sam2():
    global _sam2_predictor
    if _sam2_predictor is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2("sam2_hiera_s", SAM2_CKPT, device=DEVICE)
        _sam2_predictor = SAM2ImagePredictor(model)
    return _sam2_predictor


# ═══════════════════════════════════════════════════
# Inference Functions
# ═══════════════════════════════════════════════════
def run_rfdetr(img_array, conf=0.05):
    """Run RF-DETR inference. Returns list of {box, score}."""
    from PIL import Image
    model = get_rfdetr()
    img = Image.fromarray(img_array)
    dets = model.predict(img, threshold=conf)
    results = []
    for d in dets:
        results.append({
            "box": [float(d.xyxy[0][0]), float(d.xyxy[0][1]),
                    float(d.xyxy[0][2]), float(d.xyxy[0][3])],
            "score": float(d.score),
        })
    return results


def run_yolo(img_array, conf=0.05):
    """Run YOLOv12 inference. Returns list of {box, score, cls}."""
    model = get_yolo()
    results_raw = model(img_array, conf=conf, verbose=False)
    results = []
    if results_raw and results_raw[0].boxes is not None:
        for box in results_raw[0].boxes:
            xyxy = box.xyxy[0].tolist()
            results.append({
                "box": xyxy,
                "score": float(box.conf),
                "cls": int(box.cls),
                "cls_name": model.model.names.get(int(box.cls), str(int(box.cls))),
            })
    return results


def run_sam2_grid(img_array):
    """Run SAM2 with grid point prompts. Returns (masks, scores, bboxes)."""
    import numpy as np
    predictor = get_sam2()
    H, W = img_array.shape[:2]
    predictor.set_image(img_array)

    step = SAM2_GRID_STEP
    points = [[x, y] for y in range(step // 2, H, step)
              for x in range(step // 2, W, step)]

    raw_masks = []
    raw_scores = []
    for pt in points:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([pt]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        best = scores.argmax()
        s = float(scores[best])
        if s > SAM2_SCORE_THRESH:
            m = masks[best].astype(bool)
            area = m.sum()
            if SAM2_MIN_AREA < area < SAM2_MAX_AREA:
                raw_masks.append(m)
                raw_scores.append(s)

    # NMS merge
    merged_masks, merged_scores = merge_masks_nms(raw_masks, raw_scores, SAM2_MERGE_IOU)

    # Compute bboxes
    bboxes = []
    for m in merged_masks:
        bb = mask_to_bbox(m)
        bboxes.append(bb)

    return merged_masks, merged_scores, bboxes


def evaluate_detections_vs_masks(detections, sam2_masks, sam2_bboxes, iou_thresh=0.3):
    """Match detections to SAM2 pseudo-GT masks.
    Returns: {n_det, n_pseudo_gt, n_matched, precision_proxy, recall_proxy}
    """
    # Imported here because the function below the two early returns uses `np`
    # but this scope previously had no binding for it. The nearest
    # `import numpy as np` sits inside a *different* function, so calling this
    # one first raised NameError: name 'np' is not defined. Both early returns
    # already handle the empty-input cases, so the divisions further down are
    # safe as written.
    import numpy as np

    if not sam2_bboxes:
        return {
            "n_det": len(detections),
            "n_pseudo_gt": 0,
            "n_matched": 0,
            "precision_proxy": None,
            "recall_proxy": None,
        }

    if not detections:
        return {
            "n_det": 0,
            "n_pseudo_gt": len(sam2_bboxes),
            "n_matched": 0,
            "precision_proxy": None,
            "recall_proxy": 0.0,
        }

    # Greedy match by IoU
    matched_gt = set()
    matched_det = set()
    for i, det in enumerate(detections):
        best_iou = 0
        best_j = -1
        for j, gt_box in enumerate(sam2_bboxes):
            if j in matched_gt:
                continue
            iou = box_iou(det["box"], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            matched_gt.add(best_j)
            matched_det.add(i)

    return {
        "n_det": len(detections),
        "n_pseudo_gt": len(sam2_bboxes),
        "n_matched": len(matched_det),
        "precision_proxy": len(matched_det) / len(detections),
        "recall_proxy": len(matched_gt) / len(sam2_bboxes),
        "mean_det_score": float(np.mean([d["score"] for d in detections])),
    }


# ═══════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════
def process_single_image(img_path, entry, save_overlay=False, args_models="all"):
    """Process a single image through the full pipeline."""
    import numpy as np
    from PIL import Image

    result = {
        "model": entry["model"],
        "center": entry["center"],
        "primary_site": entry["primary_site"],
        "diagnosis": entry["diagnosis"],
        "magnification": entry["magnification"],
        "file_name": entry["file_name"],
    }

    try:
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)
        H, W = arr.shape[:2]
        result["image_size"] = [W, H]
    except Exception as e:
        result["error"] = f"Image load failed: {e!s}"
        return result

    # 1. SAM2 zero-shot (pseudo-GT)
    try:
        t0 = time.time()
        sam2_masks, sam2_scores, sam2_bboxes = run_sam2_grid(arr)
        result["sam2"] = {
            "n_masks": len(sam2_masks),
            "scores": [float(s) for s in sam2_scores],
            "bboxes": sam2_bboxes,
            "mask_areas": [int(m.sum()) for m in sam2_masks],
            "time_sec": round(time.time() - t0, 2),
        }
    except Exception as e:
        result["sam2"] = {"error": str(e)}

    # 2. RF-DETR (try lowest threshold that gives detections)
    if args_models in ("all", "rfdetr"):
        try:
            t0 = time.time()
            rfdetr_dets = []
            for conf in RFDETR_CONF_THRESHES:
                dets = run_rfdetr(arr, conf=conf)
                if dets:
                    rfdetr_dets = dets
                    result["rfdetr_conf_used"] = conf
                    break
            else:
                result["rfdetr_conf_used"] = RFDETR_CONF_THRESHES[-1]

            # Evaluate against SAM2 pseudo-GT
            rfdetr_eval = evaluate_detections_vs_masks(
                rfdetr_dets,
                sam2_masks if "sam2" in result and "error" not in result.get("sam2", {}) else [],
                sam2_bboxes if "sam2" in result and "error" not in result.get("sam2", {}) else [],
            )
            result["rfdetr"] = {
                "n_detections": len(rfdetr_dets),
                "detections": [{"box": d["box"], "score": d["score"]} for d in rfdetr_dets],
                "eval": rfdetr_eval,
                "time_sec": round(time.time() - t0, 2),
            }
        except Exception as e:
            result["rfdetr"] = {"error": str(e)}

    # 3. YOLOv12 (try lowest threshold)
    if args_models in ("all", "yolo"):
        try:
            t0 = time.time()
            yolo_dets = []
            for conf in YOLO_CONF_THRESHES:
                dets = run_yolo(arr, conf=conf)
                if dets:
                    yolo_dets = dets
                    result["yolo_conf_used"] = conf
                    break
            else:
                result["yolo_conf_used"] = YOLO_CONF_THRESHES[-1]

            yolo_eval = evaluate_detections_vs_masks(
                yolo_dets,
                sam2_masks if "sam2" in result and "error" not in result.get("sam2", {}) else [],
                sam2_bboxes if "sam2" in result and "error" not in result.get("sam2", {}) else [],
            )
            result["yolo"] = {
                "n_detections": len(yolo_dets),
                "detections": [{"box": d["box"], "score": d["score"],
                                "cls": d.get("cls"), "cls_name": d.get("cls_name")}
                               for d in yolo_dets],
                "eval": yolo_eval,
                "time_sec": round(time.time() - t0, 2),
            }
        except Exception as e:
            result["yolo"] = {"error": str(e)}

    # 4. Save overlay if requested
    if save_overlay and result.get("sam2", {}).get("n_masks", 0) > 0:
        try:
            save_overlay_image(arr, result, img_path)
        except Exception:
            pass

    return result


def save_overlay_image(arr, result, img_path):
    """Save overlay: SAM2 masks (green) + RF-DETR boxes (red) + YOLO boxes (blue)."""
    from PIL import Image, ImageDraw

    overlay_dir = OUTPUT_DIR / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # SAM2 masks in green
    sam2 = result.get("sam2", {})
    for _mask in sam2.get("mask_areas", []):  # just count
        pass

    # RF-DETR boxes in red
    rfdetr = result.get("rfdetr", {})
    for det in rfdetr.get("detections", []):
        x1, y1, x2, y2 = det["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # YOLO boxes in blue
    yolo = result.get("yolo", {})
    for det in yolo.get("detections", []):
        x1, y1, x2, y2 = det["box"]
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)

    # SAM2 bboxes in green
    for bb in sam2.get("bboxes", []):
        if bb:
            x1, y1, x2, y2 = bb
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=2)

    fname = Path(img_path).stem + "_overlay.jpg"
    img.save(overlay_dir / fname, quality=90)


def aggregate_by_center(per_image_results):
    """Aggregate results by center."""
    import numpy as np

    by_center = defaultdict(list)
    for r in per_image_results:
        by_center[r["center"]].append(r)

    center_stats = {}
    for center, results in by_center.items():
        # SAM2 stats
        sam2_mask_counts = [r["sam2"]["n_masks"] for r in results
                           if "sam2" in r and "error" not in r.get("sam2", {})]
        sam2_areas = []
        for r in results:
            if "sam2" in r and "error" not in r.get("sam2", {}):
                sam2_areas.extend(r["sam2"].get("mask_areas", []))

        # RF-DETR stats
        rfdetr_det_counts = [r["rfdetr"]["n_detections"] for r in results
                            if "rfdetr" in r and "error" not in r.get("rfdetr", {})]
        rfdetr_precisions = [r["rfdetr"]["eval"]["precision_proxy"]
                             for r in results
                             if "rfdetr" in r and "error" not in r.get("rfdetr", {})
                             and r["rfdetr"]["eval"]["precision_proxy"] is not None]
        rfdetr_recalls = [r["rfdetr"]["eval"]["recall_proxy"]
                          for r in results
                          if "rfdetr" in r and "error" not in r.get("rfdetr", {})
                          and r["rfdetr"]["eval"]["recall_proxy"] is not None]

        # YOLO stats
        yolo_det_counts = [r["yolo"]["n_detections"] for r in results
                          if "yolo" in r and "error" not in r.get("yolo", {})]
        yolo_precisions = [r["yolo"]["eval"]["precision_proxy"]
                           for r in results
                           if "yolo" in r and "error" not in r.get("yolo", {})
                           and r["yolo"]["eval"]["precision_proxy"] is not None]
        yolo_recalls = [r["yolo"]["eval"]["recall_proxy"]
                        for r in results
                        if "yolo" in r and "error" not in r.get("yolo", {})
                        and r["yolo"]["eval"]["recall_proxy"] is not None]

        def safe_stats(data):
            if not data:
                return {"count": 0, "mean": None, "median": None, "std": None,
                        "min": None, "max": None}
            arr = np.array(data, dtype=float)
            return {
                "count": len(arr),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

        center_stats[center] = {
            "n_images": len(results),
            "sam2_masks_per_image": safe_stats(sam2_mask_counts),
            "sam2_mask_area": safe_stats(sam2_areas),
            "rfdetr_dets_per_image": safe_stats(rfdetr_det_counts),
            "rfdetr_precision_proxy": safe_stats(rfdetr_precisions),
            "rfdetr_recall_proxy": safe_stats(rfdetr_recalls),
            "yolo_dets_per_image": safe_stats(yolo_det_counts),
            "yolo_precision_proxy": safe_stats(yolo_precisions),
            "yolo_recall_proxy": safe_stats(yolo_recalls),
        }

    return center_stats


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="HCMI zero-shot cross-domain evaluation")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images to process (0=all)")
    parser.add_argument("--models", choices=["all", "rfdetr", "yolo", "sam2"],
                        default="all", help="Which models to run")
    parser.add_argument("--save-overlays", action="store_true",
                        help="Save overlay images (slower)")
    parser.add_argument("--center", type=str, default=None,
                        help="Filter by center (CSHL/SANG/BROD/WCMC)")
    parser.add_argument("--rfdetr-ckpt", type=str, default=None,
                        help="RF-DETR checkpoint path (override default)")
    parser.add_argument("--yolo-ckpt", type=str, default=None,
                        help="YOLO checkpoint path (override default)")
    parser.add_argument("--sam2-ckpt", type=str, default=None,
                        help="SAM2 checkpoint path (override default)")
    args = parser.parse_args()

    # Update checkpoint paths from args
    global RFDETR_CKPT, YOLO_CKPT, SAM2_CKPT
    if args.rfdetr_ckpt:
        RFDETR_CKPT = args.rfdetr_ckpt
    if args.yolo_ckpt:
        YOLO_CKPT = args.yolo_ckpt
    if args.sam2_ckpt:
        SAM2_CKPT = args.sam2_ckpt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest
    entries = load_manifest()
    print(f"Loaded manifest: {len(entries)} entries")

    # Filter by center if specified
    if args.center:
        entries = [e for e in entries if e["center"] == args.center]
        print(f"Filtered to center {args.center}: {len(entries)} entries")

    # Limit images
    if args.max_images > 0:
        # Sample evenly across centers
        from collections import defaultdict
        by_center = defaultdict(list)
        for e in entries:
            by_center[e["center"]].append(e)
        per_center = max(1, args.max_images // max(1, len(by_center)))
        entries = []
        for _center, items in by_center.items():
            entries.extend(items[:per_center])
        print(f"Sampled {len(entries)} images (max {args.max_images}, ~{per_center}/center)")

    print(f"\n{'='*70}")
    print("HCMI Zero-shot Cross-domain Evaluation")
    print(f"{'='*70}")
    print(f"  Images: {len(entries)}")
    print(f"  Models: {args.models}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # Pre-load models
    if args.models in ("all", "rfdetr"):
        print("Loading RF-DETR...", flush=True)
        get_rfdetr()
    if args.models in ("all", "yolo"):
        print("Loading YOLOv12...")
        get_yolo()
    if args.models in ("all", "sam2"):
        print("Loading SAM2...")
        get_sam2()
    print()

    # Process images
    all_results = []
    t_start = time.time()

    for i, entry in enumerate(entries):
        img_path = get_image_path(entry)
        if not os.path.exists(img_path):
            print(f"[{i+1}/{len(entries)}] SKIP (not found): {entry['file_name']}")
            continue

        t0 = time.time()
        result = process_single_image(
            img_path, entry,
            save_overlay=args.save_overlays,
            args_models=args.models,
        )
        elapsed = time.time() - t0
        all_results.append(result)

        # Progress
        sam2_n = result.get("sam2", {}).get("n_masks", "?")
        rfdetr_n = result.get("rfdetr", {}).get("n_detections", "?")
        yolo_n = result.get("yolo", {}).get("n_detections", "?")
        print(f"[{i+1}/{len(entries)}] {entry['center']} {entry['file_name'][:35]:<35s} "
              f"SAM2={sam2_n:>3} RF={rfdetr_n:>3} YOLO={yolo_n:>3} ({elapsed:.1f}s)", flush=True)

        # Save intermediate results every 50 images
        if (i + 1) % 50 == 0:
            _save_results(all_results, t_start, len(entries))
            print(f"  [Intermediate save at {i+1} images]")

    # Final save
    _save_results(all_results, t_start, len(entries))

    # Print summary
    _print_summary(all_results)


def _save_results(results, t_start, total):
    """Save all result files."""
    # Per-image results
    with open(OUTPUT_DIR / "per_image.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Per-center aggregation
    center_stats = aggregate_by_center(results)
    with open(OUTPUT_DIR / "per_center.json", "w", encoding="utf-8") as f:
        json.dump(center_stats, f, indent=2, default=str)

    # Summary
    elapsed = time.time() - t_start
    summary = {
        "total_images": len(results),
        "total_time_sec": round(elapsed, 1),
        "avg_time_per_image_sec": round(elapsed / max(1, len(results)), 2),
        "centers": list(center_stats.keys()),
        "center_stats": center_stats,
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def _print_summary(results):
    """Print final summary table."""

    print(f"\n{'='*70}")
    print("ZERO-SHOT CROSS-DOMAIN EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total images: {len(results)}")
    print()

    center_stats = aggregate_by_center(results)

    # Header
    print(f"{'Center':<8} {'N':>4} {'SAM2/img':>8} {'RF-DETR/img':>11} "
          f"{'RF-prec':>8} {'RF-rec':>8} {'YOLO/img':>8} "
          f"{'YOLO-prec':>9} {'YOLO-rec':>8}")
    print("-" * 85)

    for center in sorted(center_stats.keys()):
        s = center_stats[center]
        sam2 = s["sam2_masks_per_image"]
        rfd = s["rfdetr_dets_per_image"]
        rfp = s["rfdetr_precision_proxy"]
        rfr = s["rfdetr_recall_proxy"]
        yol = s["yolo_dets_per_image"]
        yop = s["yolo_precision_proxy"]
        yor = s["yolo_recall_proxy"]

        def fmt(val, suffix=""):
            if val is None or val.get("count", 0) == 0:
                return "N/A"
            return f"{val['mean']:.2f}{suffix}"

        print(f"{center:<8} {s['n_images']:>4} "
              f"{fmt(sam2):>8} {fmt(rfd):>11} "
              f"{fmt(rfp):>8} {fmt(rfr):>8} "
              f"{fmt(yol):>8} {fmt(yop):>9} {fmt(yor):>8}")

    print()
    print("Legend:")
    print("  SAM2/img   = SAM2 pseudo-GT masks per image (higher = more organoids found)")
    print("  RF-DETR/img = RF-DETR detections per image")
    print("  RF-prec    = Precision proxy (det matched SAM2 / total det)")
    print("  RF-rec     = Recall proxy (SAM2 matched by det / total SAM2)")
    print("  YOLO/*     = Same metrics for YOLOv12")
    print()
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
