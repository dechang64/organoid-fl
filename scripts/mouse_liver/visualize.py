r"""
可视化检测结果：把检测到的类器官用绿色轮廓画在原图上，人工标注为红色
需要冬生本地跑（需要 RF-DETR 模型）

Usage:
    cd C:\Users\decha\organoid-fl
    .\.venv\Scripts\activate
    python scripts\mouse_liver\visualize.py --weights runs\mouse_liver_fewshot\checkpoint_best_regular.pth --src scripts\mouse_liver\yolo_format\images --gt scripts\mouse_liver\yolo_format\labels --annot ..\..\..\..\..\..\..\..\home\z\my-project\mouse_liver_organoid\data\新鼠肝AI --dst results\visualization
"""
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def detect_and_draw(model, img_path, threshold=0.25):
    """检测并返回轮廓"""
    img = Image.open(img_path)
    dets = model.predict(img, threshold=threshold)
    return dets

def draw_detection_contours(orig_img, detections, color=(0, 255, 0), thickness=3):
    """把检测框画成轮廓叠加到原图"""
    vis = orig_img.copy()
    if len(detections.xyxy) > 0:
        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            conf = detections.confidence[i]
            # 画矩形框
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            # 标注置信度
            label = f"{conf:.2f}"
            cv2.putText(vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return vis

def draw_gt_from_yolo(orig_img, label_path):
    """从YOLO格式标注画GT框"""
    vis = orig_img.copy()
    h, w = vis.shape[:2]
    if os.path.exists(label_path):
        with open(label_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, xc, yc, bw, bh = parts
                    xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for GT
    return vis

def draw_gt_from_red_annot(orig_img, annot_img):
    """从人工红线标注图提取轮廓画到原图"""
    vis = orig_img.copy()
    annot = cv2.cvtColor(annot_img, cv2.COLOR_RGB2BGR)
    r, g, b = annot[:,:,2].astype(int), annot[:,:,1].astype(int), annot[:,:,0].astype(int)
    red_mask = (r > 150) & (r - g > 50) & (r - b > 50)
    red_uint8 = (red_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(red_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 500]
    cv2.drawContours(vis, valid, -1, (0, 0, 255), 3)  # Red contours
    return vis

def main():
    parser = argparse.ArgumentParser(description='Visualize detection results vs GT annotations')
    parser.add_argument('--weights', required=True, help='RF-DETR checkpoint')
    parser.add_argument('--src', required=True, help='Image directory')
    parser.add_argument('--gt', default=None, help='YOLO format label directory')
    parser.add_argument('--annot', default=None, help='Human annotated image directory (red line)')
    parser.add_argument('--dst', default='results/visualization', help='Output directory')
    parser.add_argument('--model-variant', default='small', choices=['nano', 'small', 'base'])
    parser.add_argument('--threshold', type=float, default=0.25)
    args = parser.parse_args()
    
    os.makedirs(args.dst, exist_ok=True)
    
    from rfdetr import RFDETRSmall, RFDETRNano, RFDETRBase
    model_map = {'nano': RFDETRNano, 'small': RFDETRSmall, 'base': RFDETRBase}
    model = model_map[args.model_variant](pretrain_weights=args.weights, num_classes=1)
    
    img_dir = Path(args.src)
    images = sorted(img_dir.glob('*.jpg'))
    
    # Find matching annotated images
    annot_dir = Path(args.annot) if args.annot else None
    annot_images = sorted(annot_dir.glob('*.jpg')) if annot_dir else []
    
    for i, img_path in enumerate(images):
        orig = cv2.imread(str(img_path))
        dets = detect_and_draw(model, img_path, args.threshold)
        
        # 1. Detection only (green)
        det_vis = draw_detection_contours(orig, dets, color=(0, 255, 0))
        
        # 2. GT from YOLO labels (red)
        if args.gt:
            gt_path = Path(args.gt) / (img_path.stem + '.txt')
            gt_vis = draw_gt_from_yolo(orig, gt_path)
        else:
            gt_vis = orig.copy()
        
        # 3. Side-by-side comparison
        h, w = orig.shape[:2]
        canvas = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
        canvas[:, :w] = det_vis
        canvas[:, w+20:] = gt_vis
        cv2.putText(canvas, 'Detection (green)', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(canvas, 'GT (red)', (w + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        out_path = Path(args.dst) / f'vis_{img_path.name}'
        cv2.imwrite(str(out_path), canvas)
        print(f"  [{i+1}/{len(images)}] {out_path.name}: det={len(dets.xyxy)}")
    
    # Also create overlay: detection + GT on same image
    for i, img_path in enumerate(images):
        orig = cv2.imread(str(img_path))
        dets = detect_and_draw(model, img_path, args.threshold)
        
        # Draw GT first (red, thick)
        if args.gt:
            gt_path = Path(args.gt) / (img_path.stem + '.txt')
            overlay = draw_gt_from_yolo(orig, gt_path)
        else:
            overlay = orig.copy()
        
        # Draw detection on top (green, thick)
        overlay = draw_detection_contours(overlay, dets, color=(0, 255, 0))
        
        cv2.putText(overlay, 'Green=Detection  Red=GT', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        out_path = Path(args.dst) / f'overlay_{img_path.name}'
        cv2.imwrite(str(out_path), overlay)
    
    print(f"\nVisualization saved to {args.dst}/")
    print(f"  vis_*.jpg: side-by-side comparison")
    print(f"  overlay_*.jpg: detection + GT overlay")

if __name__ == '__main__':
    main()
