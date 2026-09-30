#!/usr/bin/env python3
"""
C3 before/after: stock COCO YOLOv8s vs our IDD-fine-tuned YOLOv8s.
SIH PS26037 · Team Epsilon · Perception (camera)

1. Per-class AP50 on the same validation images for BOTH models, computed by
   the same code, so the comparison is apples to apples.
   The COCO model's classes are mapped onto ours (cow/dog/horse... -> animal, etc).
   Classes COCO has no equivalent for (rider, autorickshaw, vehicle fallback)
   score 0 for the stock model by construction.
2. Side-by-side images (stock | ours) for the 15 Q1 frames.

Run on the server inside the sih environment:
  python ~/epsilon_yash/c3_compare.py
Outputs go to ~/epsilon_yash/runs/c3_compare/
"""
import argparse
import csv
from pathlib import Path

import numpy as np

HOME = Path.home() / "epsilon_yash"
OUR_CLASSES = ["person", "rider", "car", "bus", "truck", "autorickshaw", "motorcycle",
               "bicycle", "animal", "traffic sign", "traffic light", "vehicle fallback"]
# COCO name -> our class id (anything not listed is ignored)
COCO_TO_OURS = {
    "person": 0, "car": 2, "bus": 3, "truck": 4, "motorcycle": 6, "bicycle": 7,
    "traffic light": 10, "stop sign": 9,
    "bird": 8, "cat": 8, "dog": 8, "horse": 8, "sheep": 8, "cow": 8,
    "elephant": 8, "bear": 8, "zebra": 8, "giraffe": 8,
}


def iou_one_to_many(b, B):
    x1 = np.maximum(b[0], B[:, 0]); y1 = np.maximum(b[1], B[:, 1])
    x2 = np.minimum(b[2], B[:, 2]); y2 = np.minimum(b[3], B[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_B = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / np.maximum(area_b + area_B - inter, 1e-9)


def ap50(dets, gts, n_gt):
    """dets: list of (img_id, score, box); gts: {img_id: array Nx4}; VOC all-point AP at IoU 0.5."""
    if n_gt == 0:
        return float("nan")
    if not dets:
        return 0.0
    dets = sorted(dets, key=lambda d: -d[1])
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    for i, (img, _, box) in enumerate(dets):
        g = gts.get(img)
        if g is None or len(g) == 0:
            fp[i] = 1; continue
        ious = iou_one_to_many(box, g)
        j = int(np.argmax(ious))
        if ious[j] >= 0.5 and not used[img][j]:
            tp[i] = 1; used[img][j] = True
        else:
            fp[i] = 1
    tp = np.cumsum(tp); fp = np.cumsum(fp)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def load_gt(img_paths, label_dir, shapes):
    gts = {c: {} for c in range(len(OUR_CLASSES))}
    n_gt = np.zeros(len(OUR_CLASSES), int)
    for p in img_paths:
        h, w = shapes[p]
        lbl = label_dir / (Path(p).stem + ".txt")
        rows = [r.split() for r in lbl.read_text().splitlines() if r.strip()] if lbl.exists() else []
        per = {}
        for r in rows:
            c = int(r[0]); cx, cy, bw, bh = map(float, r[1:5])
            box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
            per.setdefault(c, []).append(box)
        for c, boxes in per.items():
            gts[c][p] = np.array(boxes)
            n_gt[c] += len(boxes)
    return gts, n_gt


def run_model(model, img_paths, mapping, batch):
    """Returns {class_id: [(img, score, box), ...]} and {img: (h, w)}."""
    dets = {c: [] for c in range(len(OUR_CLASSES))}
    shapes = {}
    for i in range(0, len(img_paths), batch):
        chunk = img_paths[i:i + batch]
        for r in model.predict(chunk, imgsz=640, conf=0.001, iou=0.7, verbose=False):
            p = r.path
            shapes[p] = r.orig_shape
            b = r.boxes
            for box, s, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
                ours = mapping(c)
                if ours is not None:
                    dets[ours].append((p, float(s), box))
        print(f"  {min(i + batch, len(img_paths))}/{len(img_paths)}", end="\r")
    print()
    return dets, shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HOME / "yolo_idd"))
    ap.add_argument("--ours", default=str(HOME / "runs/c3_yolov8s_e60/weights/best.pt"))
    ap.add_argument("--coco", default=str(HOME / "yolov8s.pt"))
    ap.add_argument("--frames", default=str(HOME / "q1_frames"))
    ap.add_argument("--out", default=str(HOME / "runs/c3_compare"))
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    from ultralytics import YOLO
    import cv2

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(str(p) for p in (Path(a.data) / "images/val").glob("*.jpg"))
    label_dir = Path(a.data) / "labels/val"
    print(f"{len(img_paths)} validation images")

    coco = YOLO(a.coco); ours = YOLO(a.ours)
    coco_names = coco.names
    print("Stock COCO model ...")
    d_coco, shapes = run_model(coco, img_paths, lambda c: COCO_TO_OURS.get(coco_names[c]), a.batch)
    print("Our fine-tuned model ...")
    d_ours, _ = run_model(ours, img_paths, lambda c: c, a.batch)

    gts, n_gt = load_gt(img_paths, label_dir, shapes)
    rows = []
    print(f"\n{'class':<18}{'GT boxes':>9}{'stock AP50':>12}{'ours AP50':>11}")
    for c, name in enumerate(OUR_CLASSES):
        a_coco = ap50(d_coco[c], gts[c], n_gt[c])
        a_ours = ap50(d_ours[c], gts[c], n_gt[c])
        note = "" if name not in ("rider", "autorickshaw", "vehicle fallback") else "  (no COCO class)"
        print(f"{name:<18}{n_gt[c]:>9}{a_coco:>12.3f}{a_ours:>11.3f}{note}")
        rows.append([name, int(n_gt[c]), round(a_coco, 4), round(a_ours, 4)])
    m_coco = np.nanmean([r[2] for r in rows]); m_ours = np.nanmean([r[3] for r in rows])
    print(f"{'mean (12 classes)':<18}{'':>9}{m_coco:>12.3f}{m_ours:>11.3f}")
    with open(out / "ap50_before_after.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class", "gt_boxes", "stock_coco_ap50", "ours_ap50"]); w.writerows(rows)
        w.writerow(["mean", "", round(m_coco, 4), round(m_ours, 4)])

    # Side-by-side images for the Q1 frames (confidence 0.25, like Q1)
    sbs = out / "side_by_side"; sbs.mkdir(exist_ok=True)
    frames = sorted(Path(a.frames).glob("*.jpg"))
    for fp in frames:
        left = coco.predict(str(fp), imgsz=640, conf=0.25, verbose=False)[0].plot()
        right = ours.predict(str(fp), imgsz=640, conf=0.25, verbose=False)[0].plot()
        for img, txt in ((left, "Stock YOLOv8s (COCO)"), (right, "Ours (fine-tuned on IDD)")):
            cv2.rectangle(img, (0, 0), (img.shape[1], 70), (255, 255, 255), -1)
            cv2.putText(img, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
        cv2.imwrite(str(sbs / f"{fp.stem}_compare.jpg"), cv2.hconcat([left, right]))
    print(f"\nSaved: {out / 'ap50_before_after.csv'} and {len(frames)} side-by-side images in {sbs}")


if __name__ == "__main__":
    main()
