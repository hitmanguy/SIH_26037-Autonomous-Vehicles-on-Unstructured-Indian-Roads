"""
SAHI Engine (Slicing Aided Hyper Inference) for C3 YOLOv8 IDD Detector
Smart India Hackathon (SIH) 2026 - Problem Statement 26037
Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads

Camera pipeline slicing loop:
- Full-frame pass: 1080p frame letterboxed (aspect-preserving) to 640x640.
- Band pass: functional road bands (defined as fractions of image height so any
  resolution works) cut into 640 px wide tiles with >= 30% overlap that always
  cover the full image width. Tiles are letterboxed WITHOUT upscaling, so pixels
  reach the network at true 1:1 sensor resolution.
    1. Far Band  (40 - 150 m): rows [400, 760] px on 1080p
    2. Near Band (15 - 30 m):  rows [600, 1000] px on 1080p
- Vectorized YOLOv8 head decoding (score pre-filter in logit space before DFL).
- Per-inference class-aware NMS, then a cross-tile merge that uses IoU *and*
  intersection-over-smaller (IoS) so boxes cut by a tile edge are suppressed or
  union-merged instead of being counted as extra objects.
- Flat-ground IPM range estimate per detection for handoff to Sensor_fusion.
- Exports structured detection results for MATLAB sensor fusion and visualization.
"""

import argparse
import glob
import os
import time

import numpy as np
import onnxruntime as ort
import scipy.io as sio
from PIL import Image

CLASSES = ['person', 'rider', 'car', 'bus', 'truck', 'autorickshaw',
           'motorcycle', 'bicycle', 'animal', 'traffic sign', 'traffic light', 'vehicle fallback']
classes = CLASSES  # backwards-compatible name

INPUT_SIZE = 640
STRIDES = (8, 16, 32)
PAD_VALUE = 114  # Ultralytics letterbox grey

# (name, y_top, y_bottom) as fractions of image height; 400/760 and 600/1000 px on 1080p
BANDS = (('far', 400 / 1080, 760 / 1080),
         ('near', 600 / 1080, 1000 / 1080))
# --fast: one 600 px band covering both (still <= 640 rows, so still 1:1). Half the
# tiles (~2x faster SAHI pass) at the cost of a few far/near boundary detections.
BANDS_FAST = (('road', 400 / 1080, 1000 / 1080),)
TILE_W = 640
MIN_TILE_OVERLAP = 0.30

CONF_THRESH = 0.25
NMS_IOU = 0.45        # per-inference NMS
MERGE_IOU = 0.45      # cross-tile merge
MERGE_IOS = 0.60      # intersection over the smaller box
NEW_IOU = 0.35        # "is new vs full-frame" matching
FRAGMENT_AREA_RATIO = 0.5  # IoS merging only for pairs this lopsided (or truncated)
EDGE_PX = 3.0         # a box this close to an interior tile edge is treated as truncated
TRUNC_PENALTY = 0.5   # truncated boxes rank below complete ones during merging

# Camera model for flat-ground inverse perspective mapping (matches Sensor_fusion)
CAM_H, FY, CY = 1.5, 1200.0, 540.0

HERE = os.path.dirname(os.path.abspath(__file__))
DETECTOR_DIR = os.path.join('C3_detector_v1', 'C3_detector_v1')
SEARCH_ROOTS = [os.getcwd(), HERE, os.path.join(HERE, '..'), os.path.join(HERE, '..', '..')]

TEST_IMAGES = [
    'frontNear__BLR-2018-04-19_18-06-55_frontNear__0000060.jpg',
    'highquality_16k__BLR-2018-05-31_10-49-32__2018-05-31_10-58-6-131756_leftImg8bit.jpg',
    'rearNear__BLR-2018-06-06_16-41-17_rearNear__0000330.jpg',
    'sideLeft__BLR-2018-05-14_13-38-46_sideLeft__000264_r.jpg',
]


def find_detector_path(*parts):
    for root in SEARCH_ROOTS:
        p = os.path.normpath(os.path.join(root, DETECTOR_DIR, *parts))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find {os.path.join(DETECTOR_DIR, *parts)} "
                            f"under any of {SEARCH_ROOTS}")


# ---------------------------------------------------------------------------
# Box utilities (all boxes are float xyxy)
# ---------------------------------------------------------------------------

def _pairwise_overlap(a, b):
    """Return (IoU, IoS) matrices between box sets a (N,4) and b (M,4)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    smaller = np.minimum(area_a[:, None], area_b[None, :])
    iou = inter / np.maximum(union, 1e-9)
    ios = inter / np.maximum(smaller, 1e-9)
    return iou, ios


def multiclass_nms(boxes, scores, cls, iou_thr=NMS_IOU):
    """Class-aware greedy NMS. Returns kept indices sorted by descending score."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(-scores)
    iou, _ = _pairwise_overlap(boxes[order], boxes[order])
    same = cls[order][:, None] == cls[order][None, :]
    suppress = same & (iou > iou_thr)
    alive = np.ones(len(order), dtype=bool)
    for i in range(len(order)):
        if alive[i]:
            alive[i + 1:] &= ~suppress[i, i + 1:]
    return order[alive]


def merge_slices(boxes, scores, cls, truncated, source,
                 iou_thr=MERGE_IOU, ios_thr=MERGE_IOS):
    """Greedy cross-tile merge.

    Boxes of the same class from *different* inferences (source ids) are grouped
    if IoU > iou_thr, or if IoS > ios_thr and the pair looks like a fragment (one
    box truncated by a tile edge, or much smaller than the other). The fragment
    condition keeps tightly packed distinct objects (parked two-wheelers) apart.
    Complete boxes outrank truncated ones; a truncated group leader is grown to
    the union of its truncated partners, reconstructing objects cut by tiles.
    Returns (kept indices, merged boxes for those indices).
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=int), np.zeros((0, 4))
    rank = scores * np.where(truncated, TRUNC_PENALTY, 1.0)
    order = np.argsort(-rank)
    iou, ios = _pairwise_overlap(boxes, boxes)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ratio = np.minimum(area[:, None], area[None, :]) / np.maximum(np.maximum(area[:, None], area[None, :]), 1e-9)
    fragment = truncated[:, None] | truncated[None, :] | (ratio < FRAGMENT_AREA_RATIO)
    match = ((cls[:, None] == cls[None, :]) & (source[:, None] != source[None, :])
             & ((iou > iou_thr) | ((ios > ios_thr) & fragment)))
    done = np.zeros(len(boxes), dtype=bool)
    keep, merged = [], []
    for i in order:
        if done[i]:
            continue
        group = match[i] & ~done
        group[i] = True
        done |= group
        box = boxes[i].copy()
        if truncated[i]:
            parts = boxes[group & truncated]
            box[:2] = parts[:, :2].min(0)
            box[2:] = parts[:, 2:].max(0)
        keep.append(i)
        merged.append(box)
    return np.array(keep, dtype=int), np.array(merged)


def tile_starts(length, tile, min_overlap=MIN_TILE_OVERLAP):
    """Evenly spaced tile origins covering [0, length) with >= min_overlap."""
    if length <= tile:
        return [0]
    n = int(np.ceil((length - tile) / (tile * (1.0 - min_overlap)))) + 1
    return [int(round(s)) for s in np.linspace(0, length - tile, n)]


def ground_range(v_bottom):
    """Flat-ground IPM longitudinal range (m) of a box's ground-contact row."""
    return CAM_H * FY / np.maximum(v_bottom - CY, 1.0)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class C3Detector:
    def __init__(self, onnx_path=None, conf=CONF_THRESH):
        onnx_path = onnx_path or find_detector_path('best_matlab.onnx')
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [p for p in ('CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider')
                     if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(onnx_path, so, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.logit_thr = float(np.log(conf / (1.0 - conf)))
        self.proj = np.arange(16, dtype=np.float32)[:, None]
        self.blob = np.empty((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)

    def _letterbox(self, img):
        """Aspect-preserving resize (never upscales) into a padded 640x640 blob.

        img is an HxWx3 uint8 RGB array or a PIL image (avoids a copy for full frames).
        """
        w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])
        r = min(INPUT_SIZE / w, INPUT_SIZE / h, 1.0)
        nw, nh = int(round(w * r)), int(round(h * r))
        if (nw, nh) != (w, h):
            pim = img if isinstance(img, Image.Image) else Image.fromarray(img)
            img = pim.resize((nw, nh), Image.BILINEAR)
        img = np.asarray(img)
        left, top = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
        blob = self.blob
        blob.fill(PAD_VALUE / 255.0)
        np.multiply(img.transpose(2, 0, 1), np.float32(1.0 / 255.0),
                    out=blob[0, :, top:top + nh, left:left + nw])
        return blob, r, left, top

    def _decode(self, outs):
        boxes, scores, cls = [], [], []
        for p, s in zip(outs, STRIDES):
            _, c, h, w = p.shape
            p = p[0].reshape(c, h * w)
            logits = p[64:]
            best = logits.argmax(0)
            best_logit = logits[best, np.arange(h * w)]
            idx = np.nonzero(best_logit > self.logit_thr)[0]
            if idx.size == 0:
                continue
            dfl = p[:64, idx].reshape(4, 16, -1)
            dfl = np.exp(dfl - dfl.max(1, keepdims=True))
            ltrb = (dfl * self.proj).sum(1) / dfl.sum(1) * s
            gx = (idx % w + 0.5) * s
            gy = (idx // w + 0.5) * s
            boxes.append(np.stack([gx - ltrb[0], gy - ltrb[1], gx + ltrb[2], gy + ltrb[3]], 1))
            scores.append(1.0 / (1.0 + np.exp(-best_logit[idx])))
            cls.append(best[idx])
        if not boxes:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        return np.concatenate(boxes), np.concatenate(scores), np.concatenate(cls)

    def detect(self, img):
        """Detect on an RGB uint8 array or PIL image. Returns NMS'd xyxy boxes in img pixels."""
        blob, r, left, top = self._letterbox(img)
        boxes, scores, cls = self._decode(self.sess.run(None, {self.input_name: blob}))
        keep = multiclass_nms(boxes, scores, cls)
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        boxes = (boxes - [left, top, left, top]) / r
        w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes, scores, cls


def _to_xywh(b):
    return np.column_stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]) if len(b) else np.zeros((0, 4))


def process_image(img_path, det=None, bands=BANDS):
    det = det or _default_detector()
    pim = Image.open(img_path).convert('RGB')
    im = np.asarray(pim)
    H, W = im.shape[:2]

    # 1. Full-frame baseline (fast loop)
    t0 = time.perf_counter()
    ff_b, ff_s, ff_c = det.detect(pim)
    t_ff = time.perf_counter() - t0

    # 2. Band slicing (slow loop)
    t0 = time.perf_counter()
    all_b, all_s, all_c, all_t = [ff_b], [ff_s], [ff_c], [np.zeros(len(ff_b), dtype=bool)]
    all_src = [np.zeros(len(ff_b), dtype=int)]
    n_tiles = 0
    for _, fy1, fy2 in bands:
        y1, y2 = int(round(fy1 * H)), int(round(fy2 * H))
        for sx in tile_starts(W, TILE_W):
            ex = min(sx + TILE_W, W)
            b, s, c = det.detect(im[y1:y2, sx:ex])
            n_tiles += 1
            # Truncated = touching a tile edge that is not also an image edge
            t = np.zeros(len(b), dtype=bool)
            if sx > 0:
                t |= b[:, 0] <= EDGE_PX
            if ex < W:
                t |= b[:, 2] >= (ex - sx) - EDGE_PX
            if y1 > 0:
                t |= b[:, 1] <= EDGE_PX
            if y2 < H:
                t |= b[:, 3] >= (y2 - y1) - EDGE_PX
            all_b.append(b + [sx, y1, sx, y1])
            all_s.append(s)
            all_c.append(c)
            all_t.append(t)
            all_src.append(np.full(len(b), n_tiles))
    sb, ss, sc, st, src = map(np.concatenate, (all_b, all_s, all_c, all_t, all_src))
    keep, merged = merge_slices(sb, ss, sc, st, src)
    t_sahi = time.perf_counter() - t0

    # 3. Which SAHI detections have no same-class full-frame counterpart?
    s_cls = sc[keep]
    if len(keep) and len(ff_b):
        iou, ios = _pairwise_overlap(merged, ff_b)
        hit = (s_cls[:, None] == ff_c[None, :]) & ((iou > NEW_IOU) | (ios > MERGE_IOS))
        is_new = ~hit.any(1)
    else:
        is_new = np.ones(len(keep), dtype=bool)

    return {
        'ff_boxes': _to_xywh(ff_b),
        'ff_scores': ff_s,
        'ff_labels': np.array([CLASSES[c] for c in ff_c], dtype=object),
        'ff_cls': ff_c + 1,  # 1-indexed for MATLAB
        'ff_range_m': ground_range(ff_b[:, 3]) if len(ff_b) else np.zeros(0),
        'sahi_boxes': _to_xywh(merged),
        'sahi_scores': ss[keep],
        'sahi_labels': np.array([CLASSES[c] for c in s_cls], dtype=object),
        'sahi_cls': s_cls + 1,
        'sahi_range_m': ground_range(merged[:, 3]) if len(merged) else np.zeros(0),
        'is_new_sahi': is_new.astype(bool),
        'n_tiles': n_tiles,
        't_fullframe_ms': 1000 * t_ff,
        't_sahi_ms': 1000 * t_sahi,
    }


_DET = None


def _default_detector():
    global _DET
    if _DET is None:
        _DET = C3Detector()
    return _DET


def main():
    ap = argparse.ArgumentParser(description='C3 YOLOv8 + SAHI dual-band slicing engine')
    ap.add_argument('images', nargs='*', help='images or folders (default: the 4 IDD test frames)')
    ap.add_argument('--onnx', default=None, help='path to best_matlab.onnx')
    ap.add_argument('--conf', type=float, default=CONF_THRESH)
    ap.add_argument('--fast', action='store_true', help='single merged road band (half the tiles)')
    ap.add_argument('--out', default='sahi_detection_results.mat')
    args = ap.parse_args()

    if args.images:
        files = []
        for p in args.images:
            files += sorted(glob.glob(os.path.join(p, '*.jpg')) + glob.glob(os.path.join(p, '*.png'))) \
                if os.path.isdir(p) else [p]
    else:
        img_dir = find_detector_path('test_images')
        files = [os.path.join(img_dir, n) for n in TEST_IMAGES]

    print("=================================================================")
    print("  Executing SAHI Slicing Engine on India Driving Dataset (IDD)   ")
    print("=================================================================")
    det = C3Detector(args.onnx, conf=args.conf)
    print(f">> ONNX Runtime providers: {det.sess.get_providers()}")
    det.detect(np.zeros((360, 640, 3), dtype=np.uint8))  # warm-up

    export_dict = {}
    for f in files:
        fname = os.path.basename(f)
        safe_key = fname.split('.')[0].replace('-', '_')
        print(f">> Processing: {fname}...")
        res = process_image(f, det, BANDS_FAST if args.fast else BANDS)
        print(f"   Full-Frame: {len(res['ff_boxes']):2d} ({res['t_fullframe_ms']:5.1f} ms) | "
              f"SAHI Total: {len(res['sahi_boxes']):2d} ({res['n_tiles']} tiles, {res['t_sahi_ms']:6.1f} ms) | "
              f"New Distant Objects: +{int(res['is_new_sahi'].sum()):2d}")
        export_dict[safe_key] = res

    sio.savemat(args.out, export_dict)
    print(f"\n>> Successfully generated and saved {args.out}!")


if __name__ == '__main__':
    main()
