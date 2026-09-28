"""
SAHI Engine (Slicing Aided Hyper Inference) for C3 YOLOv8 IDD Detector
Smart India Hackathon (SIH) 2026 - Problem Statement 26037
Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads

This module implements the camera pipeline slicing loop:
- Slices 1080p images into 2 functional road bands:
  1. Far Band (40 - 150m): Upper horizon rows [400 - 760 px] sliced into 640x640 overlapping tiles.
  2. Pothole / Near Band (15 - 30m): Mid-lower rows [620 - 1020 px].
- Runs ONNX YOLOv8s inference on each high-resolution tile.
- Remaps tile coordinates [u + x_offset, v + y_offset] to 1080p image coordinates.
- Applies Multi-Class Non-Maximum Suppression (NMS) to fuse full-frame + sliced boxes.
- Exports structured detection results for MATLAB sensor fusion and visualization.
"""

import os, glob
from PIL import Image
import numpy as np, onnxruntime as ort, scipy.io as sio

classes = ['person', 'rider', 'car', 'bus', 'truck', 'autorickshaw', 
           'motorcycle', 'bicycle', 'animal', 'traffic sign', 'traffic light', 'vehicle fallback']

candidate_onnx = [
    'C3_detector_v1/C3_detector_v1/best_matlab.onnx',
    '../C3_detector_v1/C3_detector_v1/best_matlab.onnx',
    '../../C3_detector_v1/C3_detector_v1/best_matlab.onnx',
    'c:/Users/prana/Downloads/sih/C3_detector_v1/C3_detector_v1/best_matlab.onnx'
]
onnx_path = None
for cp in candidate_onnx:
    if os.path.isfile(cp):
        onnx_path = cp
        break
assert onnx_path is not None, f"Cannot find best_matlab.onnx in candidate paths"

sess = ort.InferenceSession(onnx_path)

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25, 25)))

def decode_head(p, stride):
    bs, c, h, w = p.shape
    dfl = p[0, :64, :, :].reshape(4, 16, h, w)
    e_dfl = np.exp(dfl - np.max(dfl, axis=1, keepdims=True))
    dfl_weights = e_dfl / np.sum(e_dfl, axis=1, keepdims=True)
    conv_proj = np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)
    ltrb = np.sum(dfl_weights * conv_proj, axis=1) * stride
    gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    gx = (gx + 0.5) * stride
    gy = (gy + 0.5) * stride
    x1 = gx - ltrb[0]
    y1 = gy - ltrb[1]
    x2 = gx + ltrb[2]
    y2 = gy + ltrb[3]
    cls_scores = sigmoid(p[0, 64:, :, :])
    return x1, y1, x2, y2, cls_scores

def run_inference_on_crop(im_crop):
    cw, ch = im_crop.size
    im_res = im_crop.resize((640, 640))
    blob = (np.array(im_res, dtype=np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
    outs = sess.run(None, {'images': blob})
    boxes, scores, clses = [], [], []
    for p, s in zip(outs, [8, 16, 32]):
        x1, y1, x2, y2, cls_scores = decode_head(p, s)
        max_cls = np.max(cls_scores, axis=0)
        max_idx = np.argmax(cls_scores, axis=0)
        mask = max_cls > 0.25
        if np.any(mask):
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                boxes.append([float(x1[y,x]*cw/640), float(y1[y,x]*ch/640), 
                              float(x2[y,x]*cw/640), float(y2[y,x]*ch/640)])
                scores.append(float(max_cls[y,x]))
                clses.append(int(max_idx[y,x]))
    return boxes, scores, clses

def multiclass_nms(b, s, c, thresh=0.45):
    if len(b) == 0: return []
    b, s, c = np.array(b), np.array(s), np.array(c)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(b[i, 0], b[order[1:], 0])
        yy1 = np.maximum(b[i, 1], b[order[1:], 1])
        xx2 = np.minimum(b[i, 2], b[order[1:], 2])
        yy2 = np.minimum(b[i, 3], b[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        denom = (b[i, 2]-b[i, 0])*(b[i, 3]-b[i, 1]) + (b[order[1:], 2]-b[order[1:], 0])*(b[order[1:], 3]-b[order[1:], 1]) - inter
        same_cls = (c[order[1:]] == c[i])
        ovr = inter / denom
        inds = np.where(~(same_cls & (ovr > thresh)))[0]
        order = order[inds + 1]
    return [int(k) for k in keep]

def process_image(img_path):
    im = Image.open(img_path)
    w, h = im.size
    
    # 1. Full-frame baseline
    ff_b, ff_s, ff_c = run_inference_on_crop(im)
    ff_keep = multiclass_nms(ff_b, ff_s, ff_c)
    
    ff_boxes_final = []
    ff_scores_final = []
    ff_labels_final = []
    ff_cls_final = []
    for k in ff_keep:
        b = ff_b[k]
        ff_boxes_final.append([b[0], b[1], b[2] - b[0], b[3] - b[1]])
        ff_scores_final.append(ff_s[k])
        ff_labels_final.append(classes[ff_c[k]])
        ff_cls_final.append(ff_c[k] + 1) # 1-indexed for MATLAB
        
    # 2. SAHI Slicing
    # Start with full-frame detections to retain large near objects
    sahi_b, sahi_s, sahi_c = list(ff_b), list(ff_s), list(ff_c)
    
    # Road Band 1: Far Band (40 - 150m horizon: rows 400 - 760)
    far_y1, far_y2 = 400, 760
    slice_w = 640
    step_x = 420  # 35% overlap
    for sx in range(0, w - slice_w + 1, step_x):
        crop = im.crop((sx, far_y1, sx + slice_w, far_y2))
        cb, cs, cc = run_inference_on_crop(crop)
        for b, s, c in zip(cb, cs, cc):
            sahi_b.append([b[0] + sx, b[1] + far_y1, b[2] + sx, b[3] + far_y1])
            sahi_s.append(s)
            sahi_c.append(c)
            
    # Road Band 2: Pothole / Mid-Near Band (15 - 30m: rows 600 - 1000)
    near_y1, near_y2 = 600, 1000
    for sx in range(0, w - slice_w + 1, step_x):
        crop = im.crop((sx, near_y1, sx + slice_w, near_y2))
        cb, cs, cc = run_inference_on_crop(crop)
        for b, s, c in zip(cb, cs, cc):
            sahi_b.append([b[0] + sx, b[1] + near_y1, b[2] + sx, b[3] + near_y1])
            sahi_s.append(s)
            sahi_c.append(c)

    sahi_keep = multiclass_nms(sahi_b, sahi_s, sahi_c)
    
    sahi_boxes_final = []
    sahi_scores_final = []
    sahi_labels_final = []
    sahi_cls_final = []
    is_new_sahi = []
    
    # Determine which detections are unique to SAHI (not caught by full-frame)
    for k in sahi_keep:
        sb = sahi_b[k]
        s_box = [sb[0], sb[1], sb[2] - sb[0], sb[3] - sb[1]]
        sahi_boxes_final.append(s_box)
        sahi_scores_final.append(sahi_s[k])
        sahi_labels_final.append(classes[sahi_c[k]])
        sahi_cls_final.append(sahi_c[k] + 1)
        
        # Check IoU overlap with any full-frame box of same class
        matched = False
        for fb_idx, fb in enumerate(ff_boxes_final):
            if ff_cls_final[fb_idx] == (sahi_c[k] + 1):
                xx1 = max(s_box[0], fb[0])
                yy1 = max(s_box[1], fb[1])
                xx2 = min(s_box[0] + s_box[2], fb[0] + fb[2])
                yy2 = min(s_box[1] + s_box[3], fb[1] + fb[3])
                w_ovr = max(0, xx2 - xx1)
                h_ovr = max(0, yy2 - yy1)
                inter = w_ovr * h_ovr
                denom = (s_box[2] * s_box[3]) + (fb[2] * fb[3]) - inter
                if denom > 0 and (inter / denom) > 0.35:
                    matched = True
                    break
        is_new_sahi.append(not matched)
        
    return {
        'ff_boxes': np.array(ff_boxes_final),
        'ff_scores': np.array(ff_scores_final),
        'ff_labels': np.array(ff_labels_final, dtype=object),
        'ff_cls': np.array(ff_cls_final),
        'sahi_boxes': np.array(sahi_boxes_final),
        'sahi_scores': np.array(sahi_scores_final),
        'sahi_labels': np.array(sahi_labels_final, dtype=object),
        'sahi_cls': np.array(sahi_cls_final),
        'is_new_sahi': np.array(is_new_sahi, dtype=bool)
    }

if __name__ == '__main__':
    print("=================================================================")
    print("  Executing SAHI Slicing Engine on India Driving Dataset (IDD)   ")
    print("=================================================================")
    
    raw_names = [
        'frontNear__BLR-2018-04-19_18-06-55_frontNear__0000060.jpg',
        'highquality_16k__BLR-2018-05-31_10-49-32__2018-05-31_10-58-6-131756_leftImg8bit.jpg',
        'rearNear__BLR-2018-06-06_16-41-17_rearNear__0000330.jpg',
        'sideLeft__BLR-2018-05-14_13-38-46_sideLeft__000264_r.jpg'
    ]
    
    img_dirs = [
        'C3_detector_v1/C3_detector_v1/test_images',
        '../C3_detector_v1/C3_detector_v1/test_images',
        '../../C3_detector_v1/C3_detector_v1/test_images',
        'c:/Users/prana/Downloads/sih/C3_detector_v1/C3_detector_v1/test_images'
    ]
    img_dir = next((d for d in img_dirs if os.path.isdir(d)), None)
    assert img_dir is not None, "Cannot find test_images directory in candidate paths"
    
    test_files = [os.path.join(img_dir, n) for n in raw_names]
    
    export_dict = {}
    for f in test_files:
        fname = os.path.basename(f)
        safe_key = fname.split('.')[0].replace('-', '_')
        print(f">> Processing: {fname}...")
        res = process_image(f)
        
        num_ff = len(res['ff_boxes'])
        num_sahi = len(res['sahi_boxes'])
        num_new = np.sum(res['is_new_sahi'])
        print(f"   Full-Frame: {num_ff:2d} | SAHI Total: {num_sahi:2d} | New Distant Objects: +{num_new:2d}")
        
        export_dict[safe_key] = res
        
    sio.savemat('sahi_detection_results.mat', export_dict)
    print("\n>> Successfully generated and saved sahi_detection_results.mat!")
