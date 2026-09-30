#!/usr/bin/env python3
"""
Export our fine-tuned YOLOv8 (.pt) to an ONNX file that MATLAB's yolov8ObjectDetector understands.
SIH PS26037 · Team Epsilon · Perception (camera) · checkpoint C3

Why this exists
  New Ultralytics versions (8.3+) build the detection head's output differently from the
  2024 version MathWorks' importer was written for. MATLAB wants the network to stop at
  3 feature maps (one per scale: 80x80, 40x40, 20x20), each = [64 box channels + N class
  channels], class values NOT yet passed through sigmoid. MATLAB does the rest itself.
  So we cut the model there ourselves and export only that part.

Usage (inside the sih environment, on the server):
  python ~/epsilon_yash/export_for_matlab.py \
      --weights ~/epsilon_yash/runs/c3_yolov8s_e60/weights/best.pt
Output: best_matlab.onnx next to best.pt
"""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=14)
    a = ap.parse_args()

    from ultralytics import YOLO

    w = Path(a.weights).expanduser().resolve()
    yolo = YOLO(str(w))
    model = yolo.model.float().eval()
    model.fuse(verbose=False)                    # same fused graph Ultralytics exports
    for p in model.parameters():
        p.requires_grad_(False)

    det = model.model[-1]                        # Detect head
    nc, nl = det.nc, det.nl
    print(f"classes: {nc}  scales: {nl}  names: {list(yolo.names.values())}")

    def head_forward(x):                         # per scale: cat(box 64ch, class nc ch), raw logits
        return tuple(torch.cat((det.cv2[i](x[i]), det.cv3[i](x[i])), 1) for i in range(nl))
    det.forward = head_forward

    dummy = torch.zeros(1, 3, a.imgsz, a.imgsz)
    with torch.no_grad():
        outs = model(dummy)
    for i, o in enumerate(outs):
        print(f"  output {i}: {tuple(o.shape)}")   # expect (1, 64+nc, 80, 80) / 40 / 20

    out_path = w.with_name(w.stem + "_matlab.onnx")
    torch.onnx.export(model, dummy, str(out_path), opset_version=a.opset,
                      input_names=["images"], output_names=["p3", "p4", "p5"],
                      do_constant_folding=True, dynamo=False)
    try:                                          # tidy the graph (same tool Ultralytics uses)
        import onnx, onnxslim
        onnx.save(onnxslim.slim(onnx.load(str(out_path))), str(out_path))
    except Exception as e:                        # optional; the unslimmed file also works
        print(f"(onnxslim skipped: {e})")

    # Sanity check: ONNX Runtime gives the same numbers as PyTorch
    try:
        import numpy as np, onnxruntime as ort
        x = torch.rand(1, 3, a.imgsz, a.imgsz)
        with torch.no_grad():
            ref = [t.numpy() for t in model(x)]
        got = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"]).run(None, {"images": x.numpy()})
        print("max difference PyTorch vs ONNX:", max(float(np.abs(r - g).max()) for r, g in zip(ref, got)))
    except Exception as e:
        print(f"(onnxruntime check skipped: {e})")

    print(f"\nSaved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
