# SAHI in MATLAB / Simulink (A3)

MATLAB port of `../sahi_engine.py`, so SAHI runs inside the Simulink closed loop with no Python at runtime.
Python is used only as a test oracle (`py_reference/`).

## Files
| File | What it does |
|---|---|
| `sahiDetect.m` | Full-frame pass + band tiles + per-inference NMS + fragment-aware merge. Runs `det.Network` directly with our own letterbox and YOLOv8 decode (does **not** call `detect()`, see note below). Output: pixel boxes, scores, labels, `info` |
| `sahiDefaultOpts.m` | Modes: `'v1'` (one band from camera geometry, default), `'dual'` / `'fast'` (same bands as `sahi_engine.py`), `'full'` (full frame only) |
| `sahiBandRows.m`, `sahiCameraPlaceholder.m` | Band rows (15–150 m) from a **placeholder** camera (1920x1080, f 1200 px, h 1.5 m, pitch 0) until A1 supplies the real camera |
| `SahiSlowLoop.m`, `C3FullFrame.m`, `packDetections.m` | Simulink System objects: 30 Hz full-frame loop + 5 Hz SAHI loop, fixed-size outputs (max 128 boxes) |
| `ImageSequenceSource.m` | Test camera for Simulink (image folder); replaced by the Unreal camera in A2 |
| `test_B1..B4_*.m`, `parityMatch.m`, `evalMatch.m`, `apAllPoint.m` | The four test blocks below |

## Results (MATLAB R2025b, RTX 4050 laptop 6 GB) — logs in `results/`
**B1 · Parity with `sahi_engine.py`:** every box matched one-to-one (dual 64→94, fast 64→85; min IoU 0.985, max score diff 0.03 from `imresize` vs PIL).

**B2 · Speed** (median per 1080p frame):
| Set-up | ms |
|---|---:|
| Full frame (30 Hz loop) | 58 |
| Dual bands, one call per tile (= Python engine design) | 720 |
| v1 band, 640xN tiles, batched, reusing the full-frame pass | **84** (budget 200 ms at 5 Hz) |

Per-call overhead of the imported network (~50 ms) dominates, so batching tiles is the main win. 30 Hz full-frame needs GPU Coder / TensorRT (next step).

**B3 · Accuracy** on 1,127 IDD val front-camera frames (IoU 0.5, conf 0.25):
| Road-user recall by box height | <16 px | 16–32 | 32–64 | 64–128 | ≥128 |
|---|---:|---:|---:|---:|---:|
| Full frame only | 0.9% | 17.3% | 47.6% | 74.1% | 91.0% |
| SAHI, per-camera rows, 640xN tiles | **25.8%** | **47.2%** | **66.6%** | 79.8% | 91.5% |

mAP50 0.403 → 0.507; cost: false boxes 1.69 → 4.10 per frame (precision 82.7% → 72.1%).
**B3b:** a stricter score (0.35) only for boxes found by tiles alone gives 3.0 FP/frame at 42.6% recall (16–32 px) — default in `'v1'` mode.

**B4 · Simulink:** `sahi_B4_demo.slx` (built by `test_B4_simulink.m`): 30 Hz full-frame + 5 Hz SAHI via rate transitions; all 11 SAHI ticks identical to calling `sahiDetect` directly. Runs interpreted (0.07x real time) — fine for simulation.

## Notes
- The YOLO add-on's `detect(det, I, Threshold=0.25)` silently drops boxes scoring below ~0.5 and pads to 640x384. Use `sahiDetect` (or `sahiDefaultOpts('full')`) for anything that needs low scores (e.g. potholes).
- Bands are for the FRONT camera only. SAHI outputs pixel boxes; metres are A1's job.
- Next: GPU Coder version (blockers listed in `results/B4_simulink_log.txt`), real camera rows from A1, Unreal camera from A2.
