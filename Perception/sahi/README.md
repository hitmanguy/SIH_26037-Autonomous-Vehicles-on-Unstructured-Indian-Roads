# SAHI far-band pass (MATLAB / Simulink)

A full 1920×1080 frame is shrunk to 640 px before the detector sees it, so a person 100 m away (~18 px tall) becomes ~6 px and disappears.
SAHI ("slicing aided hyper inference") fixes this by also running the detector on **full-resolution tiles of one horizontal band** of the front camera — the strip where distant road users appear — and merging those boxes with the full-frame ones.

## Final configuration (`sahiDefaultOpts('v1')`, the default)
| Setting | Value | Why |
|---|---|---|
| Band | **one band**, 15–150 m ahead; rows from camera geometry (`sahiBandRows.m`), 470–670 on the placeholder camera | The two-band Python prototype cost 720 ms; one band keeps almost all of the gain |
| Tiles | 640 × N (N = band height, rounded to 32), 4 across, ~33% overlap, **batched in one call** | Per-call overhead (~50 ms) dominated; batching is the main speed-up |
| Full frame | **reused** from the fast loop | No second full-frame pass |
| Merge | per-inference NMS (IoU 0.45) + fragment-aware cross-tile merge (IoU 0.45 or intersection-over-smaller 0.6) | Removes half-objects cut by tile edges |
| Tile-only boxes | need score ≥ 0.35 | Cuts false alarms 4.1 → 3.0 per frame |
| Rate | 5 Hz slow loop (front camera only), next to the full-frame fast loop | 84 ms ≪ 200 ms budget |

Other modes kept for comparison: `'dual'` / `'fast'` (the Python prototype's bands), `'full'` (full frame only, own decoder).

## Files
| File | What it does |
|---|---|
| `sahiDetect.m` | The whole pass: letterbox, YOLOv8 decode, NMS, tiles, merge. Runs `det.Network` directly — does **not** call `detect()`, which hides boxes below ~0.5. Returns pixel boxes, scores, labels, `info`. |
| `sahiDefaultOpts.m` | The modes above. |
| `sahiBandRows.m` | Image rows for a 15–150 m band from camera height / focal length / pitch. |
| `sahiCameraPlaceholder.m` | Placeholder camera (1920×1080, f 1200 px, h 1.5 m, pitch 0) until the simulation camera is fixed. |
| `simulink/C3FullFrame.m` | MATLAB System block: full-frame detector (fast loop), fixed-size outputs. |
| `simulink/SahiSlowLoop.m` | MATLAB System block: SAHI band (slow loop), reuses the fast loop's boxes, marks which boxes only SAHI found. |
| `simulink/packDetections.m` | Top-N by score → fixed-size arrays (max 128 boxes) for Simulink. |
| `simulink/ImageSequenceSource.m` | Test camera that plays still images; to be replaced by the Unreal camera. |
| `tests/` | B1–B4 below (+ `parityMatch`, `evalMatch`, `apAllPoint` helpers). |
| `results/` | Logs of B1–B4 and two example images. |
| `python_reference/` | `sahi_engine.py` — first prototype (ONNX Runtime, two bands), now only a test oracle; `make_py_reference.py` makes the oracle file for B1; `b3b_sweep.py` the tile-only score sweep. |

## Tests and results (MATLAB R2025b, RTX 4050 laptop) — logs in `results/`
- **B1 · Parity:** every box matches the Python prototype one-to-one (min IoU 0.985).
- **B2 · Speed** (median per frame): Python-style dual band 720 ms → **84 ms** (one band, 640×N, batched, full frame reused).
- **B3 · Accuracy** on 1,127 IDD val front frames — recall of road users 16–32 px tall **17% → 47%**, under 16 px 1% → 26%; mAP50 0.40 → 0.51. B3b: tile-only score 0.35 → 3.0 false boxes per frame.
- **B4 · Simulink:** 30 Hz full-frame + 5 Hz SAHI via rate transitions; every SAHI tick identical to calling `sahiDetect` directly. Runs interpreted (not yet code-generation ready; blockers listed in `results/B4_simulink_log.txt`).

Running the tests needs the IDD validation set and the Python oracle file at the paths set at the top of each test script.
