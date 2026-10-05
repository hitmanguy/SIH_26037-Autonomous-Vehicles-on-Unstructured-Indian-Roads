# Perception — what the car sees

**Team Epsilon · SIH 2026 · PS 26037** (Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads)

The camera side of perception does three jobs, each with the tool that suits it:

| Job | How | Runs at |
|---|---|---|
| **Find road users and road hazards** (all 4 cameras) | YOLOv8s fine-tuned on Indian data, 18 classes, imported into MATLAB from ONNX | every frame (fast loop) |
| **Catch small, far-away objects** (front camera) | SAHI: re-read one horizontal band of the image at full resolution, in tiles | 5 Hz (slow loop) |
| **Know where the road is** (front camera) | DeepLab v3+ semantic segmentation of the drivable surface, trained in MATLAB | ~5 Hz |

Outputs are pixel boxes + class + score, and a road mask. Converting boxes to metres (camera geometry) and fusing them with radar happens in [`Sensor_fusion/`](../Sensor_fusion/).

```
 camera frame (1920x1080)
   ├── fast loop ── full frame → 640 letterbox → YOLOv8s ───────────────┐
   │                                                                     ├─ merge (NMS + fragment-aware) → boxes → Sensor_fusion
   └── slow loop (5 Hz, front cam) ── one band → 640xN tiles, batched ───┘
                                       (reuses the fast loop's full-frame result)
```

---

## Final design decisions

**Detector — v2.1, 18 classes.**
person, rider, car, bus, truck, autorickshaw, motorcycle, bicycle, animal, traffic sign, traffic light, vehicle fallback, **pothole, pushcart, tractor, emergency vehicle, cone/barrier, slow_zone** (speed bump + zebra crossing + rumble strips, all meaning "slow down"; LiDAR can tell raised from painted).
Classes follow *behaviour*: an ambulance or a pushcart needs a different reaction than a car, so they get their own class.

**SAHI — one band, not two.**
The first prototype (Python, `sahi/python_reference/sahi_engine.py`) used two overlapping bands (rows 400–760 and 600–1000, 8 tiles) and took 720 ms per frame in MATLAB. The final version uses:

- **one band** covering 15–150 m ahead, with its rows computed from the camera geometry (`sahiBandRows.m`; rows 470–670 on the placeholder camera),
- **640 × N tiles** (only as tall as the band, not padded to 640 × 640), **all tiles in one batched call**,
- **reuse of the full-frame result** from the fast loop instead of running it again,
- a stricter score (0.35) for boxes that only the tiles found, to keep false alarms down.

Result: **84 ms instead of 720 ms**, with recall within 1–3 points of the two-band version. The band is for the front camera only.

**Runtime is MATLAB/Simulink only.** Python is used for training on the GPU server and as a test oracle; the trained network is exported to ONNX with a custom head cut (`detector/C3_v1_idd/export_for_matlab.py`) and imported with `importNetworkFromONNX`.

---

## Results

**Detector** (mAP50 on validation sets)

| | v1 | v2 | **v2.1 (final)** |
|---|---:|---:|---:|
| Classes | 12 | 17 | **18** |
| Training data | IDD | IDD + DriveIndia + RDD2022 (59k imgs) | + more potholes, speed bumps (60k imgs) |
| IDD val (12 shared classes) | 0.506 | 0.499 | **0.500** |
| DriveIndia val (12 shared classes) | 0.523 | 0.773 | **0.779** (published YOLOv8 baseline: 0.787, different split) |
| Pothole, 51 unseen RDD2022 photos | — | 0.377 | **0.422** (recall 31% → 40%) |
| slow_zone (DriveIndia / RDD val) | — | — | **0.72 / 0.85** |
| Speed in MATLAB (RTX 4050 laptop) | 39–64 ms | 85 ms | **45 ms** (22 fps) |

Stock COCO YOLOv8s vs our v1 on the same IDD classes: 0.197 → 0.487 (autorickshaw 0 → 0.69, rider 0 → 0.56).

**SAHI** (1,127 IDD val front-camera frames, IoU 0.5) — recall of road users by box height:

| | <16 px | 16–32 px | 32–64 px | 64–128 px | ≥128 px |
|---|---:|---:|---:|---:|---:|
| Full frame only | 0.9% | 17.3% | 47.6% | 74.1% | 91.0% |
| + SAHI band | **25.8%** | **47.2%** | **66.6%** | 79.8% | 91.5% |

mAP50 0.40 → 0.51. Cost: false boxes 1.7 → 3.0 per frame (with the 0.35 tile-only score). Box-for-box identical to the Python prototype; runs in Simulink as a 5 Hz loop next to the full-frame loop.

**Road segmentation:** DeepLab v3+ (ResNet-18) on IDD Lite, trained in MATLAB in 27 min — drivable-area IoU 88.6%, mIoU 59.2%, 53 ms per image.

---

## Folder map

```
Perception/
├── README.md               this file
├── C3_detector_v1/         ready-to-run MATLAB detector (v1, 12 classes) used by main/AutonomousAVStack.m
│                           load_c3_detector.m → detector object · check_setup.m checks add-ons · test_images/
├── c3_idd_detections.mat   saved v1 detections on IDD frames (fallback input for the integrated stack)
├── sahi/                   SAHI far-band pass in MATLAB/Simulink   → sahi/README.md
└── detector/               data scripts + MATLAB import for v1 → v2 → v2.1   → detector/README.md
```

Only v1's weights are in this repo. The v2 / v2.1 weights are kept outside the repo; the scripts to rebuild and import them are in `detector/`.

---

## Planned next

- GPU Coder / TensorRT build so the full-frame loop reaches 30 Hz (plain MATLAB: ~22 fps).
- Real camera intrinsics from the simulation rig → band rows and box-to-metre conversion (currently a placeholder camera: 1920×1080, f = 1200 px, height 1.5 m, pitch 0).
- Unreal (Simulation 3D Camera) frames feeding the detector inside Simulink, replacing the still-image test source.
- LiDAR ground-curvature check to confirm camera pothole candidates and measure depth.
- More data for rare classes (pushcart, emergency vehicle) and night driving.

## Known limits

- MATLAB's `detect()` from the YOLO add-on silently drops boxes scoring below ~0.5. Use `sahiDetect` (own decoder) when low scores matter, e.g. potholes and slow zones.
- Pothole detection from the camera alone is modest (0.42 mAP50); it is a candidate generator, not the final word.
- No species-level animal class; dusk/night animals are sometimes labelled as person.
