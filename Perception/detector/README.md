# Camera detector: training + MATLAB import

Train in Python (Ultralytics YOLOv8s) on a GPU server → export ONNX with a custom head cut → run in MATLAB
(`importNetworkFromONNX` + `yolov8ObjectDetector`). Paths in the scripts assume the author's folders
(`D:\SIH\...` on the laptop, `~/epsilon_yash` on the server) — edit them to your setup.

## Versions
| | Classes | Training data | Key results |
|---|---|---|---|
| **v1** (`C3_v1_idd/`) | 12: person, rider, car, bus, truck, autorickshaw, motorcycle, bicycle, animal, traffic sign, traffic light, vehicle fallback | IDD Detection (31,569 imgs), 60 epochs | mAP50 0.503 on IDD val; vs stock COCO YOLOv8s on the same classes 0.197 → 0.487 (auto 0 → 0.69, rider 0 → 0.56) |
| **v2** (`v2_data/`) | v1 + pothole, pushcart, tractor, emergency vehicle, cone_barrier = 17 | IDD + DriveIndia (TiHAN) + RDD2022 India/Japan potholes = 59,089 imgs | IDD val 0.506 → 0.499 (no forgetting); DriveIndia val 0.52 → 0.77; pothole 0 → 0.45 mAP50 (RDD val, 557 boxes); cone_barrier 0.73 |
| **v2.1** (`v21_data/`, training) | v2 + **slow_zone** (speed bump / zebra / rumble strip) = 18 | v2 + RDD Czech/US potholes (5,762 total) + DriveIndia zebra & rumble strips (833) + 69 hand-labelled speed bumps; tractor/pushcart overlap fix | pending |

## How the data was merged (v2 / v2.1)
- `di_to_v2.py` — DriveIndia IDs → v2 classes (mapping checked by eye on crop sheets); person-on-bike → rider; noisy IDs held back.
- `relabel_noisy_with_v1.py` — noisy DriveIndia boxes relabelled with v1 (IoU ≥ 0.5), else image held back.
- `rdd_to_v2.py` — RDD2022 VOC → YOLO, only D40 potholes; road users in those images auto-labelled by v1.
- `pseudo_new_on_idd.py` / `pseudo_new.py` — a teacher model adds the new classes to IDD (which never labelled them), so they are not learned as background.
- `v21_build.py` — adds slow_zone and grows the class head 17 → 18 without losing v2's weights.
- `rdd_sheets.py`, `bump_autolabel.py`, `bump_export.m` — review sheets and hand-labelled speed bumps (MATLAB Image Labeler → YOLO).
- Dropped on purpose: RDD D20 "alligator cracking" (mostly cosmetic cracks, vague boxes), China_MotorBike / Drone views (camera looks down).

## MATLAB
- `C1_stock_yolov8/` — stock YOLOv8s on Indian frames (baseline).
- `C2_road_segmentation/` — DeepLab v3+ ResNet-18 on IDD Lite, trained in MATLAB (drivable IoU 88.6%).
- `C3_v1_idd/export_for_matlab.py` — cuts the Ultralytics Detect head to 3 per-scale outputs (64+nc ch) so MATLAB can import it; re-run after every retrain.
- `C4_v2_matlab/` — v2 import (84.5 ms/frame on RTX 4050 laptop) and pothole test with our own decoder: 6 of 8 labelled potholes found on 3 RDD test frames at score ≥ 0.15.
