# Camera detector: training data + MATLAB import

Train in Python (Ultralytics YOLOv8s) on a GPU server → export ONNX with a custom head cut → run in MATLAB
(`importNetworkFromONNX` + `yolov8ObjectDetector`). Paths in the scripts assume the author's folders
(`D:\SIH\...` on the laptop, `~/epsilon_yash` on the server) — edit them to your setup.
Model weights for v2 / v2.1 are not in the repo.

## Versions
| | Classes | Training data | Key results (mAP50) |
|---|---|---|---|
| **v1** | 12: person, rider, car, bus, truck, autorickshaw, motorcycle, bicycle, animal, traffic sign, traffic light, vehicle fallback | IDD Detection (31,569 imgs), 60 epochs | IDD val 0.503; stock COCO YOLOv8s on the same classes 0.197 → 0.487 (auto 0 → 0.69, rider 0 → 0.56) |
| **v2** | v1 + pothole, pushcart, tractor, emergency vehicle, cone_barrier = 17 | IDD + DriveIndia (TiHAN) + RDD2022 India/Japan potholes = 59,089 imgs | IDD val 0.499 (no forgetting); DriveIndia val 0.52 → 0.77; pothole 0.45 (RDD val); cone_barrier 0.73 |
| **v2.1 (final)** | v2 + **slow_zone** (speed bump / zebra / rumble strip) = 18 | v2 + RDD Czech/US potholes (5,762 total) + DriveIndia zebra & rumble strips + 69 hand-labelled speed bumps (60,047 imgs), 40 epochs | IDD val 0.500; DriveIndia val 0.779; pothole 0.422 on 51 unseen RDD photos (v2: 0.377); slow_zone 0.72 (DriveIndia) / 0.85 (RDD); 45 ms/frame in MATLAB |

## Folders
| Folder | What it does |
|---|---|
| `C1_stock_yolov8/` | Stock COCO YOLOv8s on Indian frames in MATLAB — the baseline (autos called car/bus, riders split, 0 animals). |
| `C2_road_segmentation/` | DeepLab v3+ ResNet-18 trained in MATLAB on IDD Lite (drivable IoU 88.6%) + figure script. |
| `C3_v1_idd/` | `export_for_matlab.py` cuts the Ultralytics Detect head to 3 per-scale outputs so MATLAB can import it (re-run after every retrain); `C3b_import_to_matlab.m` imports v1; `c3_compare.py` stock-vs-ours per-class table. |
| `C4_v2_matlab/` | v2 import + pothole test with our own decoder (6 of 8 labelled potholes found at score ≥ 0.15). |
| `C4_v21_matlab/` | v2.1 import (18 classes) + test on pothole / zebra / rumble-strip frames via `sahiDetect`. |
| `data_v2/` | Builds the v2 training set (below). |
| `data_v21/` | Adds slow_zone, more potholes and speed bumps for v2.1 (below). |

## How the data was merged
**v2 (`data_v2/`)**
- `di_to_v2.py` — DriveIndia IDs → our classes (mapping checked by eye on crop sheets); person-on-bike → rider; noisy IDs held back.
- `relabel_noisy_with_v1.py` — noisy DriveIndia boxes relabelled with v1 (IoU ≥ 0.5), else the image is held back.
- `rdd_to_v2.py` — RDD2022 (VOC) → YOLO: D40 potholes (and, with `--slow-codes D43`, faded crossings as slow_zone); road users in those images auto-labelled by v1.
- `pseudo_new_on_idd.py` — a teacher model adds the new classes to IDD (which never labelled them), so they are not learned as background.

**v2.1 (`data_v21/`)**
- `v21_build.py` — grows the class head 17 → 18 without losing v2's weights.
- `pseudo_new.py` — general version of the teacher pseudo-labelling (`--classes`, `--splits`, `--conf`), logs every added box to `added.csv`.
- `added_sheet.py`, `rdd_sheets.py` — crop sheets so every auto-added box can be checked by eye.
- `bump_autolabel.py`, `bump_export.m` — speed bumps labelled in MATLAB Image Labeler → YOLO format.
- `pothole_fair.py` — the "fair" pothole test on RDD photos no model has seen.
- Dropped on purpose: RDD D20 "alligator cracking" (mostly cosmetic cracks, vague boxes), China_MotorBike / Drone views (camera looks down).
