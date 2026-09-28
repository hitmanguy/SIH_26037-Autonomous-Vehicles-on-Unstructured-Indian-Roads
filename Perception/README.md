# Perception: What the Car Sees, and How
## C3 YOLOv8 IDD Detector + SAHI Dual-Band Slicing Engine

**Team Epsilon | Smart India Hackathon 2026 | Problem Statement: SIH26037**  
*Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads*

---

## 1. Executive Summary

Autonomous navigation on unstructured Indian roads demands a perception architecture tailored to extreme traffic heterogeneity, high density, and unexpected roadway obstacles. Standard Western autonomous driving datasets (e.g. KITTI, nuScenes, Waymo) do not capture:
- Informal lane discipline and erratic lateral cutting by three-wheelers (`autorickshaw`).
- Dense clusters of two-wheelers (`motorcycle`, `bicycle`, `rider`).
- Vulnerable pedestrians stepping unpredictably into traffic streams (`person`).
- Unrestrained animals crossing or freezing in headlights (`animal` / stray cows).
- Severe road surface anomalies, unpaved shoulders, and speed breakers.

This directory implements the **Perception Pipeline** utilizing:
1. **C3 IDD YOLOv8s Detector:** Fine-tuned on the India Driving Dataset (IDD) across 12 Indian-specific traffic classes.
2. **SAHI (Slicing Aided Hyper Inference) Dual-Band Slicing Engine:** Solves the optical downscaling bottleneck on 1080p sensors, restoring native 1:1 sensor resolution for distant targets in the 40–150m horizon band.
3. **Pinhole Inverse Perspective Mapping (IPM):** Projects 2D bounding boxes to 3D ego Cartesian coordinates to seed downstream multi-object tracking in `Sensor_fusion`.

---

## 2. Perception Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                      CAMERA PERCEPTION PIPELINE                                   |
|                                                                                                   |
|  [Surround Camera Rig: 4x 1080p Views]                                                             |
|   ├── Front Main (1080p, 60° HFOV)                                                                |
|   ├── Front Bumper / Near (1080p, Wide Ground View)                                               |
|   ├── Rear Wide (1080p, 120° HFOV)                                                                |
|   └── Side Flank (1080p, Blind-Spot & Lateral Coverage)                                           |
|                                                                                                   |
|  ============================== MULTI-RATE DUAL-LOOP INFERENCE ==================================  |
|                                                                                                   |
|  (A) FAST PERCEPTION LOOP (15.6 - 30 Hz):                                                         |
|      Full-Frame 1080p -> Letterbox Resize (640x640) -> YOLOv8s Inference                          |
|      - Low latency, full-scene situational awareness.                                             |
|      - High recall for near-field actors (0 - 40m).                                               |
|                                                                                                   |
|  (B) SLOW PERCEPTION LOOP (5 Hz - SAHI Slicing Engine):                                           |
|      Full-Frame 1080p -> Dual-Band Functional Cropping:                                           |
|      ├── Far Horizon Band (40 - 150m): Rows y in [400, 760 px] -> 640x360 Slices (35% overlap)    |
|      └── Pothole / Near Road Band (15 - 30m): Rows y in [600, 1000 px]                             |
|      - Preserves 1:1 native optical sensor resolution.                                            |
|      - Tiles letterboxed without upscaling; 4 tiles/band always span the full image width.      |
|                                                                                                   |
|  (C) TILE REMAPPING, PER-TILE NMS & FRAGMENT-AWARE MERGE:                                         |
|      Remap tile coordinates:  x_canvas = x_tile + x_offset,  y_canvas = y_tile + y_offset        |
|      Per-inference NMS (IoU 0.45), then cross-tile merge (IoU 0.45 or IoS 0.6 for fragments).    |
+---------------------------------------------------------------------------------------------------+
                                   │
                                   ▼  [3D Measurement Stream]
                 --> Handoff to `Sensor_fusion/` (Semantic IMM Tracker)
```

---

## 3. The Small & Distant Hazard Bottleneck on Indian Roads

### 3.1 Optical Resolution Degradation Under Naive Resizing
Standard deep learning vision detectors operate at a fixed square input tensor (e.g. $640 \times 640\text{ px}$). When full $1920 \times 1080\text{ px}$ automotive camera feeds are letterboxed (aspect-preserving) to $640 \times 640$, visual resolution degrades by a factor of **$3.0\times$** in both axes.

Under pinhole perspective geometry:

```
pixel_height = (target_real_height * focal_length) / distance
```

For a typical $1080\text{p}$ automotive camera ($f_y \approx 1200\text{ px}$):
- A $1.5\text{ m}$ tall pedestrian or motorcycle at $100\text{ m}$ projects to an optical height of **$18\text{ pixels}$** on the raw sensor.
- When downscaled directly to $640 \times 640$, this target shrinks to just **$5.0\text{ pixels}$ tall**.
- Because YOLOv8's finest feature pyramid level has a stride of $s = 8\text{ pixels}$, a $5\text{ px}$ target cannot trigger feature activation and is completely invisible until it closes to $< 45\text{ m}$.
- At highway cruising speeds ($80\text{ km/h} \approx 22.2\text{ m/s}$), detecting a hazard at $45\text{ m}$ gives the vehicle only **$2.0\text{ seconds}$** to brake or swerve—insufficient for safe emergency avoidance on unpredictable roads.

### 3.2 Resolution Recovery via SAHI Slicing
By cropping localized $640\text{ px}$ tiles directly from the unscaled $1080\text{p}$ image, **native $1:1$ sensor resolution is 100% preserved**. The distant pedestrian retains its full $18\text{ px}$ height, extending detection range for small road users toward $100\text{–}150\text{ m}$. In the `Sensor_fusion` Monte Carlo benchmark this confirms a 1.6 m bicycle at ~174 m instead of ~64 m (**+13.2 s earlier**, under an assumed $p_{det}=0.5$ at 12 px detection curve).

---

## 4. Dual-Band Functional Slicing Geometry

Rather than tiling the entire image (which wastes compute on sky and ego-hood), `sahi_engine.py` implements the dual-band geometry specified in Team Epsilon's perception slide:

1. **Far Horizon Band ($40\text{–}150\text{ m}$ Lookahead):**
   - Bounding row coordinates: $y \in [400, 760\text{ px}]$.
   - Covers the vanishing point and distant road horizon where oncoming traffic, stalled vehicles, pedestrians, and cattle appear.
   - Sliced into overlapping tiles of width $640\text{ px}$ with $35\%$ horizontal overlap.
   - Runs asynchronously at $5\text{ Hz}$.
2. **Pothole / Near Road Band ($15\text{–}30\text{ m}$ Lookahead):**
   - Bounding row coordinates: $y \in [600, 1000\text{ px}]$.
   - Focuses on the immediate road surface texture for negative obstacle / depression extraction.
3. **Coordinate Remapping, NMS & Fragment-Aware Merging:**
   - Tile coordinates $[x_{\text{tile}}, y_{\text{tile}}, w_{\text{tile}}, h_{\text{tile}}]$ are remapped back to full-frame canvas coordinates:
     ```
     x_canvas = x_tile + x_offset
     y_canvas = y_tile + y_offset
     ```
   - Each inference (full frame or tile) gets its own class-aware NMS (IoU $0.45$).
   - Detections from *different* inferences are then merged greedily: same-class boxes group if IoU $> 0.45$, or if intersection-over-smaller (IoS) $> 0.6$ and the pair looks like a fragment (one box touches an interior tile edge, or is < 50% of the other's area). Truncated boxes rank below complete ones, and a truncated group leader is grown to the union of its truncated partners, which reconstructs objects split across tiles.
   - Plain IoU-NMS cannot remove a half-object cut off by a tile edge (its IoU with the full box is ~0.5 or lower). In the previous version **19 of the 40 "new" SAHI detections were such fragments**; the merge above removes all of them while keeping tightly parked two-wheelers separate.
   - Tiles are letterboxed without upscaling (the previous version stretched 640×360 crops to 640×640), and tile origins are spaced evenly so they always cover the full image width (the previous stride of 420 px skipped the right-most 20 px).

---

## 5. Empirical Benchmark Across India Driving Dataset (IDD) Frames

The SAHI slicing engine was evaluated across 4 real camera angles from the India Driving Dataset (`C3_detector_v1/test_images/`):

### 5.1 Multi-View Detection Benchmark Summary

"New" = a SAHI detection with no same-class full-frame box matching it (IoU > 0.35 or IoS > 0.6). This is the conservative measure: the SAHI total can also grow when full-frame covered two adjacent objects (e.g. parked bikes) with one box.

| Test Frame | Camera Viewpoint | Scene Context | Standard Full-Frame | C3 YOLOv8s + SAHI | New Objects (no full-frame match) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `highquality_16k` | Front Center (1080p) | Dense Urban Bangalore (Flyover, Crowded Lanes) | 42 | **59** | **+10** |
| `frontNear` | Front Bumper | Village / Suburban Road (Open Horizon) | 7 | **13** | **+6** |
| `rearNear` | Rear Wide | Highway Overtaking & Tailgaters | 7 | **12** | **+5** |
| `sideLeft` | Side Flank | Lateral Blind-Spot & Pedestrians | 8 | **10** | **+2** |
| **Total Across All Views** | — | — | **64** | **94** | **+23 (+36%)** |

Compared with the previous engine (57 full-frame / 103 SAHI / +40 new): letterboxing raised full-frame detections from 57 to 64, and 19 of the previous 40 "new" detections were tile-edge fragments of already-detected objects, which the fragment-aware merge now removes. Without IDD ground-truth labels for these frames, precision/recall is not measured; these are detection counts.

#### Runtime (CPU, ONNX Runtime, per 1080p frame)

| | Previous engine | Current engine | `--fast` (single merged band) |
| :--- | :---: | :---: | :---: |
| Full-frame pass | 85 ms | 72 ms | 72 ms |
| SAHI tiles | 8 stretched tiles | 8 letterboxed tiles, ~500 ms | 4 tiles, ~265 ms |
| **Whole frame** | **~1200 ms** | **~575 ms** | **~340 ms** |

`--fast` uses one 600 px band (rows 400–1000, still 1:1) instead of the two overlapping bands; on the 4 test frames it finds 18 instead of 23 new objects. The ONNX graph has a fixed batch size of 1, so tiles run sequentially; exporting with a dynamic batch axis (or using the CUDA/DirectML execution provider, picked automatically if installed) is the next speed-up.

### 5.2 Class Breakdown in Dense Bangalore Traffic (`highquality_16k`)

- `person`: $3 \rightarrow 9$ (**$+6$ distant pedestrians**).
- `motorcycle`: $19 \rightarrow 24$ (**$+5$ two-wheelers**).
- `rider`: $6 \rightarrow 9$ (**$+3$ riders**).
- `autorickshaw`: $3 \rightarrow 4$ (**$+1$**).
- `car`: $11 \rightarrow 12$ (**$+1$**).

![SAHI Slicing Perception Benchmark](sahi_slicing_comparison.png)

*Figure: (Top-Left) Standard Full-Frame YOLOv8s (42 detections). (Top-Right) C3 YOLOv8s + SAHI Dual-Band Slicing (59 detections) with cyan markers on the +10 objects that have no full-frame match; dashed boxes show the Far (rows 400–760) and Near (rows 600–1000) bands. (Bottom-Left) Class-wise detection gain breakdown in dense Bangalore traffic. (Bottom-Right) Mathematical resolution density curve proving the 3.0x optical pixel density advantage for hazards at 40–150m.*

---

## 6. Handoff to Downstream Sensor Fusion

Detections from the perception pipeline are projected into 3D metric ego Cartesian coordinates using inverse perspective mapping (IPM) on flat ground:

```
Z = (H_cam * f_y) / (v_bottom - c_y)
X = ((u_center - c_x) * Z) / f_x
```

Where:
- `H_cam`: Camera mounting height ($1.5\text{ m}$).
- `[f_x, f_y]`: Camera focal lengths ($1200\text{ px}$).
- `[c_x, c_y]`: Optical center principal point ($960, 540\text{ px}$).
- `[u_center, v_bottom]`: Bounding box horizontal center and ground contact point.

The projected 3D coordinates `[X, Z]` and class labels feed directly into `Sensor_fusion/c3_semantic_imm_tracker.m`, seeding confirmed tracks earlier. `sahi_engine.py` now also exports this flat-ground range estimate per detection (`ff_range_m`, `sahi_range_m`).

---

## 7. Directory Structure & Execution

```
Perception/
├── README.md                              # This comprehensive perception report
├── plan.md                                # Perception system planning notes
│
├── sahi_engine.py                         # Standalone Python SAHI dual-band slicing engine (ONNX Runtime)
├── sahi_visualizer_and_benchmark.m        # MATLAB visualizer & 4-panel SAHI benchmark generator
│
├── sahi_slicing_comparison.png           # 4-panel SAHI comparison, class breakdown, and resolution curve
├── sahi_detection_results.mat            # Full-frame and sliced detections for all 4 IDD test frames
└── sahi_benchmark_summary.mat            # Numerical benchmark logs for SAHI visualizer
```

### Running the SAHI Slicing Engine

#### 1. Run Python Inference
```bash
cd Perception
python sahi_engine.py            # dual-band (default)
python sahi_engine.py --fast     # single merged band, ~2x faster SAHI pass
python sahi_engine.py my_frames/ # any folder or list of images
```
*Outputs `sahi_detection_results.mat` containing bounding boxes, confidence scores, labels, and new distant hazard flags.*

#### 2. Run MATLAB Visualizer & Benchmark Suite
```matlab
cd Perception
sahi_visualizer_and_benchmark
```
*Generates and displays the publication-grade 4-panel dashboard and saves `sahi_slicing_comparison.png`.*
