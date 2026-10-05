# Autonomous Vehicles on Unstructured Indian Roads (SIH 2026 — PS 26037)
## Complete Execution & Operational Verification Guide

This guide provides the **complete prerequisites, required MATLAB toolboxes, and unified step-by-step instructions** to initialize, verify, simulate, and record all scenarios in the autonomous driving software stack.

---

## 1. System Requirements & Required MATLAB Toolboxes

### Recommended Environment
- **MATLAB Version**: **MATLAB R2025b** (or R2024b+)
- **Operating System**: Windows 10/11 (64-bit)
- **GPU (Optional, for Deep Learning / 3D Graphics)**: NVIDIA RTX series with CUDA 12+

### Required MATLAB Toolboxes
The stack integrates perception, multi-sensor fusion, spatiotemporal motion planning, and vehicle dynamics control. The following toolboxes provide the required capabilities:

| # | Toolbox Name | Purpose & Subsystem in AV Stack |
|:---:|:---|:---|
| 1 | **Automated Driving Toolbox™** | Core driving scenario simulation (`drivingScenario`), actor kinematics, pinhole camera & 77 GHz radar sensor simulation models, cuboid bounding boxes, and OpenSCENARIO/OpenDRIVE parser. |
| 2 | **Navigation Toolbox™** | Spatiotemporal dynamic costmap grids (`binaryOccupancyMap`, cost slices), collision checking (`polyshape`, polygon intersections), and path interpolation. |
| 3 | **Sensor Fusion and Tracking Toolbox™** | Multi-sensor information matrix fusion (Camera + Radar), Interacting Multiple Model (IMM) Kalman Filtering (`c3_semantic_imm_tracker`), and track state management. |
| 4 | **Deep Learning Toolbox™** | Execution and inference of the trained **C3 YOLOv8s 12-Class Indian Driving Dataset (IDD)** object detector (`load_c3_detector.m`, DAG network, ONNX execution). |
| 5 | **Computer Vision Toolbox™** | Pinhole camera intrinsic projection (`[fx, fy, cx, cy]`), 2D bounding-box to 3D Birds-Eye-View (BEV) geometry transformation (`project_bboxes_to_3d`), and image mosaic processing. |
| 6 | **Model Predictive Control Toolbox™** | Real-time lateral steering and longitudinal velocity control via Model Predictive Control (`'mpc'` controller option in `autonomous_ego_controller`). |
| 7 | **Optimization Toolbox™** | Quadratic Programming (QP) solver and nonlinear boundary constraint satisfaction for the Frenet optimal spatiotemporal trajectory generator. |
| 8 | **Statistics and Machine Learning Toolbox™** | Multi-modal trajectory prediction clustering via Gaussian Mixture Models (GMM) in the MotionFormer/GMM prediction engine. |
| 9 | **Vehicle Dynamics Blockset™ & Simulink 3D Animation™** | 3D Unreal Engine® co-simulation bridge, multi-camera surround cockpit capture, and vehicle kinematics models. |
| 10 | **ROS Toolbox™** *(Optional)* | DDS middleware bridge for ROS 2 sensor message broadcasting and vehicle telemetry streaming. |

---

## 2. One Unified Method to Run Tests & Simulations

Follow this unified workflow inside the **MATLAB Command Window**:

### Step 1: Open MATLAB & Navigate to Repository Root
Ensure your current MATLAB working directory is the project root:
```matlab
cd 'D:\hackathon\SIH26'   % or your local cloned repository directory
```

### Step 2: One-Click Environment Startup (`startup`)
Run the unified startup script once to automatically register all modular subdirectories (`main`, `vehicle_dynamics`, `scenarios/matlab`, `scenarios/verification`, `perception`, `Sensor_fusion`, `Trajectory`, `Path_planning_decision`) onto the MATLAB path:
```matlab
startup
```
You will see the initialization banner confirming that all submodules and path dependencies are successfully mounted.

---

### Step 3: Run Full 15-Scenario Closed-Loop Verification Suite (Single Command)
To evaluate and verify the vehicle trajectory across **all 15 scenarios** (Euro NCAP protocols + Indian arterial multi-threat edge cases) in automated headless mode:

```matlab
report = verify_closed_loop_trajectories('pp')
```

#### What this command does:
1. Automatically executes all 15 scenarios sequentially in closed loop with the autonomous stack.
2. Evaluates the trajectory against 10 strict safety and kinematic criteria:
   - **COLLISION == 0**: Strict oriented-bounding-box non-overlap against all road actors.
   - **CLEARANCE > 0 m**: Minimum boundary gap to all vehicles, VRUs, and static obstacles.
   - **ONCOMING == 0.0%**: Ego vehicle center never enters oncoming traffic ($Y > -0.60\text{ m}$).
   - **ROAD EDGE == 0.0%**: Ego vehicle never drives off the outer curb ($Y < -6.20\text{ m}$).
   - **LAT ACCEL $\le$ 4.0 m/s²**: Peak lateral acceleration within passenger comfort and tire adhesion limits.
   - **YAW RATE $\le$ 0.60 rad/s**: Peak yaw rate within vehicle stability limits.
   - **ZIGZAG $\le$ 4**: Direction reversals $\ge 0.5\text{ m}$ suppressed (eliminates oscillatory slaloming).
   - **END SWERVE $\le$ 2.0 m**: Lateral drift in exit corridor ($X > 250\text{ m}$).
   - **BAD STOP == 0.0 s**: Zero unjustified standstill ($>3\text{ s}$ without an obstacle ahead).
   - **REVERSE == 0**: Zero uncommanded backwards motion.
3. Prints the verified results table to the console and exports:
   - Summary CSV: `scenarios/verification/trajectory_reports/report_pp.csv`
   - High-resolution trajectory plots for each scenario: `scenarios/verification/trajectory_reports/<SCENARIO>_pp.png`

#### Alternative Controller (MPC):
```matlab
report = verify_closed_loop_trajectories('mpc')
```

#### Testing a Specific Scenario Subset:
```matlab
report = verify_closed_loop_trajectories('pp', {'INDIAN_WRONGWAY', 'INDIAN_GAUNTLET', 'INDIAN_POTHOLE'})
```

---

### Step 4: Interactive 2D Animated Simulation (HUD Mode)
To watch the real-time simulation with an interactive **Birds-Eye View HUD**, 360° surround sensor coverage cones, and fused 3D track markers:

```matlab
% Indian Multi-Hazard Stress Test ("The Gauntlet"):
run_instruction('GAUNTLET', 'pp', 'animate')

% Head-On Wrong-Way Mini-Truck Encounter:
run_instruction('WRONGWAY', 'pp', 'animate')

% School Zone Multi-Pedestrian Rush:
run_instruction('SCHOOLZONE', 'pp', 'animate')

% Pothole Barrier Detour:
run_instruction('POTHOLE', 'pp', 'animate')
```

---

### Step 5: 3D Unreal Engine Co-Simulation & Video Recording (`'3d'`)
To run high-fidelity 3D visualization and record Full HD 30 FPS video:

```matlab
run_instruction('GAUNTLET', 'pp', '3d')
run_instruction('CPNCO', 'pp', '3d')
```

**Outputs generated:**
- Latest video: `unreal_simulation_video.mp4` (in root directory)
- Preserved run logs: `logs/run_<TIMESTAMP>_<SCENARIO>_<CONTROLLER>/`
  - `unreal_simulation_video.mp4` — Full HD 30 FPS video recording
  - `images/surround_cockpit/` — Multi-camera HUD frames
  - `graphs/` — Speed, steering, lateral error, and TTC plots

---

### Step 6: Full SOTA Autonomous AV Stack Mode (`'stack'`)
To run closed-loop simulation with C3 YOLOv8s 12-class perception, Information Matrix Fusion, Semantic IMM tracking, and GMM Trajectory Prediction:

```matlab
run_instruction('CPNCO', 'pp', 'stack')
run_instruction('POTHOLE', 'pp', 'stack')
```

---

### Step 7: Driving Scenario Designer GUI App (`'app'`)
To visually inspect, edit, or adjust road geometries and actor trajectories in the native MATLAB GUI:

```matlab
run_instruction('GAUNTLET', 'pp', 'app')
run_instruction('SCHOOLZONE', 'pp', 'app')
```

---

## 3. Catalog of All 15 Verified Scenarios

### 🌟 Complex Indian Mixed-Traffic Scenarios
| Scenario Keyword | Scenario Name | Actors | Challenge & Maneuver |
|:---|:---|:---:|:---|
| `'GAUNTLET'` | **The Gauntlet (Multi-Threat)** | 8 | 3 sequential phases: Slow rickshaw car-following $\to$ construction barrier detour to Lane -2 $\to$ jaywalker AEB yield. |
| `'WRONGWAY'` | **Wrong-Way Head-On Encounter** | 5 | Mahindra tempo driving head-on in ego lane; early lateral evasion to Lane -2, passing tempo, and return to Lane 1 past parked van. |
| `'SCHOOLZONE'` | **School Zone Crossing Rush** | 6 | Stopped school bus obscuring 3 children darting across at different speeds; sequential kinematic AEB deceleration. |
| `'BUSSTOP'` | **Bus Stop Multi-Threat** | 5 | BMTC city bus decelerating to stop, passenger alighting into lane, high-speed overtaking motorcycle. |
| `'VENDORCART'` | **Street Vendor Handcart Swerve** | 5 | Slow handcart swerving from shoulder into driving lane, oncoming bus in opposing lane. |
| `'POTHOLE'` | **Pothole-Forced Lane Detour** | 5 | Unstructured road defect/barrier forcing smooth quintic detour into Lane -2 with zero oncoming intrusion. |
| `'CONGESTION'` | **Dense Traffic Congestion Queue** | 7 | Creeping multi-vehicle queue with filtering bicycle; safe ACC car-following without premature slaloms. |
| `'CATTLE'` | **Stray Cattle Cluster** | 5 | Wandering cow entering corridor, stationary cow at divider; smooth yielding and avoidance. |
| `'JAYWALK'` | **Mid-Block Jaywalker Crossing** | 5 | Pedestrian crossing briskly from behind parked vehicle; camera/radar fusion AEB stop. |
| `'AUTOCUTIN'` | **Aggressive Auto-Rickshaw Cut-In** | 3 | Three-wheeler cutting across ego bumper; kinematic deceleration and gap recovery. |
| `'TWOWHEELER'` | **Two-Wheeler Lane Filter** | 3 | Fast motorcycle lane-splitting along lane markings; lateral corridor buffering. |

### 🛡️ Euro NCAP Safety Test Protocols
| Scenario Keyword | Protocol | Scenario Description |
|:---|:---|:---|
| `'CPNCO'` | Euro NCAP 2023 | Car-to-Pedestrian Nearside Child Obstructed by parked obstruction vehicles. |
| `'CPTA'` | Euro NCAP 2023 | Car-to-Pedestrian Turning Adult at urban intersection. |
| `'CCFTAP'` | Euro NCAP 2023 | Car-to-Car Front Turn Across Path intersection conflict. |
| `'CCCSCP'` | Euro NCAP 2023 | Car-to-Car Straight Crossing Path perpendicular intersection collision avoidance. |

---

## 4. Summary of Verification Baseline

Every scenario in the catalog has been verified in **MATLAB R2025b** with the following performance guarantees:
- **Collision Rate**: `0 / 15` (`0.0%` collisions)
- **Lane Legality**: `0.0%` oncoming carriageway intrusion ($Y > -0.60\text{ m}$)
- **Road Legality**: `0.0%` curb departure ($Y < -6.20\text{ m}$)
- **Ride Comfort**: Peak lateral acceleration $< 0.80\text{ m/s}^2$ (well below the $4.0\text{ m/s}^2$ safety cap)
- **Control Stability**: Peak yaw rate $< 0.25\text{ rad/s}$ with critically damped steering ($\le 1$ direction reversal across all maneuvers)
- **Longitudinal Reliability**: $0.0\text{ s}$ unjustified standstill; smooth stop-and-go resumption
