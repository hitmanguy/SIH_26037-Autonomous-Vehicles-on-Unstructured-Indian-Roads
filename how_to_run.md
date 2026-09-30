# Autonomous Driving Scenarios — Run Commands Guide

Run all commands directly inside the **MATLAB Command Window** (MATLAB R2025b).

---

## 1. Indian Mixed-Traffic Scenarios (Custom Designed)

### 🌟 New Complex & Multi-Actor Scenarios

| Scenario | 3D Unreal Engine Mode (Records Video) | 2D BEV Animated HUD | Description |
| :--- | :--- | :--- | :--- |
| **The Gauntlet** *(Boss Level)* | `run_instruction('GAUNTLET', 'pp', '3d')` | `run_instruction('GAUNTLET', 'pp', 'animate')` | 8 actors, 3 phases: Rickshaw overtake → construction detour → jaywalker emergency brake |
| **School Zone Rush** | `run_instruction('SCHOOLZONE', 'pp', '3d')` | `run_instruction('SCHOOLZONE', 'pp', 'animate')` | 6 actors: Stopped school bus + 3 children darting across at different speeds + adult guardian |
| **Wrong-Way Head-On** | `run_instruction('WRONGWAY', 'pp', '3d')` | `run_instruction('WRONGWAY', 'pp', 'animate')` | 5 actors: Tempo approaching head-on in ego lane + delivery van blocking shoulder + oncoming bike |
| **Bus Stop Hazard** | `run_instruction('BUSSTOP', 'pp', '3d')` | `run_instruction('BUSSTOP', 'pp', 'animate')` | 5 actors: Decelerating BMTC bus + alighting passenger stepping out + high-speed overtaking motorcycle |
| **Vendor Cart Swerve** | `run_instruction('VENDORCART', 'pp', '3d')` | `run_instruction('VENDORCART', 'pp', 'animate')` | 5 actors: Slow street vendor cart swerving into lane + oncoming KSRTC bus + parked vehicle |

---

### 🇮🇳 Standard & Improved Indian Scenarios

| Scenario | 3D Unreal Engine Mode (Records Video) | 2D BEV Animated HUD | Description |
| :--- | :--- | :--- | :--- |
| **Pothole & Detour** | `run_instruction('POTHOLE', 'pp', '3d')` | `run_instruction('POTHOLE', 'pp', 'animate')` | 5 actors: Deep road defect + barrier detour into opposing lane with oncoming car |
| **Traffic Congestion** | `run_instruction('CONGESTION', 'pp', '3d')` | `run_instruction('CONGESTION', 'pp', 'animate')` | 7 actors: Crawling multi-vehicle queue + filtering bicycle weaving between queues |
| **Stray Cattle Cluster** | `run_instruction('CATTLE', 'pp', '3d')` | `run_instruction('CATTLE', 'pp', 'animate')` | 5 actors: Wandering cow + stationary cow at lane-divider + oncoming bus + shoulder pedestrian |
| **Mid-Block Jaywalk** | `run_instruction('JAYWALK', 'pp', '3d')` | `run_instruction('JAYWALK', 'pp', 'animate')` | 5 actors: Adult jaywalker + sprinting child emerging behind parked bus + oncoming scooter |
| **Auto-Rickshaw Cut-In** | `run_instruction('AUTOCUTIN', 'pp', '3d')` | `run_instruction('AUTOCUTIN', 'pp', 'animate')` | 3 actors: Bajaj auto-rickshaw aggressive lane cut-in + slow truck |
| **Two-Wheeler Lane Filter** | `run_instruction('TWOWHEELER', 'pp', '3d')` | `run_instruction('TWOWHEELER', 'pp', 'animate')` | 3 actors: Motorcycle lane-splitting along lane markings at 50 km/h |

---

## 2. Euro NCAP Safety Test Protocols

| Scenario | 3D Unreal Engine Mode (Records Video) | 2D BEV Animated HUD | Description |
| :--- | :--- | :--- | :--- |
| **CPNCO** | `run_instruction('CPNCO', 'pp', '3d')` | `run_instruction('CPNCO', 'pp', 'animate')` | Car-to-Pedestrian Nearside Child Obstructed (Euro NCAP 2023) |
| **CPTA** | `run_instruction('CPTA', 'pp', '3d')` | `run_instruction('CPTA', 'pp', 'animate')` | Car-to-Pedestrian Turning Adult at intersection (Euro NCAP 2023) |
| **CCFTAP** | `run_instruction('CCFTAP', 'pp', '3d')` | `run_instruction('CCFTAP', 'pp', 'animate')` | Car-to-Car Front Turn Across Path (Euro NCAP 2023) |
| **CCCSCP** | `run_instruction('CCCSCP', 'pp', '3d')` | `run_instruction('CCCSCP', 'pp', 'animate')` | Car-to-Car Straight Crossing Path intersection conflict |
| **CBFA** | `run_instruction('CBFA', 'pp', '3d')` | `run_instruction('CBFA', 'pp', 'animate')` | Car-to-Bicyclist Farside Adult crossing |
| **CCR** | `run_instruction('CCR', 'pp', '3d')` | `run_instruction('CCR', 'pp', 'animate')` | Car-to-Car Rear-end collision avoidance |

---

## 3. ASAM OpenSCENARIO Standard Examples

| Scenario | 3D Unreal Engine Mode (Records Video) | 2D BEV Animated HUD | Description |
| :--- | :--- | :--- | :--- |
| **Lane Change** | `run_instruction('LANECHANGE', 'pp', '3d')` | `run_instruction('LANECHANGE', 'pp', 'animate')` | ASAM standard overtaking lane change behind slow vehicle |
| **Overtake** | `run_instruction('OVERTAKE', 'pp', '3d')` | `run_instruction('OVERTAKE', 'pp', 'animate')` | High-speed pass around slow commercial truck |
| **Pedestrian Crossing** | `run_instruction('PEDCROSSING', 'pp', '3d')` | `run_instruction('PEDCROSSING', 'pp', 'animate')` | Perpendicular pedestrian crossing with AEB response |

---

## 4. Execution Modifiers & Advanced Options

### Controller Selection
Replace `'pp'` (Pure Pursuit) with `'mpc'` (Model Predictive Control) for any scenario:
```matlab
% Example with MPC Controller:
run_instruction('GAUNTLET', 'mpc', '3d')
run_instruction('POTHOLE', 'mpc', 'animate')
```

### Custom Simulation Duration
Append duration in seconds as the 4th argument:
```matlab
% Run for 20 seconds instead of default 30s:
run_instruction('SCHOOLZONE', 'pp', '3d', 20)
run_instruction('GAUNTLET', 'pp', '3d', 45)
```

### Full Integrated AV Stack Mode (`'stack'`)
Runs closed-loop simulation with C3 YOLOv8s 12-class perception, Semantic IMM Kalman Filter, and GMM Trajectory Prediction:
```matlab
run_instruction('CPNCO', 'pp', 'stack')
run_instruction('POTHOLE', 'pp', 'stack')
```

### Driving Scenario Designer GUI App (`'app'`)
Directly inspect or edit actor waypoints and road geometries in MATLAB GUI:
```matlab
run_instruction('GAUNTLET', 'pp', 'app')
run_instruction('SCHOOLZONE', 'pp', 'app')
run_instruction('POTHOLE', 'pp', 'app')
```

---

## 5. Where to Find Generated Outputs

### 3D Unreal Engine Mode (`'3d'`)
- **Quick-Access Video (Overwritten on each run)**:
  - `D:\hackathon\SIH26\unreal_simulation_video.mp4`
- **Timestamped Session Folder (Preserved History)**:
  - `D:\hackathon\SIH26\logs\run_<TIMESTAMP>_<SCENARIO>_<CONTROLLER>\`
    - `unreal_simulation_video.mp4` — Full HD 30 FPS video
    - `images/surround_cockpit/` — Synchronized multi-camera HUD mosaics
    - `graphs/` — Speed profile, steering angle, TTC, and safety margin analysis plots

### 2D HUD Mode (`'animate'`)
- Interactive MATLAB figure with real-time Birds-Eye View, radar point clouds, camera 2D/3D bounding boxes, and TTC supervisor metrics.
