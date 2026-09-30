# How to Run Autonomous Driving Scenarios

Run all commands directly inside the **MATLAB Command Window** (MATLAB R2025b).
Before Everything, make sure to run startup.m on matlab to add all paths

---

## 1. Quick Run Commands

| Scenario | Command | Mode / Description |
| :--- | :--- | :--- |
| **Pothole & Detour** | `run_instruction('POTHOLE', 'pp', '3d')` | Unreal Engine 3D Simulation + Video Record |
| | `run_instruction('POTHOLE', 'pp', 'animate')` | Live 2D Bird's-Eye View HUD |
| **Traffic Congestion** | `run_instruction('CONGESTION', 'pp', '3d')` | Unreal Engine 3D Simulation + Video Record |
| | `run_instruction('CONGESTION', 'pp', 'animate')` | Live 2D Bird's-Eye View HUD |
| **Pedestrian AEB (CPNCO)** | `run_instruction('CPNCO', 'pp', '3d')` | Nearside Child Crossing (3D Video) |
| | `run_instruction('CPNCO', 'pp', 'animate')` | Live 2D Bird's-Eye View HUD |
| **Auto-Rickshaw Cut-In** | `run_instruction('AUTOCUTIN', 'pp', 'animate')` | Indian Auto Cut-In Avoidance |
| **Two-Wheeler Filter** | `run_instruction('TWOWHEELER', 'pp', 'animate')` | Motorbike Lane-Splitting Avoidance |
| **Stray Cattle Hazard** | `run_instruction('CATTLE', 'pp', 'animate')` | Animal Obstacle Detection & Stop |

> **Tips:**
> - Change controller: replace `'pp'` (Pure Pursuit) with `'mpc'` (Model Predictive Control).
> - Custom duration: append seconds, e.g. `run_instruction('POTHOLE', 'pp', '3d', 20)`.

---

## 2. Where to Expect Outputs

### 3D Unreal Engine Mode (`'3d'`)
- **Quick-Access Video**:
  - `D:\hackathon\SIH26\unreal_simulation_video.mp4` *(Root directory)*
- **Timestamped Session Folder**:
  - `D:\hackathon\SIH26\logs\run_<TIMESTAMP>_<SCENARIO>_<CONTROLLER>\`
    - `unreal_simulation_video.mp4` (Full recording)
    - `images/surround_cockpit/` (Multi-camera HUD mosaics)
    - `graphs/` (Speed, steering, TTC & safety margin plots)

### 2D HUD Interactive Mode (`'animate'`)
- **Live Visual Window**: Opens an interactive MATLAB Bird's-Eye View figure showing ego path, radar tracks, camera boxes, and TTC supervisor status in real time.

### Driving Scenario Designer App (`'app'`)
- `run_instruction('POTHOLE', 'pp', 'app')`
- Directly opens the scenario inside MATLAB's **Driving Scenario Designer** GUI.
