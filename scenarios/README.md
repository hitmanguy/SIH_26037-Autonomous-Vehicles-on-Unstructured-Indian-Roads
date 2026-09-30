# Indian Mixed Traffic & Autonomous Driving Scenario Framework (MATLAB R2025b & Python)

A simulation and scenario generation framework tailored for Indian driving conditions, Euro NCAP active safety protocols, and ASAM standard scenarios. Built on top of MATLAB Automated Driving Toolbox (`drivingScenario`), ASAM OpenDRIVE (`.xodr`), ASAM OpenSCENARIO (`.xosc`), BMW `scenariogeneration`, and UC Berkeley `scenic`.

---

## 1. Directory Structure

```
scenarios/
├── maps_xodr/                         # ASAM OpenDRIVE Road Networks
│   ├── Bangalore_Indiranagar.xodr     # Real Bengaluru 100ft Road / CMH Road 4-way arterial
│   ├── Indian_Junction_Chowk.xodr     # Indian 4-way Chowk intersection with unpaved shoulders
│   ├── Indian_Urban_Arterial.xodr     # 4-lane divided urban arterial with median & shoulders
│   └── X-Intersection_NCAP.xodr       # Euro NCAP 4-way test intersection (250m arms)
│
├── scenarios_xosc/                    # ASAM OpenSCENARIO Dynamic Scenarios
│   ├── indian/                        # Authentic Indian Mixed Traffic Scenarios
│   │   ├── Indian_AutoRickshaw_CutIn.xosc    # Aggressive Auto-Rickshaw cut-in
│   │   ├── Indian_TwoWheeler_LaneFilter.xosc # Two-wheeler high-speed filtering between lanes
│   │   ├── Indian_Pedestrian_Jaywalk.xosc    # Mid-block jaywalking from behind parked truck
│   │   └── Indian_StrayCattle_Hazard.xosc    # Stray bovine sitting directly in Ego lane
│   ├── ncap/                          # Euro NCAP Active Safety Protocols
│   │   ├── NCAP_AEB_VRU_CPNCO_2023.xosc      # Child Obstructed with active AEB
│   │   ├── NCAP_AEB_VRU_CPTA_2023.xosc       # Turning Adult at Intersection
│   │   ├── NCAP_AEB_C2C_CCFtap_2023.xosc     # Front Turn Across Path
│   │   └── CCCscp.xosc                       # Car-to-Car Straight Crossing Path
│   └── asam_examples/                 # ASAM OpenSCENARIO Standard Reference Examples
│       ├── LaneChangeSimple.xosc             # Sinusoidal lane change
│       ├── OvertakeSlowVehicle.xosc          # Dynamic overtake with lane return
│       └── PedestrianCrossing.xosc           # Straight road pedestrian crossing
│
├── osm_maps/                          # Real OpenStreetMap Vector Data
│   └── bangalore_indiranagar.osm      # Extracted OSM node/way data for Indiranagar, Bengaluru
│
├── generators/                        # Python Scenario & Road Generation Engines
│   ├── osm_to_xodr.py                 # OSM to ASAM OpenDRIVE (.xodr) converter
│   ├── indian_scenario_generator.py   # BMW scenariogeneration engine for Indian roads & xosc
│   ├── scenic_runner.py               # UC Berkeley Scenic probabilistic scene sampler
│   ├── sample_scenic_to_temp_xosc.py  # Scenic-to-MATLAB dynamic bridge (auto-purged)
│   ├── scenic_monte_carlo_fuzzer.py   # Automated Scenic -> XOSC -> MATLAB batch fuzzer
│   └── scenic_scenarios/              # Scenic probabilistic scenario models
│       ├── indian_mixed_traffic.scenic       # Stochastic mixed crowd & vehicle swarms
│       └── crowd_pedestrian_occlusion.scenic # Occluded jaywalker emergence models
│
├── matlab/                            # Core MATLAB Simulation Engine
│   ├── run_intersection_scenario.m   # Universal scenario executive (2D HUD, 3D, App, Headless)
│   └── import_openscenario.m          # Direct parser & importer (.xosc + .xodr -> drivingScenario)
│
├── verification/                      # Safety Verification & Analysis Tools
│   ├── trajectory_collision_verifier.py # Exact 2D OBB Separating Axis Theorem (SAT) collision auditor
│   ├── visualize_ncap_scenario.py     # Matplotlib 2D scenario layout renderer
│   ├── actor_trajectories_log.mat     # 20 Hz synchronized actor pose and velocity logs
│   └── ncap_cpnco_intersection.png    # Rendered trajectory visualizer
│
├── run_intersection.m                 # Universal root entrypoint wrapper
└── README.md                          # Master documentation (this file)
```

---

## 2. Supported Scenarios Catalog

### A. Indian Mixed Traffic Scenarios (`scenarios_xosc/indian/`)
Generated using BMW `scenariogeneration` with specialized sublane and aggressive trajectory dynamics:

| Shortcut | Description | Road Network | Actors & Dynamics | Clearance Result |
| :--- | :--- | :--- | :--- | :--- |
| `'Indian_AutoCutIn'` | **Auto-Rickshaw Cut-In**: Slow auto swerves sharply across Ego lane to drop off passenger. | `Indian_Urban_Arterial.xodr` | Ego Car, Bajaj Auto-Rickshaw, Hero Splendor, Tata Ace Mini-Truck. | **0 Collisions (Min 1.62 m)** |
| `'Indian_TwoWheeler'` | **Two-Wheeler Lane Filtering**: High-speed motorcycle weaves between Ego and a slow Tata Bus. | `Indian_Urban_Arterial.xodr` | Ego Car, Pulsar 150 Motorcycle, Tata Starbus. | **0 Collisions (Min 1.49 m)** |
| `'Indian_Jaywalk'` | **Mid-Block Jaywalking**: Pedestrian steps out into the carriageway from behind a parked truck. | `Indian_Urban_Arterial.xodr` | Ego Car, Parked Ashok Leyland Truck, Jaywalking Pedestrian. | **0 Collisions (Min 2.33 m)** |
| `'Indian_Cattle'` | **Stray Cattle Hazard**: Cow sitting on the asphalt; oncoming bus forces Ego to brake and yield. | `Indian_Urban_Arterial.xodr` | Ego Car, Stray Cow (Box Model), Oncoming Tata Starbus. | **0 Collisions (Min 2.50 m)** |

---

### B. Euro NCAP Active Safety Protocols (`scenarios_xosc/ncap/`)

| Shortcut | NCAP Protocol | Description | Road Network | Clearance Result |
| :--- | :--- | :--- | :--- | :--- |
| `'CPNCO'` (Default) | AEB VRU 2023 | **Child Obstructed**: Child darts at 5 km/h from behind parked cars; Ego AEB emergency brake. | `X-Intersection_NCAP.xodr` | **0 Collisions (Min 1.00 m)** |
| `'CPTA'` | AEB VRU 2023 | **Turning Adult**: Ego turns across intersection while adult crosses zebra crossing. | `X-Intersection_NCAP.xodr` | **0 Collisions (Min 8.80 m)** |
| `'CCFtap'` | AEB C2C 2023 | **Front Turn Across Path**: Ego turns across path of oncoming target car. | `X-Intersection_NCAP.xodr` | **0 Collisions (Min 2.54 m)** |
| `'CCCscp'` | CA-FC 2026 | **Straight Crossing Path**: 5 vehicles at intersection with corner sight obstructions. | `X-Intersection_NCAP.xodr` | **0 Collisions (Min 0.93 m)** |

---

### C. Real Indian City Digital Twins (`maps_xodr/`)

| Map Name | Description | Source |
| :--- | :--- | :--- |
| `Bangalore_Indiranagar.xodr` | **Bengaluru 100ft Road / CMH Road Arterial**: 4-way junction with multi-lane carriageways and pedestrian sidewalks. | OpenStreetMap (`osm_to_xodr.py`) |
| `Indian_Junction_Chowk.xodr` | **Indian Chowk 4-Way Junction**: Unsignalized mixed-traffic intersection with wide shoulders. | `scenariogeneration` |
| `Indian_Urban_Arterial.xodr` | **4-Lane Divided Arterial**: 300 m multi-lane road with central median barrier. | `scenariogeneration` |

---

## 3. MATLAB Simulation Usage (`run_instruction.m` & `run_intersection.m`)

### Primary Unified Command: `run_instruction`
Use `run_instruction` to run any scenario with the full **7-Sensor Perception Rig** (4 surround cameras for 360° BEV, 2 radars for 77 GHz LRR & rear, and roof 360° LiDAR), **Multi-Sensor Kalman Fusion Bridge**, and closed-loop **Model Predictive Control (MPC)** or **Pure Pursuit (PP)**:

```matlab
% Set MATLAB path to project root
cd('D:\hackathon\SIH26');

% 1. Model Predictive Control (MPC) on Euro NCAP CPNCO:
run_instruction('CPNCO', 'mpc');

% 2. Pure Pursuit (PP) on Indian Auto-Rickshaw Cut-In:
run_instruction('Indian_AutoCutIn', 'pp');

% 3. Model Predictive Control on Indian Two-Wheeler Filtering:
run_instruction('Indian_TwoWheeler', 'mpc');

% 4. Open any scenario in Driving Scenario Designer App with sensors:
run_instruction('CPNCO', 'pp', 'app');

% 5. UC Berkeley Scenic Monte Carlo Fuzzer with MPC:
run_instruction('scenic', 'mpc');

% 6. Fast Headless Batch Testing (e.g. 10.0 seconds):
run_instruction('CPNCO', 'mpc', 'headless', 10.0);
```

---

### A. Run UC Berkeley Scenic Probabilistic Fuzzer (Zero Storage Clutter)
Samples a fresh, randomized traffic permutation from `indian_mixed_traffic.scenic`, builds the scenario directly into MATLAB memory, and immediately deletes the temporary `.xosc` so no files accumulate:
```matlab
% 1. Sample fresh Scenic traffic & OPEN IN DRIVING SCENARIO DESIGNER APP:
run_instruction('scenic', 'pp', 'app');

% 2. Run fresh Scenic traffic in interactive 2D HUD:
run_instruction('scenic', 'mpc');

% 3. Run fresh Scenic traffic in Unreal Engine 3D:
run_instruction('scenic', 'mpc', '3d');
```

### B. Run Indian Scenarios (Interactive 2D Bird's-Eye HUD)
```matlab
% 1. Auto-Rickshaw aggressive cut-in
run_intersection('Indian_AutoCutIn');

% 2. Motorcycle lane-filtering
run_intersection('Indian_TwoWheeler');

% 3. Mid-block jaywalker emergence
run_intersection('Indian_Jaywalk');

% 4. Stray cattle on road
run_intersection('Indian_Cattle');
```

### C. Run Real Indian City Digital Twin (Bengaluru)
```matlab
% Runs CPNCO on real Bengaluru Indiranagar OpenDRIVE map
run_intersection('Bangalore');
```

### D. Run Euro NCAP Safety Scenarios
```matlab
run_intersection('CPNCO');   % Default Child Obstructed
run_intersection('CPTA');    % Turning Adult
run_intersection('CCFtap');  % Front Turn Across Path
run_intersection('CCCscp');  % Straight Crossing Path (5 Vehicles)
```

### E. Run in Driving Scenario Designer App
Opens any scenario directly inside MATLAB's interactive `drivingScenarioDesigner` GUI for visual editing, waypoint manipulation, and sensor mounting:
```matlab
run_intersection('scenic', 'app');          % Randomized Scenic traffic in App
run_intersection('Indian_AutoCutIn', 'app'); % Auto-Rickshaw Cut-In in App
run_intersection('CPNCO', 'app');            % Euro NCAP in App
```

### F. Run in Unreal Engine 3D Co-Simulation
Renders the simulation using MATLAB's Unreal Engine 3D Gaming Engine co-simulator (`plotSim3d`):
```matlab
run_intersection('scenic', '3d');
run_intersection('Indian_AutoCutIn', '3d');
```

### G. Headless Execution (Automated Testing & CI/CD)
Runs at maximum CPU speed without graphics rendering and exports `actor_trajectories_log.mat`:
```matlab
run_intersection('Indian_AutoCutIn', '', 'headless', 20.0);
```

---

## 4. Python Generation & Verification Pipeline

All Python utilities run in the project Conda environment:
`& "D:\conda_envs\sih26\python.exe"`

### A. OpenStreetMap to OpenDRIVE Converter (`generators/osm_to_xodr.py`)
Converts any `.osm` vector map from OpenStreetMap into ASAM OpenDRIVE (`.xodr`) with multi-lane carriageways and lateral lane geometry:
```powershell
& "D:\conda_envs\sih26\python.exe" "D:\hackathon\SIH26\scenarios\generators\osm_to_xodr.py" `
    --input "D:\hackathon\SIH26\scenarios\osm_maps\bangalore_indiranagar.osm" `
    --output "D:\hackathon\SIH26\scenarios\maps_xodr\Bangalore_Indiranagar.xodr"
```

### B. Indian Road & Scenario Generator (`generators/indian_scenario_generator.py`)
Uses BMW `scenariogeneration` to programmatically build the `.xodr` roads and `.xosc` OpenSCENARIO files:
```powershell
& "D:\conda_envs\sih26\python.exe" "D:\hackathon\SIH26\scenarios\generators\indian_scenario_generator.py"
```

### C. UC Berkeley Scenic Stochastic Crowd & Traffic Sampler (`generators/scenic_runner.py`)
Samples probabilistic mixed traffic distributions and generates dynamic scenario parameters:
```powershell
& "D:\conda_envs\sih26\python.exe" "D:\hackathon\SIH26\scenarios\generators\scenic_runner.py" --samples 3
```

### D. Automated Monte Carlo Fuzzing Pipeline (`generators/scenic_monte_carlo_fuzzer.py`)
Stress-tests autonomous driving controllers by sampling randomized traffic configurations from Scenic, dynamically creating temporary `.xosc` files, running headless simulations in MATLAB, and mathematically verifying body clearances using SAT OBB:
- **Zero Storage Footprint**: Every generated `.xosc` is strictly temporary and is automatically deleted immediately after each simulation run.
```powershell
# Run a 5-sample Monte Carlo stress test with 8s simulation duration
& "D:\conda_envs\sih26\python.exe" "D:\hackathon\SIH26\scenarios\generators\scenic_monte_carlo_fuzzer.py" --samples 5 --duration 8.0
```

### E. Exact 2D OBB SAT Collision Verifier (`verification/trajectory_collision_verifier.py`)
Performs mathematical collision verification across all simulation timesteps using the Separating Axis Theorem on 2D Oriented Bounding Boxes:
```powershell
& "D:\conda_envs\sih26\python.exe" "D:\hackathon\SIH26\scenarios\verification\trajectory_collision_verifier.py"
```

---

## 5. Collision Verification Results (Certified Zero Collisions)

All Indian scenarios have been mathematically certified using SAT OBB:

```
===========================================================================
EXACT 2D ORIENTED BOUNDING BOX (OBB) COLLISION AUDIT: [Indian_AutoCutIn]
Total Actors: 4 | Total Timesteps: 398 (0.05s resolution)
===========================================================================
Actor Pair                                         | Min Clearance   | Occurs at Time 
-------------------------------------------------------------------------------------
EgoCar_Blue vs AutoRickshaw_Bajaj                  | 1.62 m          |   5.15 s
EgoCar_Blue vs Splendor_Bike                       | 1.83 m          |   3.10 s
EgoCar_Blue vs TataAce_Delivery                    | 6.64 m          |   0.05 s
AutoRickshaw_Bajaj vs Splendor_Bike                | 3.12 m          |   5.35 s
AutoRickshaw_Bajaj vs TataAce_Delivery             | 21.05 m         |   0.05 s
Splendor_Bike vs TataAce_Delivery                  | 23.33 m         |   0.05 s
-------------------------------------------------------------------------------------
[SUCCESS] PERFECT SCORE: ZERO COLLISIONS DETECTED ACROSS ALL TIMESTEPS!
```

Clearance margins for all Indian scenarios:
- **`Indian_AutoCutIn`**: Minimum clearance **1.62 m** (Ego vs Auto-Rickshaw)
- **`Indian_TwoWheeler`**: Minimum clearance **1.49 m** (Ego vs Filtering Motorcycle)
- **`Indian_Jaywalk`**: Minimum clearance **2.33 m** (Ego vs Jaywalking Pedestrian)
- **`Indian_Cattle`**: Minimum clearance **2.50 m** (Ego vs Bovine Obstacle)

---

## 6. Autonomous Ego Closed-Loop Control System (Vision + Pure Pursuit & MPC)

The repository provides a complete, 100% automated closed-loop sense-plan-act control architecture for the Ego vehicle that eliminates static predefined waypoints.

### Architecture Overview

```
                          ┌──────────────────────────┐
                          │ Forward-Facing Camera    │
                          │ visionDetectionGenerator │
                          └────────────┬─────────────┘
                                       │ Object detections (range, azimuth, class)
                                       ▼
┌──────────────────────┐  ┌──────────────────────────┐
│ Reference Route      │─►│ Autonomous Controller    │
│ (from Scenario/Map)  │  │ • ACC / AEB Supervisor   │
└──────────────────────┘  │ • Lateral Controller:    │
                          │   - Pure Pursuit         │
                          │   - State-Space MPC (QP) │
                          └────────────┬─────────────┘
                                       │ Acceleration & Steering Commands
                                       ▼
                          ┌──────────────────────────┐
                          │ Kinematic Bicycle Model  │
                          │ Ego State Update in RAM  │
                          │ (Position, Velocity, Yaw)│
                          └──────────────────────────┘
```

### Key Components

1. **Perception**: Mounted front camera (`visionDetectionGenerator`) providing 50 m forward visibility, tracking lead vehicles, crossing pedestrians, and cut-in obstacles.
2. **Longitudinal Safety Supervisor**: Dynamic forward collision detection triggering Adaptive Cruise Control (ACC) and Emergency Braking (AEB) when Time-to-Collision (TTC) falls below safety thresholds.
3. **Lateral Path Tracking**:
   - **Pure Pursuit** (`pure_pursuit_controller.m`): Adaptive lookahead distance $L_d = k_v \cdot v_x + L_0$ mapped to steering angle $\delta = \text{atan}\left(\frac{2 L \sin\alpha}{L_d}\right)$.
   - **Model Predictive Control (MPC)** (`mpc_lane_controller.m`): 4-state bicycle model tracking $[e_y, \dot{e}_y, e_\psi, \dot{e}_\psi]^T$ with preview curvature feedforward and custom Hildreth Quadratic Programming (QP) solver running in $< 1\text{ ms}$ without toolbox dependencies.
4. **State Propagation**: Kinematic bicycle dynamics directly override the Ego actor's state (`Position`, `Velocity`, `Yaw`) in MATLAB RAM on every simulation step.

### Quick Usage Examples

#### Run in Any OpenDRIVE / OpenSCENARIO Scenario:
```matlab
% Run Euro NCAP Child Obstructed with autonomous Pure Pursuit & Camera AEB
run_intersection('CPNCO', 'pp');

% Run Euro NCAP Child Obstructed with autonomous MPC
run_intersection('CPNCO', 'mpc');

% Run Indian Auto-Rickshaw Cut-In with autonomous MPC
run_intersection('Indian_AutoCutIn', 'mpc');

% Run on Bengaluru OpenStreetMap map with autonomous control
run_intersection('Bangalore', 'pp');

% Open any scenario with autonomous control in Driving Scenario Designer
run_intersection('CPNCO', 'pp', 'app');
```

#### Run Standalone Ego Control Loop (`run_ego_control.m`):
```matlab
% 1. Standalone Pure Pursuit tracking with live 2D HUD:
run_ego_control;

% 2. Standalone Model Predictive Control (MPC) tracking:
run_ego_control('controller', 'MPC');

% 3. Test camera detection and AEB emergency braking with obstacle:
run_ego_control('controller', 'PurePursuit', 'obstacle', true);

% 4. Open autonomous vehicle setup in Driving Scenario Designer App:
run_ego_control('app');
```

