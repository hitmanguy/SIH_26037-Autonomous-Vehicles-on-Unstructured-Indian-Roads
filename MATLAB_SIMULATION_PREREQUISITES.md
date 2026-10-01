# MATLAB Simulation Prerequisites & File Reference Guide
**Smart India Hackathon (SIH) 2026 — Problem Statement 26037**  
*Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads*

This document lists all required MATLAB toolboxes and every file needed to start and run the autonomous vehicle simulation.

---

## 1. Required MATLAB Toolboxes & Products

| # | Toolbox / Product | Purpose in the Stack | Requirement |
| :-: | :--- | :--- | :-: |
| 1 | **MATLAB & Simulink** (R2023b or later) | Base numerical execution, multi-rate ODE solver, and block canvas. | **Mandatory** |
| 2 | **Automated Driving Toolbox** | Driving scenarios, actor modeling, sensor rigs, and RoadRunner co-simulation. | **Mandatory** |
| 3 | **Sensor Fusion and Tracking Toolbox** | Multi-sensor coincidence, Covariance Intersection, and IMM-Kalman filtering. | **Mandatory** |
| 4 | **Navigation Toolbox** | Dynamic `vehicleCostmap`, cost inflation, and collision checking. | **Mandatory** |
| 5 | **Stateflow** | Behavioral supervisory finite-state machine (CRUISE, SLOW_DOWN, YIELD, STOP, REROUTE). | **Mandatory** |
| 6 | **Computer Vision Toolbox** | 2D bounding boxes, camera extrinsics/intrinsics, and inverse ground projection. | **Mandatory** |
| 7 | **Deep Learning Toolbox** | MotionFormer ONNX network inference and neural trajectory prediction. | **Recommended** |
| 8 | **Vehicle Dynamics Blockset** | 3D Unreal Engine co-simulation vehicle actors and bicycle model propagation. | **Recommended** |
| 9 | **Simulink Coder / Embedded Coder** | Fixed-size static array compilation (`#codegen`) for planning and prediction blocks. | **Recommended** |
| 10 | **RoadRunner & RoadRunner Scenario** | 3D scenario playback, Indian road networks (`.xodr`), and OpenSCENARIO (`.xosc`). | **Recommended** |

---

## 2. Files Needed to Start the Simulation

### A. Master Entry Points & Launchers
- **[`main/run_instruction.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/main/run_instruction.m)**  
  Primary interactive MATLAB runner; prompts the user to select one of the 11 Indian scenarios and executes the full stack.
- **[`main/run_integrated_pipeline.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/main/run_integrated_pipeline.m)**  
  Automated multi-rate verification script executing perception, fusion, prediction, planning, and control end-to-end.
- **[`main/AutonomousAVStack.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/main/AutonomousAVStack.m)**  
  Master multi-rate class scheduler coordinating camera (15.6 Hz), radar (20 Hz), IMM fusion (50 Hz), prediction (10 Hz), and control (50 Hz).

---

### B. Track 1: Perception
- **[`Perception/sahi_visualizer_and_benchmark.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Perception/sahi_visualizer_and_benchmark.m)**  
  Evaluates C3 YOLOv8 detection accuracy and visualizes SAHI slicing on Indian Driving Dataset (IDD) camera frames.
- **[`Perception/c3_idd_detections.mat`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Perception/c3_idd_detections.mat)**  
  Precomputed YOLOv8 12-class detection bounding boxes, scores, and labels from real Indian roadway images.

---

### C. Track 2: Sensor Fusion
- **[`Sensor_fusion/c3_vision_radar_fusion_bridge.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Sensor_fusion/c3_vision_radar_fusion_bridge.m)**  
  Projects 2D camera detections to 3D ego metric coordinates and fuses 77 GHz radar Doppler reflections.
- **[`Sensor_fusion/c3_semantic_imm_tracker.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Sensor_fusion/c3_semantic_imm_tracker.m)**  
  Benchmarks the Semantic IMM tracker with autorickshaw lateral-swerve and animal freeze motion models.

---

### D. Track 3: Trajectory Prediction
- **[`Trajectory/trajectory_prediction_engine.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/trajectory_prediction_engine.m)**  
  Generates multi-modal GMM future trajectories ($K=3$: Nominal, Swerve, Freeze) over a 3.0s horizon.
- **[`Trajectory/prediction_to_costmap_bridge.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/prediction_to_costmap_bridge.m)**  
  Transforms predicted GMM trajectories and pothole tokens into dynamic spatio-temporal costmaps and Stateflow triggers.
- **[`Trajectory/setup_trajectory_simulink.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/setup_trajectory_simulink.m)**  
  Initializes Simulink Bus objects and workspace parameters for the trajectory prediction subsystem.
- **[`Trajectory/simulink_prediction_block.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/simulink_prediction_block.m)**  
  Embedded C-coder compliant (`#codegen`) MATLAB Function block executing prediction inside Simulink at 10 Hz.

---

### E. Track 4: Path Planning & Decision
- **[`Path_planning_decision/dynamic_trajectory_planner.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/dynamic_trajectory_planner.m)**  
  Online 10 Hz Frenet quintic lattice replanner with pothole detour, GMM collision checks, and oncoming traffic avoidance.
- **[`Path_planning_decision/decision_supervisor.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/decision_supervisor.m)**  
  Tactical supervisor managing state transitions across CRUISE, SLOW_DOWN, YIELD, STOP, and REROUTE modes.
- **[`Path_planning_decision/dynamic_costmap_manager.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/dynamic_costmap_manager.m)**  
  Maintains multi-layer `vehicleCostmap` with virtual lane corridors, pothole depth tiers, and actor risk zones.
- **[`Path_planning_decision/hybrid_astar_planner.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/hybrid_astar_planner.m)**  
  Non-holonomic bicycle kinematic A* search extracting safe convex corridors under heavy blockage.
- **[`Path_planning_decision/continuous_trajectory_optimizer.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/continuous_trajectory_optimizer.m)**  
  Tesla-inspired banded QP spline optimizer enforcing $C^2$ continuity, jerk bounds, and road boundaries.
- **[`Path_planning_decision/setup_planning_simulink.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/setup_planning_simulink.m)**  
  Defines Simulink Bus structures (`BusEgoState`, `BusPlannedTrajectory`) for the planning layer.
- **[`Path_planning_decision/simulink_planning_block.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/simulink_planning_block.m)**  
  Deterministic `#codegen` planning block generating a 40-point $[X, Y, \theta, \kappa, v, a]$ reference trajectory.

---

### F. Track 5: Vehicle Dynamics & Control
- **[`Vehicle_dynamics/autonomous_ego_controller.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Vehicle_dynamics/autonomous_ego_controller.m)**  
  Unified manager executing longitudinal ACC/AEB braking, lateral steering tracking, and kinematic bicycle propagation.
- **[`Vehicle_dynamics/pure_pursuit_controller.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Vehicle_dynamics/pure_pursuit_controller.m)**  
  Adaptive Pure Pursuit lateral controller dynamically scaling lookahead distance with forward velocity and lateral offset.
- **[`Vehicle_dynamics/mpc_lane_controller.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Vehicle_dynamics/mpc_lane_controller.m)**  
  Discrete bicycle-model Model Predictive Controller optimizing steering slew rate and cross-track error.
- **[`Vehicle_dynamics/sensor_fusion_bridge.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Vehicle_dynamics/sensor_fusion_bridge.m)**  
  Multi-sensor fusion bridge with Covariance Intersection, M-of-N persistence filtering, and two-tier AEB authority.
- **[`Vehicle_dynamics/sensor_rig_builder.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Vehicle_dynamics/sensor_rig_builder.m)**  
  Mounts and configures the 4 surround cameras, 2 automotive radars, and roof LiDAR onto the Ego vehicle.
- **[`Vehicle_dynamics/build_sim3d_harness.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Vehicle_dynamics/build_sim3d_harness.m)**  
  Programmatically generates the Simulink 3D Unreal Engine surround camera harness (`sim3d_surround_harness.slx`).
- **[`Vehicle_dynamics/sim3d_surround_harness.slx`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Vehicle_dynamics/sim3d_surround_harness.slx)**  
  Simulink canvas wiring 3D vehicle actor and surround camera feeds for high-fidelity Unreal Engine co-simulation.

---

### G. Scenarios & Road Networks
- **[`scenarios/scenarios_xosc/indian/`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/scenarios/scenarios_xosc/indian/)**  
  Contains the 11 verified OpenSCENARIO (`.xosc`) files:
  1. `Indian_AutoRickshaw_CutIn.xosc` — Aggressive autorickshaw cut-in and lane squeeze.
  2. `Indian_TwoWheeler_LaneFilter.xosc` — Motorcycle lane filtering between vehicles.
  3. `Indian_Pedestrian_Jaywalk.xosc` — Pedestrian crossing across unmarked arterial road.
  4. `Indian_StrayCattle_Hazard.xosc` — Stray cattle stepping into lane and freezing dead.
  5. `Indian_Traffic_Congestion.xosc` — Dense stop-and-go mixed traffic corridor.
  6. `Indian_Pothole_Detour.xosc` — Deep road surface cavity ($>5\text{ cm}$) requiring lane detour.
  7. `Indian_WrongWay_Encounter.xosc` — Oncoming vehicle traveling against traffic on our side.
  8. `Indian_SchoolZone_Rush.xosc` — High-density pedestrian and low-speed vehicle congestion.
  9. `Indian_BusStop_Hazard.xosc` — City bus stopped with pedestrians emerging from blind spot.
  10. `Indian_VendorCart_Swerve.xosc` — Slow pushcart protruding into lane forcing evasive swerve.
  11. `Indian_MultiThreat_Gauntlet.xosc` — Full composite gauntlet testing all threats simultaneously.
- **[`scenarios/maps_xodr/`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/scenarios/maps_xodr/)**  
  OpenDRIVE (`.xodr`) road geometry definitions loaded by RoadRunner and the MATLAB Driving Scenario engine.

---

### H. Closed-Loop Simulink & RoadRunner Simulation (`Simulink_Simulation/`)
- **[`Simulink_Simulation/setup_simulink_simulation.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Simulink_Simulation/setup_simulink_simulation.m)**  
  Master initialization script configuring multi-rate clocks ($100\text{ Hz}$ dynamics, $50\text{ Hz}$ control, $20\text{ Hz}$ fusion, $10\text{ Hz}$ planning), all 7 Simulink Bus objects, vehicle parameters, and RoadRunner co-simulation tokens.
- **[`Simulink_Simulation/build_closed_loop_model.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Simulink_Simulation/build_closed_loop_model.m)**  
  Programmatic Simulink model generator compiling `SIH26037_ClosedLoop_EgoSimulator.slx` connecting all 8 subsystems in closed loop.
- **[`Simulink_Simulation/run_closed_loop_simulation.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Simulink_Simulation/run_closed_loop_simulation.m)**  
  Automated multi-rate closed-loop simulation runner executing OpenSCENARIO scenarios, logging real-time telemetry, and plotting bird's-eye view trajectories.
- **[`Simulink_Simulation/test_simulink_pipeline.py`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Simulink_Simulation/test_simulink_pipeline.py)**  
  Unit test suite validating multi-rate clock synchronization, bicycle model kinematics, Pure Pursuit lateral convergence, and emergency braking buffers.

---

## 3. How to Launch Simulation in MATLAB

In the MATLAB Command Window, run:
```matlab
% 1. Navigate to repository root
cd('/path/to/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads');

% 2. Launch interactive scenario simulation
addpath(genpath('.'));
run('main/run_instruction.m');
```
Alternatively, to run the automated end-to-end benchmark across all subsystems:
```matlab
run('main/run_integrated_pipeline.m');
```
