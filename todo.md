# Closed-Loop Autonomous Driving Simulator & Pipeline Implementation Plan
## Problem Statement ID: 26037 — Team Epsilon
### Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads (RoadRunner + Simulink + MATLAB)

---

## 1. Executive Overview & Target System Architecture

The goal is to build, integrate, and test a **closed-loop simulation in MATLAB and Simulink** for a defined **Ego Vehicle** operating in **RoadRunner** 3D Indian scenes.

The closed-loop architecture operates on a multi-rate clock where the Ego Vehicle perceives, tracks, predicts, plans, controls, and moves within the 3D world:

```
+──────────────────────────────────────────────────────────────────────────────────────────+
|                              ROADRUNNER 3D SCENARIO SIMULATION                           |
|  - 3D Scenes: Village Road, Unsignalled Intersection, Highway Merge, Market, Cattle     |
|  - Traffic Actors: Auto-rickshaws, stray cattle, motorcycles, bicycles, pedestrians      |
|  - Surface Hazards: Pothole depressions, unpaved shoulders, missing lane markings        |
+──────────────────────────────────────────────────────────────────────────────────────────+
               │ [Surround Cameras / LiDAR / Radar / Actor Poses]
               ▼
+──────────────────────────────────────────────────────────────────────────────────────────+
|                               SIMULINK CLOSED-LOOP EGO PIPELINE                          |
|                                                                                          |
| 1. PERCEPTION LAYER (`Perception/`) [~15.6 - 30 Hz Camera, 10-20 Hz LiDAR]              |
|    - C3 YOLOv8 Detector: 12 Indian-specific classes on 1080p surround camera feeds       |
|    - SAHI Dual-Band Slicing Engine: Native 1:1 optical resolution for far targets (40-150m)|
|    - PatchWork++ LiDAR ground-plane curvature: Quantifies 3D pothole depth and volume    |
|                                                                                          |
| 2. SENSOR FUSION & TRACKING (`Sensor_fusion/`) [20 Hz Radar / 15.6 Hz Camera]            |
|    - Hungarian assignment & Mahalanobis gating (vision boxes + radar Doppler returns)   |
|    - 4-Model Semantic IMM Filter (Constant Velocity, CTRV, CA, Animal Freeze)            |
|    - World Model Output: `fused_tracks` in ISO 8855 Ego Body Frame                       |
|                                                                                          |
| 3. PREDICTION & TRAJECTORY LAYER (`Trajectory/`) [10 Hz / 100 ms]   <-- [BUILT & READY]  |
|    - MotionFormer BEV Cross-Attention (agents query static pothole tokens)               |
|    - Multi-modal GMM prediction (K=3 modes: Nominal, Evasive Swerve, Emergency Freeze)   |
|    - Dynamic Spatio-Temporal Costmap Slices (tau = 0.5s, 1.0s, 2.0s, 3.0s)               |
|    - Stateflow Supervisory Triggers (CRUISE, SLOW_DOWN, YIELD, STOP, REROUTE)            |
|                                                                                          |
| 4. PATH PLANNING & DECISION LOGIC (`Path_planning_decision/`) [5 Hz / 200 ms] <-- [NEXT] |
|    - Multi-Layer Dynamic Costmap Manager (`vehicleCostmap` from Navigation Toolbox)      |
|    - Stateflow Supervisory State Machine (mode transition logic & speed limiter)         |
|    - Bounded-Window Hybrid A* Local Planner (searches 35m local window in < 50ms)        |
|    - Path Smoother & Urgency-Scaled Replan Blender (limits curvature jerk)               |
|                                                                                          |
| 5. VEHICLE DYNAMICS & CONTROL (`Vehicle_dynamics/`) [100 Hz / 10 ms]                     |
|    - Speed-adaptive controller handover:                                                 |
|      * Adaptive Pure Pursuit (A-PP): Low speed, tight turns, village roads, market areas|
|      * Discrete Bicycle-Model MPC: Higher-speed tracking & highway merges                |
|    - Actuator bounds: Steering angle delta, throttle, and regenerative/friction brake    |
|    - Kinematic Bicycle Model / Vehicle Dynamics Blockset updates Ego state               |
+──────────────────────────────────────────────────────────────────────────────────────────+
               │ [Updated Ego Pose: X, Y, Z, Roll, Pitch, Yaw]
               ▼
+──────────────────────────────────────────────────────────────────────────────────────────+
| ROADRUNNER SCENARIO: Ego vehicle position updates in 3D world in real-time closed loop   |
+──────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Status of the Pipeline Layers

| Subsystem Layer | Status | Key Deliverables Present |
| :--- | :---: | :--- |
| **Track 1: Perception** | **COMPLETE** | `sahi_engine.py`, C3 YOLOv8 IDD detector, `sahi_visualizer_and_benchmark.m` |
| **Track 2: Sensor Fusion** | **COMPLETE** | `c3_vision_radar_fusion_bridge.m`, `c3_semantic_imm_tracker.m`, `topic4_sensor_fusion.m`, `Vehicle_dynamics/sensor_fusion_bridge.m` |
| **Track 3: Trajectory Prediction** | **COMPLETE** | `motionformer_engine.py`, `trajectory_prediction_engine.m`, `prediction_to_costmap_bridge.m`, `simulink_prediction_block.m`, `setup_trajectory_simulink.m`, `benchmark_5_scenarios.m`, 3x PPT slides |
| **Track 4: Path Planning & Decision** | **COMPLETE** | `dynamic_trajectory_planner.m` (Frenet quintic replanner), `dynamic_costmap_manager.m`, `hybrid_astar_planner.m`, `continuous_trajectory_optimizer.m`, `speed_profile_generator.m`, `decision_supervisor.m`, `path_smoother_blender.m`, `simulink_planning_block.m`, `test_planning_pipeline.py`, `test_dynamic_trajectory_planner.py` |
| **Track 5: Vehicle Dynamics & Control** | **COMPLETE** | `autonomous_ego_controller.m`, `pure_pursuit_controller.m`, `mpc_lane_controller.m`, `sensor_rig_builder.m`, `simulation_logger.m`, `record_unreal_simulation.m`, `sim3d_surround_harness.slx`, 11 OpenSCENARIO (.xosc) scenarios |
| **System Integration: Simulink + RoadRunner** | **COMPLETE** | `Simulink_Simulation/setup_simulink_simulation.m`, `Simulink_Simulation/build_closed_loop_model.m`, `Simulink_Simulation/run_closed_loop_simulation.m`, `Simulink_Simulation/test_simulink_pipeline.py`, closed-loop feedback wiring |

---

## 3. What Needs to Be Built Step-by-Step

### Phase 1: Build Path Planning & Decision Logic (`Path_planning_decision/`) [COMPLETED]

The planning layer consumes the costmaps and triggers from `Trajectory/` and generates smooth, collision-free paths.

- [x] **Step 1.1: Create `Path_planning_decision/plan.md`**
  - Document the technical architecture, coordinate transformations, search window constraints, cost functions, and latency budgets.

- [x] **Step 1.2: Build the Dynamic Multi-Layer Costmap Engine (`dynamic_costmap_manager.m`)**
  - Uses MATLAB Navigation Toolbox `vehicleCostmap`.
  - Layer 1 (Static Road): Ingests road boundaries and injects a soft virtual lane corridor ($0.05$ cost vs $0.20$ off-corridor) to maintain human-like behavior on unmarked roads.
  - Layer 2 (Surface Hazards): Converts LiDAR/Perception pothole tokens into graded costs ($<5\text{ cm}$ dip = $0.2\text{--}0.4$; $>5\text{ cm}$ cavity = $0.95$ solid wall).
  - Layer 3 (Dynamic Occupancy): Ingests multi-modal GMM prediction slices ($\tau = 0.5\text{s}, 1.0\text{s}, 2.0\text{s}, 3.0\text{s}$) from `prediction_to_costmap_bridge.m`.

- [x] **Step 1.3: Build the Stateflow Decision Supervisor (`decision_supervisor.m` / Stateflow Chart)**
  - Inputs: `min_TTC`, `critical_agent_id`, `is_path_blocked`, `ego_speed`.
  - Implement 5 behavioral states:
    * `CRUISE`: Normal driving ($40\text{ km/h}$).
    * `SLOW_DOWN`: Predicted actor crosses ego path; target speed reduced to $20\text{ km/h}$.
    * `YIELD / NEGOTIATE`: Unsignalled junction; closing speed and proximity logic.
    * `STOP`: Emergency halt ($0\text{ km/h}$) triggered when $\text{TTC} < 1.8\text{ s}$ (e.g. cattle sudden freeze).
    * `REROUTE`: Triggered when forward corridor cost $\ge 0.90$ across entire roadway width.
  - Timeout transitions (e.g. STOP held $> 5.0\text{ s}$ escalates to REROUTE).

- [x] **Step 1.4: Implement Bounded-Window Hybrid A\* Planner (`hybrid_astar_planner.m`)**
  - Explores non-holonomic bicycle kinematics ($R_{\min} = 4.55\text{ m}$, wheelbase $L = 2.8\text{ m}$, max steer $30^\circ$).
  - Bounded local window ($35\text{ m}$ ahead) guarantees sub-$25\text{ ms}$ search time.
  - Automatically identifies safe convex corridor $[x_{\min}, x_{\max}]$ for continuous optimization.

- [x] **Step 1.5: Implement Continuous QP Spline Trajectory Optimizer (`continuous_trajectory_optimizer.m`)**
  - Tesla-inspired banded QP optimizer minimizing curvature rate ($\mathbf{D}_2$) and lateral jerk ($\mathbf{D}_3$) within safe convex corridor.
  - Guaranteed $C^2$ continuity without discrete steering spikes.

- [x] **Step 1.6: Build Longitudinal ST Speed Profile Generator (`speed_profile_generator.m`)**
  - Curvature-aware speed governor: $v(s) \le \sqrt{a_{y,\max} / |\kappa(s)|}$.
  - Graded hazard speed limit: $v \le 15\text{ km/h}$ over shallow pothole dips.
  - Forward-backward passes enforcing acceleration and deceleration comfort limits.

- [x] **Step 1.7: Build Path Smoother & Replan Blender (`path_smoother_blender.m`)**
  - Urgency-scaled quintic $C^2$ polynomial trajectory stitcher eliminating steering kicks at $10\text{ Hz}$.

- [x] **Step 1.8: Build Simulink Embedded Code Block & Bus Setup (`simulink_planning_block.m`, `setup_planning_simulink.m`)**
  - Deterministic execution, zero dynamic memory allocation (`#codegen`), strongly typed Simulink Bus definitions.

- [x] **Step 1.9: Build Benchmarking Suite, Unit Tests & Presentation Graphics (`benchmark_planning_suite.m`, `test_planning_pipeline.py`, `planning_visualizer.py`)**
  - Evaluated across 5 Indian road scenarios ($24.4\text{ ms}$ latency, $100\%$ collision-free).
  - 3 high-resolution 16:9 presentation slides generated at 300 DPI.

---

### Phase 2: Build Vehicle Dynamics & Control (`Vehicle_dynamics/`) [COMPLETED]

The control layer turns planned paths and speed targets into steering, throttle, and brake demands.

- [x] **Step 2.1: Implement Kinematic Bicycle Model Subsystem (`autonomous_ego_controller.m`)**
  - State vector: $\mathbf{x} = [X, Y, \theta, v]^T$.
  - Equations of motion:
    $$\dot{X} = v \cos(\theta), \quad \dot{Y} = v \sin(\theta), \quad \dot{\theta} = \frac{v}{L} \tan(\delta), \quad \dot{v} = a$$
  - Vehicle parameters: Wheelbase $L = 2.8\text{ m}$, $l_f = 1.2\text{ m}$, $l_r = 1.6\text{ m}$, Max deceleration $a_{\min} = -7.5\text{ m/s}^2$, Max acceleration $a_{\max} = 1.6\text{ m/s}^2$.

- [x] **Step 2.2: Implement Adaptive Pure Pursuit (`pure_pursuit_controller.m`)**
  - Adaptive lookahead distance scaled by vehicle forward speed and cross-track error:
    $$L_d = \text{clamp}(k_v v + k_e |e_y|, L_{\min}, L_{\max})$$
  - Computes front wheel steering angle $\delta = \text{atan2}(2 L \sin(\alpha), L_d)$ with anti-chatter smoothing.

- [x] **Step 2.3: Implement Discrete Bicycle-Model Model Predictive Control (`mpc_lane_controller.m`)**
  - Discrete MPC optimizing steering angle rate and lateral error over a prediction horizon $N_p = 15$ steps.
  - Formulates QP minimizing tracking error, heading discrepancy, and steering effort subject to physical steering limits.

- [x] **Step 2.4: Implement Unified Closed-Loop Manager (`autonomous_ego_controller.m`)**
  - Ingests Perception & Sensor Fusion tracks, runs 10 Hz dynamic trajectory replanner, modulates longitudinal speed via kinematic ACC/AEB stopping distance law, and executes lateral path tracking.

- [x] **Step 2.5: Build Safety Brake Override & Multi-Sensor Harness (`sensor_fusion_bridge.m`, `sim3d_surround_harness.slx`)**
  - M-of-N persistence filter, two-tier braking authority gate, and Unreal Engine 3D co-simulation surround camera harness.

---

### Phase 3: Build the Integrated Closed-Loop Simulink Simulator (`Simulink_Simulation/`) [COMPLETED]

Create the master Simulink model connecting all five tracks into an executable simulation.

- [x] **Step 3.1: Create Master Initialization Script (`setup_simulink_simulation.m`)**
  - Loads all vehicle parameters, sensor extrinsics/intrinsics, Simulink Buses, and RoadRunner scenario waypoints into the MATLAB base workspace.
  - Sets up multi-rate solver (`ode4` Runge-Kutta, fixed step $0.01\text{ s}$).

- [x] **Step 3.2: Create Master Simulink Model (`SIH26037_ClosedLoop_EgoSimulator.slx`)**
  - Programmatically generated via `Simulink_Simulation/build_closed_loop_model.m`.
  - Subsystem 1: **RoadRunner Co-Simulation Interface** (or Synthetic ADT Scenario Engine).
  - Subsystem 2: **Perception & IPM Bridge** (Object bounding boxes, SAHI proposal merge, ground curvature).
  - Subsystem 3: **Sensor Fusion IMM Tracker** (Multi-object tracking with CV/CTRV/CA/Freeze models).
  - Subsystem 4: **Trajectory Prediction Block** (Integrates `simulink_prediction_block.m`).
  - Subsystem 5: **Stateflow Supervisory Chart** (`CRUISE`, `SLOW_DOWN`, `YIELD`, `STOP`, `REROUTE`).
  - Subsystem 6: **Dynamic Costmap & Hybrid A\* Local Planner** (Frenet optimal replanner).
  - Subsystem 7: **Adaptive Controller & Handover Subsystem** (A-PP + MPC + Kinematic ACC/AEB).
  - Subsystem 8: **Ego Vehicle Dynamics Subsystem** (Kinematic bicycle model updating $[X, Y, \theta, v]$).
  - Subsystem 9: **RoadRunner Feedback** (Sends updated ego pose back to RoadRunner in real-time closed loop).

- [x] **Step 3.3: Implement Rate Transition Blocks & Bus Selectors**
  - Connect $100\text{ Hz}$ Dynamics $\leftrightarrow$ $50\text{ Hz}$ Control $\leftrightarrow$ $20\text{ Hz}$ Fusion $\leftrightarrow$ $10\text{ Hz}$ Planning & Prediction.

- [x] **Step 3.4: Add Dashboard Instrumentation & Scopes**
  - Live trajectory visualization in `run_closed_loop_simulation.m`.
  - Scopes for tracking error, lateral jerk, steering demand, throttle/brake, and Time-To-Collision.
  - Verified via `Simulink_Simulation/test_simulink_pipeline.py`.

---

### Phase 4: Validate the Ego Vehicle across the 5 Official Scenarios

Test the closed loop against the 5 validation scenarios specified in Problem Statement 26037:

- [ ] **Scenario 1: Unmarked Village Road**
  - Challenge: Stray cattle wandering along shoulder, unpaved muddy road edge, shallow dips ($<5\text{ cm}$).
  - Validation Criteria: Ego vehicle stays in virtual corridor; slows for dip without stopping; avoids cow with $> 1.5\text{ m}$ clearance.

- [ ] **Scenario 2: Busy Urban Intersection without Signals**
  - Challenge: Auto-rickshaws nudging forward, crossing motorcycles, informal merging.
  - Validation Criteria: Predictor forecasts turn modes; Stateflow enters `YIELD`; vehicle negotiates gap without freezing.

- [ ] **Scenario 3: Highway Merge with Slow Vehicles**
  - Challenge: High-speed cruising ($60\text{ km/h}$) approaching a slow tractor or pushcart ($15\text{ km/h}$).
  - Validation Criteria: MPC controller executes smooth lane-change overtake; controller handover creates zero steering jerk.

- [ ] **Scenario 4: Dense Market Area with Mixed Traffic**
  - Challenge: High agent count ($> 10$), pedestrians darting out between parked vehicles, motorcycles gap-filling.
  - Validation Criteria: Sub-$50\text{ ms}$ replanning latency; dynamic costmap inflates pedestrian safety envelopes; vehicle cruises safely at $15\text{--}20\text{ km/h}$.

- [ ] **Scenario 5: Sudden Cattle-Crossing Freeze Event**
  - Challenge: Cow steps into ego lane and freezes dead ($v \to 0$ in $0.4\text{ s}$) inside stopping distance.
  - Validation Criteria: MotionFormer shifts to Zero-Velocity Freeze mode; Stateflow triggers emergency `STOP`; vehicle halts with $> 5.0\text{ m}$ safety margin without spinout.

---

### Phase 5: Automated Scored Benchmark Suite (`run_full_system_benchmark.m`)

Build the automated validation and grading harness scored by MathWorks:

- [ ] **Step 5.1: Replanning Latency Profiler**
  - Log timestamps at sensor input, prediction output, and planner output.
  - Measure and report: Mean latency, 95th-percentile latency, and peak worst-case latency (target: $< 100\text{ ms}$).

- [ ] **Step 5.2: Path Smoothness & Jerk Evaluator**
  - Measure curvature profile $\kappa(s)$ and lateral jerk $\dot{a}_y(t)$ (target: $|jerk| < 2.0\text{ m/s}^3$).

- [ ] **Step 5.3: Scenario Completion & Safety Margin Evaluator**
  - Measure completion rate ($100\%$), minimum Time-To-Collision ($\text{TTC}_{min} > 1.5\text{ s}$), and zero collisions across all 5 scenarios.

- [ ] **Step 5.4: Automatic Export of Figures and Benchmark Data**
  - Save summary results to `full_pipeline_benchmark_results.mat`.
  - Generate final presentation figures comparing all stages.

---

## 4. Immediate Next Step Checklist

To begin execution right now:
- [ ] Build **`Path_planning_decision/plan.md`** defining all planning equations and interfaces.
- [ ] Implement **`Path_planning_decision/dynamic_costmap_manager.m`** consuming `Trajectory/` outputs.
- [ ] Implement **`Path_planning_decision/hybrid_astar_planner.m`** with local window search.
- [ ] Implement **`Path_planning_decision/decision_supervisor.m`** (Stateflow supervisor).
- [ ] Run standalone planning benchmark and verify end-to-end handoff from Trajectory to Planning.
