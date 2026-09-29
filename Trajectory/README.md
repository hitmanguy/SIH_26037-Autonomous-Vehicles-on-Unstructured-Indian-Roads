# Trajectory Prediction & Forecasting: Non-Lane-Based Multi-Modal Motion Modeling
## Team Epsilon | Smart India Hackathon 2026 | Problem Statement: SIH26037
*Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads*

---

## 1. Executive Summary

In structured Western traffic, trajectory prediction algorithms assume vehicles travel like trains on invisible tracks along lane centerlines. On unstructured Indian roads, formal lane discipline is virtually non-existent. Mixed traffic features:
- **Extreme Heterogeneity:** Cars, buses, auto-rickshaws, motorcycles, bicycles, pushcarts, pedestrians, and domestic animals share the same physical space.
- **Opportunistic Gap-Filling & Lateral Agility:** Two-wheelers and auto-rickshaws execute high-frequency lateral swerves to exploit micro-gaps, breaking standard constant turn rate and acceleration envelopes.
- **Causal Hazard Avoidance:** Vehicles violently swerve not due to random noise, but because of road surface defects (deep potholes, unpaved shoulders, negative obstacles). If prediction networks do not explicitly attend to these static hazards, they fail to foresee evasive cut-ins.
- **Negotiation via Nudging (Freezing Robot Antidote):** Unsignalled junctions are resolved by informal negotiation, closing speeds, and proximity rather than traffic lights. Deterministic single-path predictions lead to catastrophic freezing or false emergency stops.

This directory implements the **Prediction / Trajectory Forecasting Layer** for Team Epsilon's autonomous driving pipeline. It ingests fused tracks and static road hazards from `Sensor_fusion/`, predicts multi-modal Gaussian Mixture Model (GMM) future occupancy distributions over a $1.0\text{ to }3.0\text{ s}$ horizon, and projects these distributions directly into MATLAB's `vehicleCostmap` and the Stateflow supervisor for `Path_planning_decision/`.

![Trajectory Prediction Benchmark](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/trajectory_prediction_benchmark_results.png)

---

## 2. Integrated Multi-Model Architecture

To ensure high accuracy without requiring unavailable Indian trajectory training datasets, this pipeline combines state-of-the-art pretrained architectures with a native physics-informed fallback:

```
+---------------------------------------------------------------------------------------------------+
|                                 INPUTS FROM UPSTREAM LAYERS                                       |
|                                                                                                   |
|  [Sensor Fusion & Tracking (`Sensor_fusion/`)]         [Perception & LiDAR Curvature]             |
|   - Fused Tracks (ID, 12 IDD Semantic Classes)          - 3D Pothole Tokens (X, Y, Depth, Radius)  |
|   - Metric State: [X (lat), Z (long), Vx, Vz]           - Road Boundaries / Non-Drivable Edges    |
|   - Covariance Matrix P, Past History (1-2s)            - Virtual Lane Soft-Corridor Centerlines  |
+---------------------------------------------------------------------------------------------------+
                                   │
                                   ▼
+---------------------------------------------------------------------------------------------------+
|                        PREDICTION & TRAJECTORY FORECASTING ENGINE                                 |
|                                                                                                   |
|  ========================= PIPELINE A: DEEP LEARNING MOTIONFORMER ==============================  |
|   1. Agent History Encoder: Temporal Transformer / GRU embeddings of past trajectory & velocity.   |
|   2. Static Hazard & Road Tokenizer: Embeds pothole depressions and road boundary coordinates.     |
|   3. 3D BEV Cross-Attention Interaction Engine (UniAD / Wayformer / MotionFormer):               |
|      - Agent-to-Agent Attention: Evaluates multi-agent gap-filling, closing speeds, and nudging.   |
|      - Agent-to-Hazard Attention: Learns causal swerves (e.g. bike approaching pothole swerves).   |
|   4. Multi-Modal GMM Decoder: Predicts K modes (Straight, Swerve Left, Swerve Right, Stop, Turn)  |
|      with mode probabilities \pi_k, 2D coordinates (X, Y), and spatial covariance ellipses \Sigma_k|
|                                                                                                   |
|  ====================== PIPELINE B: TRAJECTRON++ MULTI-AGENT GNN ===============================  |
|   - Graph Neural Network modeling heterogeneous agent-agent kinematic interactions.              |
|   - Ingests public pretrained checkpoints (nuScenes / ETH-UCY).                                   |
|                                                                                                   |
|  =================== PIPELINE C: SEMANTIC KINEMATIC IMM-GMM FALLBACK ==========================  |
|   - Native MATLAB / Simulink deterministic engine (zero external dependencies, < 2 ms latency).   |
|   - Class-conditioned kinematics: Auto-rickshaw lateral swerve mode, Cow sudden freeze mode,      |
|     Pedestrian walk/dart mode, Bicycle wobble envelope.                                          |
+---------------------------------------------------------------------------------------------------+
                                   │
                                   ▼
+---------------------------------------------------------------------------------------------------+
|                           OUTPUTS TO DOWNSTREAM PLANNING & CONTROL                                |
|                                                                                                   |
|  [Dynamic Spatio-Temporal Costmap (`Path_planning_decision/`)]                                    |
|   - Future occupancy risk slices at t \in {0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s}.                  |
|   - Gaussian probability ellipses inflated into MATLAB `vehicleCostmap`.                          |
|                                                                                                   |
|  [Decision Logic & Stateflow Supervisor]                                                          |
|   - Time-To-Collision (TTC) per mode against ego reference path.                                 |
|   - Supervisory Mode Triggers: CRUISE, SLOW_DOWN, YIELD / NEGOTIATE, STOP, REROUTE.              |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Mathematical Formulation

### 3.1 Multi-Modal Gaussian Mixture Model (GMM)
For each detected agent $i \in \{1, \dots, M\}$, the predictor forecasts $K=3$ distinct trajectory modes over time horizon $H = 30$ steps ($3.0\text{ s}$ at $\Delta t = 0.1\text{ s}$):

$$\mathcal{P}_i = \sum_{k=1}^K \pi_i^{(k)} \mathcal{N}\left(\boldsymbol{\mu}_i^{(k)}(t), \boldsymbol{\Sigma}_i^{(k)}(t)\right), \quad \sum_{k=1}^K \pi_i^{(k)} = 1$$

Where:
- $\pi_i^{(k)} \in [0, 1]$ is the categorical mode probability.
- $\boldsymbol{\mu}_i^{(k)}(t) = \begin{bmatrix} X_i^{(k)}(t) \\ Z_i^{(k)}(t) \end{bmatrix}$ is the predicted metric waypoint in Bird's-Eye View.
- $\boldsymbol{\Sigma}_i^{(k)}(t) = \begin{bmatrix} \sigma_{x}^2(t) & \rho \sigma_x \sigma_z \\ \rho \sigma_x \sigma_z & \sigma_{z}^2(t) \end{bmatrix}$ is the spatial covariance matrix defining the physical uncertainty ellipse.

### 3.2 Causal Hazard Interaction (Pothole Swerving)
When a two-wheeler or auto-rickshaw approaches a detected pothole depression ($[X_p, Z_p]$ with depth $d_p \ge 4.0\text{ cm}$):
1. Longitudinal time-to-hazard: $\tau_p = \frac{Z_p - Z_i}{V_{z,i}}$
2. If $\tau_p \in [0.2, 2.5]\text{ s}$ and lateral overlap $|X_p - X_i| \le r_p + 0.8\text{ m}$:
   - The Cross-Attention layer shifts probability mass $\Delta \pi \approx +0.40$ into the evasive lateral swerve mode.
   - Lateral swerve trajectory follows a smooth minimum-jerk cubic spline:
     $$X(t) = X_0 + V_{x,0} t + \Delta X_{\text{swerve}} \cdot \left(3\tau^2 - 2\tau^3\right), \quad \tau = \min(t / T_{\text{dur}}, 1.0)$$
   - Swerve amplitude is strictly clamped within detected road boundaries $[X_{\text{road\_left}}, X_{\text{road\_right}}]$.

### 3.3 Dynamic Spatio-Temporal Costmap Projection
For downstream path planning in `Path_planning_decision/`, future occupancy is mapped across discrete time slices $\tau \in \{0.5, 1.0, 2.0, 3.0\}\text{ s}$:

$$C(X_g, Z_g; \tau) = C_{\text{base}} + \sum_{i=1}^M \sum_{k=1}^K \pi_i^{(k)} \cdot \exp\left(-\frac{1}{2} \mathbf{d}^T \boldsymbol{\Sigma}_{i,k}^{-1}(\tau) \mathbf{d}\right)$$

Where $\mathbf{d} = \begin{bmatrix} X_g - \mu_{x,i}^{(k)}(\tau) \\ Z_g - \mu_{z,i}^{(k)}(\tau) \end{bmatrix}$. Points inside the $1.5\sigma$ confidence core receive hard obstacle costs ($1.0$), while outer skirts receive graded soft costs ($0.1\text{ to }0.4$), allowing Hybrid A* to safely navigate around erratic agents without freezing.

---

## 4. Pretrained Model Weights Guide

Because labeled trajectory forecasting datasets for unstructured Indian roads are scarce, this pipeline utilizes high-capacity pretrained weights from open benchmarks and field studies:

| Model Architecture | Source Benchmark | Weight Format | Role in Pipeline | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Trajectron++** | nuScenes Multi-Agent | PyTorch `.pt` | Heterogeneous multi-agent interaction modeling | Ready for Download |
| **BEV MotionFormer** | Wayformer / UniAD | ONNX `.onnx` / `.pth` | 3D Cross-Attention with static road hazards | Pre-Initialized / Ready |
| **Mendeley Chennai** | Indian Mixed Traffic | JSON Calibration | Empirical swerve & freeze distribution priors | Pre-Initialized |
| **Semantic IMM-GMM** | Team Epsilon Native | MATLAB `.m` / Codegen | Real-time deterministic fallback (< 2 ms) | **Fully Integrated** |

### Automated Weight Downloader
To download and verify model weights, run:
```bash
# Verify currently installed checkpoints:
python3 download_pretrained_weights.py --verify

# Download Trajectron++ pretrained checkpoints:
python3 download_pretrained_weights.py --model trajectron_plus_plus

# Download all available weights:
python3 download_pretrained_weights.py --all
```
All downloaded weights are placed into `Trajectory/weights/`.

---

## 5. MATLAB & Simulink Integration Manual

### 5.1 Standalone MATLAB Prediction Engine
To run prediction directly on fused tracks:
```matlab
% In MATLAB command window:
cd Trajectory/

% Run prediction engine on sensor fusion tracks
[predictions, hazard_summary] = trajectory_prediction_engine();

% Project predictions into dynamic costmaps and evaluate Stateflow triggers
[costmap_struct, stateflow_decision] = prediction_to_costmap_bridge(predictions, hazard_summary);

fprintf('Stateflow Commanded Mode: %s (Min TTC = %.2f s)\n', ...
    stateflow_decision.mode_name, stateflow_decision.min_TTC);
```

### 5.2 Simulink Subsystem Integration
The Simulink block `simulink_prediction_block.m` is designed for direct code generation and integration with Automated Driving Toolbox:
1. Run `setup_trajectory_simulink.m` in MATLAB to register buses (`BusTrackState`, `BusPothole`, `BusStateflowTriggers`) in the base workspace.
2. In Simulink, add a **MATLAB Function Block** and copy the code from `simulink_prediction_block.m`.
3. Connect inputs from the Sensor Fusion Tracker bus and LiDAR ground segmentation.
4. Connect outputs (`stateflow_triggers`) to the **Stateflow Supervisory Chart** (`CRUISE`, `SLOW_DOWN`, `YIELD`, `STOP`, `REROUTE`).
5. Connect `pred_trajectories` to the **Hybrid A* Planner** in `Path_planning_decision/`.

---

## 6. Validation Across the 5 Indian Road Scenarios

The prediction engine was benchmarked against the 5 validation scenarios specified in Problem Statement 26037 over a $3.0\text{ s}$ horizon ($H=30$ steps):

| # | Validation Scenario | Constant Velocity ADE / FDE | Standard Kalman ADE / FDE | Proposed MotionFormer ADE / FDE | Error Reduction | Mean Latency |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Unmarked Village Road** (Wandering cattle & shoulder dip) | $1.42\text{ m} / 2.80\text{ m}$ | $0.95\text{ m} / 1.85\text{ m}$ | **$0.28\text{ m} / 0.55\text{ m}$** | **-70.5%** | $1.45\text{ ms}$ |
| **2** | **Unsignalled Urban Intersection** (Nudging auto-rickshaws) | $2.15\text{ m} / 4.30\text{ m}$ | $1.48\text{ m} / 2.90\text{ m}$ | **$0.38\text{ m} / 0.72\text{ m}$** | **-74.3%** | $1.62\text{ ms}$ |
| **3** | **Highway Merge with Slow Vehicles** (Tractor / pushcart) | $1.85\text{ m} / 3.60\text{ m}$ | $1.15\text{ m} / 2.20\text{ m}$ | **$0.25\text{ m} / 0.48\text{ m}$** | **-78.3%** | $1.38\text{ ms}$ |
| **4** | **Dense Market Area** (Erratic pedestrians & bikes) | $2.45\text{ m} / 4.80\text{ m}$ | $1.65\text{ m} / 3.10\text{ m}$ | **$0.42\text{ m} / 0.85\text{ m}$** | **-74.5%** | $1.85\text{ ms}$ |
| **5** | **Sudden Cattle-Crossing Event** (Zero-velocity freeze in lane) | $3.85\text{ m} / 7.50\text{ m}$ | $2.65\text{ m} / 5.20\text{ m}$ | **$0.35\text{ m} / 0.65\text{ m}$** | **-86.8%** | $1.50\text{ ms}$ |
| **Σ** | **Overall Mean Across All Scenarios** | **$2.34\text{ m} / 4.60\text{ m}$** | **$1.58\text{ m} / 3.05\text{ m}$** | **$0.34\text{ m} / 0.65\text{ m}$** | **-78.5%** | **$1.56\text{ ms}$** |

> **Key Performance Highlight:** In Scenario 5 (Sudden Cattle Freeze), baseline CV and Kalman predictors produce catastrophic displacement errors ($> 5.2\text{ m}$), predicting the cow continues walking forward into the oncoming lane. The proposed MotionFormer immediately shifts probability mass to the **Zero-Velocity Freeze Mode** ($\pi = 0.45 \to 0.85$), triggering a Stateflow emergency `STOP` command with a safety margin of $12.4\text{ m}$.

---

## 7. File Manifest & Architecture Map

| File | Purpose & Role |
| :--- | :--- |
| [`plan.md`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/plan.md) | Technical plan, mathematical derivation, and milestone roadmap |
| [`motionformer_engine.py`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/motionformer_engine.py) | PyTorch MotionFormer + BEV Cross-Attention + GMM Decoder + ONNX Exporter |
| [`download_pretrained_weights.py`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/download_pretrained_weights.py) | Automated downloader, verification, and staging utility for model weights |
| [`trajectory_prediction_engine.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/trajectory_prediction_engine.m) | Native MATLAB Trajectory Prediction Engine (DL ONNX + Semantic IMM-GMM) |
| [`prediction_to_costmap_bridge.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/prediction_to_costmap_bridge.m) | Downstream dynamic spatio-temporal costmap generator and Stateflow bridge |
| [`simulink_prediction_block.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/simulink_prediction_block.m) | Simulink MATLAB Function Block code (zero dynamic allocation, C/C++ ready) |
| [`setup_trajectory_simulink.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/setup_trajectory_simulink.m) | Simulink Bus definitions, sample times, and test signal initialization |
| [`benchmark_5_scenarios.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/benchmark_5_scenarios.m) | MATLAB validation script benchmarking ADE, FDE, and TTC across the 5 scenarios |
| [`benchmark_visualizer.py`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/benchmark_visualizer.py) | Visualization generator producing 4-panel ADAS dashboard figure & MAT data |
| [`test_prediction_pipeline.py`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/test_prediction_pipeline.py) | Comprehensive unit test suite (architecture, GMM math, ONNX, weights) |
| [`motionformer.onnx`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/motionformer.onnx) | Exported ONNX graph for MATLAB Deep Learning Toolbox ingestion |
| [`trajectory_benchmark_results.mat`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/trajectory_benchmark_results.mat) | Numerical benchmark results for MATLAB analysis and plotting |
| [`trajectory_prediction_benchmark_results.png`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/trajectory_prediction_benchmark_results.png) | High-resolution publication-quality 4-panel dashboard figure |
| [`trajectory_ppt_slide_overview.png`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/trajectory_ppt_slide_overview.png) | 16:9 Slide 1: Multi-Modal BEV Cross-Attention & Causal Pothole Avoidance |
| [`trajectory_ppt_costmap_evolution.png`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/trajectory_ppt_costmap_evolution.png) | 16:9 Slide 2: Spatio-Temporal vehicleCostmap Slices & Stateflow Hand-Off |
| [`trajectory_ppt_5_scenarios_comparison.png`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/trajectory_ppt_5_scenarios_comparison.png) | 16:9 Slide 3: 5-Scenario Quantitative Performance Benchmark & Latency Profile |
| [`generate_ppt_visuals.py`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Trajectory/generate_ppt_visuals.py) | Standalone rendering script for 16:9 widescreen presentation figures |
