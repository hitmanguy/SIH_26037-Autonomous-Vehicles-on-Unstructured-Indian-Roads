# Trajectory Prediction and Forecasting Plan
## Problem Statement ID: 26037 — Team Epsilon
### Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads

---

## 1. Executive Summary & Objective

The objective of this track is to build a robust, multi-modal, non-lane-conditioned trajectory prediction engine that anticipates the short-term future motion ($1.0$ to $5.0$ seconds) of heterogeneous road users in chaotic Indian driving environments. 

On unstructured Indian roads, standard autonomous vehicle trajectory predictors fail because they assume:
1. **Strict Lane Discipline:** Vehicles stick to predefined lane centerlines (Frenet frame). On Indian roads, lane markings are often absent, worn, or ignored.
2. **Homogeneous Dynamics:** Most models assume standard passenger cars and gentle lane changes. Indian roads feature aggressive three-wheelers (`autorickshaw`), agile two-wheelers (`motorcycle`, `bicycle`, `rider`), vulnerable pedestrians (`person`), slow pushcarts, and unpredictable animals (`animal` / stray cows).
3. **Smooth, Continuous Motion:** Indian road users execute high-frequency lateral swerves to fill micro-gaps and evade road surface hazards (deep potholes, speed breakers, debris, unpaved shoulders).
4. **Deterministic Behavior:** Traffic negotiation at unsignalled junctions occurs via nudging and closing speed rather than traffic signals. A deterministic single-path predictor suffers from **Freezing Robot Syndrome** or dangerous false emergency stops.

This track bridges **Sensor Fusion** (`Sensor_fusion/`) and **Path Planning & Decision Logic** (`Path_planning_decision/`), consuming fused kinematic tracks with static hazard tokens and producing multi-modal Gaussian Mixture Model (GMM) future occupancy distributions that feed directly into MATLAB's `vehicleCostmap` and the Stateflow supervisor.

---

## 2. Integrated Multi-Model Architecture

To provide maximum reliability, accuracy, and real-time execution in MATLAB and Simulink without fine-tuning on proprietary Indian datasets, this pipeline combines state-of-the-art pretrained architectures with a physics-informed semantic fallback:

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
|   - Native MATLAB / Simulink deterministic engine (zero external dependencies, < 5 ms latency).   |
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
For each detected agent $i \in \{1, \dots, M\}$, the predictor forecasts $K$ distinct trajectory modes over a time horizon $H = 30$ steps (corresponding to $3.0\text{ s}$ at $\Delta t = 0.1\text{ s}$):

$$\mathcal{P}_i = \sum_{k=1}^K \pi_i^{(k)} \mathcal{N}\left(\boldsymbol{\mu}_i^{(k)}(t), \boldsymbol{\Sigma}_i^{(k)}(t)\right), \quad \sum_{k=1}^K \pi_i^{(k)} = 1$$

Where:
- $\pi_i^{(k)} \in [0, 1]$ is the categorical probability of mode $k$.
- $\boldsymbol{\mu}_i^{(k)}(t) = \begin{bmatrix} x_i^{(k)}(t) \\ y_i^{(k)}(t) \end{bmatrix}$ is the predicted 2D position at future time step $t$.
- $\boldsymbol{\Sigma}_i^{(k)}(t) = \begin{bmatrix} \sigma_{x}^2 & \rho \sigma_x \sigma_y \\ \rho \sigma_x \sigma_y & \sigma_y^2 \end{bmatrix}$ is the $2 \times 2$ spatial covariance matrix representing the physical uncertainty ellipse.

### 3.2 Causal Hazard Interaction (Pothole-Induced Swerve)
When an agent $i$ with heading $\psi_i$ and velocity $v_i$ approaches a detected road surface depression (pothole $j$ at $[x_p, y_p]$ with depth $d_p > 5\text{ cm}$):
1. Proximity horizon: $\tau_{hazard} = \frac{\|(x_i, y_i) - (x_p, y_p)\|}{v_i}$
2. If $\tau_{hazard} < \tau_{anticipate}$ and lateral distance $|\Delta y_{rel}| < \frac{w_{agent} + w_{pothole}}{2}$:
   - The Cross-Attention layer shifts probability mass $\Delta \pi$ from the straight/nominal mode to the evasive swerve modes ($k_{swerve\_left}$ and $k_{swerve\_right}$).
   - The lateral amplitude of the swerve is bounded by the road boundary constraints:
     $$\Delta y_{swerve} = \text{sign}(\Delta y_{boundary}) \cdot \min\left(w_{pothole} + 0.5\text{ m}, d_{boundary}\right)$$

### 3.3 Dynamic Costmap Inflation
The GMM uncertainty ellipse at future time slice $\tau$ is mapped onto the grid $[x_g, y_g]$ with cost:

$$C(x_g, y_g; \tau) = \sum_{i=1}^M \sum_{k=1}^K \pi_i^{(k)} \cdot \exp\left(-\frac{1}{2} \left(\mathbf{p}_g - \boldsymbol{\mu}_i^{(k)}(\tau)\right)^T \boldsymbol{\Sigma}_i^{(k)}(\tau)^{-1} \left(\mathbf{p}_g - \boldsymbol{\mu}_i^{(k)}(\tau)\right)\right)$$

Points exceeding a critical collision threshold ($\chi^2 \le 5.991$, $95\%$ confidence ellipse) are marked as hard obstacle inflations, while outer bands serve as graded soft costs discouraging near-miss trajectories.

---

## 4. Five Target Validation Scenarios

1. **Unmarked Village Road:** Stray cattle wandering across unpaved boundaries; predictor must maintain a wide multi-modal uncertainty ellipse until the animal commits to a direction or freezes.
2. **Busy Unsignalled Urban Intersection:** Multi-agent negotiation where auto-rickshaws nudge forward; predictor outputs turning and yielding modes with dynamic TTC computation.
3. **Highway Merge with Slow Vehicles:** High closing velocity scenario; predictor accurately forecasts merging angles of slow tractors and carts into 60 km/h traffic.
4. **Dense Market Area:** High agent count ($M > 10$), micro-gap filling by two-wheelers, and sudden lateral darting of pedestrians.
5. **Sudden Cattle Crossing Freeze Event:** Cow steps into ego lane and stops dead (mode probability of Zero-Velocity Freeze jumps to $> 0.85$); Stateflow immediately commands emergency `STOP`.

---

## 5. Deliverables & Directory Structure

```
Trajectory/
├── README.md                           # Comprehensive track documentation & API reference
├── plan.md                             # Architectural specification & roadmap (this file)
├── motionformer_engine.py              # PyTorch MotionFormer + GMM Neural Network + ONNX exporter
├── download_pretrained_weights.py       # Automated downloader for open-source model weights
├── trajectory_prediction_engine.m      # Native MATLAB prediction engine (DL ONNX + Semantic IMM-GMM)
├── prediction_to_costmap_bridge.m      # Downstream costmap & Stateflow supervisory bridge
├── simulink_prediction_block.m         # Simulink MATLAB Function Block implementation
├── setup_trajectory_simulink.m         # Simulink Bus configuration & simulation setup
├── test_prediction_pipeline.py         # Python unit & verification tests
├── benchmark_5_scenarios.m             # MATLAB validation suite across the 5 Indian scenarios
└── benchmark_visualizer.py             # Publication figure & .mat results generator
```
