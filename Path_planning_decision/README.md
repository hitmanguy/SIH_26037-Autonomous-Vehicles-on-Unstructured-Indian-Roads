# Path Planning & Decision Logic Subsystem
## Smart India Hackathon (SIH) 2026 — Problem Statement ID: 26037
### Team Epsilon: Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads

---

## 1. Executive Summary & Algorithmic Architecture

On unstructured Indian roads, classical Western autonomous vehicle planning architectures fail because they assume:
1. **Clearly painted lane markings:** Unstructured Indian roads lack formal lane markings, requiring continuous planning over a **free-space occupancy grid** with a **soft virtual lane corridor bias** ($0.05$ cost vs $0.20$ off-corridor) to avoid getting trapped behind static or slow-moving obstacles.
2. **Binary obstacle classification:** Surface anomalies such as potholes cannot be treated as binary obstacles. Shallow depressions ($<5\text{ cm}$) must be negotiated at reduced crawl speed ($v \le 15\text{ km/h}$, cost $0.35$), whereas deep cavities ($>5\text{ cm}$) must be treated as solid walls (cost $0.95$).
3. **Structured right-of-way rules:** Unsignalled junctions require dynamic negotiation based on closing velocity and subtle nudging rather than rigid stopping.
4. **Sub-second reaction budgets:** Sudden cut-ins by auto-rickshaws or stray cattle require deterministic replanning in **under $35\text{ ms}$**.

To solve these challenges with mathematical rigor and real-time reliability, this subsystem adopts a **Tesla-inspired 2-stage hierarchical planning architecture** combined with a **decoupled spatial-temporal (SL / ST) graph optimizer** and a **Stateflow tactical decision supervisor**.

```
+───────────────────────────────────────────────────────────────────────────────────────────────────────────+
|                                        UPSTREAM INPUT DATA STREAMS                                        |
|  [Track 3: Trajectory Prediction (`Trajectory/`)]            [Track 1 & 2: Perception & Sensor Fusion]    |
|   - Dynamic Costmap Slices (tau = 0.5s, 1.0s, 2.0s, 3.0s)     - Road Boundaries & Unpaved Shoulders        |
|   - Multi-Modal GMM Uncertainty Ellipses                      - 3D Pothole Depressions (X, Z, Depth, Rad)  |
|   - Stateflow Triggers (min_TTC, critical_agent_id)           - Fused Tracks (Auto, Cow, Bike, Pedestrian) |
+───────────────────────────────────────────────────────────────────────────────────────────────────────────+
                                                      │
                                                      ▼
+───────────────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    PATH PLANNING & DECISION SUBSYSTEM                                     |
|                                                                                                           |
|  [1. Dynamic Multi-Layer Costmap Manager] (`dynamic_costmap_manager.m`)                                   |
|   - Layer 1 (Static Road): Hard boundary clamps + Soft Virtual Lane Corridor (-1.8m center, cost=0.05)    |
|   - Layer 2 (Surface Hazards): Graded Potholes (depth <5cm -> cost 0.35; depth >=5cm -> cost 0.95 wall)   |
|   - Layer 3 (Dynamic Occupancy): Spatio-temporal GMM ellipses from upstream Trajectory layer               |
|                                                     │                                                     |
|                                                     ▼                                                     |
|  [2. Stateflow Decision Supervisor] (`decision_supervisor.m`)                                             |
|   - Finite State Machine: CRUISE (40 km/h) | SLOW_DOWN (20 km/h) | YIELD (10 km/h) | STOP | REROUTE        |
|   - Debounce counters prevent high-frequency chattering under radar/camera track noise                   |
|   - 5-second STOP timeout with road blockage automatically escalates to REROUTE                           |
|                                                     │                                                     |
|                                                     ▼                                                     |
|  [3. Stage 1: Coarse Corridor Search] (`hybrid_astar_planner.m`)                                          |
|   - Explores SE(2) non-holonomic bicycle kinematics (wheelbase L=2.8m, max steer 30 deg)                  |
|   - Bounded 35m moving horizon guarantees deterministic exploration in < 25 ms                            |
|   - Selects global homotopy class (e.g. bypass left vs right) and outputs safe convex corridor [Xmin, Xmax]|
|                                                     │                                                     |
|                                                     ▼                                                     |
|  [4. Stage 2: Continuous QP Spline Optimizer] (`continuous_trajectory_optimizer.m`)                      |
|   - Solves convex Quadratic Program minimizing curvature rate (D2) and lateral jerk (D3)                 |
|   - Guaranteed C^2 continuity (strictly zero steering wheel angle discontinuities)                        |
|   - Clamped within convex corridor: X_min(s) <= X_opt(s) <= X_max(s)                                      |
|                                                     │                                                     |
|                                                     ▼                                                     |
|  [5. ST Speed Profile Generator] (`speed_profile_generator.m`)                                            |
|   - Pointwise curvature speed limit: v_kappa(s) <= sqrt(a_y_max / |kappa(s)|)                             |
|   - Graded hazard speed limit: v_hazard(s) <= 15 km/h over shallow pothole dips                           |
|   - Forward-backward velocity integration satisfying acceleration (a_max) and comfort decel (a_dec)      |
|                                                     │                                                     |
|                                                     ▼                                                     |
|  [6. Urgency-Scaled Replan Blender] (`path_smoother_blender.m`)                                           |
|   - Smoothly stitches new replan trajectory with previously tracked path using quintic C^2 polynomial    |
|   - Blending length adapts dynamically: L_blend = 6.0m (nominal cruising) down to 1.5m (urgent swerve)    |
+───────────────────────────────────────────────────────────────────────────────────────────────────────────+
                                                      │
                                                      ▼
+───────────────────────────────────────────────────────────────────────────────────────────────────────────+
|                                OUTPUT TO VEHICLE DYNAMICS & CONTROL                                       |
|  - Reference Trajectory Vector: [X_ref(s), Z_ref(s), theta_ref(s), kappa_ref(s), v_ref(s), a_ref(s)]       |
|  - Control Target Rate: 100 Hz (Pure Pursuit / Discrete Bicycle MPC tracking)                             |
+───────────────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Mathematical Formulations

### 2.1 Bounded-Window Non-Holonomic Bicycle Model (Hybrid A*)
The vehicle state in the local body coordinate frame is $\mathbf{x} = [x, z, \theta]^T \in \text{SE}(2)$.
Motion primitives satisfy the non-holonomic bicycle model:
$$\dot{x} = v \sin \theta, \quad \dot{z} = v \cos \theta, \quad \dot{\theta} = \frac{v}{L} \tan \delta, \quad |\delta| \le \delta_{\max}$$
Cost function for node expansion:
$$f(n) = g(n) + h(n)$$
Where:
- $g(n) = \sum_{k=1}^n \left( \Delta s_k + w_\delta |\delta_k| + w_{\Delta \delta} |\delta_k - \delta_{k-1}| + w_{\text{cost}} C(x_k, z_k) \right)$
- $h(n) = \max\left( h_{\text{Reeds-Shepp}}(n), h_{\text{Euclidean}}(n) \right)$ provides an admissible, non-holonomic lower bound.

### 2.2 Continuous Convex Trajectory Optimization (Tesla-Style QP)
Given coarse discrete waypoints $\mathbf{x}_{\text{coarse}} \in \mathbb{R}^N$ from Hybrid A* and convex lateral corridor bounds $[x_{\min,k}, x_{\max,k}]$, the continuous path $\mathbf{x} = [x_1, \dots, x_N]^T$ is solved via Quadratic Programming:

$$\min_{\mathbf{x}} \quad w_{\text{smooth}} \|\mathbf{D}_2 \mathbf{x}\|^2 + w_{\text{jerk}} \|\mathbf{D}_3 \mathbf{x}\|^2 + w_{\text{ref}} \|\mathbf{x} - \mathbf{x}_{\text{coarse}}\|^2$$

Where:
- $\mathbf{D}_2 \in \mathbb{R}^{(N-2) \times N}$ is the second-order central difference matrix approximating curvature:
  $$\mathbf{D}_2 \mathbf{x} \approx x_{k+1} - 2x_k + x_{k-1}$$
- $\mathbf{D}_3 \in \mathbb{R}^{(N-3) \times N}$ is the third-order central difference matrix approximating lateral jerk:
  $$\mathbf{D}_3 \mathbf{x} \approx x_{k+2} - 3x_{k+1} + 3x_k - x_{k-1}$$

Subject to:
1. **Convex corridor bounds:** $x_{\min,k} \le x_k \le x_{\max,k}, \quad \forall k \in \{1, \dots, N\}$
2. **Initial state continuity:** $x_1 = x_{\text{ego}}, \quad x_2 - x_1 = \Delta s_1 \sin \theta_{\text{ego}}$
3. **Curvature constraint:** $|\kappa_k| \le \kappa_{\max} = 0.22\text{ m}^{-1} \implies R_{\min} \ge 4.55\text{ m}$

### 2.3 Longitudinal ST Speed Profile Optimization
Target speed is governed pointwise by lateral acceleration comfort ($a_{y,\max} \le 2.2\text{ m/s}^2$):
$$v_{\kappa}(s) = \sqrt{\frac{a_{y,\max}}{\max(|\kappa(s)|, 10^{-4})}}$$
The composite velocity upper envelope is:
$$v_{\text{upper}}(s) = \min\left( v_{\text{Stateflow}}, v_{\kappa}(s), v_{\text{hazard}}(s) \right)$$
Where $v_{\text{hazard}}(s) \le 15\text{ km/h}$ over shallow pothole dips and $0\text{ km/h}$ at emergency stop targets.

**Forward-Backward Velocity Integration:**
- **Backward pass** enforces deceleration limit ($a_{\text{dec}} \in [-2.5, -5.0]\text{ m/s}^2$):
  $$v_k^2 \le v_{k+1}^2 + 2 |a_{\text{dec}}| \Delta s_k$$
- **Forward pass** enforces acceleration limit ($a_{\max} = 2.0\text{ m/s}^2$):
  $$v_{k+1}^2 \le v_k^2 + 2 a_{\max} \Delta s_k$$

### 2.4 Replan Blender: C² Quintic Transition Window
To prevent steering torque spikes when replanning at 10 Hz, the newly replanned path $\mathbf{P}_{\text{new}}(s)$ is blended with the previously tracked path $\mathbf{P}_{\text{prev}}(s)$ over a dynamic blending length $L_{\text{blend}}$:

$$u = \text{clamp}\left(\frac{s}{L_{\text{blend}}}, 0, 1\right), \quad w(u) = 10 u^3 - 15 u^4 + 6 u^5$$
$$\mathbf{P}_{\text{blended}}(s) = (1 - w(u)) \mathbf{P}_{\text{prev}}(s) + w(u) \mathbf{P}_{\text{new}}(s)$$

Satisfying exact $C^2$ continuity:
$$w(0) = 0, \quad w'(0) = 0, \quad w''(0) = 0 \quad \text{(Smooth departure from tracked path)}$$
$$w(1) = 1, \quad w'(1) = 0, \quad w''(1) = 0 \quad \text{(Smooth merger into new optimal path)}$$

The transition horizon adapts dynamically based on supervisory urgency:
$$L_{\text{blend}} = L_{\text{nom}} (1 - \text{urgency}) + L_{\text{urg}} \times \text{urgency}$$
- Nominal cruising ($\text{urgency} = 0.0$): $L_{\text{blend}} = 6.0\text{ m}$ (silky smooth transition)
- Emergency swerve ($\text{urgency} = 1.0$): $L_{\text{blend}} = 1.8\text{ m}$ (rapid evasive reaction)

---

## 3. Subsystem File Manifest

| File | Type | Description |
| :--- | :---: | :--- |
| [`dynamic_costmap_manager.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/dynamic_costmap_manager.m) | MATLAB Class | 3-Layer `vehicleCostmap` fusion (Virtual corridor, graded potholes, dynamic ellipses) |
| [`hybrid_astar_planner.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/hybrid_astar_planner.m) | MATLAB Class | Bounded-window ($35\text{m}$) Hybrid A* coarse corridor planner ($< 25\text{ ms}$) |
| [`continuous_trajectory_optimizer.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/continuous_trajectory_optimizer.m) | MATLAB Class | Tesla-style banded QP trajectory smoother with $C^2$ continuity |
| [`speed_profile_generator.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/speed_profile_generator.m) | MATLAB Class | ST-domain velocity governor with curvature, pothole, and stopping limits |
| [`decision_supervisor.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/decision_supervisor.m) | MATLAB Class | Stateflow supervisory FSM (CRUISE, SLOW_DOWN, YIELD, STOP, REROUTE) |
| [`path_smoother_blender.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/path_smoother_blender.m) | MATLAB Class | Urgency-scaled quintic $C^2$ trajectory stitcher eliminating steering kicks |
| [`simulink_planning_block.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/simulink_planning_block.m) | MATLAB Function | Zero-dynamic-allocation Simulink block ready for C/C++ embedded code generation |
| [`setup_planning_simulink.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/setup_planning_simulink.m) | MATLAB Script | Workspace initialization, sample rates ($10\text{ Hz}$), and Simulink Bus objects |
| [`benchmark_planning_suite.m`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/benchmark_planning_suite.m) | MATLAB Script | Automated benchmark suite testing all 5 canonical Indian road scenarios |
| [`test_planning_pipeline.py`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/test_planning_pipeline.py) | Python Script | 6-part unit test suite validating mathematical correctness and constraints |
| [`planning_visualizer.py`](file:///Users/test/Desktop/SIH/SIH_26037-Autonomous-Vehicles-on-Unstructured-Indian-Roads/Path_planning_decision/planning_visualizer.py) | Python Script | Generates 3 ultra-premium 16:9 presentation slides at 300 DPI |

---





## 4. How to Run & Validate

### Running the Python Test Suite & Verification:
```bash
python3 Path_planning_decision/test_planning_pipeline.py
```
*Expected Output:* `ALL 6 TEST SUITES PASSED! 100% MATHEMATICAL INTEGRITY`

### Generating Presentation Slide Graphics:
```bash
python3 Path_planning_decision/planning_visualizer.py
```

### Running inside MATLAB / Simulink:
```matlab
% 1. Set up Simulink Workspace & Bus Definitions
setup_planning_simulink;

% 2. Run Automated Benchmarking Suite
benchmark_planning_suite;
```
