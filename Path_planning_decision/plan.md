# Path Planning & Decision Logic Plan
## Problem Statement ID: 26037 — Team Epsilon
### Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads

---

## 1. Executive Summary & Objective

The objective of this track is to generate safe, kinematically feasible, collision-free, and comfortable trajectories for an autonomous vehicle operating on unstructured Indian roads.

Unlike Western autonomous driving systems that rely on strict lane markings and structured traffic lights, this planner must operate in an environment characterized by:
1. **Absence of Lane Markings:** Formal lane discipline is nonexistent; vehicles must plan over a **free-space occupancy grid** rather than adhering rigidly to a lane centerline. A soft virtual corridor maintains human-like driving without trapping the vehicle when an obstacle blocks it.
2. **Sub-Second Reaction Budget:** A sudden cattle crossing or erratic two-wheeler cut-in requires replanning in under $1.0\text{ s}$. The computational architecture must guarantee sub-$50\text{ ms}$ replanning latency.
3. **Unsignalled Junction Negotiation:** Traffic participants resolve priority through proximity, closing speed, and subtle nudging rather than formal right-of-way rules.
4. **Graded Road Surface Anomalies:** Potholes and unpaved shoulders are not binary obstacles. Shallow potholes ($<5\text{ cm}$) can be crossed at reduced speed, whereas deep cavities ($>5\text{ cm}$) must be treated as solid obstacles.

---

## 2. Algorithmic Architecture: Tesla-Inspired 2-Stage Hierarchical Planning

Taking inspiration from industry-leading production autonomous systems (Tesla FSD Planner & Baidu Apollo EM Planner), the planning layer decouples trajectory generation into a **hierarchical two-stage optimization**:

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    INPUTS FROM UPSTREAM LAYERS                                    |
|                                                                                                   |
|  [Trajectory Prediction (`Trajectory/`)]                  [Perception & Sensor Fusion]            |
|   - Dynamic Costmap Slices (tau = 0.5s, 1s, 2s, 3s)        - Road Boundaries & Unpaved Shoulders  |
|   - Multi-Modal GMM Future Trajectories & Ellipses         - 3D Pothole Depressions (X, Z, Depth) |
|   - Stateflow Triggers (min_TTC, critical_agent_id)        - Fused Kinematic Tracks (12 classes)  |
|                                                                                                   |
|  [RoadRunner Scene / Navigation Global Route]                                                     |
|   - Sparse Global Waypoints [X_goal, Y_goal] & Target Cruise Speed (40 km/h)                      |
|   - Ego State Feedback [X, Y, theta, v, a] from Vehicle Dynamics                                  |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
                                                  │
                                                  ▼
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                               PATH PLANNING & DECISION LAYER                                      |
|                                                                                                   |
|  ====================== STAGE 1: DYNAMIC MULTI-LAYER COSTMAP MANAGER ==========================  |
|   Fuses 3 risk layers into MATLAB's `vehicleCostmap`:                                             |
|   1. Layer A (Static Road): Road boundaries + Virtual Lane Soft Bias (cost = 0.05 vs 0.20).      |
|   2. Layer B (Surface Defects): Graded pothole depth costs (<5cm = 0.35 dip; >5cm = 0.95 wall).  |
|   3. Layer C (Dynamic Agents): Spatio-temporal GMM probability uncertainty ellipses.             |
|                                                                                                   |
|  ===================== STAGE 2: STATEFLOW DECISION SUPERVISOR ==================================  |
|   Evaluates tactical behavior state machine:                                                      |
|   - CRUISE: Normal free driving (v_target = 40 km/h)                                              |
|   - SLOW_DOWN: Approaching hazard or agent crossing (v_target = 20 km/h)                          |
|   - YIELD / NEGOTIATE: Unsignalled junction closing speed logic                                   |
|   - STOP: Emergency braking when TTC < 1.8s (e.g. cattle freeze)                                  |
|   - REROUTE: Triggered when forward corridor cost >= 0.90 across entire road width               |
|                                                                                                   |
|  ================== STAGE 3: COARSE DISCRETE SEARCH (CORRIDOR FINDING) =========================  |
|   - Bounded-Window Hybrid A* Planner (`hybrid_astar_planner.m`):                                  |
|     * Explores non-holonomic vehicle kinematics (minimum turn radius R_min = 5.0m).               |
|     * Searches a local 35m moving window in < 35 ms.                                              |
|     * Selects homotopy class (e.g. bypass rickshaw on right, steer clear of deep pothole).       |
|                                                                                                   |
|  ================ STAGE 4: CONTINUOUS TRAJECTORY OPTIMIZATION (QP SPLINE) =====================  |
|   - Continuous Quadratic Programming Optimizer (`continuous_trajectory_optimizer.m`):            |
|     * Optimizes spatial path within the safe convex corridor identified by Hybrid A*.            |
|     * Minimizes curvature rate (jerk) and deviation from reference line:                          |
|       min \int ( w_smooth * ||d^2 p/ds^2||^2 + w_ref * ||p - p_coarse||^2 + w_jerk * ||d^3 p||^2 )|
|     * Guaranteed C^2 continuity (smooth curvature profile without steering spikes).              |
|                                                                                                   |
|  ================ STAGE 5: LONGITUDINAL SPEED PROFILE GENERATION (ST GRAPH) =====================  |
|   - Speed Profile Generator (`speed_profile_generator.m`):                                        |
|     * Curvature-aware speed governor: v(s) <= sqrt(a_y_max / |kappa(s)|).                         |
|     * Surface hazard deceleration: v(s) <= 15 km/h over shallow pothole dips.                     |
|     * S-curve smooth deceleration profiles to halt safely before stopped hazards.                 |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
                                                  │
                                                  ▼
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                              OUTPUT TO VEHICLE DYNAMICS & CONTROL                                 |
|                                                                                                   |
|  Reference Trajectory Vector at 100 Hz:                                                           |
|   - Trajectory Waypoints: [X_ref(t), Y_ref(t), theta_ref(t), kappa_ref(t), v_ref(t), a_ref(t)]    |
|   - Supervisory Mode & Urgency Flag: [mode_id, urgency_factor, replan_flag]                      |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 3. Mathematical Formulation

### 3.1 Bounded-Window Hybrid A* Formulation
The vehicle state in configuration space is $\mathbf{x} = [x, y, \theta]^T \in \text{SE}(2)$.
Motion primitives satisfy the non-holonomic bicycle model:
$$\dot{x} = v \cos \theta, \quad \dot{y} = v \sin \theta, \quad \dot{\theta} = \frac{v}{L} \tan \delta, \quad |\delta| \le \delta_{max}$$

Cost function for node expansion:
$$f(n) = g(n) + h(n)$$
Where:
- $g(n) = \sum_{k=1}^n \left( \Delta s_k + w_\delta |\delta_k| + w_{\Delta \delta} |\delta_k - \delta_{k-1}| + w_{\text{cost}} C(x_k, y_k) \right)$
- $h(n) = \max\left( h_{\text{Reeds-Shepp}}(n), h_{\text{Dijkstra-2D}}(n) \right)$ provides an admissible, non-holonomic distance lower bound.

### 3.2 Continuous Convex Trajectory Optimization (Tesla-Style QP)
Given the coarse discrete path points $\mathbf{p}_{\text{coarse}} = [x_k, y_k]^T$ from Hybrid A* and lateral corridor bounds $[l_{\min,k}, l_{\max,k}]$, the continuous path $\mathbf{p}_k$ is refined via Quadratic Programming:

$$\min_{\mathbf{p}_1, \dots, \mathbf{p}_N} \sum_{k=1}^{N-1} \left\| \mathbf{p}_{k+1} - 2\mathbf{p}_k + \mathbf{p}_{k-1} \right\|^2 + w_{\text{ref}} \sum_{k=1}^N \left\| \mathbf{p}_k - \mathbf{p}_{\text{coarse},k} \right\|^2 + w_{\text{jerk}} \sum_{k=1}^{N-2} \left\| \mathbf{p}_{k+2} - 3\mathbf{p}_{k+1} + 3\mathbf{p}_k - \mathbf{p}_{k-1} \right\|^2$$

Subject to:
1. Convex corridor bounds: $\mathbf{p}_{\min,k} \le \mathbf{p}_k \le \mathbf{p}_{\max,k}$
2. Curvature constraint: $\|\mathbf{p}_{k+1} - 2\mathbf{p}_k + \mathbf{p}_{k-1}\| \le \Delta s^2 \kappa_{\max}$
3. Boundary boundary clearance: $\mathbf{p}_k$ is strictly clamped within $[X_{\text{road\_left}} + 0.6\text{ m}, X_{\text{road\_right}} - 0.6\text{ m}]$

### 3.3 Speed Profile Generation (ST Domain)
Target speed is governed by lateral acceleration comfort ($a_{y,\max} \le 2.5\text{ m/s}^2$):
$$v_{\kappa}(s) = \sqrt{\frac{a_{y,\max}}{|\kappa(s)| + \epsilon}}$$
Overall target velocity profile:
$$v_{\text{target}}(s) = \min\left( v_{\text{Stateflow}}, v_{\kappa}(s), v_{\text{hazard}}(s) \right)$$
Where $v_{\text{hazard}} \le 15\text{ km/h}$ over shallow pothole dips and $0\text{ km/h}$ at emergency stop lines.

---

## 4. Deliverables in `Path_planning_decision/`

```
Path_planning_decision/
├── plan.md                              # Architectural specification (this file)
├── README.md                            # Comprehensive track documentation & API reference
├── dynamic_costmap_manager.m            # 3-Layer vehicleCostmap fusion engine
├── hybrid_astar_planner.m               # Bounded-window Hybrid A* coarse corridor planner
├── continuous_trajectory_optimizer.m    # Tesla-style continuous QP spline trajectory optimizer
├── speed_profile_generator.m            # Curvature-aware and hazard-aware ST speed profile planner
├── decision_supervisor.m                # Stateflow supervisory finite state machine
├── simulink_planning_block.m            # Simulink MATLAB Function Block (C/C++ codegen ready)
├── benchmark_planning_suite.m           # 5-Scenario automated benchmark script
├── planning_visualizer.py               # 16:9 publication-quality visualization generator
└── test_planning_pipeline.py            # Unit test suite verifying planning algorithms
```
