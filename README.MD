# PS 26037: Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads

**Team Epsilon | Smart India Hackathon 2026 | Theme: Smart Vehicles | Organization: MathWorks**

## Overview

Most autonomous driving systems are built and validated on roads with clear lane markings, standardized signage, and predictable traffic behavior. Indian roads frequently offer none of these guarantees. Vehicles of many types share the same physical space, including cars, buses, auto-rickshaws, two-wheelers, bicycles, pushcarts, pedestrians, and animals. Lane discipline is often informal or absent, intersections may be unsignaled, and road edges can be unclear or unpaved.

This project is a closed-loop MATLAB and Simulink simulation of an autonomous vehicle designed to perceive, predict, plan, and drive through exactly these conditions. The system is not built or validated on physical hardware. It is a simulation-first pipeline that integrates perception, trajectory prediction, path planning, decision logic, and vehicle control into a single closed loop, tested against scenarios representative of Indian road conditions.

## Problem Statement

The vehicle must perceive its environment using a multi-sensor setup consisting of camera, LiDAR, and radar, and identify road users and obstacles that do not appear in most Western driving datasets, including auto-rickshaws, pushcarts, and stray or domestic animals. It must predict the short-term motion of these agents, including non-lane-based and irregular movement, and generate a safe, collision-free path that can be replanned in real time as conditions change. The solution must also cope with missing lane markings, informal merging behavior, sudden pedestrian movement, and unexpected obstacles such as potholes.

## Validation Scenarios

The solution is validated against five scenarios, at least two of which are built as full RoadRunner scenes.

1. An unmarked village road
2. A busy urban intersection without traffic signals
3. A highway merge involving slow-moving vehicles
4. A dense market area with mixed traffic
5. A sudden cattle-crossing event

## System Architecture

The pipeline follows the standard four-stage autonomous driving architecture, adapted at each stage for unstructured Indian road conditions.

```
        Sensors: Camera, LiDAR, Radar, GNSS/IMU
                        |
                        v
   +--------------------------------------------+
   |  PERCEPTION                                 |
   |  Object detection (YOLOv8, fine-tuned on    |
   |  IDD, DriveIndia, RDD2022)                  |
   |  Semantic free-space segmentation            |
   |  Pothole and negative-obstacle detection     |
   +--------------------------------------------+
                        |
                        v
   +--------------------------------------------+
   |  SENSOR FUSION AND TRACKING                  |
   |  Object-level fusion (Hungarian assignment,  |
   |  Mahalanobis gating)                         |
   |  Multi-object tracking (IMM: CV/CTRV/CA/     |
   |  Stop models, ByteTrack/OC-SORT association) |
   |  World model output: position, velocity,     |
   |  covariance, occlusion status                |
   +--------------------------------------------+
                        |
                        v
   +--------------------------------------------+
   |  PREDICTION                                  |
   |  Attention-based, non-lane-conditioned       |
   |  trajectory forecasting (MotionFormer /      |
   |  Wayformer-style)                            |
   |  Multi-modal future paths (Gaussian Mixture  |
   |  Model over 1 to 5 second horizon)           |
   +--------------------------------------------+
                        |
                        v
   +--------------------------------------------+
   |  PLANNING AND DECISION LOGIC                 |
   |  Behavior layer: Stateflow state machine     |
   |  (CRUISE -> SLOW_DOWN -> STOP -> REROUTE)    |
   |  Local planner: Hybrid A* over a dynamic      |
   |  occupancy grid / costmap                    |
   +--------------------------------------------+
                        |
                        v
   +--------------------------------------------+
   |  CONTROL                                     |
   |  Kinematic bicycle model / Vehicle Dynamics  |
   |  Blockset                                    |
   |  Pure Pursuit (low speed) or MPC-based        |
   |  tracking (highway merge)                    |
   +--------------------------------------------+
                        |
                        v
              Steering, throttle, brake
                        |
                        +---> replan trigger feeds back into Planning
```

Each block above corresponds to one of the five task tracks described below and is validated independently before integration into the full closed loop.

## Track Breakdown

| Track |  Description |
|---|---|
| Perception and Object Detection | Camera-based detection and classification of Indian road agents, semantic free-space estimation, and pothole/negative-obstacle detection |
| Sensor Fusion and Tracking | Fusing camera, LiDAR, and radar into a single tracked world model, handling occlusion and sensor dropout |
| Prediction and Trajectory Forecasting | Forecasting short-term, multi-modal, non-lane-based motion for surrounding agents |
| Path Planning and Decision Logic |  Behavior state machine and geometric path planning over a dynamic costmap |
| Vehicle Dynamics and Control | Converting planned paths into steering, throttle, and brake commands while maintaining ride comfort |

## Perception in Detail

Perception splits the camera's job into three parts instead of relying on one model. Full details, results and the folder map are in [`Perception/README.md`](Perception/README.md).

A YOLOv8s detector, fine-tuned on the Indian Driving Dataset (IDD), DriveIndia and RDD2022 road-damage images (about 60,000 images, 18 classes including autorickshaw, pushcart, tractor, pothole and a merged speed-bump / zebra-crossing "slow zone" class), finds road users on every camera. It reaches 0.78 mAP50 on the DriveIndia validation set (the published YOLOv8 baseline reports 0.79) and runs at about 45 ms per frame in MATLAB after ONNX import. A second, slower pass (SAHI) re-reads one horizontal band of the front camera at full resolution to catch small or distant road users that the full-frame pass misses: on 1,127 IDD frames it lifts recall of 16-32 px road users from 17% to 47%, at 84 ms per pass, and runs as a 5 Hz Simulink loop. A third model, DeepLab v3+ trained in MATLAB, segments the drivable surface (88.6% IoU on IDD Lite), since unmarked or unpaved road edges cannot be described with boxes.

Potholes are a depression rather than an object, so camera detections are treated as candidates. The plan is to confirm them with LiDAR ground-curvature analysis and hand the planner a graded cost (shallow dip vs deep cavity) instead of a hard obstacle.

Known weak points: rare classes (pushcart, emergency vehicles) have little training data, there is no species-level animal class, and the camera calibration used so far is a placeholder until the simulation camera is fixed.

## Toolchain

| Toolbox | Role |
|---|---|
| RoadRunner | Scene design for village road and urban intersection environments |
| Automated Driving Toolbox | Sensor modeling, synthetic detections, sensor fusion, bird's-eye-view visualization |
| Computer Vision Toolbox | Object detection and image processing |
| Deep Learning Toolbox | Detection backbone and trajectory prediction network, including attention layers |
| Sensor Fusion and Tracking Toolbox | Multi-object tracking, Kalman and IMM filters |
| Navigation Toolbox | Path planning (Hybrid A*), occupancy grids and costmaps |
| Stateflow | Behavior and decision logic state machine |
| Vehicle Dynamics Blockset | Vehicle motion and actuation |

## Metrics

The problem statement explicitly scores the following.

- Replanning latency, measured as time from a new perception or prediction input to an updated path being available, reported as mean and 95th percentile
- Path smoothness, measured through the curvature profile of the planned path or its jerk
- Scenario completion rate, measured as successful runs divided by total runs, per scenario and overall

Additional benchmarks under consideration include collision rate, minimum time-to-collision, detection mAP on the IDD and DriveIndia validation splits, and prediction error using average and final displacement error.

## Repository Contents

This repository contains the complete closed-loop autonomous driving stack, simulation scenarios, and verification tools:

- **Integrated AV Stack (`main/`)**: Orchestrates the closed-loop pipeline (`AutonomousAVStack.m`, `run_instruction.m`) integrating perception, sensor fusion, prediction, spatiotemporal replanning, and tri-hybrid vehicle control.
- **Perception & 3D Grounding (`Perception/`)**: Trained C3 YOLOv8s 12-class detector for Indian traffic, SAHI sliced inference, and inverse pinhole camera projection to 3D Cartesian coordinates.
- **Sensor Fusion (`Sensor_fusion/`)**: Information Matrix fusion (camera azimuth + radar range/Doppler) and Semantic IMM Kalman filtering (CV, CTRV, CA motion models).
- **Prediction & Decision (`Trajectory/`, `Path_planning_decision/`)**: Multi-modal GMM trajectory prediction (3.0s horizon), dynamic 4D costmaps, and Stateflow behavior state machine.
- **Vehicle Dynamics & Control (`vehicle_dynamics/`)**: Quintic Frenet polynomial trajectory replanner (10 Hz) and Tri-Hybrid Pure Pursuit / MPC controller (50 Hz).
- **Simulation Scenarios (`scenarios/`)**: 15 OpenSCENARIO (.xosc) and MATLAB driving scenarios covering Euro NCAP safety benchmarks and complex Indian urban edge cases (wrong-way encounters, cattle crossing, auto cut-ins, school zones, multi-threat gauntlet).
- **Automated Verification (`scenarios/verification/`)**: Test harness (`verify_closed_loop_trajectories.m`) verifying all 15 scenarios with a 100% pass rate (0 collisions, 0 curb/lane excursions, stable lateral dynamics).
- **Documentation & Execution Guides**: Step-by-step setup and unified test execution instructions in [`how_to_run.md`](how_to_run.md) and [`startup.m`](startup.m).