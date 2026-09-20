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
   |  IDD, DriveIndia, DATS_2022, INTSD)         |
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

The perception stage is documented separately in a three-page technical breakdown included in this repository, which covers the reasoning behind the design in more depth than this file does. In summary, the approach separates detection into three complementary jobs rather than relying on a single model.

A YOLOv8 detector, fine-tuned on a merged label set drawn from the Indian Driving Dataset, DriveIndia, DATS_2022, and INTSD, handles agent detection across all four cameras at full frame rate. This model was chosen because it recorded the strongest overall mAP50 among five candidates benchmarked on the Indian DriveIndia dataset. A separate slicing pass over the front camera, run at a lower rate, is used to catch small or distant objects such as potholes and pedestrians that a full-frame pass tends to miss. A third pipeline performs per-pixel semantic segmentation of drivable surface, since road and non-road boundaries in unmarked or unpaved conditions cannot be represented reliably as bounding boxes.

Pothole detection is treated as a distinct problem from ordinary obstacle detection, since a pothole is a depression rather than a protrusion and does not produce a consistent visual signature across lighting conditions. The proposed approach generates candidate regions from the camera pipeline and confirms them geometrically using LiDAR ground-plane curvature analysis, before injecting a graduated cost into the planning layer rather than treating every pothole as a hard obstacle.

The known weak points of this pipeline, including limited detection accuracy on small or rare object classes such as pushcarts and the absence of any species-level animal classification in existing Indian driving datasets, are documented in the perception PDF and in the consolidated technical report.

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

This repository currently contains the team's submission slide deck, which lays out the full perception through control approach along with the reasoning behind each design decision, and a three-page perception deep dive covering the camera pipeline design in more detail.

Alongside these, individual folders such as perception and vehicle dynamics contain small standalone demo scripts. These are proof-of-concept snippets built to validate one idea in isolation, such as pothole costmap injection logic or a comparison between Pure Pursuit and MPC controllers under a simulated emergency replan. They are not yet part of an integrated pipeline. The full closed-loop Simulink model is in progress and will be added as each track's work is validated and merged.

## Status

Perception, sensor fusion and tracking, path planning and decision logic, and vehicle dynamics and control have completed initial research and design. The prediction and trajectory forecasting track is in progress. Several design decisions across tracks depend on its output, including the time-to-collision feed into the decision state machine and the motion model used for unclassified animals. Integration of the individual demos into a single closed-loop model has not yet begun.