%% SETUP_SIMULINK_SIMULATION
% =========================================================================
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% MASTER SIMULINK & ROADRUNNER CLOSED-LOOP SIMULATOR INITIALIZATION SCRIPT
% 
% Configures:
%   1. Multi-rate execution clocks (100 Hz dynamics, 50 Hz control, 20 Hz fusion, 10 Hz planning)
%   2. Strongly typed Simulink.Bus definitions for all inter-subsystem data buses
%   3. Ego vehicle physical dimensions, actuator limits, and bicycle model parameters
%   4. RoadRunner Scenario Co-Simulation Interface tokens & communication ports
%   5. Perception, Sensor Fusion, Prediction, and Planning workspace registers
% =========================================================================

clearvars -except fused_tracks pred_trajectories; clc;
fprintf('========================================================================\n');
fprintf('  SIH 26037: Master Setup for Closed-Loop Simulink & RoadRunner Pipeline \n');
fprintf('  Team Epsilon | Autonomous Vehicles on Unstructured Indian Roads       \n');
fprintf('========================================================================\n\n');

%% 1. Multi-Rate Execution Timing Clocks
Ts_dynamics   = 0.010;   % 100 Hz: Vehicle Dynamics Bicycle Model & Plant Integrator (10 ms)
Ts_controller = 0.020;   % 50 Hz:  Adaptive Pure Pursuit & Model Predictive Control (20 ms)
Ts_fusion     = 0.050;   % 20 Hz:  Radar Sensing & Semantic IMM Sensor Fusion (50 ms)
Ts_prediction = 0.100;   % 10 Hz:  Multi-Modal GMM Trajectory Prediction (100 ms)
Ts_planning   = 0.100;   % 10 Hz:  Frenet Spatiotemporal Dynamic Replanner (100 ms)
Ts_camera     = 0.064;   % 15.6 Hz: C3 YOLOv8s Windshield Surround ADAS Camera (64 ms)

assignin('base', 'Ts_dynamics',   Ts_dynamics);
assignin('base', 'Ts_controller', Ts_controller);
assignin('base', 'Ts_fusion',     Ts_fusion);
assignin('base', 'Ts_prediction', Ts_prediction);
assignin('base', 'Ts_planning',   Ts_planning);
assignin('base', 'Ts_camera',     Ts_camera);

%% 2. Fixed-Step Solver Configuration
sim_solver     = 'ode4'; % 4th-order Runge-Kutta fixed-step integrator
sim_step_size  = Ts_dynamics; % 0.01s base step
sim_stop_time  = 35.0;   % 35 seconds scenario duration

assignin('base', 'sim_solver', sim_solver);
assignin('base', 'sim_step_size', sim_step_size);
assignin('base', 'sim_stop_time', sim_stop_time);

%% 3. Ego Vehicle Physical & Kinematic Parameters (Tata Nexon EV / Hyundai Creta ADAS Spec)
EgoParams = struct();
EgoParams.Wheelbase   = 2.80;    % L (m)
EgoParams.TrackWidth  = 1.60;    % (m)
EgoParams.Length      = 4.50;    % (m)
EgoParams.Width       = 2.00;    % (m)
EgoParams.Height      = 1.65;    % (m)
EgoParams.Mass        = 1650.0;  % Curb weight (kg)
EgoParams.lf          = 1.25;    % Distance CG to front axle (m)
EgoParams.lr          = 1.55;    % Distance CG to rear axle (m)
EgoParams.Iz          = 2500.0;  % Yaw moment of inertia (kg*m^2)

% Actuator Constraints (ISO 26262 ASIL-D Compliant)
EgoParams.MaxSteerDeg = 35.0;    % Maximum front wheel steer (deg)
EgoParams.MaxSteerRad = deg2rad(35.0);
EgoParams.MaxSteerRate= deg2rad(28.0); % Steering slew rate limit (rad/s)
EgoParams.MaxAccel    = 1.80;    % Comfort acceleration limit (m/s^2)
EgoParams.ComfortDecel= -3.00;   % Comfort deceleration limit (m/s^2)
EgoParams.MaxDecel    = -7.50;   % Maximum emergency braking authority (m/s^2)
EgoParams.CruiseSpeed = 6.94;    % Nominal urban cruise speed: 25 km/h (m/s)

assignin('base', 'EgoParams', EgoParams);

%% 4. Road Geometry & Indian Highway Regulatory Boundaries
RoadGeometry = struct();
RoadGeometry.CenterDividerY  = -0.40;   % Critical centerline boundary: Y > -0.40m is oncoming traffic!
RoadGeometry.CruisingLaneY   = -1.75;   % Lane -1 center (Left side of road, m)
RoadGeometry.PassingLaneY    = -5.00;   % Lane -2 center (Outer lane, m)
RoadGeometry.OuterShoulderY  = -6.50;   % Road curb / edge (m)
RoadGeometry.LaneWidth       = 3.50;    % Standard IRC lane width (m)
RoadGeometry.RoadBounds      = [-6.50, -0.40];

assignin('base', 'RoadGeometry', RoadGeometry);

%% 5. Define Strongly Typed Simulink Bus Objects

% (A) Bus: Ego Vehicle State [X, Y, Yaw, Vx, Vy, YawRate, SteerAngle, Accel]
clear elems;
elems(1) = Simulink.BusElement; elems(1).Name = 'X';         elems(1).DataType = 'double';
elems(2) = Simulink.BusElement; elems(2).Name = 'Y';         elems(2).DataType = 'double';
elems(3) = Simulink.BusElement; elems(3).Name = 'Yaw';       elems(3).DataType = 'double';
elems(4) = Simulink.BusElement; elems(4).Name = 'Vx';        elems(4).DataType = 'double';
elems(5) = Simulink.BusElement; elems(5).Name = 'Vy';        elems(5).DataType = 'double';
elems(6) = Simulink.BusElement; elems(6).Name = 'YawRate';   elems(6).DataType = 'double';
elems(7) = Simulink.BusElement; elems(7).Name = 'SteerAngle';elems(7).DataType = 'double';
elems(8) = Simulink.BusElement; elems(8).Name = 'Accel';     elems(8).DataType = 'double';
BusEgoState = Simulink.Bus;
BusEgoState.Elements = elems;
assignin('base', 'BusEgoState', BusEgoState);

% (B) Bus: Fused Sensor Track State
clear elems;
elems(1) = Simulink.BusElement; elems(1).Name = 'id';         elems(1).DataType = 'double';
elems(2) = Simulink.BusElement; elems(2).Name = 'class_id';   elems(2).DataType = 'double';
elems(3) = Simulink.BusElement; elems(3).Name = 'score';      elems(3).DataType = 'double';
elems(4) = Simulink.BusElement; elems(4).Name = 'X';          elems(4).DataType = 'double';
elems(5) = Simulink.BusElement; elems(5).Name = 'Z';          elems(5).DataType = 'double';
elems(6) = Simulink.BusElement; elems(6).Name = 'Vx';         elems(6).DataType = 'double';
elems(7) = Simulink.BusElement; elems(7).Name = 'Vz';         elems(7).DataType = 'double';
elems(8) = Simulink.BusElement; elems(8).Name = 'valid_flag'; elems(8).DataType = 'double';
BusTrackState = Simulink.Bus;
BusTrackState.Elements = elems;
assignin('base', 'BusTrackState', BusTrackState);

% (C) Bus: Road Surface Pothole Defect Token
clear elems;
elems(1) = Simulink.BusElement; elems(1).Name = 'X';          elems(1).DataType = 'double';
elems(2) = Simulink.BusElement; elems(2).Name = 'Z';          elems(2).DataType = 'double';
elems(3) = Simulink.BusElement; elems(3).Name = 'depth_cm';   elems(3).DataType = 'double';
elems(4) = Simulink.BusElement; elems(4).Name = 'radius_m';   elems(4).DataType = 'double';
elems(5) = Simulink.BusElement; elems(5).Name = 'valid_flag'; elems(5).DataType = 'double';
BusPothole = Simulink.Bus;
BusPothole.Elements = elems;
assignin('base', 'BusPothole', BusPothole);

% (D) Bus: Stateflow Tactical Decision Triggers
clear elems;
elems(1) = Simulink.BusElement; elems(1).Name = 'mode_id';      elems(1).DataType = 'double'; % 1:CRUISE, 2:SLOW, 3:YIELD, 4:STOP, 5:REROUTE
elems(2) = Simulink.BusElement; elems(2).Name = 'min_TTC';      elems(2).DataType = 'double';
elems(3) = Simulink.BusElement; elems(3).Name = 'crit_agent_id';elems(3).DataType = 'double';
elems(4) = Simulink.BusElement; elems(4).Name = 'crit_mode';    elems(4).DataType = 'double';
elems(5) = Simulink.BusElement; elems(5).Name = 'is_blocked';   elems(5).DataType = 'double';
elems(6) = Simulink.BusElement; elems(6).Name = 'target_factor';elems(6).DataType = 'double';
BusStateflowTriggers = Simulink.Bus;
BusStateflowTriggers.Elements = elems;
assignin('base', 'BusStateflowTriggers', BusStateflowTriggers);

% (E) Bus: Reference Trajectory Waypoint Point
clear elems;
elems(1) = Simulink.BusElement; elems(1).Name = 'X';     elems(1).DataType = 'double';
elems(2) = Simulink.BusElement; elems(2).Name = 'Y';     elems(2).DataType = 'double';
elems(3) = Simulink.BusElement; elems(3).Name = 'theta'; elems(3).DataType = 'double';
elems(4) = Simulink.BusElement; elems(4).Name = 'kappa'; elems(4).DataType = 'double';
elems(5) = Simulink.BusElement; elems(5).Name = 'v';     elems(5).DataType = 'double';
elems(6) = Simulink.BusElement; elems(6).Name = 'a';     elems(6).DataType = 'double';
BusTrajectoryPoint = Simulink.Bus;
BusTrajectoryPoint.Elements = elems;
assignin('base', 'BusTrajectoryPoint', BusTrajectoryPoint);

% (F) Bus: Actuator Control Demands
clear elems;
elems(1) = Simulink.BusElement; elems(1).Name = 'SteerDemand'; elems(1).DataType = 'double'; % Front wheel angle (rad)
elems(2) = Simulink.BusElement; elems(2).Name = 'Throttle';    elems(2).DataType = 'double'; % [0.0, 1.0]
elems(3) = Simulink.BusElement; elems(3).Name = 'Brake';       elems(3).DataType = 'double'; % [0.0, 1.0]
elems(4) = Simulink.BusElement; elems(4).Name = 'TargetAccel'; elems(4).DataType = 'double'; % Demanded longitudinal accel (m/s^2)
elems(5) = Simulink.BusElement; elems(5).Name = 'IsAEB';       elems(5).DataType = 'double'; % Emergency brake active flag (0 or 1)
BusControlDemand = Simulink.Bus;
BusControlDemand.Elements = elems;
assignin('base', 'BusControlDemand', BusControlDemand);

% (G) Bus: RoadRunner 3D Co-Simulation Pose Feedback
clear elems;
elems(1) = Simulink.BusElement; elems(1).Name = 'X';     elems(1).DataType = 'double';
elems(2) = Simulink.BusElement; elems(2).Name = 'Y';     elems(2).DataType = 'double';
elems(3) = Simulink.BusElement; elems(3).Name = 'Z';     elems(3).DataType = 'double';
elems(4) = Simulink.BusElement; elems(4).Name = 'Roll';  elems(4).DataType = 'double';
elems(5) = Simulink.BusElement; elems(5).Name = 'Pitch'; elems(5).DataType = 'double';
elems(6) = Simulink.BusElement; elems(6).Name = 'Yaw';   elems(6).DataType = 'double';
elems(7) = Simulink.BusElement; elems(7).Name = 'Vx';    elems(7).DataType = 'double';
elems(8) = Simulink.BusElement; elems(8).Name = 'Vy';    elems(8).DataType = 'double';
elems(9) = Simulink.BusElement; elems(9).Name = 'Vz';    elems(9).DataType = 'double';
BusRoadRunnerPose = Simulink.Bus;
BusRoadRunnerPose.Elements = elems;
assignin('base', 'BusRoadRunnerPose', BusRoadRunnerPose);

fprintf('  [OK] Defined 7 Strongly Typed Simulink Buses:\n');
fprintf('       - BusEgoState, BusTrackState, BusPothole, BusStateflowTriggers,\n');
fprintf('         BusTrajectoryPoint, BusControlDemand, BusRoadRunnerPose\n\n');

%% 6. RoadRunner Scenario Co-Simulation Interface Configuration
RoadRunnerConfig = struct();
RoadRunnerConfig.Host            = '127.0.0.1';
RoadRunnerConfig.Port            = 50051;           % gRPC Co-Simulation Port
RoadRunnerConfig.EgoActorID      = 1;
RoadRunnerConfig.SyncMode        = 'Synchronous';   % Lockstep simulation
RoadRunnerConfig.TimeoutSec      = 10.0;
RoadRunnerConfig.ScenarioFile    = fullfile(pwd, 'scenarios', 'scenarios_xosc', 'indian', 'Indian_WrongWay_Encounter.xosc');
RoadRunnerConfig.OpenDRIVEFile   = fullfile(pwd, 'scenarios', 'maps_xodr', 'X-Intersection_NCAP.xodr');

assignin('base', 'RoadRunnerConfig', RoadRunnerConfig);

%% 7. Initial State Vectors for Harness Warm-Start
N_actors   = 16;
N_potholes = 5;
N_plan_pts = 40;

init_ego_state = struct(...
    'X', 20.0, 'Y', -1.75, 'Yaw', 0.0, ...
    'Vx', EgoParams.CruiseSpeed, 'Vy', 0.0, 'YawRate', 0.0, ...
    'SteerAngle', 0.0, 'Accel', 0.0);

init_tracks = repmat(struct('id', 0, 'class_id', 0, 'score', 0, ...
    'X', 0, 'Z', 0, 'Vx', 0, 'Vz', 0, 'valid_flag', 0), N_actors, 1);

init_potholes = repmat(struct('X', 0, 'Z', 0, 'depth_cm', 0, ...
    'radius_m', 0, 'valid_flag', 0), N_potholes, 1);

init_control = struct('SteerDemand', 0.0, 'Throttle', 0.35, 'Brake', 0.0, ...
    'TargetAccel', 0.0, 'IsAEB', 0.0);

init_rr_pose = struct('X', 20.0, 'Y', -1.75, 'Z', 0.0, ...
    'Roll', 0.0, 'Pitch', 0.0, 'Yaw', 0.0, ...
    'Vx', EgoParams.CruiseSpeed, 'Vy', 0.0, 'Vz', 0.0);

assignin('base', 'init_ego_state', init_ego_state);
assignin('base', 'init_tracks',    init_tracks);
assignin('base', 'init_potholes',  init_potholes);
assignin('base', 'init_control',   init_control);
assignin('base', 'init_rr_pose',   init_rr_pose);

fprintf('  [OK] Multi-rate clocks: Dynamics 100Hz | Control 50Hz | Fusion 20Hz | Planning 10Hz\n');
fprintf('  [OK] Ego Parameters: Wheelbase %.2fm | MaxDecel %.1fm/s^2 | Cruise %.1f km/h\n', ...
    EgoParams.Wheelbase, EgoParams.MaxDecel, EgoParams.CruiseSpeed * 3.6);
fprintf('  [OK] Indian Carriageway Bounds: [%.2f, %.2f] m (Centerline strictly capped)\n', ...
    RoadGeometry.RoadBounds(1), RoadGeometry.RoadBounds(2));
fprintf('\n========================================================================\n');
fprintf('  Simulink Workspace Ready! Run `build_closed_loop_model` to compile canvas.\n');
fprintf('========================================================================\n');
