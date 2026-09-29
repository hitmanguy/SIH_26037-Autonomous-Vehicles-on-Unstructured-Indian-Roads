%% SETUP_PLANNING_SIMULINK
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% This initialization script configures the MATLAB workspace, Simulink Bus objects,
% sample times, and test harnesses for the Path Planning & Decision Subsystem.
% Run this script before executing any Simulink model containing `simulink_planning_block.m`.

clearvars -except fused_tracks pred_trajectories; clc;
fprintf('========================================================================\n');
fprintf('  SIH 26037: Configuring Simulink Workspace for Path Planning & Decision \n');
fprintf('  Team Epsilon | MathWorks Automated Driving & Simulink                  \n');
fprintf('========================================================================\n\n');

%% 1. Subsystem Dimensions & Sample Time Parameters
Ts_plan     = 0.10;   % 10 Hz Planning Execution Rate (100 ms)
N_pts_plan  = 40;     % Number of reference trajectory waypoints
N_actors    = 16;     % Maximum tracked actors
N_potholes  = 5;      % Maximum active pothole defect tokens

assignin('base', 'Ts_plan', Ts_plan);
assignin('base', 'N_pts_plan', N_pts_plan);
assignin('base', 'N_actors', N_actors);
assignin('base', 'N_potholes', N_potholes);

%% 2. Define Simulink Bus Objects for Strongly Typed Interfaces

% (A) Bus: Reference Trajectory Waypoint Element
clear elems;
elems(1) = Simulink.BusElement; elems(1).Name = 'X';     elems(1).DataType = 'double';
elems(2) = Simulink.BusElement; elems(2).Name = 'Z';     elems(2).DataType = 'double';
elems(3) = Simulink.BusElement; elems(3).Name = 'theta'; elems(3).DataType = 'double';
elems(4) = Simulink.BusElement; elems(4).Name = 'kappa'; elems(4).DataType = 'double';
elems(5) = Simulink.BusElement; elems(5).Name = 'v';     elems(5).DataType = 'double';
elems(6) = Simulink.BusElement; elems(6).Name = 'a';     elems(6).DataType = 'double';
BusTrajectoryPoint = Simulink.Bus;
BusTrajectoryPoint.Elements = elems;
assignin('base', 'BusTrajectoryPoint', BusTrajectoryPoint);

% (B) Bus: Supervisory Planning Status
clear elems;
elems(1) = Simulink.BusElement; elems(1).Name = 'state_id';       elems(1).DataType = 'double';
elems(2) = Simulink.BusElement; elems(2).Name = 'replan_flag';     elems(2).DataType = 'double';
elems(3) = Simulink.BusElement; elems(3).Name = 'urgency_factor'; elems(3).DataType = 'double';
elems(4) = Simulink.BusElement; elems(4).Name = 'is_optimal';     elems(4).DataType = 'double';
elems(5) = Simulink.BusElement; elems(5).Name = 'min_clearance';  elems(5).DataType = 'double';
BusPlanningStatus = Simulink.Bus;
BusPlanningStatus.Elements = elems;
assignin('base', 'BusPlanningStatus', BusPlanningStatus);

% (C) Bus: Lateral Safety Corridor Limits
clear elems;
elems(1) = Simulink.BusElement; elems(1).Name = 'x_min'; elems(1).DataType = 'double';
elems(2) = Simulink.BusElement; elems(2).Name = 'x_max'; elems(2).DataType = 'double';
BusCorridorLimit = Simulink.Bus;
BusCorridorLimit.Elements = elems;
assignin('base', 'BusCorridorLimit', BusCorridorLimit);

%% 3. Generate Nominal Test Inputs in Workspace
ego_state_init = [-1.8; 0.0; 0.0; 11.11; 0.0]; % [x=-1.8m, z=0m, th=0rad, v=40km/h, a=0m/s^2]
stateflow_triggers_init = [1.0; 99.0; 0.0; 0.0; 0.0]; % CRUISE mode, min_TTC = 99s

potholes_init = zeros(N_potholes, 5);
% Add sample deep cavity at x = -1.8m, z = 18m, depth = 7cm (must avoid!)
potholes_init(1, :) = [-1.8, 18.0, 7.0, 0.6, 1.0];
% Add sample shallow dip at x = -0.5m, z = 28m, depth = 3.5cm (passable at reduced speed)
potholes_init(2, :) = [-0.5, 28.0, 3.5, 0.5, 1.0];

pred_trajectories_init = zeros(N_actors, 3, 30, 2);
% Add sample oncoming motorcycle moving in opposite direction
pred_trajectories_init(1, 1, :, 1) = 1.8;
for step = 1:30
    pred_trajectories_init(1, 1, step, 2) = 35.0 - (step - 1) * 0.8;
end

global_target_init = [-1.8; 50.0; 11.11]; % Forward goal along virtual lane

assignin('base', 'ego_state_init', ego_state_init);
assignin('base', 'stateflow_triggers_init', stateflow_triggers_init);
assignin('base', 'potholes_init', potholes_init);
assignin('base', 'pred_trajectories_init', pred_trajectories_init);
assignin('base', 'global_target_init', global_target_init);

fprintf('  [OK] Sample Time Ts_plan = %.2f s (10 Hz)\n', Ts_plan);
fprintf('  [OK] Trajectory Horizon: %d points x %.2f m = %.1f m\n', N_pts_plan, 0.8, N_pts_plan * 0.8);
fprintf('  [OK] Simulink Bus Objects Created: BusTrajectoryPoint, BusPlanningStatus, BusCorridorLimit\n');
fprintf('  [OK] Test Harness Variables Initialized in Base Workspace\n\n');
fprintf('Ready to run `simulink_planning_block.m` or execute closed-loop simulation!\n');
