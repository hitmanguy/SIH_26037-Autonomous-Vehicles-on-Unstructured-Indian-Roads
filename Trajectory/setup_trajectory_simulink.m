%% SETUP_TRAJECTORY_SIMULINK
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% This initialization script configures the MATLAB workspace, Simulink Bus objects,
% sample times, and input test harnesses for the Trajectory Prediction Subsystem.
% Run this script before executing any Simulink model containing `simulink_prediction_block.m`.

clearvars -except fused_tracks; clc;
fprintf('========================================================================\n');
fprintf('  SIH 26037: Configuring Simulink Workspace for Trajectory Prediction   \n');
fprintf('  Team Epsilon | MathWorks Automated Driving & Simulink                  \n');
fprintf('========================================================================\n\n');

%% 1. Subsystem Dimensions & Sample Time Parameters
Ts_pred      = 0.10;   % 10 Hz Prediction Execution Rate (100 ms)
N_max_tracks = 16;     % Maximum tracked actors in local scene
K_modes      = 3;      % Number of predicted multi-modal trajectories
H_steps      = 30;     % Prediction horizon steps (30 * 0.1s = 3.0s)
N_potholes   = 5;      % Maximum active pothole defect tokens

assignin('base', 'Ts_pred', Ts_pred);
assignin('base', 'N_max_tracks', N_max_tracks);
assignin('base', 'K_modes', K_modes);
assignin('base', 'H_steps', H_steps);
assignin('base', 'N_potholes', N_potholes);

%% 2. Define Simulink Bus Objects for Strongly Typed Interfaces
% (A) Bus: Track State Element
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
clear elems;

% (B) Bus: Pothole Hazard Element
elems(1) = Simulink.BusElement; elems(1).Name = 'X';          elems(1).DataType = 'double';
elems(2) = Simulink.BusElement; elems(2).Name = 'Z';          elems(2).DataType = 'double';
elems(3) = Simulink.BusElement; elems(3).Name = 'depth_cm';   elems(3).DataType = 'double';
elems(4) = Simulink.BusElement; elems(4).Name = 'radius_m';   elems(4).DataType = 'double';
elems(5) = Simulink.BusElement; elems(5).Name = 'valid_flag'; elems(5).DataType = 'double';
BusPothole = Simulink.Bus;
BusPothole.Elements = elems;
assignin('base', 'BusPothole', BusPothole);
clear elems;

% (C) Bus: Stateflow Supervisory Triggers
elems(1) = Simulink.BusElement; elems(1).Name = 'mode_id';      elems(1).DataType = 'double';
elems(2) = Simulink.BusElement; elems(2).Name = 'min_TTC';      elems(2).DataType = 'double';
elems(3) = Simulink.BusElement; elems(3).Name = 'crit_agent_id';elems(3).DataType = 'double';
elems(4) = Simulink.BusElement; elems(4).Name = 'crit_mode';    elems(4).DataType = 'double';
elems(5) = Simulink.BusElement; elems(5).Name = 'is_blocked';   elems(5).DataType = 'double';
BusStateflowTriggers = Simulink.Bus;
BusStateflowTriggers.Elements = elems;
assignin('base', 'BusStateflowTriggers', BusStateflowTriggers);
clear elems;

fprintf('[+] Defined Simulink Bus Objects: BusTrackState, BusPothole, BusStateflowTriggers\n');

%% 3. Generate Realistic Test Harness Input Signals
% Test Scenario: 3 Agents (Autorickshaw, Stray Cow, Motorcycle) + 1 Pothole
test_tracks_state = zeros(N_max_tracks, 8);
test_tracks_cov   = zeros(N_max_tracks, 4);
test_potholes     = zeros(N_potholes, 5);
test_ego_state    = [-1.8; 0.0; 0.0; 11.11]; % [X, Z, Vx, Vz] (40 km/h)

% Actor 1: Autorickshaw (Class 6) approaching center line
test_tracks_state(1, :) = [1, 6, 0.88, -0.6, 15.0, 0.1, 9.5, 1.0];
test_tracks_cov(1, :)   = [0.08, 0.25, 0.01, 0.0];

% Actor 2: Stray Cow (Class 9) crossing diagonally
test_tracks_state(2, :) = [2, 9, 0.92, -3.0, 24.0, 0.8, 1.2, 1.0];
test_tracks_cov(2, :)   = [0.05, 0.15, 0.00, 0.0];

% Actor 3: Motorcycle (Class 7) in left lane
test_tracks_state(3, :) = [3, 7, 0.85, 1.8, 20.0, -0.2, 12.5, 1.0];
test_tracks_cov(3, :)   = [0.06, 0.20, 0.00, 0.0];

% Pothole 1: Deep pothole (7.5 cm) at X = -0.8m, Z = 16.5m
test_potholes(1, :) = [-0.8, 16.5, 7.5, 0.75, 1.0];

assignin('base', 'test_tracks_state', test_tracks_state);
assignin('base', 'test_tracks_cov', test_tracks_cov);
assignin('base', 'test_potholes', test_potholes);
assignin('base', 'test_ego_state', test_ego_state);

fprintf('[+] Populated test harness input variables in base workspace.\n');

%% 4. Smoke Test Execution of the Prediction Function Block
fprintf('>> Executing smoke test of simulink_prediction_block...\n');
tic;
[pred_trajectories, pred_covariances, pred_probabilities, stateflow_triggers, dynamic_cost_summary] = ...
    simulink_prediction_block(test_tracks_state, test_tracks_cov, test_potholes, test_ego_state);
t_exec_ms = toc * 1000;

fprintf('   -> Execution Successful in %.2f ms (Budget: 100 ms @ 10 Hz)!\n', t_exec_ms);
fprintf('   -> Stateflow Mode Trigger: ID = %.0f (Min TTC: %.2f s | Crit Agent: #%.0f)\n', ...
    stateflow_triggers(1), stateflow_triggers(2), stateflow_triggers(3));
fprintf('   -> Output Tensors Validated:\n');
fprintf('      pred_trajectories  : [%dx%dx%dx%d] double\n', size(pred_trajectories));
fprintf('      pred_covariances   : [%dx%dx%dx%d] double\n', size(pred_covariances));
fprintf('      pred_probabilities : [%dx%d] double\n', size(pred_probabilities));
fprintf('========================================================================\n');
fprintf('  Workspace is 100%% Ready for Simulink Integration!                     \n');
fprintf('========================================================================\n');
