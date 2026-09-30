function simLog = run_ego_control(varargin)
% RUN_EGO_CONTROL Root Entrypoint for Closed-Loop Autonomous Ego Control
% =========================================================================
% Launches the autonomous Ego vehicle closed-loop control system with:
% 1. Forward-Facing Vision Camera (visionDetectionGenerator)
% 2. Modular Lateral Controllers (Pure Pursuit or MPC)
% 3. Real-Time Longitudinal Collision-Avoidance (AEB / ACC)
% 4. 2D Interactive Bird's-Eye HUD & Driving Scenario Designer App
%
% Usage:
%   run_ego_control;                       % Runs default Pure Pursuit with 2D HUD
%   run_ego_control('controller', 'MPC');  % Runs Model Predictive Control
%   run_ego_control('app');                % Opens directly in Driving Scenario Designer
%   run_ego_control('obstacle', true);     % Adds dynamic lead vehicle to test camera perception & AEB
%   run_ego_control('mpc', 'obstacle');    % MPC with obstacle avoidance
% =========================================================================

    thisDir = fileparts(mfilename('fullpath'));
    if isempty(thisDir), thisDir = pwd; end
    rootDir = thisDir;
    if ~isfolder(fullfile(rootDir, 'vehicle_dynamics'))
        rootDir = fileparts(thisDir);
    end

    addpath(fullfile(rootDir, 'vehicle_dynamics'));
    addpath(fullfile(rootDir, 'scenarios', 'matlab'));
    addpath(fullfile(rootDir, 'scenarios', 'maps_xodr'));

    simLog = ego_circle_control_loop(varargin{:});
end
