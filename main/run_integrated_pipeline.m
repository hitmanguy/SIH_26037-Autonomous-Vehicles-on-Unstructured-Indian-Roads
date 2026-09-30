function results = run_integrated_pipeline(duration, enable_viz)
% RUN_INTEGRATED_PIPELINE Master Autonomous AV Stack Pipeline Verification
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% Integrates:
%   - Perception: C3 YOLOv8s 12-Class Detector
%   - Sensor Fusion: Pinhole 2D-to-3D BEV & 77 GHz Radar Covariance Merge
%   - Trajectory: Multi-Modal GMM Lookahead & Spatio-Temporal Dynamic Costmaps
%   - Vehicle Dynamics: Closed-Loop Pure Pursuit & Kinematic AEB / ACC
%
% Usage:
%   results = run_integrated_pipeline();            % 10-second default simulation
%   results = run_integrated_pipeline(15.0, true);  % 15s with visualization plots

    if nargin < 1 || isempty(duration), duration = 10.0; end
    if nargin < 2 || isempty(enable_viz), enable_viz = false; end

    fprintf('========================================================================\n');
    fprintf('  SIH 26037: FULL-STACK AUTONOMOUS VEHICLE INTEGRATION PIPELINE        \n');
    fprintf('  Team Epsilon | MathWorks Automated Driving & Computer Vision Stack   \n');
    fprintf('========================================================================\n\n');

    base_dir = fileparts(mfilename('fullpath'));
    if ~isfolder(fullfile(base_dir, 'vehicle_dynamics'))
        base_dir = fileparts(base_dir);
    end
    addpath(base_dir);
    addpath(fullfile(base_dir, 'main'));
    addpath(fullfile(base_dir, 'vehicle_dynamics'));
    addpath(fullfile(base_dir, 'perception'));
    addpath(fullfile(base_dir, 'Sensor_fusion'));
    addpath(fullfile(base_dir, 'Trajectory'));

    % 1. Instantiate the Unified Champion Stack
    stack = AutonomousAVStack();
    stack.setup();

    % 2. Run Scenario Simulation
    results = stack.run_scenario(duration, enable_viz);

    % 3. Save Telemetry Log to Disk
    log_dir = fullfile(base_dir, 'logs');
    if ~isfolder(log_dir), mkdir(log_dir); end
    
    timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    log_file = fullfile(log_dir, sprintf('integrated_pipeline_run_%s.mat', timestamp));
    save(log_file, 'results');
    fprintf('>> Execution telemetry successfully recorded to:\n   %s\n\n', log_file);
end
