% STARTUP Automatically adds modular project directories to MATLAB path
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037: Autonomous Vehicles on Unstructured Indian Roads
projectDir = fileparts(mfilename('fullpath'));
if isempty(projectDir), projectDir = pwd; end

% Core Architecture Paths
addpath(fullfile(projectDir, 'main'));
addpath(fullfile(projectDir, 'vehicle_dynamics'));
addpath(fullfile(projectDir, 'scenarios'));
addpath(fullfile(projectDir, 'scenarios', 'matlab'));
addpath(fullfile(projectDir, 'scenarios', 'maps_xodr'));
addpath(fullfile(projectDir, 'scenarios', 'verification'));
addpath(fullfile(projectDir, 'perception'));
addpath(fullfile(projectDir, 'perception', 'matlab_sahi'));
addpath(fullfile(projectDir, 'perception', 'C3_detector_v1'));
addpath(fullfile(projectDir, 'Sensor_fusion'));
addpath(fullfile(projectDir, 'Trajectory'));
addpath(fullfile(projectDir, 'Path_planning_decision'));

fprintf('\n========================================================================\n');
fprintf('  SIH 26037: Autonomous Driving Stack Environment Initialized\n');
fprintf('  MATLAB Path configured across all perception, planning & control domains.\n');
fprintf('========================================================================\n');
fprintf('  [1] Run All 15 Trajectory Verification Tests (Headless):\n');
fprintf('      >> report = verify_closed_loop_trajectories(''pp'');\n\n');
fprintf('  [2] Run Interactive 2D Animated Scenario (BEV HUD + Sensor Cones):\n');
fprintf('      >> run_instruction(''GAUNTLET'', ''pp'', ''animate'');\n\n');
fprintf('  [3] Run 3D Unreal Engine Co-Simulation (Record Full HD Video):\n');
fprintf('      >> run_instruction(''CPNCO'', ''pp'', ''3d'');\n\n');
fprintf('  [4] Run Full SOTA Perception Stack (YOLOv8s + IMM Fusion + GMM):\n');
fprintf('      >> run_instruction(''CPNCO'', ''pp'', ''stack'');\n');
fprintf('========================================================================\n\n');
