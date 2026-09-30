% STARTUP Automatically adds modular project directories to MATLAB path
projectDir = fileparts(mfilename('fullpath'));
if isempty(projectDir), projectDir = pwd; end

addpath(fullfile(projectDir, 'main'));
addpath(fullfile(projectDir, 'vehicle_dynamics'));
addpath(fullfile(projectDir, 'scenarios'));
addpath(fullfile(projectDir, 'scenarios', 'matlab'));
addpath(fullfile(projectDir, 'scenarios', 'maps_xodr'));
addpath(fullfile(projectDir, 'perception'));
addpath(fullfile(projectDir, 'perception', 'matlab_sahi'));
addpath(fullfile(projectDir, 'perception', 'C3_detector_v1'));
addpath(fullfile(projectDir, 'Sensor_fusion'));
addpath(fullfile(projectDir, 'Trajectory'));
addpath(fullfile(projectDir, 'Path_planning_decision'));
