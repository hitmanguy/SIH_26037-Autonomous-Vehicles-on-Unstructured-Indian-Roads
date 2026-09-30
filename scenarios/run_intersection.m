function scenario = run_intersection(varargin)
% RUN_INTERSECTION Root Entry Point for Autonomous Driving Simulations
% =========================================================================
% Supports running Euro NCAP, Indian Mixed Traffic, UC Berkeley Scenic Fuzzing,
% and ASAM Standard OpenSCENARIO (.xosc) and OpenDRIVE (.xodr) files.
%
% Supported Quick Shortcuts:
%   --- UC Berkeley Scenic Probabilistic Fuzzer (Auto-Purged Temp Files) ---
%   run_intersection('scenic')              % Sample fresh Scenic traffic & animate with 2D HUD
%   run_intersection('scenic', 'app')       % Sample fresh Scenic traffic & OPEN IN DRIVING SCENARIO DESIGNER APP
%   run_intersection('scenic', '3d')        % Sample fresh Scenic traffic & view in Unreal Engine 3D
%   run_intersection('scenic', 'Bangalore') % Sample fresh Scenic traffic on Bengaluru OpenStreetMap road
%
%   --- Indian Mixed Traffic Scenarios ---
%   run_intersection('Indian_AutoCutIn')    % Auto-Rickshaw aggressive cut-in
%   run_intersection('Indian_TwoWheeler')   % Two-wheeler motorcycle filtering
%   run_intersection('Indian_Jaywalk')      % Mid-block jaywalking pedestrian
%   run_intersection('Indian_Cattle')       % Stray cattle / obstacle in lane
%
%   --- Euro NCAP Active Safety Protocols ---
%   run_intersection('CPNCO')               % Car-to-Pedestrian Nearside Child Obstructed
%   run_intersection('CPTA')                % Car-to-Pedestrian Turning Adult at Intersection
%   run_intersection('CCFtap')              % Car-to-Car Front Turn Across Path
%   run_intersection('CCCscp')              % Car-to-Car Straight Crossing (5 Vehicles)
%
%   --- ASAM Standard Examples ---
%   run_intersection('LaneChange')          % Polynomial lane change maneuver
%   run_intersection('Overtake')            % Overtake slow lead vehicle
%   run_intersection('PedCrossing')         % Pedestrian crossing road
%
% Modes (pass anywhere as an argument):
%   'app'  - Driving Scenario Designer App   (e.g. run_intersection('scenic', 'app'))
%   '3d'   - Unreal Engine 3D Co-Simulation (e.g. run_intersection('scenic', '3d'))
%   'animate' - Interactive 2D Bird's-Eye display with live HUD (default).
%   'headless' - Fast headless batch simulation.
% =========================================================================

    thisDir = fileparts(mfilename('fullpath'));
    if isempty(thisDir), thisDir = pwd; end

    % Add modular scenario paths to MATLAB search path
    addpath(fullfile(thisDir, 'matlab'));
    addpath(fullfile(thisDir, 'maps_xodr'));
    addpath(fullfile(thisDir, 'verification'));
    addpath(fullfile(thisDir, '..', 'vehicle_dynamics'));

    scenario = run_intersection_scenario(varargin{:});
end
