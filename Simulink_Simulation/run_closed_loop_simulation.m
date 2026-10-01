%% RUN_CLOSED_LOOP_SIMULATION
% =========================================================================
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% Unified Closed-Loop Autonomous Simulation Runner
%
% Executes the complete 5-subsystem autonomous stack in closed loop:
%   1. RoadRunner 3D Environment / OpenSCENARIO Ground Truth
%   2. C3 YOLOv8s Perception + Occlusion Frustum
%   3. Multi-Model Semantic IMM Sensor Fusion Tracker
%   4. MotionFormer Multi-Modal Trajectory Prediction (K=3 GMM Modes)
%   5. Frenet Optimal Spatiotemporal Replanner + Stateflow Supervisor
%   6. Level-4 Vehicle Dynamics (A-PP + MPC + Kinematic ACC/AEB)
%
% Evaluates across canonical Indian road scenarios:
%   - Indian_WrongWay_Encounter
%   - NCAP_AEB_VRU_CPNCO (Obstructed Child)
%   - Indian_AutoRickshaw_CutIn
%   - Indian_StrayCattle_Hazard
%   - Indian_Pothole_Detour
% =========================================================================

function results = run_closed_loop_simulation(scenarioName, durationSec, enablePlot)
    if nargin < 1 || isempty(scenarioName)
        scenarioName = 'WrongWay'; % Default: Challenging Indian Wrong-Way encounter
    end
    if nargin < 2 || isempty(durationSec)
        durationSec = 25.0; % 25s scenario time
    end
    if nargin < 3 || isempty(enablePlot)
        enablePlot = true;
    end

    scriptDir = fileparts(mfilename('fullpath'));
    rootDir = fileparts(scriptDir);

    addpath(rootDir);
    addpath(fullfile(rootDir, 'main'));
    addpath(fullfile(rootDir, 'Vehicle_dynamics'));
    addpath(fullfile(rootDir, 'Path_planning_decision'));
    addpath(fullfile(rootDir, 'Trajectory'));
    addpath(fullfile(rootDir, 'Sensor_fusion'));
    addpath(fullfile(rootDir, 'Perception'));
    addpath(fullfile(rootDir, 'scenarios', 'matlab'));

    fprintf('========================================================================\n');
    fprintf('  RUNNING CLOSED-LOOP SIMULATION: %s\n', upper(scenarioName));
    fprintf('  Duration: %.1fs | Fixed-Step Multi-Rate Clocks Active\n', durationSec);
    fprintf('========================================================================\n\n');

    % 1. Map Scenario Name to XOSC & XODR files
    mapsDir = fullfile(rootDir, 'scenarios', 'maps_xodr');
    scenDir = fullfile(rootDir, 'scenarios', 'scenarios_xosc');
    ncapDir = fullfile(scenDir, 'ncap');
    indDir  = fullfile(scenDir, 'indian');

    xodrFile = fullfile(mapsDir, 'X-Intersection_NCAP.xodr');

    switch upper(scenarioName)
        case {'WRONGWAY', 'WRONG_WAY', 'INDIAN_WRONGWAY'}
            xoscFile = fullfile(indDir, 'Indian_WrongWay_Encounter.xosc');
        case {'CPNCO', 'CHILD', 'NCAP_CPNCO'}
            xoscFile = fullfile(ncapDir, 'NCAP_AEB_VRU_CPNCO_2023.xosc');
        case {'AUTOCUTIN', 'RICKSHAW', 'AUTO'}
            xoscFile = fullfile(indDir, 'Indian_AutoRickshaw_CutIn.xosc');
        case {'CATTLE', 'COW', 'STRAYCATTLE'}
            xoscFile = fullfile(indDir, 'Indian_StrayCattle_Hazard.xosc');
        case {'POTHOLE', 'POTHOLES'}
            xoscFile = fullfile(indDir, 'Indian_Pothole_Detour.xosc');
        otherwise
            xoscFile = fullfile(indDir, 'Indian_WrongWay_Encounter.xosc');
    end

    % 2. Import Scenario via ASAM OpenSCENARIO Importer
    sampleTime = 0.02; % 50 Hz base rate for scenario
    [scenario, info] = import_openscenario(xoscFile, xodrFile, ...
        'SampleTime', sampleTime, 'StopTime', durationSec, 'EnableAEB', true, ...
        'LiberateEgo', true);

    ego = scenario.Actors(1);
    sensorRig = sensor_rig_builder(scenario, ego);

    % 3. Extract Route Reference Waypoints
    if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'EgoWaypoints') && ~isempty(info.ActorMap.EgoWaypoints)
        rawWps = info.ActorMap.EgoWaypoints(:, 1:2);
    else
        rawWps = [ego.Position(1), ego.Position(2);
                  ego.Position(1)+50, ego.Position(2);
                  ego.Position(1)+100, ego.Position(2);
                  ego.Position(1)+250, ego.Position(2)];
    end
    refWps = interpolate_ref_path(rawWps, 0.5);
    cruiseSpeed = 6.94; % 25 km/h

    % 4. Instantiate Unified Autonomous AV Stack
    avStack = AutonomousAVStack(refWps, cruiseSpeed, sampleTime, 'PurePursuit');
    if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'Potholes') && ~isempty(info.ActorMap.Potholes)
        avStack.Potholes = info.ActorMap.Potholes;
    end
    avStack.init_scenario(scenario, ego, sensorRig);

    % 5. Execute Multi-Rate Closed-Loop Step Loop
    maxSteps = ceil(durationSec / sampleTime);
    logTime     = zeros(maxSteps, 1);
    logEgoX     = zeros(maxSteps, 1);
    logEgoY     = zeros(maxSteps, 1);
    logEgoYaw   = zeros(maxSteps, 1);
    logEgoSpeed = zeros(maxSteps, 1);
    logSteer    = zeros(maxSteps, 1);
    logAccel    = zeros(maxSteps, 1);
    logLatErr   = zeros(maxSteps, 1);
    logMinTTC   = zeros(maxSteps, 1);
    logModes    = cell(maxSteps, 1);

    stepIdx = 0;
    minObsClearance = Inf;
    numCollisions   = 0;

    fprintf('>> Beginning closed-loop execution...\n');
    ticSim = tic;

    while advance(scenario)
        stepIdx = stepIdx + 1;
        if stepIdx > maxSteps, break; end
        tNow = scenario.SimulationTime;

        % Step full AV stack
        telem = avStack.step_scenario(scenario, ego, sensorRig, tNow);

        logTime(stepIdx)     = tNow;
        logEgoX(stepIdx)     = telem.Position(1);
        logEgoY(stepIdx)     = telem.Position(2);
        logEgoYaw(stepIdx)   = telem.Yaw;
        logEgoSpeed(stepIdx) = telem.Speed;
        logSteer(stepIdx)    = telem.Steering;
        logAccel(stepIdx)    = telem.Accel;
        logLatErr(stepIdx)   = telem.LateralError;
        logMinTTC(stepIdx)   = telem.TTC;
        logModes{stepIdx}    = telem.StateflowMode;

        % Proximity clearance audit against all scenario actors
        for a = 2:numel(scenario.Actors)
            act = scenario.Actors(a);
            dx_a = act.Position(1) - telem.Position(1);
            dy_a = act.Position(2) - telem.Position(2);
            d_actor = hypot(dx_a, dy_a);
            
            % Account for bounding box dimensions
            clearance = d_actor - (ego.Length + act.Length)/2;
            if clearance < minObsClearance
                minObsClearance = clearance;
            end
            if clearance < -0.10 && abs(dx_a) < (ego.Length+act.Length)/2 && abs(dy_a) < (ego.Width+act.Width)/2
                numCollisions = numCollisions + 1;
            end
        end

        if mod(stepIdx, 100) == 0
            fprintf('  Step %d/%d | t=%.2fs | Speed=%.1f km/h | Mode=%-10s | MinClearance=%.2fm\n', ...
                stepIdx, maxSteps, tNow, telem.Speed * 3.6, telem.StateflowMode, minObsClearance);
        end
    end

    totalSimTime = toc(ticSim);
    fprintf('\n>> Simulation completed in %.2fs (%.1fx real-time).\n', totalSimTime, (stepIdx * sampleTime) / totalSimTime);

    % Trim unused log rows
    logTime     = logTime(1:stepIdx);
    logEgoX     = logEgoX(1:stepIdx);
    logEgoY     = logEgoY(1:stepIdx);
    logEgoYaw   = logEgoYaw(1:stepIdx);
    logEgoSpeed = logEgoSpeed(1:stepIdx);
    logSteer    = logSteer(1:stepIdx);
    logAccel    = logAccel(1:stepIdx);
    logLatErr   = logLatErr(1:stepIdx);
    logMinTTC   = logMinTTC(1:stepIdx);
    logModes    = logModes(1:stepIdx);

    % Pack results struct
    results = struct();
    results.ScenarioName    = scenarioName;
    results.Time            = logTime;
    results.EgoX            = logEgoX;
    results.EgoY            = logEgoY;
    results.EgoYaw          = logEgoYaw;
    results.EgoSpeed        = logEgoSpeed;
    results.SteerCmd        = logSteer;
    results.AccelCmd        = logAccel;
    results.LateralError    = logLatErr;
    results.MinTTC          = logMinTTC;
    results.StateflowModes  = logModes;
    results.MinClearance    = minObsClearance;
    results.NumCollisions   = numCollisions;
    results.Success         = (numCollisions == 0);

    % Print Benchmark Verdict
    fprintf('========================================================================\n');
    fprintf('  CLOSED-LOOP SIMULATION VERDICT: %s\n', upper(scenarioName));
    fprintf('  Total Collisions: %d | Minimum Obstacle Clearance: %.2fm\n', numCollisions, minObsClearance);
    fprintf('  Peak Speed: %.1f km/h | Final Speed: %.1f km/h\n', max(logEgoSpeed)*3.6, logEgoSpeed(end)*3.6);
    fprintf('  Max Lateral Error: %.3fm | Status: %s\n', max(abs(logLatErr)), ...
        ternary(results.Success, 'PASS - 100% COLLISION FREE', 'FAIL - COLLISION DETECTED'));
    fprintf('========================================================================\n\n');

    % Plot summary figure if requested
    if enablePlot
        fig = figure('Name', ['Closed-Loop Simulation: ', scenarioName], 'Position', [100, 100, 1200, 750], 'Color', 'w');
        
        subplot(2, 2, 1);
        plot(logTime, logEgoSpeed * 3.6, 'b-', 'LineWidth', 1.8); grid on;
        xlabel('Time (s)'); ylabel('Speed (km/h)'); title('Ego Speed Profile');
        
        subplot(2, 2, 2);
        plot(logTime, rad2deg(logSteer), 'r-', 'LineWidth', 1.6); grid on;
        xlabel('Time (s)'); ylabel('Steer Angle (deg)'); title('Front Wheel Steering Command');
        
        subplot(2, 2, 3);
        plot(logTime, logLatErr, 'm-', 'LineWidth', 1.5); grid on;
        xlabel('Time (s)'); ylabel('Lateral Error e_y (m)'); title('Cross-Track Tracking Error');
        
        subplot(2, 2, 4);
        plot(logEgoX, logEgoY, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 6); hold on; grid on;
        yline(0.0, 'k--', 'LineWidth', 1.5, 'Label', 'Road Centerline Y=0');
        yline(-0.40, 'r--', 'LineWidth', 1.2, 'Label', 'Center Divider Limit');
        yline(-1.75, 'g:', 'LineWidth', 1.0, 'Label', 'Cruising Lane -1');
        yline(-5.00, 'c:', 'LineWidth', 1.0, 'Label', 'Passing Lane -2');
        xlabel('X (m)'); ylabel('Y (m)'); title('Closed-Loop Trajectory (Bird''s-Eye View)');
        axis equal;
        
        saveas(fig, fullfile(scriptDir, [scenarioName, '_closed_loop_result.png']));
    end
end

function val = ternary(cond, trueVal, falseVal)
    if cond, val = trueVal; else, val = falseVal; end
end
