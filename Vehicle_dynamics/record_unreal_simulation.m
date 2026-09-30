function [videoPath, scenario, runDir] = record_unreal_simulation(varargin)
% RECORD_UNREAL_SIMULATION  Two-Phase Pre-compute & Smooth Playback Pipeline
% =========================================================================
% Executes an autonomous driving scenario in TWO PHASES to guarantee
% lag-free 30 FPS Unreal Engine 3D video and complete sensor logging.
%
% PHASE 1 (Pre-compute, Headless):
%   - Imports the OpenSCENARIO scenario
%   - Runs the full closed-loop autonomous controller + sensor fusion
%   - simulation_logger captures 4 camera images + IPM BEV at 0.5s intervals
%   - Records all ego actor poses (X, Y, Z, Yaw, Speed) for trajectory baking
%   - Exports graphs, CSVs, LiDAR scans, and metadata to the log folder
%
% PHASE 2 (Smooth Unreal Playback + Video Recording):
%   - Restarts scenario and bakes pre-computed trajectory into ego actor
%   - Launches Unreal Engine 3D viewer (plotSim3d)
%   - Pre-warms GPU shaders for 5 seconds (zero CPU load)
%   - Starts Python video recorder at 30 FPS (unreal_video_recorder.py)
%   - Steps through scenario at real-time pace (ZERO control computation)
%   - Waits for recorder to flush and verifies MP4 output
%
% All outputs (video, images, graphs, sensor data) are in ONE log folder:
%   logs/run_YYYYMMDD_HHMMSS_<Scenario>_<Controller>/
%
% Syntax:
%   record_unreal_simulation
%   record_unreal_simulation('CPNCO', 'pp', 30.0)
%   record_unreal_simulation('CPNCO', 'MPC', 15.0)
% =========================================================================

    thisDir = fileparts(mfilename('fullpath'));
    rootDir = fileparts(thisDir);

    addpath(fullfile(rootDir, 'scenarios'));
    addpath(fullfile(rootDir, 'scenarios', 'matlab'));
    addpath(fullfile(rootDir, 'vehicle_dynamics'));
    addpath(fullfile(rootDir, 'perception'));
    addpath(fullfile(rootDir, 'sensor_fusion'));

    % =====================================================================
    % Parse Inputs (flexible order, same as run_instruction)
    % =====================================================================
    scenarioName = 'CPNCO';
    controllerType = 'PurePursuit';
    durationVal = 30.0;

    for i = 1:nargin
        arg = varargin{i};
        if ischar(arg) || isstring(arg)
            carg = char(arg);
            su = upper(carg);
            switch su
                case {'PP', 'PUREPURSUIT', 'PURSUIT'}
                    controllerType = 'PurePursuit';
                case {'MPC', 'NMPC'}
                    controllerType = 'MPC';
                otherwise
                    if ~isnan(str2double(carg))
                        durationVal = str2double(carg);
                    else
                        scenarioName = carg;
                    end
            end
        elseif isnumeric(arg) && isscalar(arg)
            durationVal = double(arg);
        end
    end

    sampleTime = 0.05; % 20 Hz update rate

    fprintf('============================================================\n');
    fprintf('  UNREAL ENGINE 3D SIMULATION - TWO-PHASE PIPELINE\n');
    fprintf('============================================================\n');
    fprintf('  Scenario   : %s\n', scenarioName);
    fprintf('  Controller : %s\n', controllerType);
    fprintf('  Duration   : %.1f seconds\n', durationVal);
    fprintf('  Video FPS  : 30\n');
    fprintf('  Pipeline   : Phase 1 (Headless Pre-compute) + Phase 2 (Smooth Record)\n');
    fprintf('============================================================\n\n');

    % =====================================================================
    % Map Scenario Files
    % =====================================================================
    mapsDir = fullfile(rootDir, 'scenarios', 'maps_xodr');
    indianDir = fullfile(rootDir, 'scenarios', 'scenarios_xosc', 'indian');
    ncapDir = fullfile(rootDir, 'scenarios', 'scenarios_xosc', 'ncap');
    asamDir = fullfile(rootDir, 'scenarios', 'scenarios_xosc', 'asam_examples');

    scUpper = upper(strtrim(string(scenarioName)));
    if isfile(scenarioName)
        xoscFile = char(scenarioName);
        xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
    else
        switch scUpper
            % --- Indian Mixed Traffic Scenarios ---
            case {'POTHOLE', 'INDIAN_POTHOLE', 'DETOUR', 'INDIAN_POTHOLE_DETOUR', 'PINCH', 'INDIAN6'}
                xoscFile = fullfile(indianDir, 'Indian_Pothole_Detour.xosc');
                xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
            case {'CONGESTION', 'INDIAN_CONGESTION', 'JAM', 'INDIAN_TRAFFIC_CONGESTION', 'QUEUE', 'INDIAN5'}
                xoscFile = fullfile(indianDir, 'Indian_Traffic_Congestion.xosc');
                xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
            case {'INDIAN_AUTOCUTIN', 'AUTOCUTIN', 'AUTORICKSHAW', 'AUTO', 'INDIAN1'}
                xoscFile = fullfile(indianDir, 'Indian_AutoRickshaw_CutIn.xosc');
                xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
            case {'INDIAN_TWOWHEELER', 'TWOWHEELER', 'MOTORCYCLE', 'BIKE', 'INDIAN2'}
                xoscFile = fullfile(indianDir, 'Indian_TwoWheeler_LaneFilter.xosc');
                xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
            case {'INDIAN_JAYWALK', 'JAYWALK', 'PEDESTRIAN', 'INDIAN3'}
                xoscFile = fullfile(indianDir, 'Indian_Pedestrian_Jaywalk.xosc');
                xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
            case {'INDIAN_CATTLE', 'CATTLE', 'COW', 'HAZARD', 'INDIAN4'}
                xoscFile = fullfile(indianDir, 'Indian_StrayCattle_Hazard.xosc');
                xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');

            % --- ASAM OpenSCENARIO Standard Examples ---
            case {'LANECHANGE', 'LANECHANGESIMPLE'}
                xoscFile = fullfile(asamDir, 'LaneChangeSimple.xosc');
                xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
            case {'OVERTAKE', 'OVERTAKESLOWVEHICLE'}
                xoscFile = fullfile(asamDir, 'OvertakeSlowVehicle.xosc');
                xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
            case {'PEDCROSSING', 'PEDESTRIANCROSSING'}
                xoscFile = fullfile(asamDir, 'PedestrianCrossing.xosc');
                xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');

            % --- Euro NCAP Standard Test Protocols ---
            case {'CPNCO', '1'}
                xoscFile = fullfile(ncapDir, 'NCAP_AEB_VRU_CPNCO_2023.xosc');
                xodrFile = fullfile(mapsDir, 'X-Intersection_NCAP.xodr');
            case {'CPTA', '2'}
                xoscFile = fullfile(ncapDir, 'NCAP_AEB_VRU_CPTA_2023.xosc');
                xodrFile = fullfile(mapsDir, 'X-Intersection_NCAP.xodr');
            case {'CCFTAP', '3'}
                xoscFile = fullfile(ncapDir, 'NCAP_AEB_C2C_CCFtap_2023.xosc');
                xodrFile = fullfile(mapsDir, 'X-Intersection_NCAP.xodr');
            case {'CCCSCP', '4'}
                xoscFile = fullfile(ncapDir, 'CCCscp.xosc');
                xodrFile = fullfile(mapsDir, 'X-Intersection_NCAP.xodr');
            case 'CBFA'
                xoscFile = fullfile(ncapDir, 'NCAP_AEB_VRU_CBFA_2023.xosc');
                xodrFile = fullfile(mapsDir, 'X-Intersection_NCAP.xodr');
            case 'CCR'
                xoscFile = fullfile(ncapDir, 'NCAP_AEB_C2C_CCR_2023.xosc');
                xodrFile = fullfile(mapsDir, 'X-Intersection_NCAP.xodr');

            otherwise
                % Substring fallback matching
                if contains(scUpper, 'POTHOLE') || contains(scUpper, 'DETOUR') || contains(scUpper, 'PINCH')
                    xoscFile = fullfile(indianDir, 'Indian_Pothole_Detour.xosc');
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
                elseif contains(scUpper, 'CONGEST') || contains(scUpper, 'JAM') || contains(scUpper, 'QUEUE')
                    xoscFile = fullfile(indianDir, 'Indian_Traffic_Congestion.xosc');
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
                elseif contains(scUpper, 'AUTO') || contains(scUpper, 'RICKSHAW')
                    xoscFile = fullfile(indianDir, 'Indian_AutoRickshaw_CutIn.xosc');
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
                elseif contains(scUpper, 'TWOWHEEL') || contains(scUpper, 'MOTORCYCLE') || contains(scUpper, 'BIKE')
                    xoscFile = fullfile(indianDir, 'Indian_TwoWheeler_LaneFilter.xosc');
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
                elseif contains(scUpper, 'JAYWALK') || contains(scUpper, 'PEDESTRIAN')
                    xoscFile = fullfile(indianDir, 'Indian_Pedestrian_Jaywalk.xosc');
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
                elseif contains(scUpper, 'CATTLE') || contains(scUpper, 'COW')
                    xoscFile = fullfile(indianDir, 'Indian_StrayCattle_Hazard.xosc');
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
                else
                    % Safe default
                    xoscFile = fullfile(ncapDir, 'NCAP_AEB_VRU_CPNCO_2023.xosc');
                    xodrFile = fullfile(mapsDir, 'X-Intersection_NCAP.xodr');
                end
        end
    end

    % =====================================================================
    % Import Scenario
    % =====================================================================
    [scenario, info] = import_openscenario(xoscFile, xodrFile, ...
        'SampleTime', sampleTime, 'StopTime', durationVal, 'EnableAEB', true, ...
        'LiberateEgo', true);

    egoActor = scenario.Actors(1);

    % Build reference waypoints for lateral controller
    if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'EgoWaypoints') && ~isempty(info.ActorMap.EgoWaypoints)
        rawWps = info.ActorMap.EgoWaypoints(:, 1:2);
    else
        rawWps = [egoActor.Position(1), egoActor.Position(2);
                  egoActor.Position(1)+50, egoActor.Position(2);
                  egoActor.Position(1)+100, egoActor.Position(2);
                  egoActor.Position(1)+250, egoActor.Position(2)];
    end
    refWps = interpolate_waypoints(rawWps, 0.5);
    vCruise = norm(egoActor.Velocity(1:2));
    if vCruise < 2.0 || vCruise > 7.5
        vCruise = 6.94; % 25 km/h steady smooth pace
    end

    % Build sensor rig, fusion bridge, and autonomous controller
    sensorRig = sensor_rig_builder(scenario, egoActor);
    fusionBridge = sensor_fusion_bridge(sensorRig, sampleTime);
    ctrl = autonomous_ego_controller(refWps, fusionBridge, controllerType, vCruise, sampleTime);
    ctrl.init_state(egoActor);

    % Initialize simulation logger (creates timestamped log folder)
    mainLogger = simulation_logger(scenario, scenarioName, controllerType, 0.5);
    videoPath = fullfile(mainLogger.RunDir, 'unreal_simulation_video.mp4');

    % =====================================================================
    % PHASE 1: Pre-compute Closed-Loop Simulation (Headless, No Unreal)
    % =====================================================================
    fprintf('[PHASE 1] ===================================================\n');
    fprintf('[PHASE 1] Pre-computing closed-loop simulation headlessly...\n');
    fprintf('[PHASE 1] Controller: %s | Cruise: %.1f km/h | Duration: %.1f s\n', ...
        controllerType, vCruise * 3.6, durationVal);

    % Add off-road anchor vehicle to guarantee scenario advances for the full durationVal
    anchor = vehicle(scenario, 'ClassID', 1, 'Position', [2000, 2000, 0]);
    trajectory(anchor, [2000, 2000, 0; 2000, 2000.01, 0], [0; durationVal]);

    maxSteps = ceil(durationVal / sampleTime) + 100;
    allPos  = zeros(maxSteps, 3);
    allSpd  = zeros(maxSteps, 1);
    allTime = zeros(maxSteps, 1);
    stepIdx = 0;

    tic;
    while advance(scenario)
        stepIdx = stepIdx + 1;
        if stepIdx > maxSteps, break; end
        tSim = scenario.SimulationTime;

        % Run autonomous sense-plan-act cycle
        telem = ctrl.step(egoActor, tSim);

        % Dynamic TTC-based VRU Trigger with baseline start guard
        if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'VRU') && ~info.ActorMap.VRUTriggered
            egoSpeedNow = max(0.1, telem.Speed);
            distToCross = info.ActorMap.VRUWaypoints(1,1) - telem.Position(1);
            ttcCondition = distToCross > 0 && (distToCross / egoSpeedNow) <= info.ActorMap.VRUCrossTime;
            if ttcCondition
                if tSim < info.ActorMap.VRUBaselineStart
                    trajectory(info.ActorMap.VRU, info.ActorMap.VRUWaypoints, info.ActorMap.VRUSpeeds);
                end
                info.ActorMap.VRUTriggered = true;
            end
        end

        % Log cameras, sensors, and telemetry at 0.5s intervals
        mainLogger.step(scenario, egoActor, telem, sensorRig, tSim);

        % Record ego pose for trajectory baking
        allPos(stepIdx, :) = [telem.Position(1), telem.Position(2), 0.0];
        allSpd(stepIdx) = max(0.01, telem.Speed);
        allTime(stepIdx) = tSim;
    end
    phase1Time = toc;

    % Finalize logger: export graphs, CSVs, metadata
    mainLogger.finalize();

    fprintf('[PHASE 1] Complete: %d steps in %.1f seconds (%.0fx faster than real-time)\n', ...
        stepIdx, phase1Time, allTime(max(1,stepIdx)) / max(0.001, phase1Time));
    fprintf('[PHASE 1] Camera images : %s\n', mainLogger.ImageDir);
    fprintf('[PHASE 1] Graphs        : %s\n', mainLogger.GraphDir);
    fprintf('[PHASE 1] Sensor data   : %s\n', mainLogger.SensorDataDir);

    % Close offscreen rendering canvas to free GPU memory for Unreal
    try
        if ~isempty(mainLogger.OffscreenFig) && isvalid(mainLogger.OffscreenFig)
            close(mainLogger.OffscreenFig);
        end
    catch
    end

    % Trim arrays
    allPos  = allPos(1:stepIdx, :);
    allSpd  = allSpd(1:stepIdx);
    allTime = allTime(1:stepIdx);

    % =====================================================================
    % Downsample to Trajectory Keyframes
    % =====================================================================
    keepMask = false(stepIdx, 1);
    keepMask(1) = true;
    keepMask(end) = true;
    % Keep every 0.5s checkpoint
    keepMask(mod(round(allTime * 20), 10) == 0) = true;
    % Keep speed inflection points (braking / acceleration transitions)
    dSpd = [0; diff(allSpd)];
    keepMask(abs(dSpd) > 0.10) = true;

    keyWps  = allPos(keepMask, :);
    keySpds = allSpd(keepMask);

    tol = 1e-6;
    dedupe = true(size(keyWps, 1), 1);
    for k = 2:size(keyWps, 1)
        if all(abs(keyWps(k,:) - keyWps(k-1,:)) < tol)
            dedupe(k) = false;   % drop the redundant row; keep the first speed
        end
    end
    keyWps  = keyWps(dedupe, :);
    keySpds = keySpds(dedupe);


    fprintf('[PHASE 1] Baked %d trajectory keyframes for smooth Unreal playback.\n\n', size(keyWps, 1));

    % =====================================================================
    % PHASE 2: Smooth Unreal Engine Playback + 30 FPS Video Recording
    % =====================================================================
    fprintf('[PHASE 2] ===================================================\n');
    fprintf('[PHASE 2] Launching Unreal Engine for smooth 30 FPS recording...\n');

    % Restart scenario and bake pre-computed trajectory into ego actor
    restart(scenario);
    if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'VRUTriggered')
        info.ActorMap.VRUTriggered = false;
    end
    trajectory(egoActor, keyWps, keySpds);
    fprintf('[PHASE 2] Ego trajectory baked (%d keyframes). All CPU-heavy work is done.\n', size(keyWps, 1));

    % Launch Unreal Engine 3D viewer
    fprintf('[PHASE 2] Initializing Unreal Engine 3D viewer (plotSim3d)...\n');
    try
        plotSim3d(scenario, 'ActorInFocus', 1);
        fprintf('[PHASE 2] Unreal Engine 3D viewer launched successfully.\n');
    catch ME3D
        warning('[PHASE 2] plotSim3d failed: %s', ME3D.message);
        fprintf('[PHASE 2] Skipping video recording. Phase 1 outputs are still available.\n');
        return;
    end

    % Pre-warm GPU: let Unreal compile shaders and stream textures
    fprintf('[PHASE 2] Pre-warming GPU shaders and textures (5 seconds)...\n');
    pause(5.0);

    % Launch Python video recorder in background at 30 FPS
    startFile  = fullfile(mainLogger.RunDir, 'recorder_started.txt');
    signalFile = fullfile(mainLogger.RunDir, 'recorder_done.txt');
    stopFile   = fullfile(mainLogger.RunDir, 'stop_recording.txt');
    logFile    = fullfile(mainLogger.RunDir, 'unreal_video_recorder.log');
    batFile    = fullfile(mainLogger.RunDir, 'run_recorder.bat');
    recorderPy = fullfile(thisDir, 'unreal_video_recorder.py');
    if ~isfile(recorderPy)
        recorderPy = fullfile(rootDir, 'vehicle_dynamics', 'unreal_video_recorder.py');
    end
    if ~isfile(recorderPy)
        recorderPy = fullfile(rootDir, 'unreal_video_recorder.py');
    end
    recDuration = durationVal + 40.0; % Extra margin (recorder will stop via stop-file)

    % Clean any leftover signal/stop/log files from previous runs
    if isfile(startFile),  delete(startFile);  end
    if isfile(signalFile), delete(signalFile); end
    if isfile(stopFile),   delete(stopFile);   end
    if isfile(logFile),    delete(logFile);    end

    % Resolve Python executable (prefer conda sih26 environment)
    pythonExe = 'D:\conda_envs\sih26\python.exe';
    if ~isfile(pythonExe), pythonExe = 'C:\Users\SAHIL\anaconda3\python.exe'; end
    if ~isfile(pythonExe), pythonExe = 'python'; end

    % Generate clean batch runner with -u (unbuffered) to guarantee immediate logging
    fidBat = fopen(batFile, 'w');
    if fidBat ~= -1
        fprintf(fidBat, '@echo off\r\n');
        fprintf(fidBat, '"%s" -u "%s" --output "%s" --duration %.1f --fps 30 --start-file "%s" --signal-file "%s" --stop-file "%s" > "%s" 2>&1\r\n', ...
            pythonExe, recorderPy, videoPath, recDuration, startFile, signalFile, stopFile, logFile);
        fclose(fidBat);
    end

    % Launch background recorder via batch runner
    system(sprintf('start "" /MIN "%s"', batFile));
    fprintf('[PHASE 2] Video recorder launched (30 FPS, duration: %.0fs).\n', recDuration);

    % Wait for recorder to signal that it is attached and actively capturing
    fprintf('[PHASE 2] Waiting for video recorder to attach and begin capturing (up to 90s for Unreal boot)...\n');
    tStartWait = tic;
    recorderReady = false;
    while toc(tStartWait) < 90.0
        if isfile(startFile)
            fprintf('[PHASE 2] Video recorder is ACTIVE and capturing (ready in %.1fs)! Commencing scenario playback.\n', toc(tStartWait));
            recorderReady = true;
            break;
        end
        pause(0.2);
    end

    if ~recorderReady
        warning('[PHASE 2] Recorder did not signal ready within 90s. Proceeding with playback anyway.');
    end

    % Step through scenario at real-time pace (ZERO control computation)
    fprintf('[PHASE 2] Playing back pre-computed trajectory smoothly...\n');
    playSteps = 0;
    while advance(scenario)
        playSteps = playSteps + 1;
        tPlay = scenario.SimulationTime;

        % Dynamic TTC-based VRU Trigger during playback
        if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'VRU') && ~info.ActorMap.VRUTriggered
            egoSpeedNow = max(0.1, norm(egoActor.Velocity(1:2)));
            distToCross = info.ActorMap.VRUWaypoints(1,1) - egoActor.Position(1);
            ttcCondition = distToCross > 0 && (distToCross / egoSpeedNow) <= info.ActorMap.VRUCrossTime;
            if ttcCondition
                if tPlay < info.ActorMap.VRUBaselineStart
                    trajectory(info.ActorMap.VRU, info.ActorMap.VRUWaypoints, info.ActorMap.VRUSpeeds);
                end
                info.ActorMap.VRUTriggered = true;
            end
        end

        pause(sampleTime); % Real-time pacing for smooth Unreal rendering
    end
    fprintf('[PHASE 2] Playback complete: %d steps (%.1f seconds of simulation).\n', ...
        playSteps, playSteps * sampleTime);

    % Allow 2 extra seconds for the recorder to capture the final scene
    pause(2.0);

    % Signal the recorder to stop gracefully (so it can call writer.release())
    fprintf('[PHASE 2] Sending stop signal to video recorder...\n');
    fid = fopen(stopFile, 'w');
    if fid ~= -1
        fprintf(fid, 'stop\n');
        fclose(fid);
    end

    % Wait for recorder to confirm completion via signal file (up to 45 seconds)
    fprintf('[PHASE 2] Waiting for video recorder to finalize (moov atom)...\n');
    maxWait = 45.0;
    waitStart = tic;
    videoFinalized = false;
    while toc(waitStart) < maxWait
        if isfile(signalFile)
            fprintf('[PHASE 2] Recorder confirmed completion via signal file.\n');
            videoFinalized = true;
            break;
        end
        pause(0.3);
    end

    if ~videoFinalized
        fprintf('[PHASE 2] Timeout waiting for recorder signal. Giving extra flush time...\n');
        pause(5.0);
    end

    % Print recorder log output for debugging visibility
    if isfile(logFile)
        try
            logTxt = fileread(logFile);
            fprintf('\n--- UNREAL RECORDER LOG ---\n%s\n---------------------------\n\n', strtrim(logTxt));
        catch
        end
    end

    % Clean up signal and stop files
    if isfile(startFile),  try delete(startFile);  catch, end, end
    if isfile(signalFile), try delete(signalFile); catch, end, end
    if isfile(stopFile),   try delete(stopFile);   catch, end, end

    % =====================================================================
    % Verify & Report Results
    % =====================================================================
    fprintf('\n============================================================\n');
    if isfile(videoPath)
        d = dir(videoPath);
        fprintf('  SUCCESS: Unreal Engine 3D Simulation Complete!\n');
        fprintf('============================================================\n');
        fprintf('  Session Dir    : %s\n', mainLogger.RunDir);
        fprintf('  Video          : %s\n', videoPath);
        fprintf('  Video Size     : %.1f KB\n', d.bytes / 1024);
        fprintf('  Camera Images  : %s\n', mainLogger.ImageDir);
        fprintf('  BEV Composites : %s\n', mainLogger.BevDir);
        fprintf('  Graphs         : %s\n', mainLogger.GraphDir);
        fprintf('  Sensor Data    : %s\n', mainLogger.SensorDataDir);
        fprintf('============================================================\n');
    else
        fprintf('  WARNING: Video file was not created.\n');
        fprintf('============================================================\n');
        fprintf('  Possible causes:\n');
        fprintf('    - Python (with cv2, psutil) not available in PATH\n');
        fprintf('    - AutoVrtlEnv window did not appear in time\n');
        fprintf('    - Unreal Engine support package not installed\n');
        fprintf('  Phase 1 outputs (images, graphs, sensor data) are still saved at:\n');
        fprintf('    %s\n', mainLogger.RunDir);
        fprintf('============================================================\n');
    end

    % Convenience copy to project root
    try
        convCopy = fullfile(rootDir, 'unreal_simulation_video.mp4');
        if isfile(videoPath)
            dVid = dir(videoPath);
            if ~isempty(dVid) && dVid.bytes > 1000
                copyfile(videoPath, convCopy);
                fprintf('  Also copied to: %s (%.1f KB)\n', convCopy, dVid.bytes / 1024);
            end
        end
    catch
    end

    runDir = mainLogger.RunDir;
end

% =========================================================================
% Helper: Interpolate sparse waypoints to dense path
% =========================================================================
function dense = interpolate_waypoints(pts, ds)
    diffs = diff(pts, 1, 1);
    segLens = hypot(diffs(:, 1), diffs(:, 2));
    keep = [true; segLens > 1e-4];
    pts = pts(keep, :);
    if size(pts, 1) < 2, dense = pts; return; end
    diffs = diff(pts, 1, 1);
    segLens = hypot(diffs(:, 1), diffs(:, 2));
    sCum = [0; cumsum(segLens)];
    totalLen = sCum(end);
    if totalLen < ds, dense = pts; return; end
    sQuery = (0:ds:totalLen)';
    xDense = interp1(sCum, pts(:, 1), sQuery, 'pchip');
    yDense = interp1(sCum, pts(:, 2), sQuery, 'pchip');
    dense = [xDense, yDense];
end
