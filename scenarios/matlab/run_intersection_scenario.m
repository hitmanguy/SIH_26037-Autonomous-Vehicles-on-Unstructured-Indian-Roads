function scenario = run_intersection_scenario(varargin)
% RUN_INTERSECTION_SCENARIO
% =========================================================================
% Imports and executes standard ASAM OpenSCENARIO (.xosc) and OpenDRIVE (.xodr)
% driving scenarios from Euro NCAP / OSC-NCAP in MATLAB Automated Driving Toolbox.
%
% Supported Scenarios:
%   1. 'CPNCO' (Default) - Car-to-Pedestrian Nearside Child Obstructed (AEB VRU 2023)
%   2. 'CPTA'            - Car-to-Pedestrian Turning Adult at Intersection (AEB VRU 2023)
%   3. 'CCFtap'          - Car-to-Car Front Turn Across Path (AEB C2C 2023)
%   4. 'CCCscp'          - Car-to-Car Straight Crossing Path at Intersection (CA-FC 2026)
%   5. Custom .xosc Path - Any arbitrary ASAM OpenSCENARIO XML file
%
% Syntax Examples:
%   run_intersection                    % Runs CPNCO in 2D bird's-eye view
%   run_intersection('3d')              % Runs CPNCO with Unreal Engine 3D co-simulation
%   run_intersection('app')             % Opens CPNCO in Driving Scenario Designer App
%   run_intersection('CPTA', '3d')      % Runs CPTA in Unreal Engine 3D
%   run_intersection('CPTA', 'app')     % Opens CPTA in Driving Scenario Designer App
%   run_intersection('CCFtap', '3d')    % Runs CCFtap in Unreal Engine 3D
%   run_intersection('CCCscp', 'app')   % Opens CCCscp in Driving Scenario Designer App
%   run_intersection(xoscPath, xodrPath, 'animate', 30.0, true)
% =========================================================================

    scriptDir = fileparts(mfilename('fullpath'));
    if isempty(scriptDir), scriptDir = pwd; end
    rootDir = fileparts(scriptDir); % D:\hackathon\SIH26\scenarios

    % Modular repository locations
    mapsDir = fullfile(rootDir, 'maps_xodr');
    scenDir = fullfile(rootDir, 'scenarios_xosc');
    ncapDir = fullfile(scenDir, 'ncap');
    indianDir = fullfile(scenDir, 'indian');
    asamDir = fullfile(scenDir, 'asam_examples');
    rootRepo = fileparts(rootDir);
    addpath(rootRepo);
    addpath(fullfile(rootRepo, 'main'));
    addpath(fullfile(rootRepo, 'vehicle_dynamics'));
    addpath(fullfile(rootRepo, 'perception'));
    addpath(fullfile(rootRepo, 'Sensor_fusion'));
    addpath(fullfile(rootRepo, 'Trajectory'));
    addpath(fullfile(rootRepo, 'Path_planning_decision'));

    defaultXodr = fullfile(mapsDir, 'X-Intersection_NCAP.xodr');
    defaultXosc = fullfile(ncapDir, 'NCAP_AEB_VRU_CPNCO_2023.xosc');

    % Defaults
    xoscFile = defaultXosc;
    xodrFile = defaultXodr;
    mode = 'animate';
    duration = 30.0;
    enable3D = false;
    isScenicFuzz = false;
    enableAutonomous = false;
    autoControllerType = 'PurePursuit';
    enableLog = false; % Default: false for fast app and simulation launch

    % Scan all arguments in varargin flexibly
    for i = 1:length(varargin)
        arg = varargin{i};
        if isempty(arg), continue; end

        if islogical(arg)
            enable3D = arg;
        elseif isnumeric(arg)
            if arg >= 1 && arg <= 4 && i == 1 && length(varargin) == 1
                switch arg
                    case 1, xoscFile = fullfile(ncapDir, 'NCAP_AEB_VRU_CPNCO_2023.xosc');
                    case 2, xoscFile = fullfile(ncapDir, 'NCAP_AEB_VRU_CPTA_2023.xosc');
                    case 3, xoscFile = fullfile(ncapDir, 'NCAP_AEB_C2C_CCFtap_2023.xosc');
                    case 4, xoscFile = fullfile(ncapDir, 'CCCscp.xosc');
                end
            else
                duration = double(arg);
            end
        elseif ischar(arg) || isstring(arg)
            s = char(arg);
            su = upper(s);
            switch su
                case {'APP', 'DESIGNER', 'DRIVINGSCENARIODESIGNER', 'OPEN_APP'}
                    mode = 'app';
                case {'3D', 'UNREAL', 'SIM3D'}
                    enable3D = true;
                case {'ANIMATE', 'GUI', '2D'}
                    mode = 'animate';
                case {'HEADLESS', 'BATCH', 'SILENT'}
                    mode = 'headless';

                % --- Autonomous Ego Controller Options ---
                case {'AUTONOMOUS', 'AUTONOMOUS_EGO', 'PUREPURSUIT', 'PP', 'PURSUIT'}
                    enableAutonomous = true;
                    autoControllerType = 'PurePursuit';
                case {'MPC', 'NMPC'}
                    enableAutonomous = true;
                    autoControllerType = 'MPC';

                % --- Logging Toggle Options ---
                case {'LOG', 'LOGGING', 'ENABLELOG', 'ENABLE_LOG', 'RECORD'}
                    enableLog = true;
                case {'NOLOG', 'NO_LOG', 'NOLOGGING', 'FAST', 'NOLOGS'}
                    enableLog = false;

                % --- UC Berkeley Scenic Probabilistic Fuzzing ---
                case {'SCENIC', 'FUZZ', 'MONTE_CARLO', 'RANDOM', 'SCENIC_SAMPLE'}
                    isScenicFuzz = true;
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');

                % --- Indian Mixed Traffic Scenarios ---
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
                case {'CONGESTION', 'INDIAN_CONGESTION', 'JAM', 'INDIAN_TRAFFIC_CONGESTION', 'QUEUE', 'INDIAN5'}
                    xoscFile = fullfile(indianDir, 'Indian_Traffic_Congestion.xosc');
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
                case {'POTHOLE', 'INDIAN_POTHOLE', 'DETOUR', 'INDIAN_POTHOLE_DETOUR', 'PINCH', 'INDIAN6'}
                    xoscFile = fullfile(indianDir, 'Indian_Pothole_Detour.xosc');
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
                case {'INDIAN_WRONGWAY', 'WRONGWAY', 'HEADON', 'TEMPO', 'INDIAN7'}
                    xoscFile = fullfile(indianDir, 'Indian_WrongWay_Encounter.xosc');
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
                case {'INDIAN_SCHOOLZONE', 'SCHOOLZONE', 'SCHOOL', 'CHILDREN', 'INDIAN8'}
                    xoscFile = fullfile(indianDir, 'Indian_SchoolZone_Rush.xosc');
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
                case {'INDIAN_BUSSTOP', 'BUSSTOP', 'BUS_STOP', 'ALIGHT', 'PASSENGER', 'INDIAN9'}
                    xoscFile = fullfile(indianDir, 'Indian_BusStop_Hazard.xosc');
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
                case {'INDIAN_VENDORCART', 'VENDORCART', 'VENDOR', 'HANDCART', 'CART', 'SWERVE', 'INDIAN10'}
                    xoscFile = fullfile(indianDir, 'Indian_VendorCart_Swerve.xosc');
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');
                case {'INDIAN_GAUNTLET', 'GAUNTLET', 'MULTITHREAT', 'MULTI_THREAT', 'BOSS', 'INDIAN11'}
                    xoscFile = fullfile(indianDir, 'Indian_MultiThreat_Gauntlet.xosc');
                    xodrFile = fullfile(mapsDir, 'Indian_Urban_Arterial.xodr');

                % --- Real Indian OpenStreetMap Roads ---
                case {'BANGALORE', 'INDIRANAGAR', 'BANGALORE_ROAD'}
                    xodrFile = fullfile(mapsDir, 'Bangalore_Indiranagar.xodr');

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
                    if endsWith(lower(s), '.xosc')
                        if isfile(s), xoscFile = s; else, error('Specified OpenSCENARIO file not found: %s', s); end
                    elseif endsWith(lower(s), '.xodr')
                        if isfile(s), xodrFile = s; else, error('Specified OpenDRIVE file not found: %s', s); end
                    elseif isfile(s)
                        [~,~,ext] = fileparts(s);
                        if strcmpi(ext, '.xodr')
                            xodrFile = s;
                        else
                            xoscFile = s;
                        end
                    else
                        warning('Unrecognized parameter: %s. Using default settings.', s);
                    end
            end
        end
    end

    tempCleanup = [];
    if isScenicFuzz
        fprintf('\n[SCENIC] Sampling probabilistic scene from UC Berkeley Scenic...\n');
        pythonExe = 'D:\\conda_envs\\sih26\\python.exe';
        bridgeScript = fullfile(rootDir, 'generators', 'sample_scenic_to_temp_xosc.py');
        [~, roadBase, roadExt] = fileparts(xodrFile);
        cmd = sprintf('"%s" "%s" --road "%s%s"', pythonExe, bridgeScript, roadBase, roadExt);
        [status, out] = system(cmd);
        if status ~= 0
            error('Failed to sample Scenic scenario:\n%s', out);
        end
        tokens = regexp(out, 'TEMP_XOSC:(.+)', 'tokens');
        if isempty(tokens)
            error('Scenic bridge did not return a valid temp XOSC path:\n%s', out);
        end
        xoscFile = strtrim(tokens{1}{1});
        fprintf('[SCENIC] Generated temporary sample: %s\n', xoscFile);
        % Register automatic onCleanup to delete the temporary file when function finishes
        tempCleanup = onCleanup(@() cleanup_temp_scenic(xoscFile));
    end

    sampleTime = 0.05; % 20 Hz update rate

    fprintf('============================================================\n');
    fprintf('EURO NCAP / ASAM OPENSCENARIO RUNNER FOR MATLAB\n');
    fprintf('Scenario File: %s\n', xoscFile);
    fprintf('Road Map File: %s\n', xodrFile);
    fprintf('Simulation Mode: %s | Duration: %.1f s | 3D Unreal: %s\n', ...
        mode, duration, mat2str(enable3D));
    fprintf('============================================================\n');

    % 2. Import Scenario via ASAM OpenSCENARIO Parser
    [scenario, info] = import_openscenario(xoscFile, xodrFile, ...
        'SampleTime', sampleTime, 'StopTime', duration, 'EnableAEB', true, ...
        'LiberateEgo', enableAutonomous);

    numActors = length(scenario.Actors);
    fprintf('Active Entities (%d):\n', numActors);
    for a = 1:numActors
        act = scenario.Actors(a);
        fprintf('  [%d] %-20s | ClassID: %d | Length: %.2fm | Width: %.2fm\n', ...
            act.ActorID, act.Name, act.ClassID, act.Length, act.Width);
    end

    % 3. Check if requested to open in Driving Scenario Designer App
    if strcmpi(mode, 'app')
        fprintf('\n[APP] Opening scenario in Driving Scenario Designer App...\n');
        
        if enableAutonomous
            fprintf('[APP] Running full Autonomous AV Stack pre-simulation (%s)...\n', autoControllerType);
            fprintf('[APP] C3 YOLOv8s + IMM Fusion + GMM Trajectory + Costmap + AEB active\n');
            try
                appEgo = scenario.Actors(1);
                appRig = sensor_rig_builder(scenario, appEgo);
                
                if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'EgoWaypoints') && ~isempty(info.ActorMap.EgoWaypoints)
                    appRawWps = info.ActorMap.EgoWaypoints(:, 1:2);
                else
                    appRawWps = [appEgo.Position(1), appEgo.Position(2);
                                 appEgo.Position(1)+50, appEgo.Position(2);
                                 appEgo.Position(1)+100, appEgo.Position(2);
                                 appEgo.Position(1)+250, appEgo.Position(2)];
                end
                appRefWps = interpolate_ref_path(appRawWps, 0.5);
                appCruise = norm(appEgo.Velocity(1:2));
                if appCruise < 2.0 || appCruise > 7.5, appCruise = 6.94; end % 25 km/h steady smooth cruise
                
                appStack = AutonomousAVStack(appRefWps, appCruise, sampleTime, autoControllerType);
                if isfield(info, 'ScenarioType'), appStack.ScenarioName = info.ScenarioType; end
                if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'Potholes') && ~isempty(info.ActorMap.Potholes)
                    appStack.Potholes = info.ActorMap.Potholes;
                end
                appStack.init_scenario(scenario, appEgo, appRig);
                
                % Initialize logger only if logging enabled
                if enableLog
                    appLogger = simulation_logger(scenario, info.ScenarioType, autoControllerType, 0.5);
                else
                    appLogger = [];
                end
                
                maxAppSteps = ceil(duration / sampleTime);
                appLogPos = zeros(maxAppSteps, 3);
                appLogSpd = zeros(maxAppSteps, 1);
                appLogTime = zeros(maxAppSteps, 1);
                appStepIdx = 0;
                
                while advance(scenario)
                    appStepIdx = appStepIdx + 1;
                    if appStepIdx > maxAppSteps, break; end
                    tNow = scenario.SimulationTime;
                    telem = appStack.step_scenario(scenario, appEgo, appRig, tNow);
                    appLogPos(appStepIdx, :) = [telem.Position(1), telem.Position(2), 0.0];
                    appLogSpd(appStepIdx) = max(0.01, telem.Speed);
                    appLogTime(appStepIdx) = tNow;
                    
                    % Dynamic TTC-based VRU Trigger status update
                    if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'VRU') && isfield(info.ActorMap, 'VRUTriggered') && ~info.ActorMap.VRUTriggered && ~isempty(telem)
                        egoSpeedNow = max(0.1, telem.Speed);
                        distToCross = info.ActorMap.VRUWaypoints(1,1) - telem.Position(1);
                        ttcCondition = distToCross > 0 && (distToCross / egoSpeedNow) <= info.ActorMap.VRUCrossTime;
                        if ttcCondition
                            info.ActorMap.VRUTriggered = true;
                        end
                    end
                    
                    if ~isempty(appLogger)
                        appLogger.step(scenario, appEgo, telem, appRig, tNow);
                    end
                    
                    % Progress indicator every 100 steps
                    if mod(appStepIdx, 100) == 0
                        fprintf('[APP] Step %d/%d | t=%.1fs | Speed=%.1f km/h | Mode=%s\n', ...
                            appStepIdx, maxAppSteps, tNow, telem.Speed*3.6, telem.StateflowMode);
                    end
                end
                
                if ~isempty(appLogger)
                    appLogger.finalize();
                end
                
                appLogPos = appLogPos(1:appStepIdx, :);
                appLogSpd = appLogSpd(1:appStepIdx);
                appLogTime = appLogTime(1:appStepIdx);
                
                % Curvature and speed inflection downsampling for smooth Hermite splining
                keepMask = false(appStepIdx, 1);
                keepMask(1) = true;
                keepMask(end) = true;
                keepMask(mod(round(appLogTime * 20), 10) == 0) = true; % Every 0.5s
                
                dSpd = [0; diff(appLogSpd)];
                keepMask(abs(dSpd) > 0.15) = true;
                % Keep lateral inflection points (lane changes, evasions, detours)
                dLat = [0; diff(appLogPos(:, 2))];
                keepMask(abs(dLat) > 0.04) = true;
                
                keyWps = appLogPos(keepMask, :);
                keySpds = appLogSpd(keepMask);
                
                restart(scenario);
                if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'VRUTriggered')
                    info.ActorMap.VRUTriggered = false;
                end
                trajectory(appEgo, keyWps, keySpds);
                fprintf('[APP] Autonomous AV Stack simulation complete!\n');
                fprintf('[APP] Baked %d trajectory keyframes from full stack (YOLO+Fusion+AEB) into Ego vehicle.\n', size(keyWps, 1));
            catch ME_app
                warning('run_intersection_scenario:appPreSim', ...
                    'Could not pre-simulate autonomous trajectory: %s', ME_app.message);
                fprintf('[APP] Stack trace:\n');
                for si = 1:min(5, length(ME_app.stack))
                    fprintf('  %s (line %d)\n', ME_app.stack(si).name, ME_app.stack(si).line);
                end
                restart(scenario);
                if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'VRUTriggered')
                    info.ActorMap.VRUTriggered = false;
                end
            end
        end
        
        % Ensure 100% compatibility with Driving Scenario Designer:
        % 1. MATLAB drivingScenarioDesigner strictly enforces that any object instantiated as a Vehicle
        %    must have ClassID 1 (Car) or 2 (Truck). If any vehicle has a non-conforming ClassID, normalize it.
        % 2. Any Actor with a trajectory CANNOT have ClassID 5 (Barrier/Immovable in DSD Class Editor).
        %    Normalize moving actors with ClassID 5 to ClassID 4 (Pedestrian/VRU/Movable).
        for actIdx = 1:length(scenario.Actors)
            curAct = scenario.Actors(actIdx);
            if isa(curAct, 'driving.scenario.Vehicle')
                if curAct.ClassID ~= 1 && curAct.ClassID ~= 2
                    curAct.ClassID = 1;
                end
            else
                % Generic actors: check if a trajectory is defined
                hasTraj = false;
                try
                    if isprop(curAct, 'Waypoints') && ~isempty(curAct.Waypoints) && size(curAct.Waypoints, 1) >= 2
                        hasTraj = true;
                    end
                catch
                end
                if (hasTraj || norm(curAct.Velocity) > 0.01) && curAct.ClassID == 5
                    curAct.ClassID = 4; % Reclassify as movable VRU/Pedestrian in DSD
                end
            end
        end

        % Save MAT scenario session file for direct access and reloading AFTER normalization
        matScenarioFile = '';
        if ~isScenicFuzz
            matScenarioFile = fullfile(rootDir, sprintf('%s_Scenario.mat', info.ScenarioType));
            save(matScenarioFile, 'scenario');
            fprintf('[APP] Scenario saved to session file: %s\n', matScenarioFile);
        end

        % Launch Driving Scenario Designer app
        drivingScenarioDesigner(scenario);
        fprintf('[APP] Driving Scenario Designer launched successfully with [%s]!\n', info.ScenarioType);
        fprintf('[APP] Press "Play" to watch the autonomous AV stack trajectory!\n');
        if ~isScenicFuzz
            fprintf('[APP] (You can also open the saved file anytime via: drivingScenarioDesigner(''%s''))\n', matScenarioFile);
        else
            fprintf('[APP] (Scenic session loaded directly into RAM. Temporary files purged.)\n');
        end
        return;
    end

    % 4. Launch 3D Unreal Engine Co-Simulation if Requested
    if enable3D
        fprintf('\n[3D] Initializing Unreal Engine 3D Simulation...\n');
        try
            % plotSim3d in MATLAB Automated Driving Toolbox has NO output arguments
            plotSim3d(scenario, 'ActorInFocus', 1);
            fprintf('[3D] Unreal Engine 3D scenario viewer initialized successfully.\n');
        catch ME3D
            warning('Could not initialize plotSim3d: %s. Continuing with 2D display.', ME3D.message);
            enable3D = false;
        end
    end

    % 4. Configure Multi-Panel Real-Time HUD & Telemetry Dashboard
    isInteractive = strcmpi(mode, 'animate');
    hFig = [];
    ax = [];
    axCockpit = [];
    axDist = [];
    axDyn = [];
    hHUDBoxes = [];
    hHUDLabels = [];
    hCockpitStatus = [];
    hLineDist = [];
    hLineTTC = [];
    hLineSpd = [];
    hLineAcc = [];
    
    numMaxSteps = ceil(duration / sampleTime) + 100;
    stripTime = nan(numMaxSteps, 1);
    stripDist = nan(numMaxSteps, 1);
    stripTTC  = nan(numMaxSteps, 1);
    stripSpd  = nan(numMaxSteps, 1);
    stripAcc  = nan(numMaxSteps, 1);
    maxHUDTargets = 6;

    if isInteractive
        hFig = figure('Name', sprintf('Autonomous Vehicle Dashboard: [%s - %s]', info.ScenarioType, autoControllerType), ...
            'NumberTitle', 'off', 'Color', [0.08 0.08 0.10], ...
            'Position', [40 40 1440 860], 'MenuBar', 'none', 'ToolBar', 'figure');

        % Panel 1: BEV 360 Multi-Sensor World Model (Top-Left)
        ax = subplot(2, 2, 1, 'Parent', hFig, 'Color', [0.12 0.12 0.14], ...
            'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
        grid(ax, 'on');
        set(ax, 'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.6);
        hold(ax, 'on');

        % Plot OpenDRIVE road network
        plot(scenario, 'Parent', ax, 'Meshes', 'off');
        title(ax, sprintf('Panel 1: BEV World Model [%s]', info.ScenarioType), ...
            'FontSize', 11, 'FontWeight', 'bold', 'Color', [1 1 1]);
        xlabel(ax, 'X [m]', 'Color', [0.8 0.8 0.8]);
        ylabel(ax, 'Y [m]', 'Color', [0.8 0.8 0.8]);
        axis(ax, 'equal');

        % Set view window based on scenario type
        if strcmpi(info.ScenarioType, 'CPNCO')
            xlim(ax, [30 330]); ylim(ax, [-35 35]);
        elseif strcmpi(info.ScenarioType, 'CPTA') || strcmpi(info.ScenarioType, 'CCFTAP')
            xlim(ax, [180 330]); ylim(ax, [-40 80]);
        elseif strcmpi(info.ScenarioType, 'CCCSCP')
            xlim(ax, [160 330]); ylim(ax, [-80 80]);
        else
            xlim(ax, [20 340]); ylim(ax, [-50 50]);
        end

        % HUD info text
        hHUD = text(ax, 0.02, 0.96, '', 'Units', 'normalized', ...
            'FontSize', 8.5, 'FontName', 'Consolas', 'Color', [0.1 1.0 0.4], ...
            'BackgroundColor', [0 0 0 0.8], 'EdgeColor', [0.3 0.8 0.4], ...
            'Margin', 5, 'VerticalAlignment', 'top');

        % Panel 2: Synthetic Forward Camera Cockpit HUD (Top-Right)
        axCockpit = subplot(2, 2, 2, 'Parent', hFig, 'Color', [0.04 0.04 0.06], ...
            'XColor', [0.5 0.5 0.5], 'YColor', [0.5 0.5 0.5]);
        hold(axCockpit, 'on');
        xlim(axCockpit, [-320 320]);
        ylim(axCockpit, [-240 240]);
        title(axCockpit, 'Panel 2: Forward Camera & Radar Cockpit HUD', ...
            'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.2 0.8 1.0]);
        
        % Visual ground and sky
        fill(axCockpit, [-320 320 320 -320], [-240 -240 0 0], [0.13 0.13 0.16], 'EdgeColor', 'none'); % Asphalt
        fill(axCockpit, [-320 320 320 -320], [0 0 240 240], [0.06 0.10 0.18], 'EdgeColor', 'none');    % Sky
        plot(axCockpit, [-320 320], [0 0], 'Color', [0.25 0.35 0.45], 'LineWidth', 1.5);             % Horizon
        plot(axCockpit, [0 0], [-240 0], 'Color', [0.9 0.8 0.1], 'LineStyle', '--', 'LineWidth', 1.5); % Center guide
        plot(axCockpit, [-180 -30], [-240 0], 'Color', [0.8 0.8 0.8], 'LineWidth', 1.2);           % Left lane line
        plot(axCockpit, [180 30], [-240 0], 'Color', [0.8 0.8 0.8], 'LineWidth', 1.2);            % Right lane line

        hHUDBoxes = gobjects(maxHUDTargets, 1);
        hHUDLabels = gobjects(maxHUDTargets, 1);
        for hb = 1:maxHUDTargets
            hHUDBoxes(hb) = rectangle('Parent', axCockpit, 'Position', [0 0 0 0], ...
                'EdgeColor', [0.1 0.9 0.3], 'LineWidth', 2, 'Visible', 'off');
            hHUDLabels(hb) = text(axCockpit, 0, 0, '', 'Color', [0.1 1.0 0.4], ...
                'FontSize', 8, 'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.75], ...
                'EdgeColor', [0.2 0.7 0.3], 'Margin', 2, 'Visible', 'off');
        end
        hCockpitStatus = text(axCockpit, 0, 205, '[SYSTEM ACTIVE - CRUISE]', ...
            'Color', [0.1 1.0 0.4], 'FontSize', 10, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'BackgroundColor', [0 0 0 0.85], ...
            'EdgeColor', [0.2 0.8 0.4], 'Margin', 4);

        % Panel 3: In-Lane Obstacle Distance & TTC Strip-Chart (Bottom-Left)
        axDist = subplot(2, 2, 3, 'Parent', hFig, 'Color', [0.12 0.12 0.14], ...
            'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
        grid(axDist, 'on');
        set(axDist, 'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.6);
        hold(axDist, 'on');
        xlim(axDist, [0 duration]);
        ylim(axDist, [0 55]);
        title(axDist, 'Panel 3: Obstacle Range & Time-To-Collision (TTC)', ...
            'FontSize', 11, 'FontWeight', 'bold', 'Color', [1 1 1]);
        xlabel(axDist, 'Simulation Time [s]', 'Color', [0.8 0.8 0.8]);
        ylabel(axDist, 'Range [m] / TTC [s]', 'Color', [0.8 0.8 0.8]);
        
        yline(axDist, 20.0, 'Color', [0.2 0.9 0.4], 'LineStyle', '--', 'LineWidth', 1.2, 'DisplayName', 'Safe Distance (20m)');
        yline(axDist, 3.5,  'Color', [1.0 0.2 0.2], 'LineStyle', ':',  'LineWidth', 1.5, 'DisplayName', 'Stop Buffer (3.5m)');
        hLineDist = plot(axDist, NaN, NaN, 'c-', 'LineWidth', 2.0, 'DisplayName', 'Distance (m)');
        hLineTTC  = plot(axDist, NaN, NaN, 'y--', 'LineWidth', 1.5, 'DisplayName', 'TTC (s)');
        legend(axDist, 'Location', 'northeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.1 0.1 0.1], 'FontSize', 8);

        % Panel 4: Vehicle Dynamics (Speed & Commanded Accel/Decel) (Bottom-Right)
        axDyn = subplot(2, 2, 4, 'Parent', hFig, 'Color', [0.12 0.12 0.14], ...
            'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
        grid(axDyn, 'on');
        set(axDyn, 'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.6);
        hold(axDyn, 'on');
        xlim(axDyn, [0 duration]);
        ylim(axDyn, [-8.5 50]);
        title(axDyn, 'Panel 4: Vehicle Dynamics & Control Signals', ...
            'FontSize', 11, 'FontWeight', 'bold', 'Color', [1 1 1]);
        xlabel(axDyn, 'Simulation Time [s]', 'Color', [0.8 0.8 0.8]);
        ylabel(axDyn, 'Speed [km/h] / Accel [m/s²]', 'Color', [0.8 0.8 0.8]);
        
        yline(axDyn, 0.0, 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0);
        yline(axDyn, -7.5, 'Color', [1.0 0.1 0.1], 'LineStyle', ':', 'LineWidth', 1.2, 'DisplayName', 'Max AEB (-7.5 m/s²)');
        hLineSpd = plot(axDyn, NaN, NaN, 'g-', 'LineWidth', 2.0, 'DisplayName', 'Speed (km/h)');
        hLineAcc = plot(axDyn, NaN, NaN, 'r-', 'LineWidth', 1.8, 'DisplayName', 'Accel (m/s²)');
        legend(axDyn, 'Location', 'northeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.1 0.1 0.1], 'FontSize', 8);
    end

    % Identify key actors (Ego and Primary VRU/Obstacle)
    egoActor = scenario.Actors(1);
    vruActor = [];
    if length(scenario.Actors) >= 4 && strcmpi(info.ScenarioType, 'CPNCO')
        % In CPNCO, VRU is Actor 4
        vruActor = scenario.Actors(4);
    elseif length(scenario.Actors) >= 2
        vruActor = scenario.Actors(2);
    end

    % 4.5 Initialize Autonomous Ego Controller & Unified AV Stack if Requested
    autoController = [];
    avStack = [];
    hSensorCones = struct();
    hTrackPlot = [];
    if enableAutonomous
        fprintf('\n[AUTONOMOUS] Initializing Unified Autonomous AV Stack (C3 YOLOv8s + IMM Fusion + GMM Trajectory + MPC/PP)...\n');
        if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'EgoWaypoints') && ~isempty(info.ActorMap.EgoWaypoints)
            rawWps = info.ActorMap.EgoWaypoints(:, 1:2);
        else
            rawWps = [egoActor.Position(1), egoActor.Position(2);
                      egoActor.Position(1)+50, egoActor.Position(2);
                      egoActor.Position(1)+100, egoActor.Position(2);
                      egoActor.Position(1)+250, egoActor.Position(2)];
        end
        refWps = interpolate_ref_path(rawWps, 0.5);

        % Build full 7-sensor surround suite
        sensorRig = sensor_rig_builder(scenario, egoActor);

        vCruise = norm(egoActor.Velocity(1:2));
        if vCruise < 2.0 || vCruise > 7.5, vCruise = 6.94; end % 25 km/h steady smooth cruise
        
        % Instantiate Unified Champion Autonomous AV Stack
        avStack = AutonomousAVStack(refWps, vCruise, sampleTime, autoControllerType);
        if isfield(info, 'ScenarioType'), avStack.ScenarioName = info.ScenarioType; end
        if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'Potholes') && ~isempty(info.ActorMap.Potholes)
            avStack.Potholes = info.ActorMap.Potholes;
        end
        avStack.init_scenario(scenario, egoActor, sensorRig);
        autoController = avStack.ControllerInstance;

        if isInteractive && isvalid(hFig)
            % 360 BEV Coverage Cones
            hSensorCones.CamFront = fill(ax, NaN, NaN, [0.2 0.8 1.0], 'FaceAlpha', 0.15, 'EdgeColor', [0.2 0.7 0.9], 'DisplayName', 'Front Cam');
            hSensorCones.CamNear  = fill(ax, NaN, NaN, [0.2 1.0 0.5], 'FaceAlpha', 0.15, 'EdgeColor', [0.2 0.9 0.4], 'DisplayName', 'Near Cam');
            hSensorCones.CamRear  = fill(ax, NaN, NaN, [1.0 0.8 0.2], 'FaceAlpha', 0.15, 'EdgeColor', [0.9 0.7 0.1], 'DisplayName', 'Rear Cam');
            hSensorCones.CamSide  = fill(ax, NaN, NaN, [0.9 0.4 0.9], 'FaceAlpha', 0.15, 'EdgeColor', [0.8 0.3 0.8], 'DisplayName', 'Flank Cam');
            hSensorCones.RadFront = fill(ax, NaN, NaN, [1.0 0.4 0.1], 'FaceAlpha', 0.12, 'EdgeColor', [1.0 0.3 0.0], 'LineStyle', '--', 'DisplayName', 'Front Radar');
            hTrackPlot = plot(ax, NaN, NaN, 'gs', 'MarkerSize', 8, 'MarkerFaceColor', [0.1 0.9 0.3], 'LineWidth', 1.5, 'DisplayName', 'Fused 3D Tracks');
        end
        if enableLog
            fprintf('[AUTONOMOUS] Closed-loop %s controller active (Cruise: %.1f km/h, 4 Cameras, 2 Radars, LiDAR Fused)\n', ...
                autoControllerType, vCruise * 3.6);
            mainLogger = simulation_logger(scenario, info.ScenarioType, autoControllerType, 0.5);
        else
            fprintf('[AUTONOMOUS] Closed-loop %s controller active (Cruise: %.1f km/h, fast zero-overhead mode)\n', ...
                autoControllerType, vCruise * 3.6);
            mainLogger = [];
        end
    else
        mainLogger = [];
    end

    % 5. Execute Simulation Loop & Log Poses for SAT Verification
    fprintf('\nExecuting simulation (%.1f s at %.3f s step)...\n', duration, sampleTime);
    numSteps = ceil(duration / sampleTime);
    log_time = zeros(numSteps, 1);
    log_poses = cell(numSteps, numActors);
    log_vels = cell(numSteps, numActors);

    stepIdx = 0;

    while advance(scenario)
        stepIdx = stepIdx + 1;
        tSim = scenario.SimulationTime;
        if stepIdx > numSteps
            break;
        end

        log_time(stepIdx) = tSim;

        % Execute Autonomous Sense-Plan-Act Step via Unified AV Stack
        telem = [];
        if enableAutonomous
            if ~isempty(avStack)
                telem = avStack.step_scenario(scenario, egoActor, sensorRig, tSim);
            elseif ~isempty(autoController)
                telem = autoController.step(egoActor, tSim);
            end

            % Dynamic TTC-based VRU Trigger status update
            if isfield(info, 'ActorMap') && isfield(info.ActorMap, 'VRU') && isfield(info.ActorMap, 'VRUTriggered') && ~info.ActorMap.VRUTriggered && ~isempty(telem)
                egoSpeedNow = max(0.1, telem.Speed);
                distToCross = info.ActorMap.VRUWaypoints(1,1) - telem.Position(1);
                ttcCondition = distToCross > 0 && (distToCross / egoSpeedNow) <= info.ActorMap.VRUCrossTime;
                if ttcCondition
                    info.ActorMap.VRUTriggered = true;
                end
            end
            
            % Update 360-degree sensor coverage cones in interactive visualization
            if isInteractive && isvalid(hFig) && ~isempty(fieldnames(hSensorCones)) && ~isempty(telem)
                vPos = telem.Position;
                vYaw = telem.Yaw;
                
                % Update 4-Camera Cones
                pCam1 = get_sensor_cone_pts(vPos, vYaw, 1.9, 0.0, 0.0, 45.0, deg2rad(22.5));
                set(hSensorCones.CamFront, 'XData', pCam1(:,1), 'YData', pCam1(:,2));
                
                pCam2 = get_sensor_cone_pts(vPos, vYaw, 3.5, 0.0, 0.0, 25.0, deg2rad(30.0));
                set(hSensorCones.CamNear, 'XData', pCam2(:,1), 'YData', pCam2(:,2));

                pCam3 = get_sensor_cone_pts(vPos, vYaw, -1.0, 0.0, pi, 35.0, deg2rad(30.0));
                set(hSensorCones.CamRear, 'XData', pCam3(:,1), 'YData', pCam3(:,2));

                pCam4 = get_sensor_cone_pts(vPos, vYaw, 0.8, 0.9, deg2rad(75), 28.0, deg2rad(30.0));
                set(hSensorCones.CamSide, 'XData', pCam4(:,1), 'YData', pCam4(:,2));

                % Update Front Radar Arc
                pRad1 = get_sensor_cone_pts(vPos, vYaw, 3.7, 0.0, 0.0, 65.0, deg2rad(25.0));
                set(hSensorCones.RadFront, 'XData', pRad1(:,1), 'YData', pRad1(:,2));

                % Update Fused 3D Track Points
                if ~isempty(telem.FusedTracks)
                    tX = zeros(numel(telem.FusedTracks), 1);
                    tY = zeros(numel(telem.FusedTracks), 1);
                    for trkIdx = 1:numel(telem.FusedTracks)
                        relPos = telem.FusedTracks(trkIdx).Position;
                        R_mat = [cos(vYaw) -sin(vYaw); sin(vYaw) cos(vYaw)];
                        wPos = vPos(:) + R_mat * relPos(:);
                        tX(trkIdx) = wPos(1);
                        tY(trkIdx) = wPos(2);
                    end
                    set(hTrackPlot, 'XData', tX, 'YData', tY);
                else
                    set(hTrackPlot, 'XData', NaN, 'YData', NaN);
                end
            end
            
            % Log 3D Surround Cameras and Sensor Streams at 5 Hz
            if ~isempty(mainLogger)
                mainLogger.step(scenario, egoActor, telem, sensorRig, tSim);
            end
        end

        % Log all actor states
        for a = 1:numActors
            act = scenario.Actors(a);
            log_poses{stepIdx, a} = struct('Position', act.Position, ...
                'Velocity', act.Velocity, 'Yaw', act.Yaw, 'Name', act.Name, ...
                'Length', act.Length, 'Width', act.Width, 'Height', act.Height);
            log_vels{stepIdx, a} = act.Velocity;
        end

        % Update 4-Panel Real-Time Dashboard
        if isInteractive && isvalid(hFig)
            egoPos = egoActor.Position;
            egoSpdKph = norm(egoActor.Velocity(1:2)) * 3.6;

            if enableAutonomous && ~isempty(telem)
                obsDist = telem.ObstacleDistance;
                aCmd = telem.Accel;
                if isfinite(obsDist)
                    distText = sprintf('%.2f m', obsDist);
                else
                    distText = 'CLEAR';
                end

                if aCmd <= -4.0
                    statusText = sprintf('AEB ACTIVE: Emergency Braking (a = %.1f m/s^2, Obs: %s)', aCmd, distText);
                    statusColor = [1.0 0.2 0.2];
                    cockpitStr = sprintf('[AEB EMERGENCY BRAKE ACTIVATED: a = %.1f m/s^2]', aCmd);
                    cockpitColor = [1.0 0.2 0.2];
                elseif aCmd < -0.2
                    statusText = sprintf('ACC DECEL: Slowing for Obstacle (a = %.1f m/s^2, Obs: %s)', aCmd, distText);
                    statusColor = [1.0 0.6 0.0];
                    cockpitStr = sprintf('[ACC YIELDING: Obstacle at %s]', distText);
                    cockpitColor = [1.0 0.6 0.0];
                elseif egoSpdKph < 1.0 && isfinite(obsDist)
                    statusText = sprintf('YIELDING: Standstill hold (Waiting for Obstacle at %s)', distText);
                    statusColor = [1.0 0.8 0.0];
                    cockpitStr = sprintf('[STANDSTILL: Holding for Crossing Hazard at %s]', distText);
                    cockpitColor = [1.0 0.8 0.0];
                elseif egoSpdKph < 0.9 * autoController.CruiseSpeed * 3.6
                    statusText = sprintf('ACCELERATING: Path clear, ramping speed (a = %+.1f m/s^2)', aCmd);
                    statusColor = [0.2 1.0 0.4];
                    cockpitStr = '[PATH CLEAR - RESUMING CRUISE SPEED]';
                    cockpitColor = [0.2 1.0 0.4];
                else
                    statusText = sprintf('CRUISING: Target %.1f km/h maintained', autoController.CruiseSpeed * 3.6);
                    statusColor = [0.2 0.8 1.0];
                    cockpitStr = '[CRUISING - PATH CLEAR]';
                    cockpitColor = [0.2 0.8 1.0];
                end

                hudStr = sprintf([ ...
                    'Time: %5.2f s / %5.1f s  |  Ego Speed: %5.1f km/h  [%s]\n' ...
                    'Steering: %+5.1f deg  |  Cross-Track Err: %+5.3f m  |  Tracks: %d\n' ...
                    'Sensors: 4-Cam BEV + 2-Radars + LiDAR  |  Obstacle: %s [%s]\n' ...
                    'System Status: %s'], ...
                    tSim, duration, egoSpdKph, upper(autoControllerType), ...
                    rad2deg(telem.Steering), telem.LateralError, telem.NumTracks, ...
                    distText, upper(telem.LeadClass), statusText);

                % Update Strip-Chart Time Series
                stripTime(stepIdx) = tSim;
                if isfinite(obsDist), stripDist(stepIdx) = min(50, obsDist); else, stripDist(stepIdx) = NaN; end
                if isfinite(telem.TTC), stripTTC(stepIdx) = min(50, telem.TTC); else, stripTTC(stepIdx) = NaN; end
                stripSpd(stepIdx) = egoSpdKph;
                stripAcc(stepIdx) = aCmd;

                if isgraphics(hLineDist), set(hLineDist, 'XData', stripTime(1:stepIdx), 'YData', stripDist(1:stepIdx)); end
                if isgraphics(hLineTTC),  set(hLineTTC,  'XData', stripTime(1:stepIdx), 'YData', stripTTC(1:stepIdx)); end
                if isgraphics(hLineSpd),  set(hLineSpd,  'XData', stripTime(1:stepIdx), 'YData', stripSpd(1:stepIdx)); end
                if isgraphics(hLineAcc),  set(hLineAcc,  'XData', stripTime(1:stepIdx), 'YData', stripAcc(1:stepIdx)); end

                % Update Cockpit Status Banner
                if isgraphics(hCockpitStatus)
                    set(hCockpitStatus, 'String', cockpitStr, 'Color', cockpitColor, 'EdgeColor', cockpitColor);
                end

                % Project Forward Targets to Pinhole Cockpit HUD
                if ~isempty(hHUDBoxes)
                    fx = 800; fy = 800; Hcam = 1.5;
                    fwdCount = 0;
                    for trk = 1:numel(telem.FusedTracks)
                        rPos = telem.FusedTracks(trk).Position;
                        if rPos(1) > 1.0 && rPos(1) < 45.0 && abs(rPos(2)) < 8.0 && fwdCount < maxHUDTargets
                            fwdCount = fwdCount + 1;
                            u = -fx * rPos(2) / rPos(1);
                            v = -fy * Hcam / rPos(1);
                            w = max(20, fx * 1.8 / rPos(1));
                            h = max(25, fy * 1.5 / rPos(1));
                            
                            boxColor = [0.1 0.9 0.3];
                            if rPos(1) < 8.0 || aCmd <= -4.0, boxColor = [1.0 0.1 0.1];
                            elseif rPos(1) < 20.0, boxColor = [1.0 0.8 0.1]; end
                            
                            set(hHUDBoxes(fwdCount), 'Position', [u - w/2, v, w, h], ...
                                'EdgeColor', boxColor, 'Visible', 'on');
                            set(hHUDLabels(fwdCount), 'Position', [u - w/2, v + h + 8, 0], ...
                                'String', sprintf('%s\n%.1fm', upper(telem.FusedTracks(trk).Class), rPos(1)), ...
                                'Color', boxColor, 'EdgeColor', boxColor, 'Visible', 'on');
                        end
                    end
                    for trk = (fwdCount + 1):maxHUDTargets
                        set(hHUDBoxes(trk), 'Visible', 'off');
                        set(hHUDLabels(trk), 'Visible', 'off');
                    end
                end
            else
                % Open-loop pre-scripted trajectory playback (Legacy Euro NCAP benchmark)
                distText = 'N/A';
                statusText = 'PRE-SCRIPTED BENCHMARK TRAJECTORY';
                statusColor = [0.2 0.8 1.0];
                if ~isempty(vruActor)
                    distToVru = norm(egoPos(1:2) - vruActor.Position(1:2));
                    distText = sprintf('%.2f m', distToVru);
                end
                hudStr = sprintf([ ...
                    'Time: %5.2f s / %5.1f s  |  Ego Speed: %5.1f km/h  [PRE-SCRIPTED]\n' ...
                    'Ego Pos: (X=%6.1f m, Y=%5.2f m)  |  Distance: %s\n' ...
                    'System Status: %s'], ...
                    tSim, duration, egoSpdKph, egoPos(1), egoPos(2), distText, statusText);
            end

            set(hHUD, 'String', hudStr, 'Color', statusColor);
            drawnow limitrate;
        end

        % Pace simulation to real-time when 2D animation or 3D Unreal Engine is active
        if isInteractive || enable3D
            pause(sampleTime);
        end
    end

    % Finalize simulation logger: generate graphs, CSVs, and metadata
    if enableAutonomous && ~isempty(mainLogger)
        mainLogger.finalize();
    end

    % 6. Save Trajectory Log for Python Mathematical Collision Verifier
    verifDir = fullfile(rootDir, 'verification');
    logFile = fullfile(verifDir, 'actor_trajectories_log.mat');
    timeSteps = log_time(1:stepIdx);
    
    trajectories = struct();
    for a = 1:numActors
        act = scenario.Actors(a);
        trajectories(a).Name = char(act.Name);
        trajectories(a).ClassID = double(act.ClassID);
        trajectories(a).Length = double(act.Length);
        trajectories(a).Width = double(act.Width);
        trajectories(a).Height = double(act.Height);
        
        posArr = zeros(stepIdx, 3);
        velArr = zeros(stepIdx, 3);
        yawArr = zeros(stepIdx, 1);
        for s = 1:stepIdx
            posArr(s, :) = log_poses{s, a}.Position;
            velArr(s, :) = log_poses{s, a}.Velocity;
            yawArr(s, 1) = log_poses{s, a}.Yaw;
        end
        trajectories(a).Pos = posArr;
        trajectories(a).Vel = velArr;
        trajectories(a).Yaw = yawArr;
    end

    scenario_type = info.ScenarioType;
    actor_names = {scenario.Actors.Name};

    save(logFile, 'trajectories', 'timeSteps', 'actor_names', 'scenario_type', '-v7');
    fprintf('\nSimulation complete! Exported %d timesteps to: %s\n', stepIdx, logFile);
    fprintf('============================================================\n');
end

function cleanup_temp_scenic(filePath)
    if isfile(filePath)
        try
            delete(filePath);
            tmpDir = fileparts(filePath);
            d = dir(tmpDir);
            d = d(~ismember({d.name}, {'.', '..'}));
            if isempty(d)
                rmdir(tmpDir, 's');
            end
            fprintf('\n[PURGE] Successfully deleted temporary Scenic file: %s\n', filePath);
        catch
        end
    end
end

function dense = interpolate_ref_path(pts, ds)
    diffs = diff(pts, 1, 1);
    segLens = hypot(diffs(:, 1), diffs(:, 2));
    keep = [true; segLens > 1e-4];
    pts = pts(keep, :);
    if size(pts, 1) < 2
        dense = pts;
        return;
    end
    diffs = diff(pts, 1, 1);
    segLens = hypot(diffs(:, 1), diffs(:, 2));
    sCum = [0; cumsum(segLens)];
    totalLen = sCum(end);
    if totalLen < ds
        dense = pts;
        return;
    end
    sQuery = (0:ds:totalLen)';
    xDense = interp1(sCum, pts(:, 1), sQuery, 'pchip');
    yDense = interp1(sCum, pts(:, 2), sQuery, 'pchip');
    dense = [xDense, yDense];
end

function pts = get_sensor_cone_pts(vPos, vYaw, xOff, yOff, yawOff, r, halfFov)
    R_ego = [cos(vYaw), -sin(vYaw); sin(vYaw), cos(vYaw)];
    sPos = vPos(:) + R_ego * [xOff; yOff];
    totYaw = vYaw + yawOff;
    p1 = sPos';
    p2 = sPos' + r * [cos(totYaw - halfFov), sin(totYaw - halfFov)];
    p3 = sPos' + r * [cos(totYaw), sin(totYaw)];
    p4 = sPos' + r * [cos(totYaw + halfFov), sin(totYaw + halfFov)];
    pts = [p1; p2; p3; p4; p1];
end
