function varargout = run_instruction(varargin)
% RUN_INSTRUCTION Unified Autonomous Vehicle Simulation Entry Point
% =========================================================================
% Executes full closed-loop autonomous driving simulations combining:
%   1. Multi-Sensor Rig: 4 Surround Cameras (Front, Near, Rear, Flank) +
%                        2 Automotive Radars (Front 77 GHz LRR, Rear) +
%                        360° Roof LiDAR Scanner
%   2. Sensor Fusion Bridge: Information Matrix Covariance Intersection,
%                            Multi-Hypothesis Spatial Gating, and
%                            Linear Kalman Filtering (License-Independent)
%   3. Autonomous Ego Controller: Dynamic Time-to-Collision (TTC) Supervisor,
%                                 Lateral Guidance via Model Predictive Control (MPC)
%                                 or Pure Pursuit (PP), and Longitudinal Accel/Brake
%   4. Rich Scenario Suite: Euro NCAP (CPNCO, CPTA, CCFtap, CCCscp),
%                           Indian Mixed Traffic (AutoCutIn, TwoWheeler, Jaywalk, Cattle),
%                           ASAM Standards, and UC Berkeley Scenic Fuzzing
%   5. Simulation Logger:   Timestamped Isolation Folders (logs/run_YYYYMMDD_HHMMSS_...),
%                           4 Surround 3D Camera Views & Synchronized Cockpit HUD Mosaics,
%                           Complete Sensor CSVs & LiDAR Point Clouds (.mat),
%                           and Publication-Quality Analytics Strip-Charts (.png)
%
% Syntax:
%   run_instruction                                  % Default: CPNCO with Pure Pursuit (Animate)
%   run_instruction('CPNCO', 'pp')                   % CPNCO with Pure Pursuit Controller
%   run_instruction('CPNCO', 'mpc')                  % CPNCO with Model Predictive Controller
%   run_instruction('Indian_AutoCutIn', 'mpc')       % Indian Auto Cut-In with MPC
%   run_instruction('Indian_TwoWheeler', 'pp')       % Indian Two-Wheeler Lane Filter with PP
%   run_instruction('scenic', 'mpc')                 % Scenic Monte Carlo Fuzzer with MPC
%   run_instruction('CPNCO', 'mpc', 'headless', 10)  % Headless batch run for 10 seconds
%   run_instruction('CPNCO', 'pp', 'app')            % Pre-simulate with full 3D camera/sensor logging and open App
%   run_instruction('CPNCO', 'pp', '3d')             % Unreal Engine 3D Co-Simulation
%
% Arguments (can be passed in any order):
%   - Scenario:   'CPNCO', 'CPTA', 'CCFtap', 'CCCscp',
%                 'Indian_AutoCutIn', 'Indian_TwoWheeler', 'Indian_Jaywalk', 'Indian_Cattle',
%                 'scenic', 'LaneChange', 'Overtake', 'PedCrossing', etc.
%   - Controller: 'mpc' (Model Predictive Control) or 'pp' (Pure Pursuit). Default: 'pp'
%   - Mode:       'animate' (interactive 2D HUD), 'headless' (batch), 'app' (Designer App), '3d' (Unreal)
%   - Duration:   Numeric simulation duration in seconds (e.g., 10.0, 25.0)
% =========================================================================

    thisDir = fileparts(mfilename('fullpath'));
    if isempty(thisDir), thisDir = pwd; end
    rootDir = thisDir;
    if ~isfolder(fullfile(rootDir, 'scenarios'))
        rootDir = fileparts(thisDir);
    end

    % Add all modular subsystem paths to MATLAB environment
    addpath(rootDir);
    addpath(fullfile(rootDir, 'main'));
    addpath(fullfile(rootDir, 'scenarios'));
    addpath(fullfile(rootDir, 'scenarios', 'matlab'));
    addpath(fullfile(rootDir, 'scenarios', 'maps_xodr'));
    addpath(fullfile(rootDir, 'scenarios', 'verification'));
    addpath(fullfile(rootDir, 'vehicle_dynamics'));
    addpath(fullfile(rootDir, 'perception'));
    addpath(fullfile(rootDir, 'Sensor_fusion'));
    addpath(fullfile(rootDir, 'Trajectory'));

    % Parse inputs and ensure autonomous mode is always engaged
    hasController = false;
    chosenController = 'PurePursuit';
    chosenScenario = 'CPNCO';
    chosenMode = 'animate';
    durationVal = [];
    enableLog = false; % Default is false for instant app launch and zero-overhead execution

    args = varargin;
    if isempty(args)
        args = {'CPNCO', 'pp', 'animate'};
        hasController = true;
    else
        % Inspect arguments
        for i = 1:numel(args)
            arg = args{i};
            if ischar(arg) || isstring(arg)
                su = upper(char(arg));
                switch su
                    case {'MPC', 'NMPC'}
                        hasController = true;
                        chosenController = 'MPC';
                    case {'PP', 'PUREPURSUIT', 'PURSUIT'}
                        hasController = true;
                        chosenController = 'PurePursuit';
                    case {'HEADLESS', 'BATCH', 'SILENT'}
                        chosenMode = 'headless';
                    case {'APP', 'DESIGNER', 'DRIVINGSCENARIODESIGNER'}
                        chosenMode = 'app';
                    case {'3D', 'UNREAL'}
                        chosenMode = '3d';
                    case {'SIM3D', 'SURROUND', 'HARNESS'}
                        chosenMode = 'sim3d';
                    case {'ANIMATE', 'GUI', '2D'}
                        chosenMode = 'animate';
                    case {'STACK', 'INTEGRATED', 'PIPELINE', 'FULL_STACK', 'AVSTACK'}
                        chosenMode = 'stack';
                    case {'LOG', 'LOGGING', 'ENABLELOG', 'ENABLE_LOG', 'RECORD'}
                        enableLog = true;
                    case {'NOLOG', 'NO_LOG', 'NOLOGGING', 'FAST', 'NOLOGS'}
                        enableLog = false;
                    otherwise
                        chosenScenario = arg;
                end
            elseif islogical(arg)
                enableLog = arg;
            elseif isnumeric(arg)
                durationVal = double(arg);
            end
        end
    end

    % If no controller was explicitly specified, default to Pure Pursuit
    if ~hasController
        args{end+1} = 'pp';
        chosenController = 'PurePursuit';
    end

    % Pass explicit logging flag to scenario runner
    if enableLog
        args{end+1} = 'log';
    else
        args{end+1} = 'nolog';
    end

    % Banner Output
    fprintf('=========================================================================\n');
    fprintf('  RUN_INSTRUCTION: AUTONOMOUS DRIVING SIMULATION & SENSOR FUSION SYSTEM\n');
    fprintf('=========================================================================\n');
    fprintf('  Selected Scenario  : %s\n', string(chosenScenario));
    fprintf('  Lateral Controller : %s\n', chosenController);
    fprintf('  Execution Mode     : %s\n', chosenMode);
    fprintf('  Logging Enabled    : %s (default: false for instant app/sim launch)\n', mat2str(enableLog));
    if ~isempty(durationVal)
        fprintf('  Duration           : %.1f s\n', durationVal);
    end
    fprintf('  Sensor Suite       : 4-Cam BEV (Front, Near, Rear, Flank) + 2 Radars + LiDAR\n');
    fprintf('  Fusion Core        : Kalman Covariance Intersection & In-Lane TTC Supervisor\n');
    fprintf('=========================================================================\n\n');

    % If STACK / INTEGRATED mode is selected, invoke the Unified Deep Learning AV Stack
    if strcmpi(chosenMode, 'stack')
        dur = 10.0;
        if ~isempty(durationVal), dur = durationVal; end
        results = run_integrated_pipeline(dur, true);
        if nargout >= 1
            varargout{1} = results;
        end
        return;
    end

    % If 3D mode is selected, invoke the Unreal Engine co-simulation & video recorder
    if strcmpi(chosenMode, '3d')
        dur = 30.0;
        if ~isempty(durationVal), dur = durationVal; end
        [videoPath, sc, runDir] = record_unreal_simulation(chosenScenario, chosenController, dur);
        if nargout > 2
            varargout{1} = videoPath;
            varargout{2} = sc;
            varargout{3} = runDir;
        elseif nargout == 2
            varargout{1} = videoPath;
            varargout{2} = sc;
        elseif nargout == 1
            varargout{1} = videoPath;
        end
        return;
    end

    % If SIM3D mode is selected, invoke the 4-Camera Simulink 3D Surround Harness
    if strcmpi(chosenMode, 'sim3d')
        dur = 5.0;
        if ~isempty(durationVal), dur = durationVal; end
        outStruct = run_sim3d_surround(chosenScenario, chosenController, dur);
        if nargout >= 1
            varargout{1} = outStruct;
        end
        return;
    end

    % Execute simulation runner
    scenario = run_intersection_scenario(args{:});
    if nargout >= 1
        varargout{1} = scenario;
    end
end
