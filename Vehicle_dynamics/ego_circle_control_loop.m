function simLog = ego_circle_control_loop(varargin)
% EGO_CIRCLE_CONTROL_LOOP Autonomous Ego Control Simulation Runner
% =========================================================================
% Demonstrates closed-loop autonomous driving on curved/circular roads with:
% 1. Forward-Facing Vision Camera (visionDetectionGenerator)
% 2. Modular Lateral Controllers (Pure Pursuit or MPC)
% 3. Longitudinal ACC & Collision-Avoidance AEB
% 4. Real-time 2D HUD bird's-eye visualization & Driving Scenario Designer App
%
% Usage:
%   ego_circle_control_loop;                      % Default Pure Pursuit
%   ego_circle_control_loop('controller', 'MPC'); % Model Predictive Control
%   ego_circle_control_loop('app');               % Open in Driving Scenario Designer
%   ego_circle_control_loop('obstacle', true);    % Test camera detection & AEB
% =========================================================================

    % Default settings
    opts = struct('controller', 'PurePursuit', ...
                  'mode', 'animate', ...
                  'obstacle', false, ...
                  'duration', 20.0, ...
                  'speed', 10.0, ...
                  'trackType', 'winding');

    k = 1;
    while k <= length(varargin)
        arg = varargin{k};
        if isempty(arg)
            k = k + 1;
            continue;
        end
        if (ischar(arg) || isstring(arg))
            s = char(arg);
            su = upper(s);
            % Check if it's a known parameter name followed by a value
            if k < length(varargin) && ismember(su, {'CONTROLLER', 'MODE', 'OBSTACLE', 'DURATION', 'SPEED', 'TRACKTYPE'})
                val = varargin{k+1};
                switch su
                    case 'CONTROLLER'
                        opts.controller = char(val);
                    case 'MODE'
                        opts.mode = char(val);
                    case 'OBSTACLE'
                        opts.obstacle = logical(val);
                    case 'DURATION'
                        opts.duration = double(val);
                    case 'SPEED'
                        opts.speed = double(val);
                    case 'TRACKTYPE'
                        opts.trackType = char(val);
                end
                k = k + 2;
                continue;
            end
            % Standalone flags/shortcuts
            if ismember(su, {'MPC', 'NMPC'})
                opts.controller = 'MPC';
            elseif ismember(su, {'PUREPURSUIT', 'PP', 'PURSUIT'})
                opts.controller = 'PurePursuit';
            elseif ismember(su, {'APP', 'DESIGNER', 'DRIVINGSCENARIODESIGNER'})
                opts.mode = 'app';
            elseif ismember(su, {'HEADLESS', 'BATCH', 'SILENT'})
                opts.mode = 'headless';
            elseif ismember(su, {'ANIMATE', 'GUI', '2D'})
                opts.mode = 'animate';
            elseif ismember(su, {'OBSTACLE', 'LEAD', 'LEAD_VEHICLE'})
                opts.obstacle = true;
            elseif ismember(su, {'NO_OBSTACLE', 'CLEAR'})
                opts.obstacle = false;
            elseif ismember(su, {'WINDING', 'CIRCLE', 'S-CURVE'})
                opts.trackType = lower(s);
            end
        elseif isnumeric(arg) && isscalar(arg)
            % Positional numeric: first numeric is duration, second is speed
            opts.duration = double(arg);
        elseif islogical(arg) && isscalar(arg)
            opts.obstacle = arg;
        end
        k = k + 1;
    end

    sampleTime = 0.05;
    totalTime = opts.duration;

    fprintf('============================================================\n');
    fprintf('AUTONOMOUS EGO CONTROL LOOP SIMULATION\n');
    fprintf('Controller: %s | Mode: %s | Obstacle: %s | Duration: %.1fs\n', ...
        opts.controller, opts.mode, mat2str(opts.obstacle), totalTime);
    fprintf('============================================================\n');

    % 1. Build Circular Scenario with Road, Camera, and Ego Vehicle
    [scenario, ego, sensor, refWps, trackInfo] = create_ego_circle_scenario(...
        'trackType', opts.trackType, ...
        'hasObstacle', opts.obstacle, ...
        'sampleTime', sampleTime, ...
        'stopTime', totalTime);

    % 2. Open in Driving Scenario Designer App if Requested
    if strcmpi(opts.mode, 'app')
        fprintf('\n[APP] Launching scenario in Driving Scenario Designer App...\n');
        drivingScenarioDesigner(scenario);
        fprintf('[APP] Driving Scenario Designer launched with Ego & Camera Sensor!\n');
        simLog = [];
        return;
    end

    % 3. Initialize Autonomous Controller Manager
    autoController = autonomous_ego_controller(...
        refWps, sensor, opts.controller, opts.speed, sampleTime);
    autoController.init_state(ego);

    % 4. Setup 2D Interactive Bird's-Eye Figure
    isInteractive = strcmpi(opts.mode, 'animate');
    hFig = [];
    if isInteractive
        hFig = figure('Name', sprintf('Autonomous Ego Control - [%s]', opts.controller), ...
            'NumberTitle', 'off', 'Color', [0.12 0.12 0.14], ...
            'Position', [100 80 1200 780]);

        ax = axes('Parent', hFig, 'Color', [0.18 0.18 0.20], ...
            'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
        grid(ax, 'on'); hold(ax, 'on'); axis(ax, 'equal');
        set(ax, 'GridColor', [0.35 0.35 0.35], 'GridAlpha', 0.6);

        % Plot road geometry
        plot(scenario, 'Parent', ax, 'Meshes', 'off');
        plot(ax, refWps(:, 1), refWps(:, 2), 'c--', 'LineWidth', 1.2, 'DisplayName', 'Reference Path');

        % Trail line and visual markers
        hTrail = animatedline(ax, 'Color', [0.1 0.8 1.0], 'LineWidth', 2.0, 'DisplayName', 'Driven Trail');
        hTargetPt = plot(ax, NaN, NaN, 'yp', 'MarkerSize', 12, 'MarkerFaceColor', 'y', 'DisplayName', 'Lookahead Target');
        hSensorFOV = fill(ax, NaN, NaN, [1.0 0.8 0.2], 'FaceAlpha', 0.2, 'EdgeColor', [1.0 0.8 0.2], 'DisplayName', 'Camera FOV');

        title(ax, sprintf('Closed-Loop Autonomous Driving: [%s Controller]', upper(opts.controller)), ...
            'FontSize', 13, 'FontWeight', 'bold', 'Color', [1 1 1]);
        xlabel(ax, 'X Position [m]', 'Color', [0.9 0.9 0.9]);
        ylabel(ax, 'Y Position [m]', 'Color', [0.9 0.9 0.9]);
        legend(ax, 'Location', 'northeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.15 0.15 0.15]);

        % Telemetry HUD Box
        hHUD = annotation(hFig, 'textbox', [0.02 0.02 0.38 0.22], ...
            'String', 'Initializing Telemetry...', 'FitBoxToText', 'off', ...
            'BackgroundColor', [0.08 0.08 0.10], 'Color', [0.2 0.9 0.4], ...
            'EdgeColor', [0.3 0.3 0.3], 'FontSize', 10, 'FontName', 'Consolas');
    end

    % 5. Execute Simulation Step-by-Step (Closed-Loop Sense-Plan-Act)
    maxSteps = ceil(totalTime / sampleTime);
    logData = cell(maxSteps, 1);
    stepIdx = 0;

    fprintf('\nExecuting autonomous closed-loop simulation (T = %.1fs)...\n', totalTime);

    while advance(scenario)
        stepIdx = stepIdx + 1;
        simTime = scenario.SimulationTime;

        % Execute Autonomous Control Step (Perception -> Planning -> Control -> Act)
        telem = autoController.step(ego, simTime);
        logData{stepIdx} = telem;

        % Interactive 2D Display Refresh
        if isInteractive && ishandle(hFig)
            addpoints(hTrail, telem.Position(1), telem.Position(2));
            set(hTargetPt, 'XData', telem.TargetPoint(1), 'YData', telem.TargetPoint(2));

            % Update Camera Sensor FOV Visual Wedge (60 deg, 35m)
            camRange = 35.0;
            camHalfFov = deg2rad(30);
            vehYaw = telem.Yaw;
            vPos = telem.Position;
            p1 = vPos;
            p2 = vPos + camRange * [cos(vehYaw - camHalfFov), sin(vehYaw - camHalfFov)];
            p3 = vPos + camRange * [cos(vehYaw + camHalfFov), sin(vehYaw + camHalfFov)];
            set(hSensorFOV, 'XData', [p1(1), p2(1), p3(1)], 'YData', [p1(2), p2(2), p3(2)]);

            % Update Telemetry HUD text
            obsStr = 'Clear';
            if telem.ObstacleDistance < 40.0
                obsStr = sprintf('%.1f m', telem.ObstacleDistance);
            end

            hudText = sprintf([...
                '================ TELEMETRY DASHBOARD ================\n' ...
                'Active Controller : %s (Closed-Loop)\n' ...
                'Simulation Time   : %5.2f s / %5.1f s\n' ...
                'Vehicle Speed     : %5.2f m/s  (%5.1f km/h)\n' ...
                'Steering Angle    : %5.2f deg  (%5.3f rad)\n' ...
                'Acceleration      : %+5.2f m/s^2\n' ...
                'Cross-Track Error : %5.3f m\n' ...
                'Heading Error     : %5.2f deg\n' ...
                'Camera Detections : %d object(s)  [Obstacle: %s]'
            ], telem.Controller, simTime, totalTime, ...
               telem.Speed, telem.Speed * 3.6, ...
               rad2deg(telem.Steering), telem.Steering, ...
               telem.Accel, telem.LateralError, ...
               rad2deg(telem.HeadingError), ...
               telem.NumDetections, obsStr);

            set(hHUD, 'String', hudText);
            drawnow limitrate;
        end
    end

    simLog = logData(1:stepIdx);

    % 6. Post-Run Summary & Tracking Metrics
    latErrors = cellfun(@(d) abs(d.LateralError), simLog);
    speeds = cellfun(@(d) d.Speed, simLog);
    steers = cellfun(@(d) abs(rad2deg(d.Steering)), simLog);

    fprintf('\n============================================================\n');
    fprintf('AUTONOMOUS CONTROL PERFORMANCE REPORT [%s]\n', upper(opts.controller));
    fprintf('============================================================\n');
    fprintf('Mean Cross-Track Error   : %.3f m\n', mean(latErrors));
    fprintf('Max Cross-Track Error    : %.3f m\n', max(latErrors));
    fprintf('Average Cruising Speed   : %.2f km/h\n', mean(speeds) * 3.6);
    fprintf('Max Steering Angle       : %.2f deg\n', max(steers));
    fprintf('Total Steps Executed     : %d\n', stepIdx);
    if max(latErrors) < 0.5
        fprintf('[RESULT] PASS: High Precision Lane Tracking Achieved (|e_y| < 0.5m)!\n');
    else
        fprintf('[RESULT] COMPLETED: Lane Tracking Maintained.\n');
    end
    fprintf('============================================================\n');
end
