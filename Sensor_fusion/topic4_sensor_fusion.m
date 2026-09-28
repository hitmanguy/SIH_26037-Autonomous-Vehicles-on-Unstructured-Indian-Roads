%% Topic 4: Sensor Fusion & Tracking
% SIH 2026 - Problem Statement SIH26037
% Synthetic Radar + Vision Fusion with Erratic Actor (Auto-Rickshaw) & Sensor Degradation
%
% Key Improvements & Fixes:
%  1. Uses trackerGNN to support both Constant Velocity KF and Interacting Multiple Model (IMM).
%  2. Assigns physical trajectories to Ego vehicle and Passing Car so they properly move in drivingScenario.
%  3. Adds non-lane-parallel erratic actor (Auto-Rickshaw) swerving across lanes.
%  4. Supports simulated road hazard / pothole obstacle detection.
%  5. Handles sensor degradation (radar azimuth noise + vision dropout) for KF vs. IMM analysis.
%  6. Avoids display ghosting by refreshing empty detections/tracks in bird's-eye plot.
%  7. Saves output snapshot as 'topic4_simulation_result.png'.

% Clear previous variables to avoid variable-function shadowing while preserving user flags
clearvars -except useIMM degradeSensors enablePothole useRealisticAVRates;
close all;

%% 1. Configuration Flags (Toggle these to test different cases)
if ~exist('useIMM', 'var')
    useIMM = false;          % false = Constant Velocity KF | true = Interacting Multiple Model (IMM)
end

if ~exist('degradeSensors', 'var')
    degradeSensors = false; % false = Ideal sensor specs  | true = Degraded radar noise + camera dropouts
end

if ~exist('enablePothole', 'var')
    enablePothole = true;   % true  = Includes low-profile road hazard in simulation
end

if ~exist('useRealisticAVRates', 'var')
    useRealisticAVRates = false; % false = 20 Hz synchronous | true = Realistic AV (50Hz clock, 20Hz radar, 10Hz vision DNN)
end

fprintf('=====================================================\n');
fprintf('   Topic 4: Sensor Fusion & Multi-Object Tracking    \n');
fprintf('=====================================================\n');
if useIMM
    fprintf(' [*] Filter Mode : Interacting Multiple Model (IMM)\n');
else
    fprintf(' [*] Filter Mode : Standard Constant Velocity Kalman Filter\n');
end

if degradeSensors
    fprintf(' [!] Sensors     : DEGRADED (Radar Azimuth Res: 15 deg, Vision Prob: 0.4)\n');
else
    fprintf(' [*] Sensors     : NOMINAL (Radar Azimuth Res: 4 deg, Vision Prob: 0.9)\n');
end
fprintf(' [*] Pothole     : %s\n', mat2str(enablePothole));
fprintf(' [*] AV MultiRate: %s (50Hz Clock, 20Hz Radar, 10Hz Vision DNN)\n', mat2str(useRealisticAVRates));
fprintf('=====================================================\n\n');

%% 2. Create Driving Scenario (Extended 20-Second Highway)
if useRealisticAVRates
    scenario = drivingScenario('StopTime', 20.0, 'SampleTime', 0.01); % 100 Hz clock (common divisor for 20 Hz radar and 10 Hz camera)
else
    scenario = drivingScenario('StopTime', 20.0, 'SampleTime', 0.05); % 20 Hz nominal
end

% Define a straight road with 2 lanes (each 3.6m wide), extended to 450m
roadCenters = [0 0 0; 450 0 0];
laneSpecification = lanespec(2, 'Width', 3.6);
road(scenario, roadCenters, 'Lanes', laneSpecification);

% 1. Ego Vehicle (Traveling straight in right lane at 40 km/h ~ 11.11 m/s, travels ~222m in 20s)
ego = vehicle(scenario, ...
    'ClassID', 1, ...
    'Position', [0 -1.8 0]);
egoWaypoints = [0 -1.8 0; 300 -1.8 0];
egoSpeed = 40 * (1000 / 3600);
trajectory(ego, egoWaypoints, egoSpeed);

% 2. Passing Car (Starts behind ego in left lane, moves faster at 54 km/h ~ 15.0 m/s, passes ego on the left)
car = vehicle(scenario, ...
    'ClassID', 1, ...
    'Position', [-30 1.8 0]);
carWaypoints = [-30 1.8 0; 360 1.8 0];
carSpeed = 54 * (1000 / 3600);
trajectory(car, carWaypoints, carSpeed);

% 3. Auto-Rickshaw (Erratic actor swerving across lanes 3 times over 20s at 36 km/h ~ 10 m/s)
autoRickshaw = vehicle(scenario, ...
    'ClassID', 1, ...
    'Length', 2.6, ...
    'Width', 1.3, ...
    'Height', 1.7, ...
    'Position', [35 -4.0 0]);

rickshawWaypoints = [
    35  -4.0  0;   % t = 0s: Starts on shoulder ahead of ego
    65  -1.8  0;   % t = 3s: Swerves into right lane
    95   1.8  0;   % t = 6s: Swerves into left lane
   135   1.8  0;   % t = 10s: Cruises in left lane
   165  -1.8  0;   % t = 13s: Swerves back into right lane
   195  -1.8  0;   % t = 16s: Continues in right lane
   240   1.8  0    % t = 20s: Cuts back into left lane
];
rickshawSpeed = 36 * (1000 / 3600);
trajectory(autoRickshaw, rickshawWaypoints, rickshawSpeed);

% Optional: Add Road Hazards / Potholes (low-profile static actors)
if enablePothole
    actor(scenario, 'ClassID', 4, 'Length', 1.0, 'Width', 1.0, 'Height', 0.05, 'Position', [100 -1.8 0]);
    actor(scenario, 'ClassID', 4, 'Length', 1.0, 'Width', 1.0, 'Height', 0.05, 'Position', [220 -1.8 0]);
end

%% 3. Sensor Configuration (Radar + Vision)
profiles = actorProfiles(scenario);

% Create 2 Radars (Front and Rear)
radars = cell(2, 1);
radars{1} = drivingRadarDataGenerator('SensorIndex', 1, ...
    'MountingLocation', [3.7 0 0.2], ...  % Front bumper
    'FieldOfView', [60, 5], ...
    'RangeLimits', [0 100], ...
    'AzimuthResolution', 4, ...
    'Profiles', profiles);

radars{2} = drivingRadarDataGenerator('SensorIndex', 2, ...
    'MountingLocation', [-1 0 0.2], ...   % Rear bumper
    'MountingAngles', [180 0 0], ...
    'FieldOfView', [60, 5], ...
    'RangeLimits', [0 100], ...
    'AzimuthResolution', 4, ...
    'Profiles', profiles);

% Create Front Vision Sensor
visionSensor = visionDetectionGenerator('SensorIndex', 3, ...
    'SensorLocation', [1.9 0], ...        % Front windshield [x y]
    'Height', 1.5, ...
    'MaxRange', 80, ...
    'DetectionProbability', 0.9, ...
    'ActorProfiles', profiles);

% Apply Sensor Degradation if enabled
if degradeSensors
    radars{1}.AzimuthResolution = 15;
    radars{2}.AzimuthResolution = 15;
    visionSensor.DetectionProbability = 0.4;
end

% Apply Realistic AV Multi-Rate Capture if enabled
if useRealisticAVRates
    radars{1}.UpdateRate = 20;          % 20 Hz Front Radar (50 ms)
    radars{2}.UpdateRate = 15;          % 15 Hz Rear Radar (66.7 ms)
    visionSensor.UpdateInterval = 0.10;  % 10 Hz DNN Object Detection (100 ms)
    if ~degradeSensors
        visionSensor.DetectionProbability = 0.70; % Realistic edge DNN detection probability
    end
end

%% 4. Multi-Object Tracker Setup (trackerGNN supports both KF & IMM)
if useIMM
    filterInit = @initDemoIMM;
else
    filterInit = @initDemoFilter;
end

tracker = trackerGNN(...
    'FilterInitializationFcn', filterInit, ...
    'AssignmentThreshold', 35, ...
    'ConfirmationThreshold', [2 3], ...
    'DeletionThreshold', [5 5]);

%% 5. Visualization Setup (Unified Single Bird's-Eye Figure)
bepFigure = figure('Name', 'Topic 4: Sensor Fusion & Tracking Benchmark', ...
    'NumberTitle', 'off', 'Color', 'k', 'Position', [100 100 1000 720]);
bepAxes = axes(bepFigure);
bep = birdsEyePlot('Parent', bepAxes, 'XLim', [-25 90], 'YLim', [-25 25]);

% Lane & Road Boundary Plotters
laneMarkingPlot = laneMarkingPlotter(bep);
lanePlotter = laneBoundaryPlotter(bep);

% Sensor Coverage Cones
cap = coverageAreaPlotter(bep);
for i = 1:numel(radars)
    plotCoverageArea(cap, radars{i}.MountingLocation(1:2), ...
        radars{i}.RangeLimits(2), radars{i}.MountingAngles(1), radars{i}.FieldOfView(1));
end
plotCoverageArea(cap, visionSensor.SensorLocation, ...
    visionSensor.MaxRange, visionSensor.Yaw, visionSensor.FieldOfView(1));

% Detection & Track Plotters
radarPlotter = detectionPlotter(bep, 'DisplayName', 'Radar Detections', ...
    'Marker', 'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
visionPlotter = detectionPlotter(bep, 'DisplayName', 'Vision Detections', ...
    'Marker', '^', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');
fusedTrackPlotter = trackPlotter(bep, 'DisplayName', 'Fused Tracks (Covariance Ellipse)', ...
    'Marker', 's', 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'HistoryDepth', 20);
egoPlotter = outlinePlotter(bep);

posSelector = [1 0 0 0 0 0; 0 0 1 0 0 0];

%% 6. Simulation & Tracking Loop
restart(scenario);

trackStats = struct('TotalConfirmed', 0, 'UniqueIDs', []);

% Initialize 20-second tracking error logs
trackingLogs = struct(...
    'Time', [], ...
    'RickshawGroundTruth', [], ...
    'RickshawEstimate', [], ...
    'RickshawError', [], ...
    'CarGroundTruth', [], ...
    'CarEstimate', [], ...
    'CarError', []);

confirmedTracks = objectTrack.empty(0, 1);

while advance(scenario)
    time = scenario.SimulationTime;

    % Target poses in ego vehicle coordinates
    tgtPoses = targetPoses(ego);

    % Generate synthetic sensor detections
    detections = {};
    isValidTime = false(3, 1);

    [rDets1, ~, isValidTime(1)] = radars{1}(tgtPoses, time);
    [rDets2, ~, isValidTime(2)] = radars{2}(tgtPoses, time);
    [vDets,  ~, isValidTime(3)] = visionSensor(tgtPoses, time);

    if isValidTime(1), detections = [detections; rDets1]; end %#ok<AGROW>
    if isValidTime(2), detections = [detections; rDets2]; end %#ok<AGROW>
    if isValidTime(3), detections = [detections; vDets];  end %#ok<AGROW>

    % Standardize ObjectAttributes to prevent heterogeneous concatenation issues
    for dIdx = 1:numel(detections)
        detections{dIdx}.ObjectAttributes = struct();
    end

    % Step tracker when sensor events occur
    if any(isValidTime)
        confirmedTracks = tracker(detections, time);
    end

    % Update plot: Fused Tracks
    if ~isempty(confirmedTracks)
        trackPositions = getTrackPositions(confirmedTracks, posSelector);
        numTracks = numel(confirmedTracks);
        trackCovariances = zeros(2, 2, numTracks);
        for k = 1:numTracks
            trackCovariances(:,:,k) = posSelector * confirmedTracks(k).StateCovariance * posSelector';
            trackStats.UniqueIDs = unique([trackStats.UniqueIDs, confirmedTracks(k).TrackID]);
        end
        plotTrack(fusedTrackPlotter, trackPositions, trackCovariances);
        trackStats.TotalConfirmed = max(trackStats.TotalConfirmed, numTracks);
    else
        trackPositions = zeros(0, 2);
        plotTrack(fusedTrackPlotter, zeros(0, 2), zeros(2, 2, 0));
    end

    % Capture Tracking Errors against Ground Truth across the 20 seconds
    if ~isempty(confirmedTracks)
        for p = 1:numel(tgtPoses)
            % 1. Auto-Rickshaw Tracking Error (Erratic 3-Swerve Actor)
            if tgtPoses(p).ActorID == autoRickshaw.ActorID
                r_true = tgtPoses(p).Position(1:2);
                dists_r = sqrt(sum((trackPositions - r_true).^2, 2));
                [min_dr, idx_r] = min(dists_r);
                if min_dr < 15
                    trackingLogs.Time(end+1) = time; %#ok<AGROW>
                    trackingLogs.RickshawGroundTruth = [trackingLogs.RickshawGroundTruth; r_true]; %#ok<AGROW>
                    trackingLogs.RickshawEstimate    = [trackingLogs.RickshawEstimate; trackPositions(idx_r,:)]; %#ok<AGROW>
                    trackingLogs.RickshawError(end+1) = min_dr; %#ok<AGROW>
                end
            end
            % 2. Passing Car Tracking Error (Left-Lane Fast Actor)
            if tgtPoses(p).ActorID == car.ActorID
                c_true = tgtPoses(p).Position(1:2);
                dists_c = sqrt(sum((trackPositions - c_true).^2, 2));
                [min_dc, idx_c] = min(dists_c);
                if min_dc < 15
                    trackingLogs.CarGroundTruth = [trackingLogs.CarGroundTruth; c_true]; %#ok<AGROW>
                    trackingLogs.CarEstimate    = [trackingLogs.CarEstimate; trackPositions(idx_c,:)]; %#ok<AGROW>
                    trackingLogs.CarError(end+1) = min_dc; %#ok<AGROW>
                end
            end
        end
    end

    % Update plot: Raw Radar Detections
    allRadars = [rDets1; rDets2];
    if ~isempty(allRadars)
        rPositions = cell2mat(cellfun(@(d) d.Measurement(1:2)', allRadars, 'UniformOutput', false));
        plotDetection(radarPlotter, rPositions);
    else
        plotDetection(radarPlotter, zeros(0, 2));
    end

    % Update plot: Raw Vision Detections
    if ~isempty(vDets)
        vPositions = cell2mat(cellfun(@(d) d.Measurement(1:2)', vDets, 'UniformOutput', false));
        plotDetection(visionPlotter, vPositions);
    else
        plotDetection(visionPlotter, zeros(0, 2));
    end

    % Plot Ego dimensions
    plotOutline(egoPlotter, [0 0], 0, ego.Length, ego.Width);

    % Update road boundaries & lane markings relative to moving ego
    [lbv, lbf] = laneMarkingVertices(ego);
    plotLaneMarking(laneMarkingPlot, lbv, lbf);
    rb = roadBoundaries(ego);
    plotLaneBoundary(lanePlotter, rb);

    % Render live frame
    drawnow limitrate;
    pause(0.02);
end

%% 7. 20-Second Tracking Error Benchmark & Summary Report
filterName = 'Standard Constant Velocity Kalman Filter';
if useIMM
    filterName = 'Interacting Multiple Model (IMM)';
end

noiseProfile = 'NOMINAL (Radar Res: 4 deg, Vision Prob: 0.9)';
if degradeSensors
    noiseProfile = 'DEGRADED (Radar Res: 15 deg, Vision Prob: 0.4)';
end

fprintf('\n==================================================================\n');
fprintf('         20-SECOND FUSED TRACKING ERROR BENCHMARK REPORT          \n');
fprintf('==================================================================\n');
fprintf(' Simulation Duration     : 20.0 seconds (400 time steps @ 20 Hz)\n');
fprintf(' Filter Configuration   : %s\n', filterName);
fprintf(' Sensor Conditions       : %s\n', noiseProfile);
fprintf(' Active Confirmed Tracks : %d (Unique IDs: %s)\n', ...
    trackStats.TotalConfirmed, mat2str(trackStats.UniqueIDs));
fprintf('------------------------------------------------------------------\n');

% Compute Auto-Rickshaw Tracking Statistics
if ~isempty(trackingLogs.RickshawError)
    mean_err_r = mean(trackingLogs.RickshawError);
    max_err_r  = max(trackingLogs.RickshawError);
    rmse_r     = sqrt(mean(trackingLogs.RickshawError.^2));
    fprintf(' Auto-Rickshaw (3 Aggressive Swerves Across Highway Lanes):\n');
    fprintf('   - Mean Position Error : %.3f meters\n', mean_err_r);
    fprintf('   - Peak Swerve Error   : %.3f meters\n', max_err_r);
    fprintf('   - Root Mean Sq Error  : %.3f meters\n', rmse_r);
else
    mean_err_r = NaN; max_err_r = NaN; rmse_r = NaN;
    fprintf(' Auto-Rickshaw: No confirmed track matches found.\n');
end

% Compute Passing Car Tracking Statistics
if ~isempty(trackingLogs.CarError)
    mean_err_c = mean(trackingLogs.CarError);
    max_err_c  = max(trackingLogs.CarError);
    rmse_c     = sqrt(mean(trackingLogs.CarError.^2));
    fprintf('\n Passing Car (Left Lane Constant Speed Overtake):\n');
    fprintf('   - Mean Position Error : %.3f meters\n', mean_err_c);
    fprintf('   - Peak Error          : %.3f meters\n', max_err_c);
    fprintf('   - Root Mean Sq Error  : %.3f meters\n', rmse_c);
else
    mean_err_c = NaN; max_err_c = NaN; rmse_c = NaN;
    fprintf('\n Passing Car: No confirmed track matches found.\n');
end
fprintf('==================================================================\n\n');

% Annotate Bird's-Eye Figure Title
title(bepAxes, sprintf('20s Simulation Complete | %s | RMSE: %.2fm', ...
    filterName, rmse_r), 'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');
saveas(bepFigure, 'topic4_simulation_result.png');
fprintf('[✓] Bird''s-eye view snapshot saved to: topic4_simulation_result.png\n');

%% 8. Generate & Save 20-Second Tracking Error Plot
errFig = figure('Name', 'Topic 4: 20-Second Tracking Error Analysis', ...
    'Color', [0.08 0.08 0.08], 'Position', [120 120 1100 620]);

% Subplot 1: Tracking Error Over Time
ax1 = subplot(2, 1, 1);
set(ax1, 'Color', [0.12 0.12 0.12], 'XColor', [0.9 0.9 0.9], 'YColor', [0.9 0.9 0.9], 'GridColor', [0.3 0.3 0.3]);
if ~isempty(trackingLogs.Time)
    plot(ax1, trackingLogs.Time, trackingLogs.RickshawError, 'r-', 'LineWidth', 1.8, ...
        'DisplayName', sprintf('Auto-Rickshaw (Mean: %.2fm, Peak: %.2fm)', mean_err_r, max_err_r));
    hold(ax1, 'on');
    if numel(trackingLogs.CarError) == numel(trackingLogs.Time)
        plot(ax1, trackingLogs.Time, trackingLogs.CarError, 'y--', 'LineWidth', 1.4, ...
            'DisplayName', sprintf('Passing Car (Mean: %.2fm, Peak: %.2fm)', mean_err_c, max_err_c));
    end
    grid(ax1, 'on');
    xlabel(ax1, 'Simulation Time (seconds)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'w');
    ylabel(ax1, 'Position Error (meters)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'w');
    title(ax1, sprintf('20-Second Tracking Error vs. Time (%s)', filterName), 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'w');
    legend(ax1, 'Location', 'northeast', 'FontSize', 9, 'TextColor', 'w', 'Color', [0.2 0.2 0.2], 'EdgeColor', [0.4 0.4 0.4]);
    xlim(ax1, [0 20]);
    
    % Annotate Swerve Phases
    y_lim = ylim(ax1);
    y_top = y_lim(2) * 0.90;
    text(ax1, 4.5, y_top, 'Swerve 1', 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', [0.9 0.9 0.9]);
    text(ax1, 12.5, y_top, 'Swerve 2', 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', [0.9 0.9 0.9]);
    text(ax1, 18.5, y_top, 'Swerve 3', 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', [0.9 0.9 0.9]);
end

% Subplot 2: Trajectory Tracking Profile (Ground Truth vs. Estimates)
ax2 = subplot(2, 1, 2);
set(ax2, 'Color', [0.12 0.12 0.12], 'XColor', [0.9 0.9 0.9], 'YColor', [0.9 0.9 0.9], 'GridColor', [0.3 0.3 0.3]);
if ~isempty(trackingLogs.RickshawGroundTruth)
    plot(ax2, trackingLogs.RickshawGroundTruth(:,1), trackingLogs.RickshawGroundTruth(:,2), 'c-', ...
        'LineWidth', 2.2, 'DisplayName', 'Rickshaw Ground Truth Path');
    hold(ax2, 'on');
    plot(ax2, trackingLogs.RickshawEstimate(:,1), trackingLogs.RickshawEstimate(:,2), 'm-.', ...
        'LineWidth', 1.8, 'DisplayName', sprintf('Estimated Tracks (%s)', filterName));
    grid(ax2, 'on');
    xlabel(ax2, 'Relative Longitudinal Position X (m)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'w');
    ylabel(ax2, 'Relative Lateral Position Y (m)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'w');
    title(ax2, '20-Second Relative Trajectory Tracking (Ground Truth vs. Estimator)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'w');
    legend(ax2, 'Location', 'best', 'FontSize', 9, 'TextColor', 'w', 'Color', [0.2 0.2 0.2], 'EdgeColor', [0.4 0.4 0.4]);
end

saveas(errFig, 'topic4_tracking_error_20s.png');
fprintf('[✓] 20-second tracking error plot saved to: topic4_tracking_error_20s.png\n');

% Package and export structured tracking results
trackingResults = struct(...
    'Filter', filterName, ...
    'Duration_s', 20.0, ...
    'SensorDegradation', degradeSensors, ...
    'Time', trackingLogs.Time, ...
    'RickshawError', trackingLogs.RickshawError, ...
    'RickshawMeanError', mean_err_r, ...
    'RickshawPeakError', max_err_r, ...
    'RickshawRMSE', rmse_r, ...
    'CarError', trackingLogs.CarError, ...
    'CarMeanError', mean_err_c, ...
    'CarPeakError', max_err_c, ...
    'CarRMSE', rmse_c);

assignin('base', 'trackingResults', trackingResults);
save('topic4_tracking_results_20s.mat', 'trackingResults');
fprintf('[✓] Results structure exported to base workspace: "trackingResults"\n');
fprintf('    and saved to: topic4_tracking_results_20s.mat\n\n');

%% Helper Filter Initialization Functions
function filter = initDemoFilter(detection)
% Constant Velocity 6-state Kalman Filter
% State vector: [x; vx; y; vy; z; vz]
H = [1 0 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 0 1 0;
    0 1 0 0 0 0;
    0 0 0 1 0 0;
    0 0 0 0 0 1];
filter = trackingKF('MotionModel', '3D Constant Velocity', ...
    'State', H' * detection.Measurement, ...
    'MeasurementModel', H, ...
    'ProcessNoise', 0.5 * eye(3), ...
    'StateCovariance', H' * detection.MeasurementNoise * H, ...
    'MeasurementNoise', detection.MeasurementNoise);
end

function filter = initDemoIMM(detection)
% Interacting Multiple Model (IMM) Filter
% Model 1: Low process noise (0.5 m/s^2) for smooth lane cruising
% Model 2: High process noise (20 m/s^2) for erratic diagonal swerves
H = [1 0 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 0 1 0;
    0 1 0 0 0 0;
    0 0 0 1 0 0;
    0 0 0 0 0 1];

kf1 = trackingKF('MotionModel', '3D Constant Velocity', ...
    'State', H' * detection.Measurement, ...
    'MeasurementModel', H, ...
    'ProcessNoise', 0.5 * eye(3), ...
    'StateCovariance', H' * detection.MeasurementNoise * H, ...
    'MeasurementNoise', detection.MeasurementNoise);

kf2 = trackingKF('MotionModel', '3D Constant Velocity', ...
    'State', H' * detection.Measurement, ...
    'MeasurementModel', H, ...
    'ProcessNoise', 20 * eye(3), ...
    'StateCovariance', H' * detection.MeasurementNoise * H, ...
    'MeasurementNoise', detection.MeasurementNoise);

filter = trackingIMM('TrackingFilters', {kf1, kf2}, ...
    'ModelProbabilities', [0.8 0.2], ...
    'TransitionProbabilities', [0.95 0.05; 0.10 0.90]);
end
