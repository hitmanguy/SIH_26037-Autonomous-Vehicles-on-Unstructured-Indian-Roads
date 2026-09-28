%% Realistic Autonomous Vehicle Capture Rates: Tracking Error Benchmark (20 Seconds)
% Problem Statement: SIH26037 - Sensor Fusion & Multi-Object Tracking
%
% Real Autonomous Vehicle Sensor Architecture:
%  - Base / Ego Clock        : 50 Hz (20 ms sample time)
%  - Front Radar             : 20 Hz (50 ms update rate, high range precision, 12 deg azimuth noise)
%  - Rear Radar              : 15 Hz (66.7 ms update rate)
%  - Vision Object Detection : 10 Hz (100 ms interval, representing edge DNN inference latency)
%  - Vision Dropouts         : Pd = 0.65 (realistic glare, dust, occlusion on Indian roads)
%  - Evaluates Standard Constant Velocity Kalman Filter vs. IMM Filter across 20s.

clear; close all; clc;

fprintf('==================================================================\n');
fprintf('==================================================================\n');
fprintf('   REALISTIC AUTONOMOUS VEHICLE MULTI-RATE TRACKING BENCHMARK     \n');
fprintf('==================================================================\n');
fprintf(' Base Scenario Clock   : 100 Hz (10 ms automotive CAN/IMU clock)\n');
fprintf(' Front Radar           : 20 Hz (50 ms update interval)\n');
fprintf(' Rear Radar            : 10 Hz (100 ms update interval)\n');
fprintf(' Vision Camera (DNN)   : 10 Hz (100 ms inference interval, Pd = 0.65)\n');
fprintf(' Simulation Duration   : 20.0 seconds (2000 base clock cycles)\n');
fprintf(' Maneuver Profile      : Auto-Rickshaw 3 erratic swerves across lanes\n');
fprintf('==================================================================\n\n');

%% 1. Build Driving Scenario (100 Hz Base Clock, 450m Highway)
% 100 Hz (10 ms) is the exact common divisor for 20 Hz Radar (50 ms) & 10 Hz Camera (100 ms)
scenario = drivingScenario('StopTime', 20.0, 'SampleTime', 0.01);
road(scenario, [0 0 0; 450 0 0], 'Lanes', lanespec(2, 'Width', 3.6));

% Ego Vehicle: 40 km/h (11.11 m/s) in right lane
ego = vehicle(scenario, 'Position', [0 -1.8 0]);
trajectory(ego, [0 -1.8 0; 300 -1.8 0], 40 * (1000 / 3600));

% Auto-Rickshaw: 36 km/h (10.0 m/s) with 3 aggressive swerves
rickshaw = vehicle(scenario, 'Length', 2.6, 'Width', 1.3, 'Height', 1.7, 'Position', [35 -4.0 0]);
r_waypoints = [
    35  -4.0  0;   % t = 0s: Starts on shoulder
    65  -1.8  0;   % t = 3s: Swerves into right lane
    95   1.8  0;   % t = 6s: Swerves into left lane
   135   1.8  0;   % t = 10s: Cruises in left lane
   165  -1.8  0;   % t = 13s: Swerves back into right lane
   195  -1.8  0;   % t = 16s: Continues in right lane
   240   1.8  0    % t = 20s: Cuts back into left lane
];
trajectory(rickshaw, r_waypoints, 36 * (1000 / 3600));

% Passing Car: 54 km/h (15.0 m/s) in left lane
car = vehicle(scenario, 'Position', [-30 1.8 0]);
trajectory(car, [-30 1.8 0; 360 1.8 0], 54 * (1000 / 3600));

profiles = actorProfiles(scenario);

%% 2. Configure Multi-Rate Sensors (Realistic AV Specifications)
% Front Radar: 20 Hz (50 ms update period)
radarFront = drivingRadarDataGenerator('SensorIndex', 1, ...
    'MountingLocation', [3.7 0 0.2], ...
    'FieldOfView', [60, 5], ...
    'RangeLimits', [0 100], ...
    'UpdateRate', 20, ...          % 20 Hz automotive radar (fires every 50 ms)
    'AzimuthResolution', 12, ...   % Realistic angular uncertainty
    'Profiles', profiles);

% Rear Radar: 10 Hz (100 ms update period)
radarRear = drivingRadarDataGenerator('SensorIndex', 2, ...
    'MountingLocation', [-1 0 0.2], ...
    'MountingAngles', [180 0 0], ...
    'FieldOfView', [60, 5], ...
    'RangeLimits', [0 100], ...
    'UpdateRate', 10, ...          % 10 Hz rear radar
    'AzimuthResolution', 15, ...
    'Profiles', profiles);

% Front Windshield Camera: 10 Hz Object Detection (Edge DNN Inference Rate)
camera = visionDetectionGenerator('SensorIndex', 3, ...
    'SensorLocation', [1.9 0], ...
    'Height', 1.5, ...
    'MaxRange', 80, ...
    'UpdateInterval', 0.10, ...    % 10 Hz DNN inference interval (100 ms)
    'DetectionProbability', 0.65, ... % Realistic camera dropout under Indian road conditions
    'ActorProfiles', profiles);

%% 3. Setup Estimators (Standard KF vs. IMM Filter)
% State vector: [x; vx; y; vy; z; vz]
H = [1 0 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 0 1 0;
    0 1 0 0 0 0;
    0 0 0 1 0 0;
    0 0 0 0 0 1];

% Standard Constant Velocity Kalman Filter
fInitKF = @(d) trackingKF('MotionModel', '3D Constant Velocity', ...
    'State', H' * d.Measurement, ...
    'MeasurementModel', H, ...
    'ProcessNoise', 0.5 * eye(3), ...
    'StateCovariance', H' * d.MeasurementNoise * H, ...
    'MeasurementNoise', d.MeasurementNoise);

% Interacting Multiple Model (IMM) Filter
% Model 1: Low process noise (0.5 m/s^2) for steady straight driving
% Model 2: High process noise (25 m/s^2) for aggressive swerving / lane changes
fInitIMM = @(d) trackingIMM('TrackingFilters', { ...
    trackingKF('MotionModel', '3D Constant Velocity', 'State', H'*d.Measurement, 'MeasurementModel', H, 'ProcessNoise', 0.5*eye(3), 'MeasurementNoise', d.MeasurementNoise), ...
    trackingKF('MotionModel', '3D Constant Velocity', 'State', H'*d.Measurement, 'MeasurementModel', H, 'ProcessNoise', 25*eye(3), 'MeasurementNoise', d.MeasurementNoise)}, ...
    'ModelProbabilities', [0.8 0.2], ...
    'TransitionProbabilities', [0.95 0.05; 0.10 0.90]);

trackerKF  = trackerGNN('FilterInitializationFcn', fInitKF,  'AssignmentThreshold', 35, 'ConfirmationThreshold', [2 3], 'DeletionThreshold', [5 5]);
trackerIMM = trackerGNN('FilterInitializationFcn', fInitIMM, 'AssignmentThreshold', 35, 'ConfirmationThreshold', [2 3], 'DeletionThreshold', [5 5]);

posSelector = [1 0 0 0 0 0; 0 0 1 0 0 0];

%% 4. Multi-Rate Simulation Loop
time_log      = [];
err_kf_log    = [];
err_imm_log   = [];
true_path_log = [];
est_kf_path   = [];
est_imm_path  = [];
sensor_events = []; % [time, has_radarFront, has_radarRear, has_camera]

fprintf('>> Running 20-second multi-rate simulation (1000 ticks @ 50 Hz)...\n');

while advance(scenario)
    time = scenario.SimulationTime;
    tgtPoses = targetPoses(ego);

    % Query sensors with their independent internal update rates
    [rFrontDets, ~, isRadarFrontValid] = radarFront(tgtPoses, time);
    [rRearDets,  ~, isRadarRearValid]  = radarRear(tgtPoses, time);
    [camDets,    ~, isCamValid]        = camera(tgtPoses, time);

    % Record sensor firing events
    sensor_events = [sensor_events; time, isRadarFrontValid, isRadarRearValid, isCamValid]; %#ok<AGROW>

    % Aggregate all detections arriving at this specific millisecond
    detections = {};
    if isRadarFrontValid, detections = [detections; rFrontDets]; end %#ok<AGROW>
    if isRadarRearValid,  detections = [detections; rRearDets];  end %#ok<AGROW>
    if isCamValid,        detections = [detections; camDets];    end %#ok<AGROW>

    for d = 1:numel(detections)
        detections{d}.ObjectAttributes = struct();
    end

    % Step trackers only when sensor events occur
    if isRadarFrontValid || isRadarRearValid || isCamValid
        tracksKF  = trackerKF(detections, time);
        tracksIMM = trackerIMM(detections, time);

        % Compute error against auto-rickshaw ground truth
        for p = 1:numel(tgtPoses)
            if tgtPoses(p).ActorID == rickshaw.ActorID
                r_true = tgtPoses(p).Position(1:2);

                % Standard KF Error
                if ~isempty(tracksKF)
                    posKF = getTrackPositions(tracksKF, posSelector);
                    distsKF = sqrt(sum((posKF - r_true).^2, 2));
                    [min_kf, idx_kf] = min(distsKF);
                else
                    min_kf = NaN; idx_kf = 1;
                end

                % IMM Filter Error
                if ~isempty(tracksIMM)
                    posIMM = getTrackPositions(tracksIMM, posSelector);
                    distsIMM = sqrt(sum((posIMM - r_true).^2, 2));
                    [min_imm, idx_imm] = min(distsIMM);
                else
                    min_imm = NaN; idx_imm = 1;
                end

                if ~isnan(min_kf) && ~isnan(min_imm) && min_kf < 15 && min_imm < 15
                    time_log(end+1)    = time; %#ok<AGROW>
                    err_kf_log(end+1)  = min_kf; %#ok<AGROW>
                    err_imm_log(end+1) = min_imm; %#ok<AGROW>
                    true_path_log = [true_path_log; r_true]; %#ok<AGROW>
                    est_kf_path   = [est_kf_path; posKF(idx_kf,:)]; %#ok<AGROW>
                    est_imm_path  = [est_imm_path; posIMM(idx_imm,:)]; %#ok<AGROW>
                end
            end
        end
    end
end

%% 5. Compute Comprehensive Metrics
mean_kf  = mean(err_kf_log);
max_kf   = max(err_kf_log);
rmse_kf  = sqrt(mean(err_kf_log.^2));

mean_imm = mean(err_imm_log);
max_imm  = max(err_imm_log);
rmse_imm = sqrt(mean(err_imm_log.^2));

reduction_mean = (1 - mean_imm / mean_kf) * 100;
reduction_max  = (1 - max_imm / max_kf) * 100;
reduction_rmse = (1 - rmse_imm / rmse_kf) * 100;

fprintf('\n==================================================================\n');
fprintf('     REALISTIC AV MULTI-RATE TRACKING ERROR BENCHMARK RESULTS     \n');
fprintf('==================================================================\n');
fprintf(' Simulation Duration     : 20.0 seconds (2000 base steps @ 100 Hz)\n');
fprintf(' Sensor Configuration    : Radar (20 Hz, 12° Az) + Vision DNN (10 Hz, Pd=0.65)\n');
fprintf(' Maneuver Profile        : 3 aggressive lane-cross swerves\n');
fprintf(' Total Evaluated Steps   : %d asynchronous sensor arrivals\n', numel(time_log));
fprintf('------------------------------------------------------------------\n');
fprintf(' Standard Constant Velocity Kalman Filter:\n');
fprintf('   - Mean Position Error : %.3f meters\n', mean_kf);
fprintf('   - Peak Swerve Error   : %.3f meters\n', max_kf);
fprintf('   - Root Mean Sq Error  : %.3f meters\n\n', rmse_kf);
fprintf(' Interacting Multiple Model (IMM) Filter:\n');
fprintf('   - Mean Position Error : %.3f meters\n', mean_imm);
fprintf('   - Peak Swerve Error   : %.3f meters\n', max_imm);
fprintf('   - Root Mean Sq Error  : %.3f meters\n\n', rmse_imm);
fprintf(' [★] Mean Error Difference: %+.2f%%\n', reduction_mean);
fprintf(' [★] Peak Lag Reduction  : %+.2f%%\n', reduction_max);
fprintf(' [★] RMSE Improvement    : %+.2f%%\n', reduction_rmse);
fprintf('==================================================================\n\n');

%% 6. Generate Publication-Quality Visualizations
fig = figure('Name', 'Realistic Autonomous Vehicle Multi-Rate Tracking Benchmark', ...
    'Color', [0.08 0.08 0.08], 'Position', [60 60 1280 720]);

% Panel 1: Multi-Rate Sensor Arrival Timeline (0 - 2 seconds zoom for clear spacing)
ax1 = subplot(3, 1, 1);
set(ax1, 'Color', [0.12 0.12 0.12], 'XColor', [0.9 0.9 0.9], 'YColor', [0.9 0.9 0.9], 'GridColor', [0.3 0.3 0.3]);
zoom_mask = sensor_events(:,1) <= 2.0;
t_zoom = sensor_events(zoom_mask, 1);
stem(ax1, t_zoom(sensor_events(zoom_mask,2) == 1), repmat(2.0, sum(sensor_events(zoom_mask,2) == 1), 1), ...
    'r', 'LineWidth', 1.4, 'Marker', 'o', 'MarkerSize', 5, 'DisplayName', 'Front Radar (20 Hz, 50ms)');
hold(ax1, 'on');
stem(ax1, t_zoom(sensor_events(zoom_mask,4) == 1), repmat(1.0, sum(sensor_events(zoom_mask,4) == 1), 1), ...
    'c', 'LineWidth', 1.6, 'Marker', '^', 'MarkerSize', 7, 'DisplayName', 'Camera DNN (10 Hz, 100ms, Pd=0.65)');
grid(ax1, 'on');
xlim(ax1, [0 2.0]);
ylim(ax1, [0 2.8]);
yticks(ax1, [1.0, 2.0]);
yticklabels(ax1, {'Camera (10 Hz)', 'Radar (20 Hz)'});
xlabel(ax1, 'Simulation Time (seconds)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'w');
title(ax1, 'Realistic AV Multi-Rate Capture: 20 Hz Radar (every 50ms) vs. 10 Hz Camera DNN (every 100ms)', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'w');
legend(ax1, 'Location', 'northeast', 'FontSize', 9, 'TextColor', 'w', 'Color', [0.2 0.2 0.2], 'EdgeColor', [0.4 0.4 0.4]);

% Panel 2: Tracking Error Over the Full 20 Seconds
ax2 = subplot(3, 1, 2);
set(ax2, 'Color', [0.12 0.12 0.12], 'XColor', [0.9 0.9 0.9], 'YColor', [0.9 0.9 0.9], 'GridColor', [0.3 0.3 0.3]);
plot(ax2, time_log, err_kf_log, 'r-', 'LineWidth', 1.6, ...
    'DisplayName', sprintf('Standard KF (Mean: %.2fm, Peak: %.2fm)', mean_kf, max_kf));
hold(ax2, 'on');
plot(ax2, time_log, err_imm_log, 'g-', 'LineWidth', 1.8, ...
    'DisplayName', sprintf('IMM Filter (Mean: %.2fm, Peak: %.2fm)', mean_imm, max_imm));
grid(ax2, 'on');
xlim(ax2, [0 20]);
xlabel(ax2, 'Simulation Time (seconds)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'w');
ylabel(ax2, 'Tracking Error (m)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'w');
title(ax2, '20-Second Tracking Error vs. Time under Multi-Rate AV Capture', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'w');
legend(ax2, 'Location', 'northeast', 'FontSize', 9, 'TextColor', 'w', 'Color', [0.2 0.2 0.2], 'EdgeColor', [0.4 0.4 0.4]);

% Annotate swerves
y_top = max(err_kf_log) * 0.90;
text(ax2, 4.5, y_top, 'Swerve 1', 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', [0.9 0.9 0.9]);
text(ax2, 12.5, y_top, 'Swerve 2', 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', [0.9 0.9 0.9]);
text(ax2, 18.5, y_top, 'Swerve 3', 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', [0.9 0.9 0.9]);

% Panel 3: Relative Trajectory Tracking (Ground Truth vs. Filters)
ax3 = subplot(3, 1, 3);
set(ax3, 'Color', [0.12 0.12 0.12], 'XColor', [0.9 0.9 0.9], 'YColor', [0.9 0.9 0.9], 'GridColor', [0.3 0.3 0.3]);
plot(ax3, true_path_log(:,1), true_path_log(:,2), 'c-', 'LineWidth', 2.2, 'DisplayName', 'Ground Truth Path');
hold(ax3, 'on');
plot(ax3, est_kf_path(:,1), est_kf_path(:,2), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Standard KF Track');
plot(ax3, est_imm_path(:,1), est_imm_path(:,2), 'g-.', 'LineWidth', 1.8, 'DisplayName', 'IMM Filter Track');
grid(ax3, 'on');
xlabel(ax3, 'Relative Longitudinal Position X (m)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'w');
ylabel(ax3, 'Relative Lateral Position Y (m)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'w');
title(ax3, '20-Second Swerve Trajectory: Ground Truth vs. Multi-Rate Estimates', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'w');
legend(ax3, 'Location', 'best', 'FontSize', 9, 'TextColor', 'w', 'Color', [0.2 0.2 0.2], 'EdgeColor', [0.4 0.4 0.4]);

saveas(fig, 'realistic_av_tracking_error_20s.png');
fprintf('[✓] Benchmark comparison plot saved to: realistic_av_tracking_error_20s.png\n');

% Save error logs and metrics
avBenchmarkResults = struct(...
    'Time', time_log, ...
    'ErrorKF', err_kf_log, ...
    'ErrorIMM', err_imm_log, ...
    'MeanKF', mean_kf, 'MaxKF', max_kf, 'RMSE_KF', rmse_kf, ...
    'MeanIMM', mean_imm, 'MaxIMM', max_imm, 'RMSE_IMM', rmse_imm, ...
    'SensorEvents', sensor_events);

assignin('base', 'avBenchmarkResults', avBenchmarkResults);
save('realistic_av_tracking_errors_20s.mat', 'avBenchmarkResults');
fprintf('[✓] Error arrays exported to workspace: "avBenchmarkResults"\n');
fprintf('    and saved to: realistic_av_tracking_errors_20s.mat\n\n');
