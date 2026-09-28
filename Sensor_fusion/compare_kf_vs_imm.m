%% Compare Standard Kalman Filter vs. IMM Filter under Sensor Degradation (20 Seconds)
% SIH 2026 - Problem Statement SIH26037
% Captures tracking error, trajectory lag, and filter accuracy for 20 seconds
% across multiple erratic swerve maneuvers.

clear; close all; clc;

fprintf('==================================================================\n');
fprintf('   Comparative Benchmark: Standard KF vs. IMM Filter (20 Seconds) \n');
fprintf('   (Both evaluated under degraded radar noise and vision dropouts)\n');
fprintf('==================================================================\n\n');

% Run Standard Kalman Filter (20s)
fprintf('>> [1/2] Simulating Standard Constant Velocity Kalman Filter (20s)...\n');
[time_kf, err_kf, true_kf, est_kf] = runSimulation(false);

% Run Interacting Multiple Model (IMM) Filter (20s)
fprintf('>> [2/2] Simulating Interacting Multiple Model (IMM) Filter (20s)...\n');
[time_imm, err_imm, true_imm, est_imm] = runSimulation(true);

%% Compute Comprehensive Metrics over 20 seconds
mean_err_kf   = mean(err_kf);
max_err_kf    = max(err_kf);
rmse_kf       = sqrt(mean(err_kf.^2));

mean_err_imm  = mean(err_imm);
max_err_imm   = max(err_imm);
rmse_imm      = sqrt(mean(err_imm.^2));

reduction_mean = (1 - mean_err_imm / mean_err_kf) * 100;
reduction_rmse = (1 - rmse_imm / rmse_kf) * 100;

fprintf('\n==================================================================\n');
fprintf('                20-SECOND BENCHMARK RESULTS                       \n');
fprintf('==================================================================\n');
fprintf(' Simulation Duration     : 20.0 seconds (400 time steps @ 20 Hz)\n');
fprintf(' Maneuver Profile        : 3 distinct aggressive swerves across lanes\n');
fprintf(' Sensor Conditions       : Radar Azimuth = 15 deg | Vision Prob = 0.4\n');
fprintf('------------------------------------------------------------------\n');
fprintf(' Standard Kalman Filter:\n');
fprintf('   - Mean Position Error : %.3f meters\n', mean_err_kf);
fprintf('   - Peak Swerve Error   : %.3f meters\n', max_err_kf);
fprintf('   - Root Mean Sq Error  : %.3f meters\n\n', rmse_kf);
fprintf(' Interacting Multiple Model (IMM):\n');
fprintf('   - Mean Position Error : %.3f meters\n', mean_err_imm);
fprintf('   - Peak Swerve Error   : %.3f meters\n', max_err_imm);
fprintf('   - Root Mean Sq Error  : %.3f meters\n\n', rmse_imm);
fprintf(' [★] Mean Error Difference: %.2f%% \n', reduction_mean);
fprintf('==================================================================\n\n');

%% Visual Comparison Plots (20 Seconds)
fig = figure('Name', '20-Second Tracking Error Comparison: Standard KF vs. IMM', ...
       'Color', [0.08 0.08 0.08], 'Position', [80 80 1200 620]);

% Subplot 1: Full 20-Second Relative Trajectory
ax1 = subplot(1, 2, 1);
set(ax1, 'Color', [0.12 0.12 0.12], 'XColor', [0.9 0.9 0.9], 'YColor', [0.9 0.9 0.9], 'GridColor', [0.3 0.3 0.3]);
plot(ax1, true_kf(:,1), true_kf(:,2), 'c-', 'LineWidth', 2.2, 'DisplayName', 'Ground Truth Path');
hold(ax1, 'on');
plot(ax1, est_kf(:,1), est_kf(:,2), 'r--', 'LineWidth', 1.6, 'DisplayName', sprintf('Standard KF (Mean: %.2fm)', mean_err_kf));
plot(ax1, est_imm(:,1), est_imm(:,2), 'g-.', 'LineWidth', 1.8, 'DisplayName', sprintf('IMM Filter (Mean: %.2fm)', mean_err_imm));
grid(ax1, 'on');
xlabel(ax1, 'Relative Longitudinal X (m)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'w');
ylabel(ax1, 'Relative Lateral Y (m)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'w');
title(ax1, '20s Multi-Swerve Trajectory', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'w');
legend(ax1, 'Location', 'best', 'FontSize', 10, 'TextColor', 'w', 'Color', [0.2 0.2 0.2], 'EdgeColor', [0.4 0.4 0.4]);

% Subplot 2: Position Error Across All 20 Seconds
ax2 = subplot(1, 2, 2);
set(ax2, 'Color', [0.12 0.12 0.12], 'XColor', [0.9 0.9 0.9], 'YColor', [0.9 0.9 0.9], 'GridColor', [0.3 0.3 0.3]);
plot(ax2, time_kf, err_kf, 'r-', 'LineWidth', 1.6, 'DisplayName', sprintf('Standard KF Error (RMSE: %.2fm)', rmse_kf));
hold(ax2, 'on');
plot(ax2, time_imm, err_imm, 'g-', 'LineWidth', 1.8, 'DisplayName', sprintf('IMM Filter Error (RMSE: %.2fm)', rmse_imm));
grid(ax2, 'on');
xlabel(ax2, 'Simulation Time (seconds)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'w');
ylabel(ax2, 'Position Error (meters)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'w');
title(ax2, 'Tracking Error vs. Time (0 - 20s)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'w');
legend(ax2, 'Location', 'northeast', 'FontSize', 10, 'TextColor', 'w', 'Color', [0.2 0.2 0.2], 'EdgeColor', [0.4 0.4 0.4]);
xlim(ax2, [0 20]);

% Annotate swerve phases
y_lim = ylim(ax2);
y_pos = y_lim(2) * 0.90;
text(ax2, 4.5, y_pos, 'Swerve 1', 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', [0.9 0.9 0.9]);
text(ax2, 12.5, y_pos, 'Swerve 2', 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', [0.9 0.9 0.9]);
text(ax2, 18.5, y_pos, 'Swerve 3', 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', [0.9 0.9 0.9]);

% Save high-res artifacts
saveas(fig, 'kf_vs_imm_comparison_20s.png');
saveas(fig, 'kf_vs_imm_comparison.png');
fprintf('[✓] 20-second comparison plots saved to:\n');
fprintf('    - kf_vs_imm_comparison_20s.png\n');
fprintf('    - kf_vs_imm_comparison.png\n\n');

% Export captured 20-second error data to .mat file and base workspace
save('kf_vs_imm_errors_20s.mat', 'time_kf', 'err_kf', 'time_imm', 'err_imm', ...
    'mean_err_kf', 'max_err_kf', 'rmse_kf', 'mean_err_imm', 'max_err_imm', 'rmse_imm');
assignin('base', 'time_kf', time_kf);
assignin('base', 'err_kf', err_kf);
assignin('base', 'time_imm', time_imm);
assignin('base', 'err_imm', err_imm);
fprintf('[✓] 20-second error data saved to: kf_vs_imm_errors_20s.mat\n');
fprintf('    and assigned to workspace variables: time_kf, err_kf, time_imm, err_imm\n\n');

%% 20-Second Simulation Function
function [time_log, err_log, true_log, est_log] = runSimulation(useIMM)
    % 20-second scenario with 450m extended highway
    s = drivingScenario('StopTime', 20.0, 'SampleTime', 0.05);
    road(s, [0 0 0; 450 0 0], 'Lanes', lanespec(2, 'Width', 3.6));
    
    % Ego vehicle: travels ~222m in 20s at 40 km/h (11.11 m/s)
    ego = vehicle(s, 'Position', [0 -1.8 0]);
    trajectory(ego, [0 -1.8 0; 300 -1.8 0], 40 * (1000 / 3600));
    
    % Auto-Rickshaw: travels ~200m in 20s at 36 km/h (10.0 m/s) with 3 lane swerves
    rickshaw = vehicle(s, 'Length', 2.6, 'Width', 1.3, 'Height', 1.7, 'Position', [35 -4.0 0]);
    r_waypoints = [
        35  -4.0  0;   % t = 0s: Starts on shoulder ahead of ego
        65  -1.8  0;   % t = 3s: Swerves into right lane
        95   1.8  0;   % t = 6s: Swerves into left lane
       135   1.8  0;   % t = 10s: Cruises in left lane
       165  -1.8  0;   % t = 13s: Swerves back into right lane
       195  -1.8  0;   % t = 16s: Continues in right lane
       240   1.8  0    % t = 20s: Cuts back into left lane
    ];
    trajectory(rickshaw, r_waypoints, 36 * (1000 / 3600));

    % Sensors with realistic degradation
    prof = actorProfiles(s);
    r1 = drivingRadarDataGenerator('SensorIndex', 1, 'MountingLocation', [3.7 0 0.2], ...
        'FieldOfView', [60, 5], 'AzimuthResolution', 15, 'Profiles', prof);
    v  = visionDetectionGenerator('SensorIndex', 3, 'SensorLocation', [1.9 0], ...
        'Height', 1.5, 'MaxRange', 80, 'DetectionProbability', 0.4, 'ActorProfiles', prof);

    if useIMM
        fInit = @initIMM;
    else
        fInit = @initKF;
    end
    
    tracker = trackerGNN('FilterInitializationFcn', fInit, ...
        'AssignmentThreshold', 35, 'ConfirmationThreshold', [2 3], 'DeletionThreshold', [5 5]);
    posSel = [1 0 0 0 0 0; 0 0 1 0 0 0];

    time_log = [];
    err_log  = [];
    true_log = [];
    est_log  = [];

    while advance(s)
        time = s.SimulationTime;
        tgtPoses = targetPoses(ego);
        [rd1,~,v1] = r1(tgtPoses, time);
        [vd, ~,v3] = v(tgtPoses, time);
        dets = {};
        if v1, dets = [dets; rd1]; end
        if v3, dets = [dets; vd]; end
        for d = 1:numel(dets), dets{d}.ObjectAttributes = struct(); end
        
        cTracks = tracker(dets, time);

        % Compute error against auto-rickshaw ground truth
        for p = 1:numel(tgtPoses)
            if tgtPoses(p).ActorID == rickshaw.ActorID
                r_true = tgtPoses(p).Position(1:2);
                if ~isempty(cTracks)
                    tPositions = getTrackPositions(cTracks, posSel);
                    dists = sqrt(sum((tPositions - r_true).^2, 2));
                    [min_d, idx] = min(dists);
                    if min_d < 15
                        err_log(end+1)  = min_d; %#ok<AGROW>
                        time_log(end+1) = time;  %#ok<AGROW>
                        true_log = [true_log; r_true]; %#ok<AGROW>
                        est_log  = [est_log; tPositions(idx,:)]; %#ok<AGROW>
                    end
                end
            end
        end
    end
end

function f = initKF(d)
    H = [1 0 0 0 0 0; 0 0 1 0 0 0; 0 0 0 0 1 0; 0 1 0 0 0 0; 0 0 0 1 0 0; 0 0 0 0 0 1];
    f = trackingKF('MotionModel', '3D Constant Velocity', ...
        'State', H'*d.Measurement, 'MeasurementModel', H, ...
        'ProcessNoise', 0.5 * eye(3), ...
        'StateCovariance', H'*d.MeasurementNoise*H, 'MeasurementNoise', d.MeasurementNoise);
end

function f = initIMM(d)
    H = [1 0 0 0 0 0; 0 0 1 0 0 0; 0 0 0 0 1 0; 0 1 0 0 0 0; 0 0 0 1 0 0; 0 0 0 0 0 1];
    k1 = trackingKF('MotionModel', '3D Constant Velocity', ...
        'State', H'*d.Measurement, 'MeasurementModel', H, ...
        'ProcessNoise', 0.5 * eye(3), ...
        'StateCovariance', H'*d.MeasurementNoise*H, 'MeasurementNoise', d.MeasurementNoise);
    k2 = trackingKF('MotionModel', '3D Constant Velocity', ...
        'State', H'*d.Measurement, 'MeasurementModel', H, ...
        'ProcessNoise', 20 * eye(3), ...
        'StateCovariance', H'*d.MeasurementNoise*H, 'MeasurementNoise', d.MeasurementNoise);
    f = trackingIMM('TrackingFilters', {k1, k2}, ...
        'ModelProbabilities', [0.8 0.2], ...
        'TransitionProbabilities', [0.95 0.05; 0.10 0.90]);
end
