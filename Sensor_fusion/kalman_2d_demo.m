clc;
clear;
close all;

%% Parameters
dt = 0.1;              % Sampling time (s)
N = 100;               % Number of time steps

%% State vector definition
% State: [position_x; position_y; velocity_x; velocity_y]
% Ground truth initial state
x_true = [0; 0; 2; 1];

% Initial state estimate (filter starts with slight initial offset)
x_est = [0; 0; 1.5; 0.5];

%% State transition matrix (Constant Velocity model)
F = [1 0 dt 0;
     0 1 0 dt;
     0 0 1  0;
     0 0 0  1];

%% Measurement matrix (measures [x; y] position only)
H = [1 0 0 0;
     0 1 0 0];

%% Noise Covariances
% Process noise covariance Q (represents small physical acceleration disturbances)
q_acc = 0.05;
G = [0.5*dt^2 0; 0 0.5*dt^2; dt 0; 0 dt]; % Acceleration gain matrix
Q = G * (q_acc * eye(2)) * G';

% Measurement noise covariance R (sensor variance)
R = [0.5  0;
     0    0.5];
chol_R = chol(R, 'lower');

%% Initial estimate covariance
P = eye(4);

%% Storage arrays for results
true_states  = zeros(4, N);
measurements = zeros(2, N);
estimates    = zeros(4, N);

%% Simulation Loop
for k = 1:N
    %% ----------------------------------------------------
    % 1. Physical System: Propagate Ground Truth State
    % ----------------------------------------------------
    % Ground truth propagates independently with physical acceleration disturbances
    a_dist = sqrt(q_acc) * randn(2, 1);
    w_k = G * a_dist;
    x_true = F * x_true + w_k;
    true_states(:, k) = x_true;

    %% ----------------------------------------------------
    % 2. Sensor Measurement
    % ----------------------------------------------------
    % Sensor observes ground truth with additive Gaussian noise
    v_k = chol_R * randn(2, 1);
    z = H * x_true + v_k;
    measurements(:, k) = z;

    %% ----------------------------------------------------
    % 3. Kalman Filter: Prediction Step
    % ----------------------------------------------------
    x_pred = F * x_est;
    P_pred = F * P * F' + Q;

    %% ----------------------------------------------------
    % 4. Kalman Filter: Compute Gain & Innovation
    % ----------------------------------------------------
    S = H * P_pred * H' + R;       % Innovation covariance
    K = (P_pred * H') / S;          % Kalman gain
    innovation = z - H * x_pred;    % Residual

    %% ----------------------------------------------------
    % 5. Kalman Filter: Correction (Update) Step
    % ----------------------------------------------------
    x_est = x_pred + K * innovation;
    % Joseph form update for guaranteed positive semi-definite covariance
    I_KH = eye(4) - K * H;
    P = I_KH * P_pred * I_KH' + K * R * K';

    % Store current estimate
    estimates(:, k) = x_est;
end

%% Quantitative Evaluation (RMSE)
meas_rmse_x = sqrt(mean((measurements(1,:) - true_states(1,:)).^2));
meas_rmse_y = sqrt(mean((measurements(2,:) - true_states(2,:)).^2));
meas_rmse_pos = sqrt(mean(sum((measurements - true_states(1:2,:)).^2, 1)));

kf_rmse_x = sqrt(mean((estimates(1,:) - true_states(1,:)).^2));
kf_rmse_y = sqrt(mean((estimates(2,:) - true_states(2,:)).^2));
kf_rmse_pos = sqrt(mean(sum((estimates(1:2,:) - true_states(1:2,:)).^2, 1)));

fprintf('=== 2D Kalman Filter Performance ===\n');
fprintf('Raw Sensor Measurement Position RMSE : %.3f m\n', meas_rmse_pos);
fprintf('Kalman Filter Estimated Position RMSE: %.3f m\n', kf_rmse_pos);
fprintf('Error Reduction                       : %.1f%%\n\n', ...
    (1 - kf_rmse_pos / meas_rmse_pos) * 100);

%% Visualization
figure('Name', '2D Kalman Filter Tracking Benchmark', 'NumberTitle', 'off', ...
       'Color', 'w', 'Position', [150 150 1000 650]);

subplot(2, 2, [1 3]);
plot(true_states(1,:), true_states(2,:), 'k-', 'LineWidth', 2, 'DisplayName', 'True Trajectory');
hold on;
plot(measurements(1,:), measurements(2,:), 'r.', 'MarkerSize', 8, 'DisplayName', 'Raw Noisy Measurements');
plot(estimates(1,:), estimates(2,:), 'b--', 'LineWidth', 2, 'DisplayName', 'Kalman Filter Estimate');
grid on;
axis equal;
xlabel('X Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Y Position (m)', 'FontSize', 11, 'FontWeight', 'bold');
title('2D Position Tracking Trajectory', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10);

% Subplot: Position Errors over Time
time_vec = (1:N) * dt;
raw_err = sqrt(sum((measurements - true_states(1:2,:)).^2, 1));
kf_err  = sqrt(sum((estimates(1:2,:) - true_states(1:2,:)).^2, 1));

subplot(2, 2, 2);
plot(time_vec, raw_err, 'r:', 'LineWidth', 1.2, 'DisplayName', 'Raw Measurement Error');
hold on;
plot(time_vec, kf_err, 'b-', 'LineWidth', 1.8, 'DisplayName', 'Kalman Filter Error');
grid on;
xlabel('Time (s)', 'FontSize', 10);
ylabel('Position Error (m)', 'FontSize', 10);
title('Tracking Error vs. Time', 'FontSize', 11, 'FontWeight', 'bold');
legend('FontSize', 9);

% Subplot: Estimated Velocities
subplot(2, 2, 4);
plot(time_vec, true_states(3,:), 'k-', 'LineWidth', 1.5, 'DisplayName', 'True Vx');
hold on;
plot(time_vec, estimates(3,:), 'b--', 'LineWidth', 1.5, 'DisplayName', 'Estimated Vx');
plot(time_vec, true_states(4,:), 'k:', 'LineWidth', 1.5, 'DisplayName', 'True Vy');
plot(time_vec, estimates(4,:), 'm--', 'LineWidth', 1.5, 'DisplayName', 'Estimated Vy');
grid on;
xlabel('Time (s)', 'FontSize', 10);
ylabel('Velocity (m/s)', 'FontSize', 10);
title('Velocity Estimation (Unmeasured Hidden States)', 'FontSize', 11, 'FontWeight', 'bold');
legend('FontSize', 9);

% Save figure artifact for documentation / reports
saveas(gcf, 'kalman_2d_results.png');
fprintf('[✓] Figure successfully saved as kalman_2d_results.png\n');
