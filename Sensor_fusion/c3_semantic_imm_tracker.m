%% C3_SEMANTIC_IMM_TRACKER
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% This script benchmarks the C3 Detector (YOLOv8s trained on IDD) + SAHI Slicing
% integrated with Semantic-Aware Interacting Multiple Model (IMM) Sensor Fusion.
%
% Architecture Implemented (from Team Epsilon Perception Pipeline):
%  1. Camera Pipeline (C3 YOLOv8s @ 15.6 Hz / 64 ms + SAHI Far Band @ 5 Hz)
%  2. Multi-Rate Radar Pipeline (Front @ 20 Hz)
%  3. Class-Conditioned Motion Models based on C3's 12 IDD Classes:
%     - autorickshaw (Class 6): lateral-swerve IMM (Q_lat = 28 m/s^2, Q_long = 1.5 m/s^2)
%     - animal (Class 9): 3-Mode IMM (CV + Dart + true Zero-Velocity/Stop Mode)
%  4. SAHI Slicing Far-Band Seeding (40 - 150 m):
%     - Track confirmation range computed from a pixel-height detection model
%       and M-of-N confirmation logic (not assumed).
%
% Tracking design:
%  - Filters run in a world frame (ego-motion compensated with odometry), so the
%    animal Stop mode is a genuine zero-velocity model (velocity pinned to 0).
%  - Event-driven: filters predict only to each sensor timestamp, so the IMM
%    Markov transition matrix acts once per measurement interval, not every 10 ms.
%  - Camera and radar arriving at the same instant are fused sequentially.
%  - Monte Carlo over numRuns seeded runs; mean +/- std is reported.

clear; clc; closeall = @() delete(findall(0, 'Type', 'figure')); closeall();
fprintf('========================================================================\n');
fprintf('  SIH 26037: C3 Detector + SAHI + Semantic IMM Sensor Fusion Benchmark  \n');
fprintf('  Team Epsilon | MathWorks Automated Driving & Sensor Fusion Toolbox    \n');
fprintf('========================================================================\n\n');

%% 1. Simulation Parameters
cfg.simDuration    = 18.0;    % 18-second scenario
cfg.dt_base        = 0.01;    % 100 Hz simulation base clock (10 ms)
cfg.dt_radar_front = 0.050;   % 20 Hz Front Radar (50 ms)
cfg.dt_c3_yolo     = 0.064;   % ~15.6 Hz C3 YOLOv8s (64 ms latency on RTX 4050)
cfg.dt_sahi_slice  = 0.200;   % 5 Hz SAHI Far-Band Slicing (200 ms)
cfg.p_det_animal   = 0.55;    % empirical C3 detection probability for animals

% Camera model (matches Perception/sahi_engine.py and the fusion bridge)
cfg.f_y          = 1200;      % px
cfg.ff_scale     = 640 / 1920;% full-frame letterbox downscale
cfg.px50         = 12;        % px height with 50% detection prob (~1.5 stride-8 cells)
cfg.px_spread    = 2;         % logistic spread of the detection curve (px)
cfg.confirm_M    = 3;         % M-of-N track confirmation
cfg.confirm_N    = 5;

% Semantic motion-model parameters (acceleration noise variances in (m/s^2)^2)
cfg.cow_Q      = [0.4 15.0];                 % walk / dart modes
cfg.cow_stop_q = 1e-4;                       % stop mode position jitter (truly stationary)
cfg.cow_mu0    = [0.80 0.15 0.05];
cfg.cow_tpm    = [0.94 0.04 0.02; 0.10 0.85 0.05; 0.02 0.02 0.96];
cfg.rick_Q     = {diag([0.3 0.1 0.01]), diag([1.5 28.0 0.01])};  % [long lat vert]

numRuns = 50;

t_vec = (0:cfg.dt_base:cfg.simDuration)';
N     = length(t_vec);

%% 2. Scenario Trajectories (Ego, Animal Freeze, Swerving Rickshaw, Far Bicycle)
% All trajectories are in the world frame; ego-relative = world - ego.
% Ego Vehicle: cruises at 40 km/h (11.11 m/s) in right lane (y = -1.8m).
v_ego   = 40 * (1000 / 3600);
scn.ego = [v_ego * t_vec, -1.8 * ones(N, 1)];
scn.v_ego = v_ego;

% Actor 1: Stray Animal / Cow (Class 9)
% Walks diagonally into the right lane at 10 m/s longitudinal, 0.675 m/s lateral,
% then at t = 4.0s SUDDENLY FREEZES dead in the road.
t_frz = min(t_vec, 4.0);
scn.cow = [25.0 + 10.0 * t_frz, -4.5 + 0.675 * t_frz];

% Actor 2: Auto-Rickshaw (Class 6): 38 km/h with 3 sharp lane swerves.
v_rickshaw = 38 * (1000 / 3600);
rick_y = zeros(N, 1);
for i = 1:N
    t = t_vec(i);
    if t < 2.5
        rick_y(i) = -4.0; % shoulder
    elseif t < 5.5
        tau = (t - 2.5) / 3.0;
        rick_y(i) = -4.0 + 2.2 * (3*tau^2 - 2*tau^3); % to right lane
    elseif t < 9.5
        tau = (t - 5.5) / 4.0;
        rick_y(i) = -1.8 + 3.6 * (3*tau^2 - 2*tau^3); % to left lane
    elseif t < 13.0
        rick_y(i) = 1.8; % cruising in left lane
    elseif t < 16.5
        tau = (t - 13.0) / 3.5;
        rick_y(i) = 1.8 - 3.6 * (3*tau^2 - 2*tau^3); % back to right lane
    else
        rick_y(i) = -1.8;
    end
end
scn.rick = [22.0 + v_rickshaw * t_vec, rick_y];
scn.v_rick = v_rickshaw;

% Actor 3: Distant slow Bicycle (Class 8, 1.6 m tall incl. rider) at 10 km/h,
% starting 190 m ahead (beyond both pipelines' reach, so both confirmation
% ranges are measured, not given). Small, far targets are exactly where
% full-frame downscaling fails and SAHI's 1:1 tiles help.
v_bike = 10 * (1000 / 3600);
scn.bike_x   = 190.0 + v_bike * t_vec;
scn.bike_h   = 1.6;

%% 3. Monte Carlo Runs
fprintf('>> Running %d Monte Carlo runs of the 18s multi-rate scenario...\n', numRuns);
tic;
runs = cell(numRuns, 1);
for r = 1:numRuns
    rng(r);
    runs{r} = runScenario(cfg, scn, t_vec);
end
fprintf('   done in %.1f s (%.2f s/run)\n', toc, toc / numRuns);

t_log = t_vec;
stack = @(f) cell2mat(cellfun(@(s) s.(f), runs', 'UniformOutput', false));
E.cow_kf   = stack('err_cow_kf');   E.cow_imm  = stack('err_cow_imm');
E.cow_sem  = stack('err_cow_sem');  E.rick_kf  = stack('err_rick_kf');
E.rick_imm = stack('err_rick_imm'); E.rick_sem = stack('err_rick_sem');
modeProbs = cellfun(@(s) s.cow_mode_prob, runs, 'UniformOutput', false);
cow_mode_prob = mean(cat(3, modeProbs{:}), 3);

%% 4. Statistical Benchmark Results
idx_freeze = (t_log >= 3.8 & t_log <= 5.5);
idx_rick   = (t_log >= 0.5);   % skip initial convergence
mstat = @(E) [mean(mean(E(idx_freeze, :))), std(mean(E(idx_freeze, :))), ...
              mean(max(E(idx_freeze, :))),  std(max(E(idx_freeze, :)))];
rstat = @(E) [mean(mean(E(idx_rick, :))), std(mean(E(idx_rick, :))), ...
              mean(max(E(idx_rick, :))),  std(max(E(idx_rick, :)))];

S.cow_kf  = mstat(E.cow_kf);  S.cow_imm  = mstat(E.cow_imm);  S.cow_sem  = mstat(E.cow_sem);
S.rick_kf = rstat(E.rick_kf); S.rick_imm = rstat(E.rick_imm); S.rick_sem = rstat(E.rick_sem);

conf_ff   = cellfun(@(s) s.t_confirm_yolo_only, runs);
conf_sahi = cellfun(@(s) s.t_confirm_sahi, runs);
dist_ff   = cellfun(@(s) s.d_confirm_yolo_only, runs);
dist_sahi = cellfun(@(s) s.d_confirm_sahi, runs);
t_confirm_yolo_only = mean(conf_ff);
t_confirm_sahi      = mean(conf_sahi);

pr = @(name, s) fprintf('  %-32s: mean %.3f +- %.3f m | peak %.3f +- %.3f m\n', name, s);
fprintf('\n=========== BENCHMARK RESULTS SUMMARY (%d runs, mean +- std) ===========\n', numRuns);
fprintf('Actor 1: Stray Cow Sudden Freeze Event (t = 3.8s to 5.5s):\n');
pr('Standard Agnostic KF', S.cow_kf);
pr('Standard Agnostic IMM (2-mode)', S.cow_imm);
pr('Proposed Semantic IMM (3-mode)', S.cow_sem);
fprintf('  -> Semantic IMM: %.1f%% lower mean error, %.1f%% lower peak vs agnostic IMM\n', ...
    100 * (1 - S.cow_sem(1) / S.cow_imm(1)), 100 * (1 - S.cow_sem(3) / S.cow_imm(3)));

fprintf('\nActor 2: Auto-Rickshaw (Class 6) 3-Stage Aggressive Swerve (t >= 0.5s):\n');
pr('Standard Agnostic KF', S.rick_kf);
pr('Agnostic Isotropic IMM (2-mode)', S.rick_imm);
pr('Semantic Lateral-Swerve IMM', S.rick_sem);
fprintf('  -> Semantic IMM: %.1f%% lower mean error, %.1f%% lower peak vs KF\n', ...
    100 * (1 - S.rick_sem(1) / S.rick_kf(1)), 100 * (1 - S.rick_sem(3) / S.rick_kf(3)));

fprintf('\nPerception Track Confirmation (1.6 m bicycle, %d-of-%d logic, px50 = %d px):\n', ...
    cfg.confirm_M, cfg.confirm_N, cfg.px50);
fprintf('  YOLOv8 Full-Frame Confirmation   : t = %5.2f +- %.2f s (range %5.1f +- %.1f m)\n', ...
    mean(conf_ff), std(conf_ff), mean(dist_ff), std(dist_ff));
fprintf('  SAHI Slicing (Far Band) Seed     : t = %5.2f +- %.2f s (range %5.1f +- %.1f m)\n', ...
    mean(conf_sahi), std(conf_sahi), mean(dist_sahi), std(dist_sahi));
fprintf('  Early Warning Lead Time Added    : +%.2f s (Safety Horizon Extension: +%.1f m)\n', ...
    mean(conf_ff - conf_sahi), mean(dist_sahi - dist_ff));
fprintf('=======================================================================\n\n');

%% 5. Generate 4-Panel High-Resolution Publication Figure (Modern Dark ADAS Dashboard)
bg_dark   = [0.08 0.09 0.11];
axes_dark = [0.12 0.13 0.16];
grid_col  = [0.25 0.28 0.35];
txt_white = [0.95 0.96 0.98];
styleAx = @() set(gca, 'Color', axes_dark, 'XColor', txt_white, 'YColor', txt_white, ...
                       'GridColor', grid_col, 'GridAlpha', 0.5);

fig = figure('Name', 'SIH 26037: C3 Detector + SAHI + Semantic IMM Fusion', ...
             'Color', bg_dark, 'Position', [100 80 1350 850], 'Visible', 'off');

% Panel 1: Cow Freeze Tracking Error (Monte Carlo mean with +/-1 std band)
subplot(2, 2, 1); styleAx(); hold on; grid on; box on;
w = t_log <= 5.5;
h1 = bandPlot(t_log(w), E.cow_kf(w, :),  [1.0 0.35 0.35], '--', 2.0);
h2 = bandPlot(t_log(w), E.cow_imm(w, :), [0.3 0.7 1.0],   '-.', 1.8);
h3 = bandPlot(t_log(w), E.cow_sem(w, :), [0.1 0.95 0.45], '-',  2.6);
xline(4.0, ':', 'Cow Freezes Dead in Lane', 'LineWidth', 1.5, 'Color', [1.0 0.85 0.2], 'LabelVerticalAlignment', 'top');
xlabel('Simulation Time (s)', 'Color', txt_white, 'FontSize', 10);
ylabel('Position Error (m)', 'Color', txt_white, 'FontSize', 10);
title(sprintf('Stray Cow Tracking Error (Sudden Freeze at t=4.0s, %d runs)', numRuns), ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', txt_white);
legend([h1 h2 h3], 'Standard Agnostic KF', 'Agnostic 2-Mode IMM', 'Proposed C3 Semantic IMM (3-Mode)', ...
    'Location', 'northwest', 'TextColor', txt_white, 'Color', axes_dark, 'EdgeColor', grid_col);
xlim([0 5.5]); ylim([0 2.5]);

% Panel 2: Semantic IMM Probability Evolution (Mode Switching, mean over runs)
subplot(2, 2, 2); styleAx(); hold on; grid on; box on;
plot(t_log(w), cow_mode_prob(w, 1), '-',  'Color', [0.3 0.7 1.0], 'LineWidth', 2.0);
plot(t_log(w), cow_mode_prob(w, 2), '--', 'Color', [0.9 0.4 0.9], 'LineWidth', 1.8);
plot(t_log(w), cow_mode_prob(w, 3), '-',  'Color', [1.0 0.6 0.1], 'LineWidth', 2.6);
xline(4.0, ':', 'Cow Stops Dead', 'LineWidth', 1.5, 'Color', [1.0 0.85 0.2]);
xlabel('Simulation Time (s)', 'Color', txt_white, 'FontSize', 10);
ylabel('Mode Probability \mu_j', 'Color', txt_white, 'FontSize', 10);
title('C3 Semantic IMM Dynamic Mode Probabilities for Animal', 'FontSize', 11, 'FontWeight', 'bold', 'Color', txt_white);
legend('Mode 1: Constant Velocity (Walking)', 'Mode 2: Sudden Maneuver/Dart', 'Mode 3: ZERO-VELOCITY FREEZE', ...
    'Location', 'east', 'TextColor', txt_white, 'Color', axes_dark, 'EdgeColor', grid_col);
xlim([0 5.5]); ylim([0 1.05]);

% Panel 3: Auto-Rickshaw Swerve Tracking Error
subplot(2, 2, 3); styleAx(); hold on; grid on; box on;
h1 = bandPlot(t_log, E.rick_kf,  [1.0 0.35 0.35], '--', 1.8);
h2 = bandPlot(t_log, E.rick_imm, [0.3 0.7 1.0],   '-.', 1.6);
h3 = bandPlot(t_log, E.rick_sem, [0.15 0.85 1.0], '-',  2.4);
xline(2.5, ':', 'Swerve 1', 'FontSize', 9, 'Color', [0.8 0.8 0.8]);
xline(5.5, ':', 'Swerve 2', 'FontSize', 9, 'Color', [0.8 0.8 0.8]);
xline(13.0, ':', 'Swerve 3', 'FontSize', 9, 'Color', [0.8 0.8 0.8]);
xlabel('Simulation Time (s)', 'Color', txt_white, 'FontSize', 10);
ylabel('Tracking Error (m)', 'Color', txt_white, 'FontSize', 10);
title('Auto-Rickshaw (Class 6) Erratic Swerve Tracking Error', 'FontSize', 11, 'FontWeight', 'bold', 'Color', txt_white);
legend([h1 h2 h3], 'Standard Agnostic KF', 'Agnostic Isotropic IMM', 'C3 Semantic Lateral-Swerve IMM', ...
    'Location', 'northeast', 'TextColor', txt_white, 'Color', axes_dark, 'EdgeColor', grid_col);
xlim([0 18]);

% Panel 4: SAHI Far-Band Early Warning & Track Seeding Horizon
subplot(2, 2, 4); styleAx(); hold on; grid on; box on;
categories = {'Full-Frame Only', 'Full-Frame + SAHI'};
confirm_distances = [mean(dist_ff), mean(dist_sahi)];
b = bar(confirm_distances, 0.45, 'FaceColor', 'flat');
b.CData(1, :) = [0.90 0.35 0.25];
b.CData(2, :) = [0.10 0.85 0.45];
errorbar(1:2, confirm_distances, [std(dist_ff), std(dist_sahi)], 'LineStyle', 'none', ...
    'Color', txt_white, 'LineWidth', 1.5, 'CapSize', 12);
set(gca, 'XTick', 1:2, 'XTickLabel', categories, 'FontSize', 10);
ylabel('Track Confirmation Range (m)', 'Color', txt_white, 'FontSize', 10);
title('Far Hazard Seeding Horizon (1.6 m Bicycle, 3-of-5 Confirmation)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', txt_white);
ylim([0 1.3 * max(confirm_distances + [std(dist_ff), std(dist_sahi)])]);
text(1, confirm_distances(1) + std(dist_ff) + 8, sprintf('%.1f m\n(t = %.2fs)', confirm_distances(1), mean(conf_ff)), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', txt_white);
text(2, confirm_distances(2) + std(dist_sahi) + 12, sprintf('%.1f m (+%.0f m)\n(t = %.2fs -> +%.1fs)', ...
    confirm_distances(2), diff(confirm_distances), mean(conf_sahi), mean(conf_ff - conf_sahi)), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', [0.2 1.0 0.5]);

sgtitle({'Team Epsilon (SIH 26037): C3 IDD Perception + SAHI + Semantic IMM Sensor Fusion', ...
         sprintf('Multi-Rate Monte Carlo Benchmark (%d runs) under Unstructured Indian Road Scenarios', numRuns)}, ...
        'FontSize', 13, 'FontWeight', 'bold', 'Color', txt_white);

exportgraphics(fig, 'c3_semantic_imm_results.png', 'Resolution', 150, 'BackgroundColor', bg_dark);
fprintf('>> Saved benchmark figure: c3_semantic_imm_results.png\n');

% Run-1 traces keep the original variable names for downstream scripts
err_cow_kf  = E.cow_kf(:, 1);  err_cow_imm = E.cow_imm(:, 1); err_cow_sem = E.cow_sem(:, 1);
err_rick_kf = E.rick_kf(:, 1); err_rick_sem = E.rick_sem(:, 1);
save('c3_semantic_imm_benchmark.mat', 't_log', 'err_cow_kf', 'err_cow_imm', 'err_cow_sem', ...
     'cow_mode_prob', 'err_rick_kf', 'err_rick_sem', 't_confirm_yolo_only', 't_confirm_sahi', ...
     'S', 'conf_ff', 'conf_sahi', 'dist_ff', 'dist_sahi', 'numRuns', 'cfg');
fprintf('>> Saved numerical logs: c3_semantic_imm_benchmark.mat\n');
fprintf('>> All operations completed successfully!\n');


%% ======================= Local Functions =======================

function out = runScenario(cfg, scn, t_vec)
% One seeded run of the multi-rate scenario. Filters live in the world frame.
N = numel(t_vec);
dt = cfg.dt_base;
v_ego = scn.v_ego;

H_meas = [1 0 0 0 0 0;
          0 0 1 0 0 0;
          0 0 0 0 1 0;
          0 1 0 0 0 0;
          0 0 0 1 0 0;
          0 0 0 0 0 1];
R_rad = diag([0.4, 0.8, 0.2, 0.15, 0.25, 0.1].^2);
R_cam = diag([0.25, 0.25, 0.1, 0.4, 0.4, 0.1].^2);

cvKF = @(x0, Q) trackingKF('MotionModel', '3D Constant Velocity', 'State', x0, ...
    'MeasurementModel', H_meas, 'ProcessNoise', Q, 'MeasurementNoise', R_rad);
% Zero-velocity model: position held, velocity pinned to 0. An EKF is used
% because trackingIMM predicts by dt, which a 'Custom' trackingKF can't accept.
pinVel = [1; 0; 1; 0; 1; 0];
stopKF = @(x0, q) trackingEKF(@(x, dt) x .* pinVel, @(x, varargin) H_meas * x, x0, ...
    'StateTransitionJacobianFcn', @(x, dt) diag(pinVel), ...
    'MeasurementJacobianFcn', @(x, varargin) H_meas, 'HasAdditiveProcessNoise', true, ...
    'ProcessNoise', diag([q 1e-4 q 1e-4 1e-4 1e-4]), 'MeasurementNoise', R_rad);
sameState = @(~, x1, ~, ~) x1;  % all modes share the [x vx y vy z vz] layout

% --- Cow Filters ---
xc = [scn.cow(1, 1); 10.0; scn.cow(1, 2); 0.675; 0; 0];
F.cow_kf  = cvKF(xc, 0.5 * eye(3));
F.cow_imm = trackingIMM('TrackingFilters', {cvKF(xc, 0.5 * eye(3)), cvKF(xc, 20.0 * eye(3))}, ...
    'ModelProbabilities', [0.80 0.20], 'TransitionProbabilities', [0.95 0.05; 0.10 0.90]);
% Semantic IMM for Class 9 (animal): walk / dart / freeze
F.cow_sem = trackingIMM('TrackingFilters', {cvKF(xc, cfg.cow_Q(1) * eye(3)), cvKF(xc, cfg.cow_Q(2) * eye(3)), ...
    stopKF([xc(1); 0; xc(3); 0; 0; 0], cfg.cow_stop_q)}, ...
    'ModelConversionFcn', sameState, ...
    'ModelProbabilities', cfg.cow_mu0, 'TransitionProbabilities', cfg.cow_tpm);

% --- Auto-Rickshaw Filters ---
xr = [scn.rick(1, 1); scn.v_rick; scn.rick(1, 2); 0; 0; 0];
F.rick_kf  = cvKF(xr, 0.5 * eye(3));
F.rick_imm = trackingIMM('TrackingFilters', {cvKF(xr, 0.4 * eye(3)), cvKF(xr, 28.0 * eye(3))}, ...
    'ModelProbabilities', [0.85 0.15], 'TransitionProbabilities', [0.93 0.07; 0.10 0.90]);
% Semantic IMM for Class 6: rickshaws hold speed but cut across lanes, so the
% maneuver mode's acceleration noise is lateral-dominant.
F.rick_sem = trackingIMM('TrackingFilters', {cvKF(xr, cfg.rick_Q{1}), cvKF(xr, cfg.rick_Q{2})}, ...
    'ModelProbabilities', [0.85 0.15], 'TransitionProbabilities', [0.93 0.07; 0.10 0.90]);

names  = fieldnames(F);
t_last = zeros(numel(names), 1);

out.err_cow_kf  = zeros(N, 1); out.err_cow_imm  = zeros(N, 1); out.err_cow_sem  = zeros(N, 1);
out.err_rick_kf = zeros(N, 1); out.err_rick_imm = zeros(N, 1); out.err_rick_sem = zeros(N, 1);
out.cow_mode_prob = zeros(N, 3);

% Detection streams for the far bicycle: [full-frame hits], [SAHI-tile hits]
hist_ff = false(1, cfg.confirm_N); hist_sahi = false(1, cfg.confirm_N);
out.t_confirm_yolo_only = NaN; out.t_confirm_sahi = NaN;
out.d_confirm_yolo_only = NaN; out.d_confirm_sahi = NaN;
pdet = @(px) 1 ./ (1 + exp(-(px - cfg.px50) / cfg.px_spread));

last_radar = -1; last_yolo = -1; last_sahi = -1;
for step = 1:N
    t = t_vec(step);
    radar_tick = (t - last_radar >= cfg.dt_radar_front - 1e-5);
    yolo_tick  = (t - last_yolo  >= cfg.dt_c3_yolo - 1e-5);
    sahi_tick  = (t - last_sahi  >= cfg.dt_sahi_slice - 1e-5);
    if radar_tick, last_radar = t; end
    if yolo_tick,  last_yolo  = t; end
    if sahi_tick,  last_sahi  = t; end

    % Ground-truth velocity by finite difference (forward at the first step,
    % otherwise the very first measurement would report zero velocity)
    nxt = max(step, 2);
    ego = scn.ego(step, :);

    % --- 1. Cow measurements (visible while ahead of ego bumper) ---
    cow = scn.cow(step, :);
    rel = cow - ego;
    v_cow = (scn.cow(nxt, :) - scn.cow(nxt - 1, :)) / dt;
    Z = {};
    if rel(1) > 2.0
        if yolo_tick && norm(rel) < 80 && rand() < cfg.p_det_animal
            Z{end+1} = {[cow(1) + 0.2*randn(); cow(2) + 0.15*randn(); 0.3; ...
                         v_cow(1) + 0.3*randn(); v_cow(2) + 0.3*randn(); 0], R_cam};
        end
        if radar_tick && norm(rel) < 100
            Z{end+1} = {[cow(1) + 0.25*randn(); cow(2) + 0.55*randn(); 0.3; ...
                         v_cow(1) + 0.12*randn(); v_cow(2) + 0.25*randn(); 0], R_rad};
        end
    end
    for k = find(startsWith(names, 'cow'))'
        if ~isempty(Z)
            if t > t_last(k), predict(F.(names{k}), t - t_last(k)); t_last(k) = t; end
            for m = 1:numel(Z), correct(F.(names{k}), Z{m}{1}, Z{m}{2}); end
        end
    end

    % --- 2. Auto-Rickshaw measurements (radar) ---
    rick = scn.rick(step, :);
    v_r = (scn.rick(nxt, :) - scn.rick(nxt - 1, :)) / dt;
    if radar_tick && norm(rick - ego) < 95
        z_r = [rick(1) + 0.25*randn(); rick(2) + 0.45*randn(); 0.5; ...
               v_r(1) + 0.15*randn(); v_r(2) + 0.25*randn(); 0];
        for k = find(startsWith(names, 'rick'))'
            if t > t_last(k), predict(F.(names{k}), t - t_last(k)); t_last(k) = t; end
            correct(F.(names{k}), z_r, R_rad);
        end
    end

    % --- 3. Far bicycle: detection from pixel height, M-of-N confirmation ---
    range = scn.bike_x(step) - ego(1);
    px_native = scn.bike_h * cfg.f_y / range;
    if yolo_tick
        hist_ff = [hist_ff(2:end), rand() < pdet(px_native * cfg.ff_scale)];
        if isnan(out.t_confirm_yolo_only) && sum(hist_ff) >= cfg.confirm_M
            out.t_confirm_yolo_only = t; out.d_confirm_yolo_only = range;
        end
    end
    if sahi_tick
        hist_sahi = [hist_sahi(2:end), rand() < pdet(px_native)];
    end
    if isnan(out.t_confirm_sahi) && (sum(hist_sahi) >= cfg.confirm_M || ~isnan(out.t_confirm_yolo_only))
        out.t_confirm_sahi = t; out.d_confirm_sahi = range;
    end

    % --- Logging: latest estimate extrapolated to t (what a consumer reads) ---
    for k = 1:numel(names)
        f = F.(names{k});
        p = f.State([1 3])' + f.State([2 4])' * (t - t_last(k));
        if startsWith(names{k}, 'cow'), gt = cow; else, gt = rick; end
        out.(['err_' names{k}])(step) = norm(p - gt);
    end
    out.cow_mode_prob(step, :) = F.cow_sem.ModelProbabilities;
end
end

function h = bandPlot(t, E, col, ls, lw)
% Monte Carlo mean curve with a +/-1 std shaded band.
m = mean(E, 2); s = std(E, 0, 2);
fill([t; flipud(t)], [m - s; flipud(m + s)], col, 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');
h = plot(t, m, ls, 'Color', col, 'LineWidth', lw);
end
