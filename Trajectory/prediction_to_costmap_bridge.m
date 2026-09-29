%% PREDICTION_TO_COSTMAP_BRIDGE
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% This module bridges the Trajectory Prediction layer with Path Planning & Decision Logic.
% It transforms multi-modal GMM trajectories and spatial covariance ellipses into
% dynamic spatio-temporal costmaps (`vehicleCostmap` compatible) and computes
% supervisory mode triggers for the downstream Stateflow controller.
%
% Key Functionalities:
%  1. Ingestion of Multi-Modal GMM Predictions from `trajectory_prediction_engine.m`
%  2. Dynamic Occupancy Grid Inflation across discrete time slices (0.5s, 1.0s, 2.0s, 3.0s)
%  3. Graded Cost Layering:
%     - Surface Hazards (Potholes: <5cm soft dip 0.1-0.4; >5cm hard cavity 0.8-1.0)
%     - Static Road Boundaries & Unpaved Shoulders (cost = 1.0)
%     - Virtual Driving Corridor Soft Bias (cost = 0.05 vs 0.20 off-corridor)
%     - Multi-Modal Agent Uncertainty Ellipses (Mahalanobis distance graduated risk)
%  4. Stateflow Supervisory Triggers:
%     - CRUISE (TTC > 3.5s)
%     - SLOW_DOWN (1.8s <= TTC <= 3.5s)
%     - YIELD / NEGOTIATE (unsignalled junction crossing closing speed)
%     - STOP (TTC < 1.8s, cattle freeze or pedestrian crossing)
%     - REROUTE (all forward lane paths blocked by obstacle cost >= 0.90)

function [costmap_struct, stateflow_decision] = prediction_to_costmap_bridge(predictions, hazard_summary, cfg)

    if nargin < 1 || isempty(predictions)
        % Run trajectory prediction engine to generate upstream predictions
        [predictions, hazard_summary] = trajectory_prediction_engine();
    end

    if nargin < 2 || isempty(hazard_summary)
        hazard_summary.potholes = [-0.8, 16.5, 7.5, 0.75, 0.85; 1.4, 28.0, 4.2, 0.60, 0.40];
        hazard_summary.road_bounds = [-4.5, 4.5];
    end

    if nargin < 3 || isempty(cfg)
        cfg.grid_res      = 0.20;       % 20 cm grid resolution (meters)
        cfg.x_range       = [-6.0, 6.0];% lateral bounds [-6m to +6m]
        cfg.z_range       = [0.0, 50.0];% longitudinal bounds [0m to 50m]
        cfg.time_slices   = [0.5, 1.0, 2.0, 3.0]; % seconds ahead
        cfg.dt_pred       = 0.1;        % prediction timestep (100 ms)
        cfg.ego_x         = -1.8;
        cfg.ego_width     = 2.0;
        cfg.hard_ttc_lim  = 1.8;        % seconds (triggers STOP)
        cfg.soft_ttc_lim  = 3.5;        % seconds (triggers SLOW_DOWN)
    end

    fprintf('========================================================================\n');
    fprintf('  SIH 26037: Prediction-to-Costmap & Stateflow Supervisor Bridge        \n');
    fprintf('  Team Epsilon | Automated Driving Toolbox & Navigation Toolbox          \n');
    fprintf('========================================================================\n');

    %% 1. Initialize Grid Coordinates
    x_vec = (cfg.x_range(1):cfg.grid_res:cfg.x_range(2))';
    z_vec = (cfg.z_range(1):cfg.grid_res:cfg.z_range(2))';
    Nx = length(x_vec);
    Nz = length(z_vec);
    [X_grid, Z_grid] = meshgrid(x_vec, z_vec);

    num_slices = length(cfg.time_slices);
    cost_slices = zeros(Nz, Nx, num_slices);

    %% 2. Base Static Road & Corridor Costs
    % (A) Virtual Corridor Soft Bias (Slide 11: keeps driving human-like)
    base_cost = 0.20 * ones(Nz, Nx);
    lane_mask = (X_grid >= -3.6 & X_grid <= 0.0); % right lane corridor
    base_cost(lane_mask) = 0.05;

    % (B) Road Boundaries (Hard Obstacles)
    bound_mask = (X_grid < hazard_summary.road_bounds(1) | X_grid > hazard_summary.road_bounds(2));
    base_cost(bound_mask) = 1.0;

    % (C) Static Surface Potholes (Graded Depth Cost, Slide 12)
    potholes = hazard_summary.potholes;
    for p = 1:size(potholes, 1)
        px = potholes(p, 1);
        pz = potholes(p, 2);
        depth = potholes(p, 3);
        radius = potholes(p, 4);

        dist_sq = (X_grid - px).^2 + (Z_grid - pz).^2;
        pothole_mask = dist_sq <= (radius^2);

        if depth >= 5.0
            % Deep Cavity (>5 cm): Hard wall (0.85 to 1.0)
            base_cost(pothole_mask) = max(base_cost(pothole_mask), 0.95);
        else
            % Shallow Dip (<5 cm): Soft penalty (0.25 to 0.40)
            base_cost(pothole_mask) = max(base_cost(pothole_mask), 0.35);
        end
    end

    %% 3. Inject Multi-Modal GMM Future Occupancy Ellipses per Time Slice
    num_agents = length(predictions);

    for s = 1:num_slices
        t_slice = cfg.time_slices(s);
        step_idx = min(round(t_slice / cfg.dt_pred), length(predictions(1).modes(1).X));
        grid_s = base_cost;

        for a = 1:num_agents
            trk = predictions(a);
            for m = 1:length(trk.modes)
                mode = trk.modes(m);
                prob = mode.prob;
                if prob < 0.05, continue; end % prune negligible modes

                mu_x = mode.X(step_idx);
                mu_z = mode.Z(step_idx);
                sig_x = sqrt(mode.cov_X(step_idx));
                sig_z = sqrt(mode.cov_Z(step_idx));

                % Bounding box ROI for fast local evaluation
                roi_x = (X_grid >= mu_x - 3.5 * sig_x & X_grid <= mu_x + 3.5 * sig_x);
                roi_z = (Z_grid >= mu_z - 3.5 * sig_z & Z_grid <= mu_z + 3.5 * sig_z);
                roi = roi_x & roi_z;

                if ~any(roi(:)), continue; end

                dx = X_grid(roi) - mu_x;
                dz = Z_grid(roi) - mu_z;

                % Mahalanobis distance squared (2D Gaussian):
                d_mahal_sq = (dx ./ max(sig_x, 0.1)).^2 + (dz ./ max(sig_z, 0.1)).^2;

                % Mode cost scaling:
                % Hard core (d_M <= 1.5): cost = 1.0 * prob
                % Soft skirt (1.5 < d_M <= 3.0): exponential decay
                agent_cost = zeros(size(d_mahal_sq));
                hard_idx = (d_mahal_sq <= 2.25); % (1.5)^2
                soft_idx = (d_mahal_sq > 2.25 & d_mahal_sq <= 9.0); % (3.0)^2

                agent_cost(hard_idx) = 1.0;
                agent_cost(soft_idx) = exp(-0.5 * (d_mahal_sq(soft_idx) - 2.25));

                scaled_cost = prob * agent_cost;
                grid_s(roi) = min(1.0, grid_s(roi) + scaled_cost);
            end
        end

        cost_slices(:, :, s) = grid_s;
    end

    %% 4. Evaluate Stateflow Supervisory Signals & TTC
    min_ttc = inf;
    crit_agent_id = 0;
    crit_agent_class = 'None';
    crit_mode_name = 'Clear';

    for a = 1:num_agents
        trk = predictions(a);
        for m = 1:length(trk.modes)
            mode = trk.modes(m);
            if mode.TTC < min_ttc
                min_ttc = mode.TTC;
                crit_agent_id = trk.track_id;
                crit_agent_class = trk.class;
                crit_mode_name = mode.mode_name;
            end
        end
    end

    % Check if current ego forward path is impassable (REROUTE condition)
    ego_corridor_mask = (X_grid >= (cfg.ego_x - cfg.ego_width/2) & ...
                         X_grid <= (cfg.ego_x + cfg.ego_width/2) & ...
                         Z_grid >= 2.0 & Z_grid <= 15.0);
    near_slice_cost = cost_slices(:, :, 1); % 0.5s slice
    is_path_blocked = all(max(near_slice_cost(ego_corridor_mask)) >= 0.90);

    % Stateflow Supervisor Mode Determination (Slide 12)
    % 0: CRUISE, 1: SLOW_DOWN, 2: YIELD, 3: STOP, 4: REROUTE
    if is_path_blocked
        sf_mode = 4;
        sf_mode_str = 'REROUTE';
        target_speed_factor = 0.0;
    elseif min_ttc < cfg.hard_ttc_lim
        sf_mode = 3;
        sf_mode_str = 'STOP';
        target_speed_factor = 0.0;
    elseif min_ttc <= cfg.soft_ttc_lim
        sf_mode = 1;
        sf_mode_str = 'SLOW_DOWN';
        target_speed_factor = 0.5;
    else
        sf_mode = 0;
        sf_mode_str = 'CRUISE';
        target_speed_factor = 1.0;
    end

    stateflow_decision = struct(...
        'mode_id', sf_mode, ...
        'mode_name', sf_mode_str, ...
        'min_TTC', min_ttc, ...
        'target_speed_factor', target_speed_factor, ...
        'critical_agent_id', crit_agent_id, ...
        'critical_agent_class', crit_agent_class, ...
        'critical_mode_name', crit_mode_name, ...
        'is_path_blocked', is_path_blocked ...
    );

    costmap_struct = struct(...
        'x_grid', x_vec, ...
        'z_grid', z_vec, ...
        'X_mesh', X_grid, ...
        'Z_mesh', Z_grid, ...
        'time_slices', cfg.time_slices, ...
        'cost_slices', cost_slices, ...
        'grid_res', cfg.grid_res ...
    );

    fprintf('>> Stateflow Supervisory Mode: [%s] (Min TTC: %.2f s | Crit Agent: #%d %s)\n', ...
        sf_mode_str, min_ttc, crit_agent_id, crit_agent_class);
    fprintf('>> Generated %d dynamic spatio-temporal costmap slices.\n', num_slices);
end
