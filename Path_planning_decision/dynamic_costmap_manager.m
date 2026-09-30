%% DYNAMIC_COSTMAP_MANAGER
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% This module manages the multi-layer dynamic costmap for path planning.
% Fuses three distinct risk layers:
%  Layer 1: Static Road Boundaries & Soft Virtual Lane Corridor Bias
%  Layer 2: Graded Surface Hazards (Shallow Pothole Dip vs Deep Cavity Wall)
%  Layer 3: Dynamic Multi-Modal Agent Occupancy (GMM Uncertainty Ellipses from Trajectory)
%
% Compatible with MATLAB Navigation Toolbox `vehicleCostmap` and standalone grid evaluation.

classdef dynamic_costmap_manager < handle
    properties
        grid_res       = 0.20;        % Grid resolution (meters/cell)
        x_range        = [-5.5, 5.5]; % Lateral range (meters)
        z_range        = [0.0, 45.0]; % Longitudinal range (meters)
        road_bounds    = [-4.5, 4.5]; % Road boundary limits [left, right]
        corridor_x     = 0.0;         % Center of virtual lane corridor (centered on vehicle in body frame)
        corridor_width = 2.6;         % Width of virtual lane corridor
        
        x_vec
        z_vec
        X_mesh
        Z_mesh
        Nx
        Nz
        
        base_static_cost  % Layer 1 + Layer 2
        dynamic_slices    % Layer 3 across time slices
        time_slices       % Array of slice timestamps [0.5, 1.0, 2.0, 3.0]
    end
    
    methods
        function obj = dynamic_costmap_manager(road_bounds, potholes, time_slices)
            if nargin >= 1 && ~isempty(road_bounds)
                obj.road_bounds = road_bounds;
            end
            if nargin < 3 || isempty(time_slices)
                obj.time_slices = [0.5, 1.0, 2.0, 3.0];
            else
                obj.time_slices = time_slices;
            end
            
            % Generate metric grid
            obj.x_vec = (obj.x_range(1):obj.grid_res:obj.x_range(2))';
            obj.z_vec = (obj.z_range(1):obj.grid_res:obj.z_range(2))';
            obj.Nx = length(obj.x_vec);
            obj.Nz = length(obj.z_vec);
            [obj.X_mesh, obj.Z_mesh] = meshgrid(obj.x_vec, obj.z_vec);
            
            % Build base static layers
            obj.build_base_static_layer(potholes);
            
            % Initialize dynamic slices
            num_slices = length(obj.time_slices);
            obj.dynamic_slices = zeros(obj.Nz, obj.Nx, num_slices);
        end
        
        function build_base_static_layer(obj, potholes)
            % Layer 1: Road Boundaries & Soft Virtual Lane Corridor
            % Default off-corridor cost is 0.20; inside virtual lane corridor is 0.05
            obj.base_static_cost = 0.20 * ones(obj.Nz, obj.Nx);
            
            % Soft virtual corridor (Slide 11: human-like behavior, easily leaves when blocked)
            half_cw = obj.corridor_width / 2;
            corridor_mask = (obj.X_mesh >= (obj.corridor_x - half_cw)) & ...
                            (obj.X_mesh <= (obj.corridor_x + half_cw));
            obj.base_static_cost(corridor_mask) = 0.05;
            
            % Hard road boundaries & unpaved shoulders (lethal obstacle)
            bound_mask = (obj.X_mesh < obj.road_bounds(1)) | (obj.X_mesh > obj.road_bounds(2));
            obj.base_static_cost(bound_mask) = 1.00;
            
            % Layer 2: Graded Surface Hazards (Potholes, Slide 12)
            if ~isempty(potholes)
                for p = 1:size(potholes, 1)
                    px    = potholes(p, 1);
                    pz    = potholes(p, 2);
                    depth = potholes(p, 3);
                    rad   = potholes(p, 4);
                    
                    dist_sq = (obj.X_mesh - px).^2 + (obj.Z_mesh - pz).^2;
                    p_mask  = dist_sq <= (rad^2);
                    
                    if depth >= 5.0
                        % Deep cavity (>5 cm): Hard wall (0.95) -> Planner must steer around it
                        obj.base_static_cost(p_mask) = max(obj.base_static_cost(p_mask), 0.95);
                    else
                        % Shallow dip (<5 cm): Soft graded penalty (0.35) -> Crossable at reduced speed
                        obj.base_static_cost(p_mask) = max(obj.base_static_cost(p_mask), 0.35);
                    end
                end
            end
        end
        
        function update_dynamic_occupancy(obj, predictions)
            % Ingests multi-modal GMM predictions from `Trajectory/`
            num_slices = length(obj.time_slices);
            obj.dynamic_slices = repmat(obj.base_static_cost, [1, 1, num_slices]);
            
            if isempty(predictions), return; end
            
            for s = 1:num_slices
                t_slice = obj.time_slices(s);
                grid_s = obj.base_static_cost;
                
                for a = 1:length(predictions)
                    trk = predictions(a);
                    for m = 1:length(trk.modes)
                        mode = trk.modes(m);
                        if mode.prob < 0.05, continue; end
                        
                        % Find closest time index in prediction
                        step_idx = min(round(t_slice / 0.1), length(mode.X));
                        mu_x = mode.X(step_idx);
                        mu_z = mode.Z(step_idx);
                        sig_x = max(0.2, sqrt(mode.cov_X(step_idx)));
                        sig_z = max(0.3, sqrt(mode.cov_Z(step_idx)));
                        
                        % Bounding box evaluation region
                        roi_x = (obj.X_mesh >= mu_x - 3.0*sig_x) & (obj.X_mesh <= mu_x + 3.0*sig_x);
                        roi_z = (obj.Z_mesh >= mu_z - 3.0*sig_z) & (obj.Z_mesh <= mu_z + 3.0*sig_z);
                        roi = roi_x & roi_z;
                        
                        if ~any(roi(:)), continue; end
                        
                        dx = obj.X_mesh(roi) - mu_x;
                        dz = obj.Z_mesh(roi) - mu_z;
                        d_mahal_sq = (dx / sig_x).^2 + (dz / sig_z).^2;
                        
                        % Mahalanobis distance scaled cost
                        risk = zeros(size(d_mahal_sq));
                        hard_core = (d_mahal_sq <= 2.25); % 1.5 sigma
                        soft_skirt = (d_mahal_sq > 2.25) & (d_mahal_sq <= 9.0); % 3.0 sigma
                        
                        risk(hard_core) = 1.0;
                        risk(soft_skirt) = exp(-0.5 * (d_mahal_sq(soft_skirt) - 2.25));
                        
                        grid_s(roi) = min(1.0, grid_s(roi) + mode.prob * risk);
                    end
                end
                obj.dynamic_slices(:, :, s) = grid_s;
            end
        end
        
        function cost = get_cost_at(obj, x, z, t_lookahead)
            % Returns interpolated cost at metric coordinate (x, z) for future time t
            if nargin < 4 || isempty(t_lookahead), t_lookahead = 0.5; end
            
            % Interpolate between time slices
            [~, slice_idx] = min(abs(obj.time_slices - t_lookahead));
            grid_eval = obj.dynamic_slices(:, :, slice_idx);
            
            % Clamp coordinates
            x_c = max(obj.x_range(1), min(obj.x_range(2), x));
            z_c = max(obj.z_range(1), min(obj.z_range(2), z));
            
            % Nearest cell lookup for speed
            ix = round((x_c - obj.x_range(1)) / obj.grid_res) + 1;
            iz = round((z_c - obj.z_range(1)) / obj.grid_res) + 1;
            ix = max(1, min(obj.Nx, ix));
            iz = max(1, min(obj.Nz, iz));
            
            cost = grid_eval(iz, ix);
        end
        
        function blocked = is_ego_lane_blocked(obj, ego_x, ego_w, z_start, z_end)
            % Checks if forward corridor is completely blocked across its entire width
            if nargin < 4, z_start = 2.0; end
            if nargin < 5, z_end   = 18.0; end
            
            mask = (obj.X_mesh >= (ego_x - ego_w/2)) & ...
                   (obj.X_mesh <= (ego_x + ego_w/2)) & ...
                   (obj.Z_mesh >= z_start) & (obj.Z_mesh <= z_end);
            
            near_cost = obj.dynamic_slices(:, :, 1); % near slice
            blocked = all(max(near_cost(mask)) >= 0.90);
        end
    end
end
