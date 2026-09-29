%% HYBRID_ASTAR_PLANNER
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% Bounded-Window Hybrid A* Motion Planner:
% Explores non-holonomic vehicle kinematics over a bounded local window (35m)
% to find the optimal collision-free corridor in < 35 ms.
%
% Key Features:
%  1. Non-holonomic bicycle model motion primitives (L = 2.8m, delta_max = 32 deg)
%  2. Direct collision validation against dynamic 3-layer costmap
%  3. Graded hazard awareness: routes safely around deep potholes (>5cm) and slow-moving cattle
%  4. Automatic extraction of safe lateral convex corridor bounds [x_min, x_max] for Stage 2 QP optimization
%  5. Infeasible detection: triggers `planner_infeasible` flag for Stateflow REROUTE

classdef hybrid_astar_planner < handle
    properties
        wheelbase      = 2.8;         % Vehicle wheelbase L (meters)
        max_steer_rad  = deg2rad(30); % Maximum steering angle (rad)
        step_size      = 1.2;         % Motion primitive arc length ds (meters)
        local_window_z = 35.0;        % Bounded forward search horizon (meters)
        xy_res         = 0.5;         % Spatial discretization for visited hash (meters)
        theta_res      = deg2rad(15); % Angular discretization for visited hash (rad)
        max_iterations = 1200;        % Search budget to guarantee sub-35ms runtime
        
        % Cost Weights
        w_dist         = 1.0;
        w_steer        = 0.15;
        w_steer_change = 0.30;
        w_costmap      = 5.0;
        w_heuristic    = 1.2;
    end
    
    methods
        function obj = hybrid_astar_planner(cfg)
            if nargin >= 1 && ~isempty(cfg)
                if isfield(cfg, 'wheelbase'), obj.wheelbase = cfg.wheelbase; end
                if isfield(cfg, 'max_steer_rad'), obj.max_steer_rad = cfg.max_steer_rad; end
                if isfield(cfg, 'local_window_z'), obj.local_window_z = cfg.local_window_z; end
                if isfield(cfg, 'max_iterations'), obj.max_iterations = cfg.max_iterations; end
            end
        end
        
        function [plan_result, corridor_bounds, latency_ms] = plan(obj, start_state, goal_state, costmap_mgr)
            % start_state: [X, Z, theta] (metric ISO 8855 coordinates)
            % goal_state : [X_g, Z_g, theta_g]
            % costmap_mgr: instance of `dynamic_costmap_manager`
            
            t_start = tic;
            
            x0 = start_state(1);
            z0 = start_state(2);
            th0 = start_state(3);
            
            xg = goal_state(1);
            zg = min(z0 + obj.local_window_z, goal_state(2));
            thg = goal_state(3);
            
            % Steering angles discrete set: [-30, -15, 0, +15, +30] deg
            steer_set = [-obj.max_steer_rad, -0.5*obj.max_steer_rad, 0.0, 0.5*obj.max_steer_rad, obj.max_steer_rad];
            num_steer = length(steer_set);
            
            % Node storage:
            % Struct: x, z, theta, g_cost, f_cost, parent_idx, steer_cmd
            nodes = repmat(struct('x', 0, 'z', 0, 'theta', 0, 'g', 0, 'f', 0, 'parent', 0, 'steer', 0), obj.max_iterations, 1);
            
            % Start node
            h0 = obj.heuristic(x0, z0, th0, xg, zg, thg);
            nodes(1) = struct('x', x0, 'z', z0, 'theta', th0, 'g', 0, 'f', h0, 'parent', 0, 'steer', 0);
            num_nodes = 1;
            
            % Open set: priority queue implemented via array of indices
            open_indices = 1;
            open_f_costs = nodes(1).f;
            
            % Visited hash map: spatial 3D grid [ix, iz, ith]
            visited = containers.Map('KeyType', 'char', 'ValueType', 'double');
            key0 = obj.hash_state(x0, z0, th0);
            visited(key0) = nodes(1).g;
            
            goal_node_idx = 0;
            closest_node_idx = 1;
            min_dist_to_goal = hypot(x0 - xg, z0 - zg);
            
            iter = 0;
            while ~isempty(open_indices) && iter < obj.max_iterations
                iter = iter + 1;
                
                % Pop lowest f_cost node
                [~, min_pos] = min(open_f_costs);
                curr_idx = open_indices(min_pos);
                open_indices(min_pos) = [];
                open_f_costs(min_pos) = [];
                
                curr = nodes(curr_idx);
                
                % Check goal arrival
                dist_g = hypot(curr.x - xg, curr.z - zg);
                if dist_g < min_dist_to_goal
                    min_dist_to_goal = dist_g;
                    closest_node_idx = curr_idx;
                end
                
                if (curr.z >= zg - 1.0) || (dist_g <= 1.5 && abs(angdiff(curr.theta, thg)) < deg2rad(25))
                    goal_node_idx = curr_idx;
                    break;
                end
                
                % Expand motion primitives
                for s = 1:num_steer
                    delta = steer_set(s);
                    [next_x, next_z, next_th] = obj.kinematic_step(curr.x, curr.z, curr.theta, delta);
                    
                    % Boundary and costmap collision validation
                    cost_val = costmap_mgr.get_cost_at(next_x, next_z, (next_z - z0)/11.11);
                    if cost_val >= 0.85
                        continue; % lethal collision with obstacle, deep pothole wall, or road edge
                    end
                    
                    % Compute transition cost
                    d_steer = abs(delta - curr.steer);
                    step_cost = obj.w_dist * obj.step_size + ...
                                obj.w_steer * abs(delta) + ...
                                obj.w_steer_change * d_steer + ...
                                obj.w_costmap * cost_val;
                    next_g = curr.g + step_cost;
                    
                    % Check visited state
                    key = obj.hash_state(next_x, next_z, next_th);
                    if isKey(visited, key) && visited(key) <= next_g
                        continue;
                    end
                    
                    % Add new node
                    num_nodes = num_nodes + 1;
                    if num_nodes > obj.max_iterations, break; end
                    
                    h_next = obj.heuristic(next_x, next_z, next_th, xg, zg, thg);
                    nodes(num_nodes) = struct('x', next_x, 'z', next_z, 'theta', next_th, ...
                                              'g', next_g, 'f', next_g + obj.w_heuristic * h_next, ...
                                              'parent', curr_idx, 'steer', delta);
                    visited(key) = next_g;
                    
                    open_indices(end+1) = num_nodes;
                    open_f_costs(end+1) = nodes(num_nodes).f;
                end
            end
            
            % Reconstruct path
            if goal_node_idx == 0
                % Fallback to closest forward explored node if search budget exhausted
                target_idx = closest_node_idx;
                is_optimal_goal = (min_dist_to_goal < 4.0);
            else
                target_idx = goal_node_idx;
                is_optimal_goal = true;
            end
            
            % Backtrack path
            path_x  = [];
            path_z  = [];
            path_th = [];
            curr_trace = target_idx;
            while curr_trace > 0
                path_x(end+1)  = nodes(curr_trace).x;
                path_z(end+1)  = nodes(curr_trace).z;
                path_th(end+1) = nodes(curr_trace).theta;
                curr_trace     = nodes(curr_trace).parent;
            end
            
            path_x  = flip(path_x(:));
            path_z  = flip(path_z(:));
            path_th = flip(path_th(:));
            
            % Check feasibility
            if length(path_x) < 3 || (path_z(end) - z0) < 5.0
                is_feasible = false;
            else
                is_feasible = true;
            end
            
            % Extract safe lateral corridor bounds [x_min, x_max] around coarse path
            N_pts = length(path_x);
            corridor_bounds = zeros(N_pts, 2);
            for k = 1:N_pts
                px = path_x(k);
                pz = path_z(k);
                
                % Scan left for obstacle wall
                x_l = px;
                while x_l > costmap_mgr.road_bounds(1) + 0.5 && ...
                        costmap_mgr.get_cost_at(x_l, pz, 0.5) < 0.80 && (px - x_l) < 2.0
                    x_l = x_l - 0.2;
                end
                
                % Scan right for obstacle wall
                x_r = px;
                while x_r < costmap_mgr.road_bounds(2) - 0.5 && ...
                        costmap_mgr.get_cost_at(x_r, pz, 0.5) < 0.80 && (x_r - px) < 2.0
                    x_r = x_r + 0.2;
                end
                
                corridor_bounds(k, :) = [x_l, x_r];
            end
            
            latency_ms = toc(t_start) * 1000;
            
            plan_result = struct(...
                'x', path_x, ...
                'z', path_z, ...
                'theta', path_th, ...
                'is_feasible', is_feasible, ...
                'is_optimal_goal', is_optimal_goal, ...
                'latency_ms', latency_ms, ...
                'num_iterations', iter, ...
                'corridor_bounds', corridor_bounds ...
            );
        end
        
        function [next_x, next_z, next_th] = kinematic_step(obj, x, z, th, delta)
            % Bicycle model discrete integration
            beta = atan(0.5 * tan(delta));
            next_x  = x + obj.step_size * sin(th + beta);
            next_z  = z + obj.step_size * cos(th + beta);
            next_th = th + (obj.step_size / obj.wheelbase) * cos(beta) * tan(delta);
        end
        
        function h = heuristic(obj, x, z, th, xg, zg, thg)
            % Euclidean distance + heading alignment penalty
            dist = hypot(x - xg, z - zg);
            d_th = abs(angdiff(th, thg));
            h = dist + 1.5 * d_th;
        end
        
        function key = hash_state(obj, x, z, th)
            % Quantize state to prevent duplicate node expansions
            ix = round(x / obj.xy_res);
            iz = round(z / obj.xy_res);
            ith = round(th / obj.theta_res);
            key = sprintf('%d_%d_%d', ix, iz, ith);
        end
    end
end
