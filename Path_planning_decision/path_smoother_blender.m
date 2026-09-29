%% PATH_SMOOTHER_BLENDER
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% Replan Blender and Trajectory Stitcher:
% Solves the trajectory continuity problem during periodic replanning (5-10 Hz).
% Eliminates steering wheel jerks and curvature discontinuities between consecutive
% replanning cycles using an urgency-scaled C^2 quintic polynomial transition window.
%
% Mathematical Principle:
%  Transition parameter: u = clamp(s / L_blend, 0, 1)
%  Quintic C^2 weight:   w(u) = 10*u^3 - 15*u^4 + 6*u^5
%    Satisfies: w(0) = 0, w'(0) = 0, w''(0) = 0 (Smooth join with previously tracked path)
%               w(1) = 1, w'(1) = 0, w''(1) = 0 (Smooth merge into new optimal path)
%  Blended path: P_blend(s) = (1 - w(u)) * P_prev(s) + w(u) * P_new(s)
%
% Dynamic Blending Window:
%  - High Urgency (Obstacle swerve / Emergency): L_blend = 1.2 m (fast reaction)
%  - Low Urgency (Nominal cruising / corridor adjustment): L_blend = 5.0 m (maximum comfort)

classdef path_smoother_blender < handle
    properties
        l_blend_nominal   = 5.0;  % Nominal blending window length (meters)
        l_blend_urgent    = 1.2;  % Urgent swerve blending window length (meters)
        has_previous_plan = false;
        prev_trajectory   = [];
    end
    
    methods
        function obj = path_smoother_blender(l_nom, l_urg)
            if nargin >= 1 && ~isempty(l_nom), obj.l_blend_nominal = l_nom; end
            if nargin >= 2 && ~isempty(l_urg), obj.l_blend_urgent  = l_urg; end
        end
        
        function [blended_traj, blend_stats] = blend(obj, new_traj, ego_state, urgency_factor)
            % new_traj      : Struct from speed_profile_generator (.x, .z, .s, .theta, .kappa, .v, .a, .t)
            % ego_state     : [x_ego, z_ego, theta_ego, v_ego]
            % urgency_factor: Scalar in [0.0, 1.0] (0 = smooth cruising, 1 = emergency stop/swerve)
            
            t_start = tic;
            
            if nargin < 4 || isempty(urgency_factor)
                urgency_factor = 0.0;
            end
            urgency_factor = max(0.0, min(1.0, urgency_factor));
            
            % If no previous trajectory exists, accept new trajectory directly
            if ~obj.has_previous_plan || isempty(obj.prev_trajectory)
                blended_traj = new_traj;
                obj.prev_trajectory = new_traj;
                obj.has_previous_plan = true;
                blend_stats = struct('latency_ms', toc(t_start)*1000, 'blended', false, 'l_blend', 0.0);
                return;
            end
            
            % Calculate dynamic blending window length based on urgency
            % L_blend = L_nom * (1 - urgency) + L_urg * urgency
            L_blend = obj.l_blend_nominal * (1.0 - urgency_factor) + obj.l_blend_urgent * urgency_factor;
            
            % Interpolate previous trajectory in the local frame around ego state
            s_new = new_traj.s(:);
            x_new = new_traj.x(:);
            z_new = new_traj.z(:);
            v_new = new_traj.v(:);
            a_new = new_traj.a(:);
            N     = length(s_new);
            
            % Find closest point on previous trajectory to ego
            prev_x = obj.prev_trajectory.x(:);
            prev_z = obj.prev_trajectory.z(:);
            prev_s = obj.prev_trajectory.s(:);
            
            dx_prev = prev_x - ego_state(1);
            dz_prev = prev_z - ego_state(2);
            [~, min_prev_idx] = min(hypot(dx_prev, dz_prev));
            s_offset = prev_s(min_prev_idx);
            
            % Sample previous trajectory along s_new
            s_query = s_offset + s_new;
            s_query = min(s_query, prev_s(end));
            
            prev_x_interp = interp1(prev_s, prev_x, s_query, 'linear', 'extrap');
            
            % Evaluate C^2 quintic transition weights
            u = max(0.0, min(1.0, s_new / L_blend));
            w = 10 * (u.^3) - 15 * (u.^4) + 6 * (u.^5);
            
            % Blend lateral position x(s)
            x_blended = (1.0 - w) .* prev_x_interp + w .* x_new;
            z_blended = z_new; % Longitudinal grid remains anchored to s_new
            
            % Recompute continuous derivatives for theta and kappa
            ds_vec = diff(s_new);
            ds_vec = max(ds_vec, 1e-3);
            
            dx_blend = gradient(x_blended, s_new);
            dz_blend = gradient(z_blended, s_new);
            d2x_blend = gradient(dx_blend, s_new);
            d2z_blend = gradient(dz_blend, s_new);
            
            theta_blended = atan2(dx_blend, dz_blend);
            
            denom = (dx_blend.^2 + dz_blend.^2).^(1.5);
            denom = max(denom, 1e-6);
            kappa_blended = (dx_blend .* d2z_blend - dz_blend .* d2x_blend) ./ denom;
            kappa_blended = max(-0.25, min(0.25, kappa_blended));
            
            % Blend velocity
            prev_v = obj.prev_trajectory.v(:);
            prev_v_interp = interp1(prev_s, prev_v, s_query, 'linear', 'extrap');
            v_blended = (1.0 - w) .* prev_v_interp + w .* v_new;
            
            latency_ms = toc(t_start) * 1000;
            
            blended_traj = struct(...
                'x', x_blended, ...
                'z', z_blended, ...
                's', s_new, ...
                'theta', theta_blended, ...
                'kappa', kappa_blended, ...
                'v', v_blended, ...
                'a', a_new, ...
                't', new_traj.t, ...
                'total_time', new_traj.total_time, ...
                'total_distance', new_traj.total_distance ...
            );
            
            % Store for next replan cycle
            obj.prev_trajectory = blended_traj;
            
            blend_stats = struct(...
                'latency_ms', latency_ms, ...
                'blended', true, ...
                'l_blend', L_blend, ...
                'urgency_factor', urgency_factor ...
            );
        end
        
        function reset(obj)
            obj.has_previous_plan = false;
            obj.prev_trajectory = [];
        end
    end
end
