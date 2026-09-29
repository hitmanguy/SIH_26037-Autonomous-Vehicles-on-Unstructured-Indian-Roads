%% SPEED_PROFILE_GENERATOR
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% Decoupled SL / ST Graph Longitudinal Speed Profiler:
% Computes dynamically feasible, curvature-aware, and comfort-constrained
% velocity and acceleration profiles along the continuous spatial path.
%
% Algorithmic Features:
%  1. Curvature-Governed Lateral Comfort:
%       v_kappa(s) = sqrt(a_y_max / (|kappa(s)| + eps))
%  2. Graded Surface Hazard Awareness:
%       Enforces v(s) <= 15 km/h over shallow pothole dips (< 5 cm)
%  3. Stateflow Behavioral Integration:
%       Adheres to CRUISE (40 km/h), SLOW_DOWN (20 km/h), YIELD (10 km/h), STOP (0 km/h)
%  4. Forward-Backward Pass Profile Integration:
%       Satisfies longitudinal acceleration (a_max) and deceleration limits (a_dec_comf, a_dec_emerg)
%  5. Returns time-parameterized trajectory: [x, z, s, theta, kappa, v, a, t]

classdef speed_profile_generator < handle
    properties
        v_cruise_max    = 40 / 3.6;   % Max cruise speed: 40 km/h = 11.11 m/s
        v_slowdown      = 20 / 3.6;   % Slowdown speed: 20 km/h = 5.56 m/s
        v_yield         = 10 / 3.6;   % Yielding / crawl speed: 10 km/h = 2.78 m/s
        v_pothole_dip   = 15 / 3.6;   % Shallow pothole dip speed: 15 km/h = 4.17 m/s
        
        a_max           = 2.0;        % Max comfort longitudinal acceleration (m/s^2)
        a_dec_comf      = -2.5;       % Comfort deceleration (m/s^2)
        a_dec_emerg     = -5.0;       % Maximum emergency deceleration (m/s^2)
        a_lat_max       = 2.2;        % Maximum lateral comfort acceleration (m/s^2)
        jerk_max        = 2.5;        % Maximum jerk limit (m/s^3)
    end
    
    methods
        function obj = speed_profile_generator(cfg)
            if nargin >= 1 && ~isempty(cfg)
                if isfield(cfg, 'v_cruise_max'),  obj.v_cruise_max  = cfg.v_cruise_max; end
                if isfield(cfg, 'a_max'),         obj.a_max         = cfg.a_max; end
                if isfield(cfg, 'a_dec_comf'),    obj.a_dec_comf    = cfg.a_dec_comf; end
                if isfield(cfg, 'a_dec_emerg'),   obj.a_dec_emerg   = cfg.a_dec_emerg; end
                if isfield(cfg, 'a_lat_max'),     obj.a_lat_max     = cfg.a_lat_max; end
            end
        end
        
        function [timed_traj, profile_stats] = generate(obj, opt_traj, v0, a0, supervisor_state, potholes, stop_s)
            % opt_traj        : Struct from continuous_trajectory_optimizer (.x, .z, .s, .theta, .kappa)
            % v0              : Current ego speed (m/s)
            % a0              : Current ego acceleration (m/s^2)
            % supervisor_state: String or enum ('CRUISE', 'SLOW_DOWN', 'YIELD', 'STOP', 'REROUTE')
            % potholes        : [M x 4] matrix [px, pz, depth_cm, radius_m]
            % stop_s          : Target stopping distance (scalar in meters, Inf if no stop)
            
            t_start = tic;
            
            s_vec     = opt_traj.s(:);
            kappa_vec = opt_traj.kappa(:);
            x_vec     = opt_traj.x(:);
            z_vec     = opt_traj.z(:);
            theta_vec = opt_traj.theta(:);
            N         = length(s_vec);
            
            if N < 2
                timed_traj = struct('x', x_vec, 'z', z_vec, 's', s_vec, ...
                                    'theta', theta_vec, 'kappa', kappa_vec, ...
                                    'v', v0, 'a', 0, 't', 0);
                profile_stats = struct('latency_ms', 0.1, 'emergency_stop', false);
                return;
            end
            
            % Step 1: Base Stateflow Behavioral Speed Limit
            switch upper(supervisor_state)
                case 'CRUISE'
                    v_base = obj.v_cruise_max;
                    a_dec_limit = obj.a_dec_comf;
                case 'SLOW_DOWN'
                    v_base = obj.v_slowdown;
                    a_dec_limit = obj.a_dec_comf;
                case 'YIELD'
                    v_base = obj.v_yield;
                    a_dec_limit = obj.a_dec_comf;
                case 'STOP'
                    v_base = 0.0;
                    a_dec_limit = obj.a_dec_emerg;
                case 'REROUTE'
                    v_base = obj.v_yield;
                    a_dec_limit = obj.a_dec_comf;
                otherwise
                    v_base = obj.v_cruise_max;
                    a_dec_limit = obj.a_dec_comf;
            end
            
            % Step 2: Pointwise Curvature Velocity Limit
            % v_lat(k) = sqrt(a_lat_max / (|kappa(k)| + eps))
            v_curv = sqrt(obj.a_lat_max ./ max(abs(kappa_vec), 1e-4));
            
            % Step 3: Graded Surface Hazard Speed Limit (Potholes)
            v_hazard = v_base * ones(N, 1);
            if ~isempty(potholes)
                for p = 1:size(potholes, 1)
                    px    = potholes(p, 1);
                    pz    = potholes(p, 2);
                    depth = potholes(p, 3);
                    rad   = potholes(p, 4);
                    
                    % Only process shallow dips (< 5 cm); deep cavities are avoided spatially
                    if depth < 5.0
                        dist_to_pothole = hypot(x_vec - px, z_vec - pz);
                        in_hazard = dist_to_pothole <= (rad + 0.5);
                        v_hazard(in_hazard) = min(v_hazard(in_hazard), obj.v_pothole_dip);
                    end
                end
            end
            
            % Step 4: Upper bound envelope v_max(s)
            v_upper = min([v_base * ones(N, 1), v_curv, v_hazard], [], 2);
            
            % Handle explicit stop location
            is_emergency = false;
            if nargin >= 7 && ~isinf(stop_s) && ~isnan(stop_s)
                stop_idx = find(s_vec >= stop_s, 1);
                if isempty(stop_idx)
                    stop_idx = N;
                end
                v_upper(stop_idx:end) = 0.0;
                if stop_s < 12.0
                    a_dec_limit = obj.a_dec_emerg;
                    is_emergency = true;
                end
            end
            
            if strcmpi(supervisor_state, 'STOP')
                v_upper(:) = 0.0;
                is_emergency = true;
            end
            
            % Step 5: Backward Pass (Deceleration Limit Enforcement)
            v_backward = v_upper;
            ds_vec = diff(s_vec);
            ds_vec = max(ds_vec, 1e-3);
            
            for k = (N - 1):-1:1
                ds = ds_vec(k);
                % v_k^2 <= v_{k+1}^2 + 2 * |a_dec| * ds
                max_allowable_vk = sqrt(v_backward(k + 1)^2 + 2 * abs(a_dec_limit) * ds);
                v_backward(k) = min(v_backward(k), max_allowable_vk);
            end
            
            % Step 6: Forward Pass (Acceleration Limit Enforcement)
            v_forward = zeros(N, 1);
            v_forward(1) = min(v0, v_backward(1));
            
            for k = 1:(N - 1)
                ds = ds_vec(k);
                % v_{k+1}^2 <= v_k^2 + 2 * a_max * ds
                max_allowable_vk1 = sqrt(v_forward(k)^2 + 2 * obj.a_max * ds);
                v_forward(k + 1) = min(v_backward(k + 1), max_allowable_vk1);
            end
            
            v_profile = max(0.0, v_forward);
            
            % Step 7: Time Parameterization & Acceleration Derivation
            t_vec = zeros(N, 1);
            a_vec = zeros(N, 1);
            
            for k = 1:(N - 1)
                ds = ds_vec(k);
                v_avg = 0.5 * (v_profile(k) + v_profile(k + 1));
                if v_avg > 0.1
                    dt = ds / v_avg;
                else
                    dt = ds / 0.1;
                end
                t_vec(k + 1) = t_vec(k) + dt;
                
                % Numerical acceleration: a = (v_{k+1}^2 - v_k^2) / (2 * ds)
                a_vec(k) = (v_profile(k + 1)^2 - v_profile(k)^2) / (2 * ds);
            end
            a_vec(N) = a_vec(max(1, N - 1));
            
            latency_ms = toc(t_start) * 1000;
            
            timed_traj = struct(...
                'x', x_vec, ...
                'z', z_vec, ...
                's', s_vec, ...
                'theta', theta_vec, ...
                'kappa', kappa_vec, ...
                'v', v_profile, ...
                'a', a_vec, ...
                't', t_vec, ...
                'total_time', t_vec(end), ...
                'total_distance', s_vec(end) ...
            );
            
            profile_stats = struct(...
                'latency_ms', latency_ms, ...
                'emergency_stop', is_emergency, ...
                'v_start', v_profile(1), ...
                'v_end', v_profile(end), ...
                'max_accel', max(a_vec), ...
                'min_decel', min(a_vec) ...
            );
        end
    end
end
