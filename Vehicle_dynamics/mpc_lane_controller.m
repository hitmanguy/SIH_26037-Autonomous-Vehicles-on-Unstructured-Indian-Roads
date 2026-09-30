classdef mpc_lane_controller < handle
% MPC_LANE_CONTROLLER Discrete-Time State-Space Model Predictive Controller
% =========================================================================
% Implements a 4-state dynamic bicycle model predictive controller with
% road curvature preview and hard steering & slew-rate constraints.
% Uses Hildreth's real-time quadratic programming solver (zero toolbox dependencies).
% =========================================================================

    properties
        Waypoints            % Nx2 reference path waypoints [X, Y]
        Ts = 0.05            % Sample time (s)
        Np = 10              % Prediction horizon steps
        Nc = 3               % Control horizon steps
        
        % Vehicle Parameters (Bicycle Model)
        m = 1575             % Vehicle mass (kg)
        Iz = 2875            % Yaw moment of inertia (kg*m^2)
        lf = 1.2             % Distance from CG to front axle (m)
        lr = 1.6             % Distance from CG to rear axle (m)
        Cf = 19000           % Front cornering stiffness (N/rad)
        Cr = 33000           % Rear cornering stiffness (N/rad)
        Wheelbase = 2.8      % Total wheelbase L (m)
        
        % Constraints
        MaxSteering = 0.52   % Max front wheel angle (rad) ~ 30 deg
        MaxSteerRate = 0.35  % Max steering slew rate (rad/s)
        
        % Cost Weights
        Q_ey = 10.0          % Weight on lateral position error
        Q_epsi = 15.0        % Weight on heading angle error
        R_delta = 1.0        % Weight on steering magnitude
        R_ddelta = 20.0      % Weight on steering rate (smoothness)
        
        % State Memory
        LastDelta = 0.0      % Previous steering command (rad)
        LastEy = 0.0
        LastEpsi = 0.0
    end

    methods
        function obj = mpc_lane_controller(waypoints, sampleTime)
            if nargin >= 1, obj.Waypoints = waypoints; end
            if nargin >= 2, obj.Ts = sampleTime; end
        end

        function [deltaCmd, targetPt, ey, epsi] = step(obj, currentPose, currentSpeed)
            % currentPose: [X (m), Y (m), Yaw (rad)]
            % currentSpeed: longitudinal velocity Vx (m/s)

            Vx = max(2.0, abs(currentSpeed));
            Ts = obj.Ts;
            x = currentPose(1);
            y = currentPose(2);
            psi = currentPose(3);

            % 1. Find Closest Point on Reference Path & Frenet Errors
            [closestPt, tangentYaw, curvature, targetPt] = obj.find_frenet_state(x, y, psi, Vx);

            % Lateral cross-track error: positive when vehicle is to the left of path
            dx = x - closestPt(1);
            dy = y - closestPt(2);
            ey = -sin(tangentYaw)*dx + cos(tangentYaw)*dy;

            % Heading error: angle between vehicle heading and road tangent
            epsi = wrapToPi(psi - tangentYaw);

            % Approximate error derivatives
            dey = (ey - obj.LastEy) / Ts;
            depsi = (epsi - obj.LastEpsi) / Ts;
            obj.LastEy = ey;
            obj.LastEpsi = epsi;

            % Current Error State: [ey, dey, epsi, depsi]
            x0 = [ey; dey; epsi; depsi];

            % 2. Formulate Continuous Linear Error Dynamics
            % x_dot = A_c * x + B_c * delta + B_dist * kappa
            m = obj.m; Iz = obj.Iz; lf = obj.lf; lr = obj.lr;
            Cf = obj.Cf; Cr = obj.Cr;

            Ac = [
                0, 1, 0, 0;
                0, -(2*Cf + 2*Cr)/(m*Vx), (2*Cf + 2*Cr)/m, (-2*Cf*lf + 2*Cr*lr)/(m*Vx);
                0, 0, 0, 1;
                0, (-2*Cf*lf + 2*Cr*lr)/(Iz*Vx), (2*Cf*lf - 2*Cr*lr)/Iz, -(2*Cf*lf^2 + 2*Cr*lr^2)/(Iz*Vx)
            ];

            Bc = [
                0;
                2*Cf / m;
                0;
                2*Cf*lf / Iz
            ];

            Bdist = [
                0;
                (-2*Cf*lf + 2*Cr*lr)/m - Vx^2;
                0;
                -(2*Cf*lf^2 + 2*Cr*lr^2)/Iz
            ];

            % 3. Discretize via Euler / First-Order Matrix Exponential
            Ad = eye(4) + Ac * Ts;
            Bd = Bc * Ts;
            Bdd = Bdist * Ts * curvature;

            % 4. Build Condensed Prediction Matrices (Np steps)
            Np = obj.Np;
            Nc = obj.Nc;
            
            % Extended prediction: X = F*x0 + Phi*DeltaU + G*Bdd
            F = zeros(4*Np, 4);
            Phi = zeros(4*Np, Nc);
            G_dist = zeros(4*Np, 1);
            
            A_pow = eye(4);
            for i = 1:Np
                A_pow = A_pow * Ad;
                F((i-1)*4 + (1:4), :) = A_pow;
                
                % Curvature drift accumulation
                for j = 0:(i-1)
                    G_dist((i-1)*4 + (1:4)) = G_dist((i-1)*4 + (1:4)) + (Ad^j) * Bdd;
                end
                
                for j = 1:min(i, Nc)
                    Phi((i-1)*4 + (1:4), j) = (Ad^(i-j)) * Bd;
                end
            end

            % 5. Build Cost Function: J = X' * Q_bar * X + DeltaU' * R_bar * DeltaU
            Q_step = diag([obj.Q_ey, 0.1, obj.Q_epsi, 0.1]);
            Q_bar = blkdiag(kron(eye(Np), Q_step));
            R_bar = obj.R_ddelta * eye(Nc) + obj.R_delta * ones(Nc);

            H = 2 * (Phi' * Q_bar * Phi + R_bar);
            % Ensure strictly positive-definite
            H = (H + H') / 2 + 1e-4 * eye(Nc);

            free_drift = F * x0 + G_dist;
            f_vec = 2 * (Phi' * Q_bar * free_drift);

            % 6. Apply Input Constraints via Hildreth QP Solver
            % Max steering: -MaxSteering <= u0 + sum(DeltaU) <= MaxSteering
            % Max rate: -MaxSteerRate*Ts <= DeltaU_k <= MaxSteerRate*Ts
            max_d_rate = obj.MaxSteerRate * Ts;
            lb_rate = -max_d_rate * ones(Nc, 1);
            ub_rate =  max_d_rate * ones(Nc, 1);

            A_ineq = [eye(Nc); -eye(Nc)];
            b_ineq = [ub_rate; -lb_rate];

            delta_u = obj.solve_hildreth_qp(H, f_vec, A_ineq, b_ineq);

            % Extract first control step
            du0 = delta_u(1);
            rawDelta = obj.LastDelta + du0;

            % Absolute saturation
            deltaCmd = max(-obj.MaxSteering, min(obj.MaxSteering, rawDelta));
            obj.LastDelta = deltaCmd;
        end

        function u_opt = solve_hildreth_qp(~, H, f, A, b)
            % Hildreth's active-set quadratic programming solver
            % Solves: min 0.5 * u' * H * u + f' * u  s.t.  A * u <= b
            n_ineq = length(b);
            H_inv = inv(H);
            
            % Check unconstrained solution first
            u_unc = -H_inv * f;
            if all(A * u_unc <= b + 1e-7)
                u_opt = u_unc;
                return;
            end

            P = A * H_inv * A';
            d = A * H_inv * f + b;

            lambda = zeros(n_ineq, 1);
            for iter = 1:60
                lambda_old = lambda;
                for i = 1:n_ineq
                    w = P(i, :) * lambda - P(i, i) * lambda(i);
                    lambda(i) = max(0, -(d(i) + w) / P(i, i));
                end
                if norm(lambda - lambda_old) < 1e-5
                    break;
                end
            end
            u_opt = -H_inv * (f + A' * lambda);
        end

        function [closestPt, tangentYaw, curvature, targetPt] = find_frenet_state(obj, x, y, psi, Vx)
            % Projects (x, y) to closest waypoint and computes local tangent & curvature
            wps = obj.Waypoints;
            dists = hypot(wps(:, 1) - x, wps(:, 2) - y);
            [~, idx] = min(dists);

            N = size(wps, 1);
            idx_next = mod(idx, N) + 1;
            idx_prev = mod(idx - 2 + N, N) + 1;

            p1 = wps(idx, :);
            p2 = wps(idx_next, :);
            tangentYaw = atan2(p2(2) - p1(2), p2(1) - p1(1));
            closestPt = p1;

            % 3-point local curvature calculation
            p0 = wps(idx_prev, :);
            d1 = hypot(p1(1) - p0(1), p1(2) - p0(2));
            d2 = hypot(p2(1) - p1(1), p2(2) - p1(2));
            if d1 > 0.01 && d2 > 0.01
                theta1 = atan2(p1(2) - p0(2), p1(1) - p0(1));
                theta2 = atan2(p2(2) - p1(2), p2(1) - p1(1));
                dTheta = wrapToPi(theta2 - theta1);
                curvature = dTheta / ((d1 + d2) / 2);
            else
                curvature = 0.0;
            end
            curvature = max(-0.15, min(0.15, curvature));

            % Lookahead preview target point (lookahead distance approx 4m)
            previewSteps = min(N, max(5, round(4.0 / max(0.5, hypot(p2(1)-p1(1), p2(2)-p1(2))))));
            targetIdx = mod(idx + previewSteps - 1, N) + 1;
            targetPt = wps(targetIdx, :);
        end
    end
end
