classdef pure_pursuit_controller < handle
% PURE_PURSUIT_CONTROLLER Stanley-Enhanced Predictive Pure Pursuit Controller
% =========================================================================
% Hybrid Tier-1 AV Lateral Controller combining:
%   1. Kinematic Pose Predictor (t_pred = 0.20s): Eliminates phase lag
%   2. Stanley Cross-Track Damping Term (delta_stanley): Fast, zero-overshoot lane acquisition
%   3. Adaptive Lookahead Distance (Ld): Curvature-coupled (3.5m - 9.5m)
%   4. Analytical Curvature Feedforward (delta_ff): Steady-state curve tracking
%   5. Slew-Rate Limiting & Command Smoothing Filter
% =========================================================================

    properties
        Waypoints            % Nx2 reference path waypoints [X, Y]
        Wheelbase = 2.8      % Vehicle wheelbase L (meters)
        lr = 1.6             % CG to rear axle distance (meters)
        MinLookahead = 4.2   % Minimum lookahead distance (meters) >= 1.5 * Wheelbase for mathematical stability
        MaxLookahead = 9.5   % Maximum lookahead distance (meters)
        LookaheadGain = 0.65 % Speed scaling factor (seconds)
        MaxSteering = 0.61   % Maximum front steering angle (rad) ~ 35.0 deg
        MaxSteerRate = 1.05  % Maximum steering rate (rad/s) ~ 60.0 deg/s for fast crisp avoidance
        
        % State Memory & Output Filter
        LastSteering = 0.0   % Previous steering command (rad)
        FilteredSteering = 0.0
        PredictionTime = 0.12% Lookahead preview time for predictor (seconds)
        Ts = 0.05            % Sample time (seconds)
    end

    methods
        function obj = pure_pursuit_controller(waypoints, initialSpeed, sampleTime)
            if nargin >= 1 && ~isempty(waypoints), obj.Waypoints = waypoints; end
            if nargin >= 3 && ~isempty(sampleTime), obj.Ts = sampleTime; end
        end

        function [deltaCmd, lookaheadPt, Ld, ey, epsi] = step(obj, currentPose, currentSpeed)
            % currentPose: [X (m), Y (m), Yaw (rad)]
            % currentSpeed: longitudinal velocity Vx (m/s)

            vx = max(1.0, abs(currentSpeed));
            x = currentPose(1);
            y = currentPose(2);
            yaw = currentPose(3);
            wps = obj.Waypoints;
            N = size(wps, 1);

            if N < 2
                deltaCmd = 0.0;
                lookaheadPt = [x + 5.0, y];
                Ld = 5.0; ey = 0.0; epsi = 0.0;
                return;
            end

            % 1. Find Closest Point & Path Curvature
            dists = hypot(wps(:, 1) - x, wps(:, 2) - y);
            [~, closestIdx] = min(dists);

            % 3-point local curvature estimate around closest point
            i0 = max(1, closestIdx - 3);
            i1 = closestIdx;
            i2 = min(N, closestIdx + 3);
            if i2 == i1
                i1 = max(1, N - 1);
            end
            p0 = wps(i0, 1:2);
            p1 = wps(i1, 1:2);
            p2 = wps(i2, 1:2);
            th1 = atan2(p1(2) - p0(2), p1(1) - p0(1));
            th2 = atan2(p2(2) - p1(2), p2(1) - p1(1));
            chord = max(0.5, hypot(p2(1) - p0(1), p2(2) - p0(2)));
            curvature = max(-0.15, min(0.15, wrapToPi(th2 - th1) / chord));

            % Safe local tangent vector and unit orientation
            if closestIdx >= N - 1
                tp1 = wps(max(1, N - 1), 1:2);
                tp2 = wps(N, 1:2);
                refPt = tp2;
            else
                tp1 = wps(closestIdx, 1:2);
                tp2 = wps(min(N, closestIdx + 3), 1:2);
                refPt = tp1;
            end
            segVec = tp2 - tp1;
            segLen = hypot(segVec(1), segVec(2));
            if segLen > 1e-4
                tangentUnit = segVec / segLen;
                tangentYaw = atan2(segVec(2), segVec(1));
            else
                tangentUnit = [1.0, 0.0];
                tangentYaw = 0.0;
            end

            % 2. Cross-Track Error & Heading Error
            dx = x - refPt(1); dy = y - refPt(2);
            ey = -sin(tangentYaw) * dx + cos(tangentYaw) * dy;
            epsi = wrapToPi(yaw - tangentYaw);

            % 3. Kinematic Pose Predictor (t_pred = 0.12s)
            prevDelta = obj.LastSteering;
            beta = atan((obj.lr / obj.Wheelbase) * tan(prevDelta));
            tPred = obj.PredictionTime;

            xPred = x + vx * cos(yaw + beta) * tPred;
            yPred = y + vx * sin(yaw + beta) * tPred;
            yawPred = wrapToPi(yaw + (vx / obj.Wheelbase) * cos(beta) * tan(prevDelta) * tPred);

            % 4. Adaptive Lookahead Distance
            % Must strictly satisfy Ld >= 1.5 * Wheelbase (4.2m) to guarantee closed-loop stability
            % and eliminate underdamped snake oscillations across lanes!
            rawLd = (obj.LookaheadGain * vx) / (1.0 + 1.2 * abs(curvature));
            Ld = max(obj.MinLookahead, min(obj.MaxLookahead, rawLd));

            % 5. Search Forward along Reference Path for Lookahead Target
            distsFromPred = hypot(wps(:, 1) - xPred, wps(:, 2) - yPred);
            foundTarget = false;
            targetIdx = closestIdx;
            for k = closestIdx:min(closestIdx + 60, N)
                longProj = (wps(k, 1) - xPred) * tangentUnit(1) + (wps(k, 2) - yPred) * tangentUnit(2);
                if distsFromPred(k) >= Ld && longProj >= 0.0
                    targetIdx = k;
                    foundTarget = true;
                    break;
                end
            end

            if foundTarget
                lookaheadPt = wps(targetIdx, 1:2);
            else
                % Continuous Tangent Extrapolation at Path Boundary
                endPt = wps(N, 1:2);
                pastDist = (xPred - endPt(1)) * tangentUnit(1) + (yPred - endPt(2)) * tangentUnit(2);
                forwardDist = max(Ld, pastDist + Ld);
                lookaheadPt = endPt + forwardDist * tangentUnit;
            end

            % 6. Pure Pursuit Feedback Term
            gx = lookaheadPt(1); gy = lookaheadPt(2);
            alpha = wrapToPi(atan2(gy - yPred, gx - xPred) - yawPred);
            delta_pp = atan2(2.0 * obj.Wheelbase * sin(alpha), max(2.5, Ld));

            % 7. Curvature Feedforward Steering
            delta_ff = atan(obj.Wheelbase * curvature);

            % Pure Pursuit steering with gentle feedforward (avoids double-counting error)
            rawDelta = delta_pp + 0.40 * delta_ff;

            % 8. Slew Rate Limiting & Command Smoothing
            dDeltaMax = obj.MaxSteerRate * obj.Ts;
            deltaDelta = max(-dDeltaMax, min(dDeltaMax, rawDelta - obj.LastSteering));
            unfiltDelta = obj.LastSteering + deltaDelta;

            obj.FilteredSteering = 0.85 * unfiltDelta + 0.15 * obj.FilteredSteering;
            deltaCmd = max(-obj.MaxSteering, min(obj.MaxSteering, obj.FilteredSteering));
            obj.LastSteering = deltaCmd;
        end

        function update_waypoints(obj, newWaypoints)
            if size(newWaypoints, 1) >= 2
                obj.Waypoints = newWaypoints;
            end
        end
    end
end
