classdef pure_pursuit_controller < handle
% PURE_PURSUIT_CONTROLLER Closed-Loop Geometric Lateral Tracking Controller
% =========================================================================
% Wraps MATLAB's controllerPurePursuit with adaptive lookahead scaling and
% dynamic bicycle steering angle mapping.
% =========================================================================

    properties
        PPObject             % MATLAB controllerPurePursuit instance
        Waypoints            % Nx2 reference path waypoints [X, Y]
        Wheelbase = 2.8      % Vehicle wheelbase L (meters)
        MinLookahead = 3.0   % Minimum lookahead distance (meters)
        MaxLookahead = 10.0  % Maximum lookahead distance (meters)
        LookaheadGain = 0.35 % Velocity lookahead scaling factor (s)
        MaxSteering = 0.52   % Maximum front steering angle (rad) ~ 30 deg
    end

    methods
        function obj = pure_pursuit_controller(waypoints, initialSpeed)
            if nargin < 2, initialSpeed = 10.0; end
            obj.Waypoints = waypoints;

            initLookahead = min(obj.MaxLookahead, max(obj.MinLookahead, obj.LookaheadGain * initialSpeed));

            if ~isempty(which('controllerPurePursuit'))
                obj.PPObject = controllerPurePursuit(...
                    'Waypoints', waypoints, ...
                    'LookaheadDistance', initLookahead, ...
                    'DesiredLinearVelocity', initialSpeed, ...
                    'MaxAngularVelocity', 1.5);
            else
                obj.PPObject = [];
            end
        end

        function [deltaCmd, lookaheadPt, Ld] = step(obj, currentPose, currentSpeed)
            % currentPose: [X (m), Y (m), Yaw (rad)]
            % currentSpeed: longitudinal velocity Vx (m/s)

            % 1. Dynamic Lookahead Distance Adaptation
            Ld = min(obj.MaxLookahead, max(obj.MinLookahead, obj.LookaheadGain * abs(currentSpeed)));

            if ~isempty(obj.PPObject)
                obj.PPObject.LookaheadDistance = Ld;
                obj.PPObject.DesiredLinearVelocity = max(1.0, currentSpeed);

                % 2. Query Pure Pursuit Angular Velocity Command
                [~, omega] = obj.PPObject(currentPose);

                % 3. Convert Yaw Rate (omega) to Front Wheel Steering Angle (delta)
                vxEff = max(1.5, abs(currentSpeed));
                rawDelta = atan((omega * obj.Wheelbase) / vxEff);
            else
                % Native Geometric Pure Pursuit fallback (Zero toolbox dependency)
                x = currentPose(1);
                y = currentPose(2);
                yaw = currentPose(3);
                dists = hypot(obj.Waypoints(:, 1) - x, obj.Waypoints(:, 2) - y);
                [~, closestIdx] = min(dists);
                targetIdx = min(closestIdx + 3, size(obj.Waypoints, 1));
                for k = closestIdx:min(closestIdx + 50, size(obj.Waypoints, 1))
                    if dists(k) >= Ld
                        targetIdx = k;
                        break;
                    end
                end
                gx = obj.Waypoints(targetIdx, 1);
                gy = obj.Waypoints(targetIdx, 2);
                alpha = wrapToPi(atan2(gy - y, gx - x) - yaw);
                rawDelta = atan2(2 * obj.Wheelbase * sin(alpha), max(Ld, 1.0));
            end

            % 4. Bound Steering Angle
            deltaCmd = max(-obj.MaxSteering, min(obj.MaxSteering, rawDelta));

            % 5. Estimate Lookahead Target Point for HUD Visualization
            x = currentPose(1);
            y = currentPose(2);
            dists = hypot(obj.Waypoints(:, 1) - x, obj.Waypoints(:, 2) - y);
            % Find closest waypoint ahead with distance approx Ld
            [~, closestIdx] = min(dists);
            targetIdx = closestIdx;
            for k = closestIdx:min(closestIdx + 50, size(obj.Waypoints, 1))
                if dists(k) >= Ld
                    targetIdx = k;
                    break;
                end
            end
            lookaheadPt = obj.Waypoints(targetIdx, :);
        end

        function update_waypoints(obj, newWaypoints)
            if size(newWaypoints, 1) >= 2
                obj.Waypoints = newWaypoints;
                if ~isempty(obj.PPObject)
                    obj.PPObject.Waypoints = newWaypoints;
                end
            end
        end
    end
end
