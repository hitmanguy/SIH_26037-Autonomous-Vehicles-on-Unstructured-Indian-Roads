classdef autonomous_ego_controller < handle
% AUTONOMOUS_EGO_CONTROLLER Unified Closed-Loop Autonomous Vehicle Manager
% =========================================================================
% Combines:
% 1. Forward-Facing Camera Vision Perception (visionDetectionGenerator)
% 2. Longitudinal Speed Regulation & Collision-Avoidance Braking (ACC / AEB)
% 3. Modular Lateral Lane Tracking (Pure Pursuit or MPC)
% 4. Kinematic Bicycle Model State Propagation (updates ego actor in RAM)
% =========================================================================

    properties
        ControllerType = 'PurePursuit'  % 'PurePursuit' or 'MPC'
        LateralController               % pure_pursuit_controller or mpc_lane_controller
        FusionBridge                    % sensor_fusion_bridge handle
        CameraSensor                    % Fallback visionDetectionGenerator handle
        Waypoints                       % Reference path waypoints [X, Y]
        Ts = 0.05                       % Sample time (s)
        
        % Longitudinal Parameters
        CruiseSpeed = 6.94              % Desired cruise velocity (m/s) ~ 25 km/h steady pace
        SafeDistance = 20.0             % Safe headway threshold (m)
        StopDistance = 3.5              % Standstill emergency clearance (m)
        MaxDecel = -7.5                 % Max emergency deceleration (m/s^2)
        MaxAccel = 1.6                  % Max acceleration (m/s^2) - calm, steady, smooth
        
        % Vehicle Parameters
        Wheelbase = 2.8
        lf = 1.2
        lr = 1.6
        
        % Internal State: [X (m), Y (m), Yaw (rad), Vx (m/s)]
        CurrentState = [0, 0, 0, 0]
        
        % Telemetry
        LastSteering = 0.0
        LastAccel = 0.0
        LastMinObstacleDist = Inf
        NumDetections = 0
        NumTracks = 0
    end

    methods
        function obj = autonomous_ego_controller(waypoints, sensorOrBridge, controllerType, cruiseSpeed, sampleTime)
            if nargin >= 1, obj.Waypoints = waypoints; end
            if nargin >= 2 && ~isempty(sensorOrBridge)
                if isa(sensorOrBridge, 'sensor_fusion_bridge')
                    obj.FusionBridge = sensorOrBridge;
                else
                    obj.CameraSensor = sensorOrBridge;
                end
            end
            if nargin >= 3 && ~isempty(controllerType), obj.ControllerType = char(controllerType); end
            if nargin >= 4 && ~isempty(cruiseSpeed), obj.CruiseSpeed = cruiseSpeed; end
            if nargin >= 5 && ~isempty(sampleTime), obj.Ts = sampleTime; end

            % Initialize Lateral Controller
            if strcmpi(obj.ControllerType, 'MPC')
                obj.LateralController = mpc_lane_controller(obj.Waypoints, obj.Ts);
            else
                obj.ControllerType = 'PurePursuit';
                obj.LateralController = pure_pursuit_controller(obj.Waypoints, obj.CruiseSpeed);
            end
        end

        function init_state(obj, egoActor)
            % Initializes state vector from Ego vehicle starting pose
            pos = egoActor.Position;
            yaw_rad = deg2rad(egoActor.Yaw);
            v0 = norm(egoActor.Velocity(1:2));
            if v0 == 0
                v0 = min(3.0, obj.CruiseSpeed * 0.5); % Initial rolling start
            end
            obj.CurrentState = [pos(1), pos(2), yaw_rad, v0];
        end

        function telemetry = step(obj, egoActor, scenarioTime)
            % 1. Perception & Sensor Fusion Step
            minDist = Inf;
            vClosing = 0.0;
            leadClass = 'none';
            numTracks = 0;
            fusedTracks = [];
            cutinHazard = struct('Active', false, 'Distance', Inf, 'Class', 'none');

            if ~isempty(obj.FusionBridge)
                % Full Multi-Sensor Fusion World Model (4 Cameras + 2 Radars + LiDAR)
                [fusedTracks, leadObstacle, cutinHazard] = obj.FusionBridge.step(egoActor, scenarioTime);
                minDist = leadObstacle.Distance;
                leadClass = leadObstacle.Class;
                vClosing = max(0.0, -leadObstacle.ClosingVelocity);
                numTracks = numel(fusedTracks);
            elseif ~isempty(obj.CameraSensor)
                % Fallback: Single Forward Camera
                try
                    poses = targetPoses(egoActor);
                    if ~isempty(poses)
                        [dets, numDets, isValid] = obj.CameraSensor(poses, scenarioTime);
                        if isValid && numDets > 0
                            for d = 1:numDets
                                mPos = dets{d}.Measurement(1:2); % Relative [X_forward, Y_lateral]
                                if mPos(1) > 0 && abs(mPos(2)) < 1.75 % In-lane corridor
                                    if mPos(1) < minDist
                                        minDist = mPos(1);
                                    end
                                end
                            end
                        end
                        numTracks = numDets;
                    end
                catch
                end
            end
            obj.NumDetections = numTracks;
            obj.NumTracks = numTracks;
            obj.LastMinObstacleDist = minDist;

            % 2. Longitudinal Control: First-Principles Kinematic ACC & AEB
            vx = obj.CurrentState(4);
            effClosingSpeed = max(vx, vClosing);
            
            % Compute dynamic Time-to-Collision (TTC = d / v_rel)
            if effClosingSpeed > 0.3 && isfinite(minDist)
                ttc = minDist / effClosingSpeed;
            else
                ttc = Inf;
            end

            if minDist <= obj.StopDistance || ttc < 1.4
                % Standstill hold / critical emergency brake
                aCmd = obj.MaxDecel;
            elseif minDist < obj.SafeDistance || ttc < 3.2
                % First-principles kinematic stopping equation:
                % v_f^2 = v_0^2 + 2 * a * d  =>  a_req = -v_0^2 / (2 * (d - d_stop))
                availDist = max(0.5, minDist - obj.StopDistance);
                aKinematic = -(effClosingSpeed^2) / (2 * availDist);
                aCmd = max(obj.MaxDecel, min(-1.5, aKinematic));
            elseif cutinHazard.Active && cutinHazard.Distance < 25.0
                % Flank sensor detects cut-in vehicle entering lane: anticipate & yield smoothly
                aCmd = -4.0;
            else
                % Normal cruise control acceleration tracking target speed
                speedErr = obj.CruiseSpeed - vx;
                aCmd = max(-3.0, min(obj.MaxAccel, 1.2 * speedErr));
            end
            obj.LastAccel = aCmd;

            % 3. Lateral Control: Compute Front Steering Angle Delta
            currentPose = obj.CurrentState(1:3); % [X, Y, Yaw]
            if strcmpi(obj.ControllerType, 'MPC')
                [deltaCmd, targetPt, ey, epsi] = obj.LateralController.step(currentPose, vx);
            else
                [deltaCmd, targetPt, ~] = obj.LateralController.step(currentPose, vx);
                [ey, epsi] = obj.calc_lateral_error(currentPose(1), currentPose(2), currentPose(3));
            end
            obj.LastSteering = deltaCmd;

            % 4. Act: Vehicle Kinematic Bicycle Model State Update
            Ts = obj.Ts;
            x = obj.CurrentState(1);
            y = obj.CurrentState(2);
            psi = obj.CurrentState(3);

            % Sideslip angle beta
            beta = atan((obj.lr / obj.Wheelbase) * tan(deltaCmd));
            
            % Continuous-to-discrete bicycle model integration
            x_next = x + vx * cos(psi + beta) * Ts;
            y_next = y + vx * sin(psi + beta) * Ts;
            psi_next = wrapToPi(psi + (vx / obj.Wheelbase) * cos(beta) * tan(deltaCmd) * Ts);
            vx_next = max(0.0, vx + aCmd * Ts);

            obj.CurrentState = [x_next, y_next, psi_next, vx_next];

            % 5. Write State Directly to Ego Actor in drivingScenario Memory
            egoActor.Position = [x_next, y_next, 0.0];
            egoActor.Velocity = [vx_next * cos(psi_next), vx_next * sin(psi_next), 0.0];
            egoActor.Yaw = rad2deg(psi_next);

            % 6. Pack Comprehensive Telemetry Record
            telemetry = struct();
            telemetry.Time = scenarioTime;
            telemetry.Position = [x_next, y_next];
            telemetry.Yaw = psi_next;
            telemetry.Speed = vx_next;
            telemetry.Steering = deltaCmd;
            telemetry.Accel = aCmd;
            telemetry.LateralError = ey;
            telemetry.HeadingError = epsi;
            telemetry.TargetPoint = targetPt;
            telemetry.ObstacleDistance = minDist;
            telemetry.NumDetections = numTracks;
            telemetry.NumTracks = numTracks;
            telemetry.FusedTracks = fusedTracks;
            telemetry.LeadClass = leadClass;
            telemetry.CutInHazard = cutinHazard;
            telemetry.TTC = ttc;
            telemetry.Controller = obj.ControllerType;
            if ~isempty(obj.FusionBridge) && isprop(obj.FusionBridge, 'LatestCamDets')
                telemetry.CamDets = obj.FusionBridge.LatestCamDets;
                telemetry.RadDets = obj.FusionBridge.LatestRadDets;
                telemetry.PtCloud = obj.FusionBridge.LatestPtCloud;
            else
                telemetry.CamDets = {};
                telemetry.RadDets = {};
                telemetry.PtCloud = [];
            end
        end

        function [ey, epsi] = calc_lateral_error(obj, x, y, psi)
            wps = obj.Waypoints;
            dists = hypot(wps(:, 1) - x, wps(:, 2) - y);
            [~, idx] = min(dists);
            N = size(wps, 1);
            idx_next = mod(idx, N) + 1;

            p1 = wps(idx, :);
            p2 = wps(idx_next, :);
            tangentYaw = atan2(p2(2) - p1(2), p2(1) - p1(1));

            dx = x - p1(1);
            dy = y - p1(2);
            ey = -sin(tangentYaw)*dx + cos(tangentYaw)*dy;
            epsi = wrapToPi(psi - tangentYaw);
        end

        function set_reference_trajectory(obj, newWaypoints, targetSpeed)
            if size(newWaypoints, 1) >= 2
                obj.Waypoints = newWaypoints;
                if ismethod(obj.LateralController, 'update_waypoints')
                    obj.LateralController.update_waypoints(newWaypoints);
                elseif isprop(obj.LateralController, 'Waypoints')
                    obj.LateralController.Waypoints = newWaypoints;
                    if isprop(obj.LateralController, 'PPObject') && ~isempty(obj.LateralController.PPObject)
                        obj.LateralController.PPObject.Waypoints = newWaypoints;
                    end
                end
            end
            if nargin >= 3 && ~isempty(targetSpeed) && isscalar(targetSpeed) && targetSpeed > 0
                obj.CruiseSpeed = targetSpeed;
            end
        end
    end
end
