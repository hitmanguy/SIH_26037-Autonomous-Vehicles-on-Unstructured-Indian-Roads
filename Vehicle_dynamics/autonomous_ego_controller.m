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
        
        % Dynamic Strategic Trajectory Replanning
        DynamicPlanner                      % Instance of dynamic_trajectory_planner
        EnableDynamicReplanning = true      % Real-time dynamic replanning toggle
        OriginalRouteWaypoints              % Pristine route reference waypoints
        LastPlanTime = -1.0                 % Sim time of last plan step
        PlanInterval = 0.10                 % 10 Hz replan interval (100 ms)
        PlanInfo = struct()                 % Latest planning telemetry
        
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

            % Initialize pristine route waypoints & SOTA Dynamic Replanner
            obj.OriginalRouteWaypoints = obj.Waypoints;
            obj.DynamicPlanner = dynamic_trajectory_planner(obj.CruiseSpeed, [-6.20, -0.60]);

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

            % 1.5. Dynamic Strategic Trajectory Replanning (@ 10 Hz / 100 ms)
            % Continuously evaluates forward corridor for in-lane obstacles / hazards.
            % Dynamically generates smooth, collision-free C^2 evasive trajectories to Lane -2 on our own side
            % and updates active waypoints for MPC / Pure Pursuit with persistent lane commitment!
            planInfo = struct('SelectedTargetY', obj.CurrentState(2), 'IsEvasive', false, 'LaneCommitted', false, 'TargetSpeed', obj.CruiseSpeed);
            if obj.EnableDynamicReplanning && ~isempty(obj.DynamicPlanner) && ...
               (scenarioTime - obj.LastPlanTime >= obj.PlanInterval - 1e-4)
                obj.LastPlanTime = scenarioTime;
                
                % Feed current pose, route waypoints, fused tracks, and steer angle
                [newWps, pInfo] = obj.DynamicPlanner.replan(...
                    obj.CurrentState, obj.OriginalRouteWaypoints, fusedTracks, obj.LastSteering, scenarioTime);
                
                if ~isempty(newWps) && size(newWps, 1) >= 4
                    obj.set_reference_trajectory(newWps, pInfo.TargetSpeed);
                    planInfo = pInfo;
                    obj.PlanInfo = pInfo;
                end
            elseif ~isempty(obj.PlanInfo) && isfield(obj.PlanInfo, 'IsEvasive')
                planInfo = obj.PlanInfo;
            end

            % 2. Longitudinal Control: SOTA ACC Car-Following & Emergency AEB
            vx = obj.CurrentState(4);
            vRelClosing = vClosing; % Rate of range decrease (m/s)

            dStandstill = obj.StopDistance; % 3.5m emergency clearance
            timeHeadway = 1.5;              % 1.5s comfortable following headway
            desiredGap  = dStandstill + vx * timeHeadway;

            % Dynamic TTC (time to bumper contact)
            if vRelClosing > 0.4 && isfinite(minDist)
                ttc = max(0.01, minDist - dStandstill) / vRelClosing;
            else
                ttc = Inf;
            end

            isEvasive = (isfield(planInfo, 'IsEvasive') && planInfo.IsEvasive) || ...
                        (isfield(planInfo, 'LaneCommitted') && planInfo.LaneCommitted);

            % Priority 0 Guard: Critical emergency braking commanded by planner or bumper collision
            isPlannerStop = (isfield(planInfo, 'IsEmergencyBraking') && planInfo.IsEmergencyBraking) || ...
                            (isfield(planInfo, 'TargetSpeed') && planInfo.TargetSpeed <= 0.1);

            if isPlannerStop || minDist <= dStandstill || ttc < 1.4
                % Critical Emergency Braking (AEB)
                aCmd = obj.MaxDecel;
            elseif isfinite(minDist) && minDist < 45.0
                % Adaptive Cruise Control (ACC) Car-Following Law
                gapErr = minDist - desiredGap;

                if gapErr < -2.0
                    % Closer than desired headway: smoothly decelerate to open up headway
                    aDecel = -0.6 * vRelClosing + 0.35 * gapErr;
                    aCmd = max(obj.MaxDecel, min(-0.4, aDecel));
                elseif gapErr < 4.0
                    % Within headway transition zone: smoothly match lead speed
                    vTarget = min(obj.CruiseSpeed, max(2.5, vx - 0.7 * vRelClosing + 0.20 * gapErr));
                    if isfield(planInfo, 'TargetSpeed') && planInfo.TargetSpeed > 0.5
                        vTarget = min(vTarget, planInfo.TargetSpeed);
                    end
                    aCmd = max(-2.0, min(obj.MaxAccel, 1.0 * (vTarget - vx)));
                else
                    % Ample clearance (> desired gap): cruise at target pace
                    targetSpeed = obj.CruiseSpeed;
                    if isfield(planInfo, 'TargetSpeed') && planInfo.TargetSpeed > 0.5
                        targetSpeed = min(targetSpeed, planInfo.TargetSpeed);
                    end
                    aCmd = max(-1.5, min(obj.MaxAccel, 1.0 * (targetSpeed - vx)));
                end
            elseif cutinHazard.Active && cutinHazard.Distance < 25.0
                % Cut-in vehicle detected entering our lane: yield smoothly
                aCmd = -2.5;
            else
                % Normal Cruise Control with curvature coupling
                dists_curv = hypot(obj.Waypoints(:, 1) - obj.CurrentState(1), obj.Waypoints(:, 2) - obj.CurrentState(2));
                [~, cIdx] = min(dists_curv);
                N_wps = size(obj.Waypoints, 1);
                i0 = max(1, cIdx - 3);
                i2 = min(N_wps, cIdx + 3);
                p0 = obj.Waypoints(i0, 1:2);
                p1 = obj.Waypoints(cIdx, 1:2);
                p2 = obj.Waypoints(i2, 1:2);
                th1 = atan2(p1(2) - p0(2), p1(1) - p0(1));
                th2 = atan2(p2(2) - p1(2), p2(1) - p1(1));
                chord = max(0.5, hypot(p2(1) - p0(1), p2(2) - p0(2)));
                curvNow = abs(wrapToPi(th2 - th1) / chord);

                aLatMax = 1.5; % Comfortable lateral acceleration limit (m/s^2)
                vCurvLimit = sqrt(aLatMax / max(1e-4, curvNow));
                dynamicTargetSpeed = min(obj.CruiseSpeed, max(4.0, vCurvLimit));
                if isfield(planInfo, 'TargetSpeed') && planInfo.TargetSpeed > 0.5
                    dynamicTargetSpeed = min(dynamicTargetSpeed, planInfo.TargetSpeed);
                end

                speedErr = dynamicTargetSpeed - vx;
                aCmd = max(-2.5, min(obj.MaxAccel, 1.2 * speedErr));
            end
            obj.LastAccel = aCmd;

            % 3. Lateral Control: Compute Front Steering Angle Delta (Predictive PP or MPC)
            currentPose = obj.CurrentState(1:3); % [X, Y, Yaw]
            if strcmpi(obj.ControllerType, 'MPC')
                [deltaCmd, targetPt, ey, epsi] = obj.LateralController.step(currentPose, vx);
            else
                [deltaCmd, targetPt, ~, ey, epsi] = obj.LateralController.step(currentPose, vx);
            end
            obj.LastSteering = deltaCmd;

            % 4. Act: Vehicle Kinematic Bicycle Model State Update with Full 2D Velocity (Longitudinal & Lateral)
            Ts = obj.Ts;
            x = obj.CurrentState(1);
            y = obj.CurrentState(2);
            psi = obj.CurrentState(3);

            % Sideslip angle beta at vehicle CG
            beta = atan((obj.lr / obj.Wheelbase) * tan(deltaCmd));
            
            % Full 2D velocity vector in world frame (coupling longitudinal and lateral velocity)
            VX = vx * cos(psi + beta);
            VY = vx * sin(psi + beta);

            % State integration
            x_next = x + VX * Ts;
            y_next = y + VY * Ts;
            psi_next = wrapToPi(psi + (vx / obj.Wheelbase) * cos(beta) * tan(deltaCmd) * Ts);
            vx_next = max(0.0, vx + aCmd * Ts);

            obj.CurrentState = [x_next, y_next, psi_next, vx_next];

            % 5. Write State Directly to Ego Actor in drivingScenario Memory
            % Updates BOTH position and 2D velocity (longitudinal and lateral components)
            egoActor.Position = [x_next, y_next, 0.0];
            egoActor.Velocity = [vx_next * cos(psi_next + beta), vx_next * sin(psi_next + beta), 0.0];
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
            telemetry.PlanInfo = planInfo;
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
            N = size(wps, 1);
            if N < 2
                ey = 0; epsi = 0; return;
            end
            dists = hypot(wps(:, 1) - x, wps(:, 2) - y);
            [~, idx] = min(dists);

            if idx >= N
                p1 = wps(max(1, N - 1), 1:2);
                p2 = wps(N, 1:2);
                refPt = p2;
            else
                p1 = wps(idx, 1:2);
                p2 = wps(idx + 1, 1:2);
                refPt = p1;
            end
            tangentYaw = atan2(p2(2) - p1(2), p2(1) - p1(1));

            dx = x - refPt(1);
            dy = y - refPt(2);
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
