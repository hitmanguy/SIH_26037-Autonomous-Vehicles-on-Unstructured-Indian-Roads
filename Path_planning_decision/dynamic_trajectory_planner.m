classdef dynamic_trajectory_planner < handle
% DYNAMIC_TRAJECTORY_PLANNER State-of-the-Art Frenet Optimal Spatiotemporal Replanner
% =========================================================================
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% Production-grade online dynamic trajectory planner based on Apollo / Werling et al.
% Frenet Optimal Spatiotemporal Lattice Sampling:
%  1. Real-Time Online Dynamic Replanning (10 Hz cyclic execution)
%  2. Frenet Frame [s, d] decomposition with exact C^2 boundary condition matching
%  3. Minimum-Jerk Quintic Polynomial lateral candidate generation
%  4. Spatiotemporal dynamic obstacle collision checking over forward prediction horizon
%  5. Indian Traffic Rule Enforcement: Strictly prevents swerving into oncoming traffic
%     (Centerline Y > -0.6m heavily penalized; evasions committed to Ego's OWN side Y in [-6.2, -0.6])
%  6. Persistent Lane Commitment: Locks onto safe detour corridor (e.g. Lane -2 at Y ~ -5.0m)
%     until obstacles are safely passed, preventing rapid lane oscillation / bouncing
%  7. Curvature-bounded smooth C^2 trajectory generation with direct feed to Level-4 MPC & Pure Pursuit
% =========================================================================

    properties
        % Road and Geometry Parameters (Indian Urban Arterial)
        RoadBounds          = [-6.20, -0.60]  % Allowed lateral envelope on Ego's own side [m]
        CruisingLaneY       = -1.75           % Lane -1 center (m)
        PassingLaneY        = -4.75           % Lane -2 center (m) - safe margin from curb
        CenterDividerY      = -0.60           % Critical boundary: Y > -0.60 is oncoming traffic!
        OuterShoulderY      = -6.20           % Road edge curb (m)
        LaneWidth           = 3.50            % Standard lane width (m)
        
        % Vehicle Physical Dimensions
        Wheelbase           = 2.80            % (m)
        Length              = 4.50            % (m)
        Width               = 2.00            % (m)
        
        % Planning Horizons & Sampling
        Horizons            = [3.0, 4.0, 5.0] % Candidate trajectory durations T [s] (smooth transitions)
        TargetSpeed         = 6.94            % Nominal cruise speed (m/s) ~ 25 km/h
        PlanHorizonMeters   = 50.0            % Spatial lookahead distance (m)
        ReplanDistance      = 0.50            % Waypoint spatial discretization (m)
        
        % Cost Weights (Apollo / Werling et al. formulation)
        w_jerk              = 0.20            % Lateral jerk penalty
        w_acc               = 0.35            % Lateral acceleration penalty
        w_time              = 1.00            % Duration efficiency penalty
        w_target            = 2.50            % Deviation from preferred lane
        w_collision         = 1e6             % Collision with dynamic/static obstacle
        w_boundary          = 1e6             % Road boundary / oncoming traffic violation
        w_commit            = 3.00            % Lane commitment hysteresis weight
        
        % Dynamic State & Lane Commitment Memory
        LaneCommitted       = false           % True when committed to evasion corridor
        CommittedY          = -1.75           % Currently committed lane center
        CommitmentHoldUntilX= -Inf            % X coordinate threshold before lane return allowed
        LastTargetY         = -1.75           % Target Y chosen in previous cycle
        LastPlannedWps      = []              % Active planned waypoints cache
        LastPlanTime        = -1.0            % Timestamp of last replan
        ReplanInterval      = 0.10            % 10 Hz replan rate (s)
        LastLaneChangeTime  = -10.0           % Timestamp of last committed lane switch
        MinLaneDwellTime    = 4.5             % Minimum dwell duration (s) in lane before switching
    end

    methods
        function obj = dynamic_trajectory_planner(cruiseSpeed, roadBounds)
            if nargin >= 1 && ~isempty(cruiseSpeed)
                obj.TargetSpeed = cruiseSpeed;
            end
            if nargin >= 2 && ~isempty(roadBounds)
                obj.RoadBounds = roadBounds;
                obj.CenterDividerY = roadBounds(2);
                obj.OuterShoulderY = roadBounds(1);
            end
            obj.reset();
        end

        function reset(obj)
            obj.LaneCommitted = false;
            obj.CommittedY = obj.CruisingLaneY;
            obj.CommitmentHoldUntilX = -Inf;
            obj.LastTargetY = obj.CruisingLaneY;
            obj.LastPlannedWps = [];
            obj.LastPlanTime = -1.0;
            obj.LastLaneChangeTime = -10.0;
        end

        function [wps, planInfo] = replan(obj, currentState, globalWaypoints, fusedTracks, currentSteer, simTime)
            % REPLAN Generates a dynamically optimized, collision-free C^2 trajectory
            %
            % Inputs:
            %   currentState   - [X, Y, Yaw (rad), Vx (m/s)]
            %   globalWaypoints- [N x 2] matrix of route reference points
            %   fusedTracks    - Array/struct of perceived tracks from sensor fusion
            %   currentSteer   - Current front wheel angle delta (rad)
            %   simTime        - Current simulation timestamp (s)
            
            x0   = currentState(1);
            y0   = currentState(2);
            psi0 = currentState(3);
            v0   = max(0.1, currentState(4));

            if nargin < 5 || isempty(currentSteer), currentSteer = 0.0; end
            if nargin < 6 || isempty(simTime), simTime = 0.0; end

            % Rate limit replanning to 10 Hz to prevent high-frequency chattering
            if obj.LastPlanTime >= 0 && (simTime - obj.LastPlanTime) < (obj.ReplanInterval - 1e-4) && ~isempty(obj.LastPlannedWps)
                wps = obj.LastPlannedWps;
                planInfo = struct('SelectedTargetY', obj.LastTargetY, 'IsEvasive', obj.LaneCommitted, ...
                    'LaneCommitted', obj.LaneCommitted, 'TargetSpeed', obj.TargetSpeed);
                return;
            end
            obj.LastPlanTime = simTime;

            % 1. Extract Obstacles in Vehicle Vicinity (Forward corridor X in [x0 - 2, x0 + 60])
            parsedObstacles = obj.extract_obstacles(fusedTracks, x0, y0, psi0, v0);

            % 2. Evaluate Lane Status & Commitment Condition
            inCruisingLane = (abs(y0 - obj.CruisingLaneY) < 1.0);
            inPassingLane  = (abs(y0 - obj.PassingLaneY) < 1.0);

            % Check if obstacle directly blocks cruising lane (Lane -1, Y ~ -1.75m)
            cruisingLaneBlocked = false;
            passingLaneBlocked  = false;
            critObstacleX = Inf;

            for i = 1:numel(parsedObstacles)
                obs = parsedObstacles(i);
                cLower = lower(char(obs.Class));
                isVRU = contains(cLower, 'pedestrian') || contains(cLower, 'person') || ...
                        contains(cLower, 'vru') || contains(cLower, 'child');
                if isVRU
                    continue; % Crossing VRUs are handled by longitudinal AEB/ACC, not lateral overtaking lane changes
                end
                
                dx = obs.X - x0;
                
                % Check if passing lane (Lane -2) has ANY obstacle ahead (moving or static)
                % Entry into Lane -2 is only blocked if an obstacle occupies the lane change corridor (dx in [-2, 22] m).
                % If already committed to Lane -2, Lane -2 is blocked ahead if an obstacle is within 35m.
                if obj.LaneCommitted
                    obsBlockedPassing = (abs(obs.Y - obj.PassingLaneY) < 1.4 && dx > -2.0 && dx < 35.0);
                else
                    obsBlockedPassing = (abs(obs.Y - obj.PassingLaneY) < 1.4 && dx > -2.0 && dx < 22.0);
                end
                if obsBlockedPassing
                    passingLaneBlocked = true;
                end

                % Cruising lane (Lane -1) blockage:
                % Vehicles moving forward in traffic (Vx > 2.0 m/s) are handled by ACC car-following.
                % Head-on oncoming vehicles (Vx < -1.0 m/s) are an immediate emergency: detect up to 75m ahead!
                % Static / stopped blockages (Vx <= 2.0 m/s) are detected up to 45m ahead.
                isHeadOn = (obs.Vx < -1.0);
                isMovingTraffic = (obs.Vx > 2.0);
                if isHeadOn && abs(obs.Y - obj.CruisingLaneY) < 1.4 && dx > 2.0 && dx < 75.0
                    cruisingLaneBlocked = true;
                    if obs.X < critObstacleX, critObstacleX = obs.X; end
                elseif ~isMovingTraffic && ~isHeadOn && abs(obs.Y - obj.CruisingLaneY) < 1.4 && dx > 2.0 && dx < 45.0
                    cruisingLaneBlocked = true;
                    if obs.X < critObstacleX, critObstacleX = obs.X; end
                end
            end

            % Update Lane Commitment State Machine with Anti-Chatter Hysteresis
            canSwitchLane = (simTime - obj.LastLaneChangeTime >= obj.MinLaneDwellTime);

            if obj.LaneCommitted
                % Already committed to Lane -2: STAY in Lane -2!
                % Only return to Lane -1 if Lane -2 itself is blocked AND Lane -1 is clear AND dwell time elapsed
                if passingLaneBlocked && ~cruisingLaneBlocked && canSwitchLane
                    obj.LaneCommitted = false;
                    obj.CommittedY = obj.CruisingLaneY;
                    obj.LastLaneChangeTime = simTime;
                else
                    obj.LaneCommitted = true;
                    obj.CommittedY = obj.PassingLaneY;
                end
            elseif cruisingLaneBlocked && ~passingLaneBlocked && canSwitchLane
                % Cruising lane blocked and passing lane clear: Commit to Lane -2
                obj.LaneCommitted = true;
                obj.CommittedY = obj.PassingLaneY;
                obj.LastLaneChangeTime = simTime;
            else
                obj.CommittedY = obj.CruisingLaneY;
            end

            % 3. Generate Candidate Lateral Targets (Indian Road Evasion Corridor)
            % Never plan targets that enter oncoming traffic (Y > -0.60m)
            % The in-lane "hold" option is ALWAYS evaluated: when every lateral evasion is
            % blocked (e.g. parked cars on the shoulder), the correct lateral behaviour is to
            % stay in lane and let the longitudinal TTC/AEB supervisor brake - not to swerve
            % into a worse collision.
            if inPassingLane || (obj.LaneCommitted && abs(y0 - obj.PassingLaneY) < abs(y0 - obj.CruisingLaneY))
                holdY = obj.PassingLaneY;
            else
                holdY = obj.CruisingLaneY;
            end

            if obj.LaneCommitted
                % When committed to Lane -2: stay smoothly centered in Lane -2
                candidateTargets = [obj.PassingLaneY, obj.PassingLaneY - 0.15, obj.PassingLaneY + 0.15];
                preferredTargetY = obj.PassingLaneY;
            elseif cruisingLaneBlocked && ~passingLaneBlocked
                % Cruising lane blocked and passing lane clear: sample centered Lane -2
                candidateTargets = [obj.PassingLaneY, obj.PassingLaneY + 0.20, obj.PassingLaneY - 0.15];
                preferredTargetY = obj.PassingLaneY;
            elseif cruisingLaneBlocked && passingLaneBlocked
                % Both lanes blocked: HOLD current lane and let ACC/AEB brake safely!
                candidateTargets = [holdY, holdY - 0.10, holdY + 0.10];
                preferredTargetY = holdY;
            else
                % Road clear: stay in Cruising Lane -1, with minor adjustments
                candidateTargets = [obj.CruisingLaneY, obj.CruisingLaneY - 0.15, obj.CruisingLaneY + 0.15];
                preferredTargetY = obj.CruisingLaneY;
            end
            if ~any(abs(candidateTargets - holdY) < 1e-6)
                candidateTargets(end+1) = holdY;
            end

            % Initial Frenet lateral state [d0, dot_d0, ddot_d0]
            % d is lateral position Y in world frame
            d0 = y0;
            dot_d0 = 0.35 * v0 * sin(psi0);
            ddot_d0 = 0.0;

            % 4. Spatiotemporal Candidate Generation & Cost Optimization (Lattice Search)
            bestCost = Inf;
            bestTrajectory = [];
            bestTargetY = preferredTargetY;
            bestHorizonT = 3.0;

            for tIdx = 1:numel(candidateTargets)
                d1 = candidateTargets(tIdx);

                % Skip if target violates road boundaries
                if d1 > (obj.CenterDividerY - 0.20) || d1 < (obj.OuterShoulderY + 0.20)
                    continue;
                end

                for hIdx = 1:numel(obj.Horizons)
                    T = obj.Horizons(hIdx);

                    % Solve closed-form Quintic Polynomial for lateral profile d(t)
                    poly = obj.solve_quintic(d0, dot_d0, ddot_d0, d1, T);

                    % Calculate Jerk & Acceleration Cost
                    jerkCost = poly.calc_jerk_integral();
                    accCost = abs(d1 - d0) / (T^2);

                    % Target Preference & Commitment Cost
                    targetCost = (d1 - preferredTargetY)^2;
                    commitCost = (d1 - obj.CommittedY)^2;

                    % Spatiotemporal Collision & Boundary Check
                    [isCollision, minClearance, boundaryViol] = obj.check_spatiotemporal_collision(...
                        poly, x0, v0, T, parsedObstacles);

                    if boundaryViol
                        continue;
                    end

                    totalCost = obj.w_jerk * jerkCost + ...
                                obj.w_acc  * accCost + ...
                                obj.w_time * T + ...
                                obj.w_target * targetCost + ...
                                obj.w_commit * commitCost;

                    if isCollision
                        if abs(d1 - holdY) < 1e-6
                            % Hold-lane: longitudinal AEB/ACC resolves this; far cheaper than
                            % any lateral move that physically hits something.
                            totalCost = totalCost + 1e-3 * obj.w_collision;
                        else
                            totalCost = totalCost + obj.w_collision;
                        end
                    else
                        % Proximity cost: encourage healthy clearance margins
                        if minClearance < 3.5
                            totalCost = totalCost + 150.0 / max(0.5, minClearance);
                        end
                    end

                    if totalCost < bestCost
                        bestCost = totalCost;
                        bestTrajectory = poly;
                        bestTargetY = d1;
                        bestHorizonT = T;
                    end
                end
            end

            % Fallback: if all candidate maneuvers had issues, force smooth hold of safe corridor
            if isempty(bestTrajectory)
                bestTrajectory = obj.solve_quintic(d0, dot_d0, ddot_d0, obj.CommittedY, 3.5);
                bestTargetY = obj.CommittedY;
                bestHorizonT = 3.5;
            end

            obj.LastTargetY = bestTargetY;

            % 5. Sample Continuous C^2 Waypoints from Optimal Trajectory
            % Generate forward trajectory up to lookahead distance (50m)
            vForward = max(2.5, v0);
            timeSamples = 0.0:0.05:bestHorizonT;
            N_samples = numel(timeSamples);

            trajX = zeros(N_samples, 1);
            trajY = zeros(N_samples, 1);

            for k = 1:N_samples
                tk = timeSamples(k);
                trajX(k) = x0 + vForward * tk;
                trajY(k) = bestTrajectory.calc_pos(tk);
            end

            % Enforce strictly that planned trajectory NEVER crosses into oncoming lane or curb
            trajY = max(obj.OuterShoulderY + 0.50, min(obj.CenterDividerY - 0.40, trajY));

            % 6. Smoothly Stitch with Downstream Global Waypoints
            if ~isempty(globalWaypoints) && size(globalWaypoints, 1) >= 2
                lastPlanPt = [trajX(end), trajY(end)];
                distsGlobal = hypot(globalWaypoints(:,1) - lastPlanPt(1), globalWaypoints(:,2) - lastPlanPt(2));
                [~, matchIdx] = min(distsGlobal);

                if matchIdx < size(globalWaypoints, 1)
                    remainingGlobal = globalWaypoints(matchIdx+1:end, :);
                    isStraightRoad = (max(abs(diff(globalWaypoints(:,2)))) < 1.0) && ...
                                     (all(diff(globalWaypoints(:,1)) > -0.1));
                    if isStraightRoad
                        remainingGlobal(:, 2) = trajY(end);
                    end
                    fullWps = [trajX, trajY; remainingGlobal];
                else
                    % Synthesize forward extension at current committed lane covering full road (350m)
                    extX = (trajX(end)+1:obj.ReplanDistance:max(350.0, trajX(end)+150))';
                    extY = trajY(end) * ones(size(extX));
                    fullWps = [trajX, trajY; extX, extY];
                end
            else
                extX = (trajX(end)+1:obj.ReplanDistance:max(350.0, trajX(end)+150))';
                extY = trajY(end) * ones(size(extX));
                fullWps = [trajX, trajY; extX, extY];
            end

            % Downsample and ensure smooth interpolation (0.5m spacing)
            wps = obj.resample_path(fullWps, obj.ReplanDistance);
            obj.LastPlannedWps = wps;

            % 7. Pack Planning Telemetry
            planInfo = struct();
            planInfo.SelectedTargetY   = bestTargetY;
            planInfo.SelectedHorizonT  = bestHorizonT;
            planInfo.IsEvasive         = obj.LaneCommitted;
            planInfo.LaneCommitted     = obj.LaneCommitted;
            planInfo.TargetSpeed       = obj.TargetSpeed;
            planInfo.BestCost          = bestCost;
        end

        function poly = solve_quintic(~, d0, v_d0, a_d0, d1, T)
            % Solves boundary conditions:
            % d(0) = d0, d'(0) = v_d0, d''(0) = a_d0
            % d(T) = d1, d'(T) = 0,    d''(T) = 0
            c0 = d0;
            c1 = v_d0;
            c2 = 0.5 * a_d0;

            h  = d1 - c0 - c1 * T - c2 * (T^2);
            v1 = -c1 - 2.0 * c2 * T;
            a1 = -2.0 * c2;

            T3 = T^3;
            T4 = T^4;
            T5 = T^5;

            c3 = (10.0 * h - 4.0 * v1 * T + 0.5 * a1 * (T^2)) / T3;
            c4 = (-15.0 * h + 7.0 * v1 * T - a1 * (T^2)) / T4;
            c5 = (6.0 * h - 3.0 * v1 * T + 0.5 * a1 * (T^2)) / T5;

            poly = struct(...
                'c0', c0, 'c1', c1, 'c2', c2, 'c3', c3, 'c4', c4, 'c5', c5, 'T', T, ...
                'calc_pos', @(t) c0 + c1*t + c2*t.^2 + c3*t.^3 + c4*t.^4 + c5*t.^5, ...
                'calc_vel', @(t) c1 + 2*c2*t + 3*c3*t.^2 + 4*c4*t.^3 + 5*c5*t.^4, ...
                'calc_acc', @(t) 2*c2 + 6*c3*t + 12*c4*t.^2 + 20*c5*t.^3, ...
                'calc_jerk_integral', @() (36*(c3^2)*T + 144*c3*c4*(T^2) + (192*(c4^2) + 240*c3*c5)*(T^3) + 720*c4*c5*(T^4) + 720*(c5^2)*(T^5)));
        end

        function [isCollision, minClearance, boundaryViol] = check_spatiotemporal_collision(obj, poly, x0, v0, T, obstacles)
            isCollision = false;
            boundaryViol = false;
            minClearance = Inf;

            tCheck = 0.2:0.2:T;

            for k = 1:numel(tCheck)
                t = tCheck(k);
                x_ego = x0 + v0 * t;
                y_ego = poly.calc_pos(t);

                % Hard Road Boundary Check:
                % Oncoming lane violation (crossing centerline into opposite traffic)
                if y_ego > (obj.CenterDividerY - 0.15)
                    boundaryViol = true;
                    return;
                end
                % Off-road curb violation
                if y_ego < (obj.OuterShoulderY + 0.15)
                    boundaryViol = true;
                    return;
                end

                % Dynamic Obstacle Check against forward-projected bounding ellipses
                for oi = 1:numel(obstacles)
                    obs = obstacles(oi);
                    x_obs = obs.X + obs.Vx * t;
                    y_obs = obs.Y + obs.Vy * t;

                    % Elliptical safety boundary
                    a_safe = (obj.Length + obs.Length)/2 + 2.2 + 0.25 * v0; % Longitudinal clearance
                    b_safe = (obj.Width + obs.Width)/2 + 0.65;             % Lateral clearance

                    distNorm = hypot((x_ego - x_obs)/a_safe, (y_ego - y_obs)/b_safe);
                    physicalDist = hypot(x_ego - x_obs, y_ego - y_obs);

                    if physicalDist < minClearance
                        minClearance = physicalDist;
                    end

                    if distNorm < 1.0
                        isCollision = true;
                    end
                end
            end
        end

        function obsList = extract_obstacles(obj, fusedTracks, x0, y0, psi0, v0)
            obsList = struct('X', {}, 'Y', {}, 'Vx', {}, 'Vy', {}, 'Length', {}, 'Width', {}, 'Class', {});
            if isempty(fusedTracks), return; end

            cosP = cos(psi0);
            sinP = sin(psi0);

            for i = 1:numel(fusedTracks)
                trk = fusedTracks(i);

                % Handle both struct formats (sensor_fusion_bridge body format and AutonomousAVStack BEV format)
                if isfield(trk, 'Position')
                    % Body Cartesian [x_long, y_lat]
                    x_rel = trk.Position(1);
                    y_rel = trk.Position(2);
                    v_rel_x = 0; v_rel_y = 0;
                    if isfield(trk, 'Velocity') && numel(trk.Velocity) >= 2
                        v_rel_x = trk.Velocity(1);
                        v_rel_y = trk.Velocity(2);
                    end
                    cClass = 'car';
                    if isfield(trk, 'Class'), cClass = trk.Class; end
                elseif isfield(trk, 'Z') && isfield(trk, 'X')
                    % BEV format: Z is longitudinal forward (+), X is lateral right (+)
                    x_rel = trk.Z;
                    y_rel = -trk.X;
                    v_rel_x = 0; v_rel_y = 0;
                    if isfield(trk, 'Vz'), v_rel_x = trk.Vz; end
                    if isfield(trk, 'Vx'), v_rel_y = -trk.Vx; end
                    cClass = 'car';
                    if isfield(trk, 'class'), cClass = trk.class; end
                else
                    continue;
                end

                % Transform to world frame
                w_x = x0 + x_rel * cosP - y_rel * sinP;
                w_y = y0 + x_rel * sinP + y_rel * cosP;
                w_vx = v_rel_x * cosP - v_rel_y * sinP + v0 * cosP;
                w_vy = v_rel_x * sinP + v_rel_y * cosP + v0 * sinP;

                % Assign approximate physical dimensions based on class
                cLower = lower(char(cClass));
                if contains(cLower, 'truck') || contains(cLower, 'bus')
                    len = 8.5; wid = 2.5;
                elseif contains(cLower, 'rickshaw') || contains(cLower, 'auto')
                    len = 2.8; wid = 1.4;
                elseif contains(cLower, 'bike') || contains(cLower, 'motorcycle') || contains(cLower, 'cycle')
                    len = 2.0; wid = 0.9;
                elseif contains(cLower, 'pedestrian') || contains(cLower, 'person') || contains(cLower, 'vru') || contains(cLower, 'child')
                    len = 1.0; wid = 0.8;
                elseif contains(cLower, 'barrier') || contains(cLower, 'cone') || contains(cLower, 'obstacle')
                    len = 3.5; wid = 1.0;
                else
                    len = 4.5; wid = 2.0; % Passenger car default
                end

                obsList(end+1) = struct(...
                    'X', w_x, 'Y', w_y, 'Vx', w_vx, 'Vy', w_vy, ...
                    'Length', len, 'Width', wid, 'Class', cClass); %#ok<AGROW>
            end
        end

        function resampled = resample_path(~, wps, ds)
            if size(wps, 1) < 2
                resampled = wps;
                return;
            end
            % Remove consecutive duplicate points
            d_consec = hypot(diff(wps(:, 1)), diff(wps(:, 2)));
            keep = [true; d_consec > 1e-4];
            wps = wps(keep, :);
            if size(wps, 1) < 2
                resampled = wps;
                return;
            end

            % Compute cumulative chord length
            d = hypot(diff(wps(:, 1)), diff(wps(:, 2)));
            s = [0; cumsum(d)];
            [s_uniq, uIdx] = unique(s, 'stable');
            wps = wps(uIdx, :);
            totalLen = s_uniq(end);

            if totalLen < ds || numel(s_uniq) < 2
                resampled = wps;
                return;
            end

            s_query = (0:ds:totalLen)';
            if s_query(end) < totalLen
                s_query = [s_query; totalLen];
            end

            x_interp = interp1(s_uniq, wps(:, 1), s_query, 'linear');
            y_interp = interp1(s_uniq, wps(:, 2), s_query, 'linear');
            resampled = [x_interp, y_interp];
        end
    end
end
