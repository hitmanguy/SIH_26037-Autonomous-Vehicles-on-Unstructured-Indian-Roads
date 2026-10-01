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
        RoadBounds          = [-6.50, -0.40]  % Allowed lateral envelope on Ego's own side [m]
        CruisingLaneY       = -1.75           % Lane -1 center (m)
        PassingLaneY        = -5.00           % Lane -2 center (m)
        CenterDividerY      = -0.40           % Critical boundary: Y > -0.40 enters oncoming traffic!
        OuterShoulderY      = -6.50           % Road edge curb (m)
        LaneWidth           = 3.50            % Standard lane width (m)
        
        % Vehicle Physical Dimensions
        Wheelbase           = 2.80            % (m)
        Length              = 4.50            % (m)
        Width               = 2.00            % (m)
        
        % Planning Horizons & Sampling (Agile, crisp turns without lazy wide arcs)
        Horizons            = [1.2, 1.8, 2.5] % Candidate trajectory durations T [s]
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
    end

    methods
        function obj = dynamic_trajectory_planner(cruiseSpeed, roadBounds)
            if nargin >= 1 && ~isempty(cruiseSpeed)
                obj.TargetSpeed = cruiseSpeed;
            end
            if nargin >= 2 && ~isempty(roadBounds)
                obj.RoadBounds = roadBounds;
                if roadBounds(1) < 0 && roadBounds(2) > 0
                    % Bidirectional road: negative Y is Ego's travel direction; positive Y is oncoming traffic!
                    % Clamp CenterDividerY strictly below 0 to never swerve into opposing traffic
                    obj.CenterDividerY = -0.40;
                    obj.OuterShoulderY = min(-6.50, roadBounds(1));
                else
                    obj.CenterDividerY = roadBounds(2);
                    obj.OuterShoulderY = roadBounds(1);
                end
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
        end

        function [wps, planInfo] = replan(obj, currentState, globalWaypoints, fusedTracks, currentSteer, simTime, predictions, potholes, stateflowDecision)
            % REPLAN Generates a dynamically optimized, collision-free C^2 trajectory
            %
            % Inputs:
            %   currentState      - [X, Y, Yaw (rad), Vx (m/s)]
            %   globalWaypoints   - [N x 2] matrix of route reference points
            %   fusedTracks       - Array/struct of perceived tracks from sensor fusion
            %   currentSteer      - Current front wheel angle delta (rad)
            %   simTime           - Current simulation timestamp (s)
            %   predictions       - (Optional) Multi-modal GMM trajectories from trajectory_prediction_engine
            %   potholes          - (Optional) [N x 5] road defect matrix [X, Y, depth_cm, radius_m, severity]
            %   stateflowDecision - (Optional) Supervisory tactical mode struct from decision_supervisor / bridge
            
            x0   = currentState(1);
            y0   = currentState(2);
            psi0 = currentState(3);
            v0   = max(0.1, currentState(4));

            if nargin < 5 || isempty(currentSteer), currentSteer = 0.0; end
            if nargin < 6 || isempty(simTime), simTime = 0.0; end
            if nargin < 7 || isempty(predictions), predictions = []; end
            if nargin < 8 || isempty(potholes), potholes = []; end
            if nargin < 9 || isempty(stateflowDecision), stateflowDecision = []; end

            % Rate limit replanning to 10 Hz to prevent high-frequency chattering
            if obj.LastPlanTime >= 0 && (simTime - obj.LastPlanTime) < (obj.ReplanInterval - 1e-4) && ~isempty(obj.LastPlannedWps)
                wps = obj.LastPlannedWps;
                planInfo = struct('SelectedTargetY', obj.LastTargetY, 'IsEvasive', obj.LaneCommitted, ...
                    'LaneCommitted', obj.LaneCommitted, 'TargetSpeed', obj.TargetSpeed, ...
                    'IsEmergencyBraking', false, 'BestCost', 0.0);
                return;
            end
            obj.LastPlanTime = simTime;

            % Extract nominal route Y from global reference waypoints
            nominalY = obj.CruisingLaneY;
            if ~isempty(globalWaypoints) && size(globalWaypoints, 1) >= 2
                nearCount = min(5, size(globalWaypoints, 1));
                nominalY = mean(globalWaypoints(1:nearCount, 2));
            end

            % Evaluate Stateflow Supervisory Directives
            effectiveTargetSpeed = obj.TargetSpeed;
            if ~isempty(stateflowDecision)
                if isfield(stateflowDecision, 'mode_name')
                    mName = upper(char(stateflowDecision.mode_name));
                    if contains(mName, 'STOP')
                        effectiveTargetSpeed = 0.0;
                    elseif contains(mName, 'SLOW')
                        effectiveTargetSpeed = min(effectiveTargetSpeed, 5.56); % 20 km/h
                    elseif contains(mName, 'YIELD')
                        effectiveTargetSpeed = min(effectiveTargetSpeed, 2.78); % 10 km/h
                    elseif contains(mName, 'REROUTE')
                        obj.LaneCommitted = true;
                        obj.CommittedY = obj.PassingLaneY;
                    end
                elseif isfield(stateflowDecision, 'mode_id')
                    mid = stateflowDecision.mode_id;
                    if mid == 4 || mid == 3
                        effectiveTargetSpeed = 0.0;
                    elseif mid == 1 || mid == 2
                        effectiveTargetSpeed = min(effectiveTargetSpeed, 5.56);
                    end
                end
                if isfield(stateflowDecision, 'target_speed_factor') && stateflowDecision.target_speed_factor < 1.0
                    effectiveTargetSpeed = effectiveTargetSpeed * stateflowDecision.target_speed_factor;
                end
            end

            % 1. Extract Obstacles in Vehicle Vicinity (Forward corridor X in [x0 - 2, x0 + 60])
            parsedObstacles = obj.extract_obstacles(fusedTracks, x0, y0, psi0, v0, predictions);

            % Ingest Road Surface Potholes (LiDAR/Camera ground curvature registry)
            if ~isempty(potholes) && size(potholes, 1) > 0
                for p = 1:size(potholes, 1)
                    p_val1 = potholes(p, 1);
                    p_val2 = potholes(p, 2);
                    p_depth = potholes(p, 3);
                    p_rad = potholes(p, 4);

                    % Determine coordinate frame: BEV [X_lat, Z_long] vs World [X_world, Y_world]
                    if abs(p_val1) < 6.0 && p_val2 > 8.0 && p_val2 > abs(p_val1)
                        p_wx = x0 + p_val2 * cos(psi0) + p_val1 * sin(psi0);
                        p_wy = y0 + p_val2 * sin(psi0) - p_val1 * cos(psi0);
                    else
                        p_wx = p_val1;
                        p_wy = p_val2;
                    end

                    dx_p = p_wx - x0;
                    if p_depth < 5.0 && dx_p > 0 && dx_p < 25.0 && abs(p_wy - y0) < 1.5
                        effectiveTargetSpeed = min(effectiveTargetSpeed, 4.17); % 15 km/h dip traverse
                    end

                    if p_depth >= 5.0
                        pObs = struct(...
                            'X', p_wx, 'Y', p_wy, 'Vx', 0.0, 'Vy', 0.0, ...
                            'Length', max(1.8, 2.0 * p_rad), 'Width', max(1.8, 2.0 * p_rad), ...
                            'Class', 'pothole_cavity', 'PredModes', []);
                        parsedObstacles(end+1) = pObs; %#ok<AGROW>
                    end
                end
            end

            % 2. Comprehensive Lane Status & Clearance Evaluation
            % An obstacle blocks a corridor if Ego is approaching it OR currently alongside it!
            lane1Blocked = false;
            lane2Blocked = false;
            critObstacleX = Inf;
            critObstacleDist = Inf;

            for i = 1:numel(parsedObstacles)
                obs = parsedObstacles(i);
                obsFrontX = obs.X + obs.Length/2;
                obsRearX  = obs.X - obs.Length/2;

                % Check if obstacle is ahead or currently alongside Ego
                if obsFrontX > (x0 - 2.0) && obsRearX < (x0 + 50.0)
                    distToObs = obsRearX - (x0 + obj.Length/2);

                    % Lane 1 blockage check (nominal cruising corridor)
                    obsLatDistLane1 = abs(obs.Y - nominalY);
                    corridorHalfWidth1 = min(1.40, (obj.Width + obs.Width)/2 + 0.35);
                    if obsLatDistLane1 < corridorHalfWidth1
                        lane1Blocked = true;
                        if distToObs < critObstacleDist
                            critObstacleDist = distToObs;
                            critObstacleX = obs.X;
                        end
                    end

                    % Lane 2 blockage check (outer passing/detour corridor)
                    % Only blocks Lane 2 if obstacle actually occupies Lane 2 and is within active horizon
                    obsLatDistLane2 = abs(obs.Y - obj.PassingLaneY);
                    corridorHalfWidth2 = min(1.40, (obj.Width + obs.Width)/2 + 0.35);
                    if obsLatDistLane2 < corridorHalfWidth2 && distToObs < 30.0
                        lane2Blocked = true;
                    end
                end
            end

            % Update Lane Commitment State Machine
            if lane1Blocked && ~lane2Blocked
                % Clear evasion path in Lane -2: Commit and hold until safely past obstacle
                obj.LaneCommitted = true;
                obj.CommittedY = obj.PassingLaneY;
                obj.CommitmentHoldUntilX = critObstacleX + 18.0; % Hold until 18m past hazard
            elseif obj.LaneCommitted
                % Hold commitment until Ego has fully cleared the obstacle and downstream route requests nominal
                downstreamWantsLane1 = false;
                if ~isempty(globalWaypoints)
                    futureIdx = find(globalWaypoints(:, 1) > x0 + 15.0, 1);
                    if ~isempty(futureIdx) && abs(globalWaypoints(futureIdx, 2) - nominalY) < 0.5
                        downstreamWantsLane1 = true;
                    end
                end

                if x0 >= obj.CommitmentHoldUntilX && ~lane1Blocked && downstreamWantsLane1
                    obj.LaneCommitted = false;
                    obj.CommittedY = nominalY;
                else
                    obj.LaneCommitted = true;
                    obj.CommittedY = obj.PassingLaneY;
                end
            else
                obj.CommittedY = nominalY;
            end

            % 3. Generate Candidate Lateral Targets (Indian Road Own-Side Corridor)
            % Strictly enforce: Never plan targets that cross centerline into oncoming traffic (Y > -0.40m)
            d0 = y0;
            dot_d0 = v0 * sin(psi0);
            ddot_d0 = (v0^2 / obj.Wheelbase) * tan(currentSteer);
            ddot_d0 = max(-2.5, min(2.5, ddot_d0));

            candTargets = [];
            if obj.LaneCommitted
                candTargets = [obj.PassingLaneY, obj.PassingLaneY - 0.40, obj.PassingLaneY + 0.50, -3.50, -4.20, nominalY];
                preferredTargetY = obj.PassingLaneY;
            elseif lane1Blocked && ~lane2Blocked
                candTargets = [obj.PassingLaneY, obj.PassingLaneY - 0.40, obj.PassingLaneY + 0.50, -4.20, -3.50];
                preferredTargetY = obj.PassingLaneY;
            elseif lane1Blocked && lane2Blocked
                % Both corridors blocked (e.g. crossing VRU with shoulder parked cars, or wrong-way vehicle)
                % Maintain current lane alignment and prepare emergency braking
                candTargets = [d0, nominalY, obj.PassingLaneY];
                preferredTargetY = d0;
            else
                % Road clear: follow nominal cruising route with minor dynamic nudges
                candTargets = [nominalY, nominalY - 0.35, nominalY + 0.35];
                preferredTargetY = nominalY;
            end

            % Filter candidate targets strictly within drivable road bounds on Ego's own side
            validTargets = [];
            for ti = 1:numel(candTargets)
                ty = candTargets(ti);
                if ty >= (obj.OuterShoulderY + 0.40) && ty <= (obj.CenterDividerY - 0.35)
                    if isempty(validTargets) || ~any(abs(validTargets - ty) < 0.10)
                        validTargets(end+1) = ty; %#ok<AGROW>
                    end
                end
            end
            if isempty(validTargets)
                validTargets = [max(obj.OuterShoulderY + 0.60, min(obj.CenterDividerY - 0.40, d0))];
            end

            % 4. Spatiotemporal Candidate Generation & Cost Optimization (Lattice Search)
            bestCost = Inf;
            bestTrajectory = [];
            bestTargetY = preferredTargetY;
            bestHorizonT = 1.8;

            for tIdx = 1:numel(validTargets)
                d1 = validTargets(tIdx);

                for hIdx = 1:numel(obj.Horizons)
                    T = obj.Horizons(hIdx);

                    poly = obj.solve_quintic(d0, dot_d0, ddot_d0, d1, T);

                    jerkCost = poly.calc_jerk_integral();
                    accCost = abs(d1 - d0) / (T^2);
                    targetCost = (d1 - preferredTargetY)^2;
                    commitCost = (d1 - obj.CommittedY)^2;

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
                        totalCost = totalCost + obj.w_collision;
                    else
                        if minClearance < 2.5
                            totalCost = totalCost + 80.0 / max(0.4, minClearance);
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

            % 5. Impassable Corridor Detection & Emergency Braking
            % Emergency stop (speed = 0) is commanded ONLY if no safe trajectory exists
            % AND obstacle is within critical immediate stopping distance (< 12m or TTC < 2.0s).
            % If obstacle is further ahead (> 12m), perform a controlled detour crawl (14 km/h) to steer around.
            isEmergencyStop = false;
            if isempty(bestTrajectory) || bestCost >= 0.5 * obj.w_collision
                if critObstacleDist < 12.0 || (critObstacleDist / max(0.5, v0)) < 2.0
                    isEmergencyStop = true;
                    effectiveTargetSpeed = 0.0;
                    bestTargetY = max(obj.OuterShoulderY + 0.60, min(obj.CenterDividerY - 0.40, d0));
                    bestHorizonT = 2.0;
                    bestTrajectory = obj.solve_quintic(d0, dot_d0, 0.0, bestTargetY, bestHorizonT);
                else
                    % Advance crawl: maintain 14 km/h pace to allow agile bypass
                    effectiveTargetSpeed = min(effectiveTargetSpeed, 3.89);
                    bestTargetY = obj.PassingLaneY;
                    bestHorizonT = 1.8;
                    bestTrajectory = obj.solve_quintic(d0, dot_d0, 0.0, bestTargetY, bestHorizonT);
                end
            elseif obj.LaneCommitted || (lane1Blocked && ~lane2Blocked)
                % When executing an evasive maneuver, maintain stable turning pace (16 km/h)
                effectiveTargetSpeed = min(effectiveTargetSpeed, 4.44);
            end

            obj.LastTargetY = bestTargetY;

            % 6. Sample Continuous C^2 Waypoints
            if isEmergencyStop
                % Emergency stopping trajectory: decelerate smoothly to standstill before obstacle
                dStopAvail = max(0.5, critObstacleDist - 3.5);
                if isinf(critObstacleDist) || critObstacleDist <= 0
                    dStopAvail = max(1.0, (v0^2) / (2 * 4.5));
                end
                tStop = max(0.5, 2.0 * dStopAvail / max(0.2, v0));
                timeSamples = 0.0:0.05:max(bestHorizonT, tStop);
                N_samples = numel(timeSamples);
                trajX = zeros(N_samples, 1);
                trajY = zeros(N_samples, 1);

                aDecel = min(-1.5, -(v0^2) / (2 * max(0.5, dStopAvail)));
                for k = 1:N_samples
                    tk = timeSamples(k);
                    if tk <= tStop
                        s_prog = v0 * tk + 0.5 * aDecel * (tk^2);
                        trajX(k) = x0 + max(0.0, min(dStopAvail, s_prog));
                    else
                        trajX(k) = x0 + dStopAvail;
                    end
                    trajY(k) = bestTrajectory.calc_pos(min(tk, bestHorizonT));
                end

                % Forward extension holding position at standstill
                extX = (trajX(end)+0.5:obj.ReplanDistance:trajX(end)+25.0)';
                extY = trajY(end) * ones(size(extX));
                fullWps = [trajX, trajY; extX, extY];
            else
                % Normal smooth cruising or detour trajectory
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

                % Enforce strictly that planned trajectory NEVER crosses into oncoming lane
                trajY = max(obj.OuterShoulderY + 0.15, min(obj.CenterDividerY - 0.15, trajY));

                % Stitch with downstream global waypoints
                if ~isempty(globalWaypoints) && size(globalWaypoints, 1) >= 2
                    lastPlanPt = [trajX(end), trajY(end)];
                    distsGlobal = hypot(globalWaypoints(:,1) - lastPlanPt(1), globalWaypoints(:,2) - lastPlanPt(2));
                    [~, matchIdx] = min(distsGlobal);

                    if matchIdx < size(globalWaypoints, 1)
                        remainingGlobal = globalWaypoints(matchIdx+1:end, :);
                        remainingGlobal(:, 2) = trajY(end);
                        fullWps = [trajX, trajY; remainingGlobal];
                    else
                        extX = (trajX(end)+1:obj.ReplanDistance:max(350.0, trajX(end)+150))';
                        extY = trajY(end) * ones(size(extX));
                        fullWps = [trajX, trajY; extX, extY];
                    end
                else
                    extX = (trajX(end)+1:obj.ReplanDistance:max(350.0, trajX(end)+150))';
                    extY = trajY(end) * ones(size(extX));
                    fullWps = [trajX, trajY; extX, extY];
                end
            end

            % Downsample and ensure smooth interpolation (0.5m spacing)
            wps = obj.resample_path(fullWps, obj.ReplanDistance);
            obj.LastPlannedWps = wps;

            % 7. Pack Planning Telemetry
            planInfo = struct();
            planInfo.SelectedTargetY    = bestTargetY;
            planInfo.SelectedHorizonT   = bestHorizonT;
            planInfo.IsEvasive          = obj.LaneCommitted && ~isEmergencyStop;
            planInfo.LaneCommitted      = obj.LaneCommitted && ~isEmergencyStop;
            planInfo.TargetSpeed        = effectiveTargetSpeed;
            planInfo.IsEmergencyBraking = isEmergencyStop;
            planInfo.BestCost           = bestCost;
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

            tCheck = 0.1:0.1:T;

            for k = 1:numel(tCheck)
                t = tCheck(k);
                x_ego = x0 + v0 * t;
                y_ego = poly.calc_pos(t);

                % Hard Road Boundary Check:
                % Oncoming lane violation (crossing centerline into opposite traffic)
                if y_ego > (obj.CenterDividerY - 0.10)
                    boundaryViol = true;
                    return;
                end
                % Off-road curb violation
                if y_ego < (obj.OuterShoulderY + 0.15)
                    boundaryViol = true;
                    return;
                end

                % Dynamic/Static Obstacle Check against forward-projected bounding ellipses
                for oi = 1:numel(obstacles)
                    obs = obstacles(oi);

                    if isfield(obs, 'PredModes') && ~isempty(obs.PredModes)
                        % Evaluate against multi-modal trajectory prediction modes
                        for mi = 1:numel(obs.PredModes)
                            mMode = obs.PredModes(mi);
                            if mMode.prob < 0.10, continue; end

                            % Sample predicted position at time t (prediction dt = 0.1s)
                            stepIdx = max(1, min(numel(mMode.X), round(t / 0.1)));
                            x_obs = mMode.X(stepIdx);
                            y_obs = mMode.Y(stepIdx);

                            % Elliptical safety boundary (Indian urban road lateral margin 0.35m)
                            a_safe = (obj.Length + obs.Length)/2 + 1.2 + 0.15 * v0;
                            b_safe = (obj.Width + obs.Width)/2 + 0.35;

                            distNorm = hypot((x_ego - x_obs)/a_safe, (y_ego - y_obs)/b_safe);
                            physicalDist = hypot(x_ego - x_obs, y_ego - y_obs);

                            if physicalDist < minClearance
                                minClearance = physicalDist;
                            end
                            if distNorm < 1.0
                                isCollision = true;
                            end
                        end
                    else
                        % First-principles kinematic linear forward extrapolation
                        x_obs = obs.X + obs.Vx * t;
                        y_obs = obs.Y + obs.Vy * t;

                        % Elliptical safety boundary (Indian urban road lateral margin 0.35m)
                        a_safe = (obj.Length + obs.Length)/2 + 1.2 + 0.15 * v0; % Longitudinal clearance
                        b_safe = (obj.Width + obs.Width)/2 + 0.35;             % Lateral clearance

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
        end

        function obsList = extract_obstacles(obj, fusedTracks, x0, y0, psi0, v0, predictions)
            obsList = struct('X', {}, 'Y', {}, 'Vx', {}, 'Vy', {}, 'Length', {}, 'Width', {}, 'Class', {}, 'PredModes', {});
            if isempty(fusedTracks), return; end
            if nargin < 7, predictions = []; end

            cosP = cos(psi0);
            sinP = sin(psi0);

            for i = 1:numel(fusedTracks)
                trk = fusedTracks(i);
                trkId = -1;
                if isfield(trk, 'ID'), trkId = trk.ID;
                elseif isfield(trk, 'id'), trkId = trk.id; end

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
                elseif contains(cLower, 'cattle') || contains(cLower, 'cow') || contains(cLower, 'animal')
                    len = 2.2; wid = 1.2;
                elseif contains(cLower, 'barrier') || contains(cLower, 'cone') || contains(cLower, 'obstacle')
                    len = 3.5; wid = 1.0;
                else
                    len = 4.5; wid = 2.0; % Passenger car default
                end

                % Ingest Multi-Modal GMM Trajectories if available from Trajectory layer
                predModes = [];
                if ~isempty(predictions)
                    for pi = 1:numel(predictions)
                        pTrk = predictions(pi);
                        pId = -1;
                        if isfield(pTrk, 'track_id'), pId = pTrk.track_id;
                        elseif isfield(pTrk, 'id'), pId = pTrk.id; end

                        if (trkId >= 0 && pId == trkId) || (trkId < 0 && pi == i)
                            if isfield(pTrk, 'modes') && ~isempty(pTrk.modes)
                                for m = 1:numel(pTrk.modes)
                                    mStruct = pTrk.modes(m);
                                    % Modes are in BEV coordinates [X (lat right), Z (long fwd)]
                                    z_bev = mStruct.Z;
                                    x_bev = mStruct.X;
                                    % Transform mode trajectory to world coordinates
                                    w_mx = x0 + z_bev * cosP + x_bev * sinP;
                                    w_my = y0 + z_bev * sinP - x_bev * cosP;

                                    mEntry = struct('prob', mStruct.prob, ...
                                        'X', w_mx, 'Y', w_my, 'mode_name', mStruct.mode_name);
                                    if isempty(predModes)
                                        predModes = mEntry;
                                    else
                                        predModes(end+1) = mEntry; %#ok<AGROW>
                                    end
                                end
                            end
                            break;
                        end
                    end
                end

                obsList(end+1) = struct(...
                    'X', w_x, 'Y', w_y, 'Vx', w_vx, 'Vy', w_vy, ...
                    'Length', len, 'Width', wid, 'Class', cClass, 'PredModes', predModes); %#ok<AGROW>
            end
        end

        function resampled = resample_path(~, wps, ds)
            if size(wps, 1) < 2
                resampled = wps;
                return;
            end
            % Compute cumulative chord length
            d = hypot(diff(wps(:, 1)), diff(wps(:, 2)));
            s = [0; cumsum(d)];
            totalLen = s(end);

            if totalLen < ds
                resampled = wps;
                return;
            end

            s_query = (0:ds:totalLen)';
            if s_query(end) < totalLen
                s_query = [s_query; totalLen];
            end

            x_interp = interp1(s, wps(:, 1), s_query, 'linear');
            y_interp = interp1(s, wps(:, 2), s_query, 'linear');
            resampled = [x_interp, y_interp];
        end
    end
end
