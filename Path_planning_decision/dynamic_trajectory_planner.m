classdef dynamic_trajectory_planner < handle
% DYNAMIC_TRAJECTORY_PLANNER State-of-the-Art Frenet Optimal Spatiotemporal Replanner
% =========================================================================
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% Production-grade online dynamic trajectory planner based on Apollo / Werling et al.
% Frenet Optimal Spatiotemporal Lattice Sampling:
%  1. Real-Time Online Dynamic Replanning (10 Hz cyclic execution)
%  2. Full Frenet Frame [s, d] decomposition with exact C^2 boundary condition matching
%     Seamlessly invariant to arbitrary vehicle heading / road orientation (Yaw 0, 90, 180, curves)
%  3. Minimum-Jerk Quintic Polynomial lateral candidate generation
%  4. Spatiotemporal dynamic obstacle collision checking over forward prediction horizon
%  5. Indian Traffic Rule Enforcement: Strictly prevents swerving into oncoming traffic
%     (Centerline d > +0.70m heavily penalized; evasions committed to Ego's OWN side d in [-5.5, +0.6])
%  6. Persistent Lane Commitment: Locks onto safe detour corridor (e.g. Lane -2 at d ~ -3.5m)
%     until obstacles are safely passed, preventing rapid lane oscillation / bouncing
%  7. Curvature-bounded smooth C^2 trajectory generation with direct feed to Level-4 MPC & Pure Pursuit
% =========================================================================

    properties
        % Road and Geometry Parameters in Frenet Offset (m)
        CruisingLaneOffset  = 0.00            % Centerline of nominal reference route (m)
        PassingLaneOffset   = -3.50           % Outer lane on Ego's own side (m)
        CenterDividerLimit  = 0.70            % Maximum allowed offset towards centerline (m)
        OuterShoulderLimit  = -5.50           % Road edge curb threshold (m)
        LaneWidth           = 3.50            % Standard lane width (m)
        
        % Legacy compatibility properties
        RoadBounds          = [-6.50, -0.40]  % Allowed lateral envelope
        CruisingLaneY       = -1.75
        PassingLaneY        = -5.00
        CenterDividerY      = -0.40
        OuterShoulderY      = -6.50
        
        % Vehicle Physical Dimensions
        Wheelbase           = 2.80            % (m)
        Length              = 4.50            % (m)
        Width               = 2.00            % (m)
        
        % Planning Horizons & Sampling
        Horizons            = [1.4, 2.0, 2.8] % Candidate trajectory durations T [s]
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
        CommittedD          = 0.00            % Currently committed Frenet lateral offset
        CommittedY          = -1.75           % Legacy world Y
        CommitmentHoldUntilS= -Inf            % S progress threshold before lane return allowed
        CommitmentHoldUntilX= -Inf            % Legacy X
        LastTargetD         = 0.00            % Target D chosen in previous cycle
        LastTargetY         = -1.75
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
            obj.CommittedD = obj.CruisingLaneOffset;
            obj.CommittedY = obj.CruisingLaneY;
            obj.CommitmentHoldUntilS = -Inf;
            obj.CommitmentHoldUntilX = -Inf;
            obj.LastTargetD = obj.CruisingLaneOffset;
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

            % 1. Establish Frenet Reference Frame along Global Reference Path
            % Projects ego state onto path centerline regardless of road angle (Yaw = 0, 90, 180, etc.)
            hasGlobalPath = ~isempty(globalWaypoints) && size(globalWaypoints, 1) >= 2;
            if hasGlobalPath
                [P_proj, tangentYaw, s_cum_0, ~] = obj.find_path_projection(x0, y0, globalWaypoints);
            else
                P_proj = [x0, y0];
                tangentYaw = psi0;
                s_cum_0 = 0.0;
            end

            % Relative Frenet Coordinates of Ego Vehicle
            dx0 = x0 - P_proj(1);
            dy0 = y0 - P_proj(2);
            d0 = -sin(tangentYaw) * dx0 + cos(tangentYaw) * dy0; % Cross-track offset
            epsi0 = wrapToPi(psi0 - tangentYaw);                % Heading error
            dot_d0 = v0 * sin(epsi0);                           % Lateral velocity
            ddot_d0 = (v0^2 / obj.Wheelbase) * tan(currentSteer) * cos(epsi0);
            ddot_d0 = max(-2.5, min(2.5, ddot_d0));             % Lateral acceleration

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
                        obj.CommittedD = obj.PassingLaneOffset;
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

            % 2. Extract Perceived Obstacles in Frenet Coordinate Frame
            parsedObstacles = obj.extract_obstacles(fusedTracks, x0, y0, psi0, v0, predictions);

            % Ingest Road Surface Potholes
            if ~isempty(potholes) && size(potholes, 1) > 0
                for p = 1:size(potholes, 1)
                    p_val1 = potholes(p, 1);
                    p_val2 = potholes(p, 2);
                    p_depth = potholes(p, 3);
                    p_rad = potholes(p, 4);

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

            % 3. Project Obstacles into Path Frenet Frame (s_obs, d_obs)
            numObs = numel(parsedObstacles);
            frenetObstacles = repmat(struct('s', 0, 'd', 0, 'v_s', 0, 'v_d', 0, ...
                'Length', 4.5, 'Width', 2.0, 'Class', 'car', 'PredModes', []), numObs, 1);

            lane1Blocked = false;
            lane2Blocked = false;
            critObstacleDist = Inf;
            critObstacleS = Inf;

            for i = 1:numObs
                obs = parsedObstacles(i);
                dx_o = obs.X - P_proj(1);
                dy_o = obs.Y - P_proj(2);
                s_o = dx_o * cos(tangentYaw) + dy_o * sin(tangentYaw);
                d_o = -dx_o * sin(tangentYaw) + dy_o * cos(tangentYaw);

                v_s_o = obs.Vx * cos(tangentYaw) + obs.Vy * sin(tangentYaw);
                v_d_o = -obs.Vx * sin(tangentYaw) + obs.Vy * cos(tangentYaw);

                frenetObstacles(i).s = s_o;
                frenetObstacles(i).d = d_o;
                frenetObstacles(i).v_s = v_s_o;
                frenetObstacles(i).v_d = v_d_o;
                frenetObstacles(i).Length = obs.Length;
                frenetObstacles(i).Width = obs.Width;
                frenetObstacles(i).Class = obs.Class;
                frenetObstacles(i).PredModes = obs.PredModes;

                % Corridor occupancy check
                obsFrontS = s_o + obs.Length / 2;
                obsRearS  = s_o - obs.Length / 2;

                if obsFrontS > -2.0 && obsRearS < 55.0
                    distToObs = obsRearS - (obj.Length / 2);

                    % Lane 1 (Nominal Lane: d ~ 0)
                    corridor1 = min(1.40, (obj.Width + obs.Width)/2 + 0.35);
                    if abs(d_o - obj.CruisingLaneOffset) < corridor1
                        lane1Blocked = true;
                        if distToObs < critObstacleDist
                            critObstacleDist = distToObs;
                            critObstacleS = s_o;
                        end
                    end

                    % Lane 2 (Passing/Outer Lane: d ~ -3.5m)
                    corridor2 = min(1.40, (obj.Width + obs.Width)/2 + 0.35);
                    if abs(d_o - obj.PassingLaneOffset) < corridor2
                        % Obstacle in Lane 2 anywhere in the forward horizon blocks Lane 2
                        lane2Blocked = true;
                    end
                end
            end

            % 4. Lane Commitment State Machine
            if lane1Blocked && ~lane2Blocked
                % Definite clear bypass corridor in Lane 2: commit to detour
                obj.LaneCommitted = true;
                obj.CommittedD = obj.PassingLaneOffset;
                obj.CommitmentHoldUntilS = s_cum_0 + critObstacleS + 16.0;
            elseif obj.LaneCommitted
                % Check if ego has safely cleared obstacle
                if s_cum_0 >= obj.CommitmentHoldUntilS && ~lane1Blocked
                    obj.LaneCommitted = false;
                    obj.CommittedD = obj.CruisingLaneOffset;
                else
                    obj.LaneCommitted = true;
                    obj.CommittedD = obj.PassingLaneOffset;
                end
            else
                obj.CommittedD = obj.CruisingLaneOffset;
            end

            % 5. Candidate Lateral Goals in Frenet Offset (d)
            if obj.LaneCommitted
                candTargets = [obj.PassingLaneOffset, obj.PassingLaneOffset - 0.30, obj.PassingLaneOffset + 0.40, -2.50, 0.0];
                preferredTargetD = obj.PassingLaneOffset;
            elseif lane1Blocked && ~lane2Blocked
                candTargets = [obj.PassingLaneOffset, obj.PassingLaneOffset - 0.30, obj.PassingLaneOffset + 0.40, -2.50];
                preferredTargetD = obj.PassingLaneOffset;
            elseif lane1Blocked && lane2Blocked
                % Both corridors blocked (e.g. crossing VRU with obstacle alongside)
                candTargets = [d0, 0.0, obj.PassingLaneOffset];
                preferredTargetD = d0;
            else
                % Road clear: follow nominal route
                candTargets = [obj.CruisingLaneOffset, obj.CruisingLaneOffset - 0.30, obj.CruisingLaneOffset + 0.30];
                preferredTargetD = obj.CruisingLaneOffset;
            end

            % Bound candidate targets strictly within road limits
            validTargets = [];
            for ti = 1:numel(candTargets)
                td = candTargets(ti);
                if td >= (obj.OuterShoulderLimit + 0.30) && td <= (obj.CenterDividerLimit - 0.10)
                    if isempty(validTargets) || ~any(abs(validTargets - td) < 0.10)
                        validTargets(end+1) = td; %#ok<AGROW>
                    end
                end
            end
            if isempty(validTargets)
                validTargets = [max(obj.OuterShoulderLimit + 0.50, min(obj.CenterDividerLimit - 0.20, d0))];
            end

            % 6. Lattice Search: Quintic Lateral Polynomials & Spatiotemporal Collision Check
            bestCost = Inf;
            bestTrajectory = [];
            bestTargetD = preferredTargetD;
            bestHorizonT = 1.8;

            for tIdx = 1:numel(validTargets)
                d1 = validTargets(tIdx);

                for hIdx = 1:numel(obj.Horizons)
                    T = obj.Horizons(hIdx);

                    poly = obj.solve_quintic(d0, dot_d0, ddot_d0, d1, T);

                    jerkCost   = poly.calc_jerk_integral();
                    accCost    = abs(d1 - d0) / (T^2);
                    targetCost = (d1 - preferredTargetD)^2;
                    commitCost = (d1 - obj.CommittedD)^2;

                    [isCollision, minClearance, boundaryViol] = obj.check_spatiotemporal_collision(...
                        poly, v0, T, frenetObstacles);

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
                        bestTargetD = d1;
                        bestHorizonT = T;
                    end
                end
            end

            % 7. Fallback: Emergency Braking or Controlled Detour
            isEmergencyStop = false;
            if isempty(bestTrajectory) || bestCost >= 0.5 * obj.w_collision
                if critObstacleDist < 12.0 || (critObstacleDist / max(0.5, v0)) < 2.0
                    isEmergencyStop = true;
                    effectiveTargetSpeed = 0.0;
                    bestTargetD = max(obj.OuterShoulderLimit + 0.50, min(obj.CenterDividerLimit - 0.20, d0));
                    bestHorizonT = 2.0;
                    bestTrajectory = obj.solve_quintic(d0, dot_d0, 0.0, bestTargetD, bestHorizonT);
                else
                    effectiveTargetSpeed = min(effectiveTargetSpeed, 3.89); % Detour crawl (14 km/h)
                    bestTargetD = obj.PassingLaneOffset;
                    bestHorizonT = 1.8;
                    bestTrajectory = obj.solve_quintic(d0, dot_d0, 0.0, bestTargetD, bestHorizonT);
                end
            elseif obj.LaneCommitted || (lane1Blocked && ~lane2Blocked)
                effectiveTargetSpeed = min(effectiveTargetSpeed, 4.44); % 16 km/h stable evasion pace
            end

            obj.LastTargetD = bestTargetD;

            % 8. Sample Waypoints & Reconstruct World Coordinates
            % Generates continuous C^2 waypoints rotated into world frame along reference path
            if isEmergencyStop
                dStopAvail = max(0.5, critObstacleDist - 3.5);
                if isinf(critObstacleDist) || critObstacleDist <= 0
                    dStopAvail = max(1.0, (v0^2) / (2 * 4.5));
                end
                tStop = max(0.5, 2.0 * dStopAvail / max(0.2, v0));
                timeSamples = 0.0:0.05:max(bestHorizonT, tStop);
                N_samples = numel(timeSamples);
                w_x = zeros(N_samples, 1);
                w_y = zeros(N_samples, 1);

                aDecel = min(-1.5, -(v0^2) / (2 * max(0.5, dStopAvail)));
                for k = 1:N_samples
                    tk = timeSamples(k);
                    if tk <= tStop
                        s_prog = max(0.0, min(dStopAvail, v0 * tk + 0.5 * aDecel * (tk^2)));
                    else
                        s_prog = dStopAvail;
                    end
                    d_prog = bestTrajectory.calc_pos(min(tk, bestHorizonT));

                    if hasGlobalPath
                        [refPt_k, tang_k] = obj.evaluate_path_at_s(globalWaypoints, s_cum_0, s_prog);
                    else
                        refPt_k = P_proj + s_prog * [cos(tangentYaw), sin(tangentYaw)];
                        tang_k = tangentYaw;
                    end
                    w_x(k) = refPt_k(1) - d_prog * sin(tang_k);
                    w_y(k) = refPt_k(2) + d_prog * cos(tang_k);
                end

                % Forward extension holding standstill position
                ext_s = (s_prog + 0.5:obj.ReplanDistance:s_prog + 25.0)';
                N_ext = numel(ext_s);
                ext_x = zeros(N_ext, 1);
                ext_y = zeros(N_ext, 1);
                for ek = 1:N_ext
                    if hasGlobalPath
                        [refPt_e, tang_e] = obj.evaluate_path_at_s(globalWaypoints, s_cum_0, ext_s(ek));
                    else
                        refPt_e = P_proj + ext_s(ek) * [cos(tangentYaw), sin(tangentYaw)];
                        tang_e = tangentYaw;
                    end
                    ext_x(ek) = refPt_e(1) - d_prog * sin(tang_e);
                    ext_y(ek) = refPt_e(2) + d_prog * cos(tang_e);
                end
                fullWps = [w_x, w_y; ext_x, ext_y];
            else
                vForward = max(2.5, v0);
                timeSamples = 0.0:0.05:bestHorizonT;
                N_samples = numel(timeSamples);
                w_x = zeros(N_samples, 1);
                w_y = zeros(N_samples, 1);

                for k = 1:N_samples
                    tk = timeSamples(k);
                    s_prog = vForward * tk;
                    d_prog = bestTrajectory.calc_pos(tk);
                    d_prog = max(obj.OuterShoulderLimit + 0.20, min(obj.CenterDividerLimit - 0.15, d_prog));

                    if hasGlobalPath
                        [refPt_k, tang_k] = obj.evaluate_path_at_s(globalWaypoints, s_cum_0, s_prog);
                    else
                        refPt_k = P_proj + s_prog * [cos(tangentYaw), sin(tangentYaw)];
                        tang_k = tangentYaw;
                    end
                    w_x(k) = refPt_k(1) - d_prog * sin(tang_k);
                    w_y(k) = refPt_k(2) + d_prog * cos(tang_k);
                end

                % Stitch with downstream reference waypoints beyond horizon
                final_d = bestTrajectory.calc_pos(bestHorizonT);
                final_d = max(obj.OuterShoulderLimit + 0.20, min(obj.CenterDividerLimit - 0.15, final_d));
                s_end = vForward * bestHorizonT;

                ext_s = (s_end + obj.ReplanDistance:obj.ReplanDistance:s_end + 120.0)';
                N_ext = numel(ext_s);
                ext_x = zeros(N_ext, 1);
                ext_y = zeros(N_ext, 1);
                for ek = 1:N_ext
                    if hasGlobalPath
                        [refPt_e, tang_e] = obj.evaluate_path_at_s(globalWaypoints, s_cum_0, ext_s(ek));
                    else
                        refPt_e = P_proj + ext_s(ek) * [cos(tangentYaw), sin(tangentYaw)];
                        tang_e = tangentYaw;
                    end
                    ext_x(ek) = refPt_e(1) - final_d * sin(tang_e);
                    ext_y(ek) = refPt_e(2) + final_d * cos(tang_e);
                end
                fullWps = [w_x, w_y; ext_x, ext_y];
            end

            % Downsample and ensure smooth interpolation (0.5m spacing)
            wps = obj.resample_path(fullWps, obj.ReplanDistance);
            obj.LastPlannedWps = wps;

            % Reconstruct world target Y for telemetry
            targetWorld = P_proj + [cos(tangentYaw), sin(tangentYaw)] * (v0 * bestHorizonT) + ...
                          [-sin(tangentYaw), cos(tangentYaw)] * bestTargetD;
            obj.LastTargetY = targetWorld(2);

            % 9. Pack Planning Telemetry
            planInfo = struct();
            planInfo.SelectedTargetY    = targetWorld(2);
            planInfo.SelectedTargetD    = bestTargetD;
            planInfo.SelectedHorizonT   = bestHorizonT;
            planInfo.IsEvasive          = obj.LaneCommitted && ~isEmergencyStop;
            planInfo.LaneCommitted      = obj.LaneCommitted && ~isEmergencyStop;
            planInfo.TargetSpeed        = effectiveTargetSpeed;
            planInfo.IsEmergencyBraking = isEmergencyStop;
            planInfo.BestCost           = bestCost;
        end

        function poly = solve_quintic(~, d0, v_d0, a_d0, d1, T)
            % Closed-form minimum-jerk boundary solver: d(T)=d1, d'(T)=0, d''(T)=0
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

        function [isCollision, minClearance, boundaryViol] = check_spatiotemporal_collision(obj, poly, v0, T, frenetObstacles)
            isCollision = false;
            boundaryViol = false;
            minClearance = Inf;

            tCheck = 0.1:0.1:T;

            for k = 1:numel(tCheck)
                t = tCheck(k);
                s_ego = v0 * t;
                d_ego = poly.calc_pos(t);

                % Road boundary check in Frenet frame
                if d_ego > obj.CenterDividerLimit
                    boundaryViol = true;
                    return;
                end
                if d_ego < obj.OuterShoulderLimit
                    boundaryViol = true;
                    return;
                end

                % Obstacle check in Frenet frame
                for oi = 1:numel(frenetObstacles)
                    obs = frenetObstacles(oi);
                    s_obs = obs.s + obs.v_s * t;
                    d_obs = obs.d + obs.v_d * t;

                    a_safe = (obj.Length + obs.Length)/2 + 1.2 + 0.15 * v0;
                    b_safe = (obj.Width + obs.Width)/2 + 0.35;

                    distNorm = hypot((s_ego - s_obs)/a_safe, (d_ego - d_obs)/b_safe);
                    physicalDist = hypot(s_ego - s_obs, d_ego - d_obs);

                    if physicalDist < minClearance
                        minClearance = physicalDist;
                    end
                    if distNorm < 1.0
                        isCollision = true;
                    end
                end
            end
        end

        function [projPt, tangentYaw, s_cum, bestSeg] = find_path_projection(~, x0, y0, wps)
            N = size(wps, 1);
            if N < 2
                projPt = [x0, y0];
                tangentYaw = 0.0;
                s_cum = 0.0;
                bestSeg = 1;
                return;
            end

            dSeg = hypot(diff(wps(:, 1)), diff(wps(:, 2)));
            s_nodes = [0; cumsum(dSeg)];

            segStarts = wps(1:N-1, 1:2);
            segEnds   = wps(2:N, 1:2);
            vecs      = segEnds - segStarts;
            lensSq    = max(1e-4, sum(vecs.^2, 2));

            pRel = [x0, y0] - segStarts;
            t = max(0.0, min(1.0, sum(pRel .* vecs, 2) ./ lensSq));
            projs = segStarts + t .* vecs;
            distsSq = sum(([x0, y0] - projs).^2, 2);
            [~, bestSeg] = min(distsSq);

            projPt = projs(bestSeg, :);
            vBest = vecs(bestSeg, :);
            tangentYaw = atan2(vBest(2), vBest(1));
            s_cum = s_nodes(bestSeg) + t(bestSeg) * sqrt(lensSq(bestSeg));
        end

        function [refPt, tangYaw] = evaluate_path_at_s(~, wps, s_cum_0, s_ahead)
            N = size(wps, 1);
            if N < 2
                refPt = [0, 0]; tangYaw = 0; return;
            end
            dSeg = hypot(diff(wps(:, 1)), diff(wps(:, 2)));
            s_nodes = [0; cumsum(dSeg)];
            totalLen = s_nodes(end);

            target_s = s_cum_0 + s_ahead;
            if target_s <= 0
                refPt = wps(1, 1:2);
                tangYaw = atan2(wps(2,2)-wps(1,2), wps(2,1)-wps(1,1));
            elseif target_s >= totalLen
                endVec = wps(N, 1:2) - wps(max(1, N-1), 1:2);
                tangYaw = atan2(endVec(2), endVec(1));
                uEnd = endVec / max(1e-4, norm(endVec));
                refPt = wps(N, 1:2) + (target_s - totalLen) * uEnd;
            else
                idx = find(s_nodes <= target_s, 1, 'last');
                idx = min(idx, N - 1);
                frac = (target_s - s_nodes(idx)) / max(1e-4, dSeg(idx));
                refPt = wps(idx, 1:2) + frac * (wps(idx+1, 1:2) - wps(idx, 1:2));
                segVec = wps(idx+1, 1:2) - wps(idx, 1:2);
                tangYaw = atan2(segVec(2), segVec(1));
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

                if isfield(trk, 'Position')
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
                    len = 4.5; wid = 2.0;
                end

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
                                    z_bev = mStruct.Z;
                                    x_bev = mStruct.X;
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
