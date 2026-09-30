classdef sensor_fusion_bridge < handle
% SENSOR_FUSION_BRIDGE Robust Multi-Sensor Perception & Kalman Fusion World Model
% =========================================================================
% High-Integrity Perception & Multi-Sensor Fusion Pipeline:
% 1. Ingestion: 4 Surround Cameras (360° BEV), 2 Automotive Radars, 1 Roof LiDAR.
% 2. Multi-Sensor Coincidence: Vision (high azimuth) + Radar (high range + Doppler).
% 3. Covariance Intersection: Information matrix Kalman fusion:
%      P_fused = inv(inv(P_cam) + inv(P_rad))
%      pos_fused = P_fused * (inv(P_cam)*z_cam + inv(P_rad)*z_rad)
% 4. M-out-of-N Persistence Filter: 3-of-5 temporal verification prevents phantom braking.
% 5. Two-Tier Braking Authority Gate: Only verified coincident/persisting tracks command AEB.
% 6. Dynamic Swept Path Corridor: Strict in-lane boundary (1.25m) eliminates roadside false alarms.
% =========================================================================

    properties
        SensorRig                           % Handle to sensor_rig_builder
        Tracks = struct([])                 % Array of persistent tracks
        NextTrackID = 1                     % Unique track ID generator
        Ts = 0.05                           % Fusion cycle time (s)
        
        % Filter & Fusion Parameters
        GateThreshold = 3.5                 % Spatial association gate (m)
        MaxMissedSteps = 6                  % Steps before track deletion
        MinConfirmHits = 3                  % M-of-N persistence threshold
        HistoryLength = 5                   % Circular hit history window
        
        % Spatial Corridor Envelopes (Ego Vehicle Width = 1.8m)
        InLaneCorridorHalfWidth = 1.25      % True vehicle swept path + 0.35m safety buffer (m)
        VRUBufferHalfWidth = 2.8            % Encroachment buffer for laterally moving VRUs (m)
        
        % Sensor Measurement Covariances
        R_cam_nominal = diag([0.20^2, 1.20^2])   % High lateral, moderate longitudinal
        R_rad_nominal = diag([1.50^2, 0.30^2])   % Moderate lateral, high longitudinal
        R_lidar_nominal = diag([0.25^2, 0.25^2]) % High spatial accuracy
        Q_process = diag([0.4, 0.4])             % Process acceleration noise
        
        % Latest raw sensor buffers
        LatestCamDets = {}
        LatestRadDets = {}
        LatestPtCloud = []
    end

    methods
        function obj = sensor_fusion_bridge(sensorRig, sampleTime)
            if nargin >= 1, obj.SensorRig = sensorRig; end
            if nargin >= 2, obj.Ts = sampleTime; end
            obj.Tracks = struct(...
                'ID', {}, 'Position', {}, 'Velocity', {}, ...
                'Covariance', {}, 'Class', {}, 'Age', {}, ...
                'Hits', {}, 'Missed', {}, 'HitHistory', {}, ...
                'IsCoincident', {}, 'Confirmed', {}, 'BrakingAuthority', {});
        end

        function [fusedTracks, leadObstacle, cutinHazard] = step(obj, egoActor, simTime)
            % 1. Collect measurements from sensor rig
            if isempty(obj.SensorRig)
                fusedTracks = obj.Tracks;
                leadObstacle = struct('Distance', Inf, 'ClosingVelocity', 0, 'Class', 'none', 'TrackID', -1);
                cutinHazard = struct('Active', false, 'Distance', Inf, 'Class', 'none');
                return;
            end

            [camDets, radDets, ptCloud] = obj.SensorRig.collect_measurements(egoActor, simTime);
            obj.LatestCamDets = camDets;
            obj.LatestRadDets = radDets;
            obj.LatestPtCloud = ptCloud;

            % 2. Extract Cartesian observations in Ego Body Frame
            camObs = obj.extract_camera_observations(camDets);
            radObs = obj.extract_radar_observations(radDets);
            lidarObs = obj.extract_lidar_observations(ptCloud);

            % 3. Cross-Sensor Fusion (Information Matrix Covariance Intersection)
            fusedObs = obj.fuse_cross_sensor_observations(camObs, radObs, lidarObs);

            % 4. Track Association & M-of-N Kalman State Update
            obj.update_tracks(fusedObs);

            % 5. Filter for Confirmed Active Tracks
            if isempty(obj.Tracks)
                fusedTracks = [];
            else
                confMask = [obj.Tracks.Confirmed];
                fusedTracks = obj.Tracks(confMask);
            end

            % 6. Extract Critical In-Path Collision Hazards
            leadObstacle = obj.get_lead_obstacle(obj.Tracks);
            cutinHazard  = obj.get_cutin_hazard(obj.Tracks);
        end

        function camObs = extract_camera_observations(~, camDets)
            camObs = struct('Pos', {}, 'Cov', {}, 'Class', {}, 'SensorIndex', {}, 'Source', {});
            for k = 1:numel(camDets)
                d = camDets{k};
                m = d.Measurement;
                % Coordinates are reported in Ego Body Cartesian [x_long, y_lat, ...]
                if numel(m) >= 2
                    x = m(1);
                    y = m(2);
                    
                    % Adaptive range-dependent longitudinal covariance
                    zCov = max(0.4, (0.07 * max(1.0, abs(x)))^2);
                    latCov = 0.20^2;
                    R = diag([zCov, latCov]);
                    
                    className = 'obstacle';
                    if isfield(d, 'ObjectClassID')
                        switch d.ObjectClassID
                            case 1, className = 'car';
                            case 2, className = 'truck';
                            case 4, className = 'pedestrian';
                            case 6, className = 'autorickshaw';
                            case 7, className = 'motorcycle';
                            case 9, className = 'animal';
                        end
                    end
                    camObs(end+1) = struct('Pos', [x, y], 'Cov', R, 'Class', className, ...
                        'SensorIndex', d.SensorIndex, 'Source', 'Camera'); %#ok<AGROW>
                end
            end
        end

        function radObs = extract_radar_observations(~, radDets)
            radObs = struct('Pos', {}, 'Cov', {}, 'Doppler', {}, 'SensorIndex', {}, 'Source', {});
            for k = 1:numel(radDets)
                d = radDets{k};
                m = d.Measurement;
                if numel(m) >= 2
                    x = m(1);
                    y = m(2);
                    vDoppler = 0.0;
                    if numel(m) >= 4, vDoppler = m(4); end
                    
                    % Radar has high longitudinal range accuracy, wider lateral azimuth uncertainty
                    zCov = 0.30^2;
                    latCov = max(0.4, (0.06 * max(1.0, abs(x)))^2);
                    R = diag([zCov, latCov]);
                    
                    radObs(end+1) = struct('Pos', [x, y], 'Cov', R, 'Doppler', vDoppler, ...
                        'SensorIndex', d.SensorIndex, 'Source', 'Radar'); %#ok<AGROW>
                end
            end
        end

        function lidarObs = extract_lidar_observations(~, ptCloud)
            lidarObs = struct('Pos', {}, 'Cov', {}, 'Class', {});
            if isempty(ptCloud) || ~isprop(ptCloud, 'Location'), return; end
            pts = ptCloud.Location;
            if isempty(pts), return; end

            xPts = pts(:, :, 1);
            yPts = pts(:, :, 2);
            zPts = [];
            if size(pts, 3) >= 3
                zPts = pts(:, :, 3);
            end

            % Filter out Ego vehicle bounding envelope and invalid returns
            valid = isfinite(xPts) & isfinite(yPts);
            if ~isempty(zPts)
                % Ground plane segmentation: roof scanner is at H=2.0m.
                % Road surface is at Z ~ -2.0m. Filter points with Z > -1.7m (at least 30cm above ground)
                valid = valid & (zPts > -1.7) & (zPts < 1.2);
            end
            valid = valid & (abs(xPts) > 2.5 | abs(yPts) > 1.1);

            if any(valid(:))
                validX = xPts(valid);
                validY = yPts(valid);
                
                % Focus on forward collision zone (beyond front bumper at 3.7m up to 45m)
                fwd = (validX > 3.7) & (validX < 45.0) & (abs(validY) < 10.0);
                if sum(fwd) >= 3 % Density check: require at least 3 points in cluster
                    fwdX = validX(fwd);
                    fwdY = validY(fwd);
                    
                    % Extract cluster centroid of nearest return
                    nearDists = hypot(fwdX, fwdY);
                    [minD, ~] = min(nearDists);
                    nearMask = nearDists <= (minD + 1.2);
                    
                    meanX = mean(fwdX(nearMask));
                    meanY = mean(fwdY(nearMask));
                    
                    lidarObs(1) = struct('Pos', [meanX, meanY], 'Cov', diag([0.25^2, 0.25^2]), ...
                        'Class', 'lidar_cluster');
                end
            end
        end

        function fusedObs = fuse_cross_sensor_observations(obj, camObs, radObs, lidarObs)
            % Information Matrix Covariance Intersection with Coincidence Flagging
            fusedObs = struct('Pos', {}, 'Cov', {}, 'Velocity', {}, 'Class', {}, 'IsCoincident', {});
            usedRad = false(numel(radObs), 1);

            % 1. Fuse camera detections with matching radar returns
            for c = 1:numel(camObs)
                cPos = camObs(c).Pos;
                cCov = camObs(c).Cov;
                cClass = camObs(c).Class;

                matchedRadIdx = 0;
                minDist = obj.GateThreshold;

                for r = 1:numel(radObs)
                    if ~usedRad(r)
                        d = norm(cPos - radObs(r).Pos);
                        if d < minDist
                            minDist = d;
                            matchedRadIdx = r;
                        end
                    end
                end

                if matchedRadIdx > 0
                    % High-Confidence Information Matrix Fusion:
                    % P_fused = inv(inv(P_cam) + inv(P_rad))
                    rPos = radObs(matchedRadIdx).Pos;
                    rCov = radObs(matchedRadIdx).Cov;
                    W_cam = inv(cCov);
                    W_rad = inv(rCov);
                    P_fused = inv(W_cam + W_rad);
                    pos_fused = (P_fused * (W_cam * cPos(:) + W_rad * rPos(:)))';
                    vel_fused = [radObs(matchedRadIdx).Doppler, 0.0];
                    usedRad(matchedRadIdx) = true;
                    isCoinc = true;
                else
                    % Single sensor camera observation (vision only)
                    pos_fused = cPos;
                    P_fused   = cCov;
                    vel_fused = [0.0, 0.0];
                    isCoinc   = false;
                end

                fusedObs(end+1) = struct('Pos', pos_fused, 'Cov', P_fused, ...
                    'Velocity', vel_fused, 'Class', cClass, 'IsCoincident', isCoinc); %#ok<AGROW>
            end

            % 2. Add remaining radar-only detections
            for r = 1:numel(radObs)
                if ~usedRad(r)
                    fusedObs(end+1) = struct('Pos', radObs(r).Pos, 'Cov', radObs(r).Cov, ...
                        'Velocity', [radObs(r).Doppler, 0.0], 'Class', 'radar_target', 'IsCoincident', false); %#ok<AGROW>
                end
            end

            % 3. Corroborate with LiDAR returns
            for l = 1:numel(lidarObs)
                lPos = lidarObs(l).Pos;
                isMerged = false;
                for f = 1:numel(fusedObs)
                    if norm(fusedObs(f).Pos - lPos) < 2.0
                        % Refine position with high-precision lidar
                        W_f = inv(fusedObs(f).Cov);
                        W_l = inv(lidarObs(l).Cov);
                        P_new = inv(W_f + W_l);
                        fusedObs(f).Pos = (P_new * (W_f * fusedObs(f).Pos(:) + W_l * lPos(:)))';
                        fusedObs(f).Cov = P_new;
                        fusedObs(f).IsCoincident = true; % Cross-corroborated
                        isMerged = true;
                        break;
                    end
                end
                if ~isMerged && norm(lPos) < 30.0
                    fusedObs(end+1) = struct('Pos', lPos, 'Cov', lidarObs(l).Cov, ...
                        'Velocity', [0.0, 0.0], 'Class', 'lidar_obstacle', 'IsCoincident', false); %#ok<AGROW>
                end
            end
        end

        function update_tracks(obj, fusedObs)
            numTracks = numel(obj.Tracks);
            numObs    = numel(fusedObs);
            matchedTrack = false(numTracks, 1);
            matchedObs   = false(numObs, 1);

            % Track-to-Observation Association (Nearest Neighbor Gating)
            for t = 1:numTracks
                predPos = obj.Tracks(t).Position + obj.Tracks(t).Velocity * obj.Ts;
                bestDist = obj.GateThreshold;
                bestObsIdx = 0;

                for o = 1:numObs
                    if ~matchedObs(o)
                        dist = norm(predPos - fusedObs(o).Pos);
                        if dist < bestDist
                            bestDist = dist;
                            bestObsIdx = o;
                        end
                    end
                end

                if bestObsIdx > 0
                    % Kalman Filter Measurement Update
                    matchedTrack(t) = true;
                    matchedObs(bestObsIdx) = true;
                    
                    z = fusedObs(bestObsIdx).Pos;
                    R = fusedObs(bestObsIdx).Cov;
                    P_pred = obj.Tracks(t).Covariance + obj.Q_process * obj.Ts;
                    K = P_pred / (P_pred + R);
                    
                    % Position innovation
                    innov = z - predPos;
                    newPos = predPos + (K * innov(:))';
                    newCov = (eye(2) - K) * P_pred;
                    
                    % Update velocity estimate
                    % Kinematic velocity update from spatial displacement
                    inferredVel = (newPos - obj.Tracks(t).Position) / max(0.01, obj.Ts);
                    
                    % Longitudinal velocity: blend Doppler (if valid) with inferred displacement
                    measuredVel = fusedObs(bestObsIdx).Velocity;
                    if abs(measuredVel(1)) > 0.1
                        newVelX = 0.6 * obj.Tracks(t).Velocity(1) + 0.3 * measuredVel(1) + 0.1 * inferredVel(1);
                    else
                        newVelX = 0.7 * obj.Tracks(t).Velocity(1) + 0.3 * inferredVel(1);
                    end
                    
                    % Lateral velocity: strictly from cross-range spatial displacement
                    newVelY = 0.7 * obj.Tracks(t).Velocity(2) + 0.3 * inferredVel(2);
                    newVel = [newVelX, newVelY];

                    obj.Tracks(t).Position = newPos;
                    obj.Tracks(t).Velocity = newVel;
                    obj.Tracks(t).Covariance = newCov;
                    obj.Tracks(t).Class = fusedObs(bestObsIdx).Class;
                    obj.Tracks(t).Hits = obj.Tracks(t).Hits + 1;
                    obj.Tracks(t).Missed = 0;
                    obj.Tracks(t).Age = obj.Tracks(t).Age + 1;
                    
                    % Update 5-frame sliding hit history
                    curHist = obj.Tracks(t).HitHistory;
                    obj.Tracks(t).HitHistory = [true, curHist(1:min(end, obj.HistoryLength-1))];
                    
                    if fusedObs(bestObsIdx).IsCoincident
                        obj.Tracks(t).IsCoincident = true;
                    end

                    % M-out-of-N Persistence Confirmation:
                    % Confirmed if: Coincident (Vision+Radar) OR at least 3 hits out of last 5 frames
                    sumHits = sum(obj.Tracks(t).HitHistory);
                    if obj.Tracks(t).IsCoincident || (sumHits >= obj.MinConfirmHits)
                        obj.Tracks(t).Confirmed = true;
                    end
                    
                    % Braking Authority granted ONLY to confirmed tracks with persistent hits
                    if obj.Tracks(t).IsCoincident || (sumHits >= obj.MinConfirmHits && obj.Tracks(t).Hits >= 3)
                        obj.Tracks(t).BrakingAuthority = true;
                    end
                else
                    % Missed update - propagate via dead reckoning
                    obj.Tracks(t).Position = predPos;
                    obj.Tracks(t).Covariance = obj.Tracks(t).Covariance + obj.Q_process * obj.Ts;
                    obj.Tracks(t).Missed = obj.Tracks(t).Missed + 1;
                    obj.Tracks(t).Age = obj.Tracks(t).Age + 1;
                    
                    curHist = obj.Tracks(t).HitHistory;
                    obj.Tracks(t).HitHistory = [false, curHist(1:min(end, obj.HistoryLength-1))];
                end
            end

            % Prune dead tracks
            keepMask = [obj.Tracks.Missed] <= obj.MaxMissedSteps;
            obj.Tracks = obj.Tracks(keepMask);

            % Initialize new tracks for unassociated observations (NEVER grant braking on frame 1)
            for o = 1:numObs
                if ~matchedObs(o)
                    isCoinc = fusedObs(o).IsCoincident;
                    initHist = false(1, obj.HistoryLength);
                    initHist(1) = true;
                    
                    newTrk = struct(...
                        'ID', obj.NextTrackID, ...
                        'Position', fusedObs(o).Pos, ...
                        'Velocity', fusedObs(o).Velocity, ...
                        'Covariance', fusedObs(o).Cov, ...
                        'Class', fusedObs(o).Class, ...
                        'Age', 1, ...
                        'Hits', 1, ...
                        'Missed', 0, ...
                        'HitHistory', initHist, ...
                        'IsCoincident', isCoinc, ...
                        'Confirmed', isCoinc, ... % Immediate confirmation ONLY if Vision+Radar coincident
                        'BrakingAuthority', isCoinc); % Single-sensor blips cannot brake
                    obj.NextTrackID = obj.NextTrackID + 1;
                    obj.Tracks(end+1) = newTrk; %#ok<AGROW>
                end
            end
        end

        function lead = get_lead_obstacle(obj, allTracks)
            lead = struct('Distance', Inf, 'ClosingVelocity', 0.0, 'Class', 'none', 'TrackID', -1);
            if isempty(allTracks), return; end

            for k = 1:numel(allTracks)
                % Enforce Two-Tier Braking Authority Gate:
                % Tentative single-sensor unverified tracks are strictly forbidden from commanding braking!
                if ~allTracks(k).BrakingAuthority
                    continue;
                end

                pos = allTracks(k).Position;
                vel = allTracks(k).Velocity;
                xLong = pos(1);
                yLat  = pos(2);
                vLat  = vel(2);
                cClass = allTracks(k).Class;

                % Forward zone only (in front of front bumper)
                if xLong <= 0.5 || xLong > 50.0
                    continue;
                end

                isCollisionHazard = false;

                % 1. True In-Lane Swept Path: within vehicle body envelope + margin (1.25m)
                if abs(yLat) <= obj.InLaneCorridorHalfWidth
                    isCollisionHazard = true;
                
                % 2. Encroaching VRU / Cut-in: in shoulder buffer (1.25m to 2.8m) within AEB horizon (<= 18m)
                %    Crossing VRUs / cut-ins have |vLat| >= 0.6 m/s, whereas stationary roadside noise has |vLat| < 0.35 m/s
                elseif abs(yLat) <= obj.VRUBufferHalfWidth && xLong <= 18.0
                    isMovingIn = (yLat > 0 && vLat < -0.6) || (yLat < 0 && vLat > 0.6);
                    if isMovingIn
                        isCollisionHazard = true;
                    end
                end

                if isCollisionHazard && (xLong < lead.Distance)
                    lead.Distance = xLong;
                    lead.ClosingVelocity = vel(1);
                    lead.Class = cClass;
                    lead.TrackID = allTracks(k).ID;
                end
            end
        end

        function hazard = get_cutin_hazard(obj, allTracks)
            hazard = struct('Active', false, 'Distance', Inf, 'LateralRate', 0.0, 'Class', 'none');
            if isempty(allTracks), return; end

            for k = 1:numel(allTracks)
                if ~allTracks(k).BrakingAuthority
                    continue;
                end

                pos = allTracks(k).Position;
                vel = allTracks(k).Velocity;
                xLong = pos(1);
                yLat  = pos(2);
                vLat  = vel(2);

                % Flank corridor: adjacent to driving lane (1.25m to 4.5m lateral offset)
                if xLong > 0.5 && xLong < 30.0 && abs(yLat) > obj.InLaneCorridorHalfWidth && abs(yLat) < 4.5
                    % Lateral velocity directed towards ego lane center (y -> 0)
                    isCuttingIn = (yLat > 0 && vLat < -0.2) || (yLat < 0 && vLat > 0.2);
                    if isCuttingIn
                        hazard.Active = true;
                        hazard.Distance = xLong;
                        hazard.LateralRate = abs(vLat);
                        hazard.Class = allTracks(k).Class;
                        return;
                    end
                end
            end
        end
    end
end
