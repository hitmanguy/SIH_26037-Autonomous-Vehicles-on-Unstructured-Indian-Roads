classdef simulation_logger < handle
% SIMULATION_LOGGER Automated Multi-Camera 3D Image Capture & Sensor Logging Suite
% =========================================================================
% Features:
%   1. Timestamped Run Folders: logs/run_YYYYMMDD_HHMMSS_<Scenario>_<Controller>/
%   2. 4 Surround 3D Camera Views (Front, Near Bumper, Rear, Flank) + Cockpit HUD Mosaic
%   3. Sensor Data Logging: Radars, LiDAR 3D Point Clouds (.mat), Cameras, Fused Tracks
%   4. Automated Sensor Analytics Graphs: Range/TTC, Speed/Accel, Radar Doppler, LiDAR Density
%   5. Metadata Serialization: JSON run summary and complete CSV telemetry tables
% =========================================================================

    properties
        LogRootDir
        RunDir
        ImageDir
        Cam1Dir
        Cam2Dir
        Cam3Dir
        Cam4Dir
        BevDir
        CockpitDir
        GraphDir
        SensorDataDir
        LidarDir

        ScenarioName = 'Scenario'
        ControllerType = 'PurePursuit'
        LogInterval = 0.5           % Capture snapshot every 0.5s (2 Hz)
        LastLogTime = -Inf
        FrameIndex = 0

        % In-memory log buffers
        TelemetryRows = {}
        ActorTrajectoryRows = {}
        RadarRows = {}
        CameraRows = {}
        TrackRows = {}
        LidarTimes = []
        LidarCounts = []
        ReferenceWaypoints = []

        % Offscreen 3D rendering canvas
        OffscreenFig = []
        OffscreenAx = []
        ActorPatches = struct()
        RoadPlotted = false
    end

    methods
        function obj = simulation_logger(scenario, scenarioName, controllerType, logInterval, logRootDir)
            if nargin >= 2 && ~isempty(scenarioName), obj.ScenarioName = char(scenarioName); end
            if nargin >= 3 && ~isempty(controllerType), obj.ControllerType = char(controllerType); end
            if nargin >= 4 && ~isempty(logInterval), obj.LogInterval = double(logInterval); end

            if nargin >= 5 && ~isempty(logRootDir)
                obj.LogRootDir = char(logRootDir);
            else
                thisDir = fileparts(mfilename('fullpath'));
                obj.LogRootDir = fullfile(fileparts(thisDir), 'logs');
            end

            % Create isolated timestamped folder
            tsStr = datestr(now, 'yyyymmdd_HHMMSS');
            safeScen = regexprep(obj.ScenarioName, '[^a-zA-Z0-9_]', '_');
            safeCtrl = regexprep(obj.ControllerType, '[^a-zA-Z0-9_]', '_');
            folderName = sprintf('run_%s_%s_%s', tsStr, safeScen, safeCtrl);
            obj.RunDir = fullfile(obj.LogRootDir, folderName);

            % Build directory tree
            obj.ImageDir      = fullfile(obj.RunDir, 'images');
            obj.Cam1Dir       = fullfile(obj.ImageDir, 'camera_1_front');
            obj.Cam2Dir       = fullfile(obj.ImageDir, 'camera_2_left');
            obj.Cam3Dir       = fullfile(obj.ImageDir, 'camera_3_rear');
            obj.Cam4Dir       = fullfile(obj.ImageDir, 'camera_4_right');
            obj.BevDir        = fullfile(obj.ImageDir, 'bird_eye_view');
            obj.CockpitDir    = fullfile(obj.ImageDir, 'surround_cockpit');
            obj.GraphDir      = fullfile(obj.RunDir, 'graphs');
            obj.SensorDataDir = fullfile(obj.RunDir, 'sensor_data');
            obj.LidarDir      = fullfile(obj.SensorDataDir, 'lidar_scans');

            dirsToCreate = {obj.RunDir, obj.ImageDir, obj.Cam1Dir, obj.Cam2Dir, ...
                            obj.Cam3Dir, obj.Cam4Dir, obj.BevDir, obj.CockpitDir, ...
                            obj.GraphDir, obj.SensorDataDir, obj.LidarDir};
            for k = 1:numel(dirsToCreate)
                if ~isfolder(dirsToCreate{k})
                    mkdir(dirsToCreate{k});
                end
            end

            fprintf('[LOGGER] Initialized timestamped session folder:\n');
            fprintf('         %s\n', obj.RunDir);

            % Ensure all scenario actors are equipped with authentic 3D meshes
            if nargin >= 1 && ~isempty(scenario)
                for a = 1:numel(scenario.Actors)
                    act = scenario.Actors(a);
                    try
                        if isempty(act.Mesh) || size(act.Mesh.Vertices, 1) <= 8
                            if act.ClassID == 1
                                act.Mesh = driving.scenario.carMesh;
                            elseif act.ClassID == 2
                                act.Mesh = driving.scenario.truckMesh;
                            elseif act.ClassID == 4
                                act.Mesh = driving.scenario.pedestrianMesh;
                            elseif act.ClassID == 3 || act.ClassID == 7
                                act.Mesh = driving.scenario.bicycleMesh;
                            end
                        end
                    catch
                    end
                end
            end

            % Initialize hidden offscreen 3D rendering canvas
            try
                obj.OffscreenFig = figure('Visible', 'off', 'Position', [50 50 640 480], ...
                    'Color', [0.06 0.08 0.12], 'MenuBar', 'none', 'ToolBar', 'none');
                obj.OffscreenAx = axes('Parent', obj.OffscreenFig, 'Color', [0.06 0.08 0.12]);
                hold(obj.OffscreenAx, 'on');

                if nargin >= 1 && ~isempty(scenario)
                    plot(scenario, 'Parent', obj.OffscreenAx, 'Meshes', 'on');
                    obj.RoadPlotted = true;
                    view(obj.OffscreenAx, 3);
                    camproj(obj.OffscreenAx, 'perspective');
                    lighting(obj.OffscreenAx, 'gouraud');
                    grid(obj.OffscreenAx, 'off');
                    axis(obj.OffscreenAx, 'equal');
                end
            catch ME
                warning('[LOGGER] Offscreen 3D canvas initialization warning: %s', ME.message);
            end
        end

        function step(obj, scenario, egoActor, telem, sensorRig, simTime)
            % Check logging interval
            isFirst = isempty(obj.TelemetryRows);
            if ~isFirst && (simTime - obj.LastLogTime) < (obj.LogInterval - 1e-4)
                return;
            end

            obj.FrameIndex = obj.FrameIndex + 1;
            obj.LastLogTime = simTime;

            % 1. Render 3D Perspective Images for all 4 Surround Cameras + Synthesized BEV
            obj.render_and_save_cameras(scenario, egoActor, telem, simTime);

            % 2. Log Raw & Fused Sensor Measurements
            obj.record_sensor_measurements(sensorRig, telem, egoActor, simTime);

            % 3. Log Vehicle Telemetry State
            obj.record_vehicle_telemetry(egoActor, telem, simTime);

            % 4. Log All Scenario Actor Trajectories
            obj.record_actor_trajectories(scenario, simTime);
        end

        function render_and_save_cameras(obj, scenario, egoActor, telem, simTime)
            if isempty(obj.OffscreenFig) || ~isvalid(obj.OffscreenFig), return; end
            ax = obj.OffscreenAx;

            egoPos = egoActor.Position;
            egoYawDeg = egoActor.Yaw;
            egoYaw = deg2rad(egoYawDeg);
            R_mat = [cos(egoYaw) -sin(egoYaw) 0; sin(egoYaw) cos(egoYaw) 0; 0 0 1];

            % Re-plot actors at current positions with authentic 3D meshes
            try
                cla(ax);
                plot(scenario, 'Parent', ax, 'Meshes', 'on');
                view(ax, 3);
                camproj(ax, 'perspective');
                lighting(ax, 'gouraud');
                axis(ax, 'equal');
                set(ax, 'Color', [0.06 0.08 0.12]);
            catch
            end

            fileTag = sprintf('frame_%04d_t%05.2fs.png', obj.FrameIndex, simTime);

            % Locate ego actor patch to eliminate camera obstruction
            egoPatch = findobj(ax, 'Tag', sprintf('ActorPatch%d', egoActor.ActorID));

            % 1. Physical Camera 1: Front Windshield (0 deg yaw)
            if ~isempty(egoPatch), set(egoPatch, 'Visible', 'off'); end
            c1_eye = egoPos(:) + R_mat * [2.2; 0.0; 1.35];
            c1_tgt = egoPos(:) + R_mat * [50.0; 0.0; 1.2];
            img1 = obj.capture_perspective_view(ax, c1_eye, c1_tgt, 65, 'Front Windshield (Cam 1)', telem, simTime);
            imwrite(img1, fullfile(obj.Cam1Dir, fileTag));

            % 2. Physical Camera 2: Left Side Mirror (+90 deg yaw)
            c2_eye = egoPos(:) + R_mat * [0.8; 1.05; 1.2];
            c2_tgt = egoPos(:) + R_mat * [0.8; 35.0; 1.0];
            img2 = obj.capture_perspective_view(ax, c2_eye, c2_tgt, 65, 'Left Side Mirror (Cam 2)', telem, simTime);
            imwrite(img2, fullfile(obj.Cam2Dir, fileTag));

            % 3. Physical Camera 3: Rear Tailgate (180 deg yaw)
            c3_eye = egoPos(:) + R_mat * [-2.2; 0.0; 1.15];
            c3_tgt = egoPos(:) + R_mat * [-40.0; 0.0; 1.0];
            img3 = obj.capture_perspective_view(ax, c3_eye, c3_tgt, 65, 'Rear Tailgate (Cam 3)', telem, simTime);
            imwrite(img3, fullfile(obj.Cam3Dir, fileTag));

            % 4. Physical Camera 4: Right Side Mirror (-90 deg yaw)
            c4_eye = egoPos(:) + R_mat * [0.8; -1.05; 1.2];
            c4_tgt = egoPos(:) + R_mat * [0.8; -35.0; 1.0];
            img4 = obj.capture_perspective_view(ax, c4_eye, c4_tgt, 65, 'Right Side Mirror (Cam 4)', telem, simTime);
            imwrite(img4, fullfile(obj.Cam4Dir, fileTag));

            % 5. Synthesized 360 Bird's Eye View (BEV) via Inverse Perspective Mapping
            if ~isempty(egoPatch), set(egoPatch, 'Visible', 'on'); end
            imgBEV = obj.generate_algorithmic_bev(img1, img2, img3, img4, 'Synthesized 360 BEV (Algorithmic IPM)', telem, simTime);
            imwrite(imgBEV, fullfile(obj.BevDir, fileTag));

            % Generate Clean 4-Quadrant Physical Camera Surround Mosaic (2x2 layout, NO filler tiles)
            mosaicImg = obj.build_cockpit_mosaic(img1, img2, img3, img4, telem, egoActor, simTime);
            imwrite(mosaicImg, fullfile(obj.CockpitDir, sprintf('cockpit_%04d_t%05.2fs.png', obj.FrameIndex, simTime)));
        end

        function imgBEV = generate_algorithmic_bev(obj, img1, img2, img3, img4, camTitle, telem, simTime)
            outH = 500; outW = 500;
            imgBEV = zeros(outH, outW, 3, 'uint8');
            
            [H_img, W_img, ~] = size(img1);
            
            % Generate BEV grid (-15m to 25m longitudinal, -15m to 15m lateral)
            [c_grid, r_grid] = meshgrid(1:outW, 1:outH);
            X_w = 25 - (r_grid - 1) * (40 / (outH - 1));
            Y_w = 15 - (c_grid - 1) * (30 / (outW - 1));
            pts_ground = [X_w(:)'; Y_w(:)'; ones(1, outH*outW)];
            
            % Define sector masks based on diagonals
            mask_front = (X_w >= abs(Y_w));
            mask_rear  = (X_w <= -abs(Y_w));
            mask_left  = (Y_w >= abs(X_w));
            mask_right = (Y_w <= -abs(X_w));
            % Vehicle blind spot footprint
            mask_car = (X_w >= -1.0 & X_w <= 3.5 & Y_w >= -0.9 & Y_w <= 0.9);
            
            % Common camera intrinsics
            fovRad = deg2rad(65);
            fy = (H_img / 2) / tan(fovRad / 2);
            fx = fy;
            cx = W_img / 2;
            cy = H_img / 2;
            K = [fx, 0, cx; 0, fy, cy; 0, 0, 1];
            
            % Local configs: {eye, tgt, mask, img}
            cams = {
                {[2.2; 0.0; 1.35],  [50.0; 0.0; 1.2], mask_front, img1},
                {[0.8; 1.05; 1.2],  [0.8; 35.0; 1.0], mask_left,  img2},
                {[-2.2; 0.0; 1.15], [-40.0; 0.0; 1.0], mask_rear,  img3},
                {[0.8; -1.05; 1.2], [0.8; -35.0; 1.0], mask_right, img4}
            };
            
            for i = 1:4
                eye_loc = cams{i}{1};
                tgt_loc = cams{i}{2};
                mask_sec = cams{i}{3};
                img_src = cams{i}{4};
                
                % Compute projective homography matrix
                d = tgt_loc - eye_loc;
                zc = d / norm(d);
                up = [0; 0; 1];
                xc = cross(zc, up);
                xc = xc / norm(xc);
                yc = cross(zc, xc);
                
                R_loc = [xc, yc, zc]';
                t_loc = -R_loc * eye_loc;
                H = K * [R_loc(:, 1), R_loc(:, 2), t_loc];
                
                % Warp BEV grid to image plane
                pts_img = H * pts_ground;
                valid_depth = reshape(pts_img(3, :) > 0, outH, outW);
                
                U = round(reshape(pts_img(1, :) ./ pts_img(3, :), outH, outW));
                V = round(reshape(pts_img(2, :) ./ pts_img(3, :), outH, outW));
                
                in_img = (U >= 1 & U <= W_img & V >= 1 & V <= H_img);
                
                % Apply radial seams and footprint mask
                active_mask = mask_sec & valid_depth & in_img & ~mask_car;
                
                % Copy pixels from perspective image to BEV composite
                idx_bev = find(active_mask);
                idx_src = sub2ind([H_img, W_img], V(active_mask), U(active_mask));
                
                for ch = 1:3
                    bev_ch = imgBEV(:,:,ch);
                    src_ch = img_src(:,:,ch);
                    bev_ch(idx_bev) = src_ch(idx_src);
                    imgBEV(:,:,ch) = bev_ch;
                end
            end
            
            % Fill vehicle footprint
            car_idx = find(mask_car);
            for ch = 1:3
                bev_ch = imgBEV(:,:,ch);
                bev_ch(car_idx) = 50; % Dark gray shadow
                imgBEV(:,:,ch) = bev_ch;
            end
            
            imgBEV = obj.add_cam_hud_banner(imgBEV, camTitle, telem, simTime);
        end

        function img = capture_bev_view(obj, ax, eyePos, tgtPos, yawDeg, camTitle, telem, simTime)
            campos(ax, eyePos(:)');
            camtarget(ax, tgtPos(:)');
            camva(ax, 50);
            camup(ax, [cosd(yawDeg), sind(yawDeg), 0]);
            drawnow limitrate;

            fr = getframe(ax);
            img = fr.cdata;

            % Overlay HUD banner
            img = obj.add_cam_hud_banner(img, camTitle, telem, simTime);
        end

        function img = capture_perspective_view(obj, ax, eyePos, tgtPos, fovDeg, camTitle, telem, simTime)
            campos(ax, eyePos(:)');
            camtarget(ax, tgtPos(:)');
            camva(ax, fovDeg);
            camup(ax, [0 0 1]);
            drawnow limitrate;

            fr = getframe(ax);
            img = fr.cdata;

            % Overlay HUD banner
            img = obj.add_cam_hud_banner(img, camTitle, telem, simTime);
        end

        function img = add_cam_hud_banner(~, img, camTitle, telem, simTime)
            H = size(img, 1);
            W = size(img, 2);

            % Top banner strip (Dark semi-transparent header)
            bannerHeight = min(36, round(H * 0.10));
            img(1:bannerHeight, :, 1) = uint8(double(img(1:bannerHeight, :, 1)) * 0.2 + 20);
            img(1:bannerHeight, :, 2) = uint8(double(img(1:bannerHeight, :, 2)) * 0.2 + 25);
            img(1:bannerHeight, :, 3) = uint8(double(img(1:bannerHeight, :, 3)) * 0.2 + 35);

            % Bottom status strip
            bottomHeight = min(28, round(H * 0.08));
            img((H - bottomHeight + 1):H, :, 1) = uint8(double(img((H - bottomHeight + 1):H, :, 1)) * 0.2 + 20);
            img((H - bottomHeight + 1):H, :, 2) = uint8(double(img((H - bottomHeight + 1):H, :, 2)) * 0.2 + 25);
            img((H - bottomHeight + 1):H, :, 3) = uint8(double(img((H - bottomHeight + 1):H, :, 3)) * 0.2 + 35);

            % Draw crosshair in center
            cx = round(W / 2);
            cy = round(H / 2);
            cLen = 12;
            for c = 1:3
                img(max(1, cy-cLen):min(H, cy+cLen), cx, :) = 180;
                img(cy, max(1, cx-cLen):min(W, cx+cLen), :) = 180;
            end
        end

        function mosaic = build_cockpit_mosaic(~, img1, img2, img3, img4, telem, egoActor, simTime)
            targetH = 240;
            targetW = 320;
            
            i1 = imresize(img1, [targetH, targetW]); % Front Windshield (Cam 1)
            i2 = imresize(img2, [targetH, targetW]); % Left Side Mirror (Cam 2)
            i3 = imresize(img3, [targetH, targetW]); % Rear Tailgate (Cam 3)
            i4 = imresize(img4, [targetH, targetW]); % Right Side Mirror (Cam 4)

            % Pure 4-Quadrant Physical Camera Layout (2x2 grid, 0 dummy tiles)
            % Row 1: [Front Windshield (Cam 1) | Rear Tailgate (Cam 3)]
            % Row 2: [Left Side Mirror (Cam 2) | Right Side Mirror (Cam 4)]
            topRow = [i1, i3];
            botRow = [i2, i4];
            gridImg = [topRow; botRow];

            % Clean Header Bar
            headerH = 30;
            header = zeros(headerH, size(gridImg, 2), 3, 'uint8');
            header(:, :, 1) = 18;
            header(:, :, 2) = 22;
            header(:, :, 3) = 32;

            mosaic = [header; gridImg];
        end

        function record_sensor_measurements(obj, sensorRig, telem, egoActor, simTime)
            if isempty(sensorRig), return; end

            % 1. Collect and serialize Radar detections
            if ~isempty(telem) && isfield(telem, 'RadDets') && ~isempty(telem.RadDets)
                for d = 1:numel(telem.RadDets)
                    det = telem.RadDets{d};
                    m = det.Measurement;
                    rangeVal = norm(m(1:min(3, numel(m))));
                    azVal = atan2d(m(2), m(1));
                    doppVal = 0.0;
                    if numel(m) >= 3, doppVal = m(3); end
                    obj.RadarRows(end+1, :) = {simTime, det.SensorIndex, rangeVal, azVal, doppVal, m(1), m(2)};
                end
            else
                try
                    poses = targetPoses(egoActor);
                    for rIdx = 1:numel(sensorRig.Radars)
                        radGen = sensorRig.Radars{rIdx};
                        if ~isempty(radGen)
                            [dets, numDets, isValid] = radGen(poses, simTime);
                            if isValid && numDets > 0
                                for d = 1:numDets
                                    m = dets{d}.Measurement;
                                    rangeVal = norm(m(1:min(3, numel(m))));
                                    azVal = atan2d(m(2), m(1));
                                    doppVal = 0.0;
                                    if numel(m) >= 3, doppVal = m(3); end
                                    obj.RadarRows(end+1, :) = {simTime, dets{d}.SensorIndex, rangeVal, azVal, doppVal, m(1), m(2)};
                                end
                            end
                        end
                    end
                catch
                end
            end

            % 2. Collect and serialize Camera detections
            if ~isempty(telem) && isfield(telem, 'CamDets') && ~isempty(telem.CamDets)
                for d = 1:numel(telem.CamDets)
                    det = telem.CamDets{d};
                    m = det.Measurement;
                    clsId = 0;
                    if isprop(det, 'ObjectClassID')
                        clsId = det.ObjectClassID;
                    elseif isfield(det, 'ObjectClassID')
                        clsId = det.ObjectClassID;
                    end
                    obj.CameraRows(end+1, :) = {simTime, det.SensorIndex, clsId, m(1), m(2)};
                end
            else
                try
                    poses = targetPoses(egoActor);
                    for cIdx = 1:numel(sensorRig.Cameras)
                        camGen = sensorRig.Cameras{cIdx};
                        if ~isempty(camGen)
                            [dets, numDets, isValid] = camGen(poses, simTime);
                            if isValid && numDets > 0
                                for d = 1:numDets
                                    m = dets{d}.Measurement;
                                    clsId = 0;
                                    if isprop(dets{d}, 'ObjectClassID')
                                        clsId = dets{d}.ObjectClassID;
                                    elseif isfield(dets{d}, 'ObjectClassID')
                                        clsId = dets{d}.ObjectClassID;
                                    end
                                    obj.CameraRows(end+1, :) = {simTime, dets{d}.SensorIndex, clsId, m(1), m(2)};
                                end
                            end
                        end
                    end
                catch
                end
            end

            % 3. Collect and serialize LiDAR Point Clouds
            if ~isempty(telem) && isfield(telem, 'PtCloud') && ~isempty(telem.PtCloud)
                ptCloud = telem.PtCloud;
                if isprop(ptCloud, 'Count') && ptCloud.Count > 0
                    pts = ptCloud.Location;
                    obj.LidarTimes(end+1) = simTime;
                    obj.LidarCounts(end+1) = ptCloud.Count;
                    scanFile = fullfile(obj.LidarDir, sprintf('scan_%04d_t%05.2fs.mat', obj.FrameIndex, simTime));
                    save(scanFile, 'pts', 'simTime');
                end
            else
                try
                    poses = targetPoses(egoActor);
                    if ~isempty(sensorRig.Lidar)
                        [ptCloud, isValid] = sensorRig.Lidar(poses, simTime);
                        if isValid && ptCloud.Count > 0
                            pts = ptCloud.Location;
                            obj.LidarTimes(end+1) = simTime;
                            obj.LidarCounts(end+1) = ptCloud.Count;
                            scanFile = fullfile(obj.LidarDir, sprintf('scan_%04d_t%05.2fs.mat', obj.FrameIndex, simTime));
                            save(scanFile, 'pts', 'simTime');
                        end
                    end
                catch
                end
            end

            % 4. Collect Fused Tracks
            if ~isempty(telem) && isfield(telem, 'FusedTracks') && ~isempty(telem.FusedTracks)
                for trk = 1:numel(telem.FusedTracks)
                    t = telem.FusedTracks(trk);
                    obj.TrackRows(end+1, :) = {simTime, t.ID, t.Position(1), t.Position(2), ...
                        t.Velocity(1), t.Velocity(2), string(t.Class), t.Confirmed, t.BrakingAuthority};
                end
            end
        end

        function record_vehicle_telemetry(obj, egoActor, telem, simTime)
            egoPos = egoActor.Position;
            egoVel = egoActor.Velocity;
            egoYaw = egoActor.Yaw;
            spdMps = norm(egoVel(1:2));
            spdKph = spdMps * 3.6;

            steerDeg = 0.0;
            aCmd = 0.0;
            leadDist = Inf;
            ttc = Inf;
            latErr = 0.0;
            headErr = 0.0;
            refX = NaN;
            refY = NaN;

            if ~isempty(telem)
                if isfield(telem, 'Steering'), steerDeg = rad2deg(telem.Steering); end
                if isfield(telem, 'Accel'), aCmd = telem.Accel; end
                if isfield(telem, 'ObstacleDistance'), leadDist = telem.ObstacleDistance; end
                if isfield(telem, 'TTC'), ttc = telem.TTC; end
                if isfield(telem, 'LateralError'), latErr = telem.LateralError; end
                if isfield(telem, 'HeadingError'), headErr = rad2deg(telem.HeadingError); end
                if isfield(telem, 'TargetPoint') && numel(telem.TargetPoint) >= 2
                    refX = telem.TargetPoint(1);
                    refY = telem.TargetPoint(2);
                end
            end

            obj.TelemetryRows(end+1, :) = {simTime, egoPos(1), egoPos(2), egoPos(3), ...
                egoYaw, spdMps, spdKph, steerDeg, aCmd, leadDist, ttc, latErr, headErr, refX, refY};
        end

        function record_actor_trajectories(obj, scenario, simTime)
            if isempty(scenario) || isempty(scenario.Actors), return; end
            for a = 1:numel(scenario.Actors)
                act = scenario.Actors(a);
                actPos = act.Position;
                actVel = act.Velocity;
                actSpd = norm(actVel(1:2));
                actYaw = act.Yaw;
                actName = char(act.Name);
                if isempty(actName), actName = sprintf('Actor_%d', act.ActorID); end
                
                obj.ActorTrajectoryRows(end+1, :) = {simTime, act.ActorID, string(actName), ...
                    act.ClassID, actPos(1), actPos(2), actPos(3), actYaw, actSpd};
            end
        end

        function finalize(obj)
            fprintf('\n[LOGGER] Finalizing simulation session & generating analytics graphs...\n');

            % 1. Write CSV Tables
            obj.export_csv_tables();

            % 2. Export Publication-Quality Sensor & Trajectory Graphs
            obj.export_sensor_graphs();

            % 3. Write Metadata JSON
            obj.export_metadata_json();

            % Clean up offscreen canvas
            if ~isempty(obj.OffscreenFig) && isvalid(obj.OffscreenFig)
                close(obj.OffscreenFig);
            end

            fprintf('[LOGGER] All session data exported successfully to:\n');
            fprintf('         %s\n\n', obj.RunDir);
        end

        function export_csv_tables(obj)
            % Telemetry Table
            if ~isempty(obj.TelemetryRows)
                telemHeaders = {'Time_s', 'X_world_m', 'Y_world_m', 'Z_world_m', ...
                    'Yaw_deg', 'Speed_mps', 'Speed_kph', 'Steering_deg', 'Accel_mps2', ...
                    'LeadDistance_m', 'TTC_s', 'LateralError_m', 'HeadingError_deg', 'RefX_m', 'RefY_m'};
                tTable = cell2table(obj.TelemetryRows, 'VariableNames', telemHeaders);
                writetable(tTable, fullfile(obj.SensorDataDir, 'vehicle_telemetry.csv'));

                % Dedicated Trajectory Ego Path CSV
                tX = cell2mat(obj.TelemetryRows(:, 2));
                tY = cell2mat(obj.TelemetryRows(:, 3));
                diffX = [0; diff(tX)];
                diffY = [0; diff(tY)];
                cumDist = cumsum(hypot(diffX, diffY));
                
                trajTable = table(cell2mat(obj.TelemetryRows(:, 1)), ...
                    tX, tY, cell2mat(obj.TelemetryRows(:, 4)), ...
                    cell2mat(obj.TelemetryRows(:, 5)), ...
                    cell2mat(obj.TelemetryRows(:, 6)), ...
                    cell2mat(obj.TelemetryRows(:, 7)), ...
                    cell2mat(obj.TelemetryRows(:, 8)), ...
                    cell2mat(obj.TelemetryRows(:, 9)), ...
                    cell2mat(obj.TelemetryRows(:, 12)), ...
                    cell2mat(obj.TelemetryRows(:, 13)), ...
                    cumDist, ...
                    'VariableNames', {'Time_s', 'X_world_m', 'Y_world_m', 'Z_world_m', ...
                                      'Yaw_deg', 'Speed_mps', 'Speed_kph', 'Steering_deg', ...
                                      'Accel_mps2', 'LateralError_m', 'HeadingError_deg', 'DistanceTraveled_m'});
                writetable(trajTable, fullfile(obj.SensorDataDir, 'trajectory_ego_path.csv'));
            end

            % Actor Trajectories Table
            if ~isempty(obj.ActorTrajectoryRows)
                actHeaders = {'Time_s', 'ActorID', 'ActorName', 'ClassID', ...
                    'X_world_m', 'Y_world_m', 'Z_world_m', 'Yaw_deg', 'Speed_mps'};
                actTable = cell2table(obj.ActorTrajectoryRows, 'VariableNames', actHeaders);
                writetable(actTable, fullfile(obj.SensorDataDir, 'trajectory_actor_paths.csv'));
            end

            % Radar Table
            if ~isempty(obj.RadarRows)
                radHeaders = {'Time_s', 'SensorIndex', 'Range_m', 'Azimuth_deg', 'Doppler_mps', 'X_rel_m', 'Y_rel_m'};
                rTable = cell2table(obj.RadarRows, 'VariableNames', radHeaders);
                writetable(rTable, fullfile(obj.SensorDataDir, 'radar_measurements.csv'));
            end

            % Camera Table
            if ~isempty(obj.CameraRows)
                camHeaders = {'Time_s', 'SensorIndex', 'ObjectClassID', 'X_rel_m', 'Y_rel_m'};
                cTable = cell2table(obj.CameraRows, 'VariableNames', camHeaders);
                writetable(cTable, fullfile(obj.SensorDataDir, 'camera_detections.csv'));
            end

            % Track Table
            if ~isempty(obj.TrackRows)
                trkHeaders = {'Time_s', 'TrackID', 'X_rel_m', 'Y_rel_m', 'Vx_mps', 'Vy_mps', 'Class', 'Confirmed', 'BrakingAuthority'};
                trTable = cell2table(obj.TrackRows, 'VariableNames', trkHeaders);
                writetable(trTable, fullfile(obj.SensorDataDir, 'fused_tracks.csv'));
            end
        end

        function export_sensor_graphs(obj)
            if isempty(obj.TelemetryRows), return; end

            tTime  = cell2mat(obj.TelemetryRows(:, 1));
            tSpd   = cell2mat(obj.TelemetryRows(:, 7));
            tAcc   = cell2mat(obj.TelemetryRows(:, 9));
            tDist  = cell2mat(obj.TelemetryRows(:, 10));
            tTTC   = cell2mat(obj.TelemetryRows(:, 11));
            tSteer = cell2mat(obj.TelemetryRows(:, 8));
            tLatErr= cell2mat(obj.TelemetryRows(:, 12));

            % 1. Obstacle Range & TTC Strip-Chart
            f1 = figure('Visible', 'off', 'Position', [100 100 800 450], 'Color', [0.12 0.12 0.14]);
            ax1 = axes('Parent', f1, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
            hold(ax1, 'on'); grid(ax1, 'on');
            set(ax1, 'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.6);
            title(ax1, sprintf('Obstacle Range & Time-To-Collision (TTC) - [%s]', obj.ScenarioName), ...
                'FontSize', 12, 'FontWeight', 'bold', 'Color', [1 1 1]);
            xlabel(ax1, 'Simulation Time [s]', 'Color', [0.8 0.8 0.8]);
            ylabel(ax1, 'Range [m] / TTC [s]', 'Color', [0.8 0.8 0.8]);

            yline(ax1, 20.0, 'Color', [0.2 0.9 0.4], 'LineStyle', '--', 'LineWidth', 1.5, 'DisplayName', 'Safe Distance (20m)');
            yline(ax1, 3.5,  'Color', [1.0 0.2 0.2], 'LineStyle', ':',  'LineWidth', 1.5, 'DisplayName', 'Stop Buffer (3.5m)');
            
            cleanDist = tDist; cleanDist(isinf(cleanDist)) = NaN;
            cleanTTC  = tTTC;  cleanTTC(isinf(cleanTTC)) = NaN;
            plot(ax1, tTime, cleanDist, 'c-', 'LineWidth', 2.0, 'DisplayName', 'Obstacle Range (m)');
            plot(ax1, tTime, min(25, cleanTTC), 'y--', 'LineWidth', 1.8, 'DisplayName', 'TTC (s)');
            legend(ax1, 'Location', 'northeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.1 0.1 0.1]);
            saveas(f1, fullfile(obj.GraphDir, 'obstacle_distance_and_ttc.png'));
            close(f1);

            % 2. Vehicle Dynamics: Speed & Commanded Acceleration
            f2 = figure('Visible', 'off', 'Position', [100 100 800 450], 'Color', [0.12 0.12 0.14]);
            ax2 = axes('Parent', f2, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
            hold(ax2, 'on'); grid(ax2, 'on');
            set(ax2, 'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.6);
            title(ax2, sprintf('Vehicle Dynamics: Ego Speed & Acceleration Profile - [%s]', obj.ScenarioName), ...
                'FontSize', 12, 'FontWeight', 'bold', 'Color', [1 1 1]);
            xlabel(ax2, 'Simulation Time [s]', 'Color', [0.8 0.8 0.8]);
            ylabel(ax2, 'Speed [km/h] / Accel [m/s²]', 'Color', [0.8 0.8 0.8]);

            yline(ax2, 0.0, 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0);
            yline(ax2, -7.5, 'Color', [1.0 0.1 0.1], 'LineStyle', ':', 'LineWidth', 1.5, 'DisplayName', 'Max AEB (-7.5 m/s²)');
            plot(ax2, tTime, tSpd, 'g-', 'LineWidth', 2.2, 'DisplayName', 'Ego Speed (km/h)');
            plot(ax2, tTime, tAcc, 'm--', 'LineWidth', 1.8, 'DisplayName', 'Commanded Accel (m/s²)');
            legend(ax2, 'Location', 'southeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.1 0.1 0.1]);
            saveas(f2, fullfile(obj.GraphDir, 'vehicle_dynamics_speed_accel.png'));
            close(f2);

            % 3. Lateral Guidance & Steering
            f3 = figure('Visible', 'off', 'Position', [100 100 800 450], 'Color', [0.12 0.12 0.14]);
            ax3 = axes('Parent', f3, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
            hold(ax3, 'on'); grid(ax3, 'on');
            set(ax3, 'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.6);
            title(ax3, sprintf('Lateral Guidance: Steering Angle & Cross-Track Error - [%s]', obj.ScenarioName), ...
                'FontSize', 12, 'FontWeight', 'bold', 'Color', [1 1 1]);
            xlabel(ax3, 'Simulation Time [s]', 'Color', [0.8 0.8 0.8]);
            ylabel(ax3, 'Steer [deg] / Error [m]', 'Color', [0.8 0.8 0.8]);

            plot(ax3, tTime, tSteer, 'Color', [0.2 0.8 1.0], 'LineWidth', 2.0, 'DisplayName', 'Steering Angle (deg)');
            plot(ax3, tTime, tLatErr, 'Color', [1.0 0.6 0.2], 'LineStyle', '--', 'LineWidth', 1.8, 'DisplayName', 'Cross-Track Error (m)');
            legend(ax3, 'Location', 'northeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.1 0.1 0.1]);
            saveas(f3, fullfile(obj.GraphDir, 'lateral_tracking_error.png'));
            close(f3);

            % 4. Radar Doppler Profile
            if ~isempty(obj.RadarRows)
                f4 = figure('Visible', 'off', 'Position', [100 100 800 450], 'Color', [0.12 0.12 0.14]);
                ax4 = axes('Parent', f4, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
                hold(ax4, 'on'); grid(ax4, 'on');
                set(ax4, 'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.6);
                title(ax4, 'Radar Detection Range vs Doppler Velocity', 'FontSize', 12, 'FontWeight', 'bold', 'Color', [1 1 1]);
                xlabel(ax4, 'Range [m]', 'Color', [0.8 0.8 0.8]);
                ylabel(ax4, 'Doppler Velocity [m/s]', 'Color', [0.8 0.8 0.8]);

                rRange = cell2mat(obj.RadarRows(:, 3));
                rDopp  = cell2mat(obj.RadarRows(:, 5));
                scatter(ax4, rRange, rDopp, 25, [1.0 0.4 0.1], 'filled', 'MarkerEdgeColor', [1.0 0.8 0.2]);
                saveas(f4, fullfile(obj.GraphDir, 'radar_doppler_profile.png'));
                close(f4);
            end

            % 5. LiDAR Detection Density
            if ~isempty(obj.LidarTimes)
                f5 = figure('Visible', 'off', 'Position', [100 100 800 450], 'Color', [0.12 0.12 0.14]);
                ax5 = axes('Parent', f5, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
                hold(ax5, 'on'); grid(ax5, 'on');
                set(ax5, 'GridColor', [0.3 0.3 0.3], 'GridAlpha', 0.6);
                title(ax5, 'LiDAR 3D Point Return Density Over Time', 'FontSize', 12, 'FontWeight', 'bold', 'Color', [1 1 1]);
                xlabel(ax5, 'Simulation Time [s]', 'Color', [0.8 0.8 0.8]);
                ylabel(ax5, 'Points per Scan', 'Color', [0.8 0.8 0.8]);

                plot(ax5, obj.LidarTimes, obj.LidarCounts, 'Color', [0.2 0.9 0.8], 'LineWidth', 2.0, 'Marker', 'o', 'MarkerSize', 4);
                saveas(f5, fullfile(obj.GraphDir, 'lidar_point_cloud_density.png'));
                close(f5);
            end

            % =============================================================
            % DEDICATED TRAJECTORY ANALYTICS GRAPHS
            % =============================================================
            tX = cell2mat(obj.TelemetryRows(:, 2));
            tY = cell2mat(obj.TelemetryRows(:, 3));
            diffX = [0; diff(tX)];
            diffY = [0; diff(tY)];
            cumDist = cumsum(hypot(diffX, diffY));

            % Curvature and Kinematics
            if numel(tX) >= 3
                dx = gradient(tX);
                dy = gradient(tY);
                ddx = gradient(dx);
                ddy = gradient(dy);
                curv = abs(dx .* ddy - dy .* ddx) ./ max(1e-4, (dx.^2 + dy.^2).^(1.5));
            else
                curv = zeros(size(tX));
            end
            latAcc = ((tSpd / 3.6).^2) .* curv;
            
            if numel(tAcc) >= 2 && numel(tTime) >= 2
                dt = gradient(tTime);
                jerk = gradient(tAcc) ./ max(1e-3, dt);
            else
                jerk = zeros(size(tAcc));
            end

            % 6. Trajectory Graph: 2D Global Spatial Trajectory Map
            f6 = figure('Visible', 'off', 'Position', [100 100 950 600], 'Color', [0.10 0.11 0.14]);
            ax6 = axes('Parent', f6, 'Color', [0.06 0.07 0.09], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
            hold(ax6, 'on'); grid(ax6, 'on'); axis(ax6, 'equal');
            set(ax6, 'GridColor', [0.25 0.25 0.30], 'GridAlpha', 0.6);
            title(ax6, sprintf('2D Global Trajectory & Dynamic Scenario Map - [%s]', obj.ScenarioName), ...
                'FontSize', 12, 'FontWeight', 'bold', 'Color', [1 1 1]);
            xlabel(ax6, 'Global X Position [m]', 'Color', [0.8 0.8 0.8]);
            ylabel(ax6, 'Global Y Position [m]', 'Color', [0.8 0.8 0.8]);

            % Plot Reference Waypoints if available
            if ~isempty(obj.ReferenceWaypoints)
                plot(ax6, obj.ReferenceWaypoints(:, 1), obj.ReferenceWaypoints(:, 2), ...
                    'Color', [0.4 0.6 0.8], 'LineStyle', ':', 'LineWidth', 1.8, 'DisplayName', 'Reference Waypoints');
            elseif size(obj.TelemetryRows, 2) >= 15
                tRefX = cell2mat(obj.TelemetryRows(:, 14));
                tRefY = cell2mat(obj.TelemetryRows(:, 15));
                valRef = ~isnan(tRefX) & ~isnan(tRefY);
                if any(valRef)
                    plot(ax6, tRefX(valRef), tRefY(valRef), 'Color', [0.4 0.6 0.8], ...
                        'LineStyle', ':', 'LineWidth', 1.8, 'DisplayName', 'Reference Path');
                end
            end

            % Plot Ego Vehicle Traveled Path
            plot(ax6, tX, tY, 'Color', [0.1 0.9 0.4], 'LineWidth', 2.5, 'DisplayName', 'Ego Trajectory');

            % Plot Other Scenario Actors (VRUs, Lead Vehicles, Cut-Ins)
            if ~isempty(obj.ActorTrajectoryRows)
                actIDs = cell2mat(obj.ActorTrajectoryRows(:, 2));
                uIDs = unique(actIDs);
                for ui = 1:numel(uIDs)
                    currID = uIDs(ui);
                    if currID == 1, continue; end % Ego actor
                    rowMask = (actIDs == currID);
                    aX = cell2mat(obj.ActorTrajectoryRows(rowMask, 5));
                    aY = cell2mat(obj.ActorTrajectoryRows(rowMask, 6));
                    aName = string(obj.ActorTrajectoryRows{find(rowMask, 1), 3});
                    aClass = cell2mat(obj.ActorTrajectoryRows(find(rowMask, 1), 4));

                    if aClass == 4 % VRU / Pedestrian
                        aCol = [1.0 0.3 0.8];
                        aStyle = '--';
                    else
                        aCol = [1.0 0.6 0.1];
                        aStyle = '-.';
                    end
                    plot(ax6, aX, aY, 'Color', aCol, 'LineStyle', aStyle, 'LineWidth', 2.0, ...
                        'DisplayName', sprintf('%s (Actor %d)', aName, currID));
                    if ~isempty(aX)
                        plot(ax6, aX(1), aY(1), '^', 'Color', aCol, 'MarkerSize', 7, ...
                            'MarkerFaceColor', aCol, 'HandleVisibility', 'off');
                        plot(ax6, aX(end), aY(end), 'v', 'Color', aCol, 'MarkerSize', 7, ...
                            'MarkerFaceColor', aCol, 'HandleVisibility', 'off');
                    end
                end
            end

            % Ego Start, End, and Hazard Reaction Points
            plot(ax6, tX(1), tY(1), 'o', 'Color', [0.2 1.0 0.4], 'MarkerSize', 9, ...
                'MarkerFaceColor', [0.2 1.0 0.4], 'DisplayName', 'Ego Start');
            plot(ax6, tX(end), tY(end), 's', 'Color', [0.2 0.6 1.0], 'MarkerSize', 9, ...
                'MarkerFaceColor', [0.2 0.6 1.0], 'DisplayName', 'Ego Final Stop');

            cleanDist = tDist; cleanDist(isinf(cleanDist)) = NaN;
            [minD, minDIdx] = min(cleanDist);
            if ~isnan(minD) && minDIdx <= numel(tX)
                plot(ax6, tX(minDIdx), tY(minDIdx), 'p', 'Color', [1.0 0.2 0.2], 'MarkerSize', 13, ...
                    'MarkerFaceColor', [1.0 0.2 0.2], 'DisplayName', sprintf('Min Clearance (%.1fm)', minD));
            end

            legend(ax6, 'Location', 'best', 'TextColor', [0.9 0.9 0.9], 'Color', [0.08 0.09 0.12]);
            saveas(f6, fullfile(obj.GraphDir, 'trajectory_2d_global_map.png'));
            close(f6);

            % 7. Trajectory Graph: Tracking Performance & Error Analysis
            f7 = figure('Visible', 'off', 'Position', [100 100 950 700], 'Color', [0.10 0.11 0.14]);
            
            % Subplot 7.1: Lateral Cross-Track Error
            ax71 = subplot(4, 1, 1, 'Parent', f7, 'Color', [0.06 0.07 0.09], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
            hold(ax71, 'on'); grid(ax71, 'on'); set(ax71, 'GridColor', [0.25 0.25 0.30]);
            rmsLat = sqrt(mean(tLatErr.^2));
            title(ax71, sprintf('Lateral Cross-Track Error e_{lat}(t) [RMS: %.3f m, Max: %.3f m]', rmsLat, max(abs(tLatErr))), ...
                'Color', [1 1 1], 'FontWeight', 'bold');
            ylabel(ax71, 'Error [m]', 'Color', [0.8 0.8 0.8]);
            yline(ax71, 0.0, 'Color', [0.5 0.5 0.5], 'LineStyle', '-');
            yline(ax71, 0.20, 'Color', [1.0 0.4 0.2], 'LineStyle', '--', 'DisplayName', '+0.2m Tolerance');
            yline(ax71, -0.20, 'Color', [1.0 0.4 0.2], 'LineStyle', '--', 'DisplayName', '-0.2m Tolerance');
            plot(ax71, tTime, tLatErr, 'Color', [0.2 0.9 1.0], 'LineWidth', 1.8, 'DisplayName', 'Lateral Error');
            legend(ax71, 'Location', 'northeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.08 0.09 0.12]);

            % Subplot 7.2: Heading Error
            ax72 = subplot(4, 1, 2, 'Parent', f7, 'Color', [0.06 0.07 0.09], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
            hold(ax72, 'on'); grid(ax72, 'on'); set(ax72, 'GridColor', [0.25 0.25 0.30]);
            if size(obj.TelemetryRows, 2) >= 13
                tHeadErr = cell2mat(obj.TelemetryRows(:, 13));
            else
                tHeadErr = zeros(size(tTime));
            end
            title(ax72, 'Heading / Yaw Orientation Error \Delta\psi(t)', 'Color', [1 1 1], 'FontWeight', 'bold');
            ylabel(ax72, 'Error [deg]', 'Color', [0.8 0.8 0.8]);
            yline(ax72, 0.0, 'Color', [0.5 0.5 0.5], 'LineStyle', '-');
            plot(ax72, tTime, tHeadErr, 'Color', [1.0 0.8 0.2], 'LineWidth', 1.8);

            % Subplot 7.3: Path Curvature along Distance
            ax73 = subplot(4, 1, 3, 'Parent', f7, 'Color', [0.06 0.07 0.09], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
            hold(ax73, 'on'); grid(ax73, 'on'); set(ax73, 'GridColor', [0.25 0.25 0.30]);
            title(ax73, 'Trajectory Path Curvature \kappa(s) along Traveled Distance', 'Color', [1 1 1], 'FontWeight', 'bold');
            ylabel(ax73, '\kappa [1/m]', 'Color', [0.8 0.8 0.8]);
            plot(ax73, cumDist, curv, 'Color', [0.8 0.3 1.0], 'LineWidth', 1.8);

            % Subplot 7.4: Commanded Steering Angle
            ax74 = subplot(4, 1, 4, 'Parent', f7, 'Color', [0.06 0.07 0.09], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
            hold(ax74, 'on'); grid(ax74, 'on'); set(ax74, 'GridColor', [0.25 0.25 0.30]);
            title(ax74, 'Front Wheel Steering Angle \delta(t)', 'Color', [1 1 1], 'FontWeight', 'bold');
            xlabel(ax74, 'Simulation Time [s]', 'Color', [0.8 0.8 0.8]);
            ylabel(ax74, 'Steer [deg]', 'Color', [0.8 0.8 0.8]);
            plot(ax74, tTime, tSteer, 'Color', [0.2 1.0 0.6], 'LineWidth', 1.8);

            saveas(f7, fullfile(obj.GraphDir, 'trajectory_tracking_and_error_profile.png'));
            close(f7);

            % 8. Trajectory Graph: Velocity, Acceleration & Jerk Dynamics
            f8 = figure('Visible', 'off', 'Position', [100 100 950 650], 'Color', [0.10 0.11 0.14]);
            
            % Subplot 8.1: Speed vs Time
            ax81 = subplot(3, 1, 1, 'Parent', f8, 'Color', [0.06 0.07 0.09], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
            hold(ax81, 'on'); grid(ax81, 'on'); set(ax81, 'GridColor', [0.25 0.25 0.30]);
            title(ax81, sprintf('Trajectory Speed Profile [Max: %.1f km/h]', max(tSpd)), 'Color', [1 1 1], 'FontWeight', 'bold');
            ylabel(ax81, 'Speed [km/h]', 'Color', [0.8 0.8 0.8]);
            plot(ax81, tTime, tSpd, 'Color', [0.1 0.9 0.4], 'LineWidth', 2.0, 'DisplayName', 'Ego Speed');
            legend(ax81, 'Location', 'northeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.08 0.09 0.12]);

            % Subplot 8.2: Longitudinal Acceleration & AEB Safety Bands
            ax82 = subplot(3, 1, 2, 'Parent', f8, 'Color', [0.06 0.07 0.09], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
            hold(ax82, 'on'); grid(ax82, 'on'); set(ax82, 'GridColor', [0.25 0.25 0.30]);
            title(ax82, 'Longitudinal Acceleration & AEB Intervention Thresholds', 'Color', [1 1 1], 'FontWeight', 'bold');
            ylabel(ax82, 'a_x [m/s^2]', 'Color', [0.8 0.8 0.8]);
            yline(ax82, 0.0, 'Color', [0.5 0.5 0.5]);
            yline(ax82, -1.5, 'Color', [1.0 0.7 0.2], 'LineStyle', ':', 'LineWidth', 1.5, 'DisplayName', 'Comfort Decel (-1.5 m/s^2)');
            yline(ax82, -4.0, 'Color', [1.0 0.3 0.2], 'LineStyle', '--', 'LineWidth', 1.5, 'DisplayName', 'AEB Active (-4.0 m/s^2)');
            yline(ax82, -7.5, 'Color', [1.0 0.1 0.1], 'LineStyle', '-', 'LineWidth', 1.8, 'DisplayName', 'Max AEB (-7.5 m/s^2)');
            plot(ax82, tTime, tAcc, 'Color', [1.0 0.4 0.4], 'LineWidth', 1.8, 'DisplayName', 'Commanded a_x');
            legend(ax82, 'Location', 'southeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.08 0.09 0.12]);

            % Subplot 8.3: Lateral Acceleration and Jerk
            ax83 = subplot(3, 1, 3, 'Parent', f8, 'Color', [0.06 0.07 0.09], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
            hold(ax83, 'on'); grid(ax83, 'on'); set(ax83, 'GridColor', [0.25 0.25 0.30]);
            title(ax83, 'Lateral Acceleration a_y(t) & Vehicle Longitudinal Jerk da/dt', 'Color', [1 1 1], 'FontWeight', 'bold');
            xlabel(ax83, 'Simulation Time [s]', 'Color', [0.8 0.8 0.8]);
            ylabel(ax83, 'a_y [m/s^2] / Jerk [m/s^3]', 'Color', [0.8 0.8 0.8]);
            plot(ax83, tTime, latAcc, 'Color', [0.2 0.8 1.0], 'LineWidth', 1.8, 'DisplayName', 'Lateral a_y');
            plot(ax83, tTime, jerk, 'Color', [0.9 0.6 0.2], 'LineStyle', '--', 'LineWidth', 1.6, 'DisplayName', 'Jerk da_x/dt');
            legend(ax83, 'Location', 'northeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.08 0.09 0.12]);

            saveas(f8, fullfile(obj.GraphDir, 'trajectory_velocity_and_acceleration_profile.png'));
            close(f8);

            % 9. Trajectory Graph: Friction Circle & g-g Stability Envelope
            f9 = figure('Visible', 'off', 'Position', [100 100 950 480], 'Color', [0.10 0.11 0.14]);
            
            % Subplot 9.1: g-g Acceleration Diagram
            ax91 = subplot(1, 2, 1, 'Parent', f9, 'Color', [0.06 0.07 0.09], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
            hold(ax91, 'on'); grid(ax91, 'on'); axis(ax91, 'equal'); set(ax91, 'GridColor', [0.25 0.25 0.30]);
            title(ax91, 'g-g Acceleration Diagram (Friction Limits)', 'Color', [1 1 1], 'FontWeight', 'bold');
            xlabel(ax91, 'Lateral Acceleration a_y [m/s^2]', 'Color', [0.8 0.8 0.8]);
            ylabel(ax91, 'Longitudinal Acceleration a_x [m/s^2]', 'Color', [0.8 0.8 0.8]);
            
            thCirc = linspace(0, 2*pi, 120);
            plot(ax91, 0.8 * 9.81 * cos(thCirc), 0.8 * 9.81 * sin(thCirc), 'Color', [0.3 0.8 0.4], ...
                'LineStyle', '--', 'LineWidth', 1.5, 'DisplayName', 'Dry Road Limit (\mu=0.8)');
            plot(ax91, 0.5 * 9.81 * cos(thCirc), 0.5 * 9.81 * sin(thCirc), 'Color', [0.9 0.7 0.2], ...
                'LineStyle', ':', 'LineWidth', 1.5, 'DisplayName', 'Wet Road Limit (\mu=0.5)');
            scatter(ax91, latAcc, tAcc, 30, tTime, 'filled', 'DisplayName', 'Ego Operating Points');
            cb91 = colorbar(ax91, 'Color', [0.8 0.8 0.8]);
            cb91.Label.String = 'Time [s]'; cb91.Label.Color = [0.8 0.8 0.8];
            legend(ax91, 'Location', 'southwest', 'TextColor', [0.9 0.9 0.9], 'Color', [0.08 0.09 0.12]);

            % Subplot 9.2: Total Acceleration Magnitude vs Time
            ax92 = subplot(1, 2, 2, 'Parent', f9, 'Color', [0.06 0.07 0.09], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
            hold(ax92, 'on'); grid(ax92, 'on'); set(ax92, 'GridColor', [0.25 0.25 0.30]);
            totalAcc = hypot(tAcc, latAcc);
            title(ax92, 'Total Acceleration Magnitude |a(t)|', 'Color', [1 1 1], 'FontWeight', 'bold');
            xlabel(ax92, 'Simulation Time [s]', 'Color', [0.8 0.8 0.8]);
            ylabel(ax92, '|a| [m/s^2]', 'Color', [0.8 0.8 0.8]);
            yline(ax92, 0.8 * 9.81, 'Color', [0.3 0.8 0.4], 'LineStyle', '--', 'LineWidth', 1.5, 'DisplayName', 'Dry Adhesion Limit');
            plot(ax92, tTime, totalAcc, 'Color', [0.2 0.9 0.8], 'LineWidth', 1.8, 'DisplayName', '|a(t)|');
            legend(ax92, 'Location', 'northeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.08 0.09 0.12]);

            saveas(f9, fullfile(obj.GraphDir, 'trajectory_friction_circle_dynamics.png'));
            close(f9);
        end

        function export_metadata_json(obj)
            meta = struct();
            meta.SessionID = obj.RunDir;
            meta.Scenario = obj.ScenarioName;
            meta.Controller = obj.ControllerType;
            meta.Timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
            meta.LogInterval_s = obj.LogInterval;
            meta.TotalFramesCaptured = obj.FrameIndex;

            if ~isempty(obj.TelemetryRows)
                tTime = cell2mat(obj.TelemetryRows(:, 1));
                tX    = cell2mat(obj.TelemetryRows(:, 2));
                tY    = cell2mat(obj.TelemetryRows(:, 3));
                tSpd  = cell2mat(obj.TelemetryRows(:, 7));
                tAcc  = cell2mat(obj.TelemetryRows(:, 9));
                tDist = cell2mat(obj.TelemetryRows(:, 10));
                tLat  = cell2mat(obj.TelemetryRows(:, 12));

                meta.SimulationDuration_s = max(tTime);
                diffX = [0; diff(tX)];
                diffY = [0; diff(tY)];
                meta.TrajectoryDistance_m = sum(hypot(diffX, diffY));
                meta.MaxSpeed_kph = max(tSpd);
                meta.MaxDecel_mps2 = min(tAcc);
                meta.MaxLateralError_m = max(abs(tLat));
                meta.RMSLateralError_m = sqrt(mean(tLat.^2));

                cleanD = tDist(~isinf(tDist));
                if ~isempty(cleanD)
                    meta.MinimumClearance_m = min(cleanD);
                else
                    meta.MinimumClearance_m = Inf;
                end
                meta.AEB_Activated = any(tAcc <= -4.0);
            end

            jsonText = jsonencode(meta, 'PrettyPrint', true);
            fid = fopen(fullfile(obj.RunDir, 'metadata.json'), 'w');
            if fid ~= -1
                fwrite(fid, jsonText, 'char');
                fclose(fid);
            end
        end
    end
end
