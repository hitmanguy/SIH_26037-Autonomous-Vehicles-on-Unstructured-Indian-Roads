function out = run_sim3d_surround_core(varargin)
% RUN_SIM3D_SURROUND_CORE 4-Camera Surround Simulink 3D Co-Simulation Runner
% =========================================================================
% Executes the 4-Camera Surround Simulink 3D Co-Simulation Harness
% (sim3d_surround_harness.slx).
%
% Streams 4 synchronous 720p photorealistic Unreal Engine camera feeds
% directly into MATLAB workspace RAM and exports high-definition frames
% into an isolated, timestamped session folder.
%
% Outputs:
%   - images/camera_1_front/frame_*.png  (1280x720 Unreal Engine Front View)
%   - images/camera_2_left/frame_*.png   (1280x720 Unreal Engine Left Mirror)
%   - images/camera_3_rear/frame_*.png   (1280x720 Unreal Engine Rear Tailgate)
%   - images/camera_4_right/frame_*.png  (1280x720 Unreal Engine Right Mirror)
%   - images/surround_cockpit/cockpit_*.png (4-Quadrant HD Surround Mosaic)
%   - images/bird_eye_view/frame_*.png   (Synthesized 360 BEV from 4 Cameras)
%   - metadata.json                      (Simulation & Sensor Configuration)
%
% Syntax:
%   run_sim3d_surround
%   run_sim3d_surround('CPNCO', 'pp')
%   run_sim3d_surround('CPNCO', 'pp', 5.0)
%   run_sim3d_surround('CPNCO', 10.0)
% =========================================================================

    thisDir = fileparts(mfilename('fullpath'));
    if endsWith(thisDir, 'vehicle_dynamics')
        rootDir = fileparts(thisDir);
    else
        rootDir = thisDir;
    end
    thisDir = fullfile(rootDir, 'vehicle_dynamics');

    addpath(fullfile(rootDir, 'scenarios'));
    addpath(fullfile(rootDir, 'scenarios', 'matlab'));
    addpath(fullfile(rootDir, 'vehicle_dynamics'));

    modelName = 'sim3d_surround_harness';
    modelPath = fullfile(thisDir, [modelName, '.slx']);

    if ~isfile(modelPath)
        fprintf('[SIM3D] Harness not found on disk. Building %s...\n', modelPath);
        build_sim3d_harness(thisDir);
    end

    % Parse inputs flexibly (support scenario, controller, duration in any order)
    scenarioName = 'CPNCO';
    controllerType = 'PurePursuit';
    simDuration = 5.0;

    for i = 1:nargin
        arg = varargin{i};
        if ischar(arg) || isstring(arg)
            carg = char(arg);
            su = upper(carg);
            switch su
                case {'PP', 'PUREPURSUIT', 'PURSUIT'}
                    controllerType = 'PurePursuit';
                case {'MPC', 'NMPC'}
                    controllerType = 'MPC';
                otherwise
                    if ~isnan(str2double(carg))
                        simDuration = str2double(carg);
                    else
                        scenarioName = carg;
                    end
            end
        elseif isnumeric(arg) && isscalar(arg)
            simDuration = double(arg);
        end
    end

    % Safety clamp for duration
    simDuration = max(0.5, simDuration);

    fprintf('============================================================\n');
    fprintf('  SIMULINK 3D SURROUND CO-SIMULATION RUNNER\n');
    fprintf('  Model      : %s\n', modelName);
    fprintf('  Scenario   : %s\n', scenarioName);
    fprintf('  Controller : %s\n', controllerType);
    fprintf('  Duration   : %.1f seconds\n', simDuration);
    fprintf('  Resolution : 1280x720 HD per camera (4 Surround Cameras)\n');
    fprintf('============================================================\n');

    % Create timestamped session folder
    tsStr = datestr(now, 'yyyymmdd_HHMMSS');
    safeScen = regexprep(scenarioName, '[^a-zA-Z0-9_]', '_');
    runDir = fullfile(rootDir, 'logs', sprintf('run_%s_%s_Sim3DSurround', tsStr, safeScen));

    cam1Dir = fullfile(runDir, 'images', 'camera_1_front');
    cam2Dir = fullfile(runDir, 'images', 'camera_2_left');
    cam3Dir = fullfile(runDir, 'images', 'camera_3_rear');
    cam4Dir = fullfile(runDir, 'images', 'camera_4_right');
    cockpitDir = fullfile(runDir, 'images', 'surround_cockpit');
    bevDir = fullfile(runDir, 'images', 'bird_eye_view');

    dirsToCreate = {runDir, cam1Dir, cam2Dir, cam3Dir, cam4Dir, cockpitDir, bevDir};
    for d = 1:numel(dirsToCreate)
        if ~isfolder(dirsToCreate{d}), mkdir(dirsToCreate{d}); end
    end

    % Load model into RAM
    load_system(modelPath);

    % Ensure clean model closure on exit or error
    cleanupObj = onCleanup(@() close_system(modelName, 0));

    % Configure stop time
    set_param(modelName, 'StopTime', sprintf('%.2f', simDuration));

    % Execute simulation
    fprintf('[SIM3D] Executing 4-Camera Unreal Engine Co-Simulation in Simulink...\n');
    out = sim(modelName);
    fprintf('[SIM3D] Simulation complete. Processing 720p camera data streams...\n');

    % Extract camera timeseries streams (supports both out object and base workspace)
    getVar = @(name) extract_ts(out, name);
    fcam  = getVar('sim3d_frontcamera_rgb');
    lcam  = getVar('sim3d_leftcamera_rgb');
    rcam  = getVar('sim3d_rearcamera_rgb');
    scamm = getVar('sim3d_rightcamera_rgb');

    totalFramesSaved = 0;
    if ~isempty(fcam) && isa(fcam, 'timeseries')
        timeVec = fcam.Time;
        numSamples = numel(timeVec);
        logInterval = 0.5; % Snapshot every 0.5 seconds
        lastLogTime = -Inf;

        for s = 1:numSamples
            t_now = timeVec(s);
            if s > 1 && (t_now - lastLogTime) < (logInterval - 1e-4)
                continue;
            end
            lastLogTime = t_now;
            totalFramesSaved = totalFramesSaved + 1;

            frameTag = sprintf('frame_%04d_t%05.2fs.png', totalFramesSaved, t_now);

            % 1. Extract 720p HD Unreal Engine Camera Images
            img1 = fcam.Data(:, :, :, s);
            imwrite(img1, fullfile(cam1Dir, frameTag));

            img2 = []; img3 = []; img4 = [];
            if ~isempty(lcam), img2 = lcam.Data(:, :, :, s); imwrite(img2, fullfile(cam2Dir, frameTag)); end
            if ~isempty(rcam), img3 = rcam.Data(:, :, :, s); imwrite(img3, fullfile(cam3Dir, frameTag)); end
            if ~isempty(scamm), img4 = scamm.Data(:, :, :, s); imwrite(img4, fullfile(cam4Dir, frameTag)); end

            % 2. Generate 4-Quadrant HD Surround Cockpit Mosaic
            if ~isempty(img1) && ~isempty(img2) && ~isempty(img3) && ~isempty(img4)
                sub1 = imresize(img1, [360, 640]);
                sub2 = imresize(img2, [360, 640]);
                sub3 = imresize(img3, [360, 640]);
                sub4 = imresize(img4, [360, 640]);

                sub1 = add_banner(sub1, sprintf('Front (0 deg) | t=%.2fs', t_now));
                sub2 = add_banner(sub2, sprintf('Left (+90 deg) | t=%.2fs', t_now));
                sub3 = add_banner(sub3, sprintf('Rear (180 deg) | t=%.2fs', t_now));
                sub4 = add_banner(sub4, sprintf('Right (-90 deg) | t=%.2fs', t_now));

                topRow = [sub1, sub3];   % Front | Rear
                botRow = [sub2, sub4];   % Left  | Right
                mosaic = [topRow; botRow];
                imwrite(mosaic, fullfile(cockpitDir, sprintf('cockpit_%04d_t%05.2fs.png', totalFramesSaved, t_now)));

                % 3. Synthesize Algorithmic IPM BEV from the 4 physical cameras
                bevImg = synthesize_ipm_bev(img1, img2, img3, img4, t_now);
                imwrite(bevImg, fullfile(bevDir, frameTag));
            end
        end
        fprintf('[SIM3D] Exported %d snapshot sets at %.1fs intervals to:\n        %s\n', ...
            totalFramesSaved, logInterval, runDir);
    end

    % Export metadata.json
    meta = struct();
    meta.SessionID = runDir;
    meta.Scenario = scenarioName;
    meta.Controller = controllerType;
    meta.SimulationDuration_s = simDuration;
    meta.CameraResolution = '1280x720';
    meta.NumCameras = 4;
    meta.CameraMounts = struct(...
        'Front', 'Translation: [2.2, 0.0, 1.35] m, Yaw: 0 deg', ...
        'Left',  'Translation: [0.8, 1.05, 1.20] m, Yaw: +90 deg', ...
        'Rear',  'Translation: [-2.2, 0.0, 1.15] m, Yaw: 180 deg', ...
        'Right', 'Translation: [0.8, -1.05, 1.20] m, Yaw: -90 deg');
    meta.FramesCaptured = totalFramesSaved;
    meta.Timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');

    fidMeta = fopen(fullfile(runDir, 'metadata.json'), 'w');
    if fidMeta ~= -1
        fwrite(fidMeta, jsonencode(meta, 'PrettyPrint', true), 'char');
        fclose(fidMeta);
    end

    fprintf('============================================================\n');
    fprintf('  SUCCESS: Simulink 3D Surround Co-Simulation Complete!\n');
    fprintf('  Session Folder : %s\n', runDir);
    fprintf('  Front Camera   : %s (%d frames, 1280x720)\n', cam1Dir, totalFramesSaved);
    fprintf('  Left Camera    : %s (%d frames, 1280x720)\n', cam2Dir, totalFramesSaved);
    fprintf('  Rear Camera    : %s (%d frames, 1280x720)\n', cam3Dir, totalFramesSaved);
    fprintf('  Right Camera   : %s (%d frames, 1280x720)\n', cam4Dir, totalFramesSaved);
    fprintf('  Cockpit Mosaic : %s\n', cockpitDir);
    fprintf('============================================================\n');

    outStruct = struct();
    outStruct.SessionDir = runDir;
    outStruct.Scenario = scenarioName;
    outStruct.Controller = controllerType;
    outStruct.Duration = simDuration;
    outStruct.FramesCaptured = totalFramesSaved;
    outStruct.FrontCameraDir = cam1Dir;
    outStruct.LeftCameraDir = cam2Dir;
    outStruct.RearCameraDir = cam3Dir;
    outStruct.RightCameraDir = cam4Dir;
    outStruct.CockpitMosaicDir = cockpitDir;
    outStruct.SynthesizedBevDir = bevDir;
    if exist('out', 'var'), outStruct.SimulinkOutput = out; end
    out = outStruct;
end

% =========================================================================
% Helper: Add text banner to sub-image
% =========================================================================
function img = add_banner(img, txt)
    h = size(img, 1);
    w = size(img, 2);
    banH = min(28, round(h * 0.08));
    for c = 1:3
        img(1:banH, :, c) = uint8(double(img(1:banH, :, c)) * 0.25);
    end
    try
        img = insertText(img, [10, 4], txt, ...
            'FontSize', 12, 'TextColor', [0.2 1.0 0.4], ...
            'BoxOpacity', 0, 'Font', 'LucidaSansDemiBold');
    catch
    end
end

% =========================================================================
% Helper: Synthesize Algorithmic IPM BEV composite from 4 cameras
% =========================================================================
function bevImg = synthesize_ipm_bev(img1, img2, img3, img4, simTime)
    outH = 500; outW = 500;
    bevImg = zeros(outH, outW, 3, 'uint8');

    [H_img, W_img, ~] = size(img1);

    % BEV ground grid (-15m to +25m longitudinal, -15m to +15m lateral)
    [c_grid, r_grid] = meshgrid(1:outW, 1:outH);
    X_w = 25 - (r_grid - 1) * (40 / (outH - 1));
    Y_w = 15 - (c_grid - 1) * (30 / (outW - 1));
    pts_ground = [X_w(:)'; Y_w(:)'; ones(1, outH*outW)];

    mask_front = (X_w >= abs(Y_w));
    mask_rear  = (X_w <= -abs(Y_w));
    mask_left  = (Y_w >= abs(X_w));
    mask_right = (Y_w <= -abs(X_w));
    mask_car   = (X_w >= -1.0 & X_w <= 3.5 & Y_w >= -0.9 & Y_w <= 0.9);

    fovRad = deg2rad(65);
    fy = (H_img / 2) / tan(fovRad / 2);
    fx = fy;
    cx = W_img / 2;
    cy = H_img / 2;
    K = [fx, 0, cx; 0, fy, cy; 0, 0, 1];

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

        d = tgt_loc - eye_loc;
        zc = d / norm(d);
        up = [0; 0; 1];
        xc = cross(zc, up);
        xc = xc / norm(xc);
        yc = cross(zc, xc);

        R_loc = [xc, yc, zc]';
        t_loc = -R_loc * eye_loc;
        H = K * [R_loc(:, 1), R_loc(:, 2), t_loc];

        pts_img = H * pts_ground;
        valid_depth = reshape(pts_img(3, :) > 0, outH, outW);

        U = round(reshape(pts_img(1, :) ./ pts_img(3, :), outH, outW));
        V = round(reshape(pts_img(2, :) ./ pts_img(3, :), outH, outW));

        in_img = (U >= 1 & U <= W_img & V >= 1 & V <= H_img);
        active_mask = mask_sec & valid_depth & in_img & ~mask_car;

        idx_bev = active_mask;
        idx_src = sub2ind([H_img, W_img], V(active_mask), U(active_mask));

        for ch = 1:3
            bev_ch = bevImg(:, :, ch);
            src_ch = img_src(:, :, ch);
            bev_ch(idx_bev) = src_ch(idx_src);
            bevImg(:, :, ch) = bev_ch;
        end
    end

    % Car shadow mask in center
    car_idx = mask_car;
    for ch = 1:3
        bev_ch = bevImg(:, :, ch);
        bev_ch(car_idx) = 45;
        bevImg(:, :, ch) = bev_ch;
    end

    % Add HUD banner
    banH = 30;
    for c = 1:3, bevImg(1:banH, :, c) = uint8(double(bevImg(1:banH, :, c)) * 0.2); end
    try
        bevImg = insertText(bevImg, [12, 6], sprintf('Synthesized 360 BEV (Simulink 3D) | t=%.2fs', simTime), ...
            'FontSize', 11, 'TextColor', [0.1 1.0 0.4], 'BoxOpacity', 0);
    catch
    end
end

function ts = extract_ts(outObj, varName)
    ts = [];
    if ismember(varName, outObj.who)
        ts = outObj.get(varName);
    elseif evalin('base', sprintf('exist(''%s'', ''var'')', varName))
        ts = evalin('base', varName);
    end
end

