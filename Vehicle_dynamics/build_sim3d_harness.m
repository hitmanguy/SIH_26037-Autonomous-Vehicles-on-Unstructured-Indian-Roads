function modelName = build_sim3d_harness(outputDir)
% BUILD_SIM3D_HARNESS
% =========================================================================
% Programmatically constructs the complete Simulink 3D Surround Camera
% Co-Simulation Harness (sim3d_surround_harness.slx).
%
% Architecture:
%   1. Simulation 3D Scene Configuration (AutoVrtlEnv interface)
%   2. Simulation 3D Vehicle with Ground Following (EgoCar Sim3DVehicle1)
%   3. Four Physical Surround 3D Cameras mounted to EgoCar:
%        - Front Windshield (Yaw = 0 deg)
%        - Left Side Mirror (Yaw = +90 deg)
%        - Rear Tailgate    (Yaw = 180 deg)
%        - Right Side Mirror (Yaw = -90 deg)
%   4. To Workspace sink streaming RGB pixel arrays directly to MATLAB RAM
%
% Syntax:
%   modelName = build_sim3d_harness
%   modelName = build_sim3d_harness(outputDir)
% =========================================================================

    if nargin < 1 || isempty(outputDir)
        outputDir = fileparts(mfilename('fullpath'));
    end

    modelName = 'sim3d_surround_harness';
    modelPath = fullfile(outputDir, [modelName, '.slx']);

    fprintf('============================================================\n');
    fprintf('  CONSTRUCTING SIMULINK 3D SURROUND HARNESS\n');
    fprintf('  Model Name : %s\n', modelName);
    fprintf('  Destination: %s\n', modelPath);
    fprintf('============================================================\n');

    % Close existing instance if open in RAM
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
    if isfile(modelPath)
        delete(modelPath);
    end

    % Create blank Simulink model
    new_system(modelName);
    load_system('drivingsim3d');
    load_system('simulink');

    set_param(modelName, 'Solver', 'FixedStepAuto');
    set_param(modelName, 'FixedStep', '1/60');
    set_param(modelName, 'StopTime', '10.0');

    % 1. Scene Configuration
    add_block('drivingsim3d/Simulation 3D Scene Configuration', ...
        [modelName, '/SceneConfig'], ...
        'Position', [50, 50, 220, 150]);

    % 2. Vehicle Actor with Ground Following
    add_block('drivingsim3d/Simulation 3D Vehicle with Ground Following', ...
        [modelName, '/EgoVehicle'], ...
        'Position', [320, 50, 480, 200], ...
        'ActorTag', 'Sim3DVehicle1');

    % Scalar Pose Inputs (X, Y, Yaw)
    add_block('simulink/Sources/Constant', [modelName, '/PoseX'], ...
        'Position', [150, 60, 220, 90], 'Value', '0');
    add_block('simulink/Sources/Constant', [modelName, '/PoseY'], ...
        'Position', [150, 110, 220, 140], 'Value', '0');
    add_block('simulink/Sources/Constant', [modelName, '/PoseYaw'], ...
        'Position', [150, 160, 220, 190], 'Value', '0');

    add_line(modelName, 'PoseX/1',   'EgoVehicle/1');
    add_line(modelName, 'PoseY/1',   'EgoVehicle/2');
    add_line(modelName, 'PoseYaw/1', 'EgoVehicle/3');

    % 3. Surround Camera Definitions [Name, [X Y Z], [Roll Pitch Yaw], Y-pos]
    % Pitched down slightly to view road surface, lane markings, and obstacles
    camConfigs = {
        'FrontCamera', '[2.2 0.0 1.35]',  '[0 6 0]',    60;
        'LeftCamera',  '[0.8 1.05 1.20]',  '[0 8 90]',   210;
        'RearCamera',  '[-2.2 0.0 1.15]', '[0 8 180]',  360;
        'RightCamera', '[0.8 -1.05 1.20]', '[0 8 -90]',  510
    };

    for c = 1:size(camConfigs, 1)
        cName = camConfigs{c, 1};
        cTrans = camConfigs{c, 2};
        cRot = camConfigs{c, 3};
        yPos = camConfigs{c, 4};

        % Camera Block (Unreal Engine 720p HD at 30/60 = 0.5s snapshot interval)
        blkPath = [modelName, '/', cName];
        add_block('drivingsim3d/Simulation 3D Camera', blkPath, ...
            'Position', [600, yPos, 760, yPos + 100], ...
            'tmountOffset', cTrans, ...
            'rmountOffset', cRot, ...
            'ImageSize', '[720 1280]', ...
            'SampleTime', '30/60');

        % To Workspace Sink for Image Stream
        sinkPath = [modelName, '/ToWS_', cName];
        varName = sprintf('sim3d_%s_rgb', lower(cName));
        add_block('simulink/Sinks/To Workspace', sinkPath, ...
            'Position', [860, yPos + 20, 960, yPos + 60], ...
            'VariableName', varName, ...
            'SaveFormat', 'Timeseries');

        % Connect Camera Image Port (Port 1) to Workspace Sink
        add_line(modelName, [cName, '/1'], ['ToWS_', cName, '/1']);
    end

    % Save and close model
    save_system(modelName, modelPath);
    close_system(modelName);

    fprintf('============================================================\n');
    fprintf('  SUCCESS: Built Simulink 3D Surround Harness:\n');
    fprintf('           %s\n', modelPath);
    fprintf('============================================================\n');
end
