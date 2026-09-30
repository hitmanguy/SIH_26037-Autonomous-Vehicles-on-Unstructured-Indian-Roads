function [scenario, ego, sensor, refPathWaypoints, trackInfo] = create_ego_circle_scenario(varargin)
% CREATE_EGO_CIRCLE_SCENARIO Generates circular road scenario with camera sensor
% =========================================================================
% Options:
%   'trackType'   - 'winding' (default, 8-node track) or 'smooth_circle'
%   'hasObstacle' - true / false (default: false, adds lead obstacle vehicle)
%   'sampleTime'  - scenario time step (default: 0.05 s)
%   'stopTime'    - scenario duration (default: 30.0 s)
% =========================================================================

    p = inputParser;
    addParameter(p, 'trackType', 'winding', @(x) ischar(x) || isstring(x));
    addParameter(p, 'hasObstacle', false, @islogical);
    addParameter(p, 'sampleTime', 0.05, @isnumeric);
    addParameter(p, 'stopTime', 30.0, @isnumeric);
    parse(p, varargin{:});
    opts = p.Results;

    scenario = drivingScenario('SampleTime', opts.sampleTime, 'StopTime', opts.stopTime);

    % 1. Define Road Geometry
    switch lower(char(opts.trackType))
        case 'smooth_circle'
            % Smooth circular circuit with radius R = 50m
            theta = linspace(0, 2*pi, 36)';
            R = 50;
            roadCenters = [R*cos(theta), R*sin(theta), zeros(size(theta))];
            laneWidth = 3.6;
        otherwise % 'winding' (from team's circleRoadScenario.m design)
            roadCenters = [
                 0.3   -3.0   0.0;
                20.5   -9.9   0.0;
                28.2  -30.6   0.0;
                55.7  -39.8   0.0;
                46.9   -6.3   0.0;
                61.6   20.3   0.0;
                39.7   10.1   0.0;
                 0.9   29.8   0.0;
                 0.3   -3.0   0.0  % Close the loop
            ];
            laneWidth = 3.5;
    end

    laneSpec = lanespec(2, 'Width', laneWidth);
    road(scenario, roadCenters, 'Lanes', laneSpec, 'Name', 'CircleRoad');

    % 2. Extract Dense Reference Centerline for Lane Tracking
    % Use fine spline interpolation along road centers
    denseRef = interpolate_centerline(roadCenters(:, 1:2), 0.5);
    refPathWaypoints = denseRef;

    % Initial vehicle pose: start of reference path
    x0 = denseRef(1, 1);
    y0 = denseRef(1, 2);
    dx = denseRef(2, 1) - x0;
    dy = denseRef(2, 2) - y0;
    yaw0_rad = atan2(dy, dx);
    yaw0_deg = rad2deg(yaw0_rad);

    % 3. Instantiate Ego Vehicle (No predefined trajectory - 100% automated)
    ego = vehicle(scenario, 'ClassID', 1, 'Name', 'EgoCar_Autonomous', ...
        'AssetType', 'Sedan', 'Length', 4.5, 'Width', 1.8, 'Height', 1.45, ...
        'Position', [x0, y0, 0.0], 'Yaw', yaw0_deg, ...
        'PlotColor', [0.05 0.45 0.95]);

    % 4. Mount Forward-Facing Camera Sensor
    sensor = visionDetectionGenerator('SensorIndex', 1, ...
        'SensorLocation', [1.9, 0.0], ...     % Mounted on front bumper / windshield
        'Height', 1.4, ...
        'Yaw', 0.0, ...
        'MaxRange', 50.0, ...                 % 50 m forward visibility
        'DetectorOutput', 'Objects only', ...
        'DetectionProbability', 1.0, ...
        'FalsePositivesPerImage', 0.0, ...
        'Intrinsics', cameraIntrinsics([800 800], [320 240], [480 640]));

    % 5. Optional Lead Obstacle Vehicle
    leadVehicle = [];
    if opts.hasObstacle
        % Place obstacle roughly 25 meters ahead along the track
        obsIdx = min(size(denseRef, 1), 60);
        xObs = denseRef(obsIdx, 1);
        yObs = denseRef(obsIdx, 2);
        dxObs = denseRef(obsIdx+1, 1) - xObs;
        dyObs = denseRef(obsIdx+1, 2) - yObs;
        yawObs_deg = rad2deg(atan2(dyObs, dxObs));

        leadVehicle = vehicle(scenario, 'ClassID', 1, 'Name', 'LeadVehicle_Obstacle', ...
            'AssetType', 'Hatchback', 'Length', 4.0, 'Width', 1.75, 'Height', 1.5, ...
            'Position', [xObs, yObs, 0.0], 'Yaw', yawObs_deg, ...
            'PlotColor', [0.85 0.20 0.15]);
        % Slow cruise trajectory along remaining waypoints
        obsWps = [denseRef(obsIdx:end, :), zeros(size(denseRef(obsIdx:end, :), 1), 1)];
        trajectory(leadVehicle, obsWps, 3.0); % Crawls at 3 m/s (11 km/h)
    end

    trackInfo = struct();
    trackInfo.TrackType = char(opts.trackType);
    trackInfo.RoadCenters = roadCenters;
    trackInfo.LaneWidth = laneWidth;
    trackInfo.InitialPose = [x0, y0, yaw0_rad];
    trackInfo.LeadVehicle = leadVehicle;
end

function dense = interpolate_centerline(pts, ds)
    % Arc-length parameterized cubic spline interpolation
    diffs = diff(pts, 1, 1);
    segLens = hypot(diffs(:, 1), diffs(:, 2));
    sCum = [0; cumsum(segLens)];
    totalLen = sCum(end);

    sQuery = (0:ds:totalLen)';
    xDense = interp1(sCum, pts(:, 1), sQuery, 'spline');
    yDense = interp1(sCum, pts(:, 2), sQuery, 'spline');
    dense = [xDense, yDense];
end
