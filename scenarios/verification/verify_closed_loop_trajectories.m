function report = verify_closed_loop_trajectories(varargin)
% VERIFY_CLOSED_LOOP_TRAJECTORIES  Light helper that proves the ego trajectory is sane
% in every scenario by running the full closed-loop AV stack headless and scoring
% the recorded ego/actor poses.
%
%   report = verify_closed_loop_trajectories                 % all scenarios, PurePursuit
%   report = verify_closed_loop_trajectories('mpc')          % all scenarios, MPC
%   report = verify_closed_loop_trajectories('pp', {'POTHOLE','INDIAN_GAUNTLET'})
%
% Checks per scenario (ego = Actor 1, poses exported by run_intersection_scenario):
%   COLLISION  : oriented-box overlap of ego vs every other actor at any step
%   CLEARANCE  : minimum boundary-to-boundary gap (m) to any actor
%   ONCOMING   : (Indian arterial only) ego centre enters oncoming half  Y > -0.6 m
%   ROAD_EDGE  : (Indian arterial only) ego centre beyond outer curb      Y < -6.2 m
%   LAT_ACCEL  : peak lateral acceleration  v*yawRate         (<= 4.0 m/s^2)
%   YAW_RATE   : peak |yaw rate|                              (<= 0.60 rad/s)
%   ZIGZAG     : lateral direction reversals of >= 0.5 m swing (<= 4)
%   END_SWERVE : lateral swing during the last 25% of travel   (<= 1.2 m)
%   BAD_STOP   : standstill > 3 s with nothing ahead in corridor (must be 0 s)
%   REVERSE    : ego moving backwards                          (must be false)
%
% Writes scenarios/verification/trajectory_reports/<scenario>_<ctrl>.png and a
% summary report.txt.  Returns a table of metrics (also printed).

    ctrl = 'pp';
    only = {};
    for i = 1:numel(varargin)
        a = varargin{i};
        if iscell(a), only = a;
        elseif ischar(a) || isstring(a)
            if any(strcmpi(char(a), {'pp','mpc'})), ctrl = lower(char(a)); end
        end
    end

    here    = fileparts(mfilename('fullpath'));
    repo    = fileparts(fileparts(here));
    outDir  = fullfile(here, 'trajectory_reports');
    if ~isfolder(outDir), mkdir(outDir); end
    addpath(repo); addpath(fullfile(repo,'main')); addpath(fullfile(repo,'scenarios','matlab'));
    addpath(fullfile(repo,'vehicle_dynamics')); addpath(fullfile(repo,'Sensor_fusion'));
    addpath(fullfile(repo,'Trajectory')); addpath(fullfile(repo,'Path_planning_decision'));
    addpath(fullfile(repo,'Perception','C3_detector_v1'));

    % name, indian-arterial?, duration
    S = { ...
        'CPNCO',             false, 15; ...
        'CPTA',              false, 15; ...
        'CCFtap',            false, 15; ...
        'CCCscp',            false, 15; ...
        'INDIAN_AUTOCUTIN',  true,  48; ...
        'INDIAN_TWOWHEELER', true,  48; ...
        'INDIAN_JAYWALK',    true,  48; ...
        'INDIAN_CATTLE',     true,  48; ...
        'INDIAN_CONGESTION', true,  48; ...
        'INDIAN_POTHOLE',    true,  48; ...
        'INDIAN_WRONGWAY',   true,  48; ...
        'INDIAN_SCHOOLZONE', true,  48; ...
        'INDIAN_BUSSTOP',    true,  48; ...
        'INDIAN_VENDORCART', true,  48; ...
        'INDIAN_GAUNTLET',   true,  48};
    if ~isempty(only)
        keep = ismember(upper(S(:,1)), upper(only));
        S = S(keep, :);
    end

    logFile = fullfile(repo, 'scenarios', 'verification', 'actor_trajectories_log.mat');
    rows = struct('Scenario',{},'Pass',{},'Collision',{},'MinClear_m',{},'OncomingFrac',{}, ...
        'OutsideEdge',{},'PeakLatAcc',{},'PeakYawRate',{},'Zigzag',{},'EndSwing_m',{}, ...
        'BadStop_s',{},'Reversed',{},'FinalX',{},'FinalSpeed',{},'Notes',{});

    for k = 1:size(S,1)
        name = S{k,1}; isIndian = S{k,2}; dur = S{k,3};
        fprintf('\n##### [%d/%d] %s (%s, %.0f s) #####\n', k, size(S,1), name, ctrl, dur);
        if isfile(logFile), delete(logFile); end
        row = struct('Scenario',name,'Pass',false,'Collision',NaN,'MinClear_m',NaN,'OncomingFrac',NaN, ...
            'OutsideEdge',NaN,'PeakLatAcc',NaN,'PeakYawRate',NaN,'Zigzag',NaN,'EndSwing_m',NaN, ...
            'BadStop_s',NaN,'Reversed',NaN,'FinalX',NaN,'FinalSpeed',NaN,'Notes','');
        try
            evalc("run_intersection_scenario(name, ctrl, 'headless', 'nolog', dur);");
            L = load(logFile);
            row = score_run(L, row, isIndian);
            plot_run(L, row, outDir, name, ctrl, isIndian);
        catch ME
            row.Notes = ['RUN ERROR: ' strrep(ME.message, newline, ' ')];
            fprintf(2, '   %s\n', row.Notes);
        end
        rows(end+1) = row; %#ok<AGROW>
        fprintf('   -> %s | coll=%g clear=%.2fm oncoming=%.1f%% latAcc=%.2f yaw=%.2f zig=%g endSwing=%.2fm badStop=%.1fs | %s\n', ...
            tern(row.Pass,'PASS','FAIL'), row.Collision, row.MinClear_m, 100*row.OncomingFrac, ...
            row.PeakLatAcc, row.PeakYawRate, row.Zigzag, row.EndSwing_m, row.BadStop_s, row.Notes);
    end

    report = struct2table(rows, 'AsArray', true);
    fprintf('\n================ CLOSED-LOOP TRAJECTORY VERIFICATION (%s) ================\n', upper(ctrl));
    disp(report(:, {'Scenario','Pass','Collision','MinClear_m','OncomingFrac','PeakLatAcc','PeakYawRate','Zigzag','EndSwing_m','BadStop_s','FinalX'}));
    fprintf('PASSED %d / %d\n', sum(report.Pass), height(report));
    writetable(report, fullfile(outDir, sprintf('report_%s.csv', ctrl)));
end

% ------------------------------------------------------------------------------------
function row = score_run(L, row, isIndian)
    T  = L.trajectories;  t = L.timeSteps(:);
    n  = numel(t);  dt = median(diff(t));
    ego = T(1);  P = ego.Pos(:,1:2);  V = ego.Vel(:,1:2);  yaw = unwrap(deg2rad(ego.Yaw(:)));
    spd = vecnorm(V, 2, 2);
    row.FinalX = P(end,1);  row.FinalSpeed = spd(end);

    % --- collision / clearance (oriented boxes, sampled boundary distance) ---
    coll = 0; minClear = inf;
    for a = 2:numel(T)
        Ta = T(a);
        if size(Ta.Pos,1) ~= n, continue; end
        for s = 1:2:n
            [c, d] = box_gap(P(s,:), deg2rad(ego.Yaw(s)), ego.Length, ego.Width, ...
                             Ta.Pos(s,1:2), deg2rad(Ta.Yaw(s)), Ta.Length, Ta.Width);
            if c, coll = 1; end
            minClear = min(minClear, d);
        end
    end
    row.Collision = coll;  row.MinClear_m = minClear;

    % --- road legality (Indian arterial: ego half is Y<0, curb at -6.2) ---
    if isIndian
        row.OncomingFrac = mean(P(:,2) > -0.6);
        row.OutsideEdge  = mean(P(:,2) < -6.2);
    else
        row.OncomingFrac = 0;  row.OutsideEdge = 0;
    end

    % --- smoothness ---
    win = max(1, round(0.25/dt));
    yawRate = movmean(gradient(yaw, dt), win);
    row.PeakYawRate = max(abs(yawRate));
    row.PeakLatAcc  = max(abs(spd .* yawRate));
    ysm = movmean(P(:,2), max(1, round(0.5/dt)));
    row.Zigzag = count_reversals(ysm, 0.5);

    % --- end-of-run swerve (last 25% of travelled distance along corridor) ---
    if isIndian && row.FinalX > 250
        cum = [0; cumsum(vecnorm(diff(P),2,2))];
        idx = find(cum >= 0.75*cum(end), 1, 'first');
        if isempty(idx), idx = 1; end
        row.EndSwing_m = max(P(idx:end,2)) - min(P(idx:end,2));
    else
        row.EndSwing_m = 0.0;
    end

    % --- unjustified standstill ---
    stopped = spd < 0.3;  bad = 0;
    for s = 1:n
        if ~stopped(s), continue; end
        blocked = false;
        h = [cos(deg2rad(ego.Yaw(s))), sin(deg2rad(ego.Yaw(s)))];
        for a = 2:numel(T)
            if size(T(a).Pos,1) ~= n, continue; end
            r = T(a).Pos(s,1:2) - P(s,:);
            lon = dot(r,h);  lat = abs(h(1)*r(2) - h(2)*r(1));
            if lon > -1 && lon < 45 && lat < 3.5, blocked = true; break; end
        end
        if ~blocked, bad = bad + dt; end
    end
    row.BadStop_s = bad;
    row.Reversed = double(any(V(:,1).*cos(yaw) + V(:,2).*sin(yaw) < -0.3));

    % --- verdict ---
    notes = {};
    if row.Collision,                 notes{end+1} = 'COLLISION'; end
    if row.OncomingFrac > 0.0,        notes{end+1} = sprintf('ONCOMING %.1f%%', 100*row.OncomingFrac); end
    if row.OutsideEdge  > 0.0,        notes{end+1} = sprintf('OFF-ROAD %.1f%%', 100*row.OutsideEdge); end
    if row.PeakLatAcc   > 4.0,        notes{end+1} = 'HIGH LAT-ACC'; end
    if row.PeakYawRate  > 0.60,       notes{end+1} = 'HIGH YAW-RATE'; end
    if row.Zigzag       > 4,          notes{end+1} = 'ZIGZAG'; end
    if row.EndSwing_m   > 2.0,        notes{end+1} = 'END-SWERVE'; end
    if row.BadStop_s    > 0.0,        notes{end+1} = 'UNJUSTIFIED STOP'; end
    if row.Reversed,                  notes{end+1} = 'REVERSING'; end
    row.Notes = strjoin(notes, '; ');
    row.Pass  = isempty(notes);
end

function n = count_reversals(y, th)
    % Number of direction reversals of >= th metres (out-and-back evasion = 1).
    n = 0; dirn = 0; ext = y(1);
    for i = 2:numel(y)
        if dirn == 0
            if y(i) - ext >= th, dirn = 1;  ext = y(i);
            elseif ext - y(i) >= th, dirn = -1; ext = y(i); end
        elseif dirn == 1
            if y(i) > ext, ext = y(i);
            elseif ext - y(i) >= th, dirn = -1; n = n + 1; ext = y(i); end
        else
            if y(i) < ext, ext = y(i);
            elseif y(i) - ext >= th, dirn = 1; n = n + 1; ext = y(i); end
        end
    end
end

function [overlap, gap] = box_gap(p1, th1, L1, W1, p2, th2, L2, W2)
    A = box_poly(p1, th1, L1, W1);  B = box_poly(p2, th2, L2, W2);
    overlap = overlaps(A, B);
    if overlap, gap = 0; return; end
    ea = sample_boundary(A.Vertices);  eb = sample_boundary(B.Vertices);
    gap = min(vecnorm(permute(ea,[1 3 2]) - permute(eb,[3 1 2]), 2, 3), [], 'all');
end

function pg = box_poly(c, th, L, W)
    L = max(L, 0.3); W = max(W, 0.3);
    X = [ L/2 -L/2 -L/2  L/2]; Y = [ W/2  W/2 -W/2 -W/2];
    R = [cos(th) -sin(th); sin(th) cos(th)];
    V = (R*[X; Y])' + c;
    pg = polyshape(V(:,1), V(:,2));
end

function pts = sample_boundary(V)
    pts = [];
    for i = 1:size(V,1)
        a = V(i,:); b = V(mod(i,size(V,1))+1,:);
        m = max(2, ceil(norm(b-a)/0.25));
        pts = [pts; a + (b-a).*linspace(0,1,m)']; %#ok<AGROW>
    end
end

function plot_run(L, row, outDir, name, ctrl, isIndian)
    T = L.trajectories; t = L.timeSteps(:); ego = T(1); P = ego.Pos(:,1:2);
    spd = vecnorm(ego.Vel(:,1:2),2,2);
    f = figure('Visible','off','Position',[50 50 1500 800],'Color','w');
    try, theme(f, 'light'); catch, end
    subplot(2,2,[1 2]); hold on; grid on; axis equal;
    cols = lines(numel(T));
    for a = 2:numel(T)
        plot(T(a).Pos(:,1), T(a).Pos(:,2), '-', 'Color', cols(a,:), 'LineWidth', 1.2, 'DisplayName', T(a).Name);
        plot(T(a).Pos(end,1), T(a).Pos(end,2), 'o', 'Color', cols(a,:), 'HandleVisibility','off');
    end
    plot(P(:,1), P(:,2), 'k-', 'LineWidth', 2.4, 'DisplayName', 'EGO');
    plot(P(1,1), P(1,2), 'ks', 'MarkerFaceColor','g', 'HandleVisibility','off');
    plot(P(end,1), P(end,2), 'kv', 'MarkerFaceColor','r', 'HandleVisibility','off');
    if isIndian
        yline(-0.6,'r--','centre divider / oncoming','HandleVisibility','off');
        yline(-6.2,'m--','curb','HandleVisibility','off');
        yline(-1.75,':','lane -1','HandleVisibility','off'); yline(-5.0,':','lane -2','HandleVisibility','off');
    end
    legend('Location','eastoutside','Interpreter','none'); xlabel('X (m)'); ylabel('Y (m)');
    title(sprintf('%s | %s | %s  %s', name, upper(ctrl), tern(row.Pass,'PASS','FAIL'), row.Notes), 'Interpreter','none');
    subplot(2,2,3); plot(t, P(:,2), 'k', 'LineWidth', 1.6); grid on; xlabel('t (s)'); ylabel('Ego Y (m)');
    if isIndian, yline(-0.6,'r--'); yline(-6.2,'m--'); end; title('Lateral position');
    subplot(2,2,4); plot(t, spd*3.6, 'b', 'LineWidth', 1.6); grid on; xlabel('t (s)'); ylabel('km/h'); title('Speed');
    exportgraphics(f, fullfile(outDir, sprintf('%s_%s.png', name, ctrl)), 'Resolution', 110);
    close(f);
end

function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
