%% A3 SAHI - Block 4: Simulink model with a 30 Hz full-frame loop + 5 Hz SAHI slow loop
% SIH PS26037 · Team Epsilon · Perception
%
% Builds sahi_B4_demo.slx programmatically:
%
%   Digital Clock (1/30 s) -> Camera (ImageSequenceSource) -> FullFrame (C3FullFrame)   [30 Hz]
%                                   |                             |
%                             Rate Transition (hold newest, 0.2 s) for frame + ff boxes
%                                   v                             v
%                                  SAHI (SahiSlowLoop, fixed-size outputs, N = 128)   [5 Hz]
%   bandRows constant [470 670] (placeholder camera; A1 will drive this port later)
%
% Then simulates 2 s, checks every 5 Hz SAHI output against calling sahiDetect directly
% on the same frame, and screens sahiDetect for code-generation (GPU Coder) blockers.
%
% WRITES D:\SIH\work\A3_sahi\B4_simulink\  (model .slx, log, example PNG)

clear; clc;
here   = fileparts(mfilename('fullpath'));
pkg    = 'D:\SIH\share\C3_detector_v1';
outDir = 'D:\SIH\work\A3_sahi\B4_simulink';
if ~isfolder(outDir), mkdir(outDir); end
addpath(pkg); addpath(here);
logFile = fullfile(outDir, 'B4_simulink_log.txt');
if isfile(logFile), delete(logFile); end
diary(logFile); cleanupDiary = onCleanup(@() diary('off'));
fprintf('A3 SAHI Block 4 Simulink  (%s, MATLAB %s)\n\n', datestr(now), version('-release'));

N = 128; bandRows = [470 670]; stopTime = 2;

%% 1. Build the model
mdl = 'sahi_B4_demo';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
set_param(mdl, 'Solver', 'FixedStepDiscrete', 'FixedStep', '1/30', 'StopTime', num2str(stopTime));

q = @(s) ['''' s ''''];                       % quote a char value for set_param
add_block('simulink/Sources/Digital Clock', [mdl '/Clock'], 'SampleTime', '1/30', 'Position', [30 100 90 130]);

add_block('simulink/User-Defined Functions/MATLAB System', [mdl '/Camera'], 'Position', [140 90 280 150]);
set_param([mdl '/Camera'], 'System', 'ImageSequenceSource');
set_param([mdl '/Camera'], 'SimulateUsing', 'Interpreted execution', ...
    'Folder', fullfile(pkg, 'test_images'), 'HoldSeconds', '0.5');

add_block('simulink/User-Defined Functions/MATLAB System', [mdl '/FullFrame_30Hz'], 'Position', [360 60 520 180]);
set_param([mdl '/FullFrame_30Hz'], 'System', 'C3FullFrame');
set_param([mdl '/FullFrame_30Hz'], 'SimulateUsing', 'Interpreted execution', ...
    'DetectorFolder', pkg, 'MaxDets', num2str(N));

add_block('simulink/User-Defined Functions/MATLAB System', [mdl '/SAHI_5Hz'], 'Position', [760 60 940 260]);
set_param([mdl '/SAHI_5Hz'], 'System', 'SahiSlowLoop');
set_param([mdl '/SAHI_5Hz'], 'SimulateUsing', 'Interpreted execution', ...
    'DetectorFolder', pkg, 'MaxDets', num2str(N), 'TileOnlyConf', '0.35');

add_block('simulink/Sources/Constant', [mdl '/bandRows'], 'Value', mat2str(bandRows), ...
    'SampleTime', '0.2', 'Position', [620 240 690 270]);

% Rate transitions (30 Hz -> 5 Hz, deterministic, holds the value at the slow tick)
rtNames = {'RT_frame', 'RT_boxes', 'RT_scores', 'RT_labels', 'RT_count', 'RT_frameIdx'};
for r = 1:numel(rtNames)
    add_block('simulink/Signal Attributes/Rate Transition', [mdl '/' rtNames{r}], ...
        'OutPortSampleTime', '0.2', 'Position', [620 40 + 30 * r 650 60 + 30 * r]);
end

% Loggers
logs = {'sahi_boxes', 'sahi_scores', 'sahi_labels', 'sahi_count', 'sahi_overflow', 'sahi_only'};
for r = 1:numel(logs)
    add_block('simulink/Sinks/To Workspace', [mdl '/' logs{r}], 'VariableName', logs{r}, ...
        'SaveFormat', 'Structure With Time', 'Position', [1020 40 + 40 * r 1110 65 + 40 * r]);
end
add_block('simulink/Sinks/To Workspace', [mdl '/slow_frameIdx'], 'VariableName', 'slow_frameIdx', ...
    'SaveFormat', 'Structure With Time', 'Position', [760 300 850 325]);
add_block('simulink/Sinks/To Workspace', [mdl '/ff_count'], 'VariableName', 'ff_count', ...
    'SaveFormat', 'Structure With Time', 'Position', [560 200 650 225]);
add_block('simulink/Sinks/Terminator', [mdl '/ff_overflow_term'], 'Position', [560 240 580 260]);

% Wiring
add_line(mdl, 'Clock/1', 'Camera/1');
add_line(mdl, 'Camera/1', 'FullFrame_30Hz/1', 'autorouting', 'on');
add_line(mdl, 'Camera/1', 'RT_frame/1', 'autorouting', 'on');
add_line(mdl, 'Camera/2', 'RT_frameIdx/1', 'autorouting', 'on');
add_line(mdl, 'FullFrame_30Hz/1', 'RT_boxes/1', 'autorouting', 'on');
add_line(mdl, 'FullFrame_30Hz/2', 'RT_scores/1', 'autorouting', 'on');
add_line(mdl, 'FullFrame_30Hz/3', 'RT_labels/1', 'autorouting', 'on');
add_line(mdl, 'FullFrame_30Hz/4', 'RT_count/1', 'autorouting', 'on');
add_line(mdl, 'FullFrame_30Hz/4', 'ff_count/1', 'autorouting', 'on');
add_line(mdl, 'FullFrame_30Hz/5', 'ff_overflow_term/1', 'autorouting', 'on');
add_line(mdl, 'RT_frame/1',  'SAHI_5Hz/1', 'autorouting', 'on');
add_line(mdl, 'RT_boxes/1',  'SAHI_5Hz/2', 'autorouting', 'on');
add_line(mdl, 'RT_scores/1', 'SAHI_5Hz/3', 'autorouting', 'on');
add_line(mdl, 'RT_labels/1', 'SAHI_5Hz/4', 'autorouting', 'on');
add_line(mdl, 'RT_count/1',  'SAHI_5Hz/5', 'autorouting', 'on');
add_line(mdl, 'bandRows/1',  'SAHI_5Hz/6', 'autorouting', 'on');
add_line(mdl, 'RT_frameIdx/1', 'slow_frameIdx/1', 'autorouting', 'on');
for r = 1:numel(logs)
    add_line(mdl, sprintf('SAHI_5Hz/%d', r), [logs{r} '/1'], 'autorouting', 'on');
end
save_system(mdl, fullfile(outDir, [mdl '.slx']));
fprintf('Model built and saved: %s\n', fullfile(outDir, [mdl '.slx']));

%% 2. Simulate
t0 = tic;
out = sim(mdl, 'ReturnWorkspaceOutputs', 'on');
wall = toc(t0);
fprintf('Simulated %.1f s of model time in %.1f s wall time (%.2fx real time)\n\n', stopTime, wall, stopTime / wall);

sb  = out.sahi_boxes.signals.values;    sc = out.sahi_scores.signals.values;
sl  = out.sahi_labels.signals.values;   cnt = squeeze(out.sahi_count.signals.values);
ovf = squeeze(out.sahi_overflow.signals.values); so = out.sahi_only.signals.values;
fIdx = squeeze(out.slow_frameIdx.signals.values); tS = out.sahi_count.time;
ffc = squeeze(out.ff_count.signals.values);
fprintf('Fast loop steps: %d (full-frame count per step: min %d, max %d)\n', numel(ffc), min(ffc), max(ffc));
fprintf('Slow loop ticks: %d at t = %s\n', numel(tS), mat2str(tS', 3));
fprintf('Output sizes per tick: boxes %s, scores %s, labels %s (fixed, N = %d)\n\n', ...
    mat2str(size(sb, [1 2])), mat2str(size(sc, [1 2])), mat2str(size(sl, [1 2])), N);

%% 3. Check each SAHI tick against sahiDetect called directly on the same frame
det = load_c3_detector();
files = [dir(fullfile(pkg, 'test_images', '*.jpg')); dir(fullfile(pkg, 'test_images', '*.png'))];
o = sahiDefaultOpts('v1'); o.RectInput = true; o.BatchTiles = true; o.TileOnlyConf = 0.35;
o.BandRows = bandRows; o.BandRowsImageHeight = 1080;
fprintf('%5s %6s %6s %9s %9s %8s %8s %9s\n', 't', 'frame', 'count', 'direct', 'matched', 'min IoU', 'sahiOnly', 'overflow');
allOk = true; marks = {'MISMATCH', 'ok'};
for k = 1:numel(tS)
    I = imread(fullfile(files(fIdx(k)).folder, files(fIdx(k)).name));
    [bD, sD, ~, iD] = sahiDetect(det, I, o);
    n = cnt(k);
    m = parityMatch([bD(:, 1:2) - 0.5, bD(:, 3:4)], iD.labelIds, sD, ...
                    double(sb(1:n, :, k)), sl(1:n, 1, k), double(sc(1:n, 1, k)));
    ok = m.matched == numel(sD) && m.matched == n;
    allOk = allOk && ok;
    fprintf('%5.1f %6d %6d %9d %9d %8.4f %8d %9d  %s\n', tS(k), fIdx(k), n, numel(sD), m.matched, ...
        m.minIoU, sum(so(1:n, 1, k)), ovf(k), marks{ok + 1});
end
fprintf('\nSimulink output identical to direct sahiDetect: %s\n', string(allOk));

%% 4. Example: last tick drawn from the Simulink outputs
k = numel(tS); n = cnt(k);
I = imread(fullfile(files(fIdx(k)).folder, files(fIdx(k)).name));
I = insertShape(I, 'rectangle', [1 bandRows(1) size(I, 2) - 1 diff(bandRows)], 'LineWidth', 2, 'ShapeColor', 'white');
isS = so(1:n, 1, k);
if any(~isS), I = insertShape(I, 'rectangle', double(sb(~isS, :, k)), 'LineWidth', 3, 'ShapeColor', 'yellow'); end
if any(isS),  I = insertShape(I, 'rectangle', double(sb(isS, :, k)),  'LineWidth', 3, 'ShapeColor', 'cyan'); end
I = insertText(I, [10 10], sprintf('Simulink SAHI 5 Hz, t = %.1f s: %d boxes (cyan = only SAHI found)', tS(k), n), 'FontSize', 28);
imwrite(I, fullfile(outDir, 'B4_simulink_example.png'));

%% 5. Code-generation screening (what GPU Coder would reject today)
fprintf('\n=== Code-generation screening of sahiDetect ===\n');
try
    scr = coder.screener('sahiDetect');
    msgs = scr.Messages;
    if isempty(msgs)
        fprintf('No blocking issues reported.\n');
    else
        for i = 1:numel(msgs)
            fprintf('  - %s (%s)\n', msgs(i).Text, msgs(i).File);
        end
    end
catch err
    fprintf('coder.screener unavailable or failed here: %s\n', err.message);
end
fprintf('\nSaved %s\n', outDir);
