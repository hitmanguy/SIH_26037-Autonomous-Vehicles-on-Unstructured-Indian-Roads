%% A3 SAHI - Block 2: speed on the RTX 4050
% SIH PS26037 · Team Epsilon · Perception
%
% Times sahiDetect in several set-ups on the 4 test frames and reports MEDIAN ms.
% Budget: SAHI slow loop at 5 Hz = 200 ms per tick.
%   square = every input padded to 640x640 (as sahi_engine.py)
%   rect   = padded only to a multiple of 32 (full frame 640x384, v1 tile 640x224)
%   batch  = all same-size tiles in ONE predict call
%   reuse  = slow loop reuses the 30 Hz loop's full-frame boxes (no second full-frame pass)
%
% WRITES D:\SIH\work\A3_sahi\B2_speed\  (log, results .mat, CSV)

clear; clc;
here   = fileparts(mfilename('fullpath'));
pkg    = fullfile(here, '..', '..', 'C3_detector_v1');   % detector package in this repo
outDir = 'D:\SIH\work\A3_sahi\B2_speed';
if ~isfolder(outDir), mkdir(outDir); end
addpath(pkg); addpath(here, fullfile(here, ".."), fullfile(here, "..", "simulink"));   % tests/ -> sahi/ core + simulink/
logFile = fullfile(outDir, 'B2_speed_log.txt');
if isfile(logFile), delete(logFile); end
diary(logFile); cleanupDiary = onCleanup(@() diary('off'));

nRep = 15;                                   % timed repetitions per frame and set-up
fprintf('A3 SAHI Block 2 speed  (%s, MATLAB %s)\n', datestr(now), version('-release'));
g = gpuDevice; fprintf('GPU: %s, %.1f GB\n\n', g.Name, g.TotalMemory / 2^30);

det = load_c3_detector();
names = cellstr(string(det.ClassNames));
files = dir(fullfile(pkg, 'test_images', '*.jpg'));
imgs = cell(numel(files), 1);
for k = 1:numel(files), imgs{k} = imread(fullfile(files(k).folder, files(k).name)); end

%% 1. Can the network take non-square inputs and batches?
net = det.Network;
[okRect, msgRect]   = tryPredict(net, zeros(224, 640, 3, 1, 'single'));
[okBatch, msgBatch] = tryPredict(net, zeros(640, 640, 3, 4, 'single'));
fprintf('Original network:  640x224 input %s | batch of 4 %s\n', okStr(okRect, msgRect), okStr(okBatch, msgBatch));
detUse = det;
if ~(okRect && okBatch)
    % Swap the fixed-size image input layer for a size-free one (no normalisation in
    % either, so the maths is unchanged), then re-check.
    lay = net.Layers(1);
    flexNet = replaceLayer(net, lay.Name, inputLayer([NaN NaN 3 NaN], 'SSCB', 'Name', lay.Name));
    flexNet = initialize(flexNet);
    [okRect, msgRect]   = tryPredict(flexNet, zeros(224, 640, 3, 1, 'single'));
    [okBatch, msgBatch] = tryPredict(flexNet, zeros(640, 640, 3, 4, 'single'));
    fprintf('Size-free network: 640x224 input %s | batch of 4 %s\n', okStr(okRect, msgRect), okStr(okBatch, msgBatch));
    % sanity: same boxes as the original network on a real frame
    [b0, s0] = sahiDetect(det, imgs{1}, sahiDefaultOpts('dual'));
    [b1, s1] = sahiDetect(struct('Network', flexNet, 'ClassNames', {names}), imgs{1}, sahiDefaultOpts('dual'));
    fprintf('  same result as original network: %d boxes vs %d, max box diff %.3g px\n', ...
        numel(s0), numel(s1), maxDiff(b0, b1));
    detUse = struct('Network', flexNet, 'ClassNames', {names});
end
fprintf('\n');

%% 2. Set-ups to time
%            name                          mode    rect   batch  reuse
cfg = { ...
    'full-frame, square (30 Hz loop)',    'full', false, false, false
    'full-frame, rect 640x384',           'full', true,  false, false
    'v1 band, square, one call per tile', 'v1',   false, false, false
    'v1 band, square, batched',           'v1',   false, true,  false
    'v1 band, rect, one call per tile',   'v1',   true,  false, false
    'v1 band, rect, batched',             'v1',   true,  true,  false
    'v1 band, rect, batched, reuse full', 'v1',   true,  true,  true
    'dual (Pranava), square, per tile',   'dual', false, false, false
    'dual (Pranava), rect, batched',      'dual', true,  true,  false
    'fast (Pranava), square, per tile',   'fast', false, false, false
    'fast (Pranava), rect, batched',      'fast', true,  true,  false };
if ~okRect,  cfg(cell2mat(cfg(:, 3)), :) = []; end
if ~okBatch, cfg(cell2mat(cfg(:, 4)), :) = []; end

% full-frame results of the 30 Hz loop (for the "reuse" set-up)
ffOpts = sahiDefaultOpts('full');
ffRes = cell(numel(imgs), 1);
for k = 1:numel(imgs)
    [~, ~, ~, fi] = sahiDetect(detUse, imgs{k}, ffOpts);
    ffRes{k} = struct('Bboxes', fi.ffBboxes, 'Scores', fi.ffScores, 'LabelIds', fi.ffLabelIds);
end

nC = size(cfg, 1);
T = table('Size', [nC 8], 'VariableTypes', {'string','double','double','double','double','double','double','double'}, ...
    'VariableNames', {'setup','median_ms','p90_ms','fullframe_ms','tiles_ms','merge_ms','tiles','boxes'});
fprintf('%-38s %9s %8s | %9s %9s %8s | %5s %6s\n', 'set-up', 'median', 'p90', 'full-fr', 'tiles', 'merge', 'tiles', 'boxes');
for c = 1:nC
    o = sahiDefaultOpts(cfg{c, 2});
    o.RectInput = cfg{c, 3}; o.BatchTiles = cfg{c, 4};
    tt = []; tf = []; tl = []; tm = []; nb = 0; nt = 0;
    for k = 1:numel(imgs)
        ok = o;
        if cfg{c, 5}, ok.FullFrame = ffRes{k}; end
        for w = 1:3, sahiDetect(detUse, imgs{k}, ok); end               % warm-up (new input sizes)
        for r = 1:nRep
            t0 = tic;
            [~, s, ~, info] = sahiDetect(detUse, imgs{k}, ok);
            tt(end + 1) = 1000 * toc(t0);                                %#ok<SAGROW>
            tf(end + 1) = info.tFullMs; tl(end + 1) = info.tTilesMs; tm(end + 1) = info.tMergeMs; %#ok<SAGROW>
        end
        nb = nb + numel(s); nt = info.nTiles;
    end
    T(c, :) = {cfg{c, 1}, median(tt), p90(tt), median(tf), median(tl), median(tm), nt, nb};
    fprintf('%-38s %7.1f ms %6.1f | %7.1f %9.1f %8.1f | %5d %6d\n', cfg{c, 1}, median(tt), p90(tt), ...
        median(tf), median(tl), median(tm), nt, nb);
end

%% 3. Does rect / batching change the boxes? (should be tiny: only the grey padding differs)
fprintf('\nBox agreement vs the square, one-call-per-tile version of the same mode (IoU >= 0.5, same class):\n');
for mode = ["v1" "dual"]
    base = sahiDefaultOpts(mode);
    alt = base; alt.RectInput = okRect; alt.BatchTiles = okBatch;
    nA = 0; nB = 0; nM = 0;
    for k = 1:numel(imgs)
        [bA, sA, ~, iA] = sahiDetect(detUse, imgs{k}, base);
        [bB, sB, ~, iB] = sahiDetect(detUse, imgs{k}, alt);
        m = parityMatch([bA(:, 1:2) - 0.5, bA(:, 3:4)], iA.labelIds, sA, bB, iB.labelIds, sB);
        nA = nA + m.nPy; nB = nB + m.nMat; nM = nM + m.matched;
    end
    fprintf('  %-5s square %d boxes | rect+batch %d boxes | matched %d\n', mode, nA, nB, nM);
end

%% 4. Budget
ffMs   = T.median_ms(T.setup == "full-frame, rect 640x384");
if isempty(ffMs), ffMs = T.median_ms(1); end
slowMs = T.median_ms(contains(T.setup, "reuse full"));
if isempty(slowMs), slowMs = min(T.median_ms(startsWith(T.setup, "v1"))); end
fprintf('\nBudget check (best set-ups):\n');
fprintf('  slow loop per 5 Hz tick : %.0f ms of 200 ms\n', slowMs);
fprintf('  fast loop max rate      : %.1f Hz  (%.0f ms per frame)\n', 1000 / ffMs, ffMs);
fprintf('  GPU busy per second at 30 Hz + 5 Hz : %.0f ms  (>1000 ms = does not fit)\n', 30 * ffMs + 5 * slowMs);
fprintf('  GPU busy per second at 15 Hz + 5 Hz : %.0f ms\n', 15 * ffMs + 5 * slowMs);

save(fullfile(outDir, 'B2_speed_results.mat'), 'T', 'cfg', 'okRect', 'okBatch');
writetable(T, fullfile(outDir, 'B2_speed_results.csv'));
fprintf('\nSaved %s\n', outDir);

% ---------------------------------------------------------------------
function [ok, msg] = tryPredict(net, X)
ok = true; msg = '';
try
    dlX = dlarray(X, 'SSCB');
    if canUseGPU, dlX = gpuArray(dlX); end
    o = cell(1, numel(net.OutputNames));
    [o{:}] = predict(net, dlX);
    sz = size(o{1}, 1:4);
    if sz(1) ~= size(X, 1) / 8 || sz(4) ~= size(X, 4)
        ok = false; msg = sprintf('unexpected output size %s', mat2str(sz));
    end
catch err
    ok = false; msg = err.message;
end
end

function s = okStr(ok, msg)
if ok, s = 'OK'; else, s = ['FAIL (' strtrim(msg(1:min(end, 120))) ')']; end
end

function d = maxDiff(a, b)
if ~isequal(size(a), size(b)), d = Inf; else, d = max(abs(a(:) - b(:)), [], 'omitnan'); end
if isempty(d), d = 0; end
end

function v = p90(x)
x = sort(x(:)); v = x(max(1, ceil(0.9 * numel(x))));
end
