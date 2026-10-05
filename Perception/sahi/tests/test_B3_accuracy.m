%% A3 SAHI - Block 3: accuracy on labelled IDD val frames (FRONT cameras only)
% SIH PS26037 · Team Epsilon · Perception
%
% Compares full-frame vs SAHI set-ups against IDD ground truth:
%   - recall by object size (box height scaled to 1080p), road users and signs/lights separately
%   - false detections per frame (all, and small ones) and precision
%   - AP50 per class and mAP50
% A detection counts as correct if same class and IoU >= 0.5 with an unmatched GT box.
%
% NEEDS  D:\SIH\Dataset\IDD_val_yolo\images\val, labels\val   (YOLO txt, 12 classes)
% WRITES D:\SIH\work\A3_sahi\B3_accuracy\   (log, tables CSV, results .mat, example PNGs)

clear; clc;
here    = fileparts(mfilename('fullpath'));
pkg     = fullfile(here, '..', '..', 'C3_detector_v1');   % detector package in this repo
dataDir = 'D:\SIH\Dataset\IDD_val_yolo';
outDir  = 'D:\SIH\work\A3_sahi\B3_accuracy';
if ~isfolder(outDir), mkdir(outDir); end
addpath(pkg); addpath(here, fullfile(here, ".."), fullfile(here, "..", "simulink"));   % tests/ -> sahi/ core + simulink/
logFile = fullfile(outDir, 'B3_accuracy_log.txt');
if isfile(logFile), delete(logFile); end
diary(logFile); cleanupDiary = onCleanup(@() diary('off'));

MaxImages = Inf;          % set e.g. 100 for a quick trial run
IoUThr    = 0.5;

% Band rows per IDD front camera (1080p rows, rescaled for 720p frames).
% From TRAIN labels: rows holding 90% of small (<32 px) road users (5th pct top ..
% 95th pct bottom), +/-10 px margin. Not tuned on val, so no cheating.
rowsByCam.frontFar        = [428 616];
rowsByCam.frontNear       = [439 680];
rowsByCam.highquality_16k = [365 652];

fprintf('A3 SAHI Block 3 accuracy  (%s, MATLAB %s)\n\n', datestr(now), version('-release'));
det = load_c3_detector();
classNames = cellstr(string(det.ClassNames));
roadUsers = [1:9 12];     % person..animal + vehicle fallback (1-based ids)
signs     = [10 11];      % traffic sign, traffic light

%% 1. Front-camera frames
files = dir(fullfile(dataDir, 'images', 'val', '*.jpg'));
names = {files.name};
cam = regexprep(names, '__.*', '');
isFront = ismember(cam, fieldnames(rowsByCam)) & ~contains(lower(names), {'side', 'rear'});
files = files(isFront); cam = cam(isFront);
if numel(files) > MaxImages, files = files(1:MaxImages); cam = cam(1:MaxImages); end
nI = numel(files);
fprintf('Front-camera val frames: %d  (', nI);
uc = unique(cam); for u = 1:numel(uc), fprintf('%s %d  ', uc{u}, sum(strcmp(cam, uc{u}))); end
fprintf(')\n\n');

%% 2. Set-ups
%        name                               mode    rows           rect
M = { 'full-frame only',                   'full', '',            false
      'v1 placeholder rows, 640x640 tiles','v1',   'placeholder', false
      'v1 placeholder rows, 640xN tiles',  'v1',   'placeholder', true
      'per-camera rows, 640x640 tiles',    'v1',   'camera',      false
      'per-camera rows, 640xN tiles',      'v1',   'camera',      true
      'Pranava dual bands (8 tiles)',      'dual', '',            false };
nM = size(M, 1);
D = cell(nM, nI);          % per set-up, per image: struct B (xywh), C, S
G = cell(1, nI);           % ground truth per image: struct B, C, H (image height)
tSahi = nan(nM, nI);

%% 3. Run
t0 = tic;
for k = 1:nI
    I = imread(fullfile(files(k).folder, files(k).name));
    if size(I, 3) == 1, I = repmat(I, 1, 1, 3); end
    [H, W, ~] = size(I);
    G{k} = readYolo(fullfile(dataDir, 'labels', 'val', strrep(files(k).name, '.jpg', '.txt')), W, H);

    [~, ~, ~, fi] = sahiDetect(det, I, sahiDefaultOpts('full'));
    ff = struct('Bboxes', fi.ffBboxes, 'Scores', fi.ffScores, 'LabelIds', fi.ffLabelIds);
    for m = 1:nM
        if strcmp(M{m, 2}, 'full')
            D{m, k} = struct('B', fi.ffBboxes, 'C', fi.ffLabelIds, 'S', double(fi.ffScores));
            tSahi(m, k) = fi.tFullMs;
            continue
        end
        o = sahiDefaultOpts(M{m, 2});
        o.FullFrame = ff; o.BatchTiles = true; o.RectInput = M{m, 4};
        if strcmp(M{m, 3}, 'camera')
            o.BandRows = rowsByCam.(cam{k}); o.BandRowsImageHeight = 1080;
        end
        [b, s, ~, info] = sahiDetect(det, I, o);
        D{m, k} = struct('B', b, 'C', info.labelIds, 'S', double(s));
        tSahi(m, k) = info.tSahiMs;
    end
    if mod(k, 50) == 0 || k == nI
        fprintf('  %4d / %d frames  (%.1f min)\n', k, nI, toc(t0) / 60);
    end
end
save(fullfile(outDir, 'B3_raw_detections.mat'), 'D', 'G', 'M', 'files', 'cam', 'rowsByCam', '-v7.3');

%% 4. Metrics
edges = [0 16 32 64 128 Inf];  bucket = {'<16', '16-32', '32-64', '64-128', '>=128'};
nB = numel(bucket);
recRU = zeros(nM, nB); recSG = zeros(nM, nB); nRU = zeros(1, nB); nSG = zeros(1, nB);
fpAll = zeros(nM, 1); fpSmall = zeros(nM, 1); nDet = zeros(nM, 1); nTP = zeros(nM, 1);
apCls = nan(nM, 12);
camTypes = fieldnames(rowsByCam);
recSmallCam = nan(nM, numel(camTypes)); nSmallCam = zeros(1, numel(camTypes));
hitRU = cell(nM, 1);        % per set-up: logical over all small road-user GT (for examples)

for m = 1:nM
    allS = cell(1, 12); allT = cell(1, 12); nGTc = zeros(1, 12);
    hitsRU = zeros(1, nB); hitsSG = zeros(1, nB); hitsCam = zeros(1, numel(camTypes));
    smallHit = [];
    for k = 1:nI
        g = G{k}; d = D{m, k};
        [tp, gh] = evalMatch(d.B, d.C, d.S, g.B, g.C, IoUThr);
        hG = g.B(:, 4) * 1080 / g.H;             % GT heights scaled to 1080p
        hD = d.B(:, 4) * 1080 / g.H;
        bG = discretize(hG, edges);
        isRU = ismember(g.C, roadUsers); isSG = ismember(g.C, signs);
        for b = 1:nB
            hitsRU(b) = hitsRU(b) + sum(gh & isRU & bG == b);
            hitsSG(b) = hitsSG(b) + sum(gh & isSG & bG == b);
            if m == 1
                nRU(b) = nRU(b) + sum(isRU & bG == b);
                nSG(b) = nSG(b) + sum(isSG & bG == b);
            end
        end
        sm = isRU & hG < 32;
        ci = find(strcmp(camTypes, cam{k}));
        hitsCam(ci) = hitsCam(ci) + sum(gh & sm);
        if m == 1, nSmallCam(ci) = nSmallCam(ci) + sum(sm); end
        smallHit = [smallHit; gh(sm)];           %#ok<AGROW>
        fpAll(m) = fpAll(m) + sum(~tp);
        fpSmall(m) = fpSmall(m) + sum(~tp & hD < 32);
        nDet(m) = nDet(m) + numel(tp); nTP(m) = nTP(m) + sum(tp);
        for c = 1:12
            allS{c} = [allS{c}; d.S(d.C == c)];  %#ok<AGROW>
            allT{c} = [allT{c}; tp(d.C == c)];   %#ok<AGROW>
            nGTc(c) = nGTc(c) + sum(g.C == c);
        end
    end
    recRU(m, :) = hitsRU ./ max(nRU, 1); recSG(m, :) = hitsSG ./ max(nSG, 1);
    recSmallCam(m, :) = hitsCam ./ max(nSmallCam, 1);
    hitRU{m} = smallHit;
    for c = 1:12, apCls(m, c) = apAllPoint(allS{c}, allT{c}, nGTc(c)); end
end

%% 5. Report
pct = @(x) 100 * x;
fprintf('\n=== Recall of ROAD USERS by box height (1080p px) — IoU >= %.1f ===\n', IoUThr);
fprintf('%-36s', 'set-up'); fprintf('%9s', bucket{:}); fprintf('\n');
fprintf('%-36s', '  (GT boxes)'); fprintf('%9d', nRU); fprintf('\n');
for m = 1:nM, fprintf('%-36s', M{m, 1}); fprintf('%8.1f%%', pct(recRU(m, :))); fprintf('\n'); end

fprintf('\n=== Recall of TRAFFIC SIGNS + LIGHTS by box height ===\n');
fprintf('%-36s', 'set-up'); fprintf('%9s', bucket{:}); fprintf('\n');
fprintf('%-36s', '  (GT boxes)'); fprintf('%9d', nSG); fprintf('\n');
for m = 1:nM, fprintf('%-36s', M{m, 1}); fprintf('%8.1f%%', pct(recSG(m, :))); fprintf('\n'); end

fprintf('\n=== False detections and precision (all classes) ===\n');
fprintf('%-36s %10s %14s %10s %8s %10s\n', 'set-up', 'FP/frame', 'small FP/frame', 'precision', 'mAP50', 'SAHI ms');
for m = 1:nM
    fprintf('%-36s %10.2f %14.2f %9.1f%% %8.3f %10.1f\n', M{m, 1}, fpAll(m) / nI, fpSmall(m) / nI, ...
        pct(nTP(m) / max(nDet(m), 1)), mean(apCls(m, :), 'omitnan'), median(tSahi(m, :), 'omitnan'));
end

fprintf('\n=== Small road-user recall (<32 px) by camera ===\n');
fprintf('%-36s', 'set-up'); fprintf('%18s', camTypes{:}); fprintf('\n');
fprintf('%-36s', '  (GT boxes)'); fprintf('%18d', nSmallCam); fprintf('\n');
for m = 1:nM, fprintf('%-36s', M{m, 1}); fprintf('%17.1f%%', pct(recSmallCam(m, :))); fprintf('\n'); end

fprintf('\n=== AP50 per class ===\n%-18s', 'class');
for m = 1:nM, fprintf('%8s', sprintf('S%d', m)); end
fprintf('\n');
for c = 1:12
    fprintf('%-18s', classNames{c}); fprintf('%8.3f', apCls(:, c)); fprintf('\n');
end
fprintf('(S1..S%d = set-ups in the order above)\n', nM);

Tsize = array2table([recRU, recSG], 'VariableNames', [strcat('RU_', bucket), strcat('SG_', bucket)], 'RowNames', M(:, 1));
writetable(Tsize, fullfile(outDir, 'B3_recall_by_size.csv'), 'WriteRowNames', true);
Tfp = table(fpAll / nI, fpSmall / nI, nTP ./ max(nDet, 1), mean(apCls, 2, 'omitnan'), median(tSahi, 2, 'omitnan'), ...
    'VariableNames', {'FP_per_frame', 'smallFP_per_frame', 'precision', 'mAP50', 'median_ms'}, 'RowNames', M(:, 1));
writetable(Tfp, fullfile(outDir, 'B3_fp_precision.csv'), 'WriteRowNames', true);
save(fullfile(outDir, 'B3_metrics.mat'), 'recRU', 'recSG', 'nRU', 'nSG', 'fpAll', 'fpSmall', 'nDet', 'nTP', ...
    'apCls', 'recSmallCam', 'nSmallCam', 'M', 'bucket', 'camTypes', 'rowsByCam', 'nI');

%% 6. Example pictures: frames where SAHI (per-camera rows, 640xN) finds most small road users full-frame missed
mBest = 5;
gain = zeros(1, nI);
for k = 1:nI
    g = G{k};
    [~, g1] = evalMatch(D{1, k}.B, D{1, k}.C, D{1, k}.S, g.B, g.C, IoUThr);
    [~, g2] = evalMatch(D{mBest, k}.B, D{mBest, k}.C, D{mBest, k}.S, g.B, g.C, IoUThr);
    sm = ismember(g.C, roadUsers) & g.B(:, 4) * 1080 / g.H < 32;
    gain(k) = sum(g2 & ~g1 & sm);
end
[~, top] = sort(gain, 'descend');
for r = 1:min(4, nI)
    k = top(r); if gain(k) == 0, break; end
    g = G{k};
    [~, g1] = evalMatch(D{1, k}.B, D{1, k}.C, D{1, k}.S, g.B, g.C, IoUThr);
    [~, g2] = evalMatch(D{mBest, k}.B, D{mBest, k}.C, D{mBest, k}.S, g.B, g.C, IoUThr);
    I = imread(fullfile(files(k).folder, files(k).name));
    rows = rowsByCam.(cam{k}) * size(I, 1) / 1080;
    I = insertShape(I, 'rectangle', [1 rows(1) size(I, 2) - 1 diff(rows)], 'LineWidth', 2, 'ShapeColor', 'white');
    if any(g1), I = insertShape(I, 'rectangle', g.B(g1, :), 'LineWidth', 2, 'ShapeColor', 'yellow'); end
    if any(g2 & ~g1), I = insertShape(I, 'rectangle', g.B(g2 & ~g1, :), 'LineWidth', 3, 'ShapeColor', 'cyan'); end
    if any(~g1 & ~g2), I = insertShape(I, 'rectangle', g.B(~g1 & ~g2, :), 'LineWidth', 2, 'ShapeColor', 'red'); end
    I = insertText(I, [10 10], sprintf('%s | yellow: full-frame found | cyan: only SAHI found (+%d small) | red: both missed', ...
        cam{k}, gain(k)), 'FontSize', 22);
    imwrite(I, fullfile(outDir, sprintf('B3_example_%d.png', r)));
end
fprintf('\nSaved %s  (%.1f min total)\n', outDir, toc(t0) / 60);

% ---------------------------------------------------------------------
function g = readYolo(f, W, H)
% YOLO txt (class cx cy w h, normalised) -> spatial [x y w h], 1-based class ids
g.B = zeros(0, 4); g.C = zeros(0, 1); g.H = H;
if ~isfile(f), return; end
a = readmatrix(f, 'FileType', 'text');
if isempty(a), return; end
a = reshape(a, [], 5);
g.C = a(:, 1) + 1;
w = a(:, 4) * W; h = a(:, 5) * H;
g.B = [a(:, 2) * W - w / 2 + 0.5, a(:, 3) * H - h / 2 + 0.5, w, h];
end
