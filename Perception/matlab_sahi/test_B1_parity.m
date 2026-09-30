%% A3 SAHI - Block 1 parity test: sahiDetect (MATLAB) vs sahi_engine.py (Python)
% SIH PS26037 · Team Epsilon · Perception
%
% Runs sahiDetect on Pranava's 4 test frames in 'dual' and 'fast' mode and matches
% every box one-to-one against the Python reference (same class, IoU >= 0.5).
% Target: 64 full-frame -> 94 SAHI, +23 new (dual); +18 new (fast); all boxes matched.
%
% NEEDS  D:\SIH\share\C3_detector_v1          (detector package)
%        D:\SIH\work\A3_sahi\py_reference\py_reference.mat   (made by make_py_reference.py)
% WRITES D:\SIH\work\A3_sahi\B1_parity\  (log, results .mat, comparison PNG)

clear; clc;
here    = fileparts(mfilename('fullpath'));
pkg     = 'D:\SIH\share\C3_detector_v1';
refFile = 'D:\SIH\work\A3_sahi\py_reference\py_reference.mat';
outDir  = 'D:\SIH\work\A3_sahi\B1_parity';
if ~isfolder(outDir), mkdir(outDir); end
addpath(pkg); addpath(here);

logFile = fullfile(outDir, 'B1_parity_log.txt');
if isfile(logFile), delete(logFile); end
diary(logFile); cleanupDiary = onCleanup(@() diary('off'));

fprintf('A3 SAHI Block 1 parity  (%s, MATLAB %s)\n\n', datestr(now), version('-release'));
det = load_c3_detector();
ref = load(refFile);
net = det.Network;

%% 0. Network I/O check (what predict returns for one 640x640 image)
fprintf('GPU usable: %d\n', canUseGPU);
fprintf('Input layer: %s  %s\n', class(net.Layers(1)), mat2str(net.Layers(1).InputSize));
if isprop(net.Layers(1), 'Normalization')
    fprintf('Input normalization: %s\n', string(net.Layers(1).Normalization));
end
X = dlarray(zeros(640, 640, 3, 'single'), 'SSCB');
o = cell(1, numel(net.OutputNames));
[o{:}] = predict(net, X);
for k = 1:numel(o)
    fprintf('Output %d  %-10s size %-16s format %s\n', k, net.OutputNames{k}, mat2str(size(o{k})), dims(o{k}));
end
fprintf('\n');

%% 1. Warm-up (first GPU calls are slow)
I0 = imread(fullfile(pkg, 'test_images', strtrim(ref.dual{1}.name)));
for w = 1:2, sahiDetect(det, I0, sahiDefaultOpts('dual')); end

%% 2. Parity
modes = {'dual', 'fast'};
results = struct();
for m = 1:numel(modes)
    mode = modes{m};
    opts = sahiDefaultOpts(mode);
    R = ref.(mode);
    fprintf('==== mode: %s ====\n', mode);
    fprintf('%-12s | full-frame py/mat/matched | SAHI py/mat/matched | new py/mat | min IoU | max dScore | tiles | detect() count\n', 'frame');
    tot = zeros(1, 8);
    for k = 1:numel(R)
        r = R{k};
        nm = strtrim(r.name);
        I = imread(fullfile(pkg, 'test_images', nm));
        [bb, sc, lb, info] = sahiDetect(det, I, opts);
        sf = parityMatch(r.ff_xywh, r.ff_cls, r.ff_scores, info.ffBboxes, info.ffLabelIds, info.ffScores);
        ss = parityMatch(r.sahi_xywh, r.sahi_cls, r.sahi_scores, bb, info.labelIds, sc);
        nDet = size(detect(det, I, Threshold = 0.25), 1);        % add-on detect(), for reference only
        fprintf('%-12s |        %2d / %2d / %2d        |    %2d / %2d / %2d     |   %2d / %2d  | %.4f  | %.2g   |  %2d   | %2d\n', ...
            nm(1:12), sf.nPy, sf.nMat, sf.matched, ss.nPy, ss.nMat, ss.matched, ...
            sum(r.is_new), sum(info.isNew), ss.minIoU, ss.maxScoreDiff, info.nTiles, nDet);
        for u = ss.unmatchedPy.'
            fprintf('      python-only : %-14s %.3f  [%s]\n', string(det.ClassNames(r.sahi_cls(u))), r.sahi_scores(u), num2str(r.sahi_xywh(u, :), '%7.1f'));
        end
        for u = ss.unmatchedMat.'
            fprintf('      matlab-only : %-14s %.3f  [%s]\n', string(lb(u)), sc(u), num2str(bb(u, :), '%7.1f'));
        end
        tot = tot + [sf.nPy sf.nMat sf.matched ss.nPy ss.nMat ss.matched sum(r.is_new) sum(info.isNew)];
        results.(mode)(k) = struct('name', nm, 'bboxes', bb, 'scores', sc, 'labels', lb, 'info', info, ...
                                   'ffMatch', sf, 'sahiMatch', ss);
    end
    fprintf('TOTAL        |        %2d / %2d / %2d        |    %2d / %2d / %2d     |   %2d / %2d\n\n', tot);
    results.([mode '_total']) = tot;
end
save(fullfile(outDir, 'B1_parity_results.mat'), 'results');

%% 3. Picture: full-frame vs SAHI on the dense Bangalore frame (new objects in cyan)
k = find(cellfun(@(r) contains(r.name, 'highquality_16k'), ref.dual), 1);
rr = results.dual(k);
I = imread(fullfile(pkg, 'test_images', rr.name));
A = insertShape(I, 'rectangle', rr.info.ffBboxes, 'LineWidth', 3, 'ShapeColor', 'yellow');
A = insertText(A, [20 20], sprintf('Full frame: %d', size(rr.info.ffBboxes, 1)), 'FontSize', 40);
Bimg = insertShape(I, 'rectangle', rr.info.tiles, 'LineWidth', 2, 'ShapeColor', 'white');
if any(~rr.info.isNew)
    Bimg = insertShape(Bimg, 'rectangle', rr.bboxes(~rr.info.isNew, :), 'LineWidth', 3, 'ShapeColor', 'yellow');
end
if any(rr.info.isNew)
    Bimg = insertShape(Bimg, 'rectangle', rr.bboxes(rr.info.isNew, :), 'LineWidth', 4, 'ShapeColor', 'cyan');
end
Bimg = insertText(Bimg, [20 20], sprintf('SAHI (MATLAB): %d, new %d (cyan), %d tiles', ...
    numel(rr.scores), sum(rr.info.isNew), rr.info.nTiles), 'FontSize', 40);
imwrite([A; Bimg], fullfile(outDir, 'B1_highquality16k_ff_vs_sahi.png'));
fprintf('Saved %s\n', outDir);
