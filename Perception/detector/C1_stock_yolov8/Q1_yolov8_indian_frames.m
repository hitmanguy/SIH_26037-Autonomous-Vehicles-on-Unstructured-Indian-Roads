%% Q1 - Stock YOLOv8 (COCO) on Indian road frames   [v2]
% SIH PS26037 · Team Epsilon · Perception (camera) · checkpoint C1
%
% WHAT THIS DOES
%   1. Runs the stock, COCO-trained YOLOv8 on 15 Indian road photos (no training).
%   2. Saves each photo with boxes + a readable 2x2 slide figure + a full montage.
%   3. Maps COCO classes to our roles and PRINTS every label actually found.
%   4. Times each model size (n / s / m): full detect() call AND network alone.
%
% v2 fixes: "motorbike" label mapped, readable figures, robust timing.
%
% Images : D:\SIH\Dataset\Q1_Frames   (15 IDD Detection val images)
% Results: D:\SIH\work\C1_Q1

clear; clc; close all;

%% Settings
imgDir     = "D:\SIH\Dataset\Q1_Frames";
outDir     = "D:\SIH\work\C1_Q1";
mainModel  = "yolov8s";
timeModels = ["yolov8n" "yolov8s" "yolov8m"];
minScore   = 0.25;
nWarm      = 5;     % warm-up runs before timing
nRuns      = 20;    % timed runs per model

if ~isfolder(outDir), mkdir(outDir); end

%% Load images
files = [dir(fullfile(imgDir, "*.jpg")); dir(fullfile(imgDir, "*.jpeg")); dir(fullfile(imgDir, "*.png"))];
assert(~isempty(files), "No images found in %s", imgDir);
fprintf("Found %d images. GPU available to MATLAB: %d\n", numel(files), canUseGPU());

%% COCO class -> our role (both spellings of motorbike/motorcycle included)
cocoNames = ["person" "bicycle" "motorcycle" "motorbike" "car" "bus" "truck" "train" ...
             "cow" "dog" "horse" "sheep" "elephant" "cat" "bird" ...
             "traffic light" "stop sign"];
roleNames = ["VRU" "two-wheeler" "two-wheeler" "two-wheeler" "vehicle" "vehicle" "vehicle" "vehicle" ...
             "animal" "animal" "animal" "animal" "animal" "animal" "animal" ...
             "sign" "sign"];
roleMap = dictionary(cocoNames, roleNames);

%% Run the main model on every image
det = yolov8ObjectDetector(mainModel);
I0  = imread(fullfile(files(1).folder, files(1).name));
for w = 1:nWarm, detect(det, I0); end          % warm-up (GPU needs a few runs)

rows = {}; annotated = cell(numel(files), 1); nObj = zeros(numel(files), 1);
allLabels = strings(0, 1);
for k = 1:numel(files)
    I = imread(fullfile(files(k).folder, files(k).name));
    if size(I,3) == 1, I = repmat(I, [1 1 3]); end

    t = tic;
    [bboxes, scores, labels] = detect(det, I);
    ms = toc(t) * 1000;

    keep   = scores >= minScore;
    bboxes = bboxes(keep, :); scores = scores(keep); labels = string(labels(keep));
    allLabels = [allLabels; labels(:)]; %#ok<AGROW>

    roles = strings(size(labels));
    for j = 1:numel(labels)
        if isKey(roleMap, labels(j)), roles(j) = roleMap(labels(j)); else, roles(j) = "other"; end
    end

    % Labels sized to the image so they survive shrinking onto a slide
    fs = max(14, round(size(I,2) / 55));
    lw = max(3,  round(size(I,2) / 320));
    if isempty(bboxes)
        J = I;
    else
        J = insertObjectAnnotation(I, "rectangle", bboxes, labels + " " + compose("%.2f", scores), ...
            LineWidth=lw, FontSize=min(fs, 72), TextBoxOpacity=0.8);
    end
    annotated{k} = J; nObj(k) = numel(labels);
    [~, base] = fileparts(files(k).name);
    imwrite(J, fullfile(outDir, base + "_yolov8.png"));

    rows(end+1, :) = {files(k).name, numel(labels), sum(roles=="vehicle"), sum(roles=="two-wheeler"), ...
        sum(roles=="VRU"), sum(roles=="animal"), sum(roles=="sign"), sum(roles=="other"), ms, ""}; %#ok<SAGROW>
end

log = cell2table(rows, VariableNames=["image" "n_objects" "vehicle" "two_wheeler" "VRU" ...
    "animal" "sign" "other" "ms_per_frame" "missed_or_wrong_FILL_IN"]);
writetable(log, fullfile(outDir, "q1_detections.csv"));
disp(log(:, 2:9));

% Every label the stock model actually used, with counts
[u, ~, ic] = unique(allLabels);
fprintf("\nLabels found by stock YOLOv8 (COCO):\n");
disp(table(u, accumarray(ic, 1), VariableNames=["label" "count"]));

%% Slide figure: the 4 busiest images, large and readable
[~, order] = sort(nObj, "descend");
show = order(1:min(4, numel(order)));
f1 = figure(Color="w", Position=[40 40 1600 950]);
try, f1.Theme = "light"; catch, end
t1 = tiledlayout(f1, 2, 2, TileSpacing="compact", Padding="compact");
for j = 1:numel(show)
    nexttile(t1); imshow(annotated{show(j)});
end
title(t1, "Stock YOLOv8s (COCO, no training) on Indian roads: no class for autorickshaw, rider or Indian vehicles", ...
    Color="k", FontWeight="bold", FontSize=14);
exportgraphics(f1, fullfile(outDir, "q1_slide_2x2.png"), Resolution=200, BackgroundColor="white");

%% Full montage (all 15, for reference)
f2 = figure(Color="w", Position=[60 60 1600 1000]);
try, f2.Theme = "light"; catch, end
montage(annotated, Size=[NaN 4], BorderSize=[4 4], BackgroundColor="w");
title("Stock YOLOv8s (COCO, no training) on 15 Indian road images", Color="k", FontSize=14);
exportgraphics(f2, fullfile(outDir, "q1_montage.png"), Resolution=200, BackgroundColor="white");

%% Timing per model size: full detect() and network alone
timing = table('Size', [0 5], 'VariableTypes', ["string" "double" "double" "double" "double"], ...
    'VariableNames', ["model" "detect_ms" "detect_fps" "network_ms" "network_fps"]);
for m = 1:numel(timeModels)
    try
        d = yolov8ObjectDetector(timeModels(m));
    catch ME
        fprintf("\nCould not load %s (%s). Retrying once...\n", timeModels(m), ME.message);
        try
            d = yolov8ObjectDetector(timeModels(m));
        catch ME2
            fprintf("Skipping %s: %s\n", timeModels(m), ME2.message);
            continue
        end
    end

    % Full detect() on a real full-HD image (includes resizing + box clean-up)
    for w = 1:nWarm, detect(d, I0); end
    td = zeros(nRuns, 1);
    for r = 1:nRuns, s = tic; detect(d, I0); td(r) = toc(s) * 1000; end

    % Network alone, on a 640x640 input already on the GPU
    netMs = NaN;
    try
        X = dlarray(gpuArray(single(rand(640, 640, 3))), "SSCB");
        for w = 1:nWarm, predict(d.Network, X); end
        wait(gpuDevice);
        tn = zeros(nRuns, 1);
        for r = 1:nRuns
            s = tic; predict(d.Network, X); wait(gpuDevice); tn(r) = toc(s) * 1000;
        end
        netMs = median(tn);
    catch ME
        fprintf("Network-only timing skipped for %s: %s\n", timeModels(m), ME.message);
    end

    timing(end+1, :) = {timeModels(m), median(td), 1000/median(td), netMs, 1000/netMs}; %#ok<SAGROW>
end
fprintf("\nTiming on this laptop (median of %d runs):\n", nRuns);
disp(timing);
writetable(timing, fullfile(outDir, "q1_timing.csv"));
fprintf("Budget reminder: 4 cameras at 30 Hz = 120 frames/s = ~8 ms per frame.\n");
fprintf("Done. Outputs are in: %s\n", outDir);