%% C2 - Road segmentation on IDD Lite (DeepLab v3+ / ResNet-18)
% SIH PS26037 · Team Epsilon · Perception (camera)
%
% WHAT THIS DOES
%   1. Checks the IDD Lite label values before anything else.
%   2. Trains a small DeepLab v3+ to label every pixel (drivable road, non-drivable, ...).
%   3. Scores it on the validation set (IoU per class, mIoU, ms per image).
%   4. Saves slide figures: ground truth vs our model, and drivable area highlighted.
%
% HOW TO RUN
%   Step 1: leave quickTest = true -> trains 1 epoch, tells you how long a full run takes.
%   Step 2: set quickTest = false  -> full training (30 epochs), then evaluation + figures.
%
% NEEDS (Home > Add-Ons): "Deep Learning Toolbox Model for ResNet-18 Network"
% If any line errors, send me the full red error text.

clear; clc; close all;

%% ---------------- SETTINGS ----------------
quickTest  = false;     % true = 1-epoch timing test, false = full training
nEpochs    = 30;
batchSize  = 8;        % lower to 4 if you get "out of memory" on the GPU
inputSize  = [224 320];% training size: close to the original 227x320, both divisible by 32 (DeepLab needs that)

root    = "D:\SIH\Dataset\IDD Lite\idd-lite\idd20k_lite";
workDir = "D:\SIH\work\C2_seg";
ckptDir = fullfile(workDir, "checkpoints");
if ~isfolder(ckptDir), mkdir(ckptDir); end

classNames = ["drivable" "non_drivable" "living_things" "vehicles" ...
              "roadside_objects" "far_objects" "sky"];
labelIDs   = 0:6;                 % 255 = unlabelled -> ignored
numClasses = numel(classNames);

%% ---------------- 1. PAIR IMAGES WITH LABELS ----------------
[trainImgs, trainLbls] = pairFiles(root, "train");
[valImgs,   valLbls]   = pairFiles(root, "val");
fprintf("Train pairs: %d | Val pairs: %d\n", numel(trainImgs), numel(valImgs));

%% ---------------- 2. CHECK LABEL VALUES (don't assume) ----------------
vals = [];
for k = round(linspace(1, numel(trainLbls), min(60, numel(trainLbls))))
    L = imread(trainLbls(k));
    vals = union(vals, unique(L(:)));
end
fprintf("Label values found in a sample: %s\n", mat2str(double(vals')));
unexpected = setdiff(double(vals), [labelIDs 255]);
assert(isempty(unexpected), ...
    "Unexpected label values %s. STOP and check the IDD Lite label mapping.", mat2str(unexpected));
I0 = imread(trainImgs(1));
fprintf("Original image size: %d x %d\n", size(I0,1), size(I0,2));

%% ---------------- 3. DATASTORES ----------------
imdsTrain = imageDatastore(cellstr(trainImgs));
pxdsTrain = pixelLabelDatastore(cellstr(trainLbls), classNames, labelIDs);
imdsVal   = imageDatastore(cellstr(valImgs));
pxdsVal   = pixelLabelDatastore(cellstr(valLbls), classNames, labelIDs);

dsTrain = transform(combine(imdsTrain, pxdsTrain), @(d) prep(d, inputSize, true));
dsVal   = transform(combine(imdsVal,   pxdsVal),   @(d) prep(d, inputSize, false));

% Class weights: rare classes (living things, vehicles) count for more
tbl  = countEachLabel(pxdsTrain);
freq = tbl.PixelCount ./ max(tbl.ImagePixelCount, 1);
w    = median(freq(freq > 0)) ./ max(freq, eps);
w(freq == 0) = 0;
classWeights = w';
disp(table(tbl.Name, tbl.PixelCount, w, VariableNames=["class" "pixels" "weight"]));

%% ---------------- 4. NETWORK ----------------
try
    net = deeplabv3plus([inputSize 3], numClasses, "resnet18");
catch ME
    fprintf("deeplabv3plus not available (%s). Using deeplabv3plusLayers.\n", ME.message);
    lg  = deeplabv3plusLayers([inputSize 3], numClasses, "resnet18");
    lg  = removeLayers(lg, lg.Layers(end).Name);   % drop old classification layer
    net = dlnetwork(lg);
end

%% ---------------- 5. TRAIN ----------------
epochs = nEpochs; if quickTest, epochs = 1; end
itersPerEpoch = floor(numel(trainImgs) / batchSize);

opts = trainingOptions("adam", ...
    InitialLearnRate    = 1e-3, ...
    LearnRateSchedule   = "piecewise", ...
    LearnRateDropPeriod = 10, ...
    LearnRateDropFactor = 0.3, ...
    MaxEpochs           = epochs, ...
    MiniBatchSize       = batchSize, ...
    Shuffle             = "every-epoch", ...
    ValidationData      = dsVal, ...
    ValidationFrequency = itersPerEpoch, ...
    CheckpointPath      = ckptDir, ...
    Plots               = "training-progress", ...
    Verbose             = true);

lossFcn = @(Y, T) weightedLoss(Y, T, classWeights);

fprintf("GPU available: %d\n", canUseGPU());
tStart = tic;
net = trainnet(dsTrain, net, lossFcn, opts);
tTrain = toc(tStart);
fprintf("\nTraining took %.1f min for %d epoch(s).\n", tTrain/60, epochs);

if quickTest
    fprintf("=> Estimated full run (%d epochs): about %.0f min.\n", nEpochs, tTrain/60*nEpochs);
    fprintf("If that fits, set quickTest = false and run again.\n");
    return
end
save(fullfile(workDir, "c2_deeplab_iddlite.mat"), "net", "classNames", "labelIDs", "inputSize");

%% ---------------- 6. EVALUATE ON VAL ----------------
conf = zeros(numClasses);
msPer = zeros(numel(valImgs), 1);
for k = 1:numel(valImgs)
    I  = imread(valImgs(k));  if size(I,3) == 1, I = repmat(I, [1 1 3]); end
    gt = imread(valLbls(k));
    [pred, msPer(k)] = segmentOne(net, I, inputSize);
    ok = gt ~= 255 & gt <= 6;
    conf = conf + accumarray([double(gt(ok))+1, double(pred(ok))+1], 1, [numClasses numClasses]);
end
tp  = diag(conf);
iou = tp ./ max(sum(conf,1)' + sum(conf,2) - tp, 1);
res = table(classNames', round(iou*100, 1), VariableNames=["class" "IoU_percent"]);
disp(res);
fprintf("mIoU: %.1f %% | pixel accuracy: %.1f %% | %.1f ms per image (after warm-up)\n", ...
    mean(iou)*100, sum(tp)/sum(conf(:))*100, mean(msPer(2:end)));
writetable(res, fullfile(workDir, "c2_iou_per_class.csv"));

%% ---------------- 7. SLIDE FIGURES ----------------
pick = round(linspace(1, numel(valImgs), 4));
figure(Color="w", Position=[50 50 1500 900]);
t = tiledlayout(3, 4, TileSpacing="compact", Padding="compact");
for j = 1:4
    I  = imread(valImgs(pick(j)));  if size(I,3) == 1, I = repmat(I, [1 1 3]); end
    gt = imread(valLbls(pick(j)));  gt(gt == 255) = 0;   % unlabelled drawn as class 0 only for display
    pred = segmentOne(net, I, inputSize);
    cmap = lines(numClasses);
    nexttile(j);     imshow(labeloverlay(I, categorical(gt,   labelIDs, classNames), Colormap=cmap, Transparency=0.45)); title("Ground truth");
    nexttile(j+4);   imshow(labeloverlay(I, categorical(pred, labelIDs, classNames), Colormap=cmap, Transparency=0.45)); title("Our model");
    nexttile(j+8);   imshow(labeloverlay(I, pred == 0, Colormap=[0 0 0; 0 0.8 0.3], Transparency=0.5)); title("Drivable area (ours)");
end
title(t, sprintf("Road segmentation on IDD Lite (Indian roads) - mIoU %.1f %%", mean(iou)*100));
exportgraphics(gcf, fullfile(workDir, "c2_segmentation_figure.png"), Resolution=200);
fprintf("Done. Figures and scores are in %s\n", workDir);

%% ================= HELPERS =================
function [imgs, lbls] = pairFiles(root, split)
    imgs = strings(0,1); lbls = strings(0,1);
    L = dir(fullfile(root, "gtFine", split, "**", "*_label.png"));
    L = L(~contains({L.name}, "_inst_label"));
    for k = 1:numel(L)
        id  = erase(L(k).name, "_label.png");
        [~, seq] = fileparts(L(k).folder);
        base = fullfile(root, "leftImg8bit", split, seq, id + "_image");
        cand = base + [".jpg" ".png" ".jpeg"];
        hit  = cand(isfile(cand));
        if ~isempty(hit)
            imgs(end+1,1) = hit(1);                                   %#ok<AGROW>
            lbls(end+1,1) = string(fullfile(L(k).folder, L(k).name)); %#ok<AGROW>
        end
    end
    assert(~isempty(imgs), "No image/label pairs found for '%s'. Check the root path.", split);
end

function out = prep(data, sz, augment)
    out = data;
    for i = 1:size(data, 1)
        I = data{i,1}; C = data{i,2};
        if size(I,3) == 1, I = repmat(I, [1 1 3]); end
        I = imresize(I, sz, "bilinear");
        C = imresize(C, sz, "nearest");
        if augment && rand > 0.5, I = fliplr(I); C = fliplr(C); end
        out(i,:) = {I, C};
    end
end

function loss = weightedLoss(Y, T, classWeights)
    weights = dlarray(classWeights, "C");
    mask = ~isnan(T);
    T(isnan(T)) = 0;
    loss = crossentropy(Y, T, weights, Mask=mask, NormalizationFactor="mask-included");
end

function [ids, ms] = segmentOne(net, I, sz)
    Ir = imresize(I, sz, "bilinear");
    t = tic;
    scores = minibatchpredict(net, single(Ir));
    ms = toc(t) * 1000;
    [~, idx] = max(scores, [], 3);
    ids = uint8(imresize(idx - 1, [size(I,1) size(I,2)], "nearest"));
end