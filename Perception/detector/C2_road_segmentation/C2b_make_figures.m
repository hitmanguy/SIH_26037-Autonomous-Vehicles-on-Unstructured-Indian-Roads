%% C2b - Regenerate C2 slide figures from the saved model (no retraining)
% SIH PS26037 · Team Epsilon · Perception (camera)
% Fixes: green drivable overlay, readable titles, unlabelled pixels left uncoloured.
% Outputs (in D:\SIH\work\C2_seg):
%   c2_fig_7class.png  - ground truth vs our model, all 7 classes, with legend
%   c2_fig_drivable.png - camera view vs drivable road (green) - simplest for the slide

clear; clc; close all;

root    = "D:\SIH\Dataset\IDD Lite\idd-lite\idd20k_lite";
workDir = "D:\SIH\work\C2_seg";

S = load(fullfile(workDir, "c2_deeplab_iddlite.mat"));
net = S.net; classNames = S.classNames; labelIDs = S.labelIDs; inputSize = S.inputSize;
nice = strrep(classNames, "_", " ");

[valImgs, valLbls] = pairFiles(root, "val");
pick = round(linspace(1, numel(valImgs), 4));   % same 4 images as before
cmap = lines(numel(classNames));

%% Figure A - 7 classes: ground truth vs our model
fA = figure(Color="w", Position=[50 50 1500 800]);
try, fA.Theme = "light"; catch, end
tA = tiledlayout(fA, 2, 4, TileSpacing="compact", Padding="compact");
for j = 1:4
    I  = imread(valImgs(pick(j)));  if size(I,3) == 1, I = repmat(I, [1 1 3]); end
    gt = imread(valLbls(pick(j)));                              % 255 stays undefined = uncoloured
    pred = segmentOne(net, I, inputSize);

    nexttile(tA, j);
    imshow(labeloverlay(I, categorical(gt, labelIDs, classNames), Colormap=cmap, Transparency=0.45));
    title("Ground truth", Color="k", FontSize=12);

    nexttile(tA, j+4);
    imshow(labeloverlay(I, categorical(pred, labelIDs, classNames), Colormap=cmap, Transparency=0.45));
    title("Our model (MATLAB, DeepLab v3+)", Color="k", FontSize=12);
end
ax = nexttile(tA, 8); hold(ax, "on");
h = gobjects(numel(classNames), 1);
for c = 1:numel(classNames)
    h(c) = patch(ax, NaN, NaN, cmap(c,:), EdgeColor="none");
end
lgd = legend(h, nice, Orientation="horizontal", TextColor="k", Color="w", FontSize=11);
lgd.Layout.Tile = "south";
title(tA, "Road segmentation on Indian roads (IDD Lite) - mIoU 59.2 %, drivable IoU 88.6 %", ...
    Color="k", FontWeight="bold", FontSize=14);
exportgraphics(fA, fullfile(workDir, "c2_fig_7class.png"), Resolution=200, BackgroundColor="white");

%% Figure B - drivable road only (clearest for a slide)
fB = figure(Color="w", Position=[80 80 1500 650]);
try, fB.Theme = "light"; catch, end
tB = tiledlayout(fB, 2, 4, TileSpacing="compact", Padding="compact");
for j = 1:4
    I = imread(valImgs(pick(j)));  if size(I,3) == 1, I = repmat(I, [1 1 3]); end
    pred = segmentOne(net, I, inputSize);

    nexttile(tB, j);   imshow(I);
    title("Camera view", Color="k", FontSize=12);

    nexttile(tB, j+4); imshow(labeloverlay(I, pred == 0, Colormap=[0 0.8 0.3], Transparency=0.45));
    title("Drivable road found by our model", Color="k", FontSize=12);
end
title(tB, "Where can the car drive? Drivable-area IoU 88.6 % at 52.7 ms/image (laptop RTX 4050)", ...
    Color="k", FontWeight="bold", FontSize=14);
exportgraphics(fB, fullfile(workDir, "c2_fig_drivable.png"), Resolution=200, BackgroundColor="white");

fprintf("Saved c2_fig_7class.png and c2_fig_drivable.png in %s\n", workDir);

%% ================= HELPERS (same as C2) =================
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
end

function [ids, ms] = segmentOne(net, I, sz)
    Ir = imresize(I, sz, "bilinear");
    t = tic;
    scores = minibatchpredict(net, single(Ir));
    ms = toc(t) * 1000;
    [~, idx] = max(scores, [], 3);
    ids = uint8(imresize(idx - 1, [size(I,1) size(I,2)], "nearest"));
end
