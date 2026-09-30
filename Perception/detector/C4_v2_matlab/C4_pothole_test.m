%% C4 pothole test - show v2 potholes that detect() hides
% SIH PS26037 · Team Epsilon · Perception (camera)
%
% WHY: detect(det,...) from the YOLO add-on silently drops boxes scoring < ~0.5.
%      v2 finds potholes but scores them 0.1-0.4, so they never showed up.
%      Here we use our own decoder (sahiDetect, A3) with a per-class threshold.
%
% NEEDS: C4_import_v2.m run once (makes ../../work/C4_v2/v2_yolov8s_matlab.mat)
% OUTPUT: ../../work/C4_v2/pothole_test/<image>_full.png and _tiles.png

clear; clc; close all;
here   = fileparts(mfilename('fullpath'));
addpath(fullfile(here, "..", "A3_sahi"));
work   = fullfile(here, "..", "..", "work", "C4_v2");
outDir = fullfile(work, "pothole_test");  if ~isfolder(outDir), mkdir(outDir); end
S = load(fullfile(work, "v2_yolov8s_matlab.mat"));  det = S.det;

% ---- thresholds: everything at 0.25, potholes lower (they score low) ----
POTHOLE_CONF = 0.15;
OTHER_CONF   = 0.25;

files = [dir(fullfile(here, "test_images", "*.jpg")); dir(fullfile(here, "test_images", "*.png"))];
fprintf("%-32s %-6s %9s %9s  pothole scores\n", "image", "mode", "potholes", "others");
for k = 1:numel(files)
    I = imread(fullfile(files(k).folder, files(k).name));
    if size(I,3) == 1, I = repmat(I, 1, 1, 3); end
    [~, base] = fileparts(files(k).name);
    for mode = ["full" "fast"]                 % full = whole frame; fast = + road-band tiles
        opts = sahiDefaultOpts(char(mode));
        opts.Conf = min(POTHOLE_CONF, OTHER_CONF);
        [bb, sc, lb] = sahiDetect(det, I, opts);
        isP  = lb == "pothole";
        keep = (isP & sc >= POTHOLE_CONF) | (~isP & sc >= OTHER_CONF);
        bb = bb(keep,:); sc = sc(keep); lb = lb(keep); isP = isP(keep);

        J  = I;
        lw = max(3, round(size(I,2)/320)); fs = min(72, max(14, round(size(I,2)/55)));
        if any(~isP)
            J = insertObjectAnnotation(J, "rectangle", bb(~isP,:), ...
                string(lb(~isP)) + " " + compose("%.2f", sc(~isP)), ...
                LineWidth=lw, FontSize=fs, TextBoxOpacity=0.8, AnnotationColor="yellow");
        end
        if any(isP)
            J = insertObjectAnnotation(J, "rectangle", bb(isP,:), ...
                "pothole " + compose("%.2f", sc(isP)), ...
                LineWidth=lw, FontSize=fs, TextBoxOpacity=0.8, AnnotationColor="red");
        end
        imwrite(J, fullfile(outDir, base + "_" + mode + ".png"));
        fprintf("%-32s %-6s %9d %9d  %s\n", base, mode, nnz(isP), nnz(~isP), ...
            strjoin(compose("%.2f", sort(sc(isP), 'descend')), " "));
    end
end
fprintf("\nImages in %s  (red = pothole, yellow = rest)\n", outDir);
