%% C3b - Import our IDD-fine-tuned YOLOv8s (Ultralytics -> ONNX) into MATLAB and test it
% SIH PS26037 · Team Epsilon · Perception (camera)
%
% Route (same one MathWorks' own trainYOLOv8ObjectDetector uses):
%   server:  best.pt --(export_for_matlab.py)--> best_matlab.onnx  (3 feature-map outputs)
%   laptop:  importNetworkFromONNX(best_matlab.onnx) -> yolov8ObjectDetector(net, classNames) -> detect()
%
% NEEDS: YOLOv8 add-on (already installed) + "Deep Learning Toolbox Converter for ONNX
%        Model Format" (Add-Ons; MATLAB will prompt if missing).
% OUTPUT (D:\SIH\work\C3_matlab):
%   c3_yolov8s_idd_matlab.mat  - the detector, ready for Simulink / A1
%   *_ours_matlab.png          - our model's boxes on the 15 Q1 images
%   c3_matlab_2x2.png          - slide figure (4 images)
%   c3_matlab_timing.csv       - speed on this laptop

clear; clc; close all;

onnxPath = "D:\SIH\work\C3_yolo\c3_yolov8s_e60\weights\best_matlab.onnx";
imgDir   = "D:\SIH\Dataset\Q1_Frames";
outDir   = "D:\SIH\work\C3_matlab";
if ~isfolder(outDir), mkdir(outDir); end

% Same order as training (YOLO class ids 0..11) - order matters
classes = ["person" "rider" "car" "bus" "truck" "autorickshaw" "motorcycle" ...
           "bicycle" "animal" "traffic sign" "traffic light" "vehicle fallback"];

%% 1. Import
assert(isfile(onnxPath), "best_matlab.onnx not found at %s", onnxPath);
% best_matlab.onnx is cut by export_for_matlab.py to end at the 3 feature maps MATLAB wants,
% so we import it directly (no importYOLOv8Model trimming needed).
net = importNetworkFromONNX(onnxPath);

% Remove the batch-size check layer if the importer added one
nm = string({net.Layers.Name});
bv = nm(contains(nm, "BatchSizeVerifier"));
for b = bv
    nxt = net.Connections.Destination(string(net.Connections.Source) == b);
    prv = net.Connections.Source(contains(string(net.Connections.Destination), b));
    net = removeLayers(net, b);
    for d = string(nxt)', net = connectLayers(net, string(prv(1)), d); end
end
net = initialize(net);
fprintf("Imported network. Input size: %s\n", mat2str(net.Layers(1).InputSize));
fprintf("Outputs: %s\n", strjoin(string(net.OutputNames), ", "));

det = yolov8ObjectDetector(net, classes);
save(fullfile(outDir, "c3_yolov8s_idd_matlab.mat"), "det");
disp(det.ClassNames');

%% 2. Run on the 15 Q1 images
files = dir(fullfile(imgDir, "*.jpg"));
assert(~isempty(files), "No images in %s", imgDir);
I0 = imread(fullfile(files(1).folder, files(1).name));
for w = 1:5, detect(det, I0, Threshold=0.25); end      % GPU warm-up

annotated = cell(numel(files), 1); allLabels = strings(0,1);
for k = 1:numel(files)
    I = imread(fullfile(files(k).folder, files(k).name));
    [bb, sc, lb] = detect(det, I, Threshold=0.25);
    lb = string(lb); allLabels = [allLabels; lb(:)]; %#ok<AGROW>
    fs = max(14, round(size(I,2)/55)); lw = max(3, round(size(I,2)/320));
    if isempty(bb), J = I;
    else
        J = insertObjectAnnotation(I, "rectangle", bb, lb + " " + compose("%.2f", sc), ...
            LineWidth=lw, FontSize=min(fs,72), TextBoxOpacity=0.8);
    end
    annotated{k} = J;
    [~, base] = fileparts(files(k).name);
    imwrite(J, fullfile(outDir, base + "_ours_matlab.png"));
end
[u, ~, ic] = unique(allLabels);
fprintf("\nLabels found by OUR model in MATLAB (15 images):\n");
disp(table(u, accumarray(ic, 1), VariableNames=["label" "count"]));

%% 3. Slide figure: the 4 images that tell the story best
want = ["sideLeft" "2018-05-31_10-49-32" "frontNear__0000060" "rearNear"];
pick = [];
for w = want
    idx = find(contains({files.name}, w), 1);
    if ~isempty(idx), pick(end+1) = idx; end %#ok<AGROW>
end
if numel(pick) < 4, pick = unique([pick 1:4], "stable"); pick = pick(1:4); end
f = figure(Color="w", Position=[40 40 1600 950]);
try, f.Theme = "light"; catch, end
t = tiledlayout(f, 2, 2, TileSpacing="compact", Padding="compact");
for j = 1:4, nexttile(t); imshow(annotated{pick(j)}); end
title(t, "Our IDD-fine-tuned YOLOv8s running inside MATLAB (imported from Ultralytics via ONNX)", ...
    Color="k", FontWeight="bold", FontSize=14);
exportgraphics(f, fullfile(outDir, "c3_matlab_2x2.png"), Resolution=200, BackgroundColor="white");

%% 4. Speed on this laptop (compare with Q1: stock yolov8s = 39 ms full detect)
td = zeros(20,1);
for r = 1:20, s = tic; detect(det, I0, Threshold=0.25); td(r) = toc(s)*1000; end
timing = table("yolov8s_idd", median(td), 1000/median(td), VariableNames=["model" "detect_ms" "fps"]);
disp(timing);
writetable(timing, fullfile(outDir, "c3_matlab_timing.csv"));
fprintf("Done. Outputs in %s\n", outDir);
