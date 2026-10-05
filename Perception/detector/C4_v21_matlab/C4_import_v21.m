%% C4b - Import Epsilon's v2.1 detector (18 classes: v2 + slow_zone) into MATLAB
% SIH PS26037 · Team Epsilon · Perception (camera)
%
% Put v2.1's best_matlab.onnx (exported on the server with export_for_matlab.py) next to this file.
% Test images: the 4 v1 frames in D:\SIH\share\C3_detector_v1\test_images (+ any you add to test_images\ here).
%
% NEEDS: add-ons "Object Detection And Instance Segmentation Using YOLO v8" and
%        "Deep Learning Toolbox Converter for ONNX Model Format".
% OUTPUT (rebuild_output\ next to this file):
%   v21_yolov8s_matlab.mat      - the v2 detector
%   *_det.png                  - boxes on the images in test_images\
%   timing printed in the Command Window

clear; clc; close all;

here     = fileparts(mfilename('fullpath'));
onnxPath = fullfile(here, "best_matlab.onnx");
imgDir   = fullfile(here, "test_images");
if ~isfolder(imgDir) || isempty(dir(fullfile(imgDir, "*.jpg")))
    imgDir = fullfile(here, "..", "..", "share", "C3_detector_v1", "test_images");
end
outDir   = fullfile(here, "..", "..", "work", "C4_v21");
if ~isfolder(outDir), mkdir(outDir); end
addpath(here);

% Same order as training (YOLO class ids 0..11) - order matters
classes = ["person" "rider" "car" "bus" "truck" "autorickshaw" "motorcycle" ...
           "bicycle" "animal" "traffic sign" "traffic light" "vehicle fallback" ...
           "pothole" "pushcart" "tractor" "emergency vehicle" "cone_barrier" "slow_zone"];

%% 1. Import (custom layers are generated into +best_matlab in this folder)
assert(isfile(onnxPath), "best_matlab.onnx not found at %s", onnxPath);
oldDir = cd(here);  cleanup = onCleanup(@() cd(oldDir));
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

det = yolov8ObjectDetector(net, classes);
save(fullfile(outDir, "v21_yolov8s_matlab.mat"), "det");
fprintf("v2.1 detector saved to %s\n", outDir);

%% 2. Run on test_images
files = [dir(fullfile(imgDir, "*.jpg")); dir(fullfile(imgDir, "*.png"))];
if isempty(files)
    fprintf("No images in %s - add a road frame to test.\n", imgDir);
    return
end
I0 = imread(fullfile(files(1).folder, files(1).name));
for w = 1:3, detect(det, I0, Threshold=0.25); end      % warm-up

for k = 1:numel(files)
    I = imread(fullfile(files(k).folder, files(k).name));
    [bb, sc, lb] = detect(det, I, Threshold=0.25);
    lw = max(3, round(size(I,2)/320)); fs = min(72, max(14, round(size(I,2)/55)));
    if ~isempty(bb)
        I = insertObjectAnnotation(I, "rectangle", bb, string(lb) + " " + compose("%.2f", sc), ...
            LineWidth=lw, FontSize=fs, TextBoxOpacity=0.8);
    end
    [~, base] = fileparts(files(k).name);
    imwrite(I, fullfile(outDir, base + "_det.png"));
    fprintf("%-60s %3d boxes\n", files(k).name, numel(sc));
end

%% 3. Speed on this machine
td = zeros(20,1);
for r = 1:20, s = tic; detect(det, I0, Threshold=0.25); td(r) = toc(s)*1000; end
fprintf("\nMedian detect time: %.1f ms  (%.1f fps)\nDone. Outputs in %s\n", ...
    median(td), 1000/median(td), outDir);
