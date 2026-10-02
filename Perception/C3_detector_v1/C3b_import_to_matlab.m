%% C3b - Rebuild Epsilon's C3 detector from best_matlab.onnx (share-package version)
% SIH PS26037 · Team Epsilon · Perception (camera)
%
% You normally DON'T need this: load_c3_detector() uses the shipped .mat.
% Run it only if c3_yolov8s_idd_matlab.mat won't load on your machine.
% Everything is relative to this folder, so it works wherever you unzipped it.
%
% NEEDS: add-ons "Object Detection And Instance Segmentation Using YOLO v8" and
%        "Deep Learning Toolbox Converter for ONNX Model Format".
% OUTPUT (rebuild_output\ next to this file):
%   c3_yolov8s_idd_matlab.mat  - rebuilt detector (the shipped one is left untouched)
%   *_det.png                  - boxes on the images in test_images\
%   timing printed in the Command Window

clear; clc; close all;

here     = fileparts(mfilename('fullpath'));
onnxPath = fullfile(here, "best_matlab.onnx");
imgDir   = fullfile(here, "test_images");
outDir   = fullfile(here, "rebuild_output");
if ~isfolder(outDir), mkdir(outDir); end
addpath(here);

% Same order as training (YOLO class ids 0..11) - order matters
classes = ["person" "rider" "car" "bus" "truck" "autorickshaw" "motorcycle" ...
           "bicycle" "animal" "traffic sign" "traffic light" "vehicle fallback"];

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
save(fullfile(outDir, "c3_yolov8s_idd_matlab.mat"), "det");
fprintf("Rebuilt detector saved to %s\n", outDir);

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
