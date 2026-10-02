function check_setup()
%CHECK_SETUP  Verify this MATLAB can run Epsilon's C3 detector v1 (YOLOv8s fine-tuned on IDD).
%   Run it from a FRESH MATLAB session (do NOT use restoredefaultpath first):
%       >> cd <wherever you unzipped C3_detector_v1>
%       >> check_setup
%   It checks release + toolboxes + add-ons, tries to self-repair the YOLO v8
%   add-on path, loads the detector, runs it on the images in test_images\,
%   and writes annotated results to check_output\.

here = fileparts(mfilename('fullpath'));
addpath(here);                         % puts +best_matlab (imported custom layers) on the path

fprintf('\n=== Epsilon C3 detector v1 : setup check ===\n\n');
ok = true;

%% 1) Release
r = version('-release');
fprintf('%-48s %s\n', 'MATLAB release', r);
if ~strcmp(r, '2025b')
    warning('Detector was built on R2025b (latest Update). Other releases may fail to load it.');
end

%% 2) Toolboxes
v = ver;  names = {v.Name};
for t = ["Deep Learning Toolbox", "Computer Vision Toolbox"]
    ok = report(t, any(strcmp(names, t))) && ok;
end

%% 3) Add-ons (checked by the function they provide, not by display name)
ok = report("ONNX converter add-on (importNetworkFromONNX)", ...
            ~isempty(which('importNetworkFromONNX'))) && ok;

hasYolo = ~isempty(which('yolov8ObjectDetector'));
if ~hasYolo
    % Installed-and-enabled but missing from the path (e.g. after restoredefaultpath):
    % toggling it makes the Add-On manager put its folders back on the path.
    try
        a   = matlab.addons.installedAddons;
        idx = find(contains(a.Name, "YOLO v8", 'IgnoreCase', true));
        for k = idx(:)'
            fprintf('  "%s" installed but not on path -> re-enabling...\n', a.Name(k));
            matlab.addons.disableAddon(a.Identifier(k));
            matlab.addons.enableAddon(a.Identifier(k));
        end
        hasYolo = ~isempty(which('yolov8ObjectDetector'));
    catch err
        fprintf('  (auto-repair failed: %s)\n', err.message);
    end
end
ok = report("YOLO v8 add-on (yolov8ObjectDetector)", hasYolo) && ok;

%% 4) GPU (info only; CPU works, just slower)
g = canUseGPU;
fprintf('%-48s %s\n', 'GPU usable', string(g));

if ~ok
    fprintf(2, '\nFix the FAIL lines above (see README.md), restart MATLAB, run again.\n');
    return
end

%% 5) Load detector
try
    det = load_c3_detector();
    fprintf('%-48s OK  (%d classes)\n', 'Detector loaded', numel(det.ClassNames));
catch err
    fprintf(2, '%-48s FAIL\n  %s\n', 'Detector loaded', err.message);
    fprintf(2, '  Fallback: run C3b_import_to_matlab.m to rebuild it from best_matlab.onnx\n');
    return
end

%% 6) Detect on test images
imgs = [dir(fullfile(here, 'test_images', '*.jpg')); dir(fullfile(here, 'test_images', '*.png'))];
if isempty(imgs)
    fprintf('\nNo images in test_images\\ - add a road frame there to test detection.\n');
    return
end
outDir = fullfile(here, 'check_output');
if ~isfolder(outDir), mkdir(outDir); end

fprintf('\n%-40s %6s %10s\n', 'image', 'boxes', 'ms/frame');
for k = 1:numel(imgs)
    I = imread(fullfile(imgs(k).folder, imgs(k).name));
    if size(I, 3) == 1, I = repmat(I, 1, 1, 3); end
    % First call on a new image size is slow (GPU warm-up / setup), so it is not timed.
    [bb, sc, lb] = detect(det, I);
    t = zeros(3, 1);
    for r = 1:3, s0 = tic; detect(det, I); t(r) = 1000 * toc(s0); end
    ms = median(t);
    nm = imgs(k).name; if strlength(nm) > 40, nm = extractBefore(string(nm), 38) + "..."; end
    fprintf('%-40s %6d %10.1f\n', nm, size(bb, 1), ms);
    if ~isempty(bb)
        I = insertObjectAnnotation(I, 'rectangle', bb, ...
              compose("%s %.2f", string(lb), sc), 'LineWidth', 2);
    end
    [~, base] = fileparts(imgs(k).name);
    imwrite(I, fullfile(outDir, base + "_det.png"));
end
fprintf('\nAll good. Annotated images in check_output\\\n');
end

function ok = report(label, pass)
if pass, s = 'OK'; else, s = 'FAIL'; end
fprintf('%-48s %s\n', label, s);
ok = pass;
end
