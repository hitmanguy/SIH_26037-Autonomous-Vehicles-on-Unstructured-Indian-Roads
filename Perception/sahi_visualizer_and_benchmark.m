%% SAHI_VISUALIZER_AND_BENCHMARK
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% This script visualizes and benchmarks the SAHI (Slicing Aided Hyper Inference)
% engine on real India Driving Dataset (IDD) camera frames.
%
% Key Capabilities Demonstrated:
%  1. Ingestion of multi-band SAHI detections vs. Standard Full-Frame YOLOv8s.
%  2. Visual comparison highlighting newly recovered distant/small road users.
%  3. Class-wise detection gain breakdown (pedestrians, riders, motorcycles, rickshaws).
%  4. Resolution density scaling analysis (3.0x pixel boost in the 40-150m band).

clear; clc; closeall = @() delete(findall(0, 'Type', 'figure')); closeall();
fprintf('========================================================================\n');
fprintf('  SIH 26037: SAHI Multi-Band Image Slicer Visualization & Benchmark     \n');
fprintf('  Team Epsilon | Computer Vision & Automated Driving Toolbox             \n');
fprintf('========================================================================\n\n');

%% 1. Load SAHI Detections & Image
detFile = 'sahi_detection_results.mat';
assert(isfile(detFile), 'Cannot find %s. Run python sahi_engine.py first.', detFile);
sahiData = load(detFile);

% Target Image: Dense urban Bangalore street
imgKey = 'highquality_16k__BLR_2018_05_31_10_49_32__2018_05_31_10_58_6_131756_leftImg8bit';
assert(isfield(sahiData, imgKey), 'Key %s not found in %s', imgKey, detFile);

data = sahiData.(imgKey);
ff_boxes   = data.ff_boxes;
ff_scores  = data.ff_scores(:);
ff_labels  = cellstr(data.ff_labels);
ff_cls     = data.ff_cls(:);

sahi_boxes  = data.sahi_boxes;
sahi_scores = data.sahi_scores(:);
sahi_labels = cellstr(data.sahi_labels);
sahi_cls    = data.sahi_cls(:);
is_new_sahi = logical(data.is_new_sahi(:));

imgName = 'highquality_16k__BLR-2018-05-31_10-49-32__2018-05-31_10-58-6-131756_leftImg8bit.jpg';
here = fileparts(mfilename('fullpath'));
imgCandidates = {
    fullfile(pwd, 'C3_detector_v1', 'C3_detector_v1', 'test_images', imgName), ...
    fullfile(here, '..', 'C3_detector_v1', 'C3_detector_v1', 'test_images', imgName), ...
    fullfile(here, '..', '..', 'C3_detector_v1', 'C3_detector_v1', 'test_images', imgName)};
imgIdx = find(cellfun(@isfile, imgCandidates), 1);
assert(~isempty(imgIdx), 'Cannot locate Bangalore test image');
imgFile = imgCandidates{imgIdx};
I = imread(imgFile);

num_ff   = size(ff_boxes, 1);
num_sahi = size(sahi_boxes, 1);
num_new  = sum(is_new_sahi);

fprintf('>> Dense Urban Frame Benchmark:\n');
fprintf('   Standard Full-Frame YOLOv8s : %d detections\n', num_ff);
fprintf('   C3 YOLOv8s + SAHI Slicing   : %d detections (+%d objects with no full-frame match, +%.1f%%)\n\n', ...
    num_sahi, num_new, 100 * num_new / num_ff);

%% 2. Class Palette Definition
classNames = {'person', 'rider', 'car', 'bus', 'truck', 'autorickshaw', ...
              'motorcycle', 'bicycle', 'animal', 'traffic sign', 'traffic light', 'vehicle fallback'};

classColors = [
    0.90 0.20 0.20; % person: red
    0.95 0.50 0.10; % rider: orange
    0.20 0.60 0.90; % car: light blue
    0.10 0.30 0.80; % bus: dark blue
    0.50 0.10 0.70; % truck: purple
    0.00 0.75 0.35; % autorickshaw: emerald green
    0.90 0.75 0.10; % motorcycle: yellow
    0.30 0.80 0.80; % bicycle: cyan
    0.85 0.10 0.50; % animal: pink
    0.60 0.60 0.60; % traffic sign: gray
    0.40 0.70 0.20; % traffic light: lime
    0.70 0.40 0.20  % vehicle fallback: brown
];

%% 3. Generate High-Impact 4-Panel Visualization
fig = figure('Name', 'SIH 26037: SAHI Slicing Perception Benchmark', ...
             'Color', [0.08 0.09 0.11], 'Position', [60 60 1480 880], 'Visible', 'off');

txt_white = [0.95 0.96 0.98];
axes_dark = [0.12 0.13 0.16];
grid_col  = [0.25 0.28 0.35];

% --- Panel 1 (Top-Left): Standard Full-Frame YOLOv8 ---
subplot('Position', [0.03, 0.48, 0.46, 0.42]);
imshow(I); hold on;
title(sprintf('Standard Full-Frame YOLOv8s (Total: %d Detections)', num_ff), ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', txt_white);

for k = 1:num_ff
    b = ff_boxes(k, :);
    cid = ff_cls(k);
    col = classColors(cid, :);
    rectangle('Position', b, 'EdgeColor', col, 'LineWidth', 1.8);
end

% --- Panel 2 (Top-Right): SAHI Sliced YOLOv8 (Highlighting New Distant Detections) ---
subplot('Position', [0.51, 0.48, 0.46, 0.42]);
imshow(I); hold on;
title(sprintf('C3 YOLOv8s + SAHI Multi-Band Slicing (Total: %d Detections | +%d New)', num_sahi, num_new), ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.2 1.0 0.5]);

% Draw the two SAHI slicing bands (rows scale with image height; 1080p values shown)
imH = size(I, 1); imW = size(I, 2);
farBand  = [400 760]  * imH / 1080;
nearBand = [600 1000] * imH / 1080;
rectangle('Position', [2, farBand(1), imW - 4, diff(farBand)], 'EdgeColor', [0.2 0.8 1.0], ...
    'LineStyle', '--', 'LineWidth', 1.5);
rectangle('Position', [6, nearBand(1), imW - 12, diff(nearBand)], 'EdgeColor', [1.0 0.8 0.2], ...
    'LineStyle', '--', 'LineWidth', 1.5);
text(12, farBand(1) + 18, 'Far Band (40-150m)', 'Color', [0.2 0.8 1.0], 'FontSize', 8, 'FontWeight', 'bold');
text(12, nearBand(2) - 18, 'Near Band (15-30m)', 'Color', [1.0 0.8 0.2], 'FontSize', 8, 'FontWeight', 'bold');

for k = 1:num_sahi
    b = sahi_boxes(k, :);
    cid = sahi_cls(k);
    col = classColors(cid, :);
    
    if is_new_sahi(k)
        % Highlight newly caught distant objects with vibrant cyan box and marker
        rectangle('Position', b, 'EdgeColor', [0.1 1.0 0.9], 'LineWidth', 2.5);
        plot(b(1) + b(3)/2, b(2) - 8, 'v', 'MarkerSize', 8, ...
            'MarkerFaceColor', [0.1 1.0 0.9], 'MarkerEdgeColor', 'k');
    else
        rectangle('Position', b, 'EdgeColor', col, 'LineWidth', 1.5);
    end
end

% --- Panel 3 (Bottom-Left): Class Breakdown Comparison ---
subplot('Position', [0.06, 0.08, 0.42, 0.33]);
set(gca, 'Color', axes_dark, 'XColor', txt_white, 'YColor', txt_white, 'GridColor', grid_col, 'GridAlpha', 0.5);
hold on; grid on; box on;

% Count detections per class
classes_to_plot = {'motorcycle', 'rider', 'car', 'person', 'autorickshaw', 'bicycle'};
counts_ff   = zeros(length(classes_to_plot), 1);
counts_sahi = zeros(length(classes_to_plot), 1);

for c = 1:length(classes_to_plot)
    cname = classes_to_plot{c};
    counts_ff(c)   = sum(strcmp(ff_labels, cname));
    counts_sahi(c) = sum(strcmp(sahi_labels, cname));
end

x_idx = 1:length(classes_to_plot);
bar_data = [counts_ff, counts_sahi];
hb = bar(x_idx, bar_data, 0.7, 'grouped');
hb(1).FaceColor = [0.85 0.35 0.25]; % Red/Orange (Full-frame)
hb(2).FaceColor = [0.15 0.75 0.40]; % Green (SAHI)

set(gca, 'XTick', x_idx, 'XTickLabel', classes_to_plot, 'FontSize', 10);
ylabel('Object Count', 'Color', txt_white, 'FontSize', 10);
title('Class-Wise Detection Gain (Dense Urban Bangalore IDD Frame)', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', txt_white);
legend({'Standard Full-Frame', 'C3 YOLOv8 + SAHI'}, ...
    'Location', 'northwest', 'TextColor', txt_white, 'Color', axes_dark, 'EdgeColor', grid_col);
ylim([0 max([counts_ff; counts_sahi]) + 5]);

% Value annotations above bars
for i = 1:length(classes_to_plot)
    text(x_idx(i) - 0.18, counts_ff(i) + 0.8, num2str(counts_ff(i)), ...
        'Color', [1 0.8 0.8], 'FontSize', 8, 'FontWeight', 'bold');
    text(x_idx(i) + 0.12, counts_sahi(i) + 0.8, sprintf('%d (%+d)', counts_sahi(i), counts_sahi(i)-counts_ff(i)), ...
        'Color', [0.6 1.0 0.6], 'FontSize', 8, 'FontWeight', 'bold');
end

% --- Panel 4 (Bottom-Right): Resolution Density & Track Seeding Gain ---
subplot('Position', [0.55, 0.08, 0.42, 0.33]);
set(gca, 'Color', axes_dark, 'XColor', txt_white, 'YColor', txt_white, 'GridColor', grid_col, 'GridAlpha', 0.5);
hold on; grid on; box on;

% Pixel density resolution comparison
distances = [15, 30, 50, 80, 120, 150]; % meters
% Object of size 1.5m at distance Z has pixel height: h = (H * f) / Z
px_orig = (1.5 * 1200) ./ distances; % full 1080p frame
px_ff_downscaled = px_orig * (640 / 1920); % resized down to 640x640
px_sahi_slice    = px_orig * (640 / 640);  % 640x640 crop has 1:1 original pixel density!

plot(distances, px_ff_downscaled, 'o--', 'Color', [0.9 0.35 0.25], 'LineWidth', 2.0, 'MarkerSize', 6);
plot(distances, px_sahi_slice, 's-', 'Color', [0.15 0.85 0.45], 'LineWidth', 2.4, 'MarkerSize', 7);
xline(40, ':', 'Far Band Starts (40m)', 'Color', [0.3 0.8 1.0], 'LineWidth', 1.5);
xline(150, ':', 'Horizon (150m)', 'Color', [0.3 0.8 1.0], 'LineWidth', 1.5);

xlabel('Target Distance (meters)', 'Color', txt_white, 'FontSize', 10);
ylabel('Effective Target Pixel Height (px)', 'Color', txt_white, 'FontSize', 10);
title('Effective Pixel Resolution: Full-Frame Downscale vs. SAHI Slices', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', txt_white);
legend({'Full-Frame 640x640 Downscale (1/3 Resolution Loss)', 'SAHI 640x640 Tile Crop (Full 1:1 Sensor Resolution)'}, ...
    'Location', 'northeast', 'TextColor', txt_white, 'Color', axes_dark, 'EdgeColor', grid_col);
xlim([10 160]); ylim([0 120]);

% Annotation box explaining the 3x resolution boost
dim = [0.57 0.12 0.38 0.08];
annotation('textbox', dim, 'String', ...
    {'Key Architectural Advantage: letterboxed tiles keep 1:1 optical pixel density,', ...
     'giving distant hazards 3.0x the pixels of the full-frame 640 px downscale.'}, ...
    'Color', [0.2 1.0 0.5], 'BackgroundColor', axes_dark, 'EdgeColor', [0.2 0.8 0.4], ...
    'FontSize', 9, 'FontWeight', 'bold', 'FitBoxToText', 'on');

sgtitle({'Team Epsilon (SIH 26037): SAHI Slicing Perception Engine on India Driving Dataset', ...
         sprintf('Dual-Band Road Slicing Benchmark: +%d New Objects (+%.1f%% vs Full-Frame)', ...
                 num_new, 100 * num_new / num_ff)}, ...
        'FontSize', 13, 'FontWeight', 'bold', 'Color', txt_white);

saveas(fig, 'sahi_slicing_comparison.png');
fprintf('>> Saved publication figure: sahi_slicing_comparison.png\n');

save('sahi_benchmark_summary.mat', 'counts_ff', 'counts_sahi', 'classes_to_plot', ...
     'num_ff', 'num_sahi', 'num_new', 'distances', 'px_ff_downscaled', 'px_sahi_slice');
fprintf('>> Saved numerical benchmark logs: sahi_benchmark_summary.mat\n');
fprintf('>> SAHI visualization and benchmark completed successfully!\n');
