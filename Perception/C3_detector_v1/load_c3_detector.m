function det = load_c3_detector()
%LOAD_C3_DETECTOR  Return Epsilon's C3 detector v1 (yolov8ObjectDetector, 12 IDD classes).
%   det = load_c3_detector();
%   [bboxes, scores, labels] = detect(det, I);
%
%   Keep this file, c3_yolov8s_idd_matlab.mat and the +best_matlab folder together
%   in the same folder, and do not rename +best_matlab (the saved network refers
%   to its custom layers by that package name).

here = fileparts(mfilename('fullpath'));
addpath(here);                                   % so +best_matlab is found

if isempty(which('yolov8ObjectDetector'))
    error('epsilon:noYolo', ['yolov8ObjectDetector not found. Install/enable the add-on ' ...
        '"Object Detection And Instance Segmentation Using YOLO v8", restart MATLAB, ' ...
        'or run check_setup for an automatic fix.']);
end

s = load(fullfile(here, 'c3_yolov8s_idd_matlab.mat'));
f = fieldnames(s);
for k = 1:numel(f)
    if isa(s.(f{k}), 'yolov8ObjectDetector')
        det = s.(f{k});
        return
    end
end
error('epsilon:noDetector', 'No yolov8ObjectDetector inside c3_yolov8s_idd_matlab.mat.');
end
