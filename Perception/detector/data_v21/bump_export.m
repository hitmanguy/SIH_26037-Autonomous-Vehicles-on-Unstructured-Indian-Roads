%% bump_export - hand-drawn speed-bump boxes (Image Labeler) -> bump_manual.jsonl for the server
% SIH PS26037 · Team Epsilon · v2.1 class 17 'slow_zone'
%
% BEFORE: Image Labeler -> load folder D:\SIH\Dataset\Pothole DATASET\bump
%         label: Rectangle named exactly  slow_zone
%         draw boxes only on good photos (leave bad ones empty = skipped)
%         Export -> Labels -> To File -> save as bump_gt.mat in THIS folder
% AFTER:  copy bump_manual.jsonl to the server and run
%         python bump_autolabel.py finalize --cands ~/epsilon_yash/bump_manual.jsonl --reject none

here    = fileparts(mfilename('fullpath'));
server  = "/home/<user>/epsilon_yash/data/speed_bump_kaggle/bump/";   % same photos on the server
S = load(fullfile(here, "bump_gt.mat"));
f = fieldnames(S);  gt = S.(f{1});                % the exported groundTruth object
src = string(gt.DataSource.Source);
T   = gt.LabelData;
assert(any(strcmp(T.Properties.VariableNames, "slow_zone")), ...
    "No label called slow_zone - rename the label in Image Labeler and export again.");

fid = fopen(fullfile(here, "bump_manual.jsonl"), "w");
nImg = 0; nBox = 0; skipped = strings(0);
for i = 1:numel(src)
    b = T.slow_zone{i};
    if isempty(b), continue; end
    info = imfinfo(src(i));
    if isfield(info, "Orientation") && info.Orientation > 1      % rotated phone photo:
        skipped(end+1) = src(i); continue;                         % boxes would not line up in training
    end
    W = info.Width; H = info.Height;
    x1 = (b(:,1) - 0.5) / W;  y1 = (b(:,2) - 0.5) / H;             % MATLAB pixel k spans k-0.5..k+0.5
    x2 = x1 + b(:,3) / W;     y2 = y1 + b(:,4) / H;
    bx = strjoin(compose("[%.6f,%.6f,%.6f,%.6f,1.0]", ...
         max(0,x1), max(0,y1), min(1,x2), min(1,y2)), ",");
    [~, nm, ext] = fileparts(src(i));
    fprintf(fid, '{"n": %d, "image": "%s", "boxes": [%s]}\n', i, server + nm + ext, bx);
    nImg = nImg + 1; nBox = nBox + size(b,1);
end
fclose(fid);
fprintf("%d photos, %d boxes -> %s\n", nImg, nBox, fullfile(here, "bump_manual.jsonl"));
if ~isempty(skipped)
    fprintf("Skipped %d rotated phone photos (EXIF orientation):\n", numel(skipped)); disp(skipped');
end
