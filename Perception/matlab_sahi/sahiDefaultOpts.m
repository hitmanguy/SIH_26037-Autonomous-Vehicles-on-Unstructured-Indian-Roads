function opts = sahiDefaultOpts(mode)
%SAHIDEFAULTOPTS  Default settings for sahiDetect (A3 SAHI, SIH PS26037, Team Epsilon).
%   opts = sahiDefaultOpts('v1')    % ONE band, 15-150 m, rows from the camera (default)
%   opts = sahiDefaultOpts('dual')  % Pranava's far + near bands (sahi_engine.py default)
%   opts = sahiDefaultOpts('fast')  % Pranava's --fast: one band, rows 400-1000 on 1080p
%   opts = sahiDefaultOpts('full')  % no bands: full-frame pass only (the 30 Hz loop)
%
%   Bands are for the FRONT camera only.

if nargin < 1 || isempty(mode), mode = 'v1'; end
opts.Mode = lower(mode);

% --- band geometry. Either BandRows (pixel rows, 1-based inclusive, for an image
%     BandRowsImageHeight tall; rescaled if the frame height differs) or Bands
%     (fractions of image height, as in sahi_engine.py). BandRows wins if set.
opts.BandRows = zeros(0, 2);
opts.BandRowsImageHeight = 1080;
switch opts.Mode
    case 'v1'
        cam = sahiCameraPlaceholder();                  % PLACEHOLDER camera (A1 stub)
        opts.BandRows = sahiBandRows(cam);               % [470 670] with the placeholder
        opts.BandRowsImageHeight = cam.ImageSize(1);
        opts.Bands = zeros(0, 2);
        opts.BandNames = {'v1'};
    case 'dual'
        opts.Bands = [400 760; 600 1000] / 1080;         % far, near
        opts.BandNames = {'far', 'near'};
    case 'fast'
        opts.Bands = [400 1000] / 1080;
        opts.BandNames = {'road'};
    case 'full'
        opts.Bands = zeros(0, 2);
        opts.BandNames = {};
    otherwise
        error('sahiDefaultOpts:mode', 'mode must be ''v1'', ''dual'', ''fast'' or ''full''.');
end

% --- tiling
opts.TileWidth    = 640;    % px, tiles cut at native 1:1 resolution
opts.MinOverlap   = 0.30;   % sahi_engine.py MIN_TILE_OVERLAP (33% real overlap, 4 tiles on 1920 px)
opts.InputSize    = 640;    % network input side
opts.PadValue     = 114;    % Ultralytics letterbox grey
opts.AllowUpscale = false;  % letterbox never enlarges a tile
opts.RectInput    = false;  % Block 2: pad only to a multiple of 32 (e.g. 640x224) instead of 640x640

% --- thresholds (sahi_engine.py values)
opts.Conf          = 0.25;
opts.NmsIoU        = 0.45;
opts.MergeIoU      = 0.45;
opts.MergeIoS      = 0.60;
opts.FragAreaRatio = 0.50;
opts.EdgePx        = 3;
opts.TruncPenalty  = 0.5;
opts.NewIoU        = 0.35;  % "new vs full-frame" matching (diagnostic only)

% --- execution
opts.RunFullFrame  = true;   % run the full-frame pass inside sahiDetect
opts.FullFrame     = [];     % OR reuse the 30 Hz loop's result: struct with Bboxes, Scores, LabelIds
                             %    (as returned in info.ffBboxes / ffScores / ffLabelIds)
opts.BatchTiles    = false;  % send same-size tiles in one predict call
opts.ExecutionEnvironment = 'auto';   % 'auto' | 'gpu' | 'cpu'
end
