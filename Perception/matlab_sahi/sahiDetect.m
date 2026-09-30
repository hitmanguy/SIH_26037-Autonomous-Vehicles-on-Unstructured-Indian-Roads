function [bboxes, scores, labels, info] = sahiDetect(det, I, opts)
%SAHIDETECT  Sliced (SAHI) inference for small / far objects, FRONT camera only.
%   [bboxes, scores, labels, info] = sahiDetect(det, I)
%   [bboxes, scores, labels, info] = sahiDetect(det, I, opts)
%
%   MATLAB port of Perception/sahi_engine.py (team repo). Steps:
%     1. full-frame pass  : whole image letterboxed to 640x640
%     2. band tiles       : horizontal bands (fractions of image height) cut into
%                           640 px wide tiles with >= MinOverlap overlap that span the
%                           full width; tiles are letterboxed WITHOUT upscaling (1:1 pixels)
%     3. per-inference class-aware NMS (IoU 0.45), boxes mapped back to the full image
%     4. fragment-aware cross-tile merge (IoU 0.45, or IoS 0.6 when one box is cut by
%        an interior tile edge or is < 50% of the other's area); truncated boxes rank
%        lower; a truncated leader grows to the union of its truncated partners
%
%   INPUTS
%     det   yolov8ObjectDetector (from load_c3_detector) or its dlnetwork
%     I     H x W x 3 uint8 RGB frame (any resolution; bands scale with H)
%     opts  struct from sahiDefaultOpts('v1' | 'dual' | 'fast' | 'full'); missing fields use defaults
%
%   OUTPUTS (pixel boxes only; no metres here, that is A1's job)
%     bboxes  M x 4 [x y w h], MATLAB spatial coords (pixel k spans k-0.5..k+0.5),
%             ready for insertObjectAnnotation / rectangle
%     scores  M x 1 single
%     labels  M x 1 categorical (detector class names)
%     info    struct: labelIds, isNew (no full-frame match), truncated, source,
%             tiles [x y w h], nTiles, full-frame boxes, timings (ms)
%
%   The network is run directly (predict on det.Network) with our own letterbox and
%   YOLOv8 head decode, identical to sahi_engine.py. We do not call detect(det,...)
%   because its preprocessing pads to a 640x384 rectangle and quantises boxes, which
%   would break parity and cannot batch tiles.

if nargin < 3 || isempty(opts)
    opts = sahiDefaultOpts();
else
    opts = fillDefaults(opts, sahiDefaultOpts());
end

[net, classNames] = unpackDetector(det);
nc = numel(classNames);
useGPU = strcmpi(opts.ExecutionEnvironment, 'gpu') || ...
         (strcmpi(opts.ExecutionEnvironment, 'auto') && canUseGPU());

if size(I, 3) == 1, I = repmat(I, 1, 1, 3); end
[H, W, ~] = size(I);
logitThr = log(opts.Conf / (1 - opts.Conf));

%% 1. Full-frame pass
t0 = tic;
if ~isempty(opts.FullFrame)                 % reuse the fast loop's full-frame result
    fb = double(opts.FullFrame.Bboxes);
    ffB = single([fb(:, 1:2) - 0.5, fb(:, 1:2) - 0.5 + fb(:, 3:4)]);
    ffS = single(opts.FullFrame.Scores(:)); ffC = double(opts.FullFrame.LabelIds(:));
elseif opts.RunFullFrame
    [ffB, ffS, ffC] = detectOne(net, {I}, opts, logitThr, nc, useGPU);
    ffB = ffB{1}; ffS = ffS{1}; ffC = ffC{1};
else
    ffB = zeros(0, 4, 'single'); ffS = zeros(0, 1, 'single'); ffC = zeros(0, 1);
end
tFull = 1000 * toc(t0);

%% 2. Band tiles
t0 = tic;
tiles = makeTiles(H, W, opts);            % n x 4 [x0 y0 w h], 0-based pixel origin
nT = size(tiles, 1);
crops = cell(nT, 1);
for k = 1:nT
    x0 = tiles(k, 1); y0 = tiles(k, 2);
    crops{k} = I(y0 + 1 : y0 + tiles(k, 4), x0 + 1 : x0 + tiles(k, 3), :);
end
[tB, tS, tC] = detectOne(net, crops, opts, logitThr, nc, useGPU);

allB = ffB; allS = ffS; allC = ffC;
allT = false(size(ffS)); allSrc = zeros(size(ffS));
for k = 1:nT
    b = tB{k};
    x0 = tiles(k, 1); y0 = tiles(k, 2); tw = tiles(k, 3); th = tiles(k, 4);
    % truncated = touching a tile edge that is not also an image edge
    t = false(size(b, 1), 1);
    if x0 > 0,       t = t | b(:, 1) <= opts.EdgePx;      end
    if x0 + tw < W,  t = t | b(:, 3) >= tw - opts.EdgePx; end
    if y0 > 0,       t = t | b(:, 2) <= opts.EdgePx;      end
    if y0 + th < H,  t = t | b(:, 4) >= th - opts.EdgePx; end
    b = b + [x0 y0 x0 y0];
    allB = [allB; b];                     %#ok<AGROW>
    allS = [allS; tS{k}];                 %#ok<AGROW>
    allC = [allC; tC{k}];                 %#ok<AGROW>
    allT = [allT; t];                     %#ok<AGROW>
    allSrc = [allSrc; k * ones(size(t))]; %#ok<AGROW>
end
tTiles = 1000 * toc(t0);

%% 3. Fragment-aware cross-tile merge
t0 = tic;
[keep, merged] = mergeSlices(allB, allS, allC, allT, allSrc, opts);
tMerge = 1000 * toc(t0);

outS = allS(keep);  outC = allC(keep);

%% 4. Diagnostic: which merged boxes have no same-class full-frame counterpart?
isNew = true(numel(keep), 1);
if ~isempty(keep) && ~isempty(ffS)
    [iou, ios] = pairOverlap(merged, ffB);
    hit = (outC == ffC.') & ((iou > opts.NewIoU) | (ios > opts.MergeIoS));
    isNew = ~any(hit, 2);
end

%% Outputs
bboxes = toSpatialXYWH(merged);
scores = outS;
labels = makeLabels(outC, classNames);

info.labelIds   = outC;
info.isNew      = isNew;
info.truncated  = allT(keep);
info.source     = allSrc(keep);          % 0 = full frame, k = tile k
info.tiles      = [tiles(:, 1:2) + 0.5, tiles(:, 3:4)];   % spatial [x y w h]
info.nTiles     = nT;
info.ffBboxes   = toSpatialXYWH(ffB);
info.ffScores   = ffS;
info.ffLabelIds = ffC;
info.tFullMs    = tFull;
info.tTilesMs   = tTiles;
info.tMergeMs   = tMerge;
info.tSahiMs    = tTiles + tMerge;       % comparable to t_sahi_ms in sahi_engine.py
info.opts       = opts;
end

% =====================================================================
function [net, classNames] = unpackDetector(det)
if isa(det, 'dlnetwork')
    net = det;
    classNames = {'person','rider','car','bus','truck','autorickshaw','motorcycle', ...
                  'bicycle','animal','traffic sign','traffic light','vehicle fallback'};
elseif isstruct(det)                       % test harness
    net = det.Network; classNames = det.ClassNames;
else                                       % yolov8ObjectDetector
    net = det.Network; classNames = cellstr(string(det.ClassNames));
end
classNames = classNames(:).';
end

% =====================================================================
function tiles = makeTiles(H, W, opts)
% Tiles for every band, band by band, left to right. [x0 y0 w h], 0-based origin.
if ~isempty(opts.BandRows)                      % pixel rows (1-based inclusive), rescaled to H
    sc = H / opts.BandRowsImageHeight;
    yy = [round((opts.BandRows(:, 1) - 1) * sc), round(opts.BandRows(:, 2) * sc)];
else                                            % fractions of H (sahi_engine.py)
    yy = round(opts.Bands * H);
end
tiles = zeros(0, 4);
for b = 1:size(yy, 1)
    y1 = max(0, min(H - 1, yy(b, 1)));  y2 = max(y1 + 1, min(H, yy(b, 2)));
    xs = tileStarts(W, opts.TileWidth, opts.MinOverlap);
    for k = 1:numel(xs)
        ex = min(xs(k) + opts.TileWidth, W);
        tiles(end + 1, :) = [xs(k), y1, ex - xs(k), y2 - y1]; %#ok<AGROW>
    end
end
end

function xs = tileStarts(len, tile, minOverlap)
% Evenly spaced origins covering [0, len) with at least minOverlap overlap.
if len <= tile
    xs = 0;
    return
end
n = ceil((len - tile) / (tile * (1 - minOverlap))) + 1;
xs = round(linspace(0, len - tile, n));
end

% =====================================================================
function [B, S, C] = detectOne(net, imgs, opts, logitThr, nc, useGPU)
% Run the network on a list of images (full frame or tiles).
% Returns per-image boxes as xyxy in that image's own pixel frame (0-based, continuous).
n = numel(imgs);
B = cell(n, 1); S = cell(n, 1); C = cell(n, 1);
X = cell(n, 1); lb = zeros(n, 5);                % [r left top canvasH canvasW]
for k = 1:n
    [X{k}, lb(k, 1), lb(k, 2), lb(k, 3)] = letterbox(imgs{k}, opts);
    lb(k, 4:5) = [size(X{k}, 1), size(X{k}, 2)];
end

if opts.BatchTiles
    % one predict call per group of same-size inputs
    grp = zeros(n, 1); ng = 0;
    for k = 1:n
        if grp(k) == 0
            ng = ng + 1;
            grp(k:n) = grp(k:n) + ng * (grp(k:n) == 0 & lb(k:n, 4) == lb(k, 4) & lb(k:n, 5) == lb(k, 5));
        end
    end
    for g = 1:ng
        idx = find(grp == g);
        outs = forward(net, cat(4, X{idx}), useGPU);
        for j = 1:numel(idx)
            [B{idx(j)}, S{idx(j)}, C{idx(j)}] = decodeAndNms(outs, j, lb(idx(j), 4), opts, logitThr, nc);
        end
    end
else
    for k = 1:n
        outs = forward(net, X{k}, useGPU);
        [B{k}, S{k}, C{k}] = decodeAndNms(outs, 1, lb(k, 4), opts, logitThr, nc);
    end
end

for k = 1:n
    b = B{k};
    [h, w, ~] = size(imgs{k});
    b = (b - [lb(k, 2) lb(k, 3) lb(k, 2) lb(k, 3)]) / lb(k, 1);
    b(:, [1 3]) = min(max(b(:, [1 3]), 0), w);
    b(:, [2 4]) = min(max(b(:, [2 4]), 0), h);
    B{k} = b;
end
end

function [X, r, left, top] = letterbox(img, opts)
% Aspect-preserving resize (never upscales unless AllowUpscale) into a padded canvas:
% InputSize x InputSize, or with RectInput the smallest multiple of 32 that fits.
S = opts.InputSize;
[h, w, ~] = size(img);
r = min(S / w, S / h);
if ~opts.AllowUpscale, r = min(r, 1); end
nw = round(w * r); nh = round(h * r);
if nw ~= w || nh ~= h
    img = imresize(img, [nh nw], 'bilinear');    % antialiased when shrinking (as PIL)
end
if opts.RectInput
    ch = 32 * ceil(nh / 32); cw = 32 * ceil(nw / 32);
else
    ch = S; cw = S;
end
left = floor((cw - nw) / 2); top = floor((ch - nh) / 2);
X = repmat(single(opts.PadValue / 255), ch, cw, 3);
X(top + 1 : top + nh, left + 1 : left + nw, :) = single(img) / 255;
end

function outs = forward(net, X, useGPU)
% One predict call. Returns a cell of numeric arrays [h w 64+nc batch], one per scale.
dlX = dlarray(X, 'SSCB');
if useGPU, dlX = gpuArray(dlX); end
nOut = numel(net.OutputNames);
outs = cell(1, nOut);
[outs{:}] = predict(net, dlX);
for k = 1:nOut
    outs{k} = gather(extractdata(outs{k}));
end
end

function [boxes, scores, cls] = decodeAndNms(outs, bIdx, canvasH, opts, logitThr, nc)
% YOLOv8 head decode for batch element bIdx, then class-aware NMS.
% Box coords are in the letterboxed canvas frame.
boxes = zeros(0, 4, 'single'); scores = zeros(0, 1, 'single'); cls = zeros(0, 1);
for L = 1:numel(outs)
    o = outs{L}(:, :, :, bIdx);
    [hc, wc, ch] = size(o);
    if ch ~= 64 + nc
        error('sahiDetect:layout', ['Network output %d is %s; expected [h w %d batch]. ' ...
            'Check dims() of the predict outputs.'], L, mat2str(size(o)), 64 + nc);
    end
    s = canvasH / hc;                            % stride 8 / 16 / 32
    P = reshape(o, hc * wc, ch);                 % row i <-> (row mod(i-1,hc), col floor((i-1)/hc))
    [bestLogit, best] = max(P(:, 65:64 + nc), [], 2);
    idx = find(bestLogit > logitThr);
    if isempty(idx), continue; end
    d = reshape(P(idx, 1:64), numel(idx), 16, 4);  % DFL bins: (anchor, bin, side l/t/r/b)
    d = exp(d - max(d, [], 2));
    ltrb = reshape(sum(d .* (0:15), 2) ./ sum(d, 2), numel(idx), 4) * s;
    gy = (mod(idx - 1, hc) + 0.5) * s;
    gx = (floor((idx - 1) / hc) + 0.5) * s;
    boxes  = [boxes; [gx - ltrb(:, 1), gy - ltrb(:, 2), gx + ltrb(:, 3), gy + ltrb(:, 4)]]; %#ok<AGROW>
    scores = [scores; 1 ./ (1 + exp(-bestLogit(idx)))];                                     %#ok<AGROW>
    cls    = [cls; double(best(idx))];                                                      %#ok<AGROW>
end
keep = nmsClassAware(boxes, scores, cls, opts.NmsIoU);
boxes = boxes(keep, :); scores = scores(keep); cls = cls(keep);
end

function keep = nmsClassAware(boxes, scores, cls, thr)
% Greedy class-aware NMS; kept indices in descending score order.
if isempty(scores), keep = zeros(0, 1); return; end
[~, order] = sort(scores, 'descend');
iou = pairOverlap(boxes(order, :), boxes(order, :));
co = cls(order);
suppress = (co == co.') & (iou > thr);
alive = true(numel(order), 1);
for i = 1:numel(order)
    if alive(i)
        alive(i + 1 : end) = alive(i + 1 : end) & ~suppress(i, i + 1 : end).';
    end
end
keep = order(alive);
end

% =====================================================================
function [keep, merged] = mergeSlices(B, S, C, T, src, opts)
% Greedy cross-inference merge (port of merge_slices in sahi_engine.py).
N = numel(S);
keep = zeros(0, 1); merged = zeros(0, 4, 'single');
if N == 0, return; end
rank = S .* (1 - (1 - opts.TruncPenalty) * T);
[~, order] = sort(rank, 'descend');
[iou, ios] = pairOverlap(B, B);
area = (B(:, 3) - B(:, 1)) .* (B(:, 4) - B(:, 2));
ratio = min(area, area.') ./ max(max(area, area.'), 1e-9);
frag = T | T.' | (ratio < opts.FragAreaRatio);
match = (C == C.') & (src ~= src.') & ((iou > opts.MergeIoU) | ((ios > opts.MergeIoS) & frag));
done = false(N, 1);
for i = order.'
    if done(i), continue; end
    g = match(:, i) & ~done;
    g(i) = true;
    done = done | g;
    box = B(i, :);
    if T(i)
        parts = B(g & T, :);
        box = [min(parts(:, 1:2), [], 1), max(parts(:, 3:4), [], 1)];
    end
    keep(end + 1, 1) = i;        %#ok<AGROW>
    merged(end + 1, :) = box;    %#ok<AGROW>
end
end

function [iou, ios] = pairOverlap(a, b)
% IoU and intersection-over-smaller between xyxy box sets a (N) and b (M).
x1 = max(a(:, 1), b(:, 1).'); y1 = max(a(:, 2), b(:, 2).');
x2 = min(a(:, 3), b(:, 3).'); y2 = min(a(:, 4), b(:, 4).');
inter = max(x2 - x1, 0) .* max(y2 - y1, 0);
aa = (a(:, 3) - a(:, 1)) .* (a(:, 4) - a(:, 2));
ab = (b(:, 3) - b(:, 1)) .* (b(:, 4) - b(:, 2));
iou = inter ./ max(aa + ab.' - inter, 1e-9);
ios = inter ./ max(min(aa, ab.'), 1e-9);
end

% =====================================================================
function xywh = toSpatialXYWH(b)
% 0-based continuous xyxy -> MATLAB spatial [x y w h]
xywh = double([b(:, 1:2) + 0.5, b(:, 3:4) - b(:, 1:2)]);
end

function labels = makeLabels(ids, classNames)
labels = categorical(classNames(ids(:)), classNames).';
labels = labels(:);
end

function opts = fillDefaults(opts, def)
f = fieldnames(def);
for k = 1:numel(f)
    if ~isfield(opts, f{k}), opts.(f{k}) = def.(f{k}); end
end
end
