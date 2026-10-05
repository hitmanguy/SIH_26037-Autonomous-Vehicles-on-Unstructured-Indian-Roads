function [tpDet, gtHit] = evalMatch(detB, detC, detS, gtB, gtC, iouThr)
%EVALMATCH  Greedy class-aware matching of detections to ground truth (one image).
%   detB, gtB : [x y w h] spatial boxes;  detC, gtC : class ids;  detS : scores
%   tpDet(i) = true if detection i matched a GT box (IoU >= iouThr, same class)
%   gtHit(j) = true if GT box j was matched
nD = size(detB, 1); nG = size(gtB, 1);
tpDet = false(nD, 1); gtHit = false(nG, 1);
if nD == 0 || nG == 0, return; end
a = [detB(:, 1:2), detB(:, 1:2) + detB(:, 3:4)];
b = [gtB(:, 1:2),  gtB(:, 1:2)  + gtB(:, 3:4)];
x1 = max(a(:, 1), b(:, 1).'); y1 = max(a(:, 2), b(:, 2).');
x2 = min(a(:, 3), b(:, 3).'); y2 = min(a(:, 4), b(:, 4).');
inter = max(x2 - x1, 0) .* max(y2 - y1, 0);
iou = inter ./ max(prod(detB(:, 3:4), 2) + prod(gtB(:, 3:4), 2).' - inter, 1e-9);
iou(detC(:) ~= gtC(:).') = 0;
[~, order] = sort(detS(:), 'descend');
for i = order.'
    v = iou(i, :); v(gtHit) = 0;
    [best, j] = max(v);
    if best >= iouThr
        tpDet(i) = true; gtHit(j) = true;
    end
end
end
