function s = parityMatch(pyXYWH, pyCls, pyScore, mXYWH, mCls, mScore)
%PARITYMATCH  One-to-one match of Python (sahi_engine.py) boxes to MATLAB boxes.
%   Python boxes are 0-based continuous [x1 y1 w h]; MATLAB boxes are spatial [x y w h]
%   (+0.5 px). Greedy by Python score, same class, IoU >= 0.5.
%   s.nPy, s.nMat, s.matched, s.meanIoU, s.minIoU, s.maxScoreDiff, s.unmatchedPy, s.unmatchedMat
pyXYWH = reshape(double(pyXYWH), [], 4);  mXYWH = reshape(double(mXYWH), [], 4);
pyCls = double(pyCls(:)); mCls = double(mCls(:)); pyScore = double(pyScore(:)); mScore = double(mScore(:));
pyXYWH(:, 1:2) = pyXYWH(:, 1:2) + 0.5;
a = [pyXYWH(:, 1:2), pyXYWH(:, 1:2) + pyXYWH(:, 3:4)];
b = [mXYWH(:, 1:2), mXYWH(:, 1:2) + mXYWH(:, 3:4)];
nP = size(a, 1); nM = size(b, 1);
s.nPy = nP; s.nMat = nM; s.matched = 0; s.meanIoU = NaN; s.minIoU = NaN; s.maxScoreDiff = NaN;
s.unmatchedPy = zeros(0, 1); s.unmatchedMat = zeros(0, 1);
if nP == 0 || nM == 0
    s.unmatchedPy = (1:nP).'; s.unmatchedMat = (1:nM).';
    return
end
x1 = max(a(:, 1), b(:, 1).'); y1 = max(a(:, 2), b(:, 2).');
x2 = min(a(:, 3), b(:, 3).'); y2 = min(a(:, 4), b(:, 4).');
inter = max(x2 - x1, 0) .* max(y2 - y1, 0);
aa = prod(pyXYWH(:, 3:4), 2); ab = prod(mXYWH(:, 3:4), 2);
iou = inter ./ max(aa + ab.' - inter, 1e-9);
iou(pyCls ~= mCls.') = 0;
[~, order] = sort(pyScore, 'descend');
usedM = false(nM, 1); ious = []; sd = []; hitP = false(nP, 1);
for i = order.'
    v = iou(i, :).'; v(usedM) = 0;
    [best, j] = max(v);
    if best >= 0.5
        usedM(j) = true; hitP(i) = true;
        ious(end + 1) = best;                          %#ok<AGROW>
        sd(end + 1) = abs(pyScore(i) - mScore(j));     %#ok<AGROW>
    end
end
s.matched = sum(hitP);
if ~isempty(ious), s.meanIoU = mean(ious); s.minIoU = min(ious); s.maxScoreDiff = max(sd); end
s.unmatchedPy = find(~hitP); s.unmatchedMat = find(~usedM);
end
