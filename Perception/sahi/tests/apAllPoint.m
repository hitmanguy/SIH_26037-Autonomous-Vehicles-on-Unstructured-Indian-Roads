function ap = apAllPoint(scores, tp, nGT)
%APALLPOINT  Average precision (all-point interpolation, as in Ultralytics/VOC2010+).
if nGT == 0, ap = NaN; return; end
if isempty(scores), ap = 0; return; end
[~, o] = sort(scores(:), 'descend');
tp = double(tp(o)); fp = 1 - tp;
ctp = cumsum(tp); cfp = cumsum(fp);
rec = ctp / nGT; prec = ctp ./ max(ctp + cfp, eps);
mrec = [0; rec; 1]; mpre = [1; prec; 0];
for k = numel(mpre) - 1 : -1 : 1
    mpre(k) = max(mpre(k), mpre(k + 1));
end
idx = find(mrec(2:end) ~= mrec(1:end - 1)) + 1;
ap = sum((mrec(idx) - mrec(idx - 1)) .* mpre(idx));
end
