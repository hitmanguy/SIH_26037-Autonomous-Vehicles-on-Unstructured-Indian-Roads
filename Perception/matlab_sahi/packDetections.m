function [boxes, scores, labels, count, overflow, flag] = packDetections(b, s, ids, f, N)
%PACKDETECTIONS  Variable-length detections -> fixed-size, zero-padded arrays for Simulink.
%   b [M x 4] boxes, s [M x 1] scores, ids [M x 1] class ids, f [M x 1] logical flag.
%   Keeps the N highest-scoring boxes; overflow = true if M > N.
%   Outputs: boxes [N x 4] single, scores [N x 1] single, labels [N x 1] double,
%            count (real rows at the top), overflow logical, flag [N x 1] logical.
M = numel(s);
[~, o] = sort(s(:), 'descend');
k = min(M, N);
o = o(1:k);
boxes = zeros(N, 4, 'single');  scores = zeros(N, 1, 'single');
labels = zeros(N, 1);           flag = false(N, 1);
if k > 0
    boxes(1:k, :) = single(b(o, :));
    scores(1:k)   = single(s(o));
    labels(1:k)   = double(ids(o));
    flag(1:k)     = logical(f(o));
end
count = k;
overflow = M > N;
end
