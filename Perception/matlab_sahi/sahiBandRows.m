function rows = sahiBandRows(cam, spec)
%SAHIBANDROWS  Pixel rows [top bottom] of the v1 SAHI band for a camera.  (A1 STUB)
%   rows = sahiBandRows(cam)          % default spec: 15-150 m band
%   rows = sahiBandRows(cam, spec)
%
%   Plain idea: on a flat road, something standing further away has its feet
%   higher up in the image, so a distance range = a range of image rows.
%     bottom row = where the road is spec.NearDist metres ahead   (+ margin)
%     top row    = top of a spec.TallHeight-metre object at spec.TallDist metres (- margin)
%   This is the only place metres appear, and only to place the strip. SAHI itself
%   outputs pixel boxes. A1 will replace this with monoCamera-based rows.
%
%   Rows are 1-based and inclusive, in cam.ImageSize coordinates.

if nargin < 2, spec = struct(); end
def = struct('NearDist', 15, 'FarDist', 150, 'TallHeight', 3.5, 'TallDist', 40, 'Margin', 10);
f = fieldnames(def);
for k = 1:numel(f)
    if ~isfield(spec, f{k}), spec.(f{k}) = def.(f{k}); end
end

fy = cam.FocalLength(2); cy = cam.PrincipalPoint(2); h = cam.Height; p = deg2rad(cam.Pitch);
rowAt = @(d, z) cy + fy * tan(atan2(h - z, d) - p);   % image row of a point z m above road, d m ahead

bottom = rowAt(spec.NearDist, 0) + spec.Margin;
top    = min(rowAt(spec.TallDist, spec.TallHeight), rowAt(spec.FarDist, spec.TallHeight)) - spec.Margin;
rows = [max(1, floor(top)), min(cam.ImageSize(1), ceil(bottom))];
end
