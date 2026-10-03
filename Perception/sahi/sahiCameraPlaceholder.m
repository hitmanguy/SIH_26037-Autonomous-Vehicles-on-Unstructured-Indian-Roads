function cam = sahiCameraPlaceholder()
%SAHICAMERAPLACEHOLDER  PLACEHOLDER front-camera numbers for SAHI band rows.
%   *** NOT A REAL CAMERA. Replace when real or simulation intrinsics arrive. ***
%   Owner of the real values: A1 (monoCamera, per-camera intrinsics/extrinsics).
%   When A1 exists, SAHI should receive band rows in pixels from A1 and this file goes.
%
%   Chosen to match what the team already uses (Sensor_fusion code and
%   sahi_engine.py): 1920x1080, f = 1200 px (about 77 deg horizontal field of view),
%   principal point at the image centre, 1.5 m mounting height, level (pitch 0).

cam.Name           = 'PLACEHOLDER front camera (team default, not measured)';
cam.ImageSize      = [1080 1920];   % [rows cols]
cam.FocalLength    = [1200 1200];   % [fx fy] px
cam.PrincipalPoint = [960 540];     % [cx cy] px
cam.Height         = 1.5;           % m above the road
cam.Pitch          = 0;             % deg, positive = tilted down toward the road
end
