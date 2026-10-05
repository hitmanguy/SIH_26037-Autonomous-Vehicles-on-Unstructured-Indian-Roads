classdef SahiSlowLoop < matlab.System
%SAHISLOWLOOP  SAHI for the FRONT camera as a Simulink MATLAB System block (5 Hz slow loop).
%   Runs the band tiles only, reusing the full-frame boxes from the fast loop,
%   and returns FIXED-SIZE outputs (MaxDets rows, zero-padded) plus a count.
%
%   Inputs  I          [H x W x 3] uint8 frame (same frame the fast loop used)
%           ffBoxes    [N x 4]  full-frame boxes from C3FullFrame   (spatial [x y w h])
%           ffScores   [N x 1]
%           ffLabels   [N x 1]  class ids 1..12
%           ffCount    scalar   number of real rows in the ff* inputs
%           bandRows   [1 x 2]  band [top bottom] pixel rows of THIS image (from A1;
%                               [0 0] = use the placeholder-camera rows)
%   Outputs boxes [N x 4] single, scores [N x 1] single, labels [N x 1] double,
%           count, overflow (true if > N boxes were found), sahiOnly [N x 1]
%           (true = no full-frame counterpart, i.e. found thanks to the tiles)
%
%   Pixel boxes only: no metres here (A1 converts).
%   Simulate with 'Interpreted execution'. Not code-generation ready yet (see
%   test_B4_simulink.m for the list of what GPU Coder would need changed).

    properties (Nontunable)
        DetectorFolder = ''   % '' = Perception/C3_detector_v1 in this repo
        MaxDets        = 128
        TileOnlyConf   = 0.35
        RectInput      = true
        BatchTiles     = true
    end

    properties (Access = private)
        Det
        Opts
    end

    methods (Access = protected)
        function setupImpl(obj)
            folder = obj.DetectorFolder;
            if isempty(folder)
                folder = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'C3_detector_v1');
            end
            addpath(folder);
            obj.Det = load_c3_detector();
            o = sahiDefaultOpts('v1');
            o.TileOnlyConf = obj.TileOnlyConf;
            o.RectInput = obj.RectInput;
            o.BatchTiles = obj.BatchTiles;
            obj.Opts = o;
        end

        function [boxes, scores, labels, count, overflow, sahiOnly] = ...
                stepImpl(obj, I, ffBoxes, ffScores, ffLabels, ffCount, bandRows)
            o = obj.Opts;
            n = double(ffCount);
            o.FullFrame = struct('Bboxes', double(ffBoxes(1:n, :)), ...
                                 'Scores', double(ffScores(1:n)), 'LabelIds', double(ffLabels(1:n)));
            if all(bandRows > 0)
                o.BandRows = double(bandRows(:).');
                o.BandRowsImageHeight = size(I, 1);
            end
            [b, s, ~, info] = sahiDetect(obj.Det, I, o);
            [boxes, scores, labels, count, overflow, sahiOnly] = ...
                packDetections(b, s, info.labelIds, info.isNew, obj.MaxDets);
        end

        function resetImpl(~)
        end

        % ---- fixed-size output specification for Simulink ----
        function num = getNumOutputsImpl(~), num = 6; end
        function varargout = getOutputSizeImpl(obj)
            N = obj.MaxDets;
            varargout = {[N 4], [N 1], [N 1], [1 1], [1 1], [N 1]};
        end
        function varargout = getOutputDataTypeImpl(~)
            varargout = {'single', 'single', 'double', 'double', 'logical', 'logical'};
        end
        function varargout = isOutputComplexImpl(~)
            varargout = repmat({false}, 1, 6);
        end
        function varargout = isOutputFixedSizeImpl(~)
            varargout = repmat({true}, 1, 6);
        end
        function varargout = getInputNamesImpl(~)
            varargout = {'I', 'ffBoxes', 'ffScores', 'ffLabels', 'ffCount', 'bandRows'};
        end
        function varargout = getOutputNamesImpl(~)
            varargout = {'boxes', 'scores', 'labels', 'count', 'overflow', 'sahiOnly'};
        end
    end
end
