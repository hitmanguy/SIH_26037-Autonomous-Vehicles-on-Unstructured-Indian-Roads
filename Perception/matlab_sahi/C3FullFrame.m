classdef C3FullFrame < matlab.System
%C3FULLFRAME  Full-frame C3 detector as a Simulink MATLAB System block (fast loop).
%   Same direct-network path as sahiDetect (confidence 0.25, not detect()'s hidden 0.5).
%   Input  I [H x W x 3] uint8.   Outputs (fixed size, zero-padded):
%   boxes [N x 4] single (spatial [x y w h]), scores [N x 1] single, labels [N x 1] double,
%   count, overflow.

    properties (Nontunable)
        DetectorFolder = 'D:\SIH\share\C3_detector_v1'
        MaxDets        = 128
    end

    properties (Access = private)
        Det
        Opts
    end

    methods (Access = protected)
        function setupImpl(obj)
            addpath(obj.DetectorFolder);
            obj.Det = load_c3_detector();
            obj.Opts = sahiDefaultOpts('full');
        end

        function [boxes, scores, labels, count, overflow] = stepImpl(obj, I)
            [b, s, ~, info] = sahiDetect(obj.Det, I, obj.Opts);
            [boxes, scores, labels, count, overflow] = ...
                packDetections(b, s, info.labelIds, false(size(s)), obj.MaxDets);
        end

        function num = getNumOutputsImpl(~), num = 5; end
        function varargout = getOutputSizeImpl(obj)
            N = obj.MaxDets;
            varargout = {[N 4], [N 1], [N 1], [1 1], [1 1]};
        end
        function varargout = getOutputDataTypeImpl(~)
            varargout = {'single', 'single', 'double', 'double', 'logical'};
        end
        function varargout = isOutputComplexImpl(~)
            varargout = repmat({false}, 1, 5);
        end
        function varargout = isOutputFixedSizeImpl(~)
            varargout = repmat({true}, 1, 5);
        end
        function varargout = getOutputNamesImpl(~)
            varargout = {'boxes', 'scores', 'labels', 'count', 'overflow'};
        end
    end
end
