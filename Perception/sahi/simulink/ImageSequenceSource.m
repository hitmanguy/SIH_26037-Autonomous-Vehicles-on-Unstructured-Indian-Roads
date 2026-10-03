classdef ImageSequenceSource < matlab.System
%IMAGESEQUENCESOURCE  Test "camera" for Simulink: plays still images from a folder.
%   Input t (simulation time, from a Digital Clock). Each image is held for HoldSeconds,
%   then the next one, looping. Stand-in for the Unreal camera (A2) until that exists.
%   Outputs I [ImageHeight x ImageWidth x 3] uint8 and frameIdx (which image).

    properties (Nontunable)
        Folder      = ''   % '' = Perception/C3_detector_v1/test_images in this repo
        HoldSeconds = 0.5
        ImageHeight = 1080
        ImageWidth  = 1920
    end

    properties (Access = private)
        Frames
    end

    methods (Access = protected)
        function setupImpl(obj)
            folder = obj.Folder;
            if isempty(folder)
                folder = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'C3_detector_v1', 'test_images');
            end
            f = [dir(fullfile(folder, '*.jpg')); dir(fullfile(folder, '*.png'))];
            assert(~isempty(f), 'No images in %s', folder);
            obj.Frames = cell(numel(f), 1);
            for k = 1:numel(f)
                I = imread(fullfile(f(k).folder, f(k).name));
                if size(I, 3) == 1, I = repmat(I, 1, 1, 3); end
                if size(I, 1) ~= obj.ImageHeight || size(I, 2) ~= obj.ImageWidth
                    I = imresize(I, [obj.ImageHeight obj.ImageWidth]);
                end
                obj.Frames{k} = I;
            end
        end

        function [I, frameIdx] = stepImpl(obj, t)
            frameIdx = mod(floor(t / obj.HoldSeconds + 1e-6), numel(obj.Frames)) + 1;
            I = obj.Frames{frameIdx};
        end

        function num = getNumOutputsImpl(~), num = 2; end
        function varargout = getOutputSizeImpl(obj)
            varargout = {[obj.ImageHeight obj.ImageWidth 3], [1 1]};
        end
        function varargout = getOutputDataTypeImpl(~)
            varargout = {'uint8', 'double'};
        end
        function varargout = isOutputComplexImpl(~)
            varargout = {false, false};
        end
        function varargout = isOutputFixedSizeImpl(~)
            varargout = {true, true};
        end
    end
end
