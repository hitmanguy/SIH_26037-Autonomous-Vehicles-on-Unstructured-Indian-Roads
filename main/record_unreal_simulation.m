function varargout = record_unreal_simulation(varargin)
% RECORD_UNREAL_SIMULATION Entry Point
    thisDir = fileparts(mfilename('fullpath'));
    rootDir = thisDir;
    if ~isfolder(fullfile(rootDir, 'vehicle_dynamics'))
        rootDir = fileparts(thisDir);
    end
    addpath(fullfile(rootDir, 'vehicle_dynamics'));
    origDir = pwd;
    cd(fullfile(rootDir, 'vehicle_dynamics'));
    try
        [varargout{1:nargout}] = record_unreal_simulation(varargin{:});
        cd(origDir);
    catch ME
        cd(origDir);
        rethrow(ME);
    end
end
