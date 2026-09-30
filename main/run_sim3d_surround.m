function varargout = run_sim3d_surround(varargin)
% RUN_SIM3D_SURROUND Entry Point for 4-Camera Simulink 3D Surround Harness
    thisDir = fileparts(mfilename('fullpath'));
    rootDir = thisDir;
    if ~isfolder(fullfile(rootDir, 'vehicle_dynamics'))
        rootDir = fileparts(thisDir);
    end
    addpath(fullfile(rootDir, 'vehicle_dynamics'));
    addpath(fullfile(rootDir, 'scenarios'));
    addpath(fullfile(rootDir, 'scenarios', 'matlab'));
    [varargout{1:nargout}] = run_sim3d_surround_core(varargin{:});
end
