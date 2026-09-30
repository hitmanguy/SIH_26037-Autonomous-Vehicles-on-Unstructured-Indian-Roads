function varargout = run_sim3d_surround(varargin)
% RUN_SIM3D_SURROUND 4-Camera Surround Simulink 3D Co-Simulation Runner
% Delegates to run_sim3d_surround_core to ensure robust non-recursive dispatch.
    thisDir = fileparts(mfilename('fullpath'));
    addpath(thisDir);
    [varargout{1:nargout}] = run_sim3d_surround_core(varargin{:});
end
