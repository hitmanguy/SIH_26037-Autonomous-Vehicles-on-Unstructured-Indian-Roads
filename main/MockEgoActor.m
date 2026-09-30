classdef MockEgoActor < handle
    % MOCKEGOACTOR Mock Handle Class for Ego Vehicle Actor in drivingScenario
    properties
        Position = [0, 0, 0]
        Velocity = [0, 0, 0]
        Yaw      = 0.0
    end
    methods
        function obj = MockEgoActor(pos, vel, yaw)
            if nargin >= 1, obj.Position = pos; end
            if nargin >= 2, obj.Velocity = vel; end
            if nargin >= 3, obj.Yaw = yaw; end
        end
    end
end
