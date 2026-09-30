%% DECISION_SUPERVISOR
% Smart India Hackathon (SIH) 2026 - Problem Statement 26037
% Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
%
% Stateflow Supervisory Decision Logic:
% Implements a robust finite state machine (FSM) governing high-level vehicle
% tactical modes, speed limits, urgency flags, and safety fallback transitions.
%
% 5 Tactical Behavioral States:
%  1. CRUISE    (ID 1): Free-flow progression (v_target = 40 km/h)
%  2. SLOW_DOWN (ID 2): Precautionary deceleration (v_target = 20 km/h)
%  3. YIELD     (ID 3): Unsignalled junction negotiation / nudging (v_target = 10 km/h)
%  4. STOP      (ID 4): Emergency halt before frozen obstacle / cattle (v_target = 0 km/h)
%  5. REROUTE   (ID 5): Roadway impassable; search alternate detour / shoulder bypass
%
% Features:
%  - Debounce counters to prevent state chattering under sensor noise
%  - Indian road closing-speed unsignalled negotiation logic
%  - 5-second STOP timeout escalating to REROUTE
%  - Direct interface with Simulink Stateflow

classdef decision_supervisor < handle
    properties
        current_state       = 'CRUISE';
        current_state_id    = 1;
        time_in_state       = 0.0;     % Seconds spent in current state
        stop_hold_timer     = 0.0;     % Timer for stopped escalation (seconds)
        replan_needed       = false;   % Force immediate replanning flag
        urgency_factor      = 0.0;     % Scaled [0.0 - 1.0] for blending speed
        
        % Safety Thresholds
        ttc_stop_thresh     = 1.8;     % TTC below 1.8s triggers STOP
        ttc_slow_thresh     = 3.2;     % TTC below 3.2s triggers SLOW_DOWN
        dist_stop_thresh    = 8.0;     % Absolute distance (meters) to obstacle triggering STOP
        stop_timeout_limit  = 5.0;     % Seconds held in STOP before escalating to REROUTE
        debounce_count      = 0;       % Debounce cycles
        debounce_limit      = 2;       % Minimum cycles to confirm state transition
        candidate_state     = 'CRUISE';
        v_cruise_kmh        = 25.0;    % Configurable scenario cruise speed (km/h) - calm & steady pace
    end
    
    methods
        function obj = decision_supervisor(cfg)
            if nargin >= 1 && ~isempty(cfg)
                if isfield(cfg, 'v_cruise_kmh'),       obj.v_cruise_kmh       = cfg.v_cruise_kmh; end
                if isfield(cfg, 'cruise_speed'),       obj.v_cruise_kmh       = cfg.cruise_speed * 3.6; end
                if isfield(cfg, 'ttc_stop_thresh'),    obj.ttc_stop_thresh    = cfg.ttc_stop_thresh; end
                if isfield(cfg, 'ttc_slow_thresh'),    obj.ttc_slow_thresh    = cfg.ttc_slow_thresh; end
                if isfield(cfg, 'stop_timeout_limit'), obj.stop_timeout_limit = cfg.stop_timeout_limit; end
            end
            obj.current_state    = 'CRUISE';
            obj.current_state_id = 1;
        end
        
        function [decision, stats] = step(obj, dt, min_TTC, critical_agent_id, is_path_blocked, ...
                                         ego_speed, planner_infeasible, unsignalled_conflict)
            % dt                  : Timestep in seconds (e.g., 0.1s for 10 Hz)
            % min_TTC             : Minimum time-to-collision from Trajectory layer (seconds)
            % critical_agent_id   : Track ID of the primary threatening actor
            % is_path_blocked     : Boolean if forward path intersects high costmap obstacle
            % ego_speed           : Current ego speed in m/s
            % planner_infeasible  : Boolean if Hybrid A* failed to find a valid corridor
            % unsignalled_conflict: Boolean if oncoming actor closing in unsignalled junction
            
            if nargin < 7, planner_infeasible = false; end
            if nargin < 8, unsignalled_conflict = false; end
            
            % Determine raw candidate state from current telemetry
            raw_target = obj.evaluate_raw_state(min_TTC, is_path_blocked, planner_infeasible, unsignalled_conflict);
            
            % Debounce logic for state transitions (except emergency STOP which transitions immediately)
            if strcmpi(raw_target, 'STOP') || planner_infeasible
                % Emergency actions bypass debounce
                next_state = raw_target;
                obj.candidate_state = raw_target;
                obj.debounce_count = 0;
            else
                if strcmp(raw_target, obj.candidate_state)
                    obj.debounce_count = obj.debounce_count + 1;
                else
                    obj.candidate_state = raw_target;
                    obj.debounce_count = 1;
                end
                
                if obj.debounce_count >= obj.debounce_limit
                    next_state = obj.candidate_state;
                else
                    next_state = obj.current_state;
                end
            end
            
            % Update stop hold timer
            if strcmpi(obj.current_state, 'STOP')
                obj.stop_hold_timer = obj.stop_hold_timer + dt;
                % If stopped for > 5.0s and road is still blocked, escalate to REROUTE
                if obj.stop_hold_timer >= obj.stop_timeout_limit && is_path_blocked
                    next_state = 'REROUTE';
                end
            else
                obj.stop_hold_timer = 0.0;
            end
            
            % Handle state transition
            if ~strcmp(next_state, obj.current_state)
                obj.current_state = next_state;
                obj.time_in_state = 0.0;
                obj.replan_needed = true;
            else
                obj.time_in_state = obj.time_in_state + dt;
                obj.replan_needed = false;
            end
            
            % Update numeric ID and urgency factor
            switch upper(obj.current_state)
                case 'CRUISE'
                    obj.current_state_id = 1;
                    obj.urgency_factor = 0.0;
                    target_speed_kmh = obj.v_cruise_kmh;
                case 'SLOW_DOWN'
                    obj.current_state_id = 2;
                    obj.urgency_factor = 0.4;
                    target_speed_kmh = 0.5 * obj.v_cruise_kmh;
                case 'YIELD'
                    obj.current_state_id = 3;
                    obj.urgency_factor = 0.6;
                    target_speed_kmh = 0.25 * obj.v_cruise_kmh;
                case 'STOP'
                    obj.current_state_id = 4;
                    obj.urgency_factor = 1.0;
                    target_speed_kmh = 0.0;
                case 'REROUTE'
                    obj.current_state_id = 5;
                    obj.urgency_factor = 0.8;
                    target_speed_kmh = 0.3 * obj.v_cruise_kmh;
                otherwise
                    obj.current_state_id = 1;
                    obj.urgency_factor = 0.0;
                    target_speed_kmh = obj.v_cruise_kmh;
            end
            
            decision = struct(...
                'state', obj.current_state, ...
                'state_id', obj.current_state_id, ...
                'target_speed_mps', target_speed_kmh / 3.6, ...
                'target_speed_kmh', target_speed_kmh, ...
                'urgency_factor', obj.urgency_factor, ...
                'replan_needed', obj.replan_needed, ...
                'critical_agent_id', critical_agent_id ...
            );
            
            stats = struct(...
                'time_in_state', obj.time_in_state, ...
                'stop_hold_timer', obj.stop_hold_timer, ...
                'is_emergency', strcmpi(obj.current_state, 'STOP') ...
            );
        end
        
        function target = evaluate_raw_state(obj, min_TTC, is_path_blocked, planner_infeasible, unsignalled_conflict)
            % Evaluate priority order
            if planner_infeasible || (strcmpi(obj.current_state, 'REROUTE') && is_path_blocked)
                target = 'REROUTE';
                return;
            end
            
            % If path is blocked by a static obstacle/barrier/pothole, trigger REROUTE immediately to detour
            if is_path_blocked && min_TTC >= obj.ttc_stop_thresh
                target = 'REROUTE';
                return;
            end

            if min_TTC < obj.ttc_stop_thresh
                target = 'STOP';
                return;
            end
            
            if unsignalled_conflict
                target = 'YIELD';
                return;
            end
            
            if min_TTC < obj.ttc_slow_thresh || is_path_blocked
                target = 'SLOW_DOWN';
                return;
            end
            
            target = 'CRUISE';
        end
        
        function reset(obj)
            obj.current_state    = 'CRUISE';
            obj.current_state_id = 1;
            obj.time_in_state    = 0.0;
            obj.stop_hold_timer  = 0.0;
            obj.replan_needed    = false;
            obj.urgency_factor   = 0.0;
            obj.candidate_state  = 'CRUISE';
            obj.debounce_count   = 0;
        end
    end
end
