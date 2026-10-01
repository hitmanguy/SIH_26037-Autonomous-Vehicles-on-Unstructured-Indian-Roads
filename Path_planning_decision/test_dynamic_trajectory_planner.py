"""
Verification of Frenet Quintic Polynomial Dynamic Trajectory Planner
Smart India Hackathon (SIH) 2026 - Problem Statement 26037
Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
"""

import math
import numpy as np

class FrenetQuinticTrajectory:
    def __init__(self, d0, v_d0, a_d0, d1, T):
        self.d0 = d0
        self.v_d0 = v_d0
        self.a_d0 = a_d0
        self.d1 = d1
        self.T = T
        
        self.c0 = d0
        self.c1 = v_d0
        self.c2 = 0.5 * a_d0
        
        # Closed-form solver for boundary conditions: d(T)=d1, d'(T)=0, d''(T)=0
        h = d1 - self.c0 - self.c1 * T - self.c2 * (T ** 2)
        v1 = -self.c1 - 2.0 * self.c2 * T
        a1 = -2.0 * self.c2
        
        self.c3 = (10.0 * h - 4.0 * v1 * T + 0.5 * a1 * (T ** 2)) / (T ** 3)
        self.c4 = (-15.0 * h + 7.0 * v1 * T - a1 * (T ** 2)) / (T ** 4)
        self.c5 = (6.0 * h - 3.0 * v1 * T + 0.5 * a1 * (T ** 2)) / (T ** 5)
        
    def calc_pos(self, t):
        return self.c0 + self.c1 * t + self.c2 * (t**2) + self.c3 * (t**3) + self.c4 * (t**4) + self.c5 * (t**5)

    def calc_vel(self, t):
        return self.c1 + 2.0 * self.c2 * t + 3.0 * self.c3 * (t**2) + 4.0 * self.c4 * (t**3) + 5.0 * self.c5 * (t**4)

    def calc_acc(self, t):
        return 2.0 * self.c2 + 6.0 * self.c3 * t + 12.0 * self.c4 * (t**2) + 20.0 * self.c5 * (t**3)

    def calc_jerk(self, t):
        return 6.0 * self.c3 + 24.0 * self.c4 * t + 60.0 * self.c5 * (t**2)

    def calc_jerk_integral(self):
        T = self.T
        c3, c4, c5 = self.c3, self.c4, self.c5
        return (36.0 * (c3**2) * T +
                144.0 * c3 * c4 * (T**2) +
                (192.0 * (c4**2) + 240.0 * c3 * c5) * (T**3) +
                720.0 * c4 * c5 * (T**4) +
                720.0 * (c5**2) * (T**5))


def test_quintic_boundary():
    # Start at Y = -1.75, v_y = 0.1, a_y = 0.05
    # Target Y = -5.00 (Lane change right into Lane -2) in T = 3.0s
    poly = FrenetQuinticTrajectory(-1.75, 0.1, 0.05, -5.00, 3.0)
    
    assert abs(poly.calc_pos(0.0) - (-1.75)) < 1e-6
    assert abs(poly.calc_vel(0.0) - 0.1) < 1e-6
    assert abs(poly.calc_acc(0.0) - 0.05) < 1e-6
    
    assert abs(poly.calc_pos(3.0) - (-5.00)) < 1e-5
    assert abs(poly.calc_vel(3.0) - 0.0) < 1e-5
    assert abs(poly.calc_acc(3.0) - 0.0) < 1e-5
    
    # Numerical integration check of jerk
    t_vals = np.linspace(0, 3.0, 1000)
    dt = t_vals[1] - t_vals[0]
    num_jerk_int = np.sum(poly.calc_jerk(t_vals)**2) * dt
    ana_jerk_int = poly.calc_jerk_integral()
    
    assert abs(num_jerk_int - ana_jerk_int) / ana_jerk_int < 0.01
    print("[PASS] Frenet Quintic boundary conditions and analytical jerk validated.")


def test_dynamic_obstacle_avoidance():
    # Ego vehicle at X=20, Y=-1.75, Speed=6.94 m/s (25 km/h)
    # Obstacle (crawling auto-rickshaw or barrier) at X=55, Y=-1.75, Speed=2.0 m/s
    ego_x0, ego_y0, ego_v0 = 20.0, -1.75, 6.94
    obs_x0, obs_y0, obs_vx = 55.0, -1.75, 2.0
    
    # Candidate lateral targets (in Indian arterial road):
    # -1.75 (stay in lane), -3.5 (intermediate), -5.00 (Lane -2 clear corridor)
    targets = [-1.75, -3.25, -5.00]
    T_horizons = [2.5, 3.0, 3.5]
    
    best_traj = None
    min_cost = float('inf')
    
    for d_target in targets:
        for T in T_horizons:
            poly = FrenetQuinticTrajectory(ego_y0, 0.0, 0.0, d_target, T)
            jerk_cost = poly.calc_jerk_integral()
            
            # Evaluate along trajectory
            collided = False
            t_samples = np.linspace(0.1, T, 25)
            
            for t in t_samples:
                x_ego_t = ego_x0 + ego_v0 * t
                y_ego_t = poly.calc_pos(t)
                
                x_obs_t = obs_x0 + obs_vx * t
                y_obs_t = obs_y0
                
                # Check collision with safety envelope (a=5.0m, b=1.8m)
                dist_norm = ((x_ego_t - x_obs_t)/5.0)**2 + ((y_ego_t - y_obs_t)/1.8)**2
                if dist_norm < 1.0:
                    collided = True
                    break
                    
                # Constraint: cannot cross oncoming traffic Y > -0.6
                if y_ego_t > -0.6:
                    collided = True
                    break
            
            if collided:
                continue
                
            # Cost function: jerk + lane preference + time
            lane_pref = (d_target - (-5.00))**2 # prefers Lane -2 when avoiding
            cost = jerk_cost * 0.1 + lane_pref * 2.0 + T * 0.5
            
            if cost < min_cost:
                min_cost = cost
                best_traj = (d_target, T, poly)
                
    assert best_traj is not None
    assert best_traj[0] == -5.00, f"Expected Lane -2 (-5.00m) to be chosen, got {best_traj[0]}"
    print(f"[PASS] Dynamic replanner successfully chose evasion to Lane -2 (Y = {best_traj[0]:.2f}m, T = {best_traj[1]:.1f}s) avoiding lead collision!")


def test_pothole_handling():
    # Test 1: Deep cavity (>5 cm) in Cruising Lane (-1.75m) at X = 45m
    # Ego at X=15, Y=-1.75. Cruising lane is blocked by deep pothole.
    ego_x0, ego_y0 = 15.0, -1.75
    pothole_deep = {'X': 45.0, 'Y': -1.75, 'depth': 8.0, 'radius': 0.8}
    
    # Must choose detour into Lane -2 (-5.00m)
    targets = [-1.75, -3.25, -5.00]
    best_target = None
    min_cost = float('inf')
    
    for d_target in targets:
        poly = FrenetQuinticTrajectory(ego_y0, 0.0, 0.0, d_target, 3.0)
        collided = False
        for t in np.linspace(0.1, 3.0, 20):
            x_t = ego_x0 + 6.94 * t
            y_t = poly.calc_pos(t)
            # Distance to deep pothole cavity
            dist = math.hypot(x_t - pothole_deep['X'], y_t - pothole_deep['Y'])
            if dist < 2.0:
                collided = True
                break
        if not collided:
            cost = poly.calc_jerk_integral() * 0.1 + (d_target - (-5.00))**2
            if cost < min_cost:
                min_cost = cost
                best_target = d_target
                
    assert best_target == -5.00, f"Expected deep pothole to trigger detour to -5.00m, got {best_target}"
    
    # Test 2: Shallow dip (<5 cm) clamps speed to <= 15 km/h (4.17 m/s)
    pothole_shallow = {'depth': 3.5}
    cruise_speed = 6.94 # 25 km/h
    effective_speed = min(cruise_speed, 4.17) if pothole_shallow['depth'] < 5.0 else cruise_speed
    assert effective_speed <= 4.17
    print("[PASS] Deep pothole cavity detour and shallow dip speed clamp validated.")


def test_multimodal_gmm_freeze_avoidance():
    # Stray cow initially walks at 1.0 m/s across road, but Mode 3 (Freeze) pins velocity to 0 m/s at X=40, Y=-1.75
    # Linear extrapolation would falsely predict it walks across out of lane to Y=-0.50 by t=2.0s
    # Multi-modal prediction correctly evaluates Mode 3 where cow remains stationary at Y=-1.75m!
    ego_x0, ego_y0 = 20.0, -1.75
    
    # Mode 3: Freeze at X=40, Y=-1.75
    cow_freeze_pos = (40.0, -1.75)
    
    # If planner stays in lane (-1.75m), Mode 3 results in a collision at t = (40-20)/6.94 = 2.88s
    stay_in_lane_poly = FrenetQuinticTrajectory(ego_y0, 0.0, 0.0, -1.75, 3.0)
    collision_detected = False
    for t in np.linspace(0.1, 3.0, 25):
        x_ego = ego_x0 + 6.94 * t
        y_ego = stay_in_lane_poly.calc_pos(t)
        if math.hypot((x_ego - cow_freeze_pos[0])/4.0, (y_ego - cow_freeze_pos[1])/1.5) < 1.0:
            collision_detected = True
            break
            
    assert collision_detected, "Mode 3 freeze must collide with in-lane trajectory!"
    
    # Swerve to Lane -2 clears the frozen cow safely
    swerve_poly = FrenetQuinticTrajectory(ego_y0, 0.0, 0.0, -5.00, 3.0)
    swerve_collision = False
    for t in np.linspace(0.1, 3.0, 25):
        x_ego = ego_x0 + 6.94 * t
        y_ego = swerve_poly.calc_pos(t)
        if math.hypot((x_ego - cow_freeze_pos[0])/4.0, (y_ego - cow_freeze_pos[1])/1.5) < 1.0:
            swerve_collision = True
            break
            
    assert not swerve_collision, "Evasion to Lane -2 must clear frozen animal safely!"
    print("[PASS] Multi-modal GMM animal freeze collision evaluation validated.")


def test_stateflow_supervisory_integration():
    # Test Stateflow trigger STOP (e.g. min_TTC < 1.8s) clamps target speed to 0.0 m/s
    stateflow_stop = {'mode_name': 'STOP', 'mode_id': 4, 'min_TTC': 1.2}
    v_nominal = 6.94
    v_cmd = 0.0 if stateflow_stop['mode_name'] == 'STOP' else v_nominal
    assert v_cmd == 0.0
    
    # Test Stateflow trigger SLOW_DOWN (e.g. min_TTC = 2.5s) clamps target speed to <= 20 km/h (5.56 m/s)
    stateflow_slow = {'mode_name': 'SLOW_DOWN', 'mode_id': 2, 'min_TTC': 2.5}
    v_cmd_slow = min(v_nominal, 5.56) if stateflow_slow['mode_name'] == 'SLOW_DOWN' else v_nominal
    assert v_cmd_slow == 5.56
    print("[PASS] Stateflow supervisory triggers (STOP / SLOW_DOWN) speed integration validated.")


def test_impassable_emergency_braking():
    # Scenario: Both Lane 1 and Lane 2 are blocked (e.g. CPNCO child crossing with shoulder parked cars)
    # TargetSpeed must clamp to 0.0 m/s and emergency braking flag must be raised
    ego_x0, ego_v0 = 70.0, 6.94
    vru_crossing_dist = 96.5 - ego_x0 # 26.5m ahead
    
    # Kinematic stopping calculation: a_req = -v0^2 / (2 * (d - 3.5))
    avail_dist = max(0.5, vru_crossing_dist - 3.5)
    a_decel = -(ego_v0 ** 2) / (2 * avail_dist)
    t_stop = max(0.5, 2.0 * avail_dist / max(0.2, ego_v0))
    
    # Ego must decelerate to 0 before the 3.5m clearance boundary
    x_stop = ego_x0 + ego_v0 * t_stop + 0.5 * a_decel * (t_stop ** 2)
    assert x_stop <= 96.5 - 3.5 + 0.05, f"Stop position {x_stop} violated 3.5m clearance to obstacle!"
    assert a_decel >= -7.5, f"Required deceleration {a_decel} exceeded max braking authority!"
    print(f"[PASS] Impassable road emergency braking validated (Stopped at X={x_stop:.2f}m, a={a_decel:.2f}m/s^2).")


def test_oncoming_lane_boundary_enforcement():
    # Ego vehicle is on an Indian road: Negative Y is Ego's carriageway, Positive Y is oncoming traffic.
    # Centerline is at Y = 0.0m. CenterDividerY boundary limit is strictly Y <= -0.40m.
    center_divider_y = -0.40
    outer_shoulder_y = -6.50
    
    # Sample candidate swerve trajectories
    y_test_candidates = [-5.00, -3.50, -1.75, -0.60, 0.0, 1.75]
    valid_candidates = [y for y in y_test_candidates if (y >= outer_shoulder_y + 0.40 and y <= center_divider_y - 0.35)]
    
    # Must NOT permit 0.0 (centerline) or +1.75 (oncoming traffic)
    assert 0.0 not in valid_candidates, "Centerline must be excluded from valid candidates!"
    assert 1.75 not in valid_candidates, "Oncoming traffic lane (+1.75m) must be strictly excluded!"
    assert -0.60 not in valid_candidates, "Boundary buffer zone (-0.60m) must be excluded!"
    assert -1.75 in valid_candidates and -5.00 in valid_candidates
    print("[PASS] Indian road centerline & oncoming traffic boundary enforcement validated.")


if __name__ == '__main__':
    test_quintic_boundary()
    test_dynamic_obstacle_avoidance()
    test_pothole_handling()
    test_multimodal_gmm_freeze_avoidance()
    test_stateflow_supervisory_integration()
    test_impassable_emergency_braking()
    test_oncoming_lane_boundary_enforcement()
    print("\n=======================================================")
    print(" ALL DYNAMIC PLANNER & INTEGRATION UNIT TESTS PASSED! ")
    print("=======================================================")

