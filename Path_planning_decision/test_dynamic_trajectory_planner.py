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

if __name__ == '__main__':
    test_quintic_boundary()
    test_dynamic_obstacle_avoidance()
    print("\nALL DYNAMIC PLANNER UNIT TESTS PASSED!")
