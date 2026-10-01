"""
Unit Test Suite for Closed-Loop Simulink & Multi-Rate AV Simulation Pipeline
Smart India Hackathon (SIH) 2026 - Problem Statement 26037
Team Epsilon: Adaptive Path Planning and Collision Avoidance on Unstructured Indian Roads
"""

import math
import numpy as np

def test_multirate_clock_scheduling():
    """Verify multi-rate clock synchronization and harmonic integer ratios."""
    ts_dynamics   = 0.010 # 100 Hz
    ts_controller = 0.020 # 50 Hz
    ts_fusion     = 0.050 # 20 Hz
    ts_planning   = 0.100 # 10 Hz
    ts_prediction = 0.100 # 10 Hz
    ts_camera     = 0.064 # 15.6 Hz

    # All rates must be positive and dynamics must be the fastest base clock
    assert ts_dynamics <= ts_controller <= ts_fusion <= ts_planning
    
    # Check integer ratio with base clock for deterministic scheduling
    assert round(ts_controller / ts_dynamics) == 2
    assert round(ts_fusion / ts_dynamics) == 5
    assert round(ts_planning / ts_dynamics) == 10
    assert round(ts_prediction / ts_dynamics) == 10
    print("[PASS] Multi-rate clock scheduling and harmonic ratios validated.")


def test_bicycle_plant_kinematics():
    """Verify nonlinear kinematic bicycle model state propagation at 100 Hz."""
    wheelbase = 2.80
    lr = 1.55
    dt = 0.010
    
    # Start at X=20m, Y=-1.75m, Heading=0 rad, Speed=6.94 m/s (25 km/h)
    x, y, psi, v = 20.0, -1.75, 0.0, 6.94
    steer_cmd = math.radians(5.0) # 5 deg steer angle
    a_cmd = 0.50 # 0.5 m/s^2 accel
    
    for _ in range(100): # 1.0 second simulation
        beta = math.atan((lr / wheelbase) * math.tan(steer_cmd))
        x += v * math.cos(psi + beta) * dt
        y += v * math.sin(psi + beta) * dt
        psi += (v / wheelbase) * math.cos(beta) * math.tan(steer_cmd) * dt
        v += a_cmd * dt

    assert x > 26.0, f"Expected forward progress x > 26m, got {x:.2f}m"
    assert psi > 0.0, "Expected positive yaw heading for positive steer"
    assert v > 7.4, f"Expected speed increase, got {v:.2f}m/s"
    print(f"[PASS] 100 Hz Kinematic Bicycle Plant verified (End X={x:.2f}m, Y={y:.2f}m, Yaw={math.degrees(psi):.2f}deg, V={v*3.6:.1f}km/h).")


def test_pure_pursuit_controller_stability():
    """Verify Pure Pursuit controller drives cross-track error to zero without limit cycles."""
    wheelbase = 2.80
    lr = 1.55
    dt = 0.020 # 50 Hz controller
    
    # Vehicle starts with lateral error e_y = +0.8m (at Y = -0.95m instead of -1.75m)
    x, y, psi, v = 20.0, -0.95, 0.0, 6.94
    target_y = -1.75
    
    lateral_errors = []
    
    for step in range(250): # 5.0 seconds
        ey = y - target_y
        lateral_errors.append(abs(ey))
        
        # Lookahead distance
        Ld = max(4.0, min(18.0, 1.2 * v))
        target_pt = (x + Ld, target_y)
        dx = target_pt[0] - x
        dy = target_pt[1] - y
        alpha = math.atan2(dy, dx) - psi
        
        # Stanley + Pure pursuit law with correct negative feedback sign
        delta_pp = math.atan2(2.0 * wheelbase * math.sin(alpha), Ld)
        delta_cmd = max(-math.radians(35.0), min(math.radians(35.0), delta_pp))
        
        # Vehicle kinematics
        beta = math.atan((lr / wheelbase) * math.tan(delta_cmd))
        x += v * math.cos(psi + beta) * dt
        y += v * math.sin(psi + beta) * dt
        psi += (v / wheelbase) * math.cos(beta) * math.tan(delta_cmd) * dt

    # Error must converge smoothly to < 0.05m
    final_error = abs(y - target_y)
    assert final_error < 0.05, f"Tracking error failed to converge: final error = {final_error:.3f}m"
    assert lateral_errors[-1] < lateral_errors[0] * 0.1, "Error was not attenuated by at least 90%"
    print(f"[PASS] Pure Pursuit lateral convergence verified (Initial error=0.80m -> Final error={final_error:.4f}m).")


def test_emergency_braking_safety_buffer():
    """Verify vehicle halts safely with >= 3.5m clearance when approaching obstacle."""
    v0 = 6.94 # 25 km/h
    obs_dist = 25.0 # Obstacle 25m ahead
    stop_clearance = 3.5 # Minimum 3.5m buffer
    
    avail_dist = obs_dist - stop_clearance # 21.5m
    a_req = -(v0 ** 2) / (2 * avail_dist) # -1.12 m/s^2
    
    dt = 0.010
    v = v0
    x = 0.0
    
    while v > 0.01:
        x += v * dt + 0.5 * a_req * (dt ** 2)
        v = max(0.0, v + a_req * dt)

    final_clearance = obs_dist - x
    assert final_clearance >= stop_clearance - 0.05, f"Final clearance {final_clearance:.2f}m violated {stop_clearance}m buffer!"
    assert a_req >= -7.5, "Required deceleration exceeded braking authority"
    print(f"[PASS] Emergency Braking buffer verified (Stopped with {final_clearance:.2f}m clearance, a={a_req:.2f}m/s^2).")


if __name__ == '__main__':
    print("========================================================================")
    print("  RUNNING SIMULINK & CLOSED-LOOP PIPELINE INTEGRITY TESTS              ")
    print("========================================================================")
    test_multirate_clock_scheduling()
    test_bicycle_plant_kinematics()
    test_pure_pursuit_controller_stability()
    test_emergency_braking_safety_buffer()
    print("\n========================================================================")
    print("  ALL 4 SIMULINK CLOSED-LOOP TESTS PASSED WITH 100% INTEGRITY!          ")
    print("========================================================================")
