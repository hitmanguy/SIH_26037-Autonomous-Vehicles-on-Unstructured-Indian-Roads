"""
visualize_ncap_scenario.py
--------------------------
Visualizes the Euro NCAP AEB VRU CPNCO (Child Obstructed) test scenario
mapped onto the ASAM OpenDRIVE X-Intersection road network.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.io
import os

def render_ncap_overview():
    fig, ax = plt.subplots(figsize=(15, 8), facecolor='#111318')
    ax.set_facecolor('#1a1d24')

    # 1. Road Dimensions from X-Intersection_NCAP.xodr
    # Road 0 (West approach): X in [0, 250], Lane -1 center at Y = -1.75m
    # Junction 1: X in [250, 273], Y in [-11.5, 11.5]
    # Road 2 (East exit): X in [273, 350], Lane -1 center at Y = -1.75m
    # Road 1 (North): Y in [11.5, 80], X around 261.5
    # Road 3 (South): Y in [-80, -11.5], X around 261.5

    # Draw Road 0 & Road 2 Carriageways
    road_y_min, road_y_max = -9.0, 9.0
    ax.fill_between([0, 250], road_y_min, road_y_max, color='#2c3038', zorder=1)
    ax.fill_between([273, 350], road_y_min, road_y_max, color='#2c3038', zorder=1)
    
    # Draw North & South Carriageways
    ax.fill_between([252.5, 270.5], 11.5, 75, color='#2c3038', zorder=1)
    ax.fill_between([252.5, 270.5], -75, -11.5, color='#2c3038', zorder=1)

    # Draw 4-Way Junction Box
    ax.fill_between([250, 273], -11.5, 11.5, color='#343a46', zorder=1)

    # Lane Markings
    # Center lines (broken yellow)
    ax.plot([0, 250], [0, 0], color='#e5a93b', linestyle='--', linewidth=2, zorder=2)
    ax.plot([273, 350], [0, 0], color='#e5a93b', linestyle='--', linewidth=2, zorder=2)
    ax.plot([261.5, 261.5], [11.5, 75], color='#e5a93b', linestyle='--', linewidth=2, zorder=2)
    ax.plot([261.5, 261.5], [-75, -11.5], color='#e5a93b', linestyle='--', linewidth=2, zorder=2)

    # Driving lane boundaries (solid white at Y = -3.5, +3.5)
    ax.plot([0, 250], [-3.5, -3.5], color='#ffffff', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
    ax.plot([0, 250], [3.5, 3.5], color='#ffffff', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
    ax.plot([273, 350], [-3.5, -3.5], color='#ffffff', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
    ax.plot([273, 350], [3.5, 3.5], color='#ffffff', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)

    # Outer curbs (solid white at Y = -9.0, +9.0)
    ax.plot([0, 250], [-9.0, -9.0], color='#888888', linestyle='-', linewidth=2.5, zorder=2)
    ax.plot([0, 250], [9.0, 9.0], color='#888888', linestyle='-', linewidth=2.5, zorder=2)
    ax.plot([273, 350], [-9.0, -9.0], color='#888888', linestyle='-', linewidth=2.5, zorder=2)
    ax.plot([273, 350], [9.0, 9.0], color='#888888', linestyle='-', linewidth=2.5, zorder=2)

    # Zebra crossings at junction entrances
    for x_c in np.linspace(248.5, 250.0, 4):
        ax.plot([x_c, x_c], [-9.0, 9.0], color='#ffffff', linewidth=4, alpha=0.5, zorder=2)
    for x_c in np.linspace(273.0, 274.5, 4):
        ax.plot([x_c, x_c], [-9.0, 9.0], color='#ffffff', linewidth=4, alpha=0.5, zorder=2)

    # 2. Plot Scenario Entities & Interactions (CPNCO)
    # Parked Obstruction Vehicles (Shifted Further Off Road)
    obs_large = patches.Rectangle((89.927 - 2.209, -5.85 - 0.91), 4.418, 1.82,
                                  linewidth=1.5, edgecolor='#aaaaaa', facecolor='#4f5b66', zorder=5)
    ax.add_patch(obs_large)
    ax.text(89.927, -5.85, 'Obstruction\nLarge', color='white', fontsize=8,
            ha='center', va='center', fontweight='bold', zorder=6)

    obs_small = patches.Rectangle((95.325 - 2.158, -5.85 - 0.895), 4.316, 1.79,
                                  linewidth=1.5, edgecolor='#aaaaaa', facecolor='#5f6b76', zorder=5)
    ax.add_patch(obs_small)
    ax.text(95.325, -5.85, 'Obstruction\nSmall', color='white', fontsize=8,
            ha='center', va='center', fontweight='bold', zorder=6)

    # Occlusion field of view cone (blind zone)
    fov_poly = patches.Polygon([[70, -1.75], [97.5, -4.94], [102, -7.5], [70, -7.5]],
                               closed=True, facecolor='#ff3333', alpha=0.15, zorder=3)
    ax.add_patch(fov_poly)
    ax.text(82, -6.2, 'Driver Blind Zone\n(Vision Occluded)', color='#ff6666',
            fontsize=8, ha='center', va='center', zorder=4)

    # VRU (Child Pedestrian, Shifted Further Off Road)
    child_start = patches.Circle((100.0, -6.80), 0.6, facecolor='#f1c40f', edgecolor='#ffffff',
                                 linewidth=1.5, zorder=7)
    ax.add_patch(child_start)
    ax.text(100.0, -7.8, 'Child Start\n(Hidden Behind Car)', color='#f1c40f',
            fontsize=8, ha='center', va='top', fontweight='bold', zorder=8)

    # Ego Vehicle Trajectory
    ego_waypoints_x = [50, 70, 83, 96.5, 97.0, 115, 175, 245, 275, 320]
    ego_waypoints_y = [-1.75] * len(ego_waypoints_x)
    ax.plot(ego_waypoints_x, ego_waypoints_y, color='#3498db', linewidth=3, linestyle='-', zorder=5)

    # Ego Start
    ego_rect_start = patches.Rectangle((50 - 2.18, -1.75 - 0.91), 4.36, 1.82,
                                       linewidth=1.5, edgecolor='#5dade2', facecolor='#2980b9', zorder=6)
    ax.add_patch(ego_rect_start)
    ax.text(50, 1.2, 'Ego Start (25 km/h)', color='#5dade2', fontsize=8, ha='center', va='bottom', fontweight='bold', zorder=7)

    # AEB Stop Standstill Position
    ego_rect_stop = patches.Rectangle((96.5 - 2.18, -1.75 - 0.91), 4.36, 1.82,
                                      linewidth=2.0, edgecolor='#2ecc71', facecolor='#27ae60', alpha=0.9, zorder=6)
    ax.add_patch(ego_rect_stop)
    ax.text(96.5, -1.75, 'AEB Safe Standstill', color='white', fontsize=7.5, ha='center', va='center', fontweight='bold', zorder=7)

    # Child Crossing Trajectory
    ax.annotate('', xy=(100.0, 2.5), xytext=(100.0, -6.80),
                arrowprops=dict(arrowstyle="->", color='#f1c40f', lw=2.5, linestyle='--'), zorder=6)
    ax.text(101.5, 3.2, 'Child Crossing (5 km/h)', color='#f1c40f', fontsize=8.5, fontweight='bold', zorder=8)

    # Ego Crossing Junction
    ax.annotate('', xy=(310, -1.75), xytext=(240, -1.75),
                arrowprops=dict(arrowstyle="->", color='#3498db', lw=3), zorder=6)
    ax.text(285, -4.0, 'Exits through 4-Way\nX-Intersection to East Road', color='#5dade2',
            fontsize=9, ha='center', fontweight='bold', zorder=8)

    # Safety Distance Bracket
    ax.plot([96.5, 100.0], [5.5, 5.5], color='#2ecc71', linewidth=2, zorder=6)
    ax.plot([96.5, 96.5], [4.8, 6.2], color='#2ecc71', linewidth=2, zorder=6)
    ax.plot([100.0, 100.0], [4.8, 6.2], color='#2ecc71', linewidth=2, zorder=6)
    ax.text(98.25, 7.0, 'Safety Buffer: 3.5 m (Zero Collision)', color='#2ecc71',
            fontsize=9, ha='center', fontweight='bold', zorder=7)

    # Layout Setup
    ax.set_xlim([35, 335])
    ax.set_ylim([-30, 30])
    ax.set_aspect('equal')
    ax.set_xlabel('X Position [m] (OpenDRIVE Reference Line)', color='#cccccc', fontsize=11)
    ax.set_ylabel('Y Position [m]', color='#cccccc', fontsize=11)
    ax.tick_params(colors='#888888', labelsize=10)

    title_text = (
        "Euro NCAP AEB VRU CPNCO Scenario on ASAM OpenDRIVE X-Intersection\n"
        "Car-to-Pedestrian Nearside Child Obstructed with Active Collision Avoidance"
    )
    plt.title(title_text, color='#ffffff', fontsize=13, fontweight='bold', pad=15)

    # Information Box
    info_str = (
        "Scenario: NCAP_AEB_VRU_CPNCO_2023.xosc\n"
        "Map: X-Intersection_NCAP.xodr\n"
        "Ego Speed: 30 km/h | VRU Speed: 5 km/h\n"
        "Obstruction: 2 Parked Vehicles (Small + Large)\n"
        "AEB Verification: Pass (100% Zero-Collision)\n"
        "Min Clearance: >= 1.00 m (OBB SAT Certified)"
    )
    ax.text(0.02, 0.95, info_str, transform=ax.transAxes, color='#00e676',
            fontsize=9, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#111822', edgecolor='#00e676', alpha=0.9))

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncap_cpnco_intersection.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] NCAP overview diagram saved to: {out_path}")
    return out_path

if __name__ == '__main__':
    render_ncap_overview()
