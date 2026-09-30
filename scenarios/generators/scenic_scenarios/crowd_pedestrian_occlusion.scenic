"""
=============================================================================
UC BERKELEY SCENIC: CROWD & OCCLUDED PEDESTRIAN SCENARIO
=============================================================================
Defines probabilistic pedestrian emergence from behind heavy parked vehicles
in an Indian urban environment.
=============================================================================
"""

param pedSpeed = Range(1.2, 1.8)       # 4.3 - 6.5 km/h

# Ego Vehicle in driving lane
ego = new Object at (10 @ -1.75),
    facing 0,
    with width 1.8,
    with length 4.5

# Stationary BMTC / Tata Bus on shoulder creating total blind spot
parked_bus = new Object at (Range(70, 90) @ -5.25),
    facing 0,
    with width 2.6,
    with length 11.0

# Jaywalking Pedestrian emerging from behind the bus
pedestrian = new Object at (Range(75, 85) @ -7.0),
    facing 1.5707,
    with width 0.45,
    with length 0.28
