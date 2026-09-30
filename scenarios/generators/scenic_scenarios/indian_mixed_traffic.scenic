"""
=============================================================================
UC BERKELEY SCENIC: INDIAN MIXED TRAFFIC PROBABILISTIC SCENARIO
=============================================================================
Defines a stochastic, probabilistic traffic scenario for Indian roadways:
- Heterogeneous traffic distribution: Auto-rickshaws, motorcycles, trucks, cars.
- Non-lane-following lateral displacements and randomized headway distances.
- Probabilistic cut-in triggers.
=============================================================================
"""

param carSpeed = Range(8.0, 12.0)      # 30 - 45 km/h
param autoSpeed = Range(6.0, 9.0)      # 20 - 32 km/h
param bikeSpeed = Range(11.0, 15.0)    # 40 - 55 km/h

# Ego vehicle (Passenger Car in driving lane)
ego = new Object at (20 @ -1.75),
    facing 0,
    with width 1.8,
    with length 4.5

# Auto-Rickshaw (Spawned on outer lane with aggressive cut-in heading)
auto_rickshaw = new Object at (Range(45, 60) @ -5.25),
    facing Range(-0.15, 0.05),
    with width 1.3,
    with length 2.65

# Two-Wheeler / Motorcycle (Filtering between lanes)
motorcycle = new Object at (Range(5, 15) @ -3.5),
    facing 0,
    with width 0.75,
    with length 2.1

# Slow Heavy Truck ahead in lane
truck = new Object at (Range(90, 120) @ -1.75),
    facing 0,
    with width 2.5,
    with length 8.0
