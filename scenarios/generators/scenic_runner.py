"""
=============================================================================
UC BERKELEY SCENIC RUNNER & SAMPLER
=============================================================================
Loads and compiles .scenic scenario programs, verifies syntax and constraints,
and samples concrete scene instances from the probabilistic distributions.

Usage:
    python scenic_runner.py [scenario_file.scenic] [num_samples]
=============================================================================
"""

import sys
import os

try:
    import scenic
    from scenic.core.scenarios import Scenario
except ImportError:
    print("Scenic is not installed. Please run: pip install scenic")
    sys.exit(1)


def sample_scenic_scenario(scenic_path, num_samples=3):
    print(f"[SCENIC] Loading and compiling Scenic program: {scenic_path}")
    scenario = scenic.scenarioFromFile(scenic_path)
    print(f"[SCENIC] Scenario compiled successfully. Generating {num_samples} sample scenes...\n")

    for i in range(num_samples):
        scene, _ = scenario.generate()
        print(f"--- Sample Scene #{i + 1} ---")
        for obj in scene.objects:
            obj_type = type(obj).__name__
            pos = (round(obj.position.x, 2), round(obj.position.y, 2), round(obj.position.z, 2))
            speed = round(getattr(obj, 'speed', 0.0), 2)
            heading = round(getattr(obj, 'heading', 0.0), 3)
            print(f"  [{obj_type:10s}] Pos: {pos} | Speed: {speed:5.2f} m/s | Heading: {heading:6.3f} rad")
        print()

    print("[SCENIC] Sampling completed successfully.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_scenic = os.path.join(script_dir, "scenic_scenarios", "indian_mixed_traffic.scenic")
    
    scenic_file = sys.argv[1] if len(sys.argv) > 1 else default_scenic
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    sample_scenic_scenario(scenic_file, samples)
