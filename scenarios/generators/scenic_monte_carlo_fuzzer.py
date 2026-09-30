"""
=============================================================================
AUTOMATED MONTE CARLO FUZZING PIPELINE (SCENIC -> XOSC -> MATLAB)
=============================================================================
Probabilistically stress-tests autonomous driving scenarios using UC Berkeley
Scenic to generate randomized traffic variations, executes them in MATLAB
drivingScenario, and verifies safety envelopes using 2D OBB SAT Collision Audit.

NOTE: All generated OpenSCENARIO (.xosc) files are strictly TEMPORARY and are
automatically purged after each simulation run to prevent repository clutter.

Usage:
    python scenic_monte_carlo_fuzzer.py [--samples 5] [--duration 10.0]
=============================================================================
"""

import os
import sys
import argparse
import subprocess
import shutil
import numpy as np

# Ensure path to verification package
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
VERIF_DIR = os.path.join(SCENARIOS_ROOT, "verification")
if VERIF_DIR not in sys.path:
    sys.path.insert(0, VERIF_DIR)

try:
    import scenic
    from trajectory_collision_verifier import audit_trajectories
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    sys.exit(1)

MATLAB_EXE = r"E:\Program Files\MATLAB\R2025b\bin\matlab.exe"
ROAD_XODR = os.path.join(SCENARIOS_ROOT, "maps_xodr", "Indian_Urban_Arterial.xodr")
SCENIC_MODEL = os.path.join(SCRIPT_DIR, "scenic_scenarios", "indian_mixed_traffic.scenic")
LOG_MAT_FILE = os.path.join(VERIF_DIR, "actor_trajectories_log.mat")


def build_temp_xosc(ego_speed: float, auto_speed: float, auto_start_x: float,
                    cutin_dist: float, cutin_duration: float, output_path: str):
    """
    Constructs an authentic ASAM OpenSCENARIO 1.0 XML file with sampled parameters.
    """
    road_rel = os.path.relpath(ROAD_XODR, os.path.dirname(output_path)).replace("\\", "/")

    xosc_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="1" date="2026-09-29T00:00:00" description="Indian Auto-Rickshaw Aggressive Cut-In (Scenic Monte Carlo Sample)" author="Scenic Fuzzer"/>
  <ParameterDeclarations>
    <ParameterDeclaration name="Ego_Speed" parameterType="double" value="{ego_speed:.2f}"/>
    <ParameterDeclaration name="Auto_Speed" parameterType="double" value="{auto_speed:.2f}"/>
    <ParameterDeclaration name="Auto_StartX" parameterType="double" value="{auto_start_x:.2f}"/>
    <ParameterDeclaration name="CutIn_Distance" parameterType="double" value="{cutin_dist:.2f}"/>
    <ParameterDeclaration name="CutIn_Duration" parameterType="double" value="{cutin_duration:.2f}"/>
  </ParameterDeclarations>
  <CatalogLocations>
    <VehicleCatalog><Directory path="../catalogs/vehicles"/></VehicleCatalog>
  </CatalogLocations>
  <RoadNetwork>
    <LogicFile filepath="{road_rel}"/>
  </RoadNetwork>
  <Entities>
    <ScenarioObject name="Ego">
      <Vehicle name="EgoCar_Blue" vehicleCategory="car">
        <Dimensions width="1.82" length="4.45" height="1.48"/>
        <Performance maxSpeed="55.5" maxAcceleration="6.0" maxDeceleration="9.0"/>
        <BoundingBox><Center x="1.4" y="0.0" z="0.7"/><Dimension width="1.82" length="4.45" height="1.48"/></BoundingBox>
        <Axles><FrontAxle maxSteering="0.5" wheelDiameter="0.6" trackWidth="1.55" positionX="2.8" positionZ="0.3"/>
               <RearAxle maxSteering="0.0" wheelDiameter="0.6" trackWidth="1.55" positionX="0.0" positionZ="0.3"/></Axles>
      </Vehicle>
    </ScenarioObject>
    <ScenarioObject name="AutoRickshaw">
      <Vehicle name="AutoRickshaw_Bajaj" vehicleCategory="car">
        <Dimensions width="1.30" length="2.65" height="1.70"/>
        <Performance maxSpeed="15.0" maxAcceleration="3.0" maxDeceleration="6.0"/>
        <BoundingBox><Center x="1.0" y="0.0" z="0.85"/><Dimension width="1.30" length="2.65" height="1.70"/></BoundingBox>
        <Axles><FrontAxle maxSteering="0.6" wheelDiameter="0.4" trackWidth="1.1" positionX="1.8" positionZ="0.2"/>
               <RearAxle maxSteering="0.0" wheelDiameter="0.4" trackWidth="1.1" positionX="0.0" positionZ="0.2"/></Axles>
      </Vehicle>
    </ScenarioObject>
    <ScenarioObject name="SlowTruck">
      <Vehicle name="TataTruck" vehicleCategory="truck">
        <Dimensions width="2.45" length="7.50" height="3.20"/>
        <Performance maxSpeed="20.0" maxAcceleration="2.0" maxDeceleration="5.0"/>
        <BoundingBox><Center x="2.5" y="0.0" z="1.6"/><Dimension width="2.45" length="7.50" height="3.20"/></BoundingBox>
        <Axles><FrontAxle maxSteering="0.4" wheelDiameter="1.0" trackWidth="2.0" positionX="5.0" positionZ="0.5"/>
               <RearAxle maxSteering="0.0" wheelDiameter="1.0" trackWidth="2.0" positionX="0.0" positionZ="0.5"/></Axles>
      </Vehicle>
    </ScenarioObject>
  </Entities>
  <Storyboard>
    <Init>
      <Actions>
        <Private entityRef="Ego">
          <PrivateAction><TeleportAction><Position><WorldPosition x="20.0" y="-1.75" z="0.0" h="0.0"/></Position></TeleportAction></PrivateAction>
          <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="$Ego_Speed"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
        </Private>
        <Private entityRef="AutoRickshaw">
          <PrivateAction><TeleportAction><Position><WorldPosition x="$Auto_StartX" y="-5.25" z="0.0" h="0.0"/></Position></TeleportAction></PrivateAction>
          <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="$Auto_Speed"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
        </Private>
        <Private entityRef="SlowTruck">
          <PrivateAction><TeleportAction><Position><WorldPosition x="110.0" y="-5.25" z="0.0" h="0.0"/></Position></TeleportAction></PrivateAction>
          <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="5.55"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
        </Private>
      </Actions>
    </Init>
    <Story name="IndianCutInStory">
      <Act name="CutInAct">
        <ManeuverGroup maximumExecutionCount="1" name="ManeuverGroup_Auto">
          <Actors selectTriggeringEntities="false"><EntityRef entityRef="AutoRickshaw"/></Actors>
          <Maneuver name="Maneuver_CutIn">
            <Event name="CutInEvent" priority="overwrite">
              <Action name="CutInLaneChange">
                <PrivateAction>
                  <LateralAction>
                    <LaneChangeAction>
                      <LaneChangeActionDynamics dynamicsShape="sinusoidal" value="$CutIn_Duration" dynamicsDimension="time"/>
                      <LaneChangeTarget><RelativeTargetLane entityRef="Ego" value="0"/></LaneChangeTarget>
                    </LaneChangeAction>
                  </LateralAction>
                </PrivateAction>
              </Action>
              <StartTrigger>
                <ConditionGroup>
                  <Condition name="TimeCondition" delay="0" conditionEdge="rising">
                    <ByValueCondition><SimulationTimeCondition value="3.5" rule="greaterThan"/></ByValueCondition>
                  </Condition>
                </ConditionGroup>
              </StartTrigger>
            </Event>
          </Maneuver>
        </ManeuverGroup>
      </Act>
    </Story>
    <StopTrigger/>
  </Storyboard>
</OpenSCENARIO>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xosc_content)


def run_matlab_batch(xosc_path: str, duration: float = 8.0) -> bool:
    """Executes a single headless simulation run in MATLAB."""
    road_escaped = ROAD_XODR.replace("\\", "/")
    xosc_escaped = xosc_path.replace("\\", "/")
    matlab_cmd = (
        f"cd('{SCENARIOS_ROOT.replace(chr(92), '/')}'); "
        f"run_intersection('{xosc_escaped}', '{road_escaped}', 'headless', {duration});"
    )
    cmd = [MATLAB_EXE, "-batch", matlab_cmd]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        print("[FUZZER] MATLAB execution timed out after 120s.")
        return False


def run_monte_carlo_fuzzing(num_samples: int = 3, sim_duration: float = 8.0):
    """
    Main Monte Carlo fuzzing loop:
    1. Loads Scenic probabilistic distribution.
    2. Samples randomized traffic scenarios.
    3. Generates TEMPORARY OpenSCENARIO (.xosc) file.
    4. Simulates in MATLAB drivingScenario.
    5. Audits safety with SAT OBB collision verifier.
    6. Automatically deletes the temporary .xosc.
    7. Displays comprehensive safety envelope report.
    """
    print("=" * 80)
    print("AUTOMATED MONTE CARLO FUZZING PIPELINE (SCENIC -> XOSC -> MATLAB)")
    print(f"Target Samples: {num_samples} | Simulation Duration: {sim_duration}s")
    print(f"Scenic Source:  {SCENIC_MODEL}")
    print("Temporary Storage: Auto-purged after every run (0 storage footprint)")
    print("=" * 80)

    # 1. Compile Scenic Program
    print("\n[STEP 1/3] Compiling UC Berkeley Scenic probabilistic model...")
    scenario = scenic.scenarioFromFile(SCENIC_MODEL)
    print("Scenario compiled successfully. Initiating randomized fuzzing loop...\n")

    results = []
    temp_dir = os.path.join(SCENARIOS_ROOT, "generators", ".tmp_scenic_fuzzing")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        for idx in range(1, num_samples + 1):
            print(f"{'-' * 35} RUN {idx}/{num_samples} {'-' * 35}")

            # Sample concrete scene from probabilistic distribution
            scene, _ = scenario.generate()
            
            # Extract randomized parameters
            # Ego vehicle is obj 0, Auto is obj 1
            auto_obj = scene.objects[1]
            auto_start_x = float(auto_obj.position.x)
            
            # Sample speeds from Scenic distributions or realistic ranges
            ego_speed = float(np.random.uniform(9.5, 11.5))       # 34 - 41 km/h
            auto_speed = float(np.random.uniform(7.0, 8.8))       # 25 - 32 km/h
            cutin_dist = float(np.random.uniform(25.0, 35.0))     # 25 - 35 m straight before cut
            cutin_duration = float(np.random.uniform(2.0, 2.6))   # 2.0 - 2.6 s swerve duration

            print(f"  [Sampled Parameters]")
            print(f"    * Auto Start X:     {auto_start_x:6.2f} m (Headway: {auto_start_x - 20.0:.2f} m)")
            print(f"    * Ego Speed:        {ego_speed:6.2f} m/s ({ego_speed * 3.6:5.1f} km/h)")
            print(f"    * Auto Speed:       {auto_speed:6.2f} m/s ({auto_speed * 3.6:5.1f} km/h)")
            print(f"    * Cut-in Distance:  {cutin_dist:6.2f} m")
            print(f"    * Cut-in Duration:  {cutin_duration:6.2f} s")

            # 2. Write temporary .xosc file
            temp_xosc = os.path.join(temp_dir, f"temp_Indian_AutoCutIn_sample_{idx:03d}.xosc")
            build_temp_xosc(ego_speed, auto_speed, auto_start_x, cutin_dist, cutin_duration, temp_xosc)
            print(f"  [XOSC] Created temporary scenario: {os.path.basename(temp_xosc)}")

            # 3. Execute in MATLAB
            print(f"  [MATLAB] Simulating headless in drivingScenario...")
            matlab_ok = run_matlab_batch(temp_xosc, duration=sim_duration)

            # 4. Audit safety using SAT OBB Collision Verifier
            min_clearance = 0.0
            is_collision_free = False
            
            if matlab_ok and os.path.exists(LOG_MAT_FILE):
                print(f"  [SAT OBB] Auditing 2D oriented bounding box trajectories...")
                success, pair_dists = audit_trajectories(LOG_MAT_FILE)
                is_collision_free = success
                if pair_dists:
                    min_clearance = min(pair_dists.values())
            else:
                print(f"  [ERROR] MATLAB simulation failed for sample {idx}!")

            # 5. IMMEDIATELY PURGE TEMPORARY .XOSC
            if os.path.exists(temp_xosc):
                os.remove(temp_xosc)
                print(f"  [PURGE] Successfully deleted temporary {os.path.basename(temp_xosc)}")

            status_str = "PASS (Zero Collisions)" if is_collision_free else "CRITICAL / COLLISION"
            print(f"  --> Result: {status_str} | Min Body Clearance: {min_clearance:.2f} m\n")

            results.append({
                'run': idx,
                'auto_x': auto_start_x,
                'ego_speed': ego_speed,
                'auto_speed': auto_speed,
                'cutin_dist': cutin_dist,
                'min_clearance': min_clearance,
                'passed': is_collision_free
            })

    finally:
        # Final cleanup of temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    # 6. Display Monte Carlo Safety Envelope Summary Report
    print("=" * 80)
    print("MONTE CARLO FUZZING SAFETY ENVELOPE REPORT")
    print("=" * 80)
    passed_runs = sum(1 for r in results if r['passed'])
    pass_rate = (passed_runs / len(results) * 100) if results else 0.0
    clearances = [r['min_clearance'] for r in results]
    worst_clearance = min(clearances) if clearances else 0.0
    best_clearance = max(clearances) if clearances else 0.0
    mean_clearance = np.mean(clearances) if clearances else 0.0

    print(f"Total Scenarios Evaluated:        {len(results)}")
    print(f"Pass Rate (Zero Collisions):      {passed_runs}/{len(results)} ({pass_rate:.1f}%)")
    print(f"Closest Proximity (Worst Margin): {worst_clearance:.2f} m")
    print(f"Mean Safety Margin:               {mean_clearance:.2f} m")
    print(f"Maximum Separation:               {best_clearance:.2f} m\n")

    print(f"{'Run':<5} | {'Auto X':<10} | {'Ego Speed':<12} | {'Auto Speed':<12} | {'Min Clearance':<15} | {'Outcome':<10}")
    print("-" * 75)
    for r in results:
        out = "PASS" if r['passed'] else "FAIL"
        print(f"#{r['run']:<4} | {r['auto_x']:6.2f} m   | {r['ego_speed']*3.6:5.1f} km/h   | {r['auto_speed']*3.6:5.1f} km/h   | {r['min_clearance']:6.2f} m        | {out:<10}")
    print("-" * 75)
    print("All temporary files automatically cleaned. No repository clutter created.\n")
    return pass_rate == 100.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Fuzzing Pipeline (Scenic -> XOSC -> MATLAB)")
    parser.add_argument("--samples", type=int, default=3, help="Number of probabilistic samples to run (default: 3)")
    parser.add_argument("--duration", type=float, default=6.0, help="Simulation duration in seconds (default: 6.0)")
    args = parser.parse_args()

    success = run_monte_carlo_fuzzing(num_samples=args.samples, sim_duration=args.duration)
    sys.exit(0 if success else 1)
