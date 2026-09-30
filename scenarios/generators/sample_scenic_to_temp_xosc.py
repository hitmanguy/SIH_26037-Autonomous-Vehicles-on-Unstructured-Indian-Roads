"""
=============================================================================
SAMPLE SCENIC TO TEMPORARY OPENSENARIO (.xosc)
=============================================================================
Samples a single concrete traffic scenario from UC Berkeley Scenic and
writes an authentic, temporary OpenSCENARIO (.xosc) file for MATLAB.

Usage:
    python sample_scenic_to_temp_xosc.py [--road Indian_Urban_Arterial.xodr]
=============================================================================
"""

import os
import sys
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIOS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SCENIC_MODEL = os.path.join(SCRIPT_DIR, "scenic_scenarios", "indian_mixed_traffic.scenic")
MAPS_DIR = os.path.join(SCENARIOS_ROOT, "maps_xodr")

try:
    import scenic
except ImportError:
    print("[ERROR] Scenic is not installed in current Python environment.", file=sys.stderr)
    sys.exit(1)


def generate_temp_scenic_xosc(road_name: str = "Indian_Urban_Arterial.xodr") -> str:
    road_path = os.path.join(MAPS_DIR, road_name)
    if not os.path.isfile(road_path):
        road_path = os.path.join(MAPS_DIR, "Indian_Urban_Arterial.xodr")

    scenario = scenic.scenarioFromFile(SCENIC_MODEL)
    scene, _ = scenario.generate()

    # Extract randomized positions and speeds
    auto_obj = scene.objects[1]
    auto_start_x = float(auto_obj.position.x)

    ego_speed = float(np.random.uniform(9.5, 11.5))       # 34 - 41 km/h
    auto_speed = float(np.random.uniform(7.0, 8.8))       # 25 - 32 km/h
    cutin_dist = float(np.random.uniform(25.0, 35.0))     # 25 - 35 m straight before cut
    cutin_duration = float(np.random.uniform(2.0, 2.6))   # 2.0 - 2.6 s swerve duration

    temp_dir = os.path.join(SCENARIOS_ROOT, "scenarios_xosc", ".tmp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_xosc = os.path.join(temp_dir, "temp_scenic_active.xosc")

    road_rel = os.path.relpath(road_path, os.path.dirname(temp_xosc)).replace("\\", "/")

    xosc_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="1" date="2026-09-29T00:00:00" description="Indian Auto-Rickshaw Aggressive Cut-In (Scenic Sample)" author="Scenic Generator"/>
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
    with open(temp_xosc, "w", encoding="utf-8") as f:
        f.write(xosc_content)

    print(f"TEMP_XOSC:{temp_xosc}")
    return temp_xosc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--road", type=str, default="Indian_Urban_Arterial.xodr")
    args = parser.parse_args()

    generate_temp_scenic_xosc(args.road)
