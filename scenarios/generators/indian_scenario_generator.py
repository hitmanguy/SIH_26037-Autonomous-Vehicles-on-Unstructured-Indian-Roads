"""
=============================================================================
INDIAN TRAFFIC SCENARIO GENERATOR USING SCENARIOGENERATION (ASAM OPENX)
=============================================================================
Programmatically generates authentic Indian road networks (.xodr) and dynamic
traffic scenarios (.xosc) modeling realistic edge-cases:
1. Auto-Rickshaw Aggressive Cut-In
2. Two-Wheeler Lane Filtering & Splitting
3. Mid-Block Jaywalking Pedestrian Swarm
4. Stray Roadway Hazard / Cattle Avoidance

Dependencies:
    pip install scenariogeneration numpy
=============================================================================
"""

import os
import math
from scenariogeneration import xodr, xosc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(SCRIPT_DIR, "..", "maps_xodr")
SCENARIOS_DIR = os.path.join(SCRIPT_DIR, "..", "scenarios_xosc", "indian")


def generate_indian_roads():
    """Generates authentic Indian road layouts in ASAM OpenDRIVE format."""
    os.makedirs(MAPS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Indian Urban Arterial (4-lane undivided, 300m straight corridor)
    # -------------------------------------------------------------------------
    odr_arterial = xodr.OpenDrive("Indian_Urban_Arterial")
    road_geom = [xodr.Line(350.0)]
    
    # 2 lanes left, 2 lanes right, 3.5m driving lanes
    arterial_road = xodr.create_road(
        road_geom,
        id=1,
        left_lanes=2,
        right_lanes=2,
        lane_width=3.5
    )
    arterial_road.name = "Indian_4Lane_Arterial"
    arterial_road.planview.set_start_point(0.0, 0.0, 0.0)
    odr_arterial.add_road(arterial_road)
    odr_arterial.adjust_roads_and_lanes()
    
    arterial_path = os.path.join(MAPS_DIR, "Indian_Urban_Arterial.xodr")
    odr_arterial.write_xml(arterial_path)
    print(f"[INDIAN-GEN] Generated: {arterial_path}")

    # -------------------------------------------------------------------------
    # 2. Indian Junction Chowk (4-way intersection)
    # -------------------------------------------------------------------------
    odr_chowk = xodr.OpenDrive("Indian_Junction_Chowk")
    
    # West approach
    r_west = xodr.create_road([xodr.Line(150.0)], id=1, left_lanes=2, right_lanes=2, lane_width=3.5)
    r_west.name = "West_Arm"
    r_west.planview.set_start_point(-150.0, 0.0, 0.0)
    odr_chowk.add_road(r_west)

    # East approach
    r_east = xodr.create_road([xodr.Line(150.0)], id=2, left_lanes=2, right_lanes=2, lane_width=3.5)
    r_east.name = "East_Arm"
    r_east.planview.set_start_point(0.0, 0.0, 0.0)
    odr_chowk.add_road(r_east)

    # South approach
    r_south = xodr.create_road([xodr.Line(150.0)], id=3, left_lanes=1, right_lanes=1, lane_width=3.5)
    r_south.name = "South_Arm"
    r_south.planview.set_start_point(0.0, -150.0, math.pi / 2)
    odr_chowk.add_road(r_south)

    # North approach
    r_north = xodr.create_road([xodr.Line(150.0)], id=4, left_lanes=1, right_lanes=1, lane_width=3.5)
    r_north.name = "North_Arm"
    r_north.planview.set_start_point(0.0, 0.0, math.pi / 2)
    odr_chowk.add_road(r_north)

    odr_chowk.adjust_roads_and_lanes()
    chowk_path = os.path.join(MAPS_DIR, "Indian_Junction_Chowk.xodr")
    odr_chowk.write_xml(chowk_path)
    print(f"[INDIAN-GEN] Generated: {chowk_path}")

    return arterial_path, chowk_path


def generate_indian_scenarios():
    """Generates standard ASAM OpenSCENARIO (.xosc) files for Indian traffic."""
    os.makedirs(SCENARIOS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Scenario 1: Indian Auto-Rickshaw Sudden Cut-In
    # -------------------------------------------------------------------------
    xosc_cutin = os.path.join(SCENARIOS_DIR, "Indian_AutoRickshaw_CutIn.xosc")
    with open(xosc_cutin, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="1" date="2026-09-29T00:00:00" description="Indian Auto-Rickshaw Aggressive Cut-In" author="SIH26 Autonomous Driving Team"/>
  <ParameterDeclarations>
    <ParameterDeclaration name="Ego_Speed" parameterType="double" value="11.11"/> <!-- 40 km/h -->
    <ParameterDeclaration name="Auto_Speed" parameterType="double" value="8.33"/>  <!-- 30 km/h -->
    <ParameterDeclaration name="CutIn_Time" parameterType="double" value="4.0"/>
  </ParameterDeclarations>
  <CatalogLocations>
    <VehicleCatalog><Directory path="../catalogs/vehicles"/></VehicleCatalog>
  </CatalogLocations>
  <RoadNetwork>
    <LogicFile filepath="../../maps_xodr/Indian_Urban_Arterial.xodr"/>
  </RoadNetwork>
  <Entities>
    <ScenarioObject name="Ego">
      <Vehicle name="EgoVehicle" vehicleCategory="car">
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
          <PrivateAction><TeleportAction><Position><WorldPosition x="55.0" y="-5.25" z="0.0" h="0.0"/></Position></TeleportAction></PrivateAction>
          <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="$Auto_Speed"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
        </Private>
        <Private entityRef="SlowTruck">
          <PrivateAction><TeleportAction><Position><WorldPosition x="110.0" y="-1.75" z="0.0" h="0.0"/></Position></TeleportAction></PrivateAction>
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
                      <LaneChangeActionDynamics dynamicsShape="sinusoidal" value="1.5" dynamicsDimension="time"/>
                      <LaneChangeTarget><RelativeTargetLane entityRef="Ego" value="0"/></LaneChangeTarget>
                    </LaneChangeAction>
                  </LateralAction>
                </PrivateAction>
              </Action>
              <StartTrigger>
                <ConditionGroup>
                  <Condition name="TimeCondition" delay="0" conditionEdge="rising">
                    <ByValueCondition><SimulationTimeCondition value="$CutIn_Time" rule="greaterThan"/></ByValueCondition>
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
""")
    print(f"[INDIAN-GEN] Generated: {xosc_cutin}")

    # -------------------------------------------------------------------------
    # Scenario 2: Indian Two-Wheeler Lane Filtering
    # -------------------------------------------------------------------------
    xosc_twowheeler = os.path.join(SCENARIOS_DIR, "Indian_TwoWheeler_LaneFilter.xosc")
    with open(xosc_twowheeler, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="1" date="2026-09-29T00:00:00" description="Two-Wheeler Motorcycle Filtering Between Lanes" author="SIH26 Autonomous Driving Team"/>
  <ParameterDeclarations>
    <ParameterDeclaration name="Ego_Speed" parameterType="double" value="8.33"/>  <!-- 30 km/h -->
    <ParameterDeclaration name="Bike_Speed" parameterType="double" value="13.88"/> <!-- 50 km/h -->
  </ParameterDeclarations>
  <RoadNetwork>
    <LogicFile filepath="../../maps_xodr/Indian_Urban_Arterial.xodr"/>
  </RoadNetwork>
  <Entities>
    <ScenarioObject name="Ego">
      <Vehicle name="EgoCar" vehicleCategory="car">
        <Dimensions width="1.80" length="4.50" height="1.50"/>
      </Vehicle>
    </ScenarioObject>
    <ScenarioObject name="Motorcycle">
      <Vehicle name="RoyalEnfield" vehicleCategory="motorbike">
        <Dimensions width="0.75" length="2.10" height="1.20"/>
      </Vehicle>
    </ScenarioObject>
    <ScenarioObject name="ParallelCar">
      <Vehicle name="MarutiSwift" vehicleCategory="car">
        <Dimensions width="1.75" length="3.90" height="1.50"/>
      </Vehicle>
    </ScenarioObject>
  </Entities>
  <Storyboard>
    <Init>
      <Actions>
        <Private entityRef="Ego">
          <PrivateAction><TeleportAction><Position><WorldPosition x="30.0" y="-1.75" z="0.0" h="0.0"/></Position></TeleportAction></PrivateAction>
          <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="$Ego_Speed"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
        </Private>
        <Private entityRef="ParallelCar">
          <PrivateAction><TeleportAction><Position><WorldPosition x="40.0" y="-5.25" z="0.0" h="0.0"/></Position></TeleportAction></PrivateAction>
          <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="6.94"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
        </Private>
        <Private entityRef="Motorcycle">
          <PrivateAction><TeleportAction><Position><WorldPosition x="5.0" y="-3.50" z="0.0" h="0.0"/></Position></TeleportAction></PrivateAction>
          <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="$Bike_Speed"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
        </Private>
      </Actions>
    </Init>
    <Story name="MotorcycleFilteringStory">
      <Act name="FilterAct">
        <ManeuverGroup maximumExecutionCount="1" name="MG_Motorcycle">
          <Actors selectTriggeringEntities="false"><EntityRef entityRef="Motorcycle"/></Actors>
          <Maneuver name="FilterWeave">
            <Event name="OvertakeEvent" priority="overwrite">
              <Action name="SpeedMaintain">
                <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="linear" value="1.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="$Bike_Speed"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
              </Action>
              <StartTrigger><ConditionGroup><Condition name="Immediate" delay="0" conditionEdge="none"><ByValueCondition><SimulationTimeCondition value="0.0" rule="greaterThan"/></ByValueCondition></Condition></ConditionGroup></StartTrigger>
            </Event>
          </Maneuver>
        </ManeuverGroup>
      </Act>
    </Story>
    <StopTrigger/>
  </Storyboard>
</OpenSCENARIO>
""")
    print(f"[INDIAN-GEN] Generated: {xosc_twowheeler}")

    # -------------------------------------------------------------------------
    # Scenario 3: Indian Mid-Block Jaywalking Pedestrians
    # -------------------------------------------------------------------------
    xosc_jaywalk = os.path.join(SCENARIOS_DIR, "Indian_Pedestrian_Jaywalk.xosc")
    with open(xosc_jaywalk, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="1" date="2026-09-29T00:00:00" description="Mid-block Pedestrians Jaywalking Between Parked Buses" author="SIH26 Autonomous Driving Team"/>
  <RoadNetwork>
    <LogicFile filepath="../../maps_xodr/Indian_Urban_Arterial.xodr"/>
  </RoadNetwork>
  <Entities>
    <ScenarioObject name="Ego">
      <Vehicle name="EgoCar" vehicleCategory="car"><Dimensions width="1.80" length="4.50" height="1.50"/></Vehicle>
    </ScenarioObject>
    <ScenarioObject name="ParkedBus">
      <Vehicle name="BMTC_Bus" vehicleCategory="bus"><Dimensions width="2.60" length="11.00" height="3.20"/></Vehicle>
    </ScenarioObject>
    <ScenarioObject name="Pedestrian">
      <Pedestrian name="Pedestrian_Jaywalker" pedestrianCategory="pedestrian" mass="70.0"><Dimensions width="0.45" length="0.28" height="1.75"/></Pedestrian>
    </ScenarioObject>
  </Entities>
  <Storyboard>
    <Init>
      <Actions>
        <Private entityRef="Ego">
          <PrivateAction><TeleportAction><Position><WorldPosition x="20.0" y="-1.75" z="0.0" h="0.0"/></Position></TeleportAction></PrivateAction>
          <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="11.11"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
        </Private>
        <Private entityRef="ParkedBus">
          <PrivateAction><TeleportAction><Position><WorldPosition x="90.0" y="-5.25" z="0.0" h="0.0"/></Position></TeleportAction></PrivateAction>
          <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="0.0"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
        </Private>
        <Private entityRef="Pedestrian">
          <PrivateAction><TeleportAction><Position><WorldPosition x="95.0" y="-7.0" z="0.0" h="1.5707"/></Position></TeleportAction></PrivateAction>
          <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="1.5"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
        </Private>
      </Actions>
    </Init>
    <Story name="JaywalkStory"><Act name="JaywalkAct"><ManeuverGroup maximumExecutionCount="1" name="MG_Ped"><Actors selectTriggeringEntities="false"><EntityRef entityRef="Pedestrian"/></Actors><Maneuver name="Maneuver_Cross"><Event name="CrossEvent" priority="overwrite"><Action name="CrossWalk"><PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="linear" value="0.5" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="1.5"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction></Action><StartTrigger><ConditionGroup><Condition name="TriggerDist" delay="0" conditionEdge="rising"><EntityCondition><DistanceCondition value="35.0" freespace="false" rule="lessThan"><EntityRef entityRef="Ego"/></DistanceCondition></EntityCondition></Condition></ConditionGroup></StartTrigger></Event></Maneuver></ManeuverGroup></Act></Story>
    <StopTrigger/>
  </Storyboard>
</OpenSCENARIO>
""")
    print(f"[INDIAN-GEN] Generated: {xosc_jaywalk}")

    # -------------------------------------------------------------------------
    # Scenario 4: Indian Stray Cattle / Road Hazard Avoidance
    # -------------------------------------------------------------------------
    xosc_cattle = os.path.join(SCENARIOS_DIR, "Indian_StrayCattle_Hazard.xosc")
    with open(xosc_cattle, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="1" date="2026-09-29T00:00:00" description="Stationary Cattle / Obstacle in Lane with Oncoming Traffic" author="SIH26 Autonomous Driving Team"/>
  <RoadNetwork>
    <LogicFile filepath="../../maps_xodr/Indian_Urban_Arterial.xodr"/>
  </RoadNetwork>
  <Entities>
    <ScenarioObject name="Ego">
      <Vehicle name="EgoCar" vehicleCategory="car"><Dimensions width="1.80" length="4.50" height="1.50"/></Vehicle>
    </ScenarioObject>
    <ScenarioObject name="StrayCattle">
      <!-- Modeled as barrier/cuboid object in roadway -->
      <MiscObject name="Cow_Obstacle" miscObjectCategory="obstacle" mass="400.0"><Dimensions width="1.10" length="2.10" height="1.40"/></MiscObject>
    </ScenarioObject>
    <ScenarioObject name="OncomingVehicle">
      <Vehicle name="OncomingBus" vehicleCategory="bus"><Dimensions width="2.50" length="10.50" height="3.20"/></Vehicle>
    </ScenarioObject>
  </Entities>
  <Storyboard>
    <Init>
      <Actions>
        <Private entityRef="Ego">
          <PrivateAction><TeleportAction><Position><WorldPosition x="20.0" y="-1.75" z="0.0" h="0.0"/></Position></TeleportAction></PrivateAction>
          <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="11.11"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
        </Private>
        <Private entityRef="StrayCattle">
          <PrivateAction><TeleportAction><Position><WorldPosition x="100.0" y="-1.75" z="0.0" h="0.0"/></Position></TeleportAction></PrivateAction>
          <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="0.0"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
        </Private>
        <Private entityRef="OncomingVehicle">
          <PrivateAction><TeleportAction><Position><WorldPosition x="250.0" y="1.75" z="0.0" h="3.14159"/></Position></TeleportAction></PrivateAction>
          <PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="9.72"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction>
        </Private>
      </Actions>
    </Init>
    <Story name="CattleStory"><Act name="CattleAct"><ManeuverGroup maximumExecutionCount="1" name="MG_Hold"><Actors selectTriggeringEntities="false"><EntityRef entityRef="StrayCattle"/></Actors><Maneuver name="M_Static"><Event name="HoldEvent" priority="overwrite"><Action name="HoldStill"><PrivateAction><LongitudinalAction><SpeedAction><SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/><SpeedActionTarget><AbsoluteTargetSpeed value="0.0"/></SpeedActionTarget></SpeedAction></LongitudinalAction></PrivateAction></Action><StartTrigger><ConditionGroup><Condition name="StartImmediate" delay="0" conditionEdge="none"><ByValueCondition><SimulationTimeCondition value="0.0" rule="greaterThan"/></ByValueCondition></Condition></ConditionGroup></StartTrigger></Event></Maneuver></ManeuverGroup></Act></Story>
    <StopTrigger/>
  </Storyboard>
</OpenSCENARIO>
""")
    print(f"[INDIAN-GEN] Generated: {xosc_cattle}")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING INDIAN ROAD MAPS (.XODR)")
    print("=" * 60)
    generate_indian_roads()

    print("\n" + "=" * 60)
    print("GENERATING INDIAN OPENSCENARIO FILES (.XOSC)")
    print("=" * 60)
    generate_indian_scenarios()
    print("\nAll Indian road networks and scenarios generated successfully!")
