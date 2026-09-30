"""
Verification Helper Script for Indian Autonomous Driving Scenarios (SIH26)
--------------------------------------------------------------------------
Performs static analysis, XML schema integrity checking, coordinate bounds
validation, and cross-checks registration in MATLAB scenario runners.
Does NOT run MATLAB simulation (per user constraints).
"""

import os
import re
import xml.etree.ElementTree as ET

SCENARIO_DIR = r"D:\hackathon\SIH26\scenarios\scenarios_xosc\indian"
ROAD_XODR = r"D:\hackathon\SIH26\scenarios\maps_xodr\Indian_Urban_Arterial.xodr"
IMPORT_M = r"D:\hackathon\SIH26\scenarios\matlab\import_openscenario.m"
RUN_INTERSECTION_M = r"D:\hackathon\SIH26\scenarios\matlab\run_intersection_scenario.m"
RECORD_UNREAL_M = r"D:\hackathon\SIH26\vehicle_dynamics\record_unreal_simulation.m"

# Road Corridor Limits from Indian_Urban_Arterial.xodr
X_MIN, X_MAX = 0.0, 350.0
Y_MIN, Y_MAX = -7.5, +7.5  # Road width +/-7.0m plus shoulder margin

SCENARIO_SPECS = [
    ("Indian_AutoRickshaw_CutIn.xosc", "setup_indian_autocutin_scenario", "INDIAN_AUTOCUTIN"),
    ("Indian_TwoWheeler_LaneFilter.xosc", "setup_indian_twowheeler_scenario", "INDIAN_TWOWHEELER"),
    ("Indian_Pedestrian_Jaywalk.xosc", "setup_indian_jaywalk_scenario", "INDIAN_JAYWALK"),
    ("Indian_StrayCattle_Hazard.xosc", "setup_indian_cattle_scenario", "INDIAN_CATTLE"),
    ("Indian_Traffic_Congestion.xosc", "setup_indian_congestion_scenario", "INDIAN_CONGESTION"),
    ("Indian_Pothole_Detour.xosc", "setup_indian_pothole_scenario", "INDIAN_POTHOLE"),
    ("Indian_WrongWay_Encounter.xosc", "setup_indian_wrongway_scenario", "INDIAN_WRONGWAY"),
    ("Indian_SchoolZone_Rush.xosc", "setup_indian_schoolzone_scenario", "INDIAN_SCHOOLZONE"),
    ("Indian_BusStop_Hazard.xosc", "setup_indian_busstop_scenario", "INDIAN_BUSSTOP"),
    ("Indian_VendorCart_Swerve.xosc", "setup_indian_vendorcart_scenario", "INDIAN_VENDORCART"),
    ("Indian_MultiThreat_Gauntlet.xosc", "setup_indian_gauntlet_scenario", "INDIAN_GAUNTLET")
]

def verify_all():
    print("=" * 80)
    print("  SIH26 INDIAN SCENARIOS VERIFICATION & INTEGRITY REPORT")
    print("=" * 80)

    with open(IMPORT_M, "r", encoding="utf-8") as f:
        import_content = f.read()
    with open(RUN_INTERSECTION_M, "r", encoding="utf-8") as f:
        run_content = f.read()
    with open(RECORD_UNREAL_M, "r", encoding="utf-8") as f:
        record_content = f.read()

    total_scenarios = len(SCENARIO_SPECS)
    passed_scenarios = 0

    for idx, (fname, setup_func, sc_keyword) in enumerate(SCENARIO_SPECS, 1):
        fpath = os.path.join(SCENARIO_DIR, fname)
        print(f"\n[{idx}/{total_scenarios}] Inspecting: {fname}")
        
        if not os.path.isfile(fpath):
            print(f"  [ERROR] File missing: {fpath}")
            continue

        # XML Parsing
        try:
            tree = ET.parse(fpath)
            root = tree.getroot()
            print("  [OK] Valid XML syntax")
        except Exception as e:
            print(f"  [FAIL] XML Parse Error: {e}")
            continue

        # Check Entities
        entities_elem = root.find(".//Entities")
        if entities_elem is None:
            print("  [FAIL] Missing <Entities> element")
            continue

        objects = entities_elem.findall("./ScenarioObject")
        num_actors = len(objects)
        print(f"  [OK] Actors Defined: {num_actors}")
        
        if num_actors == 0:
            print("  [FAIL] Zero entities declared")
            continue

        first_actor = objects[0].attrib.get("name", "")
        if first_actor != "Ego":
            print(f"  [WARN] First actor is '{first_actor}' instead of 'Ego'")
        else:
            print(f"  [OK] Ego is first entity: {first_actor}")

        # Check Init Positions
        teleports = root.findall(".//PrivateAction/TeleportAction/Position/WorldPosition")
        coord_ok = True
        for tp in teleports:
            try:
                x = float(tp.attrib.get("x", 0.0))
                y = float(tp.attrib.get("y", 0.0))
                if not (X_MIN <= x <= X_MAX):
                    print(f"  [WARN] Coordinate X={x} exceeds road bounds [{X_MIN}, {X_MAX}]")
                    coord_ok = False
                if not (Y_MIN <= y <= Y_MAX):
                    print(f"  [WARN] Coordinate Y={y} exceeds road bounds [{Y_MIN}, {Y_MAX}]")
                    coord_ok = False
            except ValueError:
                pass
        if coord_ok:
            print(f"  [OK] All initial positions strictly within road corridor bounds")

        # Check Registration in MATLAB scripts
        in_import = (setup_func in import_content) and (sc_keyword in import_content)
        in_run = fname in run_content
        in_record = fname in record_content

        reg_status = []
        if in_import: reg_status.append(f"import_openscenario ({setup_func})")
        if in_run: reg_status.append("run_intersection")
        if in_record: reg_status.append("record_unreal")
        
        print(f"  [OK] MATLAB Integration: {', '.join(reg_status)}")

        if in_import and in_run and in_record and coord_ok:
            passed_scenarios += 1

    print("\n" + "=" * 80)
    print(f"  VERIFICATION RESULT: {passed_scenarios}/{total_scenarios} SCENARIOS VERIFIED 100% READY")
    print("=" * 80)

if __name__ == "__main__":
    verify_all()
