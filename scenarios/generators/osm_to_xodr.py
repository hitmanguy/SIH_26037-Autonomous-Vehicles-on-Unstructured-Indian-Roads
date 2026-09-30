"""
=============================================================================
OSM TO ASAM OPENDRIVE CONVERTER FOR INDIAN ROAD NETWORKS
=============================================================================
Converts OpenStreetMap (.osm) XML files into standard ASAM OpenDRIVE (.xodr)
road networks compatible with MATLAB Automated Driving Toolbox and Driving
Scenario Designer.

Usage:
    python osm_to_xodr.py [input_file.osm] [output_file.xodr]

Dependencies:
    pip install scenariogeneration numpy
=============================================================================
"""

import sys
import os
import math
import xml.etree.ElementTree as ET
import numpy as np
from scenariogeneration import xodr

EARTH_RADIUS_METERS = 6378137.0


def latlon_to_xy(lat, lon, origin_lat, origin_lon):
    """Converts WGS-84 lat/lon to local Cartesian meters relative to origin."""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(origin_lat)
    lon0_rad = math.radians(origin_lon)

    dx = (lon_rad - lon0_rad) * EARTH_RADIUS_METERS * math.cos(lat0_rad)
    dy = (lat_rad - lat0_rad) * EARTH_RADIUS_METERS
    return dx, dy


def parse_osm(osm_path):
    """Parses OSM XML file and extracts nodes and highway ways."""
    tree = ET.parse(osm_path)
    root = tree.getroot()

    nodes = {}
    for node in root.findall('node'):
        node_id = node.get('id')
        lat = float(node.get('lat'))
        lon = float(node.get('lon'))
        nodes[node_id] = (lat, lon)

    if not nodes:
        raise ValueError(f"No nodes found in OSM file: {osm_path}")

    # Compute origin as center of all nodes
    lats = [n[0] for n in nodes.values()]
    lons = [n[1] for n in nodes.values()]
    origin_lat = sum(lats) / len(lats)
    origin_lon = sum(lons) / len(lons)

    # Convert nodes to local Cartesian coordinates (x, y)
    projected_nodes = {}
    for nid, (lat, lon) in nodes.items():
        x, y = latlon_to_xy(lat, lon, origin_lat, origin_lon)
        projected_nodes[nid] = (x, y)

    # Parse highway ways
    ways = []
    for way in root.findall('way'):
        tags = {tag.get('k'): tag.get('v') for tag in way.findall('tag')}
        if 'highway' in tags:
            nd_refs = [nd.get('ref') for nd in way.findall('nd')]
            pts = [projected_nodes[ref] for ref in nd_refs if ref in projected_nodes]
            if len(pts) >= 2:
                ways.append({
                    'id': way.get('id'),
                    'name': tags.get('name', f"Road_{way.get('id')}"),
                    'highway': tags.get('highway', 'secondary'),
                    'lanes': int(tags.get('lanes', 2)),
                    'oneway': tags.get('oneway', 'no').lower() in ('yes', 'true', '1'),
                    'points': pts
                })

    return projected_nodes, ways, (origin_lat, origin_lon)


def convert_osm_to_xodr(osm_path, output_xodr_path):
    """Converts OSM file to ASAM OpenDRIVE (.xodr)."""
    print(f"[OSM2ODR] Parsing OpenStreetMap: {osm_path}")
    nodes, ways, origin = parse_osm(osm_path)
    print(f"[OSM2ODR] Extracted {len(nodes)} nodes, {len(ways)} highway ways. Origin: {origin}")

    odr = xodr.OpenDrive(os.path.splitext(os.path.basename(osm_path))[0])

    road_id = 1
    for way in ways:
        pts = way['points']
        lanes_total = way['lanes']
        oneway = way['oneway']

        if oneway:
            left_lanes = 0
            right_lanes = max(1, lanes_total)
        else:
            right_lanes = max(1, lanes_total // 2)
            left_lanes = max(1, lanes_total - right_lanes)

        lane_width = 3.5  # standard lane width in meters

        # Build geometry segment chain
        plan_elements = []
        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            seg_len = math.hypot(dx, dy)
            if seg_len < 0.5:
                continue

            seg_heading = math.atan2(dy, dx)
            plan_elements.append((p1[0], p1[1], seg_heading, seg_len))

        if not plan_elements:
            continue

        # Create OpenDRIVE road geometries
        for sub_idx, (sx, sy, shdg, slen) in enumerate(plan_elements):
            sub_id = road_id * 100 + sub_idx
            road_geom = [xodr.Line(slen)]
            
            road = xodr.create_road(
                road_geom,
                id=sub_id,
                left_lanes=left_lanes,
                right_lanes=right_lanes,
                lane_width=lane_width
            )
            road.name = f"{way['name']}_Seg{sub_idx}"
            road.planview.set_start_point(sx, sy, shdg)
            odr.add_road(road)

        road_id += 1

    odr.adjust_roads_and_lanes()
    os.makedirs(os.path.dirname(output_xodr_path), exist_ok=True)
    odr.write_xml(output_xodr_path)
    print(f"[OSM2ODR] Successfully exported OpenDRIVE network to: {output_xodr_path}")
    return output_xodr_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_osm = os.path.join(script_dir, "..", "osm_maps", "bangalore_indiranagar.osm")
    default_xodr = os.path.join(script_dir, "..", "maps_xodr", "Bangalore_Indiranagar.xodr")

    in_osm = sys.argv[1] if len(sys.argv) > 1 else default_osm
    out_xodr = sys.argv[2] if len(sys.argv) > 2 else default_xodr

    convert_osm_to_xodr(in_osm, out_xodr)
