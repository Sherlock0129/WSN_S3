import csv
import math
import numpy as np
import os
import sys

# Allow imports like from src.utils.scenario_loader import ...
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.scenario_loader import build_transform_from_S3

EARTH_RADIUS = 6371000.0

NEW_RIS_LOCAL = [
    {"name": "RIS_0_3_1", "pos": [2394.51, 1345.78, 767.15]},
    {"name": "RIS_0_3_2", "pos": [2544.51, 1405.78, 801.28]},
    {"name": "RIS_1_2_1", "pos": [2316.15, 405.82, 705.22]},
    {"name": "RIS_1_2_2", "pos": [2526.15, 405.82, 785.21]},
    {"name": "RIS_3_4",   "pos": [3001.83, 1405.78, 495.34]},
    {"name": "RIS_4_5",   "pos": [3741.32, 1758.86, 278.67]},
]

S3_PATH = 'data/S3.csv'
SINK_CSV_PATH = 'data/sink.csv'


def local_to_geodetic(final_xyz: np.ndarray, origin, R: np.ndarray):
    """Invert the transform in scenario_loader: final = R @ ENU.
    Return (lon, lat, h) in degrees/meters.
    """
    lon0, lat0, h0 = origin
    lon0_rad = math.radians(lon0)
    lat0_rad = math.radians(lat0)
    # ENU = R^{-1} @ final
    enu = np.linalg.inv(R) @ final_xyz.reshape(3, 1)
    x_enu, y_enu, z_enu = enu.flatten().tolist()
    # Inverse equirectangular
    lat_rad = lat0_rad + y_enu / EARTH_RADIUS
    # avoid cos(lat) near zero, but our area is not at poles
    lon_rad = lon0_rad + x_enu / (EARTH_RADIUS * math.cos(lat0_rad))
    lon = math.degrees(lon_rad)
    lat = math.degrees(lat_rad)
    h = z_enu + h0
    return lon, lat, h


def main():
    # Build transform (to get origin & R)
    meta, _ = build_transform_from_S3(S3_PATH)
    origin = meta['origin']
    R = meta['R']

    # Read existing sink.csv and keep non-RIS rows
    rows = []
    with open(SINK_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            name_lower = str(row.get('name', '')).lower()
            # remove existing RIS/PIS
            if 'ris' in name_lower or 'pis' in name_lower:
                continue
            rows.append(row)
    if not rows:
        # attempt to restore header if empty
        fieldnames = ['id', 'name', 'WKT']

    # Append new RIS lines
    next_id_base = 1000000
    for i, item in enumerate(NEW_RIS_LOCAL):
        name = item['name']
        x, y, z = item['pos']
        lon, lat, h = local_to_geodetic(np.array([x, y, z], dtype=float), origin, R)
        wkt = f"POINT Z ({lon} {lat} {h})"
        rid = f"RISNEW_{next_id_base + i}"
        rows.append({'id': rid, 'name': name, 'WKT': wkt})

    # Write back
    with open(SINK_CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Updated {SINK_CSV_PATH} with {len(NEW_RIS_LOCAL)} new RIS entries (old RIS removed).")


if __name__ == '__main__':
    main()

