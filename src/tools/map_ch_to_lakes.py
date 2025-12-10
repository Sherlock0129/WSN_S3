import os
import sys
import re
import json
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

# ensure src on path
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
for p in [ROOT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.scenario_loader import build_transform_from_S3
from src.network.WSN import WSN

pair_re_lonlat = re.compile(r'(-?[\d\.]+)\s+(-?[\d\.]+)')


def parse_polygon_lonlat(wkt: str):
    # extract lon lat pairs in sequence (ignore Z)
    pts = []
    for m in pair_re_lonlat.finditer(wkt):
        lon = float(m.group(1)); lat = float(m.group(2))
        pts.append((lon, lat))
    # polygons are closed (first point repeated) - keep as-is
    return pts


def main():
    s3_path = os.path.join('src', 'data', 'S3.csv') if os.path.exists(os.path.join('src','data','S3.csv')) else os.path.join('data','S3.csv')
    lake_path = os.path.join('src', 'data', 'LAKE.csv') if os.path.exists(os.path.join('src','data','LAKE.csv')) else os.path.join('data','LAKE.csv')

    meta, transform = build_transform_from_S3(s3_path)

    # Build WSN to get CH positions
    wsn = WSN()
    ch_positions = {}
    for idx, cl in enumerate(wsn.clusters):
        ch = cl.cluster_head
        ch_positions[f'CH_{idx}'] = np.array(ch.position, dtype=float)

    # Load lakes and transform to local XY
    df = pd.read_csv(lake_path)
    lakes_local = []  # list of (name, np.array Nx2)
    for _, row in df.iterrows():
        name = str(row.get('name'))
        wkt = row.get('WKT','')
        lonlats = parse_polygon_lonlat(wkt)
        if len(lonlats) < 3:
            continue
        poly_xy = []
        for lon, lat in lonlats:
            x, y, z = transform(lon, lat, 0.0)
            poly_xy.append((x, y))
        lakes_local.append((name, np.array(poly_xy, dtype=float)))

    # Map CH -> lake containing, else nearest centroid
    mapping = {}
    for ch_name, pos in ch_positions.items():
        pt = pos[:2]
        chosen = None
        for lname, poly in lakes_local:
            path = MplPath(poly)
            if path.contains_point(pt):
                chosen = lname
                break
        if chosen is None:
            # nearest by centroid distance
            min_d = 1e18; min_name = None
            for lname, poly in lakes_local:
                centroid = np.mean(poly[:-1], axis=0) if np.allclose(poly[0], poly[-1]) else np.mean(poly, axis=0)
                d = float(np.linalg.norm(pt - centroid))
                if d < min_d:
                    min_d = d; min_name = lname
            chosen = min_name
        mapping[ch_name] = chosen

    print("CH->Lake mapping:")
    for k in sorted(mapping.keys()):
        print(f"  {k} -> {mapping[k]}")

    with open('ch_lake_map.json','w',encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print("Saved ch_lake_map.json")


if __name__ == '__main__':
    main()

