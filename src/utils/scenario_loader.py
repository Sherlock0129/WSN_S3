import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from matplotlib.tri import Triangulation, LinearTriInterpolator
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

EARTH_RADIUS = 6371000.0  # meters

# ----------------------
# Parsing helpers
# ----------------------
_wkt_triplet = re.compile(r'(-?[\d\.]+)\s+(-?[\d\.]+)\s+(-?[\d\.]+)')

def parse_wkt_triplets(wkt: str):
    """Parse WKT-like string that contains space-separated triplets: lon lat h.
    Returns list of (lon, lat, h) floats.
    """
    if not isinstance(wkt, str):
        return []
    out = []
    for m in _wkt_triplet.finditer(wkt):
        lon = float(m.group(1))
        lat = float(m.group(2))
        h = float(m.group(3))
        out.append((lon, lat, h))
    return out

# ----------------------
# Coordinate transform based on S3.csv
# ----------------------

def build_transform_from_S3(s3_csv_path: str):
    """
    Build local metric coordinate transform using S3.csv anchors:
      - Node '0' is origin
      - X axis along vector 0 -> -1 (horizontal)
      - Y axis along vector 0 -> -2 (horizontal, Gram-Schmidt to be orthogonal to X)
      - Z axis is Up
    Returns a dict with:
      { 'origin': (lon0, lat0, h0), 'R': rotation_matrix(3x3) }
    and a function transform(lon, lat, h) -> np.array([x,y,z]) in meters.
    """
    df = pd.read_csv(s3_csv_path)

    # Build id -> (lon,lat,h) map
    id_map = {}
    for _, row in df.iterrows():
        wkt = row.get('WKT')
        pts = parse_wkt_triplets(wkt)
        if not pts:
            continue
        # Use first triplet if multiple
        lon, lat, h = pts[0]
        rid = str(row.get('name', row.get('id')))
        if rid is None:
            continue
        id_map[rid] = (lon, lat, h)

    required_ids = ['0', '-1', '-2']
    for rid in required_ids:
        if rid not in id_map:
            raise ValueError(f"S3.csv 缺少关键锚点ID: {rid}")

    lon0, lat0, h0 = id_map['0']
    lon_m1, lat_m1, h_m1 = id_map['-1']
    lon_m2, lat_m2, h_m2 = id_map['-2']

    # Helper: geodetic to local ENU (equirectangular approx)
    def geodetic_to_local(lon, lat, h):
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
        lon0_rad = np.deg2rad(lon0)
        lat0_rad = np.deg2rad(lat0)
        x = (lon_rad - lon0_rad) * np.cos(lat0_rad) * EARTH_RADIUS
        y = (lat_rad - lat0_rad) * EARTH_RADIUS
        z = h - h0
        return np.array([x, y, z], dtype=float)

    # Build horizontal direction vectors in local coords
    v_x = geodetic_to_local(lon_m1, lat_m1, h_m1)
    v_y = geodetic_to_local(lon_m2, lat_m2, h_m2)
    # Force to horizontal (zero z component)
    v_x[2] = 0.0
    v_y[2] = 0.0

    # Orthonormalize: x_axis = norm(v_x); y_axis = norm(v_y - proj_on_x); z=[0,0,1]
    if np.linalg.norm(v_x[:2]) < 1e-9 or np.linalg.norm(v_y[:2]) < 1e-9:
        raise ValueError("S3.csv 中 -1 或 -2 与 0 水平距离过小，无法定义坐标轴")

    x_axis = v_x / np.linalg.norm(v_x)
    y_temp = v_y - np.dot(v_y, x_axis) * x_axis
    y_axis = y_temp / np.linalg.norm(y_temp)
    z_axis = np.array([0.0, 0.0, 1.0])

    # Rotation matrix that maps local-ENU to final XYZ where axes are [x_axis, y_axis, z_axis]
    A = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns are basis vectors in ENU
    R = np.linalg.inv(A)  # so that X_final = R @ X_enu

    def transform(lon, lat, h):
        enu = geodetic_to_local(lon, lat, h)
        return (R @ enu.reshape(3, 1)).reshape(3)

    return {
        'origin': (lon0, lat0, h0),
        'R': R,
    }, transform

# ----------------------
# Scenario loading from sink.csv
# ----------------------

def load_scenario(s3_csv_path='src/data/S3.csv', sink_csv_path='src/data/sink.csv'):
    meta, transform = build_transform_from_S3(s3_csv_path)
    df = pd.read_csv(sink_csv_path)

    sink_pos = None
    rf_positions = []
    ris_positions = []

    for _, row in df.iterrows():
        wkt = row.get('WKT')
        pts = parse_wkt_triplets(wkt)
        if not pts:
            continue
        lon, lat, h = pts[0]
        pos = transform(lon, lat, h)

        # Determine type
        t = str(row.get('type', '')).lower()
        name = str(row.get('name', '')).lower()
        id_str = str(row.get('id', '')).lower()
        key = t or name or id_str
        if 'sink' in key:
            sink_pos = pos
        elif key.startswith('rf') or 'rf' in key or 'ch' in key or 'cluster' in key:
            rf_positions.append(pos)
        elif 'ris' in key:
            ris_positions.append(pos)
        else:
            # Fallback: if there's a category column
            cat = str(row.get('category', '')).lower()
            if 'sink' in cat:
                sink_pos = pos
            elif 'ris' in cat:
                ris_positions.append(pos)
            else:
                rf_positions.append(pos)

    if sink_pos is None:
        raise ValueError("sink.csv 中未找到 sink 位置（需在 type/name/id/category 中包含 'sink' 关键词）")

    return {
        'sink_pos': sink_pos,
        'rf_positions': rf_positions,
        'ris_positions': ris_positions,
    }

# ----------------------
# DEM building from S3.csv (linear triangulation + nearest fill + optional smoothing)
# ----------------------

def _load_point_cloud_from_S3(s3_csv_path: str, transform_fn) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(s3_csv_path)
    xs, ys, zs = [], [], []
    for _, row in df.iterrows():
        wkt = row.get('WKT')
        pts = parse_wkt_triplets(wkt)
        for lon, lat, h in pts:
            x, y, z = transform_fn(lon, lat, h)
            xs.append(x)
            ys.append(y)
            zs.append(z)
    return np.array(xs, dtype=float), np.array(ys, dtype=float), np.array(zs, dtype=float)


def _auto_resolution(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 10:
        return 10.0
    try:
        pts = np.column_stack([x, y])
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        nn = dists[:, 1]
        med = np.median(nn)
        res = float(np.clip(0.5 * med, 2.0, 10.0))
        return res
    except Exception:
        return 10.0


def build_dem_from_S3(
    s3_csv_path: str = 'src/data/S3.csv',
    lake_csv_path: str = 'src/data/LAKE.csv',
    grid_resolution_m: float | None = None,
    method: str = 'linear',
    fill: str = 'nearest',
    smooth_sigma: float | None = None,
) -> Dict[str, Any]:
    """
    Build DEM from S3.csv point cloud in the local XYZ frame defined by build_transform_from_S3.
    - Interpolation: linear (Triangulation+LinearTriInterpolator)
    - Fill outside convex hull: nearest (griddata) if requested
    - Lake flattening: read polygons from LAKE.csv and set DEM inside each polygon to its min elevation

    Returns dict with keys: dem (H×W), origin_xy (xmin,ymin), resolution, x_coords, y_coords
    """
    meta, transform = build_transform_from_S3(s3_csv_path)
    x, y, z = _load_point_cloud_from_S3(s3_csv_path, transform)

    # Grid spec
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    res = grid_resolution_m if grid_resolution_m is not None else _auto_resolution(x, y)

    xi = np.arange(xmin, xmax + res, res, dtype=float)
    yi = np.arange(ymin, ymax + res, res, dtype=float)
    XI, YI = np.meshgrid(xi, yi)

    # Linear triangulation interpolation
    tri = Triangulation(x, y)
    interp_lin = LinearTriInterpolator(tri, z)
    ZI = interp_lin(XI, YI)
    Z = np.array(ZI, dtype=float)

    # Fill outside convex hull
    if np.isnan(Z).any() and fill == 'nearest':
        Zn = griddata(np.column_stack([x, y]), z, (XI, YI), method='nearest')
        mask = np.isnan(Z)
        Z[mask] = Zn[mask]

    # Lake flattening
    try:
        lake_df = pd.read_csv(lake_csv_path)
        for _, row in lake_df.iterrows():
            wkt = row.get('WKT', '')
            # Extract lon lat pairs (ignore z if present)
            lonlats = []
            for tok in re.findall(r'(-?[\d\.]+\s+-?[\d\.]+)', wkt):
                parts = tok.strip().split()
                if len(parts) == 2:
                    lon = float(parts[0]); lat = float(parts[1])
                    lonlats.append((lon, lat))
            if len(lonlats) < 3:
                continue
            # Transform polygon to local XY
            poly_xy = []
            for lon, lat in lonlats:
                px, py, pz = transform(lon, lat, 0.0)
                poly_xy.append((px, py))
            path = MplPath(poly_xy)
            pts_xy = np.column_stack([XI.ravel(), YI.ravel()])
            inside = path.contains_points(pts_xy)
            inside_mask = inside.reshape(Z.shape)
            if not np.any(inside_mask):
                continue
            lake_min = float(np.nanmin(Z[inside_mask]))
            Z[inside_mask] = lake_min
    except Exception as e:
        print(f"[scenario_loader] lake flatten failed: {e}")

    # Optional smoothing
    if smooth_sigma is not None and smooth_sigma > 0:
        Z = gaussian_filter(Z, sigma=float(smooth_sigma))

    return {
        'dem': Z,
        'origin_xy': (xmin, ymin),
        'resolution': res,
        'x_coords': xi,
        'y_coords': yi,
        'points': np.column_stack([x, y, z]),
    }
