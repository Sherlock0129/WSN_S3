import os
import sys
import csv
import math
from typing import List, Tuple, Dict, Union
import numpy as np

# Ensure project root is on sys.path so that 'import src.xxx' works
CURRENT_DIR = os.path.dirname(__file__)              # .../src/tools
SRC_DIR = os.path.dirname(CURRENT_DIR)               # .../src
ROOT_DIR = os.path.dirname(SRC_DIR)                  # project root
for p in [ROOT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.tools import ris_placement_simple as rps
from src.utils.scenario_loader import build_transform_from_S3
from src.network.WSN import WSN
from src.core.RIS import RIS
from src.utils import rf_propagation_model

EARTH_RADIUS = 6371000.0


def resolve(path: str) -> str:
    # Allow both src/data/... and data/...
    candidates = [
        path,
        os.path.join(ROOT_DIR, path),
        os.path.join(SRC_DIR, path),
        path.replace('src/', ''),
        path.replace('data/', 'src/data/'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


def local_to_geodetic(final_xyz: np.ndarray, origin, R: np.ndarray):
    lon0, lat0, h0 = origin
    lon0_rad = math.radians(lon0)
    lat0_rad = math.radians(lat0)
    enu = np.linalg.inv(R) @ final_xyz.reshape(3, 1)
    x_enu, y_enu, z_enu = enu.flatten().tolist()
    lat_rad = lat0_rad + y_enu / EARTH_RADIUS
    lon_rad = lon0_rad + x_enu / (EARTH_RADIUS * math.cos(lat0_rad))
    lon = math.degrees(lon_rad)
    lat = math.degrees(lat_rad)
    h = z_enu + h0
    return lon, lat, h


def uniq_positions(positions: List[np.ndarray], tol: float = 1.0) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for p in positions:
        keep = True
        for q in out:
            if float(np.linalg.norm(p - q)) <= tol:
                keep = False
                break
        if keep:
            out.append(p)
    return out

# ---------------- Generic endpoints (sink or CH) helpers -----------------

def _pos_of(node) -> np.ndarray:
    return np.array(node.position, dtype=float)


def _line_bbox(p1: np.ndarray, p2: np.ndarray, margin: float):
    xmin = min(p1[0], p2[0]) - margin
    xmax = max(p1[0], p2[0]) + margin
    ymin = min(p1[1], p2[1]) - margin
    ymax = max(p1[1], p2[1]) + margin
    return xmin, xmax, ymin, ymax


def _build_grid(xmin, xmax, ymin, ymax, step):
    xi = np.arange(xmin, xmax + step, step, dtype=float)
    yi = np.arange(ymin, ymax + step, step, dtype=float)
    XI, YI = np.meshgrid(xi, yi)
    return XI, YI


def compute_viewshed_mask(env, origin: np.ndarray, XI: np.ndarray, YI: np.ndarray, install_height: float) -> np.ndarray:
    H, W = XI.shape
    mask = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            x = float(XI[i, j]); y = float(YI[i, j])
            z = float(env.get_elevation(x, y)) + install_height
            p = np.array([x, y, z], dtype=float)
            if env.check_los(origin, p):
                mask[i, j] = True
    return mask


def choose_single_viewshed_generic(env, src_obj, dst_obj, corridor_half_width, install_height, coarse_step_factor):
    step = max(float(env.resolution) * coarse_step_factor, 20.0)
    p1 = _pos_of(src_obj)
    p2 = _pos_of(dst_obj)
    xmin, xmax, ymin, ymax = _line_bbox(p1, p2, corridor_half_width)
    XI, YI = _build_grid(xmin, xmax, ymin, ymax, step)
    mask_a = compute_viewshed_mask(env, p1, XI, YI, install_height)
    mask_b = compute_viewshed_mask(env, p2, XI, YI, install_height)
    mask = np.logical_and(mask_a, mask_b)
    best_power = 0.0; best_pos = None
    it = np.argwhere(mask)
    for (i, j) in it:
        x = float(XI[i, j]); y = float(YI[i, j])
        z = float(env.get_elevation(x, y)) + install_height
        candidate = np.array([x, y, z], dtype=float)
        ris = RIS(panel_id=-1, position=candidate)
        pr_w = rf_propagation_model.calculate_ris_assisted_power(src_obj, ris, dst_obj, env)
        if pr_w > best_power:
            best_power = pr_w; best_pos = candidate
    return {'ok': best_pos is not None, 'best_pos': best_pos, 'best_power_w': float(best_power), 'step': float(step)}


def choose_single_corridor_generic(env, src_obj, dst_obj, corridor_half_width, install_height, coarse_step_factor):
    step = max(float(env.resolution) * coarse_step_factor, 20.0)
    p1 = _pos_of(src_obj); p2 = _pos_of(dst_obj)
    xmin, xmax, ymin, ymax = _line_bbox(p1, p2, corridor_half_width)
    XI, YI = _build_grid(xmin, xmax, ymin, ymax, step)
    best_power = 0.0; best_pos = None
    for i in range(XI.shape[0]):
        for j in range(XI.shape[1]):
            x = float(XI[i, j]); y = float(YI[i, j])
            z = float(env.get_elevation(x, y)) + install_height
            candidate = np.array([x, y, z], dtype=float)
            if not env.check_los(p1, candidate) or not env.check_los(candidate, p2):
                continue
            ris = RIS(panel_id=-1, position=candidate)
            pr_w = rf_propagation_model.calculate_ris_assisted_power(src_obj, ris, dst_obj, env)
            if pr_w > best_power:
                best_power = pr_w; best_pos = candidate
    return {'ok': best_pos is not None, 'best_pos': best_pos, 'best_power_w': float(best_power), 'step': float(step)}


# -----------------------------------------------------------------------

def main():
    # I/O paths
    s3_path = resolve('src/data/S3.csv') if os.path.exists('src/data/S3.csv') else resolve('data/S3.csv')
    sink_path = resolve('src/data/sink.csv') if os.path.exists('src/data/sink.csv') else resolve('data/sink.csv')

    # Build transform (origin, R)
    meta, _ = build_transform_from_S3(s3_path)
    origin = meta['origin']
    R = meta['R']

    # Build environment
    wsn = WSN()
    env = wsn.environment

    # Requested pairs (cluster ids, 0-based) and sink pairs
    pairs_cc: List[Tuple[int, int]] = [(0, 3), (3, 4), (4, 5), (1, 2), (2, 4), (4, 5)]
    seen = set(); pairs_cc = [p for p in pairs_cc if not (p in seen or seen.add(p))]
    pairs_sc: List[Tuple[str, int]] = [('sink', 1), ('sink', 1)]
    seen2 = set(); pairs_sc = [p for p in pairs_sc if not (p in seen2 or seen2.add(p))]

    # Parameters
    corridor_half_width = 1000.0
    coarse_step_factor = 3.0
    install_height = 6.0

    placed: Dict[str, List[np.ndarray]] = {}

    # Cluster->Cluster pairs via rps helper
    for a, b in pairs_cc:
        ch_a = rps._get_ch(wsn, a)
        ch_b = rps._get_ch(wsn, b)
        if ch_a is None or ch_b is None:
            print(f"[WARN] cluster {a} or {b} not found; skip")
            continue
        p1 = np.array(ch_a.position, dtype=float); p2 = np.array(ch_b.position, dtype=float)
        key = f"{a}->{b}"
        if env.check_los(p1, p2):
            print(f"Pair {key}: LoS direct; no RIS needed")
            placed[key] = []
            continue
        res = rps.choose_single_ris_viewshed(wsn, a, b, corridor_half_width, install_height, coarse_step_factor)
        if res.get('ok'):
            placed[key] = [np.array(res['best_pos'], dtype=float)]
            print(f"Pair {key}: 1-RIS (viewshed)")
            continue
        res2 = rps.choose_single_ris_corridor(wsn, a, b, corridor_half_width, install_height, coarse_step_factor)
        if res2.get('ok'):
            placed[key] = [np.array(res2['best_pos'], dtype=float)]
            print(f"Pair {key}: 1-RIS (corridor)")
            continue
        res3 = rps.choose_two_ris_viewsheds(wsn, a, b, corridor_half_width, install_height, coarse_step_factor, topk_each=200)
        if res3.get('ok'):
            pA, pB = res3['best_pos_pair']
            placed[key] = [np.array(pA, dtype=float), np.array(pB, dtype=float)]
            print(f"Pair {key}: 2-RIS (viewsheds)")
        else:
            placed[key] = []
            print(f"[WARN] Pair {key}: no feasible placement")

    # Sink->Cluster pairs via generic helpers
    sink = wsn.rf_transmitter
    for _, b in pairs_sc:
        ch_b = rps._get_ch(wsn, b)
        if ch_b is None:
            print(f"[WARN] sink->{b}: cluster not found; skip")
            continue
        p1 = _pos_of(sink); p2 = _pos_of(ch_b)
        key = f"sink->{b}"
        if env.check_los(p1, p2):
            print(f"Pair {key}: LoS direct; no RIS needed")
            placed[key] = []
            continue
        r1 = choose_single_viewshed_generic(env, sink, ch_b, corridor_half_width, install_height, coarse_step_factor)
        if r1.get('ok'):
            placed[key] = [np.array(r1['best_pos'], dtype=float)]
            print(f"Pair {key}: 1-RIS (viewshed)")
            continue
        r2 = choose_single_corridor_generic(env, sink, ch_b, corridor_half_width, install_height, coarse_step_factor)
        if r2.get('ok'):
            placed[key] = [np.array(r2['best_pos'], dtype=float)]
            print(f"Pair {key}: 1-RIS (corridor)")
        else:
            placed[key] = []
            print(f"[WARN] Pair {key}: no feasible placement")

    # Aggregate positions (unique within 1 m)
    all_pos: List[np.ndarray] = []
    for seq in placed.values():
        all_pos.extend(seq)
    unique_pos = uniq_positions(all_pos, tol=1.0)

    # Update sink.csv: remove old RIS/PIS and append new
    rows = []
    fieldnames = None
    with open(sink_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ['id','name','WKT']
        for row in reader:
            key = (str(row.get('type',''))+str(row.get('name',''))+str(row.get('id',''))+str(row.get('category',''))).lower()
            if 'ris' in key or 'pis' in key:
                continue
            rows.append(row)

    rid_base = 800000; idx = 0
    for label, seq in placed.items():
        for j, p in enumerate(seq, start=1):
            lon, lat, h = local_to_geodetic(p, origin, R)
            name = f"RIS_{label}_{j}"
            rid = f"RISNEW_{rid_base+idx}"
            idx += 1
            rows.append({'id': rid, 'name': name, 'WKT': f"POINT Z ({lon} {lat} {h})"})

    with open(sink_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Placement summary:")
    for k, v in placed.items():
        print(f"  {k}: {len(v)} RIS")
    print(f"Applied {len(unique_pos)} unique RIS positions to {sink_path}.")


if __name__ == '__main__':
    main()
