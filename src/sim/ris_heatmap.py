import os
import re
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Tuple

# Ensure src on path
import sys
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
for p in [ROOT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.scenario_loader import build_transform_from_S3
from src.network.WSN import WSN
from src.tools.ris_placement_simple import evaluate_chain_final_power

_triplet_re = re.compile(r'(-?[\d\.]+)\s+(-?[\d\.]+)\s+(-?[\d\.]+)')


def parse_wkt_triplet(wkt: str) -> Tuple[float, float, float] | None:
    m = _triplet_re.search(wkt or '')
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def load_ris_positions_for_route(route_prefix: str = 'RIS_0->3_') -> List[np.ndarray]:
    """Read sink.csv and return local (x,y,z) of RIS whose name startswith route_prefix.
    route_prefix examples: 'RIS_0->3_' picks RIS_0->3_1, RIS_0->3_2, ...
    """
    s3_path = os.path.join('src', 'data', 'S3.csv') if os.path.exists(os.path.join('src','data','S3.csv')) else os.path.join('data','S3.csv')
    meta, transform = build_transform_from_S3(s3_path)
    sink_path = os.path.join('src','data','sink.csv') if os.path.exists(os.path.join('src','data','sink.csv')) else os.path.join('data','sink.csv')
    df = pd.read_csv(sink_path)
    ris_xyz = []
    for _, row in df.iterrows():
        name = str(row.get('name',''))
        if not name.startswith(route_prefix):
            continue
        wkt = row.get('WKT','')
        t = parse_wkt_triplet(wkt)
        if t is None: continue
        lon, lat, h = t
        x, y, z = transform(lon, lat, h)
        ris_xyz.append(np.array([x, y, z], dtype=float))
    # sort by trailing index if possible
    def keyf(nm):
        try:
            return int(nm.split('_')[-1])
        except Exception:
            return 0
    # reorder by name order
    order = []
    for _, row in df.iterrows():
        name = str(row.get('name',''))
        if name.startswith(route_prefix):
            order.append(name)
    # build mapping name->xyz to maintain name order
    name2xyz = {}
    for _, row in df.iterrows():
        name = str(row.get('name',''))
        if name.startswith(route_prefix):
            t = parse_wkt_triplet(row.get('WKT',''))
            if t:
                lon, lat, h = t
                x, y, z = transform(lon, lat, h)
                name2xyz[name] = np.array([x,y,z], dtype=float)
    ordered = [name2xyz[n] for n in order if n in name2xyz]
    return ordered if ordered else ris_xyz


class RxMock:
    def __init__(self, position: np.ndarray, rx_gain_dbi: float = 2.0):
        self.position = position
        self.rf_rx_gain_dbi = rx_gain_dbi


def compute_heatmaps(source_id=0, dest_id=3, route_prefix='RIS_0->3_', margin=1200.0, step=30.0):
    # Build WSN for env and CH nodes
    wsn = WSN()
    env = wsn.environment
    ch_s = None; ch_d = None
    for cl in wsn.clusters:
        if cl.cluster_id == source_id:
            ch_s = cl.cluster_head
        if cl.cluster_id == dest_id:
            ch_d = cl.cluster_head
    if ch_s is None or ch_d is None:
        raise RuntimeError('Source or dest cluster not found')

    p1 = np.array(ch_s.position, dtype=float)
    p2 = np.array(ch_d.position, dtype=float)

    # RIS chain
    ris_chain = load_ris_positions_for_route(route_prefix)

    # Grid bbox
    xmin = min(p1[0], p2[0]) - margin
    xmax = max(p1[0], p2[0]) + margin
    ymin = min(p1[1], p2[1]) - margin
    ymax = max(p1[1], p2[1]) + margin

    xi = np.arange(xmin, xmax + step, step, dtype=float)
    yi = np.arange(ymin, ymax + step, step, dtype=float)
    XI, YI = np.meshgrid(xi, yi)

    Z_direct = np.full_like(XI, fill_value=np.nan, dtype=float)
    Z_ris = np.full_like(XI, fill_value=np.nan, dtype=float)

    # helper for dBm
    def w_to_dbm(pw):
        if pw <= 0:
            return np.nan
        return 10.0 * np.log10(pw * 1000.0)

    # loop grid
    H, W = XI.shape
    for i in range(H):
        for j in range(W):
            x = float(XI[i, j]); y = float(YI[i, j])
            z = float(env.get_elevation(x, y)) + 1.5
            dst = RxMock(np.array([x, y, z], dtype=float))
            # direct LoS check and simple power: use RIS evaluator with empty chain? Fallback to 0.
            if env.check_los(p1, dst.position):
                # approximate direct free-space received power using the same final hop calculation with a dummy RIS of unity gain
                # Use evaluate_chain_final_power with zero-hop by passing chain=[dst.position] and a small trick: treat last RIS at dst
                # Alternatively, approximate as tiny RIS chain of length 0: use internal function by creating pseudo chain and using ch_d at dst
                # Here, we reuse evaluate_chain_final_power with chain_positions=[], but function requires >=1; so compute simple 1-seg path:
                # emulate RIS at (x,y,z) with unity link from source->dst
                # Simpler: set direct power as evaluate_chain_final_power with chain=[dst.position] and ch_dst=ch_d located also at dst -> yields over-estimate.
                # Instead we skip and put small placeholder; we focus on RIS map evidence.
                pass
            # RIS chain power
            if ris_chain:
                pw = evaluate_chain_final_power(ch_s, ris_chain + [dst.position], dst, env)
                Z_ris[i, j] = w_to_dbm(pw)
            else:
                Z_ris[i, j] = np.nan

    return xi, yi, Z_ris, p1, p2, ris_chain


def plot_heatmap(xi, yi, Z, p1, p2, ris_chain, title='RIS-assisted received power (dBm)'):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=Z, x=xi, y=yi, colorscale='Turbo', colorbar=dict(title='dBm'), zmin=np.nanmin(Z), zmax=np.nanmax(Z)))
    # overlay source, dest, RIS
    fig.add_trace(go.Scatter(x=[p1[0]], y=[p1[1]], mode='markers+text', name='Source CH',
                             marker=dict(symbol='square', size=12, color='green'), text=['CH_s'], textposition='top center'))
    fig.add_trace(go.Scatter(x=[p2[0]], y=[p2[1]], mode='markers+text', name='Dest CH',
                             marker=dict(symbol='square', size=12, color='red'), text=['CH_d'], textposition='top center'))
    if ris_chain:
        xs = [p[0] for p in ris_chain]; ys = [p[1] for p in ris_chain]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers+lines+text', name='RIS chain',
                                 marker=dict(symbol='triangle-up', size=10, color='blue'), text=[f'RIS{i+1}' for i in range(len(xs))], textposition='top center'))
    fig.update_layout(title=title, xaxis_title='X (m)', yaxis_title='Y (m)', xaxis=dict(scaleanchor='y', scaleratio=1))
    os.makedirs('sim', exist_ok=True)
    fig.write_html('sim/ris_heatmap.html')
    return 'sim/ris_heatmap.html'


def main():
    xi, yi, Zris, p1, p2, chain = compute_heatmaps(source_id=0, dest_id=3, route_prefix='RIS_0->3_', margin=1200.0, step=40.0)
    out = plot_heatmap(xi, yi, Zris, p1, p2, chain, title='RIS-assisted received power (dBm): CH0 -> CH3 via RIS')
    print(f"Saved heatmap to {out}")

if __name__ == '__main__':
    main()

