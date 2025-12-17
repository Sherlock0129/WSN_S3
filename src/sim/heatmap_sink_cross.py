import os
import sys
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Tuple

# ensure src on path
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
for p in [ROOT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.scenario_loader import build_transform_from_S3, build_dem_from_S3
from src.network.WSN import WSN
from src.tools.ris_placement_simple import evaluate_chain_final_power
from src.utils import rf_propagation_model

_triplet_re = re.compile(r'(-?[\d\.]+)\s+(-?[\d\.]+)\s+(-?[\d\.]+)')

def parse_wkt_triplet(wkt: str):
    m = _triplet_re.search(wkt or '')
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))

def load_ris_chain(prefix: str) -> List[np.ndarray]:
    s3_path = os.path.join('src','data','S3.csv') if os.path.exists(os.path.join('src','data','S3.csv')) else os.path.join('data','S3.csv')
    meta, transform = build_transform_from_S3(s3_path)
    sink_path = os.path.join('src','data','sink.csv') if os.path.exists(os.path.join('src','data','sink.csv')) else os.path.join('data','sink.csv')
    df = pd.read_csv(sink_path)
    chain = []
    names = [str(n) for n in df['name'].tolist()]
    for nm in names:
        if nm.startswith(prefix):
            row = df[df['name'] == nm].iloc[0]
            t = parse_wkt_triplet(row['WKT'])
            if t:
                lon, lat, h = t
                x, y, z = transform(lon, lat, h)
                chain.append(np.array([x,y,z], dtype=float))
    return chain

def w_to_dbm(pw: float) -> float:
    if pw is None or pw <= 0:
        return np.nan
    return 10.0 * np.log10(pw * 1000.0)

def compute_bboxes(p1: np.ndarray, p2: np.ndarray, margin: float):
    xmin = min(p1[0], p2[0]) - margin
    xmax = max(p1[0], p2[0]) + margin
    ymin = min(p1[1], p2[1]) - margin
    ymax = max(p1[1], p2[1]) + margin
    return xmin, xmax, ymin, ymax

def direct_power_heatmap(env, src, bbox, step=40.0):
    xmin, xmax, ymin, ymax = bbox
    xi = np.arange(xmin, xmax + step, step, dtype=float); yi = np.arange(ymin, ymax + step, step, dtype=float)
    XI, YI = np.meshgrid(xi, yi)
    Z = np.full_like(XI, fill_value=np.nan, dtype=float)
    H, W = XI.shape
    for i in range(H):
        for j in range(W):
            x = float(XI[i, j]); y = float(YI[i, j])
            z = float(env.get_elevation(x, y)) + 1.5
            dst = np.array([x, y, z], dtype=float)
            if not env.check_los(src.position, dst):
                continue
            # Use log-distance path loss helper
            dist = float(np.linalg.norm(src.position - dst))
            tx_dbm = src.get_tx_power_dbm()
            # reflection_gain_dbi ~ 0 for direct; rx_gain 2 dBi nominal; assume LoS
            pr_dbm = rf_propagation_model._log_distance_path_loss(tx_dbm, 0.0, 2.0, getattr(src, 'frequency_hz', None), dist, True)
            Z[i, j] = pr_dbm
    return xi, yi, Z

def ris_power_heatmap(env, src, ris_chain: List[np.ndarray], bbox, step=40.0):
    xmin, xmax, ymin, ymax = bbox
    xi = np.arange(xmin, xmax + step, step, dtype=float); yi = np.arange(ymin, ymax + step, step, dtype=float)
    XI, YI = np.meshgrid(xi, yi)
    Z = np.full_like(XI, fill_value=np.nan, dtype=float)
    H, W = XI.shape

    class RxMock:
        def __init__(self, pos):
            self.position = pos
            self.rf_rx_gain_dbi = 2.0
    for i in range(H):
        for j in range(W):
            x = float(XI[i, j]); y = float(YI[i, j])
            z = float(env.get_elevation(x, y)) + 1.5
            dst = RxMock(np.array([x, y, z], dtype=float))
            # Evaluate A->RIS...->dst (要求相邻LoS，否则返回0)
            pw = evaluate_chain_final_power(src, ris_chain + [dst.position], dst, env)
            Z[i, j] = w_to_dbm(pw)
    return xi, yi, Z


def plot_heatmap(xi, yi, Z, overlays, title, out_html):
    # robust z-scale
    zmin = np.nanpercentile(Z, 5) if np.isfinite(Z).any() else 0
    zmax = np.nanpercentile(Z, 95) if np.isfinite(Z).any() else 1
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=Z, x=xi, y=yi, colorscale='Turbo', colorbar=dict(title='dBm'), zmin=zmin, zmax=zmax))
    for ov in overlays:
        fig.add_trace(ov)
    fig.update_layout(title=title, xaxis_title='X (m)', yaxis_title='Y (m)', xaxis=dict(scaleanchor='y', scaleratio=1))
    os.makedirs('sim', exist_ok=True)
    fig.write_html(out_html)
    return out_html


def main():
    wsn = WSN()
    env = wsn.environment
    sink = wsn.rf_transmitter
    # choose CH1 as opposite side target for bbox
    ch1 = wsn.clusters[1].cluster_head
    bbox = compute_bboxes(sink.position, ch1.position, margin=1200.0)

    # 1) Direct-only heatmap (期望出现“sink一侧偏红，另一侧纯蓝/无信号”)。
    xi, yi, Zdir = direct_power_heatmap(env, sink, bbox, step=40.0)
    overlays = [
        go.Scatter(x=[sink.position[0]], y=[sink.position[1]], mode='markers+text', name='SINK',
                   marker=dict(symbol='star', size=14, color='green'), text=['SINK'], textposition='top center'),
        go.Scatter(x=[ch1.position[0]], y=[ch1.position[1]], mode='markers+text', name='CH1',
                   marker=dict(symbol='square', size=10, color='red'), text=['CH1'], textposition='top center'),
    ]
    out1 = plot_heatmap(xi, yi, Zdir, overlays, 'Direct received power from SINK (dBm)', 'sim/direct_heatmap.html')

    # 2) RIS-assisted heatmap (使用 sink->1 的 RIS 链，红色越过山)
    chain = load_ris_chain('RIS_sink->1_')
    ovs2 = overlays.copy()
    if chain:
        xs = [p[0] for p in chain]; ys = [p[1] for p in chain]
        ovs2.append(go.Scatter(x=xs, y=ys, mode='markers+lines+text', name='RIS chain',
                               marker=dict(symbol='triangle-up', size=10, color='blue'), text=[f'RIS{i+1}' for i in range(len(xs))], textposition='top center'))
    xi2, yi2, Zris = ris_power_heatmap(env, sink, chain, bbox, step=40.0)
    out2 = plot_heatmap(xi2, yi2, Zris, ovs2, 'RIS-assisted received power from SINK (dBm)', 'sim/ris_cross_heatmap.html')

    print(f"Saved: {out1} and {out2}")

if __name__ == '__main__':
    main()
