import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
from datetime import datetime

# Reuse helpers similar to energy_opt
EARTH_RADIUS = 6371000.0
_triplet_re = re.compile(r'(-?[\d\.]+ -?[\d\.]+ -?[\d\.]+)')


def parse_wkt_points(wkt: str):
    return [list(map(float, s.split())) for s in _triplet_re.findall(wkt or '')]


def load_points(file_path: str, id_field_first: str = 'name', id_field_second: str = 'id'):
    df = pd.read_csv(file_path)
    pts = []
    for _, row in df.iterrows():
        pid = row.get(id_field_first, row.get(id_field_second))
        for p in parse_wkt_points(row['WKT']):
            pts.append({'id': str(pid), 'lon': p[0], 'lat': p[1], 'h': p[2]})
    return pts


def geodetic_to_enu(lon, lat, h, lon0, lat0, h0):
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    lon0_rad = np.deg2rad(lon0)
    lat0_rad = np.deg2rad(lat0)
    x = (lon_rad - lon0_rad) * np.cos(lat0_rad) * EARTH_RADIUS
    y = (lat_rad - lat0_rad) * EARTH_RADIUS
    z = h - h0
    return np.array([x, y, z])


def efficiency(distance_m: float, cfg: dict) -> float:
    model = cfg.get('model', 'exp')
    base = float(cfg.get('base', 0.6))
    min_eta = float(cfg.get('min_eta', 1e-3))
    if model == 'exp':
        alpha = float(cfg.get('alpha', 2e-4))
        eta = base * np.exp(-alpha * max(distance_m, 0.0))
    elif model == 'friis':
        k = float(cfg.get('k', 1e-6))
        eta = base / (1.0 + k * distance_m ** 2)
    else:
        eta = base
    return float(np.clip(eta, min_eta, 1.0))


def solve_lp_max_coverage(donors, recipients, etacfg):
    # donors/recipients: id -> {'enu': np.array, 'supply' or 'demand'}
    I = list(donors.keys())
    J = list(recipients.keys())
    nI, nJ = len(I), len(J)
    # build eta matrix
    eta = np.zeros((nI, nJ))
    for ii, i in enumerate(I):
        for jj, j in enumerate(J):
            d = float(np.linalg.norm(donors[i]['enu'][:2] - recipients[j]['enu'][:2]))
            eta[ii, jj] = efficiency(d, etacfg)
    # maximize sum d_ij subject to donor supply (transmit) and recipient caps
    nvars = nI * nJ
    lam = 1e-6
    c = (-np.ones((nI, nJ)) + lam * (1.0 / eta)).reshape(-1)
    A_ub = []
    b_ub = []
    # donor supply on transmit energy
    for ii, i in enumerate(I):
        row = np.zeros(nvars)
        for jj in range(nJ):
            row[ii * nJ + jj] = 1.0 / eta[ii, jj]
        A_ub.append(row)
        b_ub.append(float(donors[i]['supply']))
    # recipient cap
    for jj, j in enumerate(J):
        row = np.zeros(nvars)
        for ii in range(nI):
            row[ii * nJ + jj] = 1.0
        A_ub.append(row)
        b_ub.append(float(recipients[j]['demand']))
    from scipy.optimize import linprog
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0.0, None)] * nvars, method='highs')
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    d_vars = res.x.reshape(nI, nJ)
    t_vars = d_vars / eta
    return {'I': I, 'J': J, 'eta': eta, 'delivered': d_vars, 'transmit': t_vars, 'objective': float(res.fun)}


def load_sim_config(path='src/sim/sim_config.json'):
    if not os.path.exists(path):
        cfg = {
            'days': 3,
            'efficiency': {'model': 'exp', 'base': 0.6, 'alpha': 2e-4, 'min_eta': 1e-3},
            'donor_surplus_daily': {'RF1': 1200.0, 'RF2': 1200.0},
            'recipient_demand_daily': {'RF3': 900.0, 'RF4': 900.0, 'RF5': 900.0, 'RF6': 900.0}
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return cfg
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_positions():
    # origin from S3 node 0
    s3_pts = load_points('src/data/S3.csv')
    node0 = next(p for p in s3_pts if p['id'] == '0')
    lon0, lat0, h0 = node0['lon'], node0['lat'], node0['h']
    sink_pts = load_points('src/data/sink.csv')
    enu = {}
    for d in sink_pts:
        enu[d['id']] = geodetic_to_enu(d['lon'], d['lat'], d['h'], lon0, lat0, h0)
    return enu


def simulate(days: int, etacfg: dict, donor_sup_daily: dict, rec_dem_daily: dict):
    pos = build_positions()
    donors_ids = ['RF1', 'RF2']
    recip_ids = ['RF3', 'RF4', 'RF5', 'RF6']
    outputs = []
    for day in range(1, days + 1):
        donors = {i: {'enu': pos[i], 'supply': float(donor_sup_daily.get(i, 0.0))} for i in donors_ids}
        recips = {j: {'enu': pos[j], 'demand': float(rec_dem_daily.get(j, 0.0))} for j in recip_ids}
        plan = solve_lp_max_coverage(donors, recips, etacfg)
        outputs.append({
            'day': day,
            'delivered': plan['delivered'].tolist(),
            'transmit': plan['transmit'].tolist(),
            'donor_ids': plan['I'],
            'recipient_ids': plan['J'],
            'eta': plan['eta'].tolist(),
            'objective': plan['objective']
        })
    return outputs


def plot_summary(outputs, pos):
    # aggregate delivered
    I = outputs[0]['donor_ids']
    J = outputs[0]['recipient_ids']
    agg = np.zeros((len(I), len(J)))
    for o in outputs:
        agg += np.array(o['delivered'])
    fig = go.Figure()
    # plot donors and recipients
    for i in I:
        p = pos[i]
        fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode='markers+text', name=f'{i} (donor)',
                                 marker=dict(symbol='square', size=12, color='#2ca02c'), text=[i], textposition='top center'))
    for j in J:
        p = pos[j]
        fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode='markers+text', name=f'{j} (recipient)',
                                 marker=dict(symbol='circle', size=10, color='#d62728'), text=[j], textposition='top center'))
    max_del = agg.max() if agg.size else 1.0
    for ii, i in enumerate(I):
        for jj, j in enumerate(J):
            e = agg[ii, jj]
            if e <= 1e-6:
                continue
            p0 = pos[i]; p1 = pos[j]
            fig.add_trace(go.Scatter(x=[p0[0], p1[0]], y=[p0[1], p1[1]], mode='lines',
                                     line=dict(width=2 + 6 * (e / max_del), color='rgba(200,50,50,0.8)'),
                                     showlegend=False))
    fig.update_layout(title='Network-layer Energy Transfer Summary (aggregated over days)',
                      xaxis=dict(title='X (m)', scaleanchor='y', scaleratio=1),
                      yaxis=dict(title='Y (m)'), margin=dict(l=40, r=40, t=80, b=40))
    os.makedirs('sim', exist_ok=True)
    fig.write_html('sim/network_energy_summary.html')


def aggregate_and_save(outputs):
    I = outputs[0]['donor_ids']
    J = outputs[0]['recipient_ids']
    nI, nJ = len(I), len(J)
    agg_del = np.zeros((nI, nJ))
    agg_tx = np.zeros((nI, nJ))
    for o in outputs:
        agg_del += np.array(o['delivered'])
        agg_tx += np.array(o['transmit'])
    per_recipient = agg_del.sum(axis=0).tolist()
    per_donor_tx = agg_tx.sum(axis=1).tolist()
    summary = {
        'donor_ids': I,
        'recipient_ids': J,
        'delivered_agg': agg_del.tolist(),
        'delivered_per_recipient': per_recipient,
        'transmit_per_donor': per_donor_tx,
        'total_delivered': float(agg_del.sum()),
        'total_transmit': float(agg_tx.sum()),
        'days': len(outputs)
    }
    with open(os.path.join('sim', 'network_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

def main():
    cfg = load_sim_config()
    outputs = simulate(cfg.get('days', 3), cfg.get('efficiency', {}), cfg.get('donor_surplus_daily', {}), cfg.get('recipient_demand_daily', {}))
    # save
    os.makedirs('sim', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sim_path = os.path.join('sim', f'network_sim_{ts}.json')
    with open(sim_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    # aggregate and save summary JSON
    summary = aggregate_and_save(outputs)
    # plot summary
    pos = build_positions()
    plot_summary(outputs, pos)
    print(f"Saved sim results: {sim_path}")
    print("Summary JSON: sim/network_summary.json; map: sim/network_energy_summary.html")


if __name__ == '__main__':
    main()

