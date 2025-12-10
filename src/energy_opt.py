import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
from scipy.optimize import linprog

EARTH_RADIUS = 6371000.0
coord_re = re.compile(r'(-?[\d\.]+ -?[\d\.]+ -?[\d\.]+)')


def parse_wkt_points(wkt: str):
    return [list(map(float, s.split())) for s in coord_re.findall(wkt)]


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


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
    eta = float(np.clip(eta, min_eta, 1.0))
    return eta

# ---- Passive multi-hop path (same as before) ----

def path_eta(points_list, etacfg, ris_passive_gain):
    total_eta = 1.0
    for a, b in zip(points_list[:-1], points_list[1:]):
        d = float(np.linalg.norm(a[:2] - b[:2]))
        total_eta *= efficiency(d, etacfg)
    hops = max(len(points_list) - 2, 0)
    total_eta *= (float(ris_passive_gain) ** hops)
    return float(np.clip(total_eta, etacfg.get('min_eta', 1e-3), 1.0))


def best_path(donor, recipient, ris_nodes, etacfg, max_hops=1, ris_passive_gain=1.1):
    best = {'eta': 0.0, 'ids': [donor['id'], recipient['id']], 'coords': [donor['enu'], recipient['enu']]}
    # direct
    eta_dir = path_eta([donor['enu'], recipient['enu']], etacfg, ris_passive_gain)
    best['eta'] = eta_dir
    # 1-hop
    if max_hops >= 1 and ris_nodes:
        for r in ris_nodes:
            pts = [donor['enu'], r['enu'], recipient['enu']]
            eta1 = path_eta(pts, etacfg, ris_passive_gain)
            if eta1 > best['eta']:
                best = {'eta': eta1, 'ids': [donor['id'], r['id'], recipient['id']], 'coords': [donor['enu'], r['enu'], recipient['enu']]}
    # 2-hop
    if max_hops >= 2 and len(ris_nodes) >= 2:
        for r1 in ris_nodes:
            for r2 in ris_nodes:
                if r1['id'] == r2['id']:
                    continue
                pts = [donor['enu'], r1['enu'], r2['enu'], recipient['enu']]
                eta2 = path_eta(pts, etacfg, ris_passive_gain)
                if eta2 > best['eta']:
                    best = {'eta': eta2, 'ids': [donor['id'], r1['id'], r2['id'], recipient['id']], 'coords': pts}
    return best

# ---- New: RIS active energy contribution ----


def build_lp_with_ris_energy(donors, recipients, ris_nodes, etacfg, max_hops, ris_passive_gain, ris_surplus_map):
    I = list(donors.keys())
    J = list(recipients.keys())
    R = list(ris_surplus_map.keys())  # RIS IDs present in budget map

    nI, nJ, nR = len(I), len(J), len(R)

    # Donor->Recipient best passive path eff
    eta_d = np.zeros((nI, nJ))
    paths = [[None for _ in J] for _ in I]
    for ii, i in enumerate(I):
        for jj, j in enumerate(J):
            bp = best_path(donors[i], recipients[j], ris_nodes, etacfg, max_hops=max_hops, ris_passive_gain=ris_passive_gain)
            eta_d[ii, jj] = max(bp['eta'], 1e-9)
            paths[ii][jj] = bp

    # RIS->Recipient direct efficiency (active transmit from RIS)
    eta_r = np.zeros((nR, nJ))
    for rr, r_id in enumerate(R):
        rnode = next((x for x in ris_nodes if x['id'] == r_id), None)
        if rnode is None:
            continue
        for jj, j in enumerate(J):
            d = float(np.linalg.norm(rnode['enu'][:2] - recipients[j]['enu'][:2]))
            eta_r[rr, jj] = max(efficiency(d, etacfg), 1e-9)

    # Decision variables: [d_ij (nI*nJ), s_rj (nR*nJ)] delivered energies
    nvars = nI * nJ + nR * nJ

    # Objective: maximize total delivered (with tiny tx regularizer)
    lam = 1e-6
    c = np.zeros(nvars)
    # delivered coefficients negative for maximization via minimization
    c[:nI * nJ] = -1.0 + lam * (1.0 / eta_d).reshape(-1)
    c[nI * nJ:] = -1.0 + lam * (1.0 / eta_r).reshape(-1)

    A_ub = []
    b_ub = []

    # Donor supply on transmit energy: sum_j d_ij / eta_d_ij <= supply_i
    for ii, i in enumerate(I):
        row = np.zeros(nvars)
        for jj in range(nJ):
            row[ii * nJ + jj] = 1.0 / eta_d[ii, jj]
        A_ub.append(row)
        b_ub.append(float(donors[i]['supply']))

    # RIS supply on transmit energy: sum_j s_rj / eta_r_rj <= supply_r
    for rr, r_id in enumerate(R):
        row = np.zeros(nvars)
        base = nI * nJ + rr * nJ
        for jj in range(nJ):
            row[base + jj] = 1.0 / eta_r[rr, jj]
        A_ub.append(row)
        b_ub.append(float(ris_surplus_map[r_id]))

    # Recipient caps: sum_i d_ij + sum_r s_rj <= demand_j
    for jj, j in enumerate(J):
        row = np.zeros(nvars)
        # donor parts
        for ii in range(nI):
            row[ii * nJ + jj] = 1.0
        # ris parts
        for rr in range(nR):
            row[nI * nJ + rr * nJ + jj] = 1.0
        A_ub.append(row)
        b_ub.append(float(recipients[j]['demand']))

    bounds = [(0.0, None)] * nvars
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")

    x = res.x
    d_vars = x[:nI * nJ].reshape(nI, nJ)
    s_vars = x[nI * nJ:].reshape(nR, nJ)

    t_d = d_vars / eta_d
    t_r = s_vars / eta_r

    return {
        'I': I, 'J': J, 'R': R,
        'eta_d': eta_d, 'eta_r': eta_r,
        'paths': paths,
        'delivered_donor': d_vars, 'delivered_ris': s_vars,
        'tx_donor': t_d, 'tx_ris': t_r,
        'objective': float(res.fun)
    }


def main():
    # Load positions
    s3_pts = load_points('src/data/S3.csv')
    sink_pts = load_points('src/data/sink.csv')
    lakes_df = pd.read_csv('src/data/LAKE.csv')
    node0 = next((p for p in s3_pts if p['id'] == '0'), None)
    if node0 is None:
        raise RuntimeError("Node '0' not found in S3.csv")
    lon0, lat0, h0 = node0['lon'], node0['lat'], node0['h']

    # ENU for sink nodes
    sink_pts_enu = []
    for d in sink_pts:
        enu = geodetic_to_enu(d['lon'], d['lat'], d['h'], lon0, lat0, h0)
        sink_pts_enu.append({**d, 'enu': enu})

    # classify donors/recipients/ris
    donors = {}
    recipients = {}
    ris_nodes = []
    for p in sink_pts_enu:
        uid = p['id'].upper()
        if uid in ['RF1', 'RF2']:
            donors[p['id']] = {'enu': p['enu'], 'supply': 0.0, 'id': p['id']}
        elif uid in ['RF3', 'RF4', 'RF5', 'RF6']:
            recipients[p['id']] = {'enu': p['enu'], 'demand': 0.0, 'id': p['id']}
        elif uid.startswith('RIS') or uid.startswith('PIS'):
            ris_nodes.append({'id': p['id'], 'enu': p['enu']})

    # Load or create config
    try:
        cfg = load_config('src/energy_config.json')
    except FileNotFoundError:
        cfg = {
            'donor_surplus': {'RF1': 1200.0, 'RF2': 1200.0},
            'recipient_demand': {'RF3': 900.0, 'RF4': 900.0, 'RF5': 900.0, 'RF6': 900.0},
            'efficiency': {'model': 'exp', 'base': 0.6, 'alpha': 2e-4, 'min_eta': 1e-3},
            'max_hops': 1,
            'ris_passive_gain': 1.15,
            'ris_surplus': {'default': 600.0}
        }
        save_json('src/energy_config.json', cfg)
        print("Created default config at src/energy_config.json with RIS energy fields. Adjust and rerun if needed.")

    # Fill supplies/demands
    for k in donors.keys():
        donors[k]['supply'] = float(cfg.get('donor_surplus', {}).get(k, 0.0))
    for k in recipients.keys():
        recipients[k]['demand'] = float(cfg.get('recipient_demand', {}).get(k, 0.0))

    # RIS budgets
    ris_budget_cfg = cfg.get('ris_surplus', {'default': 0.0})
    default_ris_budget = float(ris_budget_cfg.get('default', 0.0))
    ris_surplus_map = {}
    for r in ris_nodes:
        ris_surplus_map[r['id']] = float(ris_budget_cfg.get(r['id'], default_ris_budget))

    max_hops = int(cfg.get('max_hops', 1))
    ris_passive_gain = float(cfg.get('ris_passive_gain', 1.15))

    # Solve LP with donor passive paths + RIS active transmit
    plan = build_lp_with_ris_energy(donors, recipients, ris_nodes, cfg.get('efficiency', {}), max_hops, ris_passive_gain, ris_surplus_map)

    # Save plan JSON
    out = {
        'mode': f'max_coverage_with_donor_paths_and_ris_energy(hops<= {max_hops})',
        'donor_ids': plan['I'],
        'recipient_ids': plan['J'],
        'ris_ids': plan['R'],
        'eta_donor': plan['eta_d'].tolist(),
        'eta_ris': plan['eta_r'].tolist(),
        'delivered_from_donors': plan['delivered_donor'].tolist(),
        'delivered_from_ris': plan['delivered_ris'].tolist(),
        'tx_from_donors': plan['tx_donor'].tolist(),
        'tx_from_ris': plan['tx_ris'].tolist(),
        'paths': [[{'ids': p['ids'], 'eta': p['eta']} for p in row] for row in plan['paths']],
        'ris_surplus': ris_surplus_map,
        'objective': plan['objective']
    }
    save_json('energy_transfer_plan.json', out)

    # Visualization
    fig = go.Figure()

    # Lakes
    palette = ['#99d8c9','#a6bddb','#c7e9b4','#b3cde3','#cbe8f6','#d0f0c0']
    lakes_df = pd.read_csv('src/data/LAKE.csv')
    for i, row in lakes_df.iterrows():
        name = str(row['name'])
        coords = parse_wkt_points(row['WKT'])
        if not coords:
            continue
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        poly = np.array([geodetic_to_enu(lon, lat, 0.0, lon0, lat0, h0) for lon, lat in zip(lons, lats)])
        fig.add_trace(go.Scatter(x=poly[:,0], y=poly[:,1], mode='lines', fill='toself', name=name,
                                 line=dict(color='rgba(50,120,180,0.6)', width=1),
                                 fillcolor=palette[i % len(palette)], opacity=0.35))

    # Markers
    for k, v in donors.items():
        fig.add_trace(go.Scatter(x=[v['enu'][0]], y=[v['enu'][1]], mode='markers+text', name=f'{k} (donor)',
                                 marker=dict(symbol='square', size=12, color='#2ca02c'), text=[k], textposition='top center'))
    for k, v in recipients.items():
        fig.add_trace(go.Scatter(x=[v['enu'][0]], y=[v['enu'][1]], mode='markers+text', name=f'{k} (recipient)',
                                 marker=dict(symbol='circle', size=10, color='#d62728'), text=[k], textposition='top center'))
    for r in ris_nodes:
        fig.add_trace(go.Scatter(x=[r['enu'][0]], y=[r['enu'][1]], mode='markers+text', name=f"{r['id']} (RIS)",
                                 marker=dict(symbol='triangle-up', size=9, color='#1f77b4'), text=[r['id']], textposition='top center'))

    # Draw donor-path contributions (red)
    d_vars = plan['delivered_donor']
    I, J = plan['I'], plan['J']
    max_del_d = float(np.max(d_vars)) if d_vars.size else 1.0
    for ii, i in enumerate(I):
        for jj, j in enumerate(J):
            e = float(d_vars[ii, jj])
            if e <= 1e-6:
                continue
            path = plan['paths'][ii][jj]
            # recreate coords by ids
            ids = path['ids']
            coords = []
            # map id->enu
            id2enu = {**{k: donors[k]['enu'] for k in donors}, **{k: recipients[k]['enu'] for k in recipients}, **{r['id']: r['enu'] for r in ris_nodes}}
            for nid in ids:
                coords.append(id2enu[nid])
            for a, b in zip(coords[:-1], coords[1:]):
                fig.add_trace(go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode='lines',
                                         line=dict(width=2 + 6 * (e / max_del_d), color='rgba(200,50,50,0.85)'),
                                         showlegend=False))

    # Draw RIS active contributions (blue)
    s_vars = plan['delivered_ris']
    R = plan['R']
    max_del_r = float(np.max(s_vars)) if s_vars.size else 1.0
    for rr, r_id in enumerate(R):
        rnode = next((x for x in ris_nodes if x['id'] == r_id), None)
        if rnode is None:
            continue
        for jj, j in enumerate(J):
            e = float(s_vars[rr, jj])
            if e <= 1e-6:
                continue
            p0 = rnode['enu']
            p1 = recipients[j]['enu']
            fig.add_trace(go.Scatter(x=[p0[0], p1[0]], y=[p0[1], p1[1]], mode='lines',
                                     line=dict(width=2 + 6 * (e / max_del_r), color='rgba(30,120,200,0.85)', dash='dot'),
                                     showlegend=False))

    total_supply = sum(v['supply'] for v in donors.values()) + sum(v for v in ris_surplus_map.values())
    total_demand = sum(v['demand'] for v in recipients.values())
    total_delivered = float(np.sum(d_vars) + np.sum(s_vars))
    subtitle = f"Supply(donors+RIS)={total_supply:.1f}J, Demand={total_demand:.1f}J, Delivered={total_delivered:.1f}J"

    fig.update_layout(title=f'Energy Transfer with RIS Active Amplification (hops≤{max_hops})<br>{subtitle}',
                      xaxis=dict(title='X (m)', scaleanchor='y', scaleratio=1),
                      yaxis=dict(title='Y (m)'), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                      margin=dict(l=40, r=40, t=100, b=40))

    fig.write_html('energy_transfer_map.html')
    print('Saved: energy_transfer_plan.json and energy_transfer_map.html (RIS energy model)')


if __name__ == '__main__':
    main()
