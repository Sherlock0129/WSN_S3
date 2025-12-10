import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
from collections import deque, defaultdict
from scipy.optimize import linprog

EARTH_RADIUS = 6371000.0
coord_re = re.compile(r'(-?[\d\.]+ -?[\d\.]+ -?[\d\.]+)')


def parse_wkt_points(wkt: str):
    return [list(map(float, s.split())) for s in coord_re.findall(wkt)]


def load_points(file_path: str, id_field_first: str = 'name', id_field_second: str = 'id'):
    df = pd.read_csv(file_path)
    pts = []
    for _, row in df.iterrows():
        pid = str(row.get(id_field_first, row.get(id_field_second)))
        for p in parse_wkt_points(row['WKT']):
            pts.append({'id': pid, 'lon': p[0], 'lat': p[1], 'h': p[2]})
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


def build_graph(sequences):
    edges = []
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            edges.append((a, b))
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for e in edges:
        if e not in seen:
            uniq.append(e)
            seen.add(e)
    # adjacency
    adj = defaultdict(list)
    for u, v in uniq:
        adj[u].append(v)
    return uniq, adj


def shortest_path(adj, start, goal):
    # BFS on directed graph for a path
    q = deque([(start, [start])])
    seen = {start}
    while q:
        u, path = q.popleft()
        if u == goal:
            return path
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append((v, path + [v]))
    return None


def path_eta(path_nodes, id2enu, etacfg, ris_passive_gain):
    if path_nodes is None or len(path_nodes) < 2:
        return 0.0
    eta = 1.0
    ris_hops = 0
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        pa = id2enu[a]
        pb = id2enu[b]
        d = float(np.linalg.norm(pa[:2] - pb[:2]))
        eta *= efficiency(d, etacfg)
        # count passive RIS hop when the intermediate node is RIS/PIS (source node of edge)
        if a.upper().startswith('RIS') or a.upper().startswith('PIS'):
            ris_hops += 1
    if ris_hops > 0:
        eta *= (float(ris_passive_gain) ** ris_hops)
    return float(np.clip(eta, etacfg.get('min_eta', 1e-3), 1.0))


def run():
    # Load nodes
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

    # Index
    id2enu = {p['id']: p['enu'] for p in sink_pts_enu}

    # Alias: RIS1 == PIS1 if RIS1 not present
    ids = set(id2enu.keys())
    if 'RIS1' not in ids and 'PIS1' in ids:
        id2enu['RIS1'] = id2enu['PIS1']

    # User-specified sequences
    route1 = ['RF1','RIS24','RIS6','RIS5','RIS4','RIS25','RIS1','RIS7','RIS8','RF4','RIS9','RIS10','RIS11','RF5','RIS12','RIS13','RF6']
    route2 = ['RF2','RIS15','RIS14','RIS3','RIS16','RIS17','RF3']
    ridge  = ['RIS3','RIS19','RIS20','RIS21','RIS2','RIS22','RIS23','RIS1']

    sequences = [route1, route2, ridge]
    edges, adj = build_graph(sequences)

    # Config
    try:
        cfg = load_config('src/energy_config.json')
    except FileNotFoundError:
        cfg = {
            'donor_surplus': {'RF1': 1200.0, 'RF2': 1200.0},
            'recipient_demand': {'RF3': 900.0, 'RF4': 900.0, 'RF5': 900.0, 'RF6': 900.0},
            'efficiency': {'model': 'exp', 'base': 0.6, 'alpha': 2e-4, 'min_eta': 1e-3},
            'ris_passive_gain': 1.2,
            'ris_surplus': {'default': 500.0}
        }
        save_json('src/energy_config.json', cfg)

    etacfg = cfg.get('efficiency', {})
    ris_passive_gain = float(cfg.get('ris_passive_gain', 1.2))

    # classify
    donors = ['RF1','RF2']
    recipients = ['RF3','RF4','RF5','RF6']
    ris_nodes = [nid for nid in id2enu.keys() if nid.upper().startswith('RIS') or nid.upper().startswith('PIS')]

    # Supplies/demands
    donor_supply = {d: float(cfg.get('donor_surplus', {}).get(d, 0.0)) for d in donors}
    rec_demand = {r: float(cfg.get('recipient_demand', {}).get(r, 0.0)) for r in recipients}
    ris_budget_cfg = cfg.get('ris_surplus', {'default': 0.0})
    default_ris_budget = float(ris_budget_cfg.get('default', 0.0))
    ris_supply = {r: float(ris_budget_cfg.get(r, default_ris_budget)) for r in ris_nodes}

    # Build donor->recipient paths restricted to graph
    donor_paths = {}
    for d in donors:
        for r in recipients:
            path_nodes = shortest_path(adj, d, r)
            if path_nodes:
                eta = path_eta(path_nodes, id2enu, etacfg, ris_passive_gain)
                donor_paths[(d, r)] = {'path': path_nodes, 'eta': eta}

    # RIS active edges only if edge ends at a recipient
    ris_edges = []  # tuples (ris, rec)
    for (u, v) in edges:
        if v in recipients and (u.upper().startswith('RIS') or u.upper().startswith('PIS')):
            ris_edges.append((u, v))

    # Build LP
    I = list(donor_paths.keys())  # donor-recipient pairs with a path
    K = list(ris_edges)           # ris->recipient active edges

    nI = len(I)
    nK = len(K)

    # variables: x_i for donor path delivered, y_k for ris edge delivered
    # objective: maximize sum(x_i) + sum(y_k)
    c = np.zeros(nI + nK)
    c[:nI] = -1.0
    c[nI:] = -1.0

    A_ub = []
    b_ub = []

    # donor budgets: sum_i x_i / eta_i for those with donor d
    donors_set = donors
    for d in donors_set:
        row = np.zeros(nI + nK)
        for idx, (dd, rr) in enumerate(I):
            if dd == d:
                eta = max(donor_paths[(dd, rr)]['eta'], 1e-9)
                row[idx] = 1.0 / eta
        A_ub.append(row)
        b_ub.append(donor_supply.get(d, 0.0))

    # RIS budgets: sum_k y_k / eta_edge <= ris_supply
    for ris in ris_nodes:
        row = np.zeros(nI + nK)
        for kk, (u, v) in enumerate(K):
            if u == ris:
                pa = id2enu[u]
                pb = id2enu[v]
                d = float(np.linalg.norm(pa[:2] - pb[:2]))
                eta = max(efficiency(d, etacfg), 1e-9)
                row[nI + kk] = 1.0 / eta
        if np.any(row != 0):
            A_ub.append(row)
            b_ub.append(ris_supply.get(ris, 0.0))

    # recipient caps: sum donor_paths to r + sum ris_edges to r <= demand_r
    for r in recipients:
        row = np.zeros(nI + nK)
        for idx, (dd, rr) in enumerate(I):
            if rr == r:
                row[idx] = 1.0
        for kk, (u, v) in enumerate(K):
            if v == r:
                row[nI + kk] = 1.0
        A_ub.append(row)
        b_ub.append(rec_demand.get(r, 0.0))

    bounds = [(0.0, None)] * (nI + nK)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")

    x = res.x
    donor_delivered = {I[idx]: x[idx] for idx in range(nI)}
    ris_delivered = {K[kk]: x[nI + kk] for kk in range(nK)}

    # Prepare outputs
    plan = {
        'mode': 'route_constrained_max_coverage',
        'routes': {'route1': route1, 'route2': route2, 'ridge': ridge},
        'donor_paths': {f'{d}->{r}': {'path': donor_paths[(d,r)]['path'], 'eta': donor_paths[(d,r)]['eta'], 'delivered': donor_delivered.get((d,r), 0.0)} for (d,r) in I},
        'ris_edges': {f'{u}->{v}': {'eta': float(efficiency(float(np.linalg.norm(id2enu[u][:2]-id2enu[v][:2])), etacfg)), 'delivered': ris_delivered.get((u,v), 0.0)} for (u,v) in K},
        'donor_supply': donor_supply,
        'ris_supply': ris_supply,
        'recipient_demand': rec_demand,
        'objective': float(res.fun)
    }
    save_json('route_energy_plan.json', plan)

    # Visualization
    fig = go.Figure()

    # Lakes
    palette = ['#99d8c9','#a6bddb','#c7e9b4','#b3cde3','#cbe8f6','#d0f0c0']
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

    # Draw all route edges faintly
    for (u, v) in edges:
        pa, pb = id2enu[u], id2enu[v]
        fig.add_trace(go.Scatter(x=[pa[0], pb[0]], y=[pa[1], pb[1]], mode='lines',
                                 line=dict(width=1, color='rgba(120,120,120,0.4)'), showlegend=False))

    # Nodes
    def add_node(nid, symbol, color, name):
        p = id2enu.get(nid)
        if p is not None:
            fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode='markers+text', name=name,
                                     marker=dict(symbol=symbol, size=11, color=color), text=[nid], textposition='top center'))

    for d in donors:
        add_node(d, 'square', '#2ca02c', f'{d} (donor)')
    for r in recipients:
        add_node(r, 'circle', '#d62728', f'{r} (recipient)')
    for rid in ris_nodes:
        add_node(rid, 'triangle-up', '#1f77b4', f'{rid} (RIS)')

    # Donor path flows (red)
    if donor_delivered:
        max_d = max(donor_delivered.values()) if donor_delivered else 1.0
        for (d, r), val in donor_delivered.items():
            if val <= 1e-6:
                continue
            path_nodes = donor_paths[(d, r)]['path']
            for a, b in zip(path_nodes[:-1], path_nodes[1:]):
                pa, pb = id2enu[a], id2enu[b]
                fig.add_trace(go.Scatter(x=[pa[0], pb[0]], y=[pa[1], pb[1]], mode='lines',
                                         line=dict(width=2 + 6 * (val / max_d), color='rgba(200,50,50,0.85)'),
                                         showlegend=False))

    # RIS active flows (blue dashed)
    if ris_delivered:
        max_r = max(ris_delivered.values()) if ris_delivered else 1.0
        for (u, v), val in ris_delivered.items():
            if val <= 1e-6:
                continue
            pa, pb = id2enu[u], id2enu[v]
            fig.add_trace(go.Scatter(x=[pa[0], pb[0]], y=[pa[1], pb[1]], mode='lines',
                                     line=dict(width=2 + 6 * (val / max_r), color='rgba(30,120,200,0.85)', dash='dot'),
                                     showlegend=False))

    fig.update_layout(title='Route-constrained Energy Flow (donor paths + RIS active edges)',
                      xaxis=dict(title='X (m)', scaleanchor='y', scaleratio=1),
                      yaxis=dict(title='Y (m)'), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                      margin=dict(l=40, r=40, t=80, b=40))

    fig.write_html('route_energy_map.html')
    print('Saved: route_energy_plan.json and route_energy_map.html')


if __name__ == '__main__':
    run()

