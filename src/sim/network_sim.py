import os
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Ensure project root on sys.path for src imports
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.network.WSN import WSN
from src.utils.scenario_loader import build_transform_from_S3
from src.utils import rf_propagation_model
from src.config.simulation_config import ClusterHeadConfig

_triplet_re = re.compile(r'(-?[\d\.]+ -?[\d\.]+ -?[\d\.]+)')


def parse_wkt_points(wkt: str):
    return [list(map(float, s.split())) for s in _triplet_re.findall(wkt or '')]


def resolve_path(path: str) -> str:
    candidates = [
        path,
        os.path.join(ROOT_DIR, path),
        path.replace('src/', 'data/'),
        path.replace('data/', 'src/data/'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


def link_eta(src_pos: np.ndarray, dst_pos: np.ndarray, env, cfg: dict) -> float:
    """LoS + 3D 距离 + 物理功率模型转效率"""
    min_eta = float(cfg.get('min_eta', 1e-4))
    # LoS 阻挡直接视为不可达
    if not env.check_los(src_pos, dst_pos):
        return min_eta

    dist = float(np.linalg.norm(src_pos - dst_pos))
    if dist <= 1e-6:
        return 1.0

    tx_power_w = float(cfg.get('tx_power_w', ClusterHeadConfig.CH_RF_TX_POWER_W))
    tx_power_dbm = float(cfg.get('tx_power_dbm', 10 * np.log10(tx_power_w * 1000.0)))
    tx_gain = float(cfg.get('tx_gain_dbi', ClusterHeadConfig.CH_RF_TX_GAIN_DBI))
    rx_gain = float(cfg.get('rx_gain_dbi', ClusterHeadConfig.RF_RX_GAIN_DBI))
    freq = float(cfg.get('frequency_hz', ClusterHeadConfig.CH_RF_TX_FREQUENCY_HZ))

    pr_dbm = rf_propagation_model._log_distance_path_loss(
        tx_power_dbm,
        tx_gain,
        rx_gain,
        freq,
        dist,
        True,  # 已检查 LoS
    )
    pr_w = 10 ** ((pr_dbm - 30) / 10)
    eta = pr_w / max(tx_power_w, 1e-9)
    return float(np.clip(eta, min_eta, 1.0))


def solve_lp_max_coverage(donors, recipients, etacfg, env):
    # donors/recipients: id -> {'enu': np.array, 'supply' or 'demand'}
    I = list(donors.keys())
    J = list(recipients.keys())
    nI, nJ = len(I), len(J)
    # build eta matrix
    eta = np.zeros((nI, nJ))
    for ii, i in enumerate(I):
        for jj, j in enumerate(J):
            eta[ii, jj] = link_eta(donors[i]['enu'], recipients[j]['enu'], env, etacfg)
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


def build_env_and_positions(install_height: float = 1.5):
    """构建与 RIS 放置一致的环境 + 3D 位置（地表+安装高），LoS 可用。"""
    # 环境与 DEM
    wsn = WSN()
    env = wsn.environment

    # 坐标变换
    s3_path = resolve_path('src/data/S3.csv') if os.path.exists(resolve_path('src/data/S3.csv')) else resolve_path('data/S3.csv')
    sink_path = resolve_path('src/data/sink.csv') if os.path.exists(resolve_path('src/data/sink.csv')) else resolve_path('data/sink.csv')
    meta, transform = build_transform_from_S3(s3_path)

    df = pd.read_csv(sink_path)
    pos = {}
    for _, row in df.iterrows():
        wkt = row.get('WKT', '')
        pts = parse_wkt_points(wkt)
        if not pts:
            continue
        lon, lat, h = pts[0]
        xyz = transform(lon, lat, h)
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        z = env.get_elevation(x, y) + install_height
        pid = str(row.get('name', row.get('id', '')))
        if pid:
            pos[pid] = np.array([x, y, z], dtype=float)
    return env, pos


def simulate(days: int, etacfg: dict, donor_sup_daily: dict, rec_dem_daily: dict, env=None, pos=None):
    """
    两阶段逻辑：
      1) RF1/2（阳面）向所有阴面簇(3-6)送能
      2) RF3/4 将自身收到的能量作为二次供给，向 RF5/6 转发
    受能端不设需求上限，目标是尽可能送达更多能量。
    """
    if env is None or pos is None:
        env, pos = build_env_and_positions(etacfg.get('install_height', 1.5))
    donors_base = ['RF1', 'RF2']
    donors_forward = ['RF3', 'RF4']
    recip_ids = ['RF3', 'RF4', 'RF5', 'RF6']
    donors_all = donors_base + donors_forward
    missing = [nid for nid in donors_all + recip_ids if nid not in pos]
    if missing:
        raise RuntimeError(f"Missing positions for nodes: {missing}")
    outputs = []
    for day in range(1, days + 1):
        # 阶段1：阳面 -> 阴面（所有）
        donors1 = {i: {'enu': pos[i], 'supply': float(donor_sup_daily.get(i, 0.0))} for i in donors_base}
        recips_all = {j: {'enu': pos[j], 'demand': float(rec_dem_daily.get(j, 0.0))} for j in recip_ids}
        plan1 = solve_lp_max_coverage(donors1, recips_all, etacfg, env)

        # 阶段2：RF3/4 将收到的能量转发给 RF5/6
        supply_stage2 = {}
        delivered_stage1 = plan1['delivered']
        for idx, did in enumerate(['RF3', 'RF4']):
            supply_stage2[did] = float(delivered_stage1[:, idx].sum())
        donors2 = {i: {'enu': pos[i], 'supply': supply_stage2[i]} for i in donors_forward if supply_stage2.get(i, 0.0) > 1e-9}
        plan2 = None
        recip_stage2_ids = ['RF5', 'RF6']
        if donors2:
            recips_stage2 = {j: recips_all[j] for j in recip_stage2_ids}
            plan2 = solve_lp_max_coverage(donors2, recips_stage2, etacfg, env)

        # 汇总到统一矩阵（donors_all x recip_ids）
        nI, nJ = len(donors_all), len(recip_ids)
        agg_del = np.zeros((nI, nJ))
        agg_tx = np.zeros((nI, nJ))
        # 阶段1填充
        agg_del[:len(donors_base), :] = delivered_stage1
        agg_tx[:len(donors_base), :] = plan1['transmit']
        # 阶段2填充（仅 RF3/4 -> RF5/6）
        if plan2:
            for ii, did in enumerate(plan2['I']):
                di = donors_all.index(did)
                for jj, rid in enumerate(plan2['J']):
                    rj = recip_ids.index(rid)
                    agg_del[di, rj] = plan2['delivered'][ii, jj]
                    agg_tx[di, rj] = plan2['transmit'][ii, jj]

        outputs.append({
            'day': day,
            'delivered': agg_del.tolist(),
            'transmit': agg_tx.tolist(),
            'donor_ids': donors_all,
            'recipient_ids': recip_ids,
            'eta': None,
            'objective': plan1['objective'] + (plan2['objective'] if plan2 else 0.0)
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
    etacfg = cfg.get('efficiency', {})
    env, pos = build_env_and_positions(etacfg.get('install_height', 1.5))
    outputs = simulate(cfg.get('days', 3), etacfg, cfg.get('donor_surplus_daily', {}), cfg.get('recipient_demand_daily', {}), env, pos)
    # save
    os.makedirs('sim', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sim_path = os.path.join('sim', f'network_sim_{ts}.json')
    with open(sim_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    # aggregate and save summary JSON
    summary = aggregate_and_save(outputs)
    # plot summary
    plot_summary(outputs, pos)
    print(f"Saved sim results: {sim_path}")
    print("Summary JSON: sim/network_summary.json; map: sim/network_energy_summary.html")


if __name__ == '__main__':
    main()

