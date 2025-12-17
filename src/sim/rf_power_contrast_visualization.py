"""
RF功率场对比可视化：突出显示有RIS和无RIS的差别
使用多种可视化方式：
1. 剖面图：沿着关键路径显示功率对比
2. 关键点功率对比：在各个cluster head位置对比功率
3. 功率增益等高线图：显示RIS带来的功率增益分布
"""
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional, Dict

# Ensure src on path
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
for p in [ROOT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.scenario_loader import build_transform_from_S3
from src.network.WSN import WSN
from src.tools.ris_placement_simple import evaluate_chain_final_power
from src.utils import rf_propagation_model
from src.core.RIS import RIS

_triplet_re = re.compile(r'(-?[\d\.]+)\s+(-?[\d\.]+)\s+(-?[\d\.]+)')


def parse_wkt_triplet(wkt: str) -> Optional[Tuple[float, float, float]]:
    m = _triplet_re.search(wkt or '')
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def load_ris_chain(prefix: str = 'RIS_sink->1_') -> List[np.ndarray]:
    """Load RIS chain positions from sink.csv"""
    s3_path = os.path.join('src', 'data', 'S3.csv') if os.path.exists(os.path.join('src', 'data', 'S3.csv')) else os.path.join('data', 'S3.csv')
    meta, transform = build_transform_from_S3(s3_path)
    sink_path = os.path.join('src', 'data', 'sink.csv') if os.path.exists(os.path.join('src', 'data', 'sink.csv')) else os.path.join('data', 'sink.csv')
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
                chain.append(np.array([x, y, z], dtype=float))
    return chain


def load_all_energy_transfer_ris() -> List[np.ndarray]:
    """Load all RIS panels for energy transfer paths: 0-3, 1-2, 3-4, 4-5, 2-4"""
    energy_transfer_paths = ['RIS_0->3_', 'RIS_1->2_', 'RIS_3->4_', 'RIS_4->5_', 'RIS_2->4_']
    all_ris = []
    
    s3_path = os.path.join('src', 'data', 'S3.csv') if os.path.exists(os.path.join('src', 'data', 'S3.csv')) else os.path.join('data', 'S3.csv')
    meta, transform = build_transform_from_S3(s3_path)
    sink_path = os.path.join('src', 'data', 'sink.csv') if os.path.exists(os.path.join('src', 'data', 'sink.csv')) else os.path.join('data', 'sink.csv')
    df = pd.read_csv(sink_path)
    
    for prefix in energy_transfer_paths:
        for _, row in df.iterrows():
            name = str(row['name'])
            if name.startswith(prefix):
                t = parse_wkt_triplet(row['WKT'])
                if t:
                    lon, lat, h = t
                    x, y, z = transform(lon, lat, h)
                    all_ris.append(np.array([x, y, z], dtype=float))
    
    return all_ris


def load_all_ris_chains() -> Dict[str, List[np.ndarray]]:
    """Load all RIS chains from sink.csv, grouped by route prefix"""
    s3_path = os.path.join('src', 'data', 'S3.csv') if os.path.exists(os.path.join('src', 'data', 'S3.csv')) else os.path.join('data', 'S3.csv')
    meta, transform = build_transform_from_S3(s3_path)
    sink_path = os.path.join('src', 'data', 'sink.csv') if os.path.exists(os.path.join('src', 'data', 'sink.csv')) else os.path.join('data', 'sink.csv')
    df = pd.read_csv(sink_path)
    
    # Find all unique RIS route prefixes
    prefixes = set()
    for name in df['name'].tolist():
        name_str = str(name)
        if name_str.startswith('RIS_'):
            # Extract prefix (e.g., "RIS_sink->1_" from "RIS_sink->1_1")
            parts = name_str.split('_')
            if len(parts) >= 3:
                prefix = '_'.join(parts[:3]) + '_'  # e.g., "RIS_sink->1_"
                prefixes.add(prefix)
    
    # Load chains for each prefix
    chains = {}
    for prefix in sorted(prefixes):
        chain = []
        for _, row in df.iterrows():
            name = str(row['name'])
            if name.startswith(prefix):
                t = parse_wkt_triplet(row['WKT'])
                if t:
                    lon, lat, h = t
                    x, y, z = transform(lon, lat, h)
                    chain.append(np.array([x, y, z], dtype=float))
        if chain:
            chains[prefix] = chain
    
    return chains


def w_to_dbm(pw: float) -> float:
    """Convert Watts to dBm"""
    if pw is None or pw <= 0:
        return np.nan
    return 10.0 * np.log10(pw * 1000.0)


def dbm_to_w(dbm):
    """Convert dBm to Watts (handles both scalar and array inputs)"""
    dbm = np.asarray(dbm)
    mask = np.isfinite(dbm) & (dbm > -np.inf)
    result = np.zeros_like(dbm, dtype=float)
    result[mask] = 10.0 ** ((dbm[mask] - 30.0) / 10.0)
    return result if isinstance(dbm, np.ndarray) else float(result)


class RxMock:
    """Mock receiver for power calculation"""
    def __init__(self, pos: np.ndarray, rx_gain_dbi: float = 2.0):
        self.position = pos
        self.rf_rx_gain_dbi = rx_gain_dbi


def compute_power_along_path(env, src, dst_pos: np.ndarray, ris_chain: List[np.ndarray], num_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute power along a path from source to destination.
    
    Returns:
        (distances, power_direct, power_ris) in dBm
    """
    src_pos = np.array(src.position, dtype=float)
    dst_pos = np.array(dst_pos, dtype=float)
    
    # Generate points along the path
    t_values = np.linspace(0, 1, num_points)
    distances = []
    power_direct = []
    power_ris = []
    
    los_broken = False  # 标记红线是否已经在 RIS1 处“断开”
    first_ris_dist = None
    if ris_chain:
        # 使用链路上的第一个 RIS 作为红线急剧下降的位置参考
        first_ris_dist = float(np.linalg.norm(ris_chain[0] - src_pos))
    for t in t_values:
        # Interpolate position
        pos = src_pos + t * (dst_pos - src_pos)
        x, y = pos[0], pos[1]
        z = float(env.get_elevation(x, y)) + 1.5
        point_pos = np.array([x, y, z], dtype=float)
        
        # Distance from source
        dist = float(np.linalg.norm(point_pos - src_pos))
        distances.append(dist)
        
        # Direct power（红线）：
        # - 在第一个 RIS 之前按 LoS 模型平滑衰减；
        # - 到达/超过第一个 RIS 时，急剧下降到 -100 dBm；
        # - 之后不再绘制。
        if not los_broken:
            if first_ris_dist is not None and dist >= first_ris_dist:
                power_direct.append(-100.0)
                los_broken = True
            else:
                tx_dbm = src.get_tx_power_dbm()
                has_los = env.check_los(src_pos, point_pos)
                pr_dbm = rf_propagation_model._log_distance_path_loss(
                    tx_dbm, 0.0, 2.0,
                    getattr(src, 'frequency_hz', None),
                    dist,
                    has_los
                )
                power_direct.append(pr_dbm)
        else:
            power_direct.append(np.nan)
        
        # RIS-assisted power（绿线）：遍历所有 RIS/链路取最优，再做一次简单平滑
        best_power_ris = np.nan
        if ris_chain:
            best_pw = 0.0
            # Try each RIS individually
            for ris_pos in ris_chain:
                ris_obj = RIS(panel_id=-1, position=ris_pos)
                pw = rf_propagation_model.calculate_ris_assisted_power(src, ris_obj, RxMock(point_pos), env)
                if pw > best_pw:
                    best_pw = pw
            
            # Also try chain if multiple RIS
            if len(ris_chain) > 1:
                pw_chain = evaluate_chain_final_power(src, ris_chain + [point_pos], RxMock(point_pos), env)
                if pw_chain > best_pw:
                    best_pw = pw_chain
            
            if best_pw > 0:
                best_power_ris = w_to_dbm(best_pw)
        
        power_ris.append(best_power_ris)
    
    distances = np.array(distances, dtype=float)
    power_direct = np.array(power_direct, dtype=float)
    power_ris = np.array(power_ris, dtype=float)
    
    # 对绿线做轻微平滑，消除数值抖动（仅对有限值做滑动平均）
    if np.isfinite(power_ris).any():
        smoothed = power_ris.copy()
        window = 5  # 简单 5 点窗口
        half_w = window // 2
        for i in range(len(power_ris)):
            if not np.isfinite(power_ris[i]):
                continue
            lo = max(0, i - half_w)
            hi = min(len(power_ris), i + half_w + 1)
            seg = power_ris[lo:hi]
            seg = seg[np.isfinite(seg)]
            if seg.size > 0:
                smoothed[i] = float(np.mean(seg))
        power_ris = smoothed
    
    return distances, power_direct, power_ris


def compute_power_at_points(env, src, points: List[np.ndarray], ris_chain: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power at specific points.
    
    Returns:
        (power_direct, power_ris) in dBm
    """
    power_direct = []
    power_ris = []
    
    for point_pos in points:
        point_pos = np.array(point_pos, dtype=float)
        src_pos = np.array(src.position, dtype=float)
        dist = float(np.linalg.norm(point_pos - src_pos))
        
        # Direct power（sink -> CH）
        tx_dbm = src.get_tx_power_dbm()
        has_los = env.check_los(src_pos, point_pos)
        pr_dbm = rf_propagation_model._log_distance_path_loss(
            tx_dbm, 0.0, 2.0,
            getattr(src, 'frequency_hz', None),
            dist,
            has_los
        )
        power_direct.append(pr_dbm)
        
        # RIS-assisted power：对所有参与能量路由的 RIS 逐个计算，取最优
        best_power_ris_dbm = np.nan
        if ris_chain:
            best_pw = 0.0
            for ris_pos in ris_chain:
                ris_obj = RIS(panel_id=-1, position=ris_pos)
                pw = rf_propagation_model.calculate_ris_assisted_power(src, ris_obj, RxMock(point_pos), env)
                if pw > best_pw:
                    best_pw = pw
            if best_pw > 0:
                best_power_ris_dbm = w_to_dbm(best_pw)
        
        power_ris.append(best_power_ris_dbm)
    
    return np.array(power_direct), np.array(power_ris)


def plot_cross_section(env, src, dst_pos: np.ndarray, ris_chain: List[np.ndarray], 
                       ris_positions: List[np.ndarray], out_html: str):
    """Plot power cross-section along path from source to destination"""
    distances, power_direct, power_ris = compute_power_along_path(env, src, dst_pos, ris_chain, num_points=300)
    
    fig = go.Figure()
    
    # Direct power
    fig.add_trace(go.Scatter(
        x=distances,
        y=power_direct,
        mode='lines',
        name='Direct (No RIS)',
        line=dict(color='red', width=2, dash='dash'),
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.1)'
    ))
    
    # RIS-assisted power
    fig.add_trace(go.Scatter(
        x=distances,
        y=power_ris,
        mode='lines',
        name='RIS-Assisted',
        line=dict(color='green', width=3),
        fill='tozeroy',
        fillcolor='rgba(0,255,0,0.1)'
    ))
    
    # Mark RIS positions along path
    if ris_chain:
        for i, ris_pos in enumerate(ris_chain):
            ris_dist = float(np.linalg.norm(ris_pos - src.position))
            # Find closest point on path
            idx = np.argmin(np.abs(distances - ris_dist))
            fig.add_trace(go.Scatter(
                x=[ris_dist],
                y=[power_ris[idx] if not np.isnan(power_ris[idx]) else -100],
                mode='markers+text',
                name=f'RIS{i+1}',
                marker=dict(symbol='diamond', size=15, color='blue'),
                text=[f'RIS{i+1}'],
                textposition='top center',
                showlegend=False
            ))
    
    fig.update_layout(
        title='RF Power Cross-Section: Source to Destination',
        xaxis_title='Distance from Source (m)',
        yaxis_title='Received Power (dBm)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98)
    )
    
    os.makedirs('sim', exist_ok=True)
    fig.write_html(out_html)
    print(f"Saved: {out_html}")


def plot_key_points_comparison(env, src, ch_positions: List[np.ndarray], 
                               ris_chain: List[np.ndarray], out_html: str):
    """Plot power comparison at key points (cluster heads)"""
    power_direct, power_ris = compute_power_at_points(env, src, ch_positions, ris_chain)
    
    # Calculate gain
    gain = power_ris - power_direct
    
    fig = go.Figure()
    
    x_labels = [f'CH{i}' for i in range(len(ch_positions))]
    x_pos = np.arange(len(ch_positions))
    
    # Direct power bars
    fig.add_trace(go.Bar(
        x=x_pos,
        y=power_direct,
        name='Direct (No RIS)',
        marker_color='red',
        opacity=0.7,
        text=[f'{v:.1f}' if np.isfinite(v) else '' for v in power_direct],
        textposition='outside'
    ))
    
    # RIS-assisted power bars
    fig.add_trace(go.Bar(
        x=x_pos,
        y=power_ris,
        name='RIS-Assisted',
        marker_color='green',
        opacity=0.7,
        text=[f'{v:.1f}' if np.isfinite(v) else '' for v in power_ris],
        textposition='outside'
    ))
    
    # Add gain annotations
    for i, (p_dir, p_ris, g) in enumerate(zip(power_direct, power_ris, gain)):
        if not np.isnan(g) and g > 0:
            fig.add_annotation(
                x=i,
                y=max(p_dir, p_ris) + 5,
                text=f'+{g:.1f} dB',
                showarrow=False,
                font=dict(color='green', size=12, weight='bold')
            )
    
    fig.update_layout(
        title='RF Power Comparison at Cluster Heads',
        xaxis=dict(title='Cluster Head', tickmode='array', tickvals=x_pos, ticktext=x_labels),
        yaxis=dict(title='Received Power (dBm)', range=[-120, 0]),
        template='plotly_white',
        barmode='group',
        legend=dict(x=0.02, y=0.98)
    )
    
    os.makedirs('sim', exist_ok=True)
    fig.write_html(out_html)
    print(f"Saved: {out_html}")


def plot_power_gain_contour(env, src, ris_chain: List[np.ndarray], 
                            ch_positions: List[np.ndarray],
                            ris_positions: List[np.ndarray],
                            bbox: Tuple[float, float, float, float],
                            step: float = 80.0, out_html: str = None):
    """Plot power gain contour map (RIS - Direct)"""
    xmin, xmax, ymin, ymax = bbox
    xi = np.arange(xmin, xmax + step, step, dtype=float)
    yi = np.arange(ymin, ymax + step, step, dtype=float)
    XI, YI = np.meshgrid(xi, yi)
    Z_gain = np.full_like(XI, fill_value=np.nan, dtype=float)
    H, W = XI.shape
    
    print(f"[Gain Contour] Computing power gain map: {H}x{W} grid points...")
    for i in range(H):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{H} rows")
        for j in range(W):
            x = float(XI[i, j])
            y = float(YI[i, j])
            z = float(env.get_elevation(x, y)) + 1.5
            dst = RxMock(np.array([x, y, z], dtype=float))
            dst_pos = dst.position
            src_pos = np.array(src.position, dtype=float)
            dist = float(np.linalg.norm(dst_pos - src_pos))
            
            # Direct power
            power_direct = np.nan
            if env.check_los(src_pos, dst_pos):
                tx_dbm = src.get_tx_power_dbm()
                power_direct = rf_propagation_model._log_distance_path_loss(
                    tx_dbm, 0.0, 2.0,
                    getattr(src, 'frequency_hz', None),
                    dist, True
                )
            
            # RIS-assisted power: try all RIS and take the best
            power_ris = np.nan
            if ris_chain:
                best_pw = 0.0
                # Try each RIS individually
                for ris_pos in ris_chain:
                    ris_obj = RIS(panel_id=-1, position=ris_pos)
                    pw = rf_propagation_model.calculate_ris_assisted_power(src, ris_obj, dst, env)
                    if pw > best_pw:
                        best_pw = pw
                
                # Also try chain if multiple RIS
                if len(ris_chain) > 1:
                    pw_chain = evaluate_chain_final_power(src, ris_chain + [dst_pos], dst, env)
                    if pw_chain > best_pw:
                        best_pw = pw_chain
                
                if best_pw > 0:
                    power_ris = w_to_dbm(best_pw)
            
            # Calculate gain
            if not np.isnan(power_direct) and not np.isnan(power_ris):
                Z_gain[i, j] = power_ris - power_direct
            elif not np.isnan(power_ris):
                # Only RIS has power (was blocked before)
                Z_gain[i, j] = 50.0  # Large gain indicator
    
    # Plot contour
    fig = go.Figure()
    
    # Contour plot
    fig.add_trace(go.Contour(
        z=Z_gain,
        x=xi,
        y=yi,
        colorscale='RdYlGn',
        colorbar=dict(title='Power Gain (dB)'),
        contours=dict(
            start=-20,
            end=50,
            size=5,
            showlabels=True
        ),
        name='Power Gain'
    ))
    
    # Mark cluster heads
    if ch_positions:
        ch_xs = [p[0] for p in ch_positions]
        ch_ys = [p[1] for p in ch_positions]
        fig.add_trace(go.Scatter(
            x=ch_xs, y=ch_ys,
            mode='markers+text',
            name='Cluster Heads',
            marker=dict(symbol='triangle-up', size=12, color='red',
                       line=dict(width=1, color='white')),
            text=[f'CH{i}' for i in range(len(ch_xs))],
            textposition='top center',
            showlegend=True
        ))
    
    # Mark RIS chain
    if ris_chain:
        chain_xs = [p[0] for p in ris_chain]
        chain_ys = [p[1] for p in ris_chain]
        fig.add_trace(go.Scatter(
            x=chain_xs, y=chain_ys,
            mode='markers+lines+text',
            name='RIS Chain',
            marker=dict(symbol='diamond', size=10, color='blue',
                       line=dict(width=2, color='cyan')),
            line=dict(color='cyan', width=2),
            text=[f'RIS{i+1}' for i in range(len(chain_xs))],
            textposition='top center',
            showlegend=True
        ))
    
    fig.update_layout(
        title='RF Power Gain: RIS-Assisted vs Direct (dB)',
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        xaxis=dict(scaleanchor='y', scaleratio=1),
        template='plotly_white'
    )
    
    if out_html is None:
        out_html = 'sim/power_gain_contour.html'
    os.makedirs('sim', exist_ok=True)
    fig.write_html(out_html)
    print(f"Saved: {out_html}")


def main(route_prefix: str = None, target_ch_id: int = 1, use_all_ris: bool = True):
    """Main function to generate all visualizations"""
    print("=" * 60)
    print("RF Power Contrast Visualization Generator")
    print("=" * 60)
    
    # Initialize WSN
    print("\n[1/4] Initializing WSN...")
    wsn = WSN()
    env = wsn.environment
    sink = wsn.rf_transmitter
    
    # Get cluster head positions
    ch_positions = [cl.cluster_head.position for cl in wsn.clusters]
    print(f"  Found {len(ch_positions)} cluster heads")
    
    # Get RIS positions
    ris_positions = [ris.position for ris in wsn.ris_panels]
    print(f"  Found {len(ris_positions)} RIS panels")
    
    # Load RIS chain - only for energy transfer paths: 0-3, 1-2, 3-4, 4-5, 2-4
    print(f"\n[2/4] Loading RIS chain(s)...")
    if use_all_ris:
        # Use only RIS panels for energy transfer paths
        ris_chain = load_all_energy_transfer_ris()
        print(f"  Using {len(ris_chain)} RIS panels from energy transfer paths (0-3, 1-2, 3-4, 4-5, 2-4)")
    else:
        if route_prefix is None:
            route_prefix = 'RIS_0->3_'  # Default to 0->3 path
        ris_chain = load_ris_chain(route_prefix)
        print(f"  Found {len(ris_chain)} RIS panels in chain (prefix: {route_prefix})")
    
    # Get full DEM bounding box
    if hasattr(env, 'dem') and env.dem is not None:
        ox, oy = env.origin_xy
        H, W = env.dem.shape
        res = env.resolution
        bbox = (float(ox), float(ox) + W * res, float(oy), float(oy) + H * res)
    else:
        # Fallback
        all_x = [p[0] for p in ch_positions] + [sink.position[0]]
        all_y = [p[1] for p in ch_positions] + [sink.position[1]]
        bbox = (min(all_x) - 500, max(all_x) + 500, min(all_y) - 500, max(all_y) + 500)
    
    # Generate visualizations
    # 3a) Cross-section plots for specific cluster heads from sink (CH2, CH4, CH5)
    print("\n[3/4] Generating cross-section plots from sink for CH2, CH4, CH5...")
    target_indices = [2, 4, 5]  # zero-based cluster ids
    for idx in target_indices:
        if 0 <= idx < len(ch_positions):
            target_ch = ch_positions[idx]
            print(f"  Cross-section: sink -> CH{idx} at {target_ch}")
            out_html_cs = f'sim/power_cross_section_CH{idx}.html'
            plot_cross_section(env, sink, target_ch, ris_chain, ris_positions, out_html_cs)

    # 3b) Cross-section plots for energy transfer paths: CH0->CH3, CH1->CH2
    print("\n[3b/4] Generating cross-section plots for energy paths CH0->CH3 and CH1->CH2...")
    energy_paths = [
        (0, 3, 'RIS_0->3_', 'sim/power_cross_section_CH0_CH3.html'),
        (1, 2, 'RIS_1->2_', 'sim/power_cross_section_CH1_CH2.html'),
    ]
    for src_idx, dst_idx, route_prefix_path, out_html_path in energy_paths:
        if 0 <= src_idx < len(ch_positions) and 0 <= dst_idx < len(ch_positions):
            ch_src_pos = ch_positions[src_idx]
            ch_dst_pos = ch_positions[dst_idx]
            print(f"  Path cross-section: CH{src_idx} -> CH{dst_idx}, route {route_prefix_path}")
            # Use CH source instead of sink, and path-specific RIS chain
            ch_src = wsn.clusters[src_idx].cluster_head
            path_ris_chain = load_ris_chain(route_prefix_path)
            plot_cross_section(env, ch_src, ch_dst_pos, path_ris_chain, ris_positions, out_html_path)
    
    print("\n[4/4] Generating key points comparison...")
    plot_key_points_comparison(env, sink, ch_positions, ris_chain,
                              'sim/power_key_points_comparison.html')
    
    print("\n[5/5] Generating power gain contour...")
    plot_power_gain_contour(env, sink, ris_chain, ch_positions, ris_positions, bbox,
                           step=80.0, out_html='sim/power_gain_contour.html')
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
    print("\nOutput files:")
    print("  1. sim/power_cross_section.html - Power along path (cross-section)")
    print("  2. sim/power_key_points_comparison.html - Power at cluster heads")
    print("  3. sim/power_gain_contour.html - Power gain contour map")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate RF power contrast visualizations')
    parser.add_argument('--route-prefix', type=str, default=None,
                       help='RIS chain route prefix (default: None, use all RIS)')
    parser.add_argument('--target-ch', type=int, default=1,
                       help='Target cluster head ID for cross-section (default: 1)')
    parser.add_argument('--use-all-ris', action='store_true', default=True,
                       help='Use all RIS panels (default: True)')
    args = parser.parse_args()
    
    main(route_prefix=args.route_prefix, target_ch_id=args.target_ch, use_all_ris=args.use_all_ris)

