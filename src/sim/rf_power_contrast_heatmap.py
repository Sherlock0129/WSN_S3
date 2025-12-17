"""
生成RF功率场对比热力图：有RIS vs 无RIS
- 无RIS：山阴面大面积蓝色（能量无法越过山脊）
- 有RIS：山阴面也有高亮（能量通过RIS越过山脊）

输出4个文件：
1. direct_heatmap_full.html - 无RIS，铺满整个DEM
2. ris_cross_heatmap_full.html - 有RIS，铺满整个DEM
3. ris_vs_direct_side_by_side.html - 左右对比
4. ris_minus_direct.html - 差值热力图
"""
import os
import sys
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional

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


def load_all_ris_positions() -> List[np.ndarray]:
    """Load all RIS panel positions from sink.csv (used as fallback when no specific chain)."""
    s3_path = os.path.join('src', 'data', 'S3.csv') if os.path.exists(os.path.join('src', 'data', 'S3.csv')) else os.path.join('data', 'S3.csv')
    meta, transform = build_transform_from_S3(s3_path)
    sink_path = os.path.join('src', 'data', 'sink.csv') if os.path.exists(os.path.join('src', 'data', 'sink.csv')) else os.path.join('data', 'sink.csv')
    df = pd.read_csv(sink_path)
    ris_positions: List[np.ndarray] = []
    for _, row in df.iterrows():
        name = str(row['name'])
        if name.startswith('RIS_'):
            t = parse_wkt_triplet(row['WKT'])
            if t:
                lon, lat, h = t
                x, y, z = transform(lon, lat, h)
                ris_positions.append(np.array([x, y, z], dtype=float))
    return ris_positions


def w_to_dbm(pw):
    """Convert Watts to dBm (handles both scalar and array inputs)"""
    pw = np.asarray(pw)
    # Handle None, NaN, and non-positive values
    mask = np.isfinite(pw) & (pw > 0)
    result = np.full_like(pw, np.nan, dtype=float)
    result[mask] = 10.0 * np.log10(pw[mask] * 1000.0)
    return result if isinstance(pw, np.ndarray) or isinstance(pw, (list, tuple)) else float(result)


def dbm_to_w(dbm):
    """Convert dBm to Watts (handles both scalar and array inputs)"""
    dbm = np.asarray(dbm)
    # Handle NaN and -inf values
    mask = np.isfinite(dbm) & (dbm > -np.inf)
    result = np.zeros_like(dbm, dtype=float)
    result[mask] = 10.0 ** ((dbm[mask] - 30.0) / 10.0)
    return result if isinstance(dbm, np.ndarray) else float(result)


class RxMock:
    """Mock receiver for power calculation"""
    def __init__(self, pos: np.ndarray, rx_gain_dbi: float = 2.0):
        self.position = pos
        self.rf_rx_gain_dbi = rx_gain_dbi


def compute_direct_power_heatmap(env, src, bbox: Tuple[float, float, float, float], step: float = 40.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute direct (no RIS) RF power heatmap.
    
    Args:
        env: Environment object
        src: RF transmitter (sink)
        bbox: (xmin, xmax, ymin, ymax) bounding box
        step: Grid step size in meters
    
    Returns:
        (xi, yi, Z) where Z is power in dBm
    """
    xmin, xmax, ymin, ymax = bbox
    xi = np.arange(xmin, xmax + step, step, dtype=float)
    yi = np.arange(ymin, ymax + step, step, dtype=float)
    XI, YI = np.meshgrid(xi, yi)
    Z = np.full_like(XI, fill_value=np.nan, dtype=float)
    H, W = XI.shape
    
    print(f"[direct] Computing direct power heatmap: {H}x{W} grid points...")
    for i in range(H):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{H} rows")
        for j in range(W):
            x = float(XI[i, j])
            y = float(YI[i, j])
            z = float(env.get_elevation(x, y)) + 1.5  # 1.5m above ground
            dst = np.array([x, y, z], dtype=float)
            
            # Check line-of-sight
            if not env.check_los(src.position, dst):
                continue
            
            # Calculate received power using log-distance path loss
            dist = float(np.linalg.norm(src.position - dst))
            tx_dbm = src.get_tx_power_dbm()
            # Direct path: reflection_gain = 0, rx_gain = 2 dBi, LoS = True
            pr_dbm = rf_propagation_model._log_distance_path_loss(
                tx_dbm, 0.0, 2.0, 
                getattr(src, 'frequency_hz', None), 
                dist, True
            )
            Z[i, j] = pr_dbm
    
    print(f"[direct] Completed direct power heatmap")
    return xi, yi, Z


def compute_ris_power_heatmap(env, src, ris_chain: List[np.ndarray], bbox: Tuple[float, float, float, float], step: float = 40.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute RIS-assisted RF power heatmap.
    
    Args:
        env: Environment object
        src: RF transmitter (sink)
        ris_chain: List of RIS panel positions
        bbox: (xmin, xmax, ymin, ymax) bounding box
        step: Grid step size in meters
    
    Returns:
        (xi, yi, Z) where Z is power in dBm
    """
    xmin, xmax, ymin, ymax = bbox
    xi = np.arange(xmin, xmax + step, step, dtype=float)
    yi = np.arange(ymin, ymax + step, step, dtype=float)
    XI, YI = np.meshgrid(xi, yi)
    Z = np.full_like(XI, fill_value=np.nan, dtype=float)
    H, W = XI.shape
    
    print(f"[RIS] Computing RIS-assisted power heatmap: {H}x{W} grid points...")
    valid_count = 0
    for i in range(H):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{H} rows, valid points: {valid_count}")
        for j in range(W):
            x = float(XI[i, j])
            y = float(YI[i, j])
            z = float(env.get_elevation(x, y)) + 1.5  # 1.5m above ground
            dst = RxMock(np.array([x, y, z], dtype=float))
            
            # Evaluate power through RIS chain: src -> RIS1 -> RIS2 -> ... -> dst
            if ris_chain:
                # First try strict LoS path
                pw = evaluate_chain_final_power(src, ris_chain + [dst.position], dst, env)
                
                # If strict LoS fails, try computing power from last RIS to target
                # (even without LoS, apply path loss with additional attenuation)
                if pw <= 0 and len(ris_chain) > 0:
                    last_ris_pos = ris_chain[-1]
                    dist_ris_target = float(np.linalg.norm(last_ris_pos - dst.position))
                    has_los = env.check_los(last_ris_pos, dst.position)
                    
                    # Calculate power from source to last RIS
                    if env.check_los(src.position, last_ris_pos):
                        dist_src_ris = float(np.linalg.norm(src.position - last_ris_pos))
                        tx_dbm = src.get_tx_power_dbm()
                        # Power at RIS (assuming RIS acts as receiver then transmitter)
                        power_at_ris_dbm = rf_propagation_model._log_distance_path_loss(
                            tx_dbm, 0.0, 0.0,  # No reflection gain for intermediate calculation
                            getattr(src, 'frequency_hz', None),
                            dist_src_ris, True
                        )
                        
                        # Power from RIS to target (with RIS reflection gain if LoS, otherwise extra attenuation)
                        if has_los:
                            # Use RIS reflection gain
                            from src.core.RIS import RIS
                            ris_obj = RIS(panel_id=-1, position=last_ris_pos)
                            reflection_gain = ris_obj.get_reflection_gain()
                        else:
                            # No LoS: apply extra 20dB attenuation for NLoS
                            reflection_gain = -20.0
                        
                        pr_dbm = rf_propagation_model._log_distance_path_loss(
                            power_at_ris_dbm, reflection_gain, dst.rf_rx_gain_dbi,
                            getattr(src, 'frequency_hz', None),
                            dist_ris_target, has_los
                        )
                        pw = dbm_to_w(pr_dbm)
                
                if pw > 0:
                    dbm_val = w_to_dbm(pw)
                    if np.isfinite(dbm_val):
                        Z[i, j] = dbm_val
                        valid_count += 1
            else:
                # No RIS chain, fallback to direct
                if env.check_los(src.position, dst.position):
                    dist = float(np.linalg.norm(src.position - dst.position))
                    tx_dbm = src.get_tx_power_dbm()
                    pr_dbm = rf_propagation_model._log_distance_path_loss(
                        tx_dbm, 0.0, 2.0,
                        getattr(src, 'frequency_hz', None),
                        dist, True
                    )
                    Z[i, j] = pr_dbm
                    valid_count += 1
    
    print(f"[RIS] Completed RIS-assisted power heatmap. Valid points: {valid_count}/{H*W} ({100*valid_count/(H*W):.1f}%)")
    return xi, yi, Z


def get_full_dem_bbox(env) -> Tuple[float, float, float, float]:
    """Get bounding box covering full DEM extent - ensures heatmap covers entire DEM"""
    if env is None:
        raise ValueError("Environment is None")
    
    # Check if DEM is available
    if hasattr(env, 'dem') and env.dem is not None:
        if not hasattr(env, 'origin_xy') or env.origin_xy is None:
            raise ValueError("Environment has DEM but no origin_xy")
        if not hasattr(env, 'resolution') or env.resolution is None:
            raise ValueError("Environment has DEM but no resolution")
        
        ox, oy = env.origin_xy
        H, W = env.dem.shape
        res = env.resolution
        xmin = float(ox)
        xmax = float(ox) + W * res
        ymin = float(oy)
        ymax = float(oy) + H * res
        
        print(f"  Full DEM extent: X=[{xmin:.1f}, {xmax:.1f}], Y=[{ymin:.1f}, {ymax:.1f}]")
        print(f"  DEM size: {W}x{H} cells, resolution: {res:.2f} m/cell")
        return xmin, xmax, ymin, ymax
    else:
        # Fallback: use environment width/height if available
        if hasattr(env, 'width') and hasattr(env, 'height'):
            print("  Warning: No DEM available, using environment width/height")
            xmin, ymin = 0.0, 0.0
            xmax = float(env.width)
            ymax = float(env.height)
            return xmin, xmax, ymin, ymax
        else:
            raise ValueError("Environment has no DEM and no width/height information")


def plot_single_heatmap(xi: np.ndarray, yi: np.ndarray, Z: np.ndarray,
                       ch_positions: List[np.ndarray],
                       ris_positions: List[np.ndarray],
                       ris_chain: List[np.ndarray],
                       title: str,
                       out_html: str,
                       zmin: Optional[float] = None,
                       zmax: Optional[float] = None):
    """Plot a single heatmap with cluster heads and RIS markers"""
    # Robust z-scale
    if zmin is None:
        zmin = np.nanpercentile(Z, 5) if np.isfinite(Z).any() else -100.0
    if zmax is None:
        zmax = np.nanpercentile(Z, 95) if np.isfinite(Z).any() else -50.0
    
    fig = go.Figure()
    
    # Main heatmap
    fig.add_trace(go.Heatmap(
        z=Z, x=xi, y=yi,
        colorscale='Turbo',
        colorbar=dict(title='Received Power (dBm)'),
        zmin=zmin, zmax=zmax
    ))
    
    # Mark all cluster heads
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
    
    # Mark all RIS panels
    if ris_positions:
        ris_xs = [p[0] for p in ris_positions]
        ris_ys = [p[1] for p in ris_positions]
        fig.add_trace(go.Scatter(
            x=ris_xs, y=ris_ys,
            mode='markers',
            name='RIS Panels',
            marker=dict(symbol='star', size=10, color='yellow',
                       line=dict(width=1, color='black')),
            showlegend=True
        ))
    
    # Mark RIS chain (if different from all RIS)
    if ris_chain:
        chain_xs = [p[0] for p in ris_chain]
        chain_ys = [p[1] for p in ris_chain]
        # Only add if not already in ris_positions
        fig.add_trace(go.Scatter(
            x=chain_xs, y=chain_ys,
            mode='markers+lines+text',
            name='RIS Chain',
            marker=dict(symbol='diamond', size=8, color='blue',
                       line=dict(width=2, color='cyan')),
            line=dict(color='cyan', width=2),
            text=[f'RIS{i+1}' for i in range(len(chain_xs))],
            textposition='top center',
            showlegend=True
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        xaxis=dict(scaleanchor='y', scaleratio=1),
        template='plotly_white'
    )
    
    os.makedirs('sim', exist_ok=True)
    fig.write_html(out_html)
    print(f"Saved: {out_html}")
    return out_html


def plot_coverage_masks(xi: np.ndarray, yi: np.ndarray,
                       mask_direct: np.ndarray,
                       mask_ris: np.ndarray,
                       out_html: str):
    """
    Plot coverage masks for direct vs RIS.
    Values:
      0: no coverage
      1: direct only
      2: RIS only
      3: both
    """
    coverage = np.zeros_like(mask_direct, dtype=int)
    coverage[np.logical_and(mask_direct, ~mask_ris)] = 1
    coverage[np.logical_and(~mask_direct, mask_ris)] = 2
    coverage[np.logical_and(mask_direct, mask_ris)] = 3

    colorscale = [
        [0.0, 'rgba(200,200,200,0.6)'],
        [0.25, 'rgba(200,200,200,0.6)'],
        [0.25, 'rgba(0,120,200,0.8)'],
        [0.5, 'rgba(0,120,200,0.8)'],
        [0.5, 'rgba(0,180,0,0.8)'],
        [0.75, 'rgba(0,180,0,0.8)'],
        [0.75, 'rgba(240,120,0,0.8)'],
        [1.0, 'rgba(240,120,0,0.8)'],
    ]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=coverage, x=xi, y=yi,
        colorscale=colorscale,
        colorbar=dict(
            title='Coverage',
            tickmode='array',
            tickvals=[0, 1, 2, 3],
            ticktext=['None', 'Direct only', 'RIS only', 'Both']
        ),
        zmin=0, zmax=3
    ))
    fig.update_layout(
        title='Coverage Mask: Direct vs RIS',
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        xaxis=dict(scaleanchor='y', scaleratio=1),
        template='plotly_white'
    )
    os.makedirs('sim', exist_ok=True)
    fig.write_html(out_html)
    print(f"Saved: {out_html}")
    return out_html


def plot_side_by_side(xi: np.ndarray, yi: np.ndarray,
                      Z_direct: np.ndarray, Z_ris: np.ndarray,
                      ch_positions: List[np.ndarray],
                      ris_positions: List[np.ndarray],
                      ris_chain: List[np.ndarray],
                      out_html: str):
    """Plot side-by-side comparison"""
    # Common z-scale
    zmin = min(np.nanpercentile(Z_direct, 5), np.nanpercentile(Z_ris, 5)) if (np.isfinite(Z_direct).any() and np.isfinite(Z_ris).any()) else -100.0
    zmax = max(np.nanpercentile(Z_direct, 95), np.nanpercentile(Z_ris, 95)) if (np.isfinite(Z_direct).any() and np.isfinite(Z_ris).any()) else -50.0
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Direct (No RIS)', 'RIS-Assisted'),
        horizontal_spacing=0.1
    )
    
    # Left: Direct
    fig.add_trace(
        go.Heatmap(z=Z_direct, x=xi, y=yi, colorscale='Turbo',
                  colorbar=dict(title='Power (dBm)', x=0.45),
                  zmin=zmin, zmax=zmax, showscale=False, coloraxis='coloraxis'),
        row=1, col=1
    )
    
    # Right: RIS
    fig.add_trace(
        go.Heatmap(z=Z_ris, x=xi, y=yi, colorscale='Turbo',
                  colorbar=dict(title='Power (dBm)', x=1.02),
                  zmin=zmin, zmax=zmax, showscale=True, coloraxis='coloraxis'),
        row=1, col=2
    )
    
    # Add cluster heads to both subplots
    if ch_positions:
        ch_xs = [p[0] for p in ch_positions]
        ch_ys = [p[1] for p in ch_positions]
        for col in [1, 2]:
            fig.add_trace(
                go.Scatter(x=ch_xs, y=ch_ys, mode='markers+text',
                          marker=dict(symbol='triangle-up', size=10, color='red',
                                     line=dict(width=1, color='white')),
                          text=[f'CH{i}' for i in range(len(ch_xs))],
                          textposition='top center',
                          showlegend=(col == 1),
                          name='Cluster Heads'),
                row=1, col=col
            )
    
    # Add RIS chain to right subplot
    if ris_chain:
        chain_xs = [p[0] for p in ris_chain]
        chain_ys = [p[1] for p in ris_chain]
        fig.add_trace(
            go.Scatter(x=chain_xs, y=chain_ys, mode='markers+lines+text',
                      marker=dict(symbol='diamond', size=8, color='blue',
                                 line=dict(width=2, color='cyan')),
                      line=dict(color='cyan', width=2),
                      text=[f'RIS{i+1}' for i in range(len(chain_xs))],
                      textposition='top center',
                      showlegend=True,
                      name='RIS Chain'),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text='X (m)', scaleanchor='y', scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text='X (m)', scaleanchor='y', scaleratio=1, row=1, col=2)
    fig.update_yaxes(title_text='Y (m)', row=1, col=1)
    fig.update_yaxes(title_text='Y (m)', row=1, col=2)
    fig.update_layout(
        title='RF Power Field Comparison: Direct vs RIS-Assisted',
        template='plotly_white',
        height=600,
        coloraxis=dict(colorscale='Turbo', cmin=zmin, cmax=zmax, colorbar=dict(title='Power (dBm)', x=1.02))
    )
    
    os.makedirs('sim', exist_ok=True)
    fig.write_html(out_html)
    print(f"Saved: {out_html}")
    return out_html


def plot_difference_heatmap(xi: np.ndarray, yi: np.ndarray,
                            Z_direct: np.ndarray, Z_ris: np.ndarray,
                            ch_positions: List[np.ndarray],
                            ris_positions: List[np.ndarray],
                            ris_chain: List[np.ndarray],
                            out_html: str,
                            gain_fill_db: float = 40.0):
    """
    Plot difference heatmap (RIS - Direct) in dB.
    - When both have coverage: use dB difference.
    - When only RIS has coverage: use gain_fill_db as highlight.
    - When no coverage: keep NaN (will be masked).
    """
    mask_direct = np.isfinite(Z_direct)
    mask_ris = np.isfinite(Z_ris)
    mask_both = mask_direct & mask_ris
    mask_unlock = (~mask_direct) & mask_ris

    Z_diff = np.full_like(Z_direct, np.nan, dtype=float)
    Z_diff[mask_both] = Z_ris[mask_both] - Z_direct[mask_both]
    Z_diff[mask_unlock] = gain_fill_db

    # Robust bounds: prefer fixed bounds, fallback to percentiles if data narrow
    if np.isfinite(Z_diff).any():
        zmin = max(-20.0, np.nanpercentile(Z_diff, 5))
        zmax = min(40.0, np.nanpercentile(Z_diff, 95))
        zmin = min(zmin, -20.0)
        zmax = max(zmax, 40.0)
    else:
        zmin, zmax = -20.0, 40.0
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=Z_diff, x=xi, y=yi,
        colorscale='RdBu_r',
        colorbar=dict(title='Power Gain (dB)'),
        zmin=zmin, zmax=zmax
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
            marker=dict(symbol='diamond', size=8, color='blue',
                       line=dict(width=2, color='cyan')),
            line=dict(color='cyan', width=2),
            text=[f'RIS{i+1}' for i in range(len(chain_xs))],
            textposition='top center',
            showlegend=True
        ))
    # Overlay mask for no-coverage areas (gray)
    mask_no_cov = ~(mask_direct | mask_ris)
    if mask_no_cov.any():
        mask_overlay = np.where(mask_no_cov, 1.0, np.nan)
        fig.add_trace(go.Heatmap(
            z=mask_overlay, x=xi, y=yi,
            colorscale=[[0, 'rgba(120,120,120,0.35)'], [1, 'rgba(120,120,120,0.35)']],
            showscale=False,
            hoverinfo='skip'
        ))

    # Zero-gain contour
    if np.isfinite(Z_diff).any():
        fig.add_trace(go.Contour(
            z=Z_diff, x=xi, y=yi,
            showscale=False,
            contours=dict(showlines=True, coloring='none', start=0, end=0, size=1),
            line=dict(color='black', width=1, dash='dash'),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title='RF Power Gain: RIS-Assisted vs Direct (dB)',
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        xaxis=dict(scaleanchor='y', scaleratio=1),
        template='plotly_white'
    )
    
    os.makedirs('sim', exist_ok=True)
    fig.write_html(out_html)
    print(f"Saved: {out_html}")
    return out_html


def main(route_prefix: str = 'RIS_0->5_', step: float = 40.0):
    """Main function to generate all contrast heatmaps (source: CH0 -> target: CH5 route)"""
    print("=" * 60)
    print("RF Power Field Contrast Heatmap Generator")
    print("=" * 60)
    
    # Initialize WSN
    print("\n[1/5] Initializing WSN...")
    wsn = WSN()
    env = wsn.environment
    # Use CH0 as the transmitter (阳面) instead of sink
    if len(wsn.clusters) == 0:
        raise ValueError("No clusters available to select CH0 as source.")
    src = wsn.clusters[0].cluster_head
    print(f"  Using CH0 as source at {src.position}")
    
    # Get full DEM bounding box (ensures heatmap covers entire DEM)
    print("\n[2/5] Getting full DEM extent...")
    bbox = get_full_dem_bbox(env)
    xmin, xmax, ymin, ymax = bbox
    print(f"  Grid step: {step} m")
    print(f"  Estimated grid size: {int((xmax-xmin)/step)} x {int((ymax-ymin)/step)} cells")
    
    # Get all cluster head positions
    ch_positions = [cl.cluster_head.position for cl in wsn.clusters]
    print(f"  Found {len(ch_positions)} cluster heads")
    
    # Get all RIS positions
    ris_positions = [ris.position for ris in wsn.ris_panels]
    print(f"  Found {len(ris_positions)} RIS panels")
    
    # Load RIS chain for the route
    print(f"\n[3/5] Loading RIS chain (prefix: {route_prefix})...")
    ris_chain = load_ris_chain(route_prefix)
    if len(ris_chain) == 0:
        print("  No RIS found for the route prefix, falling back to all RIS panels")
        ris_chain = load_all_ris_positions()
    print(f"  Using {len(ris_chain)} RIS panels in chain/fallback")
    
    # Compute direct power heatmap
    print("\n[4/5] Computing heatmaps...")
    xi, yi, Z_direct = compute_direct_power_heatmap(env, src, bbox, step)
    
    # Compute RIS-assisted power heatmap
    xi2, yi2, Z_ris = compute_ris_power_heatmap(env, src, ris_chain, bbox, step)
    
    # Ensure same grid
    assert np.allclose(xi, xi2) and np.allclose(yi, yi2), "Grid mismatch!"
    
    # Generate all plots
    print("\n[5/5] Generating plots...")
    
    # Common z-scale for single plots
    zmin = min(np.nanpercentile(Z_direct, 5), np.nanpercentile(Z_ris, 5)) if (np.isfinite(Z_direct).any() and np.isfinite(Z_ris).any()) else -100.0
    zmax = max(np.nanpercentile(Z_direct, 95), np.nanpercentile(Z_ris, 95)) if (np.isfinite(Z_direct).any() and np.isfinite(Z_ris).any()) else -50.0
    
    # 1. Direct heatmap (full)
    plot_single_heatmap(
        xi, yi, Z_direct,
        ch_positions, ris_positions, [],
        'Direct RF Power Field (No RIS)',
        'sim/direct_heatmap_full.html',
        zmin=zmin, zmax=zmax
    )
    
    # 2. RIS-assisted heatmap (full)
    plot_single_heatmap(
        xi, yi, Z_ris,
        ch_positions, ris_positions, ris_chain,
        'RIS-Assisted RF Power Field',
        'sim/ris_cross_heatmap_full.html',
        zmin=zmin, zmax=zmax
    )
    
    # 3. Side-by-side comparison
    plot_side_by_side(
        xi, yi, Z_direct, Z_ris,
        ch_positions, ris_positions, ris_chain,
        'sim/ris_vs_direct_side_by_side.html'
    )
    
    # 4. Difference heatmap
    plot_difference_heatmap(
        xi, yi, Z_direct, Z_ris,
        ch_positions, ris_positions, ris_chain,
        'sim/ris_minus_direct.html'
    )

    # 5. Coverage masks
    mask_direct = np.isfinite(Z_direct)
    mask_ris = np.isfinite(Z_ris)
    plot_coverage_masks(
        xi, yi, mask_direct, mask_ris,
        'sim/coverage_masks.html'
    )
    
    print("\n" + "=" * 60)
    print("All heatmaps generated successfully!")
    print("=" * 60)
    print("\nOutput files:")
    print("  1. sim/direct_heatmap_full.html - Direct (no RIS, source=CH0)")
    print("  2. sim/ris_cross_heatmap_full.html - RIS-assisted (source=CH0)")
    print("  3. sim/ris_vs_direct_side_by_side.html - Side-by-side comparison")
    print("  4. sim/ris_minus_direct.html - Difference (gain, dB)")
    print("  5. sim/coverage_masks.html - Coverage masks (direct vs RIS)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate RF power contrast heatmaps')
    parser.add_argument('--route-prefix', type=str, default='RIS_sink->1_',
                       help='RIS chain route prefix (default: RIS_sink->1_)')
    parser.add_argument('--step', type=float, default=40.0,
                       help='Grid step size in meters (default: 40.0)')
    args = parser.parse_args()
    
    main(route_prefix=args.route_prefix, step=args.step)

