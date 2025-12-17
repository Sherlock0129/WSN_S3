"""
Energy distribution heatmap (spatial) for sensor nodes.

- Builds a 2D grid over the simulation area and aggregates node energies per cell (mean).
- For empty cells, performs height-weighted interpolation using nearby nodes.
- Produces both a static PNG (matplotlib) and an interactive HTML (plotly) when available.
"""
from __future__ import annotations
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

from src.config.simulation_config import EnvConfig, SimConfig, SensorNodeConfig


def _is_ch(node_id: str) -> bool:
    return 'CH' in str(node_id)


def _extract_xy(positions: Dict[str, np.ndarray], node_ids: List[str], sensors_only=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    xs, ys, zs, ids = [], [], [], []
    for nid in node_ids:
        if sensors_only and _is_ch(nid):
            continue
        p = positions.get(nid)
        if p is None:
            continue
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2] if len(p) > 2 else 0.0)
        ids.append(nid)
    return np.array(xs, dtype=float), np.array(ys, dtype=float), np.array(zs, dtype=float), ids


def _extract_energy(energy_history: Dict[str, np.ndarray], ids: List[str], step: int) -> np.ndarray:
    vals = []
    for nid in ids:
        arr = energy_history.get(nid)
        if arr is None or len(arr) == 0:
            vals.append(np.nan)
        else:
            vals.append(arr[step])
    return np.array(vals, dtype=float)


def _interpolate_empty_cells(grid: np.ndarray, 
                              xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
                              energies: np.ndarray,
                              min_x: float, max_x: float, min_y: float, max_y: float,
                              nx: int, ny: int,
                              environment = None,
                              ris_positions: List[np.ndarray] = None,
                              height_weight: float = 0.3,
                              max_search_radius: float = None):
    """
    Interpolate empty grid cells using distance-weighted and height-weighted interpolation.
    Height-based decay: higher elevation = lower energy (especially at peaks without RIS).
    
    Args:
        grid: (ny, nx) grid with NaN for empty cells
        xs, ys, zs: node positions and elevations
        energies: node energy values
        min_x, max_x, min_y, max_y: grid bounds
        nx, ny: grid dimensions
        environment: optional Environment object for elevation queries
        ris_positions: list of RIS panel positions (np.array([x,y,z]))
        height_weight: weight factor for height similarity (0-1, higher = more weight on height)
        max_search_radius: maximum search radius for interpolation (None = auto)
    """
    if max_search_radius is None:
        # Auto: use larger of 10% of dimension or 3x the typical node spacing
        dim_x = max_x - min_x
        dim_y = max_y - min_y
        # Estimate typical node spacing
        if len(xs) > 1:
            node_spacing = np.sqrt(dim_x * dim_y / len(xs))
            max_search_radius = max(dim_x * 0.1, dim_y * 0.1, node_spacing * 3.0)
        else:
            max_search_radius = max(dim_x, dim_y) * 0.1
    
    # Find empty cells
    empty_mask = np.isnan(grid)
    if not empty_mask.any():
        return grid
    
    # Get max elevation and max node energy for normalization
    if environment is not None and hasattr(environment, 'dem'):
        max_elevation = float(np.nanmax(environment.dem))
        min_elevation = float(np.nanmin(environment.dem))
    else:
        max_elevation = float(np.max(zs)) if len(zs) > 0 else 1000.0
        min_elevation = float(np.min(zs)) if len(zs) > 0 else 0.0
    
    max_node_energy = float(np.max(energies)) if len(energies) > 0 else 0.0
    min_node_energy = float(np.min(energies)) if len(energies) > 0 else 0.0
    
    # Check if RIS positions are provided
    ris_pos_2d = []
    if ris_positions is not None:
        for ris_pos in ris_positions:
            if len(ris_pos) >= 2:
                ris_pos_2d.append(np.array([ris_pos[0], ris_pos[1]]))
    
    # Grid cell centers
    cell_x = np.linspace(min_x, max_x, nx)
    cell_y = np.linspace(min_y, max_y, ny)
    
    # For each empty cell, find nearby nodes and interpolate
    grid_filled = grid.copy()
    empty_indices = np.argwhere(empty_mask)
    
    for idx_y, idx_x in empty_indices:
        cx = cell_x[idx_x]
        cy = cell_y[idx_y]
        
        # Get elevation at this cell (prefer environment, otherwise use nearest node)
        if environment is not None and hasattr(environment, 'get_elevation'):
            try:
                cz = float(environment.get_elevation(cx, cy))
            except Exception:
                # Fallback: use nearest node's elevation
                distances_2d = np.sqrt((xs - cx)**2 + (ys - cy)**2)
                if len(distances_2d) > 0:
                    nearest_idx = np.argmin(distances_2d)
                    cz = zs[nearest_idx]
                else:
                    cz = np.mean(zs) if len(zs) > 0 else 0.0
        else:
            # Use nearest node's elevation as approximation
            distances_2d = np.sqrt((xs - cx)**2 + (ys - cy)**2)
            if len(distances_2d) > 0:
                nearest_idx = np.argmin(distances_2d)
                cz = zs[nearest_idx]
            else:
                cz = np.mean(zs) if len(zs) > 0 else 0.0
        
        # Check if there's a RIS nearby (within reasonable distance, e.g., 100m)
        has_ris_nearby = False
        if ris_pos_2d:
            for ris_pos in ris_pos_2d:
                dist_to_ris = np.sqrt((ris_pos[0] - cx)**2 + (ris_pos[1] - cy)**2)
                if dist_to_ris < 100.0:  # 100m threshold
                    has_ris_nearby = True
                    break
        
        # Find nodes within search radius
        distances_2d = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        height_diffs = np.abs(zs - cz)
        
        # Combined distance metric: spatial distance + height difference
        # Normalize height difference to similar scale as spatial distance
        height_scale = max_search_radius * 0.1  # Scale height diff to ~10% of search radius
        combined_dist = distances_2d + height_weight * height_diffs * (max_search_radius / max(height_scale, 1e-6))
        
        within_radius = combined_dist <= max_search_radius
        
        # Calculate base interpolated energy
        if within_radius.any():
            # Inverse distance weighting (IDW)
            valid_dist = combined_dist[within_radius]
            valid_energies = energies[within_radius]
            
            # Avoid division by zero
            valid_dist = np.maximum(valid_dist, 1e-6)
            weights = 1.0 / (valid_dist ** 2)  # Power of 2 for IDW
            
            # Weighted average
            base_interpolated = np.sum(weights * valid_energies) / np.sum(weights)
        else:
            # If no nodes within radius, use nearest node
            if len(distances_2d) > 0:
                nearest_idx = np.argmin(distances_2d)
                base_interpolated = energies[nearest_idx]
            else:
                base_interpolated = np.mean(energies) if len(energies) > 0 else 0.0
        
        # Apply height-based decay for areas without nodes
        # Higher elevation = lower energy
        # At peak (max elevation) without RIS: energy = 0
        # At min elevation: use base interpolated energy
        elev_range = max_elevation - min_elevation
        if elev_range > 1e-6:
            # Normalized elevation (0 = min, 1 = max)
            elev_norm = (cz - min_elevation) / elev_range
            
            # If at peak and no RIS nearby, energy = 0
            if elev_norm > 0.95 and not has_ris_nearby:
                interpolated = 0.0
            else:
                # Decay factor: higher elevation = more decay
                # Linear decay from 1.0 (at min) to 0.0 (at max, if no RIS)
                if has_ris_nearby:
                    # With RIS: less decay, minimum 0.3
                    decay_factor = 1.0 - elev_norm * 0.7
                else:
                    # Without RIS: full decay
                    decay_factor = 1.0 - elev_norm
                
                interpolated = base_interpolated * max(0.0, decay_factor)
        else:
            interpolated = base_interpolated
        
        # Ensure interpolated energy doesn't exceed max node energy
        interpolated = min(interpolated, max_node_energy)
        
        grid_filled[idx_y, idx_x] = max(0.0, interpolated)
    
    return grid_filled


def plot_energy_heatmap(node_positions: Dict[str, np.ndarray],
                        energy_history: Dict[str, np.ndarray],
                        node_ids: List[str],
                        step: int = -1,
                        grid_bins: Tuple[int, int] = (100, 60),
                        out_dir: str = None,
                        environment = None,
                        enable_height_interpolation: bool = True,
                        cluster_head_positions: List[np.ndarray] = None,
                        ris_positions: List[np.ndarray] = None):
    """
    Render a spatial energy heatmap for sensor nodes at a given time step.

    Args:
        node_positions: dict node_id -> np.array([x,y,z])
        energy_history: dict node_id -> np.array(steps)
        node_ids: list of node ids
        step: which step to render (default last)
        grid_bins: (nx, ny) grid bins across the area
        out_dir: directory to place outputs. Defaults to project root / 'outputs'
        environment: optional Environment object for elevation queries
        enable_height_interpolation: if True, interpolate empty cells using height-weighted method
        cluster_head_positions: list of cluster head positions (np.array([x,y,z]))
        ris_positions: list of RIS panel positions (np.array([x,y,z]))
    """
    if out_dir is None:
        proj_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        out_dir = os.path.join(proj_root, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    # Select sensor nodes only
    xs, ys, zs, ids = _extract_xy(node_positions, node_ids, sensors_only=True)
    if xs.size == 0:
        print('[energy_heatmap] No sensor node positions found.')
        return

    energies = _extract_energy(energy_history, ids, step)
    
    # Get elevations from environment if available
    if environment is not None and hasattr(environment, 'get_elevation'):
        try:
            zs = np.array([environment.get_elevation(x, y) for x, y in zip(xs, ys)], dtype=float)
        except Exception:
            pass  # Fall back to z from positions

    # Build grid - use full DEM extent if environment available, otherwise use node bounds
    if environment is not None and hasattr(environment, 'origin_xy') and hasattr(environment, 'resolution') and hasattr(environment, 'dem'):
        # Use full DEM extent
        ox, oy = environment.origin_xy
        H, W = environment.dem.shape
        res = environment.resolution
        min_x = float(ox)
        max_x = float(ox) + W * res
        min_y = float(oy)
        max_y = float(oy) + H * res
        print(f'[energy_heatmap] Using full DEM extent: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})')
    else:
        # Fallback to node bounds
        min_x, max_x = float(np.min(xs)), float(np.max(xs))
        min_y, max_y = float(np.min(ys)), float(np.max(ys))
        # Add small padding
        padding_x = (max_x - min_x) * 0.05 if max_x > min_x else 100.0
        padding_y = (max_y - min_y) * 0.05 if max_y > min_y else 100.0
        min_x -= padding_x
        max_x += padding_x
        min_y -= padding_y
        max_y += padding_y
        print(f'[energy_heatmap] Using node bounds with padding: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})')
    
    nx, ny = grid_bins

    # Avoid zero-size extents
    if max_x - min_x < 1e-6:
        max_x += 1.0
        min_x -= 1.0
    if max_y - min_y < 1e-6:
        max_y += 1.0
        min_y -= 1.0

    # Bin indices
    ix = np.clip(((xs - min_x) / (max_x - min_x) * nx).astype(int), 0, nx - 1)
    iy = np.clip(((ys - min_y) / (max_y - min_y) * ny).astype(int), 0, ny - 1)

    acc = np.zeros((ny, nx), dtype=float)
    cnt = np.zeros((ny, nx), dtype=int)

    for e, i, j in zip(energies, ix, iy):
        if np.isnan(e):
            continue
        acc[ny - 1 - j, i] += e  # flip Y for imshow origin='upper'
        cnt[ny - 1 - j, i] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        grid = acc / np.maximum(cnt, 1)
        grid[cnt == 0] = np.nan
    
    # Interpolate empty cells using height-weighted method
    if enable_height_interpolation and np.isnan(grid).any():
        print('[energy_heatmap] Interpolating empty cells using height-weighted method...')
        grid = _interpolate_empty_cells(grid, xs, ys, zs, energies, 
                                       min_x, max_x, min_y, max_y, nx, ny,
                                       environment=environment,
                                       ris_positions=ris_positions,
                                       height_weight=0.3)

    # Plot static PNG
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(grid, cmap='viridis', interpolation='nearest',
                   extent=(min_x, max_x, min_y, max_y), origin='lower', aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Energy (J)', fontsize=12)
    
    # Mark cluster head positions
    if cluster_head_positions is not None:
        ch_xs, ch_ys = [], []
        for ch_pos in cluster_head_positions:
            if len(ch_pos) >= 2:
                ch_xs.append(ch_pos[0])
                ch_ys.append(ch_pos[1])
        if ch_xs:
            ax.scatter(ch_xs, ch_ys, c='red', marker='^', s=100, 
                      edgecolors='white', linewidths=1.5, 
                      label='Cluster Heads', zorder=5)
            # Add text labels
            for i, (x, y) in enumerate(zip(ch_xs, ch_ys)):
                ax.annotate(f'CH{i}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8,
                           color='white', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    
    # Mark RIS positions
    if ris_positions is not None:
        ris_xs, ris_ys = [], []
        for ris_pos in ris_positions:
            if len(ris_pos) >= 2:
                ris_xs.append(ris_pos[0])
                ris_ys.append(ris_pos[1])
        if ris_xs:
            ax.scatter(ris_xs, ris_ys, c='yellow', marker='*', s=80,
                      edgecolors='black', linewidths=1,
                      label='RIS Panels', zorder=4, alpha=0.8)
    
    ax.set_title('Sensor Energy Spatial Heatmap', fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    if cluster_head_positions is not None or ris_positions is not None:
        ax.legend(loc='upper right', fontsize=9)
    png_path = os.path.join(out_dir, 'energy_heatmap.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    try:
        plt.close(fig)
    except Exception:
        pass
    print(f'[energy_heatmap] Saved PNG: {png_path}')

    # Optional interactive
    if _HAS_PLOTLY:
        # Replace NaNs for heatmap (plotly accepts None)
        z = np.where(np.isnan(grid), None, grid)
        fig2 = go.Figure(data=go.Heatmap(z=z, x=np.linspace(min_x, max_x, nx), y=np.linspace(min_y, max_y, ny),
                                         colorscale='Viridis', colorbar=dict(title='Energy (J)')))
        
        # Add cluster head markers
        if cluster_head_positions is not None:
            ch_xs, ch_ys = [], []
            for ch_pos in cluster_head_positions:
                if len(ch_pos) >= 2:
                    ch_xs.append(ch_pos[0])
                    ch_ys.append(ch_pos[1])
            if ch_xs:
                fig2.add_trace(go.Scatter(x=ch_xs, y=ch_ys, mode='markers+text',
                                        marker=dict(symbol='triangle-up', size=12, color='red',
                                                   line=dict(width=1, color='white')),
                                        text=[f'CH{i}' for i in range(len(ch_xs))],
                                        textposition='top center',
                                        name='Cluster Heads',
                                        showlegend=True))
        
        # Add RIS markers
        if ris_positions is not None:
            ris_xs, ris_ys = [], []
            for ris_pos in ris_positions:
                if len(ris_pos) >= 2:
                    ris_xs.append(ris_pos[0])
                    ris_ys.append(ris_pos[1])
            if ris_xs:
                fig2.add_trace(go.Scatter(x=ris_xs, y=ris_ys, mode='markers',
                                        marker=dict(symbol='star', size=10, color='yellow',
                                                   line=dict(width=1, color='black')),
                                        name='RIS Panels',
                                        showlegend=True))
        
        fig2.update_layout(title='Sensor Energy Spatial Heatmap', xaxis_title='X (m)', yaxis_title='Y (m)',
                           template='plotly_white')
        html_path = os.path.join(out_dir, 'energy_heatmap.html')
        fig2.write_html(html_path, include_plotlyjs='cdn')
        print(f'[energy_heatmap] Saved HTML: {html_path}')

