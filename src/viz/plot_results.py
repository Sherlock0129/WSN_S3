"""
Visualization script for plotting simulation results.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    go = None

from src.config.simulation_config import SimConfig


def _time_axis_minutes(num_steps: int):
    # Convert to minutes for higher time resolution on x-axis
    return (np.arange(num_steps) * SimConfig.TIME_STEP_S) / 60.0


def plot_energy_history(energy_history, node_ids):
    """
    Plots the energy level of each node over time (static PNG, all nodes).
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    num_steps = len(next(iter(energy_history.values())))
    time_axis = np.arange(num_steps) * SimConfig.TIME_STEP_S / 60  # minutes

    # Separate Cluster Heads and Sensor Nodes for different line styles
    ch_ids = [nid for nid in node_ids if 'CH' in str(nid)]
    sn_ids = [nid for nid in node_ids if 'CH' not in str(nid)]

    # Plot sensor nodes
    for node_id in sn_ids:
        ax.plot(time_axis, energy_history[node_id], lw=1.0, alpha=0.6, label=f'Node {node_id}')

    # Plot cluster heads with thicker lines
    for node_id in ch_ids:
        ax.plot(time_axis, energy_history[node_id], lw=2.5, linestyle='--', label=f'CH {node_id}')

    ax.set_xlabel("Time (minutes)", fontsize=14)
    ax.set_ylabel("Energy (Joules)", fontsize=14)
    # 移除大标题（按需可在此处恢复）
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend

    # Save the figure to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_path = os.path.join(project_root, "simulation_energy_results.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n结果图已保存: {output_path}")

    # Save to paper figure directory as well
    paper_fig_dir = os.path.join(project_root, "paper", "sections", "figures")
    try:
        os.makedirs(paper_fig_dir, exist_ok=True)
        paper_fig_path = os.path.join(paper_fig_dir, "simulation_energy_results.png")
        plt.savefig(paper_fig_path, dpi=300)
        print(f"论文图另存: {paper_fig_path}")
    except Exception as e:
        print(f"论文图保存失败（可忽略）: {e}")

    try:
        plt.close(fig)
    except Exception:
        pass


def plot_energy_history_interactive(energy_history, node_ids, out_dir="sim"):
    """
    Create interactive Plotly figures:
      1) Cluster Heads only
      2) Sensor nodes only

    Saves to HTML under out_dir.
    """
    if not HAS_PLOTLY:
        print("Plotly 未安装，跳过交互图生成。可通过 pip install plotly 安装。")
        return

    os.makedirs(out_dir, exist_ok=True)
    num_steps = len(next(iter(energy_history.values())))
    t_min = _time_axis_minutes(num_steps)

    # 可配置的放大倍数（默认3倍），用于放大图例、刻度及刻度文字大小
    SCALE = getattr(SimConfig, 'PLOT_FONT_SCALE', 3.0)
    TITLE_SIZE = int(16 * SCALE)
    AXIS_TITLE_SIZE = int(14 * SCALE)
    TICK_SIZE = int(12 * SCALE)
    LEGEND_SIZE = int(12 * SCALE)
    WIDTH = 1600
    HEIGHT = 900

    ch_ids = [nid for nid in node_ids if 'CH' in str(nid)]
    sn_ids = [nid for nid in node_ids if 'CH' not in str(nid)]

    # Filter out hidden cluster heads based on config
    hidden_ch_ids = getattr(SimConfig, 'HIDDEN_CH_IDS_IN_PLOT', None)
    if hidden_ch_ids is not None and len(hidden_ch_ids) > 0:
        hidden_set = set(str(hid) for hid in hidden_ch_ids)
        ch_ids = [nid for nid in ch_ids if str(nid) not in hidden_set]
        if SimConfig.ENABLE_LOGGING:
            print(f"[Plot] 隐藏的簇头: {hidden_ch_ids}, 显示的簇头: {ch_ids}")

    # Rough downsampling to keep HTML size manageable (~5000 points per trace)
    stride = max(1, num_steps // 5000)
    t_ds = t_min[::stride]

    # Figure 1: Cluster Heads only
    fig_ch = go.Figure()
    for nid in ch_ids:
        y = np.array(energy_history[nid])[::stride]
        fig_ch.add_trace(go.Scattergl(x=t_ds, y=y, mode='lines', name=str(nid),
                                      line=dict(width=2)))
    fig_ch.update_layout(
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=LEGEND_SIZE)),
        width=WIDTH,
        height=HEIGHT,
        margin=dict(l=int(60*SCALE), r=int(20*SCALE), b=int(60*SCALE), t=int(40*SCALE))
    )
    fig_ch.update_xaxes(title_text='Time (minutes)', title_font=dict(size=AXIS_TITLE_SIZE), tickfont=dict(size=TICK_SIZE), automargin=True, tickwidth=1)
    fig_ch.update_yaxes(title_text='Energy (J)', title_font=dict(size=AXIS_TITLE_SIZE), tickfont=dict(size=TICK_SIZE), automargin=True, ticks='outside', ticklen=int(8*SCALE), ticklabelposition='outside', tickwidth=1)
    ch_path = os.path.join(out_dir, 'energy_ch_only.html')
    fig_ch.write_html(ch_path, include_plotlyjs='cdn')
    print(f"交互图（簇头）: {ch_path}")

    # Figure 2: Sensor nodes only
    fig_sn = go.Figure()
    # Optional: color by cluster id inferred from node_id prefix before '-'
    def cluster_of(nid: str):
        try:
            return str(nid).split('-')[0]
        except Exception:
            return 'N/A'

    for nid in sn_ids:
        y = np.array(energy_history[nid])[::stride]
        fig_sn.add_trace(go.Scattergl(x=t_ds, y=y, mode='lines', name=str(nid),
                                      line=dict(width=1), opacity=0.6,
                                      legendgroup=f"C{cluster_of(nid)}"))
    fig_sn.update_layout(
        template='plotly_white',
        showlegend=False,
        width=WIDTH,
        height=HEIGHT,
        margin=dict(l=int(60*SCALE), r=int(20*SCALE), b=int(60*SCALE), t=int(40*SCALE))
    )
    fig_sn.update_xaxes(title_text='Time (minutes)', title_font=dict(size=AXIS_TITLE_SIZE), tickfont=dict(size=TICK_SIZE), automargin=True, tickwidth=1)
    fig_sn.update_yaxes(title_text='Energy (J)', title_font=dict(size=AXIS_TITLE_SIZE), tickfont=dict(size=TICK_SIZE), automargin=True, ticks='outside', ticklen=int(8*SCALE), ticklabelposition='outside')
    sn_path = os.path.join(out_dir, 'energy_sensors_only.html')
    fig_sn.write_html(sn_path, include_plotlyjs='cdn')
    print(f"交互图（簇成员）: {sn_path}")


def plot_compare_runs(energy_a, energy_b, node_ids, label_a="No RIS", label_b="RIS boost", out_path="sim/ris_boost_comparison.png"):
    """
    对比两组仿真结果（无RIS增益 vs 有RIS增益）。
    输出一张包含三幅子图的PNG：
      1) 全部节点能量总和
      2) 簇头能量总和
      3) 簇成员能量总和
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    num_steps = len(next(iter(energy_a.values())))
    t_min = _time_axis_minutes(num_steps)

    ch_ids = [nid for nid in node_ids if 'CH' in str(nid)]
    sn_ids = [nid for nid in node_ids if 'CH' not in str(nid)]

    def sum_series(keys, energy_dict):
        return np.sum([np.asarray(energy_dict[k]) for k in keys], axis=0) if keys else np.zeros(num_steps)

    total_a = sum_series(node_ids, energy_a)
    total_b = sum_series(node_ids, energy_b)
    ch_a = sum_series(ch_ids, energy_a)
    ch_b = sum_series(ch_ids, energy_b)
    sn_a = sum_series(sn_ids, energy_a)
    sn_b = sum_series(sn_ids, energy_b)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(t_min, total_a, label=label_a, lw=2)
    axes[0].plot(t_min, total_b, label=label_b, lw=2, linestyle='--')
    axes[0].set_title('Total Energy (All Nodes)')
    axes[0].set_ylabel('Energy (J)')
    axes[0].legend()

    axes[1].plot(t_min, ch_a, label=label_a, lw=2)
    axes[1].plot(t_min, ch_b, label=label_b, lw=2, linestyle='--')
    axes[1].set_title('Cluster Heads Total Energy')
    axes[1].set_ylabel('Energy (J)')

    axes[2].plot(t_min, sn_a, label=label_a, lw=2)
    axes[2].plot(t_min, sn_b, label=label_b, lw=2, linestyle='--')
    axes[2].set_title('Sensor Nodes Total Energy')
    axes[2].set_xlabel('Time (minutes)')
    axes[2].set_ylabel('Energy (J)')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"对比图已保存: {out_path}")
    try:
        plt.close(fig)
    except Exception:
        pass

