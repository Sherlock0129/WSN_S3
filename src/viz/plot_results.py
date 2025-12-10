"""
Visualization script for plotting simulation results.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from src.config.simulation_config import SimConfig

def plot_energy_history(energy_history, node_ids):
    """
    Plots the energy level of each node over time.

    Args:
        energy_history (dict): A dictionary where keys are node_ids and values
                               are numpy arrays of energy levels over time.
        node_ids (list): A list of all node IDs.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    num_steps = len(next(iter(energy_history.values())))
    time_axis = np.arange(num_steps) * SimConfig.TIME_STEP_S / 60 # Time in minutes

    # Separate Cluster Heads and Sensor Nodes for different line styles
    ch_ids = [nid for nid in node_ids if 'CH' in str(nid)]
    sn_ids = [nid for nid in node_ids if 'CH' not in str(nid)]

    # Plot sensor nodes
    for node_id in sn_ids:
        ax.plot(time_axis, energy_history[node_id], lw=1.5, alpha=0.7, label=f'Node {node_id}')

    # Plot cluster heads with thicker lines
    for node_id in ch_ids:
        ax.plot(time_axis, energy_history[node_id], lw=3, linestyle='--', label=f'CH {node_id}')

    ax.set_xlabel("Time (minutes)", fontsize=14)
    ax.set_ylabel("Energy (Joules)", fontsize=14)
    ax.set_title("Node Energy Levels Over Time", fontsize=16)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
    
    # Save the figure到项目根目录，便于查找
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_path = os.path.join(project_root, "simulation_energy_results.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n结果图已保存: {output_path}")

    # 另存到论文图目录，便于直接 \includegraphics 引用
    paper_fig_dir = os.path.join(project_root, "paper", "sections", "figures")
    try:
        os.makedirs(paper_fig_dir, exist_ok=True)
        paper_fig_path = os.path.join(paper_fig_dir, "simulation_energy_results.png")
        plt.savefig(paper_fig_path, dpi=300)
        print(f"论文图另存: {paper_fig_path}")
    except Exception as e:
        print(f"论文图保存失败（可忽略）: {e}")

    # 非交互环境下不显示窗口，直接关闭释放内存
    try:
        plt.close(fig)
    except Exception:
        pass

