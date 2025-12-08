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
    # 阻塞显示，避免窗口一闪而过（若在无图形环境可注释）
    plt.show(block=True)

