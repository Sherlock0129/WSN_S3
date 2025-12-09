"""
Main simulation loop for the Hierarchical WPT System.
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np

from src.network.WSN import WSN
from src.routing import routing_algorithm
from src.scheduling import scheduler
from src.utils import mrc_model
from src.utils.simulation_logger import SimulationLogger
from src.config.simulation_config import SimConfig, SensorNodeConfig, WSNConfig
from src.viz.plot_results import plot_energy_history

def run_simulation():
    """
    Initializes and runs the main simulation loop.
    """
    # 1. Initialize the WSN and Logger
    wsn = WSN()
    logger = SimulationLogger()
    
    # Data storage for results
    # We'll store the energy level of each node at each time step
    all_nodes = wsn.get_all_nodes()
    node_ids = [node.node_id for node in all_nodes]
    num_steps = int(SimConfig.SIMULATION_TIME_S / SimConfig.TIME_STEP_S)
    energy_history = {node_id: np.zeros(num_steps) for node_id in node_ids}

    print("\nStarting simulation...")

    # Cross-cluster donation state
    donation_state = {
        'donor': None,      # ClusterHead acting as RF source
        'receiver': None,   # ClusterHead to receive energy
        'next_eval_step': 0
    }
    # Precompute baseline cluster energies
    cluster_baselines = {}
    for cluster in wsn.clusters:
        baseline = (
            # CH initial energy
            1.0 * cluster.cluster_head.initial_energy +
            # Sum of sensor initial energies (assume uniform config value)
            len(cluster.sensor_nodes) * SensorNodeConfig.INITIAL_ENERGY_J
        )
        cluster_baselines[cluster.cluster_id] = max(baseline, 1e-9)

    # 2. Main simulation loop
    for t_step in range(num_steps):
        current_time = t_step * SimConfig.TIME_STEP_S
        logger.log_step(t_step, current_time)

        # a. Optional: baseline scheduler for MRC only (RF target disabled in donation mode)
        actions = scheduler.schedule_power_transfer(wsn)
        rf_target_ch = None  # 禁用主站RF充电
        mrc_transmitting_chs = actions['mrc_transmitters']

        # Cross-cluster trigger evaluation
        if SimConfig.ENABLE_CROSS_CLUSTER_DONATION and t_step >= donation_state['next_eval_step']:
            # Compute cluster energy percentage
            cluster_pct = {}
            for cluster in wsn.clusters:
                ch = cluster.cluster_head
                total_e = ch.current_energy + sum(n.current_energy for n in cluster.sensor_nodes)
                pct = total_e / cluster_baselines[cluster.cluster_id]
                cluster_pct[cluster.cluster_id] = pct
            # Select receiver (lowest pct below threshold)
            receiver_cluster = min(wsn.clusters, key=lambda c: cluster_pct[c.cluster_id])
            if cluster_pct[receiver_cluster.cluster_id] < SimConfig.TRIGGER_LOW_PCT:
                # Select donors: exclude receiver; require high pct
                donors = [c for c in wsn.clusters if c is not receiver_cluster and cluster_pct[c.cluster_id] > SimConfig.TRIGGER_HIGH_PCT]
                # Score donors: p/distance to receiver CH
                if donors:
                    rx_ch = receiver_cluster.cluster_head
                    def score(c):
                        d = np.linalg.norm(c.cluster_head.position - rx_ch.position)
                        d = max(d, 1.0)
                        return cluster_pct[c.cluster_id] / d
                    donor_cluster = max(donors, key=score)
                    donation_state['donor'] = donor_cluster.cluster_head
                    donation_state['receiver'] = receiver_cluster.cluster_head
                else:
                    donation_state['donor'] = None
                    donation_state['receiver'] = None
            else:
                donation_state['donor'] = None
                donation_state['receiver'] = None
            donation_state['next_eval_step'] = t_step + SimConfig.CROSS_CLUSTER_TRIGGER_PERIOD_STEPS

        # If we have an active donor/receiver pair, perform donation this step
        best_path, max_power_w = [], 0.0
        rf_sent_energy_j, rf_delivered_energy_j = None, 0.0
        if donation_state['donor'] is not None and donation_state['receiver'] is not None:
            donor_ch = donation_state['donor']
            recv_ch = donation_state['receiver']
            # If donor lacks energy for TX this step, skip
            energy_needed = donor_ch.rf_tx_power_w * SimConfig.TIME_STEP_S
            if donor_ch.current_energy >= energy_needed:
                best_path, max_power_w = routing_algorithm.find_optimal_energy_path(wsn, donor_ch, recv_ch)
                logger.log_routing(best_path, max_power_w)
                rf_sent_energy_j = energy_needed
                # Deduct donor energy (TX cost)
                donor_ch.current_energy -= energy_needed
                if max_power_w > 0:
                    rf_delivered_energy_j = max_power_w * SimConfig.TIME_STEP_S
                    recv_ch.receive_rf_power(max_power_w, SimConfig.TIME_STEP_S)
            else:
                # Not enough donor energy; skip this step
                pass

        # e. Perform local MRC power transfer and collect per-CH send/deliver
        # 排除 donor CH 避免同时MRC
        mrc_entries = []
        for ch in mrc_transmitting_chs:
            if donation_state['donor'] is not None and ch is donation_state['donor']:
                continue
            # The CH transmits power to its own sensor nodes
            target_nodes = [c.sensor_nodes for c in wsn.clusters if c.cluster_head == ch][0]
            delivered_j, sent_j = ch.transmit_mrc_power(target_nodes, SimConfig.TIME_STEP_S, mrc_model)
            mrc_entries.append({
                'ch_id': ch.node_id,
                'sent_j': sent_j,
                'delivered_j': delivered_j,
            })

        # d. Log energy flow (RF + MRC)
        logger.log_scheduling({'rf_target': donation_state['receiver'], 'mrc_transmitters': mrc_transmitting_chs})
        logger.log_routing(best_path, max_power_w)
        logger.log_energy_transfer(donation_state['receiver'], rf_sent_energy_j, rf_delivered_energy_j, mrc_entries)

        # f. Per-node energy update (solar harvest + idle decay via update_energy)
        current_time_min = (current_time % (24 * 3600)) / 60.0  # minutes in a day
        for node in all_nodes:
            # Respect global solar enable: disable harvesting when turned off
            if hasattr(node, 'enable_energy_harvesting'):
                node.enable_energy_harvesting = getattr(node, 'has_solar', False) and WSNConfig.ENABLE_SOLAR
            node.update_energy(current_time_min)

        # g. Record energy levels for plotting
        for i, node in enumerate(all_nodes):
            energy_history[node.node_id][t_step] = node.current_energy
            # Check if a node has died
            if node.current_energy < SensorNodeConfig.MIN_ENERGY_J:
                print(f"!!! Node {node.node_id} has died at {current_time}s !!!")
                # For now, we just print. We could also stop the simulation.

        # h. Log cluster energy status
        logger.log_cluster_energy(wsn)

        # i. Print progress
        if (t_step + 1) % 100 == 0:
            print(f"... Step {t_step + 1}/{num_steps} completed.")

    print("Simulation finished.")
    logger.close()
    return energy_history, node_ids

if __name__ == "__main__":
    energy_data, node_ids = run_simulation()
    
    # Print final energy status
    print("\n--- Final Energy Status ---")
    for node_id in node_ids:
        final_energy = energy_data[node_id][-1]
        print(f"Node {node_id}: {final_energy:.4f} J")

    # 4. Plot the results (if enabled)
    if SimConfig.ENABLE_PLOT_RESULTS:
        plot_energy_history(energy_data, node_ids)

