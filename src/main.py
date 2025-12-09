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
from src.config.simulation_config import SimConfig, SensorNodeConfig
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
    # 2. Main simulation loop
    for t_step in range(num_steps):
        current_time = t_step * SimConfig.TIME_STEP_S
        logger.log_step(t_step, current_time)

        # a. Get scheduling decision
        actions = scheduler.schedule_power_transfer(wsn)
        rf_target_ch = actions['rf_target']
        mrc_transmitting_chs = actions['mrc_transmitters']
        logger.log_scheduling(actions)

        # b. Perform long-range RF power transfer
        best_path, max_power_w = [], 0.0
        if rf_target_ch:
            best_path, max_power_w = routing_algorithm.find_optimal_energy_path(wsn, rf_target_ch)
            logger.log_routing(best_path, max_power_w)
            
            if max_power_w > 0:
                # c. Update target CH energy
                rf_target_ch.receive_rf_power(max_power_w, SimConfig.TIME_STEP_S)

        # d. Log energy flow
        logger.log_energy_transfer(rf_target_ch, max_power_w, mrc_transmitting_chs, SimConfig.TIME_STEP_S)

        # e. Perform local MRC power transfer
        for ch in mrc_transmitting_chs:
            # The CH transmits power to its own sensor nodes
            target_nodes = [c.sensor_nodes for c in wsn.clusters if c.cluster_head == ch][0]
            ch.transmit_mrc_power(target_nodes, SimConfig.TIME_STEP_S, mrc_model)

        # f. Update energy for all nodes due to idle consumption
        for node in all_nodes:
            idle_consumption = SensorNodeConfig.IDLE_POWER_W * SimConfig.TIME_STEP_S
            node.current_energy -= idle_consumption
            # Ensure energy doesn't go below zero
            node.current_energy = max(0, node.current_energy)

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

