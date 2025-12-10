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

    # 2. Main simulation loop
    for t_step in range(num_steps):
        current_time = t_step * SimConfig.TIME_STEP_S
        logger.log_step(t_step, current_time)

        # Hourly info upload from sensors to CH (distance-aware energy cost)
        if current_time > 0 and current_time % 3600 == 0:
            hourly_consumption = {}
            for cluster in wsn.clusters:
                ch = cluster.cluster_head
                for sensor in cluster.sensor_nodes:
                    # Config parameters
                    tx_e_base = getattr(SensorNodeConfig, 'TX_ENERGY_J', 0.0)
                    B = getattr(SensorNodeConfig, 'REPORT_PACKET_BITS', 0)
                    E_elec = getattr(SensorNodeConfig, 'REPORT_E_ELEC_J_PER_BIT', 0.0)
                    epsilon = getattr(SensorNodeConfig, 'REPORT_EPSILON_AMP_J_PER_BIT_MTAU', 0.0)
                    tau = getattr(SensorNodeConfig, 'REPORT_PATH_LOSS_EXPONENT', 2.0)
                    include_rx = getattr(SensorNodeConfig, 'REPORT_INCLUDE_CH_RX', False)

                    # Distance between sensor and CH
                    d = sensor.distance_to(ch)
                    # TX energy for this report (electronics + amplifier term) + base overhead
                    E_tx_dist = E_elec * B + epsilon * B * (d ** tau)
                    E_total_sensor = tx_e_base + E_tx_dist

                    # Deduct from sensor (clamped by available energy)
                    e_s = min(E_total_sensor, sensor.current_energy)
                    if e_s > 0:
                        sensor.current_energy -= e_s
                        hourly_consumption[sensor.node_id] = e_s

                    # Optional: CH receive energy (electronics only)
                    if include_rx and hasattr(ch, 'current_energy'):
                        E_rx = E_elec * B
                        ch.current_energy = max(0.0, ch.current_energy - min(E_rx, ch.current_energy))

            # Log the information upload energy costs
            if hourly_consumption:
                logger.log_energy_transfer(
                    rf_target=None,
                    rf_sent_energy_j=None,
                    rf_delivered_energy_j=None,
                    mrc_entries=[],
                    sensor_tx_consumption=hourly_consumption
                )



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




        # c. Intra-cluster energy convergence via MRC (Sensor -> CH)
        abundant_threshold = getattr(SensorNodeConfig, 'ABUNDANT_THRESHOLD_J', 0.0)
        for cluster in wsn.clusters:
            ch = cluster.cluster_head
            for sensor in cluster.sensor_nodes:
                # Only transfer if sensor energy is above the abundant threshold
                if sensor.current_energy > abundant_threshold:
                    # Per-step intended send capped by surplus above threshold
                    intended_send_j = sensor.mrc_tx_power_w * SimConfig.TIME_STEP_S
                    surplus_j = sensor.current_energy - abundant_threshold
                    energy_to_send_j = min(intended_send_j, surplus_j)
                    if energy_to_send_j > 0:
                        # 1. Deduct energy from the sensor
                        sensor.current_energy -= energy_to_send_j
                        sensor.record_transfer(transferred=energy_to_send_j)

                        # 2. Calculate received power at the cluster head using the actual TX power
                        actual_tx_power_w = energy_to_send_j / SimConfig.TIME_STEP_S
                        received_power_w = mrc_model.calculate_received_mrc_power(sensor, ch, tx_power_w=actual_tx_power_w)
                        
                        # 3. Add energy to the cluster head
                        ch.receive_mrc_power(received_power_w, SimConfig.TIME_STEP_S)

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

