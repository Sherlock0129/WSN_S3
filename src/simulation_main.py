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
from src.config.simulation_config import SimConfig, SensorNodeConfig, WSNConfig, ClusterHeadConfig, RFLinkConfig, MRCConfig
from src.viz.plot_results import plot_energy_history, plot_energy_history_interactive, plot_compare_runs
from src.viz.plot_energy_heatmap import plot_energy_heatmap

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

        # A) 调度（当前仅用于日志与避免目标CH做MRC）
        sched_actions = scheduler.schedule_power_transfer(wsn)
        if SimConfig.ENABLE_LOGGING:
            logger.log_scheduling(sched_actions)

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
                    mult = getattr(SensorNodeConfig, 'REPORT_ENERGY_MULTIPLIER', 1.0)
                    E_total_sensor = (tx_e_base + E_tx_dist) * mult

                    # Deduct from sensor (clamped by available energy)
                    e_s = min(E_total_sensor, sensor.current_energy)
                    if e_s > 0:
                        sensor.current_energy -= e_s
                        hourly_consumption[sensor.node_id] = e_s

                    # Optional: CH receive energy (electronics only)
                    if include_rx and hasattr(ch, 'current_energy'):
                        mult = getattr(SensorNodeConfig, 'REPORT_ENERGY_MULTIPLIER', 1.0)
                        E_rx = (E_elec * B) * mult
                        ch.current_energy = max(0.0, ch.current_energy - min(E_rx, ch.current_energy))

            # Log the information upload energy costs
            if hourly_consumption and SimConfig.ENABLE_LOGGING:
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
        if getattr(SimConfig, 'ENABLE_SENSOR_TO_CH_CONVERGENCE', True):
            abundant_threshold = getattr(SensorNodeConfig, 'ABUNDANT_THRESHOLD_J', 0.0)
            for cluster in wsn.clusters:
                    # 按阴/阳面与方向控制簇内上行（成员->CH）
                    is_solar = getattr(cluster, 'has_solar_nodes', False)
                    allow_uplink = (is_solar and getattr(SimConfig, 'ENABLE_UPLINK_SOLAR', True)) or \
                                   ((not is_solar) and getattr(SimConfig, 'ENABLE_UPLINK_NON_SOLAR', True))
                    if not allow_uplink:
                        continue
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

        # c+1. Regular MRC downlink: CH maintains sensor energy above threshold
        if WSNConfig.ENABLE_MRC_LOCAL_TRANSFER:
            sensor_min_pct = getattr(SimConfig, 'SENSOR_MIN_ENERGY_PCT', 0.80)
            sensor_min_energy_j = SensorNodeConfig.MAX_ENERGY_J * sensor_min_pct
            ch_min_pct = getattr(SimConfig, 'DOWNLINK_CH_MIN_PCT', 0.10)
            ch_min_energy_j = ClusterHeadConfig.INITIAL_ENERGY_J * ch_min_pct
            
            for cluster in wsn.clusters:
                ch = cluster.cluster_head
                # 按阴/阳面与方向控制簇内下行（CH->成员）
                is_solar = getattr(cluster, 'has_solar_nodes', False)
                if is_solar and not getattr(SimConfig, 'ENABLE_DOWNLINK_SOLAR', True):
                    continue
                if (not is_solar) and not getattr(SimConfig, 'ENABLE_DOWNLINK_NON_SOLAR', True):
                    continue
                # 只有当簇头能量高于阈值才参与下发
                if ch.current_energy <= ch_min_energy_j:
                    continue
                
                # 找出能量低于80%阈值的簇成员
                needy = [n for n in cluster.sensor_nodes if n.current_energy < sensor_min_energy_j]
                if needy:
                    dt = SimConfig.TIME_STEP_S
                    delivered_j, sent_j = ch.transmit_mrc_power(needy, dt, mrc_model)
                    if SimConfig.ENABLE_LOGGING and delivered_j > 0:
                        logger.log_energy_transfer(None, None, None,
                            mrc_entries=[{'ch_id': ch.node_id, 'sent_j': sent_j, 'delivered_j': delivered_j}])

        # c+2. Overflow-based donation: if a CH exceeds its battery cap, forward overflow cross-cluster now
        if getattr(SimConfig, 'ENABLE_ROUTING', True) and getattr(SimConfig, 'ENABLE_CROSS_CLUSTER_DONATION', True):
            for cluster in wsn.clusters:
                ch = cluster.cluster_head
                cap = getattr(ch, 'max_energy_j', ClusterHeadConfig.INITIAL_ENERGY_J)
                overflow = ch.current_energy - cap
                if overflow > 1e-9 and ch.rf_tx_power_w > 0:
                    # find recipient: prefer non-solar clusters with room and lowest energy
                    candidates = []
                    for cl2 in wsn.clusters:
                        ch2 = cl2.cluster_head
                        if cl2 is cluster:
                            continue
                        # only accept if has room below its cap
                        cap2 = getattr(ch2, 'max_energy_j', ClusterHeadConfig.INITIAL_ENERGY_J)
                        room = cap2 - ch2.current_energy
                        if room <= 1e-9:
                            continue
                        # prioritize non-solar recipients or low energy
                        priority = (0 if not getattr(cl2, 'has_solar_nodes', False) else 1, ch2.current_energy)
                        candidates.append((priority, cl2, ch2, room))
                    if candidates:
                        # pick best candidate
                        _, target_cluster, target_ch, room = sorted(candidates, key=lambda x: x[0])[0]
                        # route and send up to TX power cap and target room
                        dt = SimConfig.TIME_STEP_S
                        e_tx_cap = ch.rf_tx_power_w * dt
                        e_tx = max(0.0, min(overflow, e_tx_cap, room / 1e-12))  # room/eta upper bound approximated; refine below
                        # compute path & efficiency
                        path, p_deliver_full = routing_algorithm.find_optimal_energy_path(wsn, source=ch, target_ch=target_ch, max_hops=2)
                        if p_deliver_full > 0.0 and e_tx > 0.0:
                            eta = p_deliver_full / max(ch.rf_tx_power_w, 1e-12)
                            # enforce minimum efficiency if RIS-assisted
                            try:
                                if any(hasattr(p, 'panel_id') for p in path):
                                    eta = max(eta, getattr(RFLinkConfig, 'RIS_ASSISTED_MIN_EFFICIENCY', eta))
                            except Exception:
                                pass
                            # correct e_tx by recipient room constraint under final eta
                            if eta > 0:
                                e_tx = min(e_tx, room / eta)
                            e_rx = eta * e_tx
                            # settle energies
                            ch.current_energy = max(0.0, ch.current_energy - e_tx)
                            ch.record_transfer(transferred=e_tx)
                            target_ch.current_energy += e_rx
                            target_ch.record_transfer(received=e_rx)
                            # clamp donor to cap to avoid drift
                            ch.current_energy = min(ch.current_energy, cap)
                            if SimConfig.ENABLE_LOGGING:
                                logger.log_routing(path, p_deliver_full)
                                logger.log_energy_transfer(target_ch, rf_sent_energy_j=e_tx, rf_delivered_energy_j=e_rx, mrc_entries=[])

        # d. Cross-cluster RF donation (CH -> CH) + target MRC downlink (CH -> sensors)
        if SimConfig.ENABLE_CROSS_CLUSTER_DONATION and (t_step % int(SimConfig.CROSS_CLUSTER_TRIGGER_PERIOD_STEPS) == 0):
            donors = []
            candidates = []
            for cluster in wsn.clusters:
                ch = cluster.cluster_head
                init = ClusterHeadConfig.INITIAL_ENERGY_J
                pct = ch.current_energy / max(init, 1e-9)
                if getattr(cluster, 'has_solar_nodes', False) and pct >= SimConfig.TRIGGER_HIGH_PCT:
                    donors.append((cluster, ch, pct))
                else:
                    # treat as potential recipient if non-solar, or any low-energy cluster
                    if (not getattr(cluster, 'has_solar_nodes', False)) or (pct <= SimConfig.TRIGGER_LOW_PCT):
                        candidates.append((cluster, ch, pct))

            if donors and candidates:
                # choose the lowest-energy candidate as target, highest-energy donor as source
                target_cluster, target_ch, _ = min(candidates, key=lambda x: x[2])
                donor_cluster, donor_ch, _ = max(donors, key=lambda x: x[2])

                # compute best RF/RIS path from donor CH to target CH
                path, p_deliver_full = routing_algorithm.find_optimal_energy_path(
                    wsn, source=donor_ch, target_ch=target_ch, max_hops=2
                )
                if p_deliver_full > 0.0:
                    dt = SimConfig.TIME_STEP_S
                    high_j = ClusterHeadConfig.INITIAL_ENERGY_J * SimConfig.TRIGGER_HIGH_PCT
                    e_tx_cap = donor_ch.rf_tx_power_w * dt
                    e_tx = max(0.0, min(e_tx_cap, donor_ch.current_energy - high_j))
                    if e_tx > 0.0:
                        eta = p_deliver_full / max(donor_ch.rf_tx_power_w, 1e-12)
                        # enforce minimum efficiency if RIS-assisted
                        try:
                            if any(hasattr(p, 'panel_id') for p in path):
                                eta = max(eta, getattr(RFLinkConfig, 'RIS_ASSISTED_MIN_EFFICIENCY', eta))
                        except Exception:
                            pass
                        e_rx = eta * e_tx

                        donor_ch.current_energy = max(0.0, donor_ch.current_energy - e_tx)
                        donor_ch.record_transfer(transferred=e_tx)
                        target_ch.current_energy += e_rx
                        target_ch.record_transfer(received=e_rx)

                        if SimConfig.ENABLE_LOGGING:
                            logger.log_routing(path, p_deliver_full)
                            logger.log_energy_transfer(target_ch, rf_sent_energy_j=e_tx, rf_delivered_energy_j=e_rx, mrc_entries=[])

                        # target downlink via MRC to low-energy members
                        needy = [n for n in target_cluster.sensor_nodes if n.current_energy < SensorNodeConfig.ABUNDANT_THRESHOLD_J]
                        if needy:
                            delivered_j, sent_j = target_ch.transmit_mrc_power(needy, dt, mrc_model)
                            if SimConfig.ENABLE_LOGGING:
                                logger.log_energy_transfer(None, None, None,
                                    mrc_entries=[{'ch_id': target_ch.node_id, 'sent_j': sent_j, 'delivered_j': delivered_j}])

        # h. Log cluster energy status
        logger.log_cluster_energy(wsn)

        # i. Print progress
        if (t_step + 1) % 100 == 0:
            print(f"... Step {t_step + 1}/{num_steps} completed.")

    print("Simulation finished.")
    logger.close()
    # Collect final positions
    node_positions = {node.node_id: node.position for node in wsn.get_all_nodes()}
    # Collect cluster head positions
    ch_positions = [cluster.cluster_head.position for cluster in wsn.clusters]
    # Collect RIS positions
    ris_positions = [ris.position for ris in wsn.ris_panels]
    return energy_history, node_ids, node_positions, wsn.environment, ch_positions, ris_positions

def run_comparison_and_plot():
    """
    运行两次仿真：
      - 基线：关闭簇内RIS增益（INTRA_CLUSTER_RIS_CLUSTERS=[]）
      - 增强：使用当前配置中的簇内RIS列表
    并输出对比图 sim/ris_boost_comparison.png
    """
    # 保存原配置
    orig_list = list(getattr(WSNConfig, 'INTRA_CLUSTER_RIS_CLUSTERS', []))

    # 基线：无RIS增益
    WSNConfig.INTRA_CLUSTER_RIS_CLUSTERS = []
    print("\n=== Running baseline (No intra-cluster RIS boost) ===")
    energy_base, node_ids, _, _, _, _ = run_simulation()

    # 增强：恢复原配置
    WSNConfig.INTRA_CLUSTER_RIS_CLUSTERS = orig_list
    print("\n=== Running with intra-cluster RIS boost ===")
    energy_boost, _, _, _, _, _ = run_simulation()

    # 对比图
    boost_label = f"RIS boost x{getattr(MRCConfig, 'INTRA_CLUSTER_RIS_EFF_BOOST', 1.0):.2f}"
    out_path = "sim/ris_boost_comparison.png"
    plot_compare_runs(energy_base, energy_boost, node_ids, label_a="No intra-RIS", label_b=boost_label, out_path=out_path)


if __name__ == "__main__":
    if getattr(SimConfig, 'ENABLE_RIS_COMPARISON_PLOT', False):
        run_comparison_and_plot()
    else:
        energy_data, node_ids, node_positions, environment, ch_positions, ris_positions = run_simulation()
    
    # Print final energy status
    print("\n--- Final Energy Status ---")
    for node_id in node_ids:
        final_energy = energy_data[node_id][-1]
        print(f"Node {node_id}: {final_energy:.4f} J")

    # 4. Plot the results (if enabled)
    if SimConfig.ENABLE_PLOT_RESULTS:
        plot_energy_history(energy_data, node_ids)
        # Interactive figures: CH-only and Sensors-only
        try:
            plot_energy_history_interactive(energy_data, node_ids, out_dir="sim")
        except Exception as e:
            print(f"Interactive plot generation failed: {e}")
            # Spatial energy heatmap (final step)
            try:
                plot_energy_heatmap(node_positions, energy_data, node_ids, step=-1, 
                                  environment=environment,
                                  cluster_head_positions=ch_positions,
                                  ris_positions=ris_positions)
            except Exception as e:
                print(f"Energy heatmap generation failed: {e}")

