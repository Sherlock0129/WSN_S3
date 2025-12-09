"""
Multi-RIS Energy Routing Algorithm.

This module finds the optimal energy path from the RF transmitter to a target
cluster head, potentially via one or more RIS panels.
"""

import numpy as np
import itertools

from src.utils import rf_propagation_model

def find_optimal_energy_path(wsn, source, target_ch, max_hops=2):
    """
    Finds the optimal energy path to a target cluster head.

    The algorithm explores paths with different numbers of RIS hops:
    - 0 hops: Direct path from RF Transmitter to Cluster Head.
    - 1 hop: RF_Tx -> RIS -> CH
    - 2 hops: RF_Tx -> RIS_i -> RIS_j -> CH

    Args:
        wsn (WSN): The main WSN object containing all components.
        target_ch (ClusterHead): The target cluster head to receive energy.
        max_hops (int): The maximum number of RIS panels to use in a path.

    Returns:
        tuple: A tuple containing:
            - list: The best path found (a list of objects, e.g., [tx, ris1, ch]).
            - float: The maximum power in Watts delivered via that path.
    """
    env = wsn.environment
    ris_panels = wsn.ris_panels

    best_path = []
    max_power = 0.0

    # --- Path 1: Direct Transmission (0 hops) ---
    direct_power = rf_propagation_model.calculate_received_rf_power(source, target_ch, env)
    if direct_power > max_power:
        max_power = direct_power
        best_path = [source, target_ch]

    # --- Path 2: Single RIS Hop (1 hop) ---
    if max_hops >= 1:
        for ris in ris_panels:
            power = rf_propagation_model.calculate_ris_assisted_power(source, ris, target_ch, env)
            if power > max_power:
                max_power = power
                best_path = [source, ris, target_ch]

    # --- Path 3: Double RIS Hop (2 hops) ---
    if max_hops >= 2 and len(ris_panels) >= 2:
        # Iterate through all permutations of 2 RIS panels
        for ris_i, ris_j in itertools.permutations(ris_panels, 2):
            # Power from source to RIS_j via RIS_i
            power_at_ris_j = rf_propagation_model.calculate_ris_assisted_power(source, ris_i, ris_j, env)
            
            if power_at_ris_j > 0:
                # Treat RIS_j as a new source for the final hop approximation
                ris_j_as_source = type('RISSource', (), {})
                ris_j_as_source.position = ris_j.position
                ris_j_as_source.get_tx_power_dbm = lambda: 10 * np.log10(power_at_ris_j * 1000)
                ris_j_as_source.get_reflection_gain = ris_j.get_reflection_gain
                # Use source's frequency for the cascaded hop approximation
                dist_j_ch = np.linalg.norm(ris_j.position - target_ch.position)
                if env.check_los(ris_j.position, target_ch.position):
                    final_power_dbm = rf_propagation_model._log_distance_path_loss(
                        ris_j_as_source.get_tx_power_dbm(),
                        ris_j.get_reflection_gain(),
                        target_ch.rf_rx_gain_dbi,
                        source.frequency_hz,
                        dist_j_ch,
                        True  # 已通过 RIS2→CH 的 LoS 检查
                    )
                    final_power_w = 10**((final_power_dbm - 30) / 10)

                    if final_power_w > max_power:
                        max_power = final_power_w
                        best_path = [source, ris_i, ris_j, target_ch]

    return best_path, max_power

