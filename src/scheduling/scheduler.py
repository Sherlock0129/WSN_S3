"""
Cross-layer scheduler for the hierarchical WPT system.

This module decides which cluster head to charge via the RF-RIS system and
which cluster heads should perform local MRC power transmission.
"""

from src.config.simulation_config import ClusterHeadConfig, WSNConfig

def schedule_power_transfer(wsn):
    """
    A simple energy-aware scheduler.

    The policy is as follows:
    1. Find the cluster head with the lowest energy percentage.
    2. Designate this cluster head as the target for the long-range RF-RIS system.
    3. Any cluster head with energy above a certain threshold will perform local
       MRC power transmission to its sensor nodes.

    Args:
        wsn (WSN): The main WSN object.

    Returns:
        dict: A dictionary of scheduled actions, e.g.,
              {
                  'rf_target': cluster_head_object,
                  'mrc_transmitters': [ch1, ch2, ...]
              }
    """
    if not wsn.clusters:
        return {'rf_target': None, 'mrc_transmitters': []}

    # 1. Find the cluster head with the lowest energy
    lowest_energy_ch = None
    min_energy_j = float('inf')

    for cluster in wsn.clusters:
        ch = cluster.cluster_head
        if ch.current_energy < min_energy_j:
            min_energy_j = ch.current_energy
            lowest_energy_ch = ch

    # 2. Designate it as the RF target
    rf_target = lowest_energy_ch

    # 3. Find cluster heads that should perform local MRC transmission (guarded by config)
    mrc_transmitters = []
    if WSNConfig.ENABLE_MRC_LOCAL_TRANSFER:
        # A CH will transmit locally if its energy is above 50% of its initial capacity
        mrc_threshold = ClusterHeadConfig.INITIAL_ENERGY_J * 0.5
        for cluster in wsn.clusters:
            ch = cluster.cluster_head
            # 避免被远程充电的目标CH在同一步执行MRC
            if ch is rf_target:
                continue
            if ch.current_energy > mrc_threshold:
                mrc_transmitters.append(ch)

    actions = {
        'rf_target': rf_target,
        'mrc_transmitters': mrc_transmitters
    }

    return actions

