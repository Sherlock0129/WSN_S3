"""
Cross-layer scheduler for the hierarchical WPT system.

This module decides which cluster head to charge via the RF-RIS system and
which cluster heads should perform local MRC power transmission.
"""

from src.config.simulation_config import ClusterHeadConfig, WSNConfig, SimConfig


def schedule_power_transfer(wsn):
    """
    A simple energy-aware scheduler.

    Policy:
    1) Pick the cluster head with the lowest absolute energy as rf_target
       (main loop may ignore RF charging).
    2) Any cluster head with energy > threshold (DOWNLINK_CH_MIN_PCT) will perform
       local MRC power transmission to its own sensor nodes (if enabled).
       Note: rf_target is no longer excluded from MRC transmission.

    Args:
        wsn (WSN): The main WSN object.

    Returns:
        dict: {
            'rf_target': cluster_head_object or None,
            'mrc_transmitters': [ch1, ch2, ...]
        }
    """
    if not wsn.clusters:
        return {'rf_target': None, 'mrc_transmitters': []}

    # 1) Find the cluster head with the lowest energy (absolute J)
    lowest_energy_ch = None
    min_energy_j = float('inf')
    for cluster in wsn.clusters:
        ch = cluster.cluster_head
        if ch.current_energy < min_energy_j:
            min_energy_j = ch.current_energy
            lowest_energy_ch = ch

    # 2) Designate it as the RF target (may be ignored by main loop's donation mode)
    rf_target = lowest_energy_ch

    # 3) Select CHs for local MRC transmission (if enabled)
    mrc_transmitters = []
    if WSNConfig.ENABLE_MRC_LOCAL_TRANSFER:
        # Trigger threshold from config: only stop when CH energy < DOWNLINK_CH_MIN_PCT
        pct = max(0.0, min(1.0, getattr(SimConfig, 'DOWNLINK_CH_MIN_PCT', 0.10)))
        mrc_threshold = ClusterHeadConfig.INITIAL_ENERGY_J * pct
        for cluster in wsn.clusters:
            ch = cluster.cluster_head
            # 按阴/阳面与方向控制簇内下行（CH->成员）
            is_solar = getattr(cluster, 'has_solar_nodes', False)
            if is_solar and not getattr(SimConfig, 'ENABLE_DOWNLINK_SOLAR', True):
                continue
            if (not is_solar) and not getattr(SimConfig, 'ENABLE_DOWNLINK_NON_SOLAR', True):
                continue
            # 只有当簇头能量高于阈值才参与下发
            if ch.current_energy > mrc_threshold:
                mrc_transmitters.append(ch)

    return {
        'rf_target': rf_target,
        'mrc_transmitters': mrc_transmitters,
    }
