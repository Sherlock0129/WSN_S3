"""
Physics model for Magnetic Resonant Coupling (MRC) near-field power transfer.
"""

import numpy as np

from src.config.simulation_config import ClusterHeadConfig, MRCConfig

def calculate_received_mrc_power(tx_node, rx_node, tx_power_w=None):
    """
    Calculates the received power via MRC between any two nodes.
    This is a simplified model based on an inverse power law, suitable for system-level simulation.

    Args:
        tx_node (SensorNode or ClusterHead): The transmitting node.
        rx_node (SensorNode or ClusterHead): The receiving node.
        tx_power_w (float, optional): Override transmit power (W). If None, use tx_node.mrc_tx_power_w.

    Returns:
        float: Received power in Watts.
    """
    distance = np.linalg.norm(tx_node.position - rx_node.position)

    if distance <= 0:
        return 0.0

    # Intra-cluster: efficiency varies with distance within cluster radius [eff_min, eff_max]
    R = float(getattr(MRCConfig, 'INTRA_CLUSTER_FIXED_RADIUS_M', 0.0))
    eff_min = float(getattr(MRCConfig, 'INTRA_CLUSTER_EFF_MIN', 0.5))
    eff_max = float(getattr(MRCConfig, 'INTRA_CLUSTER_EFF_MAX', 0.8))

    if distance <= R and R > 0:
        # Linear interpolation: d=0 -> eff_max, d=R -> eff_min
        t = max(0.0, min(1.0, distance / R))
        efficiency = eff_max - (eff_max - eff_min) * t
    else:
        # Fallback: near-field inverse power law model
        ref_distance = 1.0  # meters
        ref_efficiency = eff_min  # be conservative beyond cluster radius
        decay_exponent = 3.0
        efficiency = ref_efficiency * (ref_distance / distance) ** decay_exponent
        efficiency = max(0.0, min(1.0, efficiency))

    # Apply intra-cluster RIS efficiency boost if either endpoint is in a boosted cluster
    try:
        if getattr(tx_node, 'intra_cluster_ris', False) or getattr(rx_node, 'intra_cluster_ris', False):
            boost = float(getattr(MRCConfig, 'INTRA_CLUSTER_RIS_EFF_BOOST', 1.0))
            efficiency = min(1.0, efficiency * max(1.0, boost))
    except Exception:
        pass

    # Determine TX power to use
    tx_power = tx_power_w if tx_power_w is not None else getattr(tx_node, 'mrc_tx_power_w', 0.0)

    # Received power is the transmitter's power multiplied by the efficiency
    received_power_w = tx_power * efficiency

    return received_power_w

