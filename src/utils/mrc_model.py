"""
Physics model for Magnetic Resonant Coupling (MRC) near-field power transfer.
"""

import numpy as np

from src.config.simulation_config import ClusterHeadConfig

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

    # Near-field power drops off very quickly with distance. We use a simple model.
    # Let's assume a certain efficiency at a reference distance (e.g., 1 meter)
    # and a rapid decay exponent.
    ref_distance = 1.0  # meters
    ref_efficiency = 0.8  # 80% efficiency at 1m
    decay_exponent = 3.0  # Power decays with d^3 or faster in some models

    if distance <= 0:
        return 0.0

    # Efficiency calculation
    efficiency = ref_efficiency * (ref_distance / distance) ** decay_exponent

    # Clamp efficiency to a realistic range [0, 1]
    efficiency = max(0, min(1, efficiency))

    # Determine TX power to use
    tx_power = tx_power_w if tx_power_w is not None else getattr(tx_node, 'mrc_tx_power_w', 0.0)

    # Received power is the transmitter's power multiplied by the efficiency
    received_power_w = tx_power * efficiency

    return received_power_w

