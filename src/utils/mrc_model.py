"""
Physics model for Magnetic Resonant Coupling (MRC) near-field power transfer.
"""

import numpy as np

from src.config.simulation_config import ClusterHeadConfig

def calculate_received_mrc_power(tx_ch, rx_node):
    """
    Calculates the received power at a sensor node from a cluster head via MRC.
    This is a simplified model based on an inverse power law, suitable for system-level simulation.

    Args:
        tx_ch (ClusterHead): The transmitting cluster head.
        rx_node (SensorNode): The receiving sensor node.

    Returns:
        float: Received power in Watts.
    """
    distance = np.linalg.norm(tx_ch.position - rx_node.position)
    
    # Near-field power drops off very quickly with distance. We use a simple model.
    # Let's assume a certain efficiency at a reference distance (e.g., 1 meter)
    # and a rapid decay exponent.
    ref_distance = 1.0  # meters
    ref_efficiency = 0.8 # 80% efficiency at 1m
    decay_exponent = 3.0 # Power decays with d^3 or faster in some models

    if distance <= 0:
        return 0.0

    # Efficiency calculation
    efficiency = ref_efficiency * (ref_distance / distance)**decay_exponent
    
    # Clamp efficiency to a realistic range [0, 1]
    efficiency = max(0, min(1, efficiency))
    
    # Received power is the transmitter's power multiplied by the efficiency
    received_power_w = tx_ch.mrc_tx_power_w * efficiency
    
    return received_power_w

