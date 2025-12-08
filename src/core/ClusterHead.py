"""
Cluster Head class, which inherits from SensorNode but has additional capabilities.
"""

import numpy as np

from src.core.SensorNode import SensorNode
from src.config.simulation_config import ClusterHeadConfig, SensorNodeConfig

class ClusterHead(SensorNode):
    def __init__(self, node_id, position):
        """
        Initializes a Cluster Head node.

        Args:
            node_id (int): Unique ID for the node.
            position (np.array): 3D coordinates of the cluster head.
        """
        # Initialize the base SensorNode class with ClusterHead parameters
        # We disable solar for this simulation as per the proposal's focus.
        super().__init__(
            node_id=node_id,
            initial_energy=ClusterHeadConfig.INITIAL_ENERGY_J,
            position=position,
            # The following are inherited from the base SensorNode but may not be used
            # in the same way for a Cluster Head.
            low_threshold=0.1, # Example value
            high_threshold=0.9, # Example value
            has_solar=False # As per proposal, we rely on WPT
        )

        # RF receiver properties for long-range energy harvesting
        self.rf_rx_gain_dbi = ClusterHeadConfig.RF_RX_GAIN_DBI

        # MRC transmitter properties for local energy distribution
        self.mrc_tx_power_w = ClusterHeadConfig.MRC_TX_POWER_W
        self.mrc_tx_frequency_hz = ClusterHeadConfig.MRC_TX_FREQUENCY_HZ
        self.mrc_tx_gain_dbi = ClusterHeadConfig.MRC_TX_GAIN_DBI
        
        print(f"ClusterHead {self.node_id} created at position {self.position}")

    def receive_rf_power(self, received_power_w, time_step_s):
        """
        Updates the cluster head's energy from received RF power.

        Args:
            received_power_w (float): The power received at the CH in Watts.
            time_step_s (float): The duration of the time step in seconds.
        """
        # For now, assume perfect energy harvesting efficiency at the CH
        harvested_energy_j = received_power_w * time_step_s
        self.current_energy += harvested_energy_j
        self.record_transfer(received=harvested_energy_j)

    def transmit_mrc_power(self, target_nodes, time_step_s, mrc_model):
        """
        Transmits power to nearby sensor nodes via MRC and consumes energy.

        Args:
            target_nodes (list[SensorNode]): List of sensor nodes in the cluster.
            time_step_s (float): The duration of the power transmission.
            mrc_model (MRC_Model): The physics model for MRC power transfer.
        """
        energy_consumed = self.mrc_tx_power_w * time_step_s
        if self.current_energy < energy_consumed:
            # Not enough energy to transmit
            return

        self.current_energy -= energy_consumed
        self.record_transfer(transferred=energy_consumed)

        # Calculate and deliver power to each target node
        for node in target_nodes:
            received_power_w = mrc_model.calculate_received_mrc_power(self, node)
            node.receive_mrc_power(received_power_w, time_step_s)

    def __repr__(self):
        return f"ClusterHead(ID={self.node_id}, Energy={self.current_energy:.3f}J)"

