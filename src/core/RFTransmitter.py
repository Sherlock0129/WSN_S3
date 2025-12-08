"""
RF Transmitter class for the main power source.
"""

import numpy as np

from src.config.simulation_config import SinkConfig

class RFTransmitter:
    def __init__(self):
        """
        Initializes the RF power transmitter.
        """
        # 这里将“RF 发射机”视为汇聚节点（Sink）
        self.position = SinkConfig.POSITION
        self.power_w = SinkConfig.TRANSMIT_POWER_W
        self.frequency_hz = SinkConfig.FREQUENCY_HZ
        self.antenna_gain_dbi = SinkConfig.ANTENNA_GAIN_DBI

    def get_tx_power_dbm(self):
        """
        Returns the transmit power in dBm.
        """
        power_mw = self.power_w * 1000
        return 10 * np.log10(power_mw)

    def __repr__(self):
        return f"RFTransmitter(Position={self.position}, Power={self.power_w}W)"

