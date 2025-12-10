"""
RF Transmitter class for the main power source.
"""

import numpy as np

from src.config.simulation_config import SinkConfig

class RFTransmitter:
    def __init__(self, position=None, power_w=None, frequency_hz=None, antenna_gain_dbi=None):
        """
        Initializes the RF power transmitter.
        Optionally override defaults from SinkConfig via parameters.
        """
        # 这里将“RF 发射机”视为汇聚节点（Sink）
        self.position = position if position is not None else SinkConfig.POSITION
        self.power_w = power_w if power_w is not None else SinkConfig.TRANSMIT_POWER_W
        self.frequency_hz = frequency_hz if frequency_hz is not None else SinkConfig.FREQUENCY_HZ
        self.antenna_gain_dbi = antenna_gain_dbi if antenna_gain_dbi is not None else SinkConfig.ANTENNA_GAIN_DBI

    def get_tx_power_dbm(self):
        """
        Returns the transmit power in dBm.
        """
        power_mw = self.power_w * 1000
        return 10 * np.log10(power_mw)

    def __repr__(self):
        return f"RFTransmitter(Position={self.position}, Power={self.power_w}W)"

