"""
Physics model for RF (far-field) propagation, including Friis path loss and RIS reflection.
"""

import numpy as np

from src.config.simulation_config import SinkConfig, EnvConfig

def friis_path_loss(tx_power_dbm, tx_gain_dbi, rx_gain_dbi, frequency_hz, distance_m):
    """
    Calculates the received power in dBm using the Friis transmission equation.

    Args:
        tx_power_dbm (float): Transmitter power in dBm.
        tx_gain_dbi (float): Transmitter antenna gain in dBi.
        rx_gain_dbi (float): Receiver antenna gain in dBi.
        frequency_hz (float): Signal frequency in Hz.
        distance_m (float): Distance between transmitter and receiver in meters.

    Returns:
        float: Received power in dBm.
    """
    if distance_m <= 0:
        return -np.inf

    c = 3e8  # Speed of light
    wavelength = c / frequency_hz
    
    # Path loss in dB
    path_loss_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) - 147.55
    
    received_power_dbm = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss_db
    return received_power_dbm

def calculate_received_rf_power(tx, rx, env):
    """
    Calculates the received RF power in Watts from a transmitter to a receiver,
    considering path loss and line-of-sight.

    Args:
        tx (RFTransmitter): The RF transmitter object.
        rx (ClusterHead): The receiver object (Cluster Head).
        env (Environment): The simulation environment for LoS check.

    Returns:
        float: Received power in Watts.
    """
    distance = np.linalg.norm(tx.position - rx.position)
    is_los = env.check_los(tx.position, rx.position)

    tx_power_dbm = tx.get_tx_power_dbm()
    received_power_dbm = _log_distance_path_loss(
        tx_power_dbm,
        tx.antenna_gain_dbi,
        rx.rf_rx_gain_dbi,
        tx.frequency_hz,
        distance,
        is_los
    )
    # 转换回瓦特
    return 10 ** ((received_power_dbm - 30) / 10)

def calculate_ris_assisted_power(source, ris, target, env):
    """
    Calculates the received power at a target via a single RIS panel.
    This uses a two-hop Friis model (source -> RIS -> target).

    Args:
        source (RFTransmitter or RIS): The source of the signal.
        ris (RIS): The RIS panel.
        target (RIS or ClusterHead): The target receiver.
        env (Environment): The simulation environment for LoS checks.

    Returns:
        float: Received power in Watts.
    """
    # 1. Path from source to RIS
    dist_source_ris = np.linalg.norm(source.position - ris.position)
    if not env.check_los(source.position, ris.position):
        return 0.0

    # Power received at the RIS (as if it were an isotropic antenna)
    tx_power_dbm = source.get_tx_power_dbm() if hasattr(source, 'get_tx_power_dbm') else 10 * np.log10(source.power_w * 1000)
    source_gain = source.antenna_gain_dbi if hasattr(source, 'antenna_gain_dbi') else source.get_reflection_gain()

    power_at_ris_dbm = _log_distance_path_loss(
        tx_power_dbm, 
        source_gain, 
        0, # Isotropic antenna gain for RIS element
        SinkConfig.FREQUENCY_HZ, 
        dist_source_ris,
        True  # RIS 反射要求 LoS，前面已检查
    )

    # 2. Path from RIS to target
    dist_ris_target = np.linalg.norm(ris.position - target.position)
    if not env.check_los(ris.position, target.position):
        return 0.0

    # The RIS now acts as a transmitter with a certain gain
    ris.configure_phases(source.position, target.position)
    ris_gain_dbi = ris.get_reflection_gain()
    
    # Power received at the target from the RIS
    rx_gain = target.rf_rx_gain_dbi if hasattr(target, 'rf_rx_gain_dbi') else target.get_reflection_gain()

    received_power_dbm = _log_distance_path_loss(
        power_at_ris_dbm, 
        ris_gain_dbi, 
        rx_gain, 
        SinkConfig.FREQUENCY_HZ, 
        dist_ris_target,
        True  # RIS 反射要求 LoS，前面已检查
    )

    # Convert dBm to Watts
    return 10 ** ((received_power_dbm - 30) / 10)

def _log_distance_path_loss(tx_power_dbm, tx_gain_dbi, rx_gain_dbi, frequency_hz, distance_m, is_los):
    """
    对 Friis 模型进行扩展，加入对数距离路径损耗指数与 NLOS 额外损耗。
    """
    if distance_m <= 0:
        return -np.inf

    # 基础自由空间路径损耗 (dB)
    fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) - 147.55

    # 对数距离模型修正：当路径损耗指数 n != 2 时进行补偿
    n = EnvConfig.PATH_LOSS_EXPONENT
    fspl_ref_db = 20 * np.log10(distance_m)
    log_dist_correction = 10 * (n - 2) * np.log10(distance_m)

    path_loss_db = fspl_db + log_dist_correction

    # NLOS 额外损耗
    if not is_los:
        path_loss_db += EnvConfig.NLOS_EXTRA_LOSS_DB

    return tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss_db

