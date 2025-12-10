import math

import numpy as np

# 引入仿真配置，用于能量采集/衰减基于时间步长与空闲功耗
from src.config.simulation_config import SimConfig, SensorNodeConfig


class SensorNode:
    def __init__(self,
                 node_id: int,
                 initial_energy: float,
                 low_threshold: float,
                 high_threshold: float,
                 position: list,
                 has_solar: bool = True,
                 # 电池参数
                 capacity: float = 3.5,
                 voltage: float = 3.7,
                 # 太阳能参数
                 enable_energy_harvesting: bool = True,
                 solar_efficiency: float = 0.18,
                 solar_area: float = 0.001,
                 max_solar_irradiance: float = 800.0,
                 env_correction_factor: float = 0.6,
                 # 传输参数
                 energy_char: float = 1000.0,
                 energy_elec: float = 1e-4,
                 epsilon_amp: float = 1e-5,
                 bit_rate: float = 1000000.0,
                 path_loss_exponent: float = 2.0,
                 energy_decay_rate: float = 0.0,
                 sensor_energy: float = 0.1,
                 # 移动性参数
                 is_mobile: bool = False,
                 mobility_pattern: str = None,
                 mobility_params: dict = None,
                 # 物理中心标识
                 is_physical_center: bool = False):
        """
        初始化传感器节点
        
        :param node_id: 节点唯一ID
        :param initial_energy: 初始能量 (Joules)
        :param low_threshold: 低能量阈值 (容量百分比)
        :param high_threshold: 高能量阈值 (容量百分比)
        :param position: 节点位置 [x, y]
        :param has_solar: 是否具有太阳能收集能力
        :param capacity: 电池容量 (mAh)
        :param voltage: 电压 (V)
        :param solar_efficiency: 太阳能效率
        :param solar_area: 太阳能板面积 (m^2)
        :param max_solar_irradiance: 最大太阳辐射 (W/m^2)
        :param env_correction_factor: 环境修正因子
        :param energy_char: 充电能量 (J)
        :param energy_elec: 电子能量 (J per bit)
        :param epsilon_amp: 放大能量 (J per bit per distance^2)
        :param bit_rate: 传输比特率 (bits)
        :param path_loss_exponent: 路径损耗指数
        :param energy_decay_rate: 能量衰减率 (J per time step)
        :param sensor_energy: 传感器能量 (J per time step)
        :param is_mobile: 是否可移动
        :param mobility_pattern: 移动模式
        :param mobility_params: 移动参数
        :param is_physical_center: 是否为物理中心节点（特殊节点，不参与WET）
        """
        self.node_id = node_id
        self.position = position  # [x, y] position of the node
        self.has_solar = has_solar
        self.is_physical_center = is_physical_center  # 物理中心节点标识

        # Energy management parameters
        self.initial_energy = initial_energy
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.capacity = capacity  # Battery capacity in mAh
        self.V = voltage  # Voltage (V)
        self.energy_history = []  # To track energy consumption and generation over time
        self.current_energy = initial_energy

        # Solar panel parameters (if the node has a solar panel)
        self.enable_energy_harvesting = enable_energy_harvesting
        self.solar_efficiency = solar_efficiency
        self.solar_area = solar_area  # Area of a solar panel in m^2
        self.G_max = max_solar_irradiance  # Max solar irradiance in W/m^2
        self.env_correction_factor = env_correction_factor  # Environmental factor for solar collection

        # Wireless Energy Transfer (WET) parameters
        self.E_char = energy_char  # Energy consumed for charging during WET (J)
        self.E_elec = energy_elec  # Energy consumed for electronics (per bit) (J)
        self.epsilon_amp = epsilon_amp  # Amplification energy for transmission (per bit per distance^2) (J)
        self.B = bit_rate  # Transmission bit rate in bits
        self.d = 1  # Distance between nodes (will be calculated dynamically)
        self.tau = path_loss_exponent  # Path loss exponent

        # Threshold energy calculation
        self.low_threshold_energy = self.low_threshold * self.capacity * self.V * 3600  # Convert to Joules
        self.high_threshold_energy = self.high_threshold * self.capacity * self.V * 3600  # Convert to Joules

        # To track energy transferred and received (for WET)
        self.received_energy = 0
        self.transferred_energy = 0
        self.received_history = []  # List to track received energy history
        self.transferred_history = []  # List to track transferred energy history

        self.is_mobile = is_mobile
        self.mobility_pattern = mobility_pattern  # e.g., "circle", "line", "oscillate"
        self.mobility_params = mobility_params or {}  # custom parameters

        # Energy decay and sensor consumption
        self.energy_decay_rate = energy_decay_rate
        self.sensor_energy = sensor_energy

        # MRC transmitter properties for energy convergence
        self.mrc_tx_power_w = SensorNodeConfig.MRC_TX_POWER_W

        # 在 __init__ 里：
        self.position_history = [tuple(self.position)]
        
        # 如果是物理中心节点，打印特殊信息
        if self.is_physical_center:
            print(f"[SensorNode] 创建物理中心节点 ID={self.node_id}, "
                  f"位置=({self.position[0]:.3f}, {self.position[1]:.3f}), "
                  f"初始能量={self.initial_energy:.1f}J")

    def record_transfer(self, received=0, transferred=0):
        """
        Record energy transfer activities: received and transferred energy.

        :param received: Energy received by the node (in Joules).
        :param transferred: Energy transferred by the node (in Joules).
        """
        self.received_history.append(received)
        self.transferred_history.append(transferred)
        self.received_energy += received
        self.transferred_energy += transferred

    def distance_to(self, other_node):
        """
        Calculate the Euclidean distance between this node and another node.

        :param other_node: The other node to calculate distance to.
        :return: The Euclidean distance between the two nodes.
        """
        pos1 = np.array(self.position)
        pos2 = np.array(other_node.position)
        return np.linalg.norm(pos1 - pos2)


    def solar_irradiance(self, t):
        """
        Calculate the solar irradiance based on time of day.
        This is a simplified calculation using a sinusoidal model for the daylight period.

        :param t: Time in minutes since the start of the day (e.g., 0 to 1440 minutes).
        :return: Solar irradiance in W/m^2 at time `t`.
        """
        t = t % 1440  # Normalize time to cycle daily
        if 360 <= t <= 1080:  # Between sunrise and sunset (6:00 AM to 6:00 PM)
            return self.G_max * np.sin(np.pi * (t - 360) / 720)  # Simplified sinusoidal model
        return 0

    def energy_harvest(self, t):
        """
        Calculate the energy harvested from the solar panel at a given time.

        :param t: Time in minutes since the start of the day (e.g., 0 to 1440 minutes).
        :return: The energy harvested in Joules during the current time step.
        """
        if not self.has_solar or not self.enable_energy_harvesting:
            return 0.0
        # Irradiance at time t (W/m^2)
        G_t = self.solar_irradiance(t)
        # Harvested power (W)
        power_w = self.solar_efficiency * self.solar_area * G_t * self.env_correction_factor
        # Convert to energy for the current simulation time step (J)
        harvested_energy = power_w * SimConfig.TIME_STEP_S
        return harvested_energy

    def energy_decay(self):
        """
        Calculate the energy decay of the node's battery over one simulation step.
        Uses idle power model: E_idle = P_idle * dt.

        :return: The decayed energy (in Joules).
        """
        decay_energy = SensorNodeConfig.IDLE_POWER_W * SimConfig.TIME_STEP_S
        return decay_energy

    def energy_generation(self, t, receive_WET=0):
        """
        Calculate the total energy generated by the node, considering both solar energy harvesting
        and received energy from WET.

        :param t: Time step in minutes.
        :param receive_WET: The energy received through Wireless Energy Transfer (WET) (in Joules).
        :return: Total energy generated (in Joules).
        """
        harvested_energy = self.energy_harvest(t)
        total_generated_energy = harvested_energy + receive_WET
        return total_generated_energy

    def energy_consumption(self, target_node=None, transfer_WET=False):
        """
        Calculate the total energy consumed for a single communication (TX + RX),
        optionally including Wireless Energy Transfer (WET) overhead.

        :param target_node: The node to which this node is communicating.
        :param transfer_WET: Whether this node also performs energy transfer (e.g., WET).
        :return: Total energy consumed (in Joules)
        """
        B = self.B
        d = self.d if target_node is None else self.distance_to(target_node)

        # 发射能耗
        E_tx = self.E_elec * B + self.epsilon_amp * B * (d ** self.tau)

        # 接收能耗（假设双向确认通信）
        E_rx = self.E_elec * B

        # 通信总能耗
        E_com = E_tx + E_rx
        E_com = E_com/2
        E_sen = 0.1 #J

        if transfer_WET:
            E_com += self.E_char  # 加上传能附加开销

        return E_com + E_sen  # 返回通信能耗 + 传感器能耗

    def energy_transfer_efficiency(self, target_node):
        """
        Calculate wireless energy transfer efficiency based on distance.

        :param target_node: Receiver node.
        :return: Efficiency (0~1)
        """
        d = self.distance_to(target_node)
        eta_0 = 0.6  # 1米处最大效率
        gamma = 2.0  # 衰减因子
        
        # 修复：使用更合理的效率公式
        # 当距离很小时，效率接近但不等于1
        # 使用指数衰减模型：eta = eta_0 * exp(-gamma * (d - 1))
        if d <= 1.0:
            # 距离≤1m时，效率为eta_0到1之间的线性插值
            efficiency = eta_0 + (1.0 - eta_0) * (1.0 - d)
        else:
            # 距离>1m时，使用指数衰减
            efficiency = eta_0 * (1.0 / (d ** gamma))
        
        return min(1.0, max(0.0, efficiency))  # 限定在 [0, 1] 之间

    def receive_mrc_power(self, received_power_w, time_step_s):
        """
        Updates the node's energy from received MRC power.

        Args:
            received_power_w (float): The power received at the node in Watts.
            time_step_s (float): The duration of the time step in seconds.
        """
        # Assume perfect energy harvesting efficiency from MRC for now
        harvested_energy_j = received_power_w * time_step_s
        self.current_energy += harvested_energy_j
        self.record_transfer(received=harvested_energy_j)

    def update_energy(self, t):
        """
        Update the energy state of the node at time t, only considering solar harvesting and decay.

        :param t: Time step in minutes.
        :return: Tuple of (generated_energy, decay_energy)
        """
        E_gen = self.energy_harvest(t)
        E_decay = self.energy_decay()

        self.current_energy = self.current_energy + E_gen - E_decay
        self.current_energy = max(0, min(self.current_energy, self.capacity * self.V * 3600))

        self.energy_history.append({"time": t, "generated": E_gen, "consumed": E_decay})
        return E_gen, E_decay

    def update_position(self, t):
        """If the node is mobile, update its position based on its mobility pattern."""
        if not self.is_mobile or not self.mobility_pattern:
            return

        if self.mobility_pattern == "circle":
            radius = self.mobility_params.get('radius', 1.0)
            speed = self.mobility_params.get('speed', 0.01)  # radians per time step
            cx, cy = self.mobility_params.get('center', self.position)  # circle center
            angle = speed * t
            self.position[0] = cx + radius * math.cos(angle)
            self.position[1] = cy + radius * math.sin(angle)

        elif self.mobility_pattern == "line":
            amplitude = self.mobility_params.get('amplitude', 1.0)
            speed = self.mobility_params.get('speed', 0.1)
            direction = self.mobility_params.get('direction', [1, 0])  # [dx, dy]
            self.position[0] += direction[0] * speed
            self.position[1] += direction[1] * speed

        elif self.mobility_pattern == "oscillate":
            amplitude = self.mobility_params.get('amplitude', 1.0)
            freq = self.mobility_params.get('freq', 0.01)
            axis = self.mobility_params.get('axis', 'x')
            delta = amplitude * math.sin(freq * t)
            if axis == 'x':
                self.position[0] += delta
            else:
                self.position[1] += delta

        # 在 update_position(t) 最后：
        self.position_history.append((self.position[0], self.position[1]))

    # def record_transfer(self, received=0, transferred=0):
    #     """
    #     Record energy transfer activities: received and transferred energy.
    #
    #     :param received: Energy received by the node (in Joules).
    #     :param transferred: Energy transferred by the node (in Joules).
    #     """
    #     self.received_energy += received
    #     self.transferred_energy += transferred
    #
    # def distance_to(self, other_node):
    #     """
    #     Calculate the Euclidean distance between this node and another node.
    #
    #     :param other_node: The other node to calculate distance to.
    #     :return: The Euclidean distance between the two nodes.
    #     """
    #     x1, y1 = self.position
    #     x2, y2 = other_node.position
    #     return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
