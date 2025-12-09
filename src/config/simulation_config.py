"""
Simulation Configuration for Hierarchical WPT System
Based on the proposal: "A Scalable Hierarchical Wireless Power Transfer System
Based on Reconfigurable Intelligent Surfaces for Complex-Environment WSNs"
"""

import numpy as np

# ==============================================================================
# Environment Configuration
# ==============================================================================
class EnvConfig:
    # 仿真区域（单位：米），根据地图估算
    AREA_WIDTH = 4000
    AREA_HEIGHT = 2000
    # 功能开关
    ENABLE_HEIGHTMAP = False   # 是否启用真实高度热力图（需设置 HEIGHTMAP_PATH）
    ENABLE_TERRAIN_LOS = True # 是否进行地形遮挡的视距判断
    ENABLE_TERRAIN_MODEL = False  # 若无需地形，设为 False 完全禁用 DEM
    # 数字高程模型（DEM）地形设置
    TERRAIN_RESOLUTION = 10  # 每个网格点的分辨率（单位：米）
    TERRAIN_MAX_ELEVATION = 1000  # 最大海拔高度（单位：米），以容纳925米的最高RIS
    # 若提供真实高度热力图，则填写路径和比例尺；否则回退到内置随机地形
    HEIGHTMAP_PATH = None  # 示例: "data/heightmap.png"
    HEIGHTMAP_SCALE_BAR_METERS = 500  # 图中标注的标尺长度（米）
    HEIGHTMAP_SCALE_BAR_PIXELS = 250  # 标尺对应的像素长度，需量取后填写
    HEIGHTMAP_MIN_ELEV = 81   # 热力图最低处对应的海拔（米），可用 Sink 海拔近似
    HEIGHTMAP_MAX_ELEV = 925  # 热力图最高处对应的海拔（米），可用最高 RIS 海拔近似
    # 大气和路径损耗参数
    PATH_LOSS_EXPONENT = 1.5  # 自由空间近似：路径损耗指数 n=1.5
    REFERENCE_DISTANCE = 1.0  # meters
    REFERENCE_PATH_LOSS = 30.0  # dB
    NLOS_EXTRA_LOSS_DB = 20.0   # 非视距链路额外损耗（dB）

# ==============================================================================
# 汇聚节点（主站）配置
# ==============================================================================
class SinkConfig:
    # 汇聚节点（Blackvalley）的位置
    POSITION = np.array([0, 900, 81])  # x, y, z (单位：米)
    # 发射功率（单位：瓦）
    TRANSMIT_POWER_W = 10.0
    # 工作频率（单位：赫兹）
    FREQUENCY_HZ = 300e6  # 300 MHz（用于灵敏度分析）
    # 天线增益（单位：dBi）
    ANTENNA_GAIN_DBI = 6.0

# ==============================================================================
# 可重构智能表面 (RIS) 配置
# ==============================================================================
class RISConfig:
    # 功能开关
    ENABLE_RIS = True          # 是否启用 RIS 辅助链路
    ENABLE_DOUBLE_HOP = True   # 是否允许双跳（两级 RIS）路由
    ENABLE_PHASE_QUANT = True  # 是否对相位进行量化（3 bit）
    # RIS面板的数量
    NUM_RIS_PANELS = 3
    # 每个RIS面板的单元数量 (例如, 16x16)
    NUM_ELEMENTS_H = 16
    NUM_ELEMENTS_V = 16
    # 相位分辨率 (比特)
    PHASE_RESOLUTION_BITS = 3  # 2^3 = 8个离散相位级别
    # 单元间距与波长的关系
    ELEMENT_SPACING_FACTOR = 0.5  # d = 0.5 * lambda
    # RIS面板的位置，根据地图估算
    POSITIONS = [
        np.array([2250, 1150, 885]), # RIS1 (顶部)
        np.array([2250, 950, 925]),  # RIS2 (中部)
        np.array([2250, 600, 866]),  # RIS3 (底部)
    ]
    # RIS单元的能量收集效率 (如果是自供电)
    HARVESTING_EFFICIENCY = 0.5

# ==============================================================================
# 无线传感器网络 (WSN) 配置
# ==============================================================================
class WSNConfig:
    # 功能开关
    ENABLE_SOLAR = True             # 是否启用节点太阳能收集
    ENABLE_MRC_LOCAL_TRANSFER = True # 是否启用簇内 MRC 能量下发
    # 簇的数量 (湖泊)
    NUM_CLUSTERS = 6
    # 每个簇中的传感器节点数量
    NODES_PER_CLUSTER = 10
    # 簇头位置 (根据地图估算)，代表每个湖泊的中心
    CLUSTER_HEAD_POSITIONS = [
        np.array([1200, 1300, 287]), # 湖泊1
        np.array([1300, 500, 497]),  # 湖泊2
        np.array([2800, 450, 657]),  # 湖泊3 (Lough Cummeennageasta)
        np.array([2500, 1100, 536]), # 湖泊4
        np.array([3200, 1200, 331]), # 湖泊5 (Lough Callee)
        np.array([3700, 1300, 343]), # 湖泊6 (Lough Gouragh)
    ]
    # 每个簇的半径 (节点部署在簇头周围的这个半径内)
    CLUSTER_RADIUS = 50  # 单位：米

# ==============================================================================
# 传感器节点与簇头配置
# ==============================================================================
class SensorNodeConfig:
    # 初始能量 (单位：焦耳)
    INITIAL_ENERGY_J = 0.05
    # 最低工作能量水平 (单位：焦耳)
    MIN_ENERGY_J = 0.001
    # 传感能耗 (单位：焦耳/样本)
    SENSING_ENERGY_J = 50e-6  # 50 uJ
    # 发送数据包能耗 (单位：焦耳/包)
    TX_ENERGY_J = 100e-6 # 100 uJ
    # 空闲状态功耗 (单位：瓦)
    IDLE_POWER_W = 1e-6  # 1 uW
    # MRC接收天线增益 (单位：dBi)
    MRC_RX_GAIN_DBI = 2.0

class ClusterHeadConfig:
    # 簇头与传感器节点有相似的基础属性，但电池容量可能更大
    INITIAL_ENERGY_J = 1.0
    # RF接收天线增益 (单位：dBi)
    RF_RX_GAIN_DBI = 3.0
    # MRC发射器属性
    MRC_TX_POWER_W = 0.5
    MRC_TX_FREQUENCY_HZ = 13.56e6 # 13.56 MHz 用于近场通信
    MRC_TX_GAIN_DBI = 3.0

# ==============================================================================
# 仿真控制
# ==============================================================================
class SimConfig:
    # 功能开关
    ENABLE_ROUTING = True           # 是否进行能量路由（直射/RIS）
    ENABLE_SCHEDULER = True         # 是否启用调度器选择充电对象
    ENABLE_LOGGING = True           # 是否打印仿真过程日志
    ENABLE_PLOT_RESULTS = True      # 是否在仿真结束后绘制能量曲线
    ENABLE_SCENE_VIZ = True         # 是否启用场景可视化脚本（单独运行时也可关闭）
    # 总仿真时间 (单位：秒)
    SIMULATION_TIME_S = 3600  # 1小时
    # 时间步长 (单位：秒)
    TIME_STEP_S = 1.0
    # 用于可复现性的随机种子
    RANDOM_SEED = 42

# ------------------------------------------------------------------------------
# 兼容旧代码的别名（RF_TxConfig -> SinkConfig）
# ------------------------------------------------------------------------------
RF_TxConfig = SinkConfig

