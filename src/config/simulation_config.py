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
    ENABLE_TERRAIN_MODEL = True  # 使用真实地形（若提供高度图则加载，否则回退内置 DEM）
    # 数字高程模型（DEM）地形设置
    TERRAIN_RESOLUTION = 10  # 每个网格点的分辨率（单位：米）
    TERRAIN_MAX_ELEVATION = 1000  # 最大海拔高度（单位：米），以容纳925米的最高RIS
    # 若提供真实高度热力图，则填写路径和比例尺；否则回退到内置随机地形
    HEIGHTMAP_PATH = "src/data/heightmap.png"  # Carrauntoohil 高度热力图
    HEIGHTMAP_SCALE_BAR_METERS = 500  # 图中标注的标尺长度（米）
    HEIGHTMAP_SCALE_BAR_PIXELS = 250  # 标尺对应的像素长度，需量取后填写
    HEIGHTMAP_MIN_ELEV = 81   # 热力图最低处对应的海拔（米），可用 Sink 海拔近似
    HEIGHTMAP_MAX_ELEV = 925  # 热力图最高处对应的海拔（米），可用最高 RIS 海拔近似
    # 大气和路径损耗参数（CI模型）
    REFERENCE_DISTANCE = 1.0  # d0, meters
    # LoS / NLoS 路径损耗指数（可根据场景校准）
    PATH_LOSS_EXPONENT_LOS = 1.5
    PATH_LOSS_EXPONENT_NLOS = 3.5
    # 固定 NLOS 额外损耗（建议为 0；若场景需要，可设为 0–5 dB）
    NLOS_EXTRA_LOSS_DB = 0.0

# ==============================================================================
# 汇聚节点（主站）配置
# ==============================================================================
class SinkConfig:
    # 位置改由 sink.csv 提供（此处不再设定坐标）
    # 发射功率（单位：瓦）
    TRANSMIT_POWER_W = 10.0
    # 工作频率（单位：赫兹）
    FREQUENCY_HZ = 100e6  # 频率可在此配置，位置由数据文件提供
    # 天线增益（单位：dBi）
    ANTENNA_GAIN_DBI = 18.0

# ==============================================================================
# 可重构智能表面 (RIS) 配置
# ==============================================================================
class RISConfig:
    # 功能开关
    ENABLE_RIS = True          # 是否启用 RIS 辅助链路
    ENABLE_DOUBLE_HOP = True   # 是否允许双跳（两级 RIS）路由
    ENABLE_PHASE_QUANT = True  # 是否对相位进行量化（3 bit）
    # RIS 面板电磁参数（位置与数量改由 sink.csv 提供）
    NUM_ELEMENTS_H = 16
    NUM_ELEMENTS_V = 16
    # 相位分辨率 (比特)
    PHASE_RESOLUTION_BITS = 3  # 2^3 = 8个离散相位级别
    # 单元间距与波长的关系
    ELEMENT_SPACING_FACTOR = 0.5  # d = 0.5 * lambda
    # RIS单元的能量收集效率 (如果是自供电)
    HARVESTING_EFFICIENCY = 0.5

# ==============================================================================
# 无线传感器网络 (WSN) 配置
# ==============================================================================
class WSNConfig:
    # 功能开关
    ENABLE_SOLAR = True              # 是否启用节点太阳能收集（全局开关，具体到簇由下方列表控制）
    ENABLE_MRC_LOCAL_TRANSFER = True # 是否启用簇内 MRC 能量下发
    # 基本簇规模配置
    NUM_CLUSTERS = 6
    NODES_PER_CLUSTER = 10           # 仅在未按面积分配时使用
    # 是否按湖泊面积分配各簇成员数
    ALLOCATE_BY_LAKE_AREA = True
    LAKE_CSV_PATH = "src/data/LAKE.csv"
    # 可选：总的传感器节点数量（若为 None 则使用 NUM_CLUSTERS*NODES_PER_CLUSTER）
    TOTAL_SENSOR_NODES = None
    # 按簇顺序对应的湖泊名称（用于面积-簇映射）。顺序需与簇创建顺序一致（先SOLAR后NON_SOLAR）。
    LAKE_NAME_PER_CLUSTER = ["lake1", "lake2", "lake3", "lake5", "lake4", "lake6"]

    # 将簇按“是否采集太阳能”分为两类，分别维护坐标列表
    SOLAR_CLUSTER_HEAD_POSITIONS = [
        np.array([1200, 1300, 287]), # 湖泊1
        np.array([1300, 500, 497]),  # 湖泊2
        
    ]
    NON_SOLAR_CLUSTER_HEAD_POSITIONS = [
        
        np.array([2800, 450, 657]),  # 湖泊3
        np.array([3200, 1200, 331]), # 湖泊5
        np.array([2500, 1100, 536]), # 湖泊4
        np.array([3700, 1300, 343]), # 湖泊6
    ]

    # 每个簇的半径 (节点部署在簇头周围的这个半径内)
    CLUSTER_RADIUS = 45  # 单位：米（随机半径R*sqrt(U)的期望为2R/3，取45m使平均距离≈30m）

# ==============================================================================
# 传感器节点与簇头配置
# ==============================================================================
class SensorNodeConfig:
    # 初始能量 (单位：焦耳)
    INITIAL_ENERGY_J = 0.5
    # 最低工作能量水平 (单位：焦耳)
    MIN_ENERGY_J = 0.001
    # 传感能耗 (单位：焦耳/样本)
    SENSING_ENERGY_J = 50e-6  # 50 uJ
    # 发送数据包能耗 (单位：焦耳/包)
    TX_ENERGY_J = 100e-6 # 100 uJ (base overhead)
    # 空闲状态功耗 (单位：瓦)
    IDLE_POWER_W = 0.0  # Disabled idle consumption
    # 传感器判定“能量富足”的阈值 (单位：焦耳)
    ABUNDANT_THRESHOLD_J = 0.4

    # 上报能耗的距离相关模型（独立于节点通信参数，避免过大能耗）
    REPORT_PACKET_BITS = 4000                    # 上报包大小 (bits)
    REPORT_E_ELEC_J_PER_BIT = 50e-9              # 电子学能耗 (J/bit)
    REPORT_EPSILON_AMP_J_PER_BIT_MTAU = 10e-12   # 放大器能耗 (J/bit/m^tau) - set to 10 pJ/bit/m^2 as per radio model
    REPORT_PATH_LOSS_EXPONENT = 2.0              # 路径损耗指数 tau
    REPORT_INCLUDE_CH_RX = True                  # 是否计入簇头接收能耗（E_elec * B）

    # MRC接收天线增益 (单位：dBi)
    MRC_RX_GAIN_DBI = 2.0
    # MRC发射功率 (单位：瓦)
    MRC_TX_POWER_W = 0.1

class ClusterHeadConfig:
    # 簇头与传感器节点有相似的基础属性，但电池容量可能更大
    INITIAL_ENERGY_J = 1.0
    # RF接收天线增益 (单位：dBi)
    RF_RX_GAIN_DBI = 9.0
    # 远场RF发射能力（用于跨簇供能）
    CH_RF_TX_POWER_W = 1.0
    CH_RF_TX_FREQUENCY_HZ = SinkConfig.FREQUENCY_HZ
    CH_RF_TX_GAIN_DBI = 9.0
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
    ENABLE_CROSS_CLUSTER_DONATION = True  # 是否启用跨簇触发式供能（由CH作为源）
    # 触发式跨簇供能参数
    CROSS_CLUSTER_TRIGGER_PERIOD_STEPS = 100
    TRIGGER_LOW_PCT = 0.30
    TRIGGER_HIGH_PCT = 0.80
    # 总仿真时间 (单位：秒)
    SIMULATION_TIME_S = 7200  # 2小时（论文图快速生成，可调）
    # 时间步长 (单位：秒)
    TIME_STEP_S = 1.0
    # 用于可复现性的随机种子
    RANDOM_SEED = 42

# ------------------------------------------------------------------------------
# 兼容旧代码的别名（RF_TxConfig -> SinkConfig）
# ------------------------------------------------------------------------------
RF_TxConfig = SinkConfig

