# WSN_S3 项目架构与能量传输调度/路由逻辑详解

## 📋 项目概述

**WSN_S3** 是一个基于**可重构智能表面 (RIS)** 的**分层无线能量传输系统**的仿真平台。该系统为复杂环境下的无线传感器网络 (WSN) 提供可扩展的能量供给方案。

### 核心创新点：
- **多级能量传输架构**：RF远场 → RIS反射 → 簇头 → 传感器节点
- **智能路由**：支持直射、单跳RIS、双跳RIS等多种路径
- **分层调度**：全局RF充电 + 簇内MRC能量下发
- **地形感知**：基于DEM的视距判断和能量传输优化

---

## 🏗️ 系统架构

### 1. **核心组件层次**

```
┌─────────────────────────────────────────────────────────┐
│                    RF Transmitter (Sink)                │
│                    [远场能量发射源]                      │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   [直射路径]      [RIS反射]         [双跳RIS]
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
   [Cluster Head]                  [RIS Panel]
   [簇头 - 能量聚合点]              [可重构智能表面]
        │
        ▼
   [Sensor Nodes]
   [传感器节点 - MRC接收]
```

### 2. **文件结构**

```
src/
├── core/                          # 核心物理对象
│   ├── RFTransmitter.py          # RF发射机（汇聚节点）
│   ├── RIS.py                    # 可重构智能表面
│   ├── ClusterHead.py            # 簇头（能量聚合+下发）
│   ├── SensorNode.py             # 传感器节点
│   └── Environment.py            # 环境与地形管理
│
├── network/                       # 网络拓扑
│   ├── WSN.py                    # 整体网络初始化
│   └── Cluster.py                # 簇的定义和管理
│
├── scheduling/                    # 能量调度策略
│   └── scheduler.py              # RF充电目标选择 + MRC触发
│
├── routing/                       # 能量路由算法
│   └── routing_algorithm.py       # 多跳路径优化
│
├── utils/                         # 物理模型与工具
│   ├── rf_propagation_model.py   # RF远场传播（Friis模型）
│   ├── mrc_model.py              # MRC近场传输（逆幂律）
│   ├── scenario_loader.py        # CSV场景加载
│   └── simulation_logger.py       # 日志记录
│
├── config/
│   └── simulation_config.py       # 全局配置参数
│
└── simulation_main.py             # 主仿真循环
```

---

## ⚡ 能量传输调度逻辑

### 1. **调度器 (Scheduler)** - `src/scheduling/scheduler.py`

#### 核心策略：**能量感知调度**

```python
def schedule_power_transfer(wsn):
    """
    决策：
    1) 选择能量最低的簇头作为 RF 充电目标
    2) 能量充足（>20% 初始容量）的簇头执行 MRC 下发
    """
```

#### 调度决策流程：

```
┌─────────────────────────────────────────┐
│  每个时间步 (TIME_STEP_S = 1.0s)        │
└────────────────┬────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │ 1. 扫描所有簇头能量        │
    │    找最低能量的簇头        │
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │ 2. 将其设为 RF 充电目标    │
    │    (rf_target)             │
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │ 3. 筛选 MRC 发射簇头       │
    │    条件：能量 > 20% 初值   │
    │    且不是 rf_target        │
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │ 4. 返回调度决策            │
    │    {rf_target, mrc_txs}    │
    └────────────────────────────┘
```

#### 关键参数：

| 参数 | 值 | 含义 |
|------|-----|------|
| `ENABLE_MRC_LOCAL_TRANSFER` | True | 启用簇内MRC能量下发 |
| MRC阈值 | 20% | 簇头能量超过初值20%时才发送 |
| 排除条件 | rf_target | RF充电目标不同时进行MRC |

---

## 🛣️ 能量路由算法

### 1. **路由器 (Router)** - `src/routing/routing_algorithm.py`

#### 核心功能：**多路径能量传输优化**

```python
def find_optimal_energy_path(wsn, source, target_ch, max_hops=2):
    """
    探索所有可能的能量路径，返回最优路径和功率
    """
```

#### 支持的路径类型：

```
┌─────────────────────────────────────────────────────────┐
│           能量传输路径类型 (max_hops=2)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 路径0: 直射 (0 hops)                                    │
│ RF_Tx ──────────────────────► CH                       │
│                                                         │
│ 路径1: 单跳RIS (1 hop)                                  │
│ RF_Tx ──► RIS ──────────────► CH                       │
│                                                         │
│ 路径2: 双跳RIS (2 hops)                                 │
│ RF_Tx ──► RIS_i ──► RIS_j ──► CH                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 路由决策算法：

```
┌──────────────────────────────────────────┐
│ 输入: RF发射机, 目标簇头, RIS面板列表    │
└────────────┬─────────────────────────────┘
             │
             ▼
    ┌────────────────────────────┐
    │ 评估路径0: 直射             │
    │ P0 = calculate_received_   │
    │      rf_power(RF, CH, env) │
    │ max_power = P0             │
    │ best_path = [RF, CH]       │
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │ 评估路径1: 单跳RIS          │
    │ for each RIS:              │
    │   P1 = calculate_ris_      │
    │        assisted_power()    │
    │   if P1 > max_power:       │
    │     max_power = P1         │
    │     best_path = [RF,RIS,CH]│
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │ 评估路径2: 双跳RIS          │
    │ for RIS_i, RIS_j in perms: │
    │   P_at_j = ris_assisted()  │
    │   P2 = ris_assisted(j→CH)  │
    │   if P2 > max_power:       │
    │     max_power = P2         │
    │     best_path = [RF,i,j,CH]│
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │ 返回最优路径和功率         │
    │ (best_path, max_power)     │
    └────────────────────────────┘
```

#### 关键特性：

- **贪心选择**：每次选择功率最大的路径
- **视距检查**：所有路径段都需要LoS验证
- **RIS相位优化**：自动配置RIS相位以最大化反射增益
- **级联功率计算**：多跳路径逐段计算功率衰减

---

## 📡 物理传输模型

### 1. **RF远场传播** - `src/utils/rf_propagation_model.py`

#### 模型：Close-In (CI) 路径损耗模型

```
PL(d) = FSPL(f, d0) + 10·n·log₁₀(d/d0)
```

其中：
- **FSPL(f, d0)**：参考距离d0处的自由空间路径损耗
- **n**：路径损耗指数
  - LoS: n = 1.5 (良好视距)
  - NLoS: n = 3.5 (非视距)
- **d**：实际传输距离

#### 接收功率计算：

```python
def calculate_received_rf_power(tx, rx, env):
    """
    P_rx [dBm] = P_tx [dBm] + G_tx [dBi] + G_rx [dBi] - PL(d)
    """
    distance = ||tx.position - rx.position||
    is_los = env.check_los(tx.position, rx.position)
    
    P_rx_dbm = _log_distance_path_loss(
        tx.get_tx_power_dbm(),
        tx.antenna_gain_dbi,
        rx.rf_rx_gain_dbi,
        tx.frequency_hz,
        distance,
        is_los
    )
    return 10^((P_rx_dbm - 30) / 10)  # 转换为瓦特
```

#### 参数配置：

| 参数 | 值 | 含义 |
|------|-----|------|
| REFERENCE_DISTANCE | 1.0 m | 参考距离 |
| PATH_LOSS_EXPONENT_LOS | 1.5 | LoS路径损耗指数 |
| PATH_LOSS_EXPONENT_NLOS | 3.5 | NLoS路径损耗指数 |
| NLOS_EXTRA_LOSS_DB | 0.0 dB | 额外NLoS损耗 |

### 2. **RIS反射增益** - `src/core/RIS.py`

#### 相位配置（波束成形）：

```python
def configure_phases(incident_point, reflection_point):
    """
    对每个RIS单元 (m,n) 配置相位以对齐反射信号
    φ_mn = (2π/λ) · (d_in + d_out)
    """
```

#### 反射增益计算：

```
G_RIS [dBi] = 10·log₁₀(4π·A_aperture / λ²)
            = 10·log₁₀(4π·N·d_element² / λ²)
```

其中：
- **N** = 256 (16×16 单元)
- **d_element** = 0.5λ (单元间距)
- **λ** = c/f (波长)

#### RIS参数：

| 参数 | 值 | 含义 |
|------|-----|------|
| NUM_ELEMENTS_H | 16 | 水平单元数 |
| NUM_ELEMENTS_V | 16 | 竖直单元数 |
| ELEMENT_SPACING_FACTOR | 0.5 | 单元间距（波长倍数） |
| PHASE_RESOLUTION_BITS | 3 | 相位量化位数（8级） |

### 3. **MRC近场传输** - `src/utils/mrc_model.py`

#### 模型：逆幂律衰减

```
η(d) = η₀ · (d_ref / d)^γ
P_rx = P_tx · η(d)
```

其中：
- **η₀** = 0.8 (1米处效率)
- **γ** = 3.0 (衰减指数)
- **d_ref** = 1.0 m (参考距离)

#### 特点：
- **近场传输**：距离衰减极快（d³）
- **高效率**：短距离内效率可达80%
- **应用场景**：簇内传感器→簇头能量下发

---

## 🔄 仿真主循环

### 1. **时间步执行流程** - `src/simulation_main.py`

```
┌─────────────────────────────────────────────────────────┐
│           每个时间步 (TIME_STEP_S = 1.0s)               │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────┐
    │ a. 小时级信息上报                  │
    │    (每3600秒触发一次)              │
    │    - 传感器→簇头通信能耗           │
    │    - 距离相关的发射功率            │
    │    - 簇头接收能耗                  │
    └────────────┬───────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │ b. 全局太阳能采集与空闲衰减        │
    │    - 每个节点更新能量              │
    │    - 太阳能收集 (if enabled)       │
    │    - 空闲功耗衰减                  │
    └────────────┬───────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │ c. 簇内MRC能量下发                 │
    │    (Sensor → ClusterHead)          │
    │    - 检查传感器能量是否富足        │
    │    - 计算MRC接收功率               │
    │    - 更新簇头和传感器能量          │
    └────────────┬───────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │ d. 调度决策                        │
    │    schedule_power_transfer(wsn)    │
    │    返回: rf_target, mrc_txs        │
    └────────────┬───────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │ e. RF远场充电                      │
    │    (RF_Tx → rf_target)             │
    │    - 路由选择最优路径              │
    │    - 计算接收功率                  │
    │    - 更新簇头能量                  │
    └────────────┬───────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │ f. 能量历史记录                    │
    │    energy_history[node_id][t]      │
    │    用于后续分析和绘图              │
    └────────────┬───────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │ g. 节点死亡检测                    │
    │    if energy < MIN_ENERGY_J:       │
    │      print("Node died!")           │
    └────────────────────────────────────┘
```

### 2. **关键时间参数**

| 参数 | 值 | 含义 |
|------|-----|------|
| SIMULATION_TIME_S | 7200 s | 总仿真时间（2小时） |
| TIME_STEP_S | 1.0 s | 时间步长 |
| REPORT_INTERVAL | 3600 s | 信息上报间隔（1小时） |
| CROSS_CLUSTER_TRIGGER_PERIOD | 100 steps | 跨簇供能检查周期 |

---

## 🎯 能量流向总结

### 完整的能量传输链路：

```
┌──────────────────────────────────────────────────────────┐
│                  RF Transmitter (Sink)                   │
│                   [10W @ 100MHz]                         │
└────────────────────────┬─────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    [直射]          [RIS反射]         [双跳RIS]
    ~0.1W          ~0.05W            ~0.01W
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │   Cluster Head (CH)     │
            │  [接收RF能量]           │
            │  [聚合能量]             │
            │  [下发MRC能量]          │
            └────────────┬────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
   [Sensor Node 1]                  [Sensor Node 2]
   [MRC接收 ~0.1W]                  [MRC接收 ~0.1W]
   [太阳能采集]                      [太阳能采集]
   [传感/通信消耗]                   [传感/通信消耗]
```

### 能量流向决策树：

```
RF发射机能量
    │
    ├─► 直射到簇头 (最直接，损耗最小)
    │   └─► 簇头能量充足
    │       ├─► 启动MRC下发给传感器
    │       └─► 参与跨簇供能
    │
    ├─► 通过RIS反射 (增加覆盖范围)
    │   └─► 簇头能量充足
    │       └─► 同上
    │
    └─► 通过双跳RIS (覆盖远距离)
        └─► 簇头能量充足
            └─► 同上

传感器能量来源：
    ├─► 太阳能采集 (if has_solar && ENABLE_SOLAR)
    ├─► MRC接收 (from ClusterHead)
    └─► 初始能量 (0.5J)

簇头能量来源：
    ├─► RF远场充电 (from RF_Tx via routing)
    ├─► 跨簇供能 (from other ClusterHeads)
    └─► 初始能量 (1.0J)
```

---

## 📊 配置参数速查表

### RF发射机配置 (SinkConfig)
```python
TRANSMIT_POWER_W = 10.0          # 10瓦
FREQUENCY_HZ = 100e6            # 100MHz
ANTENNA_GAIN_DBI = 18.0         # 18dBi
```

### 簇头配置 (ClusterHeadConfig)
```python
INITIAL_ENERGY_J = 1.0           # 初始1焦耳
RF_RX_GAIN_DBI = 9.0            # RF接收增益
MRC_TX_POWER_W = 0.5            # MRC发射0.5瓦
```

### 传感器节点配置 (SensorNodeConfig)
```python
INITIAL_ENERGY_J = 0.5           # 初始0.5焦耳
ABUNDANT_THRESHOLD_J = 0.4       # 富足阈值
MRC_TX_POWER_W = 0.1            # MRC发射0.1瓦
```

### 调度参数 (SimConfig)
```python
ENABLE_ROUTING = True            # 启用路由
ENABLE_SCHEDULER = True          # 启用调度
ENABLE_MRC_LOCAL_TRANSFER = True # 启用簇内MRC
ENABLE_SOLAR = True              # 启用太阳能
```

---

## 🔍 关键算法伪代码

### 调度算法
```python
def schedule_power_transfer(wsn):
    # 1. 找最低能量簇头
    rf_target = min(wsn.clusters, 
                    key=lambda c: c.cluster_head.current_energy)
    
    # 2. 筛选MRC发射簇头
    mrc_transmitters = [
        ch for ch in wsn.clusters
        if ch.current_energy > 0.2 * INITIAL_ENERGY
        and ch != rf_target
    ]
    
    return {'rf_target': rf_target, 'mrc_transmitters': mrc_transmitters}
```

### 路由算法
```python
def find_optimal_energy_path(wsn, source, target_ch):
    best_path = []
    max_power = 0.0
    
    # 路径0: 直射
    power = calculate_received_rf_power(source, target_ch, env)
    if power > max_power:
        max_power = power
        best_path = [source, target_ch]
    
    # 路径1: 单跳RIS
    for ris in wsn.ris_panels:
        power = calculate_ris_assisted_power(source, ris, target_ch, env)
        if power > max_power:
            max_power = power
            best_path = [source, ris, target_ch]
    
    # 路径2: 双跳RIS
    for ris_i, ris_j in permutations(wsn.ris_panels, 2):
        power = calculate_ris_assisted_power(source, ris_i, ris_j, env)
        power = calculate_ris_assisted_power(ris_i, ris_j, target_ch, env)
        if power > max_power:
            max_power = power
            best_path = [source, ris_i, ris_j, target_ch]
    
    return best_path, max_power
```

---

## 🎓 总结

### 能量传输的三层架构：

1. **第一层：RF远场充电**
   - 从汇聚节点(Sink)到簇头
   - 支持多路径选择（直射/RIS）
   - 使用Friis路径损耗模型

2. **第二层：簇内MRC下发**
   - 从簇头到传感器节点
   - 近场磁共振耦合
   - 逆幂律衰减模型

3. **第三层：太阳能采集**
   - 传感器节点的补充能源
   - 日周期正弦模型
   - 全局开关控制

### 调度策略的核心：

- **贪心选择**：优先充电能量最低的簇头
- **能量阈值**：簇头能量>20%初值时才下发MRC
- **避免冲突**：RF目标不同时进行MRC发射
- **触发式供能**：跨簇供能基于能量水位触发

### 路由策略的核心：

- **多路径探索**：评估所有可能的传输路径
- **功率最大化**：选择接收功率最大的路径
- **视距验证**：所有路径段都需要LoS检查
- **RIS波束成形**：自动配置相位以最大化反射增益

这个系统通过**分层设计**和**智能调度**，在复杂环境中实现了高效的能量传输和网络自供电。





