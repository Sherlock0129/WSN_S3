# WSN_S3 能量流动详细执行流程

## 📍 仿真主循环的完整执行流程

### 初始化阶段 (main.py → simulation_main.py)

```python
# 1. 创建WSN网络
wsn = WSN()
    ├─ 加载场景数据 (S3.csv, sink.csv)
    ├─ 创建RF发射机 (RFTransmitter)
    ├─ 创建RIS面板列表 (RIS panels)
    ├─ 创建簇及其内部传感器节点 (Clusters + SensorNodes)
    └─ 地形感知放置 (_place_nodes_on_terrain)

# 2. 初始化日志记录器
logger = SimulationLogger()

# 3. 准备数据存储
energy_history = {node_id: zeros(num_steps) for node_id in node_ids}
num_steps = SIMULATION_TIME_S / TIME_STEP_S = 7200 / 1.0 = 7200
```

---

## ⏰ 每个时间步的执行顺序

### 时间步 t (从 0 到 7199)

```
current_time = t * TIME_STEP_S = t * 1.0 秒
current_time_min = (current_time % 86400) / 60 = 分钟级时间 (用于太阳能计算)
```

---

### 第1步：小时级信息上报 (每3600秒触发)

```python
if current_time > 0 and current_time % 3600 == 0:
    # 触发时刻: t=3600, 7200, ...
    
    for cluster in wsn.clusters:
        ch = cluster.cluster_head
        for sensor in cluster.sensor_nodes:
            # 计算上报能耗
            d = sensor.distance_to(ch)
            E_tx_dist = E_elec * B + epsilon * B * (d ** tau)
            E_total_sensor = tx_e_base + E_tx_dist
            
            # 从传感器扣除能量
            e_s = min(E_total_sensor, sensor.current_energy)
            sensor.current_energy -= e_s
            
            # 可选：簇头接收能耗
            if REPORT_INCLUDE_CH_RX:
                E_rx = E_elec * B
                ch.current_energy -= min(E_rx, ch.current_energy)
    
    # 记录日志
    logger.log_energy_transfer(
        rf_target=None,
        rf_sent_energy_j=None,
        rf_delivered_energy_j=None,
        mrc_entries=[],
        sensor_tx_consumption=hourly_consumption
    )
```

**能量变化示例：**
```
传感器初始能量: 0.5 J
距离簇头: 30 m
E_tx_dist = 50e-9 * 4000 + 10e-12 * 4000 * (30^2)
         = 200e-6 + 36e-6 = 236e-6 J
E_total = 100e-6 + 236e-6 = 336e-6 J

传感器能量: 0.5 - 0.000336 = 0.499664 J
```

---

### 第2步：全局能量更新 (太阳能采集 + 空闲衰减)

```python
current_time_min = (current_time % 86400) / 60  # 分钟级时间

for node in all_nodes:
    # 更新太阳能采集开关
    if hasattr(node, 'enable_energy_harvesting'):
        node.enable_energy_harvesting = (
            node.has_solar and WSNConfig.ENABLE_SOLAR
        )
    
    # 调用能量更新函数
    node.update_energy(current_time_min)
```

**update_energy 函数执行：**

```python
def update_energy(self, t):
    # t: 分钟级时间 (0-1440)
    
    # 1. 计算太阳能采集
    E_gen = self.energy_harvest(t)
        # 如果 360 <= t <= 1080 (6:00-18:00):
        #   G_t = G_max * sin(π * (t-360) / 720)
        #   power_w = solar_efficiency * solar_area * G_t * env_correction
        #   E_gen = power_w * TIME_STEP_S
        # 否则: E_gen = 0
    
    # 2. 计算空闲衰减
    E_decay = IDLE_POWER_W * TIME_STEP_S
        # 默认 IDLE_POWER_W = 0.0，所以 E_decay = 0
    
    # 3. 更新能量
    self.current_energy = self.current_energy + E_gen - E_decay
    
    # 4. 限制在有效范围内
    self.current_energy = max(0, min(self.current_energy, capacity_max))
    
    # 5. 记录历史
    self.energy_history.append({
        "time": t,
        "generated": E_gen,
        "consumed": E_decay
    })
```

**太阳能采集示例 (假设 t=720 分钟 = 中午12:00)：**

```
G_t = 800 * sin(π * (720-360) / 720)
    = 800 * sin(π/2)
    = 800 * 1.0
    = 800 W/m²

power_w = 0.18 * 0.001 * 800 * 0.6
        = 0.0864 W

E_gen = 0.0864 * 1.0 = 0.0864 J (每秒)

在1小时内累积: 0.0864 * 3600 = 310.6 J
```

---

### 第3步：簇内MRC能量下发 (Sensor → ClusterHead)

```python
abundant_threshold = SensorNodeConfig.ABUNDANT_THRESHOLD_J = 0.4 J

for cluster in wsn.clusters:
    ch = cluster.cluster_head
    for sensor in cluster.sensor_nodes:
        # 检查传感器能量是否富足
        if sensor.current_energy > abundant_threshold:
            # 计算本步长内的发送能量
            intended_send_j = sensor.mrc_tx_power_w * TIME_STEP_S
                            = 0.1 * 1.0 = 0.1 J
            
            # 计算可发送的能量 (不超过富余部分)
            surplus_j = sensor.current_energy - abundant_threshold
            energy_to_send_j = min(intended_send_j, surplus_j)
            
            if energy_to_send_j > 0:
                # 1. 传感器扣除能量
                sensor.current_energy -= energy_to_send_j
                sensor.record_transfer(transferred=energy_to_send_j)
                
                # 2. 计算簇头接收功率
                actual_tx_power_w = energy_to_send_j / TIME_STEP_S
                received_power_w = mrc_model.calculate_received_mrc_power(
                    sensor, ch, tx_power_w=actual_tx_power_w
                )
                
                # 3. 簇头接收能量
                ch.receive_mrc_power(received_power_w, TIME_STEP_S)
```

**MRC功率计算示例：**

```
传感器位置: (100, 100, 10)
簇头位置: (130, 100, 10)
距离: d = 30 m

MRC效率计算:
η(d) = η₀ * (d_ref / d)^γ
     = 0.8 * (1.0 / 30)^3
     = 0.8 * 3.7e-5
     = 2.96e-5

发射功率: P_tx = 0.1 W
接收功率: P_rx = 0.1 * 2.96e-5 = 2.96e-6 W

接收能量: E_rx = 2.96e-6 * 1.0 = 2.96e-6 J (极小)
```

**关键观察：** MRC距离衰减极快（d³），所以只有近距离传输才有效。

---

### 第4步：调度决策 (Scheduler)

```python
schedule_result = schedule_power_transfer(wsn)
rf_target = schedule_result['rf_target']
mrc_transmitters = schedule_result['mrc_transmitters']
```

**调度算法执行：**

```python
def schedule_power_transfer(wsn):
    # 1. 找最低能量的簇头
    lowest_energy_ch = None
    min_energy_j = float('inf')
    
    for cluster in wsn.clusters:
        ch = cluster.cluster_head
        if ch.current_energy < min_energy_j:
            min_energy_j = ch.current_energy
            lowest_energy_ch = ch
    
    rf_target = lowest_energy_ch
    
    # 2. 筛选MRC发射簇头
    mrc_transmitters = []
    mrc_threshold = ClusterHeadConfig.INITIAL_ENERGY_J * 0.2 = 1.0 * 0.2 = 0.2 J
    
    for cluster in wsn.clusters:
        ch = cluster.cluster_head
        if ch is rf_target:
            continue  # 排除RF目标
        if ch.current_energy > mrc_threshold:
            mrc_transmitters.append(ch)
    
    return {
        'rf_target': rf_target,
        'mrc_transmitters': mrc_transmitters
    }
```

**调度决策示例：**

```
簇头能量状态:
  CH_0: 0.15 J  ← 最低 (rf_target)
  CH_1: 0.85 J  ✓ (能量充足，加入mrc_transmitters)
  CH_2: 0.25 J  ✓ (能量充足，加入mrc_transmitters)
  CH_3: 0.10 J  ✗ (能量不足)
  CH_4: 0.95 J  ✓ (能量充足，加入mrc_transmitters)
  CH_5: 0.18 J  ✗ (能量不足)

结果:
  rf_target = CH_0
  mrc_transmitters = [CH_1, CH_2, CH_4]
```

---

### 第5步：RF远场充电 (RF_Tx → rf_target)

```python
if WSNConfig.ENABLE_ROUTING and rf_target is not None:
    # 1. 调用路由算法找最优路径
    best_path, max_power_w = routing_algorithm.find_optimal_energy_path(
        wsn, wsn.rf_transmitter, rf_target, max_hops=2
    )
    
    # 2. 簇头接收RF能量
    if max_power_w > 0:
        rf_target.receive_rf_power(max_power_w, TIME_STEP_S)
        
        # 记录日志
        logger.log_energy_transfer(
            rf_target=rf_target,
            rf_sent_energy_j=wsn.rf_transmitter.power_w * TIME_STEP_S,
            rf_delivered_energy_j=max_power_w * TIME_STEP_S,
            mrc_entries=[],
            sensor_tx_consumption={}
        )
```

**路由算法执行 (find_optimal_energy_path)：**

```python
def find_optimal_energy_path(wsn, source, target_ch, max_hops=2):
    env = wsn.environment
    ris_panels = wsn.ris_panels
    
    best_path = []
    max_power = 0.0
    
    # ===== 路径0: 直射 =====
    direct_power = calculate_received_rf_power(source, target_ch, env)
    # 计算过程:
    #   distance = ||RF_pos - CH_pos||
    #   is_los = env.check_los(RF_pos, CH_pos)
    #   P_rx_dbm = _log_distance_path_loss(
    #       tx_power_dbm=10*log10(10*1000)=40dBm,
    #       tx_gain_dbi=18,
    #       rx_gain_dbi=9,
    #       frequency_hz=100e6,
    #       distance_m=distance,
    #       is_los=is_los
    #   )
    #   P_rx_w = 10^((P_rx_dbm-30)/10)
    
    if direct_power > max_power:
        max_power = direct_power
        best_path = [source, target_ch]
    
    # ===== 路径1: 单跳RIS =====
    if max_hops >= 1:
        for ris in ris_panels:
            power = calculate_ris_assisted_power(source, ris, target_ch, env)
            # 计算过程:
            #   1. RF → RIS 的功率
            #      dist_source_ris = ||RF_pos - RIS_pos||
            #      is_los = env.check_los(RF_pos, RIS_pos)
            #      power_at_ris_dbm = _log_distance_path_loss(...)
            #
            #   2. RIS 配置相位
            #      ris.configure_phases(RF_pos, CH_pos)
            #      计算每个单元的相位: φ_mn = (2π/λ) * (d_in + d_out)
            #
            #   3. RIS → CH 的功率
            #      ris_gain_dbi = ris.get_reflection_gain()
            #      received_power_dbm = _log_distance_path_loss(
            #          power_at_ris_dbm,
            #          ris_gain_dbi,
            #          ch.rf_rx_gain_dbi,
            #          ...
            #      )
            #      power = 10^((received_power_dbm-30)/10)
            
            if power > max_power:
                max_power = power
                best_path = [source, ris, target_ch]
    
    # ===== 路径2: 双跳RIS =====
    if max_hops >= 2 and len(ris_panels) >= 2:
        for ris_i, ris_j in itertools.permutations(ris_panels, 2):
            # RF → RIS_i → RIS_j → CH
            power_at_ris_j = calculate_ris_assisted_power(source, ris_i, ris_j, env)
            
            if power_at_ris_j > 0:
                # 将RIS_j的接收功率作为新的发射源
                final_power_w = calculate_ris_assisted_power(
                    ris_j_as_source, ris_j, target_ch, env
                )
                
                if final_power_w > max_power:
                    max_power = final_power_w
                    best_path = [source, ris_i, ris_j, target_ch]
    
    return best_path, max_power
```

**RF功率计算示例 (假设直射路径)：**

```
RF发射机参数:
  位置: (0, 0, 100)
  功率: 10 W = 40 dBm
  频率: 100 MHz
  增益: 18 dBi

簇头参数:
  位置: (1000, 1000, 100)
  增益: 9 dBi

计算:
  distance = sqrt(1000² + 1000²) = 1414.2 m
  is_los = True (假设视距)
  
  λ = 3e8 / 100e6 = 3 m
  
  FSPL_d0 = 20*log10(100e6) + 20*log10(1) - 147.55
          = 160 - 147.55 = 12.45 dB
  
  PL = 12.45 + 10*1.5*log10(1414.2/1)
     = 12.45 + 15*3.15
     = 12.45 + 47.25 = 59.7 dB
  
  P_rx_dbm = 40 + 18 + 9 - 59.7 = 7.3 dBm
  P_rx_w = 10^((7.3-30)/10) = 10^(-2.27) = 0.0054 W = 5.4 mW
```

**接收能量：**
```
E_rx = 0.0054 * 1.0 = 0.0054 J (每秒)
```

---

### 第6步：跨簇供能 (可选)

```python
if SimConfig.ENABLE_CROSS_CLUSTER_DONATION:
    if t_step % CROSS_CLUSTER_TRIGGER_PERIOD_STEPS == 0:
        # 每100步检查一次
        
        for cluster in wsn.clusters:
            ch = cluster.cluster_head
            
            # 检查能量是否过低
            if ch.current_energy < TRIGGER_LOW_PCT * INITIAL_ENERGY:
                # 寻找能量充足的簇头进行供能
                for donor_cluster in wsn.clusters:
                    donor_ch = donor_cluster.cluster_head
                    if donor_ch.current_energy > TRIGGER_HIGH_PCT * INITIAL_ENERGY:
                        # 进行跨簇供能
                        # 使用RF远场传输模型
                        ...
```

---

### 第7步：能量历史记录

```python
for i, node in enumerate(all_nodes):
    energy_history[node.node_id][t_step] = node.current_energy
    
    # 检查节点是否死亡
    if node.current_energy < SensorNodeConfig.MIN_ENERGY_J:
        print(f"!!! Node {node.node_id} has died at {current_time}s !!!")
```

---

## 📊 完整的能量流向示例 (单个时间步)

假设仿真参数：
- 6个簇，每簇10个传感器
- RF功率: 10W @ 100MHz
- 时间步: 1秒
- 当前时间: t=3600秒 (1小时，中午12:00)

### 初始状态：

```
RF发射机:
  位置: (0, 0, 100)
  功率: 10 W

簇头能量:
  CH_0: 0.15 J  ← 最低
  CH_1: 0.85 J
  CH_2: 0.25 J
  CH_3: 0.10 J
  CH_4: 0.95 J
  CH_5: 0.18 J

传感器能量 (示例CH_0的10个传感器):
  S_0_0: 0.45 J
  S_0_1: 0.38 J
  ...
  S_0_9: 0.42 J
```

### 执行流程：

#### 1️⃣ 小时级上报 (t=3600秒触发)

```
每个传感器上报能耗: ~0.0003 J
总消耗: 6簇 × 10传感器 × 0.0003 = 0.018 J

CH_0 能量: 0.15 - 0.0003*10 = 0.147 J
```

#### 2️⃣ 太阳能采集 (中午12:00)

```
太阳辐照度: G_t = 800 W/m² (正午最大)
采集功率: 0.0864 W
采集能量 (1秒): 0.0864 J

有太阳能的传感器:
  S_0_0: 0.45 + 0.0864 = 0.5364 J
  
无太阳能的传感器:
  S_0_1: 0.38 J (不变)
```

#### 3️⃣ 簇内MRC下发

```
CH_1 (能量0.85J > 0.4J阈值):
  MRC发射功率: 0.5 W
  发送能量: 0.5 * 1.0 = 0.5 J
  
  传感器接收 (距离30m):
    η = 0.8 * (1/30)³ = 2.96e-5
    P_rx = 0.5 * 2.96e-5 = 1.48e-5 W
    E_rx = 1.48e-5 * 1.0 = 1.48e-5 J (极小)
  
  CH_1 能量: 0.85 - 0.5 = 0.35 J
  传感器能量: 增加 1.48e-5 J (可忽略)
```

#### 4️⃣ 调度决策

```
rf_target = CH_0 (能量最低: 0.147 J)
mrc_transmitters = [CH_1, CH_2, CH_4]
  (能量 > 0.2J 且不是 rf_target)
```

#### 5️⃣ RF远场充电

```
路由选择:
  路径0 (直射): P = 5.4 mW
  路径1 (RIS_0): P = 2.1 mW
  路径1 (RIS_1): P = 1.8 mW
  路径2 (RIS_0→RIS_1): P = 0.3 mW
  
最优路径: 直射 (5.4 mW)

CH_0 接收:
  E_rx = 0.0054 * 1.0 = 0.0054 J
  
CH_0 能量: 0.147 + 0.0054 = 0.1524 J
```

#### 6️⃣ 最终状态

```
RF发射机:
  能量消耗: 10 * 1.0 = 10 J (假设无限能量)

簇头能量:
  CH_0: 0.1524 J (充电后)
  CH_1: 0.35 J (MRC消耗后)
  CH_2: 0.25 J (不变)
  CH_3: 0.10 J (不变)
  CH_4: 0.95 J (不变)
  CH_5: 0.18 J (不变)

传感器能量:
  有太阳能: +0.0864 J
  无太阳能: 不变
  MRC接收: +1.48e-5 J (可忽略)
```

---

## 🔑 关键数值总结

### 能量消耗速率

| 操作 | 能量消耗 | 时间 | 总消耗 |
|------|---------|------|--------|
| 传感器上报 | 336 μJ | 1次/小时 | 336 μJ |
| 传感器MRC发射 | 0.1 W | 1秒 | 0.1 J |
| 簇头MRC发射 | 0.5 W | 1秒 | 0.5 J |
| RF发射 | 10 W | 1秒 | 10 J |

### 能量收获速率

| 来源 | 功率 | 时间 | 总收获 |
|------|------|------|--------|
| 太阳能 (正午) | 0.0864 W | 1秒 | 0.0864 J |
| RF直射 | 5.4 mW | 1秒 | 0.0054 J |
| RF+RIS | 2.1 mW | 1秒 | 0.0021 J |
| MRC (30m) | 1.48e-5 W | 1秒 | 1.48e-5 J |

### 能量平衡

```
传感器 (有太阳能):
  收入: 0.0864 J (太阳) + 1.48e-5 J (MRC) = 0.0864 J
  支出: 0.0003 J (上报) = 0.0003 J
  净增: +0.0861 J ✓ (能量充足)

传感器 (无太阳能):
  收入: 1.48e-5 J (MRC) ≈ 0 J
  支出: 0.0003 J (上报) = 0.0003 J
  净增: -0.0003 J ✗ (能量枯竭)

簇头:
  收入: 0.0054 J (RF) = 0.0054 J
  支出: 0.5 J (MRC) = 0.5 J
  净增: -0.4946 J ✗ (能量枯竭)
```

**结论：** 系统需要更频繁的RF充电或更高的RF功率才能维持能量平衡。

---

## 🎯 优化建议

1. **增加RF功率**：从10W增加到20-50W
2. **增加RIS数量**：提高覆盖范围和功率
3. **优化RIS位置**：放在高地以改善LoS
4. **减少传感器消耗**：降低上报频率或数据量
5. **增加太阳能面积**：提高单个节点的采集功率





