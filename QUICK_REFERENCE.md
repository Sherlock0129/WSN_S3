# WSN_S3 快速参考卡片

## 🚀 核心概念速记

### 三层能量传输架构

```
┌─────────────────────────────────────────┐
│ 第1层: RF远场充电 (RF_Tx → CH)          │
│ 模型: Friis/CI路径损耗                  │
│ 距离衰减: d²(LoS) / d³.5(NLoS)         │
│ 最大距离: 1000+ m                       │
└─────────────────────────────────────────┘
           ↓ (通过RIS优化)
┌─────────────────────────────────────────┐
│ 第2层: 簇内MRC下发 (CH → Sensor)        │
│ 模型: 磁共振耦合                        │
│ 距离衰减: d³                            │
│ 最大距离: 10 m                          │
└─────────────────────────────────────────┘
           ↓ (补充能源)
┌─────────────────────────────────────────┐
│ 第3层: 太阳能采集 (Sensor)              │
│ 模型: 日周期正弦                        │
│ 日采集: 40 kJ                           │
│ 时间: 6:00-18:00                        │
└─────────────────────────────────────────┘
```

---

## 📂 文件导航

### 关键文件速查

| 功能 | 文件 | 关键函数 |
|------|------|---------|
| **调度** | `src/scheduling/scheduler.py` | `schedule_power_transfer(wsn)` |
| **路由** | `src/routing/routing_algorithm.py` | `find_optimal_energy_path(wsn, source, target_ch, max_hops=2)` |
| **RF模型** | `src/utils/rf_propagation_model.py` | `calculate_received_rf_power(tx, rx, env)` |
| **RIS模型** | `src/core/RIS.py` | `configure_phases()`, `get_reflection_gain()` |
| **MRC模型** | `src/utils/mrc_model.py` | `calculate_received_mrc_power(tx_node, rx_node)` |
| **主循环** | `src/simulation_main.py` | `run_simulation()` |
| **网络初始化** | `src/network/WSN.py` | `WSN.__init__()` |
| **场景加载** | `src/utils/scenario_loader.py` | `load_scenario()`, `build_dem_from_S3()` |

---

## ⚙️ 调度策略速记

### 调度器决策流程

```python
# 1. 找最低能量簇头
rf_target = min(clusters, key=lambda c: c.cluster_head.current_energy)

# 2. 筛选MRC发射簇头
mrc_transmitters = [
    ch for ch in clusters
    if ch.current_energy > 0.2 * INITIAL_ENERGY  # 20%阈值
    and ch != rf_target  # 避免冲突
]

return {'rf_target': rf_target, 'mrc_transmitters': mrc_transmitters}
```

### 关键参数

| 参数 | 值 | 含义 |
|------|-----|------|
| MRC触发阈值 | 20% | 簇头能量超过初值20%时才发送 |
| 传感器富足阈值 | 0.4 J | 传感器能量超过此值时参与MRC |
| 调度周期 | 每步 | 每个时间步都重新调度 |

---

## 🛣️ 路由策略速记

### 路由决策流程

```python
# 评估所有可能的路径
paths = [
    ("直射", calculate_received_rf_power(RF, CH, env)),
    ("RIS_0", calculate_ris_assisted_power(RF, RIS_0, CH, env)),
    ("RIS_1", calculate_ris_assisted_power(RF, RIS_1, CH, env)),
    ("RIS_0→RIS_1", calculate_ris_assisted_power(RF, RIS_0, RIS_1, env) 
                    + calculate_ris_assisted_power(RIS_1, CH, env))
]

# 选择功率最大的路径
best_path, max_power = max(paths, key=lambda x: x[1])

return best_path, max_power
```

### 关键参数

| 参数 | 值 | 含义 |
|------|-----|------|
| 最大跳数 | 2 | 支持直射、单跳、双跳 |
| RIS单元数 | 256 | 16×16阵列 |
| RIS增益 | 29 dBi | 约800倍功率放大 |
| 相位量化 | 3 bit | 8个离散相位级别 |

---

## 📡 物理模型速记

### RF远场 (Close-In模型)

```
PL(d) = FSPL(f,d₀) + 10·n·log₁₀(d/d₀)

参数:
  f = 100 MHz
  d₀ = 1 m
  n_LoS = 1.5
  n_NLoS = 3.5
  
例: d=1414m, LoS
  PL = 12.45 + 15×3.15 = 59.7 dB
  P_rx = 40 + 18 + 9 - 59.7 = 7.3 dBm = 5.4 mW
```

### RIS反射

```
G_RIS = 10·log₁₀(4π·A_aperture / λ²)
      = 10·log₁₀(4π·256·1.5² / 3²)
      = 29 dBi

相位配置:
  φ_mn = (2π/λ) × (d_in + d_out)
  量化: 3 bit → 8级 → 45°间隔
```

### MRC近场

```
η(d) = 0.8 × (1/d)³

例: d=30m, P_tx=0.1W
  η = 0.8 × (1/30)³ = 2.96×10⁻⁵
  P_rx = 0.1 × 2.96×10⁻⁵ = 2.96 μW
```

### 太阳能

```
G(t) = 800 × sin(π(t-360)/720)  [360≤t≤1080分钟]

P_harvest = 0.18 × 0.001 × G(t) × 0.6
          = 1.08×10⁻⁴ × G(t)

日采集: ≈ 40 kJ
```

---

## 🔋 能量消耗速记

### 消耗速率

| 操作 | 功率 | 时间 | 消耗 |
|------|------|------|------|
| RF发射 | 10 W | 1 s | 10 J |
| CH MRC | 0.5 W | 1 s | 0.5 J |
| 传感器MRC | 0.1 W | 1 s | 0.1 J |
| 上报 | - | 1次/h | 0.0003 J |
| 太阳能(正午) | 0.0864 W | 1 s | 0.0864 J |

### 能量平衡

```
传感器(有太阳):
  日收入: 40 kJ
  日消耗: 0.0072 J
  结果: ✓ 充足

传感器(无太阳):
  日收入: 0 J
  日消耗: 0.0072 J
  结果: ✗ 枯竭

簇头:
  日收入: 466 J (RF)
  日消耗: 43200 J (MRC)
  结果: ✗ 严重不足
```

---

## 🎯 配置修改指南

### 增加RF功率

```python
# src/config/simulation_config.py
class SinkConfig:
    TRANSMIT_POWER_W = 50.0  # 从10W改为50W
```

**效果：** 接收功率增加5倍

### 增加RIS数量

```python
# src/data/sink.csv
# 添加更多RIS条目
```

**效果：** 提高覆盖范围和路径选择

### 调整MRC阈值

```python
# src/scheduling/scheduler.py
mrc_threshold = ClusterHeadConfig.INITIAL_ENERGY_J * 0.3  # 从0.2改为0.3
```

**效果：** 更早触发MRC下发

### 启用太阳能

```python
# src/config/simulation_config.py
class WSNConfig:
    ENABLE_SOLAR = True  # 全局开关
```

**效果：** 传感器获得额外能源

---

## 🔍 调试技巧

### 打印调度决策

```python
# 在 src/simulation_main.py 中添加
schedule_result = schedule_power_transfer(wsn)
print(f"RF Target: {schedule_result['rf_target']}")
print(f"MRC Transmitters: {schedule_result['mrc_transmitters']}")
```

### 打印路由结果

```python
# 在路由调用后添加
best_path, max_power = find_optimal_energy_path(wsn, source, target_ch)
print(f"Best Path: {[obj.node_id for obj in best_path]}")
print(f"Max Power: {max_power*1000:.2f} mW")
```

### 打印能量历史

```python
# 在仿真结束后
for node_id in node_ids:
    final_energy = energy_history[node_id][-1]
    print(f"Node {node_id}: {final_energy:.4f} J")
```

### 启用详细日志

```python
# src/config/simulation_config.py
class SimConfig:
    ENABLE_LOGGING = True
    ENABLE_PLOT_RESULTS = True
```

---

## 📊 性能指标

### 关键指标

| 指标 | 计算方法 | 目标值 |
|------|---------|--------|
| **网络寿命** | 最后一个节点死亡的时间 | > 7200 s |
| **能量利用率** | 收获能量 / 消耗能量 | > 1.0 |
| **覆盖率** | 能接收RF的簇头数 / 总簇头数 | > 80% |
| **平均功率** | 总接收功率 / 时间 | > 1 mW |

### 快速评估

```python
# 在仿真结束后
total_harvested = sum(energy_history[node_id][-1] for node_id in node_ids)
total_consumed = sum(INITIAL_ENERGY - energy_history[node_id][-1] 
                     for node_id in node_ids)
efficiency = total_harvested / total_consumed if total_consumed > 0 else 0

print(f"Energy Efficiency: {efficiency:.2%}")
print(f"Network Lifetime: {num_steps * TIME_STEP_S:.0f} s")
```

---

## 🔗 常见问题速答

### Q: 为什么节点快速死亡？
**A:** 
1. RF功率太低 → 增加 TRANSMIT_POWER_W
2. RIS数量不足 → 添加更多RIS面板
3. MRC距离太远 → 减小簇半径 CLUSTER_RADIUS
4. 没有太阳能 → 启用 ENABLE_SOLAR

### Q: 为什么路由总是选直射？
**A:**
1. RIS位置不优 → 调整RIS坐标
2. RIS增益不足 → 增加RIS单元数
3. 相位量化误差大 → 增加 PHASE_RESOLUTION_BITS
4. 双跳损耗太大 → 禁用 ENABLE_DOUBLE_HOP

### Q: 为什么MRC没有效果？
**A:**
1. 距离太远 → MRC只在10m内有效
2. 能量不足 → 增加RF功率
3. 阈值设置不当 → 调整 ABUNDANT_THRESHOLD_J
4. 簇头能量低 → 增加 INITIAL_ENERGY_J

### Q: 如何优化能量平衡？
**A:**
1. 增加RF功率 (10W → 50W)
2. 增加RIS数量 (2 → 4+)
3. 优化RIS位置 (高地优先)
4. 启用太阳能 (ENABLE_SOLAR=True)
5. 减少消耗 (降低上报频率)

---

## 📈 实验建议

### 基准实验

```python
# 实验1: 基础配置
TRANSMIT_POWER_W = 10.0
NUM_CLUSTERS = 6
ENABLE_SOLAR = False
ENABLE_MRC_LOCAL_TRANSFER = True
ENABLE_ROUTING = True
```

### 对比实验

```python
# 实验2: 增加RF功率
TRANSMIT_POWER_W = 50.0  # 5倍

# 实验3: 增加RIS数量
# 在sink.csv中添加RIS条目

# 实验4: 启用太阳能
ENABLE_SOLAR = True

# 实验5: 禁用MRC
ENABLE_MRC_LOCAL_TRANSFER = False
```

### 记录指标

```
实验 | RF功率 | RIS数 | 太阳能 | 网络寿命 | 效率 | 覆盖率
----|--------|-------|--------|---------|------|-------
1   | 10W    | 2     | No     | ?       | ?    | ?
2   | 50W    | 2     | No     | ?       | ?    | ?
3   | 10W    | 4     | No     | ?       | ?    | ?
4   | 10W    | 2     | Yes    | ?       | ?    | ?
5   | 10W    | 2     | No(MRC)| ?       | ?    | ?
```

---

## 🚀 快速启动

### 1. 运行基础仿真

```bash
cd src
python simulation_main.py
```

### 2. 查看结果

```bash
# 能量曲线图
# 日志文件: src/simulation_logs/

# 检查最后一行
tail -20 src/simulation_logs/simulation_*.log
```

### 3. 修改配置

```python
# 编辑 src/config/simulation_config.py
# 改变参数后重新运行
```

### 4. 对比结果

```bash
# 保存不同配置的日志
cp src/simulation_logs/energy_*.csv results/exp1_energy.csv
# 修改配置后再运行
cp src/simulation_logs/energy_*.csv results/exp2_energy.csv
# 用Python对比
```

---

## 📚 学习路径

### 初级 (理解基础)
1. 阅读 PROJECT_ARCHITECTURE_CN.md
2. 运行 simulation_main.py
3. 查看能量曲线图

### 中级 (理解机制)
1. 阅读 ENERGY_FLOW_DETAILED.md
2. 修改调度参数，观察效果
3. 修改路由参数，观察效果

### 高级 (优化系统)
1. 阅读 PHYSICS_MODELS_REFERENCE.md
2. 设计对比实验
3. 优化RF功率、RIS位置、簇配置

---

## 📞 关键联系点

### 调度器入口
```
src/scheduling/scheduler.py:schedule_power_transfer()
```

### 路由器入口
```
src/routing/routing_algorithm.py:find_optimal_energy_path()
```

### 主循环入口
```
src/simulation_main.py:run_simulation()
```

### 配置中心
```
src/config/simulation_config.py
```

---

**最后更新:** 2025-12-10
**版本:** 1.0





