# WSN_S3 项目分析总结

## 📖 文档导航

本分析包含以下4份详细文档，建议按顺序阅读：

### 1. **PROJECT_ARCHITECTURE_CN.md** ⭐ 从这里开始
   - **内容**：项目整体架构、系统设计、三层能量传输体系
   - **适合**：快速理解项目全貌
   - **阅读时间**：15-20分钟
   - **关键章节**：
     - 系统架构（文件结构、组件层次）
     - 能量传输调度逻辑（调度器工作原理）
     - 能量路由算法（路由器工作原理）
     - 物理传输模型概览

### 2. **ENERGY_FLOW_DETAILED.md** 深入细节
   - **内容**：每个时间步的完整执行流程、能量流向示例
   - **适合**：理解代码执行细节
   - **阅读时间**：20-30分钟
   - **关键章节**：
     - 初始化阶段
     - 7个时间步执行顺序（小时级上报、太阳能、MRC、调度、RF充电等）
     - 完整的能量流向示例（数值计算）
     - 关键数值总结

### 3. **PHYSICS_MODELS_REFERENCE.md** 物理模型详解
   - **内容**：RF、RIS、MRC、太阳能等物理模型的详细公式和参数
   - **适合**：理解物理模型和参数含义
   - **阅读时间**：25-35分钟
   - **关键章节**：
     - RF远场传播（Close-In模型）
     - RIS反射增益和相位配置
     - MRC近场传输
     - 太阳能采集模型
     - 能量消耗模型
     - 参数对比表

### 4. **QUICK_REFERENCE.md** 快速查询
   - **内容**：速记卡片、快速参考、常见问题、调试技巧
   - **适合**：快速查询和日常参考
   - **阅读时间**：5-10分钟（按需查询）
   - **关键章节**：
     - 核心概念速记
     - 文件导航
     - 调度/路由策略速记
     - 物理模型速记
     - 常见问题速答
     - 快速启动指南

---

## 🎯 核心发现总结

### 1. 系统架构的三层设计

```
第1层: RF远场充电 (1000+ m)
  ├─ 直射路径
  ├─ 单跳RIS反射
  └─ 双跳RIS反射
       ↓
第2层: 簇内MRC下发 (10 m)
  ├─ 簇头→传感器
  └─ 传感器→簇头
       ↓
第3层: 太阳能采集 (全天)
  └─ 补充能源
```

**特点**：
- 分层设计实现了从远到近的能量传输
- 每层采用不同的物理模型和参数
- 通过调度和路由实现智能决策

### 2. 调度策略的核心逻辑

```
调度器 (scheduler.py)
  ├─ 选择最低能量簇头 → RF充电目标
  ├─ 筛选能量充足簇头 (>20%) → MRC发射源
  └─ 避免冲突 (rf_target不参与MRC)
```

**特点**：
- **贪心策略**：优先充电最需要的节点
- **阈值机制**：能量>20%才能下发
- **冲突避免**：RF目标不同时进行MRC

### 3. 路由策略的核心逻辑

```
路由器 (routing_algorithm.py)
  ├─ 评估路径0: 直射
  ├─ 评估路径1: 单跳RIS
  ├─ 评估路径2: 双跳RIS
  └─ 选择功率最大的路径
```

**特点**：
- **多路径探索**：考虑所有可能的传输路径
- **功率最大化**：选择接收功率最大的路径
- **LoS验证**：所有路径段都需要视距检查
- **RIS优化**：自动配置相位以最大化反射增益

### 4. 物理模型的选择

| 传输方式 | 模型 | 距离衰减 | 应用 |
|---------|------|---------|------|
| RF远场 | Friis/CI | d²(LoS)/d³.5(NLoS) | 远距离充电 |
| RIS反射 | 孔径近似 | 29 dBi增益 | 覆盖优化 |
| MRC近场 | 逆幂律 | d³ | 近距离下发 |
| 太阳能 | 日周期正弦 | 无 | 全天采集 |

---

## 💡 关键发现

### 发现1：能量平衡的困难

```
传感器(有太阳):
  日收入: 40 kJ (太阳能)
  日消耗: 0.0072 J (上报)
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

**结论**：系统需要更高的RF功率或更优的路由策略。

### 发现2：MRC的距离限制

```
距离 | 效率 | 功率(0.1W发射)
-----|------|---------------
5 m  | 0.64 | 0.064 W
10 m | 0.08 | 0.008 W
30 m | 2.96e-5 | 2.96 μW
```

**结论**：MRC只在5-10米内有效，不适合远距离传输。

### 发现3：RIS的有限增益

```
RIS增益: 29 dBi (约800倍)
但两跳总损耗增加，可能不如直射
```

**结论**：RIS最适合覆盖直射盲区，而非所有场景。

### 发现4：太阳能的关键作用

```
有太阳能: 节点可自给自足
无太阳能: 完全依赖RF和MRC
```

**结论**：太阳能是系统稳定运行的关键。

---

## 🔧 实现细节速览

### 调度器实现 (src/scheduling/scheduler.py)

```python
def schedule_power_transfer(wsn):
    # 1. 找最低能量簇头
    rf_target = min(clusters, key=lambda c: c.cluster_head.current_energy)
    
    # 2. 筛选MRC发射簇头
    mrc_transmitters = [
        ch for ch in clusters
        if ch.current_energy > 0.2 * INITIAL_ENERGY
        and ch != rf_target
    ]
    
    return {'rf_target': rf_target, 'mrc_transmitters': mrc_transmitters}
```

**代码行数**：~50行
**复杂度**：O(n) 其中n为簇数

### 路由器实现 (src/routing/routing_algorithm.py)

```python
def find_optimal_energy_path(wsn, source, target_ch, max_hops=2):
    # 1. 评估直射
    direct_power = calculate_received_rf_power(source, target_ch, env)
    
    # 2. 评估单跳RIS
    for ris in ris_panels:
        power = calculate_ris_assisted_power(source, ris, target_ch, env)
    
    # 3. 评估双跳RIS
    for ris_i, ris_j in permutations(ris_panels, 2):
        power = calculate_ris_assisted_power(source, ris_i, ris_j, env)
    
    # 4. 返回最优路径
    return best_path, max_power
```

**代码行数**：~100行
**复杂度**：O(n²) 其中n为RIS数量

### 主循环实现 (src/simulation_main.py)

```python
for t_step in range(num_steps):
    # 1. 小时级上报
    # 2. 太阳能采集
    # 3. 簇内MRC下发
    # 4. 调度决策
    # 5. RF远场充电 (可选)
    # 6. 能量记录
    # 7. 节点死亡检测
```

**代码行数**：~200行
**时间复杂度**：O(n·m) 其中n为时间步数，m为节点数

---

## 📊 性能特征

### 时间复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| 调度 | O(n) | n=簇数 |
| 路由 | O(m²) | m=RIS数 |
| MRC下发 | O(n·k) | n=簇数，k=每簇传感器数 |
| 主循环 | O(t·n·k) | t=时间步数 |

### 空间复杂度

| 数据结构 | 复杂度 | 说明 |
|---------|--------|------|
| 能量历史 | O(t·n·k) | t=时间步数，n=簇数，k=传感器数 |
| RIS相位 | O(m·256) | m=RIS数，256=16×16单元 |
| 地形DEM | O(w·h) | w,h=网格尺寸 |

---

## 🎓 学习建议

### 初级学习路径

1. **第1天**：阅读 PROJECT_ARCHITECTURE_CN.md
   - 理解系统整体架构
   - 理解三层能量传输体系
   - 理解调度和路由的基本概念

2. **第2天**：阅读 ENERGY_FLOW_DETAILED.md
   - 理解每个时间步的执行流程
   - 理解能量流向的具体计算
   - 运行 simulation_main.py 观察结果

3. **第3天**：阅读 PHYSICS_MODELS_REFERENCE.md
   - 理解各物理模型的公式
   - 理解参数的含义和取值
   - 修改参数观察效果

4. **第4天**：阅读 QUICK_REFERENCE.md
   - 掌握快速查询技巧
   - 学习常见问题的解决方法
   - 设计简单的对比实验

### 中级学习路径

1. **修改调度策略**
   - 改变MRC触发阈值
   - 改变RF目标选择策略
   - 观察对网络寿命的影响

2. **修改路由策略**
   - 禁用双跳RIS
   - 改变RIS位置
   - 观察对能量利用率的影响

3. **优化系统参数**
   - 增加RF功率
   - 增加RIS数量
   - 启用太阳能
   - 观察综合效果

### 高级学习路径

1. **设计新的调度算法**
   - 基于能量预测的调度
   - 基于距离的调度
   - 基于负载均衡的调度

2. **设计新的路由算法**
   - 基于能量效率的路由
   - 基于延迟的路由
   - 基于覆盖率的路由

3. **优化物理模型**
   - 引入更精确的路径损耗模型
   - 考虑多径衰落
   - 考虑天线方向性

---

## 🚀 快速开始

### 1. 理解项目 (5分钟)
```bash
# 阅读项目概述
cat PROJECT_ARCHITECTURE_CN.md | head -100
```

### 2. 运行仿真 (2分钟)
```bash
cd src
python simulation_main.py
```

### 3. 查看结果 (3分钟)
```bash
# 查看能量曲线
python viz/plot_results.py

# 查看日志
tail -50 simulation_logs/simulation_*.log
```

### 4. 修改参数 (5分钟)
```python
# 编辑 src/config/simulation_config.py
# 例如: TRANSMIT_POWER_W = 50.0
# 重新运行仿真
```

### 5. 对比结果 (5分钟)
```bash
# 保存结果
cp src/simulation_logs/energy_*.csv results/exp1.csv

# 修改参数后再运行
cp src/simulation_logs/energy_*.csv results/exp2.csv

# 对比两个实验
python compare_experiments.py results/exp1.csv results/exp2.csv
```

---

## 📈 实验建议

### 基准实验
- RF功率: 10W
- RIS数量: 2
- 太阳能: 禁用
- 网络寿命: ?

### 对比实验1：增加RF功率
- RF功率: 50W (5倍)
- 预期: 网络寿命增加

### 对比实验2：增加RIS数量
- RIS数量: 4 (2倍)
- 预期: 覆盖率提高

### 对比实验3：启用太阳能
- 太阳能: 启用
- 预期: 网络寿命大幅增加

### 对比实验4：优化调度
- MRC阈值: 30% (从20%)
- 预期: MRC更早触发

---

## 🔗 文档交叉引用

### 如果你想了解...

| 问题 | 查看文档 | 章节 |
|------|---------|------|
| 系统整体架构 | PROJECT_ARCHITECTURE_CN.md | 系统架构 |
| 调度如何工作 | PROJECT_ARCHITECTURE_CN.md | 能量传输调度逻辑 |
| 路由如何工作 | PROJECT_ARCHITECTURE_CN.md | 能量路由算法 |
| 时间步执行流程 | ENERGY_FLOW_DETAILED.md | 每个时间步的执行顺序 |
| 能量流向示例 | ENERGY_FLOW_DETAILED.md | 完整的能量流向示例 |
| RF模型公式 | PHYSICS_MODELS_REFERENCE.md | RF远场传播模型 |
| RIS增益计算 | PHYSICS_MODELS_REFERENCE.md | RIS反射增益模型 |
| MRC功率计算 | PHYSICS_MODELS_REFERENCE.md | MRC近场传输模型 |
| 太阳能采集 | PHYSICS_MODELS_REFERENCE.md | 太阳能采集模型 |
| 快速查询参数 | QUICK_REFERENCE.md | 物理模型速记 |
| 常见问题解答 | QUICK_REFERENCE.md | 常见问题速答 |
| 调试技巧 | QUICK_REFERENCE.md | 调试技巧 |

---

## 📞 关键代码位置

### 调度相关
```
src/scheduling/scheduler.py
  └─ schedule_power_transfer(wsn)
```

### 路由相关
```
src/routing/routing_algorithm.py
  └─ find_optimal_energy_path(wsn, source, target_ch, max_hops=2)
```

### 物理模型
```
src/utils/rf_propagation_model.py
  ├─ calculate_received_rf_power(tx, rx, env)
  └─ calculate_ris_assisted_power(source, ris, target, env)

src/utils/mrc_model.py
  └─ calculate_received_mrc_power(tx_node, rx_node, tx_power_w=None)

src/core/RIS.py
  ├─ configure_phases(incident_point, reflection_point)
  └─ get_reflection_gain()
```

### 主循环
```
src/simulation_main.py
  └─ run_simulation()
```

### 配置
```
src/config/simulation_config.py
  ├─ EnvConfig
  ├─ SinkConfig
  ├─ RISConfig
  ├─ WSNConfig
  ├─ SensorNodeConfig
  ├─ ClusterHeadConfig
  └─ SimConfig
```

---

## 📝 总结

这个项目实现了一个**分层无线能量传输系统**，具有以下特点：

1. **三层架构**：RF远场 → RIS反射 → MRC下发 → 太阳能补充
2. **智能调度**：基于能量水位的贪心调度策略
3. **多路径路由**：支持直射、单跳、双跳等多种路径
4. **物理精确**：采用真实的传播模型和参数
5. **地形感知**：基于DEM的视距判断

通过这4份文档，你可以：
- ✅ 理解系统的整体架构
- ✅ 理解调度和路由的工作原理
- ✅ 理解物理模型的公式和参数
- ✅ 快速查询和修改配置
- ✅ 设计和执行对比实验
- ✅ 优化系统性能

**建议阅读顺序**：
1. PROJECT_ARCHITECTURE_CN.md (15-20分钟)
2. ENERGY_FLOW_DETAILED.md (20-30分钟)
3. PHYSICS_MODELS_REFERENCE.md (25-35分钟)
4. QUICK_REFERENCE.md (按需查询)

祝你学习愉快！🚀





