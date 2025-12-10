# WSN_S3 项目文档索引

## 📚 完整文档列表

### 核心分析文档 (4份)

| 文档 | 大小 | 阅读时间 | 难度 | 推荐指数 |
|------|------|---------|------|---------|
| **README_ANALYSIS.md** | 中 | 10分钟 | ⭐ | ⭐⭐⭐⭐⭐ |
| **PROJECT_ARCHITECTURE_CN.md** | 大 | 15-20分钟 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **ENERGY_FLOW_DETAILED.md** | 大 | 20-30分钟 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **PHYSICS_MODELS_REFERENCE.md** | 大 | 25-35分钟 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **QUICK_REFERENCE.md** | 中 | 5-10分钟 | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎯 按用途查找文档

### 我想快速理解项目
👉 **README_ANALYSIS.md** (10分钟)
- 项目概述
- 核心发现
- 关键发现
- 快速开始

### 我想理解系统架构
👉 **PROJECT_ARCHITECTURE_CN.md** (15-20分钟)
- 系统架构图
- 文件结构
- 调度逻辑详解
- 路由逻辑详解
- 物理模型概览

### 我想理解代码执行流程
👉 **ENERGY_FLOW_DETAILED.md** (20-30分钟)
- 初始化阶段
- 7个时间步执行顺序
- 完整的能量流向示例
- 关键数值总结

### 我想理解物理模型
👉 **PHYSICS_MODELS_REFERENCE.md** (25-35分钟)
- RF远场传播模型
- RIS反射增益模型
- MRC近场传输模型
- 太阳能采集模型
- 能量消耗模型

### 我想快速查询参数
👉 **QUICK_REFERENCE.md** (按需查询)
- 核心概念速记
- 文件导航
- 调度/路由速记
- 物理模型速记
- 常见问题速答

---

## 📖 按学习阶段推荐

### 初级 (第1-2天)

**目标**：理解项目整体架构和基本概念

1. **第1天上午** (30分钟)
   - 阅读 README_ANALYSIS.md (10分钟)
   - 阅读 PROJECT_ARCHITECTURE_CN.md 的前3个章节 (20分钟)

2. **第1天下午** (1小时)
   - 继续阅读 PROJECT_ARCHITECTURE_CN.md (40分钟)
   - 运行 simulation_main.py (20分钟)

3. **第2天上午** (1小时)
   - 阅读 ENERGY_FLOW_DETAILED.md 的前2个章节 (40分钟)
   - 查看仿真日志和能量曲线 (20分钟)

4. **第2天下午** (1小时)
   - 继续阅读 ENERGY_FLOW_DETAILED.md (40分钟)
   - 修改一个简单的参数，重新运行 (20分钟)

### 中级 (第3-4天)

**目标**：理解物理模型和代码细节

1. **第3天上午** (1.5小时)
   - 阅读 PHYSICS_MODELS_REFERENCE.md 的RF模型部分 (50分钟)
   - 计算一个RF功率的例子 (40分钟)

2. **第3天下午** (1.5小时)
   - 阅读 PHYSICS_MODELS_REFERENCE.md 的RIS和MRC部分 (50分钟)
   - 计算一个RIS和MRC功率的例子 (40分钟)

3. **第4天上午** (1小时)
   - 阅读 PHYSICS_MODELS_REFERENCE.md 的太阳能部分 (40分钟)
   - 理解能量消耗模型 (20分钟)

4. **第4天下午** (1.5小时)
   - 修改调度参数，观察效果 (45分钟)
   - 修改路由参数，观察效果 (45分钟)

### 高级 (第5-7天)

**目标**：优化系统，设计实验

1. **第5天** (2小时)
   - 设计基准实验
   - 运行基准实验
   - 记录结果

2. **第6天** (2小时)
   - 设计对比实验1-3
   - 运行对比实验
   - 分析结果

3. **第7天** (2小时)
   - 设计优化实验
   - 运行优化实验
   - 总结改进方案

---

## 🔍 按问题查找答案

### 系统设计相关

**Q: 系统的整体架构是什么？**
- 查看: PROJECT_ARCHITECTURE_CN.md → 系统架构

**Q: 调度器如何工作？**
- 查看: PROJECT_ARCHITECTURE_CN.md → 能量传输调度逻辑
- 查看: QUICK_REFERENCE.md → 调度策略速记

**Q: 路由器如何工作？**
- 查看: PROJECT_ARCHITECTURE_CN.md → 能量路由算法
- 查看: QUICK_REFERENCE.md → 路由策略速记

**Q: 支持哪些能量传输路径？**
- 查看: PROJECT_ARCHITECTURE_CN.md → 能量路由算法 → 支持的路径类型

### 执行流程相关

**Q: 仿真的每个时间步做什么？**
- 查看: ENERGY_FLOW_DETAILED.md → 每个时间步的执行顺序

**Q: 能量是如何流动的？**
- 查看: ENERGY_FLOW_DETAILED.md → 完整的能量流向示例

**Q: 为什么节点快速死亡？**
- 查看: QUICK_REFERENCE.md → 常见问题速答 → Q: 为什么节点快速死亡？

**Q: 为什么路由总是选直射？**
- 查看: QUICK_REFERENCE.md → 常见问题速答 → Q: 为什么路由总是选直射？

### 物理模型相关

**Q: RF功率如何计算？**
- 查看: PHYSICS_MODELS_REFERENCE.md → RF远场传播模型

**Q: RIS增益如何计算？**
- 查看: PHYSICS_MODELS_REFERENCE.md → RIS反射增益模型

**Q: MRC功率如何计算？**
- 查看: PHYSICS_MODELS_REFERENCE.md → MRC近场传输模型

**Q: 太阳能采集如何计算？**
- 查看: PHYSICS_MODELS_REFERENCE.md → 太阳能采集模型

**Q: 参数的含义是什么？**
- 查看: PHYSICS_MODELS_REFERENCE.md → 参数配置
- 查看: QUICK_REFERENCE.md → 物理模型速记

### 配置修改相关

**Q: 如何增加RF功率？**
- 查看: QUICK_REFERENCE.md → 配置修改指南 → 增加RF功率

**Q: 如何增加RIS数量？**
- 查看: QUICK_REFERENCE.md → 配置修改指南 → 增加RIS数量

**Q: 如何调整MRC阈值？**
- 查看: QUICK_REFERENCE.md → 配置修改指南 → 调整MRC阈值

**Q: 如何启用太阳能？**
- 查看: QUICK_REFERENCE.md → 配置修改指南 → 启用太阳能

### 调试相关

**Q: 如何打印调度决策？**
- 查看: QUICK_REFERENCE.md → 调试技巧 → 打印调度决策

**Q: 如何打印路由结果？**
- 查看: QUICK_REFERENCE.md → 调试技巧 → 打印路由结果

**Q: 如何打印能量历史？**
- 查看: QUICK_REFERENCE.md → 调试技巧 → 打印能量历史

**Q: 如何启用详细日志？**
- 查看: QUICK_REFERENCE.md → 调试技巧 → 启用详细日志

---

## 📊 文档内容对应表

### 调度器 (scheduler.py)

| 内容 | 文档 | 章节 |
|------|------|------|
| 工作原理 | PROJECT_ARCHITECTURE_CN.md | 能量传输调度逻辑 |
| 决策流程 | ENERGY_FLOW_DETAILED.md | 第4步：调度决策 |
| 速记 | QUICK_REFERENCE.md | 调度策略速记 |
| 代码实现 | QUICK_REFERENCE.md | 调度策略速记 → 调度器决策流程 |
| 修改方法 | QUICK_REFERENCE.md | 配置修改指南 → 调整MRC阈值 |

### 路由器 (routing_algorithm.py)

| 内容 | 文档 | 章节 |
|------|------|------|
| 工作原理 | PROJECT_ARCHITECTURE_CN.md | 能量路由算法 |
| 决策流程 | ENERGY_FLOW_DETAILED.md | 第5步：RF远场充电 |
| 速记 | QUICK_REFERENCE.md | 路由策略速记 |
| 代码实现 | QUICK_REFERENCE.md | 路由策略速记 → 路由决策流程 |
| 常见问题 | QUICK_REFERENCE.md | 常见问题速答 → Q: 为什么路由总是选直射？ |

### RF模型 (rf_propagation_model.py)

| 内容 | 文档 | 章节 |
|------|------|------|
| 公式 | PHYSICS_MODELS_REFERENCE.md | RF远场传播模型 → 公式 |
| 参数 | PHYSICS_MODELS_REFERENCE.md | RF远场传播模型 → 参数配置 |
| 计算示例 | PHYSICS_MODELS_REFERENCE.md | RF远场传播模型 → 计算示例 |
| 速记 | QUICK_REFERENCE.md | 物理模型速记 → RF远场 |

### RIS模型 (RIS.py)

| 内容 | 文档 | 章节 |
|------|------|------|
| 基本参数 | PHYSICS_MODELS_REFERENCE.md | RIS反射增益模型 → RIS基本参数 |
| 反射增益 | PHYSICS_MODELS_REFERENCE.md | RIS反射增益模型 → RIS反射增益 |
| 相位配置 | PHYSICS_MODELS_REFERENCE.md | RIS反射增益模型 → RIS相位配置 |
| 功率计算 | PHYSICS_MODELS_REFERENCE.md | RIS反射增益模型 → RIS辅助功率计算 |
| 速记 | QUICK_REFERENCE.md | 物理模型速记 → RIS反射 |

### MRC模型 (mrc_model.py)

| 内容 | 文档 | 章节 |
|------|------|------|
| 公式 | PHYSICS_MODELS_REFERENCE.md | MRC近场传输模型 → 模型公式 |
| 参数 | PHYSICS_MODELS_REFERENCE.md | MRC近场传输模型 → 参数配置 |
| 计算示例 | PHYSICS_MODELS_REFERENCE.md | MRC近场传输模型 → 计算示例 |
| 速记 | QUICK_REFERENCE.md | 物理模型速记 → MRC近场 |

### 太阳能模型 (SensorNode.py)

| 内容 | 文档 | 章节 |
|------|------|------|
| 日照周期 | PHYSICS_MODELS_REFERENCE.md | 太阳能采集模型 → 日照周期模型 |
| 参数 | PHYSICS_MODELS_REFERENCE.md | 太阳能采集模型 → 参数配置 |
| 计算示例 | PHYSICS_MODELS_REFERENCE.md | 太阳能采集模型 → 计算示例 |
| 速记 | QUICK_REFERENCE.md | 物理模型速记 → 太阳能 |

---

## 🎓 推荐学习路径

### 路径1：快速入门 (2小时)
```
1. README_ANALYSIS.md (10分钟)
2. PROJECT_ARCHITECTURE_CN.md (50分钟)
3. QUICK_REFERENCE.md (20分钟)
4. 运行仿真 (20分钟)
5. 修改参数 (20分钟)
```

### 路径2：深入学习 (1天)
```
1. README_ANALYSIS.md (10分钟)
2. PROJECT_ARCHITECTURE_CN.md (50分钟)
3. ENERGY_FLOW_DETAILED.md (60分钟)
4. PHYSICS_MODELS_REFERENCE.md (60分钟)
5. QUICK_REFERENCE.md (20分钟)
6. 运行和修改实验 (60分钟)
```

### 路径3：完全掌握 (3天)
```
第1天:
  - README_ANALYSIS.md (10分钟)
  - PROJECT_ARCHITECTURE_CN.md (50分钟)
  - 运行仿真 (20分钟)

第2天:
  - ENERGY_FLOW_DETAILED.md (60分钟)
  - PHYSICS_MODELS_REFERENCE.md (60分钟)
  - 修改参数实验 (60分钟)

第3天:
  - QUICK_REFERENCE.md (20分钟)
  - 设计对比实验 (120分钟)
  - 分析结果 (60分钟)
```

---

## 📋 文档检查清单

### 初级检查清单
- [ ] 阅读了 README_ANALYSIS.md
- [ ] 理解了系统的三层架构
- [ ] 理解了调度器的工作原理
- [ ] 理解了路由器的工作原理
- [ ] 成功运行了仿真

### 中级检查清单
- [ ] 阅读了 PROJECT_ARCHITECTURE_CN.md
- [ ] 阅读了 ENERGY_FLOW_DETAILED.md
- [ ] 理解了每个时间步的执行流程
- [ ] 理解了能量的完整流向
- [ ] 修改过至少一个参数

### 高级检查清单
- [ ] 阅读了 PHYSICS_MODELS_REFERENCE.md
- [ ] 理解了所有物理模型的公式
- [ ] 能手工计算RF/RIS/MRC功率
- [ ] 设计过对比实验
- [ ] 分析过实验结果
- [ ] 提出过优化方案

---

## 🔗 快速导航

### 最常用的3份文档
1. **QUICK_REFERENCE.md** - 日常参考
2. **PROJECT_ARCHITECTURE_CN.md** - 系统理解
3. **PHYSICS_MODELS_REFERENCE.md** - 模型查询

### 最常查看的3个章节
1. PROJECT_ARCHITECTURE_CN.md → 能量传输调度逻辑
2. PROJECT_ARCHITECTURE_CN.md → 能量路由算法
3. QUICK_REFERENCE.md → 常见问题速答

### 最常修改的3个参数
1. `TRANSMIT_POWER_W` (RF功率)
2. `ENABLE_SOLAR` (太阳能开关)
3. MRC触发阈值 (20%)

---

## 📞 文档反馈

如果你发现：
- 文档有错误
- 文档不清楚
- 文档缺少内容
- 文档有改进建议

请提出反馈，我们会持续改进文档质量。

---

## 📝 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| 1.0 | 2025-12-10 | 初始版本，包含5份核心文档 |

---

**最后更新:** 2025-12-10
**总文档数:** 5份
**总字数:** ~50,000字
**预计阅读时间:** 2-3小时 (快速) / 1-2天 (深入) / 3-5天 (完全掌握)





