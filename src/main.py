"""
Main simulation entrypoint.
- 构建 DEM（由 data/S3.csv 插值得到 + 湖面拉平）
- 读取 data/sink.csv 获取 Sink/RF/RIS 位置
- 运行仿真主循环，输出日志到 logs/log_YYYYMMDD_HHMMSS.txt
注意：请在 PyCharm 的 Run/Debug 配置里将 Working directory 设置为 项目/src
"""
import os
import sys
import numpy as np

# 确保可以用 "from src.xxx import ..." 的形式导入
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.network.WSN import WSN
from src.routing import routing_algorithm
from src.scheduling import scheduler
from src.utils import mrc_model
from src.utils.simulation_logger import SimulationLogger
from src.config.simulation_config import SimConfig, SensorNodeConfig, ClusterHeadConfig, WSNConfig
from src.viz.plot_results import plot_energy_history


def run_simulation():
    """初始化并运行仿真循环"""
    # 1) 初始化 WSN（内部会：用 S3.csv 生成 DEM -> 用 sink.csv 读取 Sink/RF/RIS）
    wsn = WSN()

    # 结果缓存：每个节点在每个时间步的能量
    all_nodes = wsn.get_all_nodes()
    node_ids = [node.node_id for node in all_nodes]
    num_steps = int(SimConfig.SIMULATION_TIME_S / SimConfig.TIME_STEP_S)
    energy_history = {node_id: np.zeros(num_steps) for node_id in node_ids}

    # 初始化日志
    logger = SimulationLogger()

    print("\nStarting simulation...")

    # 触发式跨簇供能状态
    donation_state = {'donor': None, 'receiver': None, 'period_end': -1}
    # 预计算各簇基线能量（CH初始 + 簇内节点初始）
    cluster_baseline = []
    for cl in wsn.clusters:
        base = cl.cluster_head.initial_energy + sum(n.initial_energy for n in cl.sensor_nodes)
        cluster_baseline.append(base if base > 0 else 1e-9)

    # 2) 主循环
    for t_step in range(num_steps):
        current_time = t_step * SimConfig.TIME_STEP_S
        logger.log_step(t_step, current_time)

        # a) 调度：仅用于 MRC（主站不再作为源发射远场能量）
        actions = scheduler.schedule_power_transfer(wsn)
        rf_target_ch = None  # 禁用主站作为源
        mrc_transmitting_chs = actions['mrc_transmitters']

        # b) 每K步重新评估跨簇供能配对
        if getattr(SimConfig, 'ENABLE_CROSS_CLUSTER_DONATION', False) and \
           (t_step % getattr(SimConfig, 'CROSS_CLUSTER_TRIGGER_PERIOD_STEPS', 100) == 0):
            # 计算各簇能量百分比
            pct = []
            for idx, cl in enumerate(wsn.clusters):
                cur = cl.cluster_head.current_energy + sum(n.current_energy for n in cl.sensor_nodes)
                pct.append(cur / cluster_baseline[idx])
            low_thr = getattr(SimConfig, 'TRIGGER_LOW_PCT', 0.30)
            high_thr = getattr(SimConfig, 'TRIGGER_HIGH_PCT', 0.80)
            # 选 receiver
            receiver_idx = None
            min_pct = 1e9
            for i, p in enumerate(pct):
                if p < low_thr and p < min_pct:
                    receiver_idx = i
                    min_pct = p
            donor_idx = None
            if receiver_idx is not None:
                # 候选 donor：pct>high 且非 receiver
                rx_ch = wsn.clusters[receiver_idx].cluster_head
                best_score = -1.0
                for i, p in enumerate(pct):
                    if i == receiver_idx or p <= high_thr:
                        continue
                    dx = np.linalg.norm(wsn.clusters[i].cluster_head.position - rx_ch.position)
                    score = p / (dx + 1e-6)
                    if score > best_score:
                        best_score = score
                        donor_idx = i
            if donor_idx is not None and receiver_idx is not None:
                donation_state['donor'] = wsn.clusters[donor_idx].cluster_head
                donation_state['receiver'] = wsn.clusters[receiver_idx].cluster_head
                donation_state['period_end'] = t_step + getattr(SimConfig, 'CROSS_CLUSTER_TRIGGER_PERIOD_STEPS', 100)
            else:
                donation_state['donor'] = None
                donation_state['receiver'] = None
                donation_state['period_end'] = t_step

        # c) 跨簇供能执行（在评估周期内持续）
        best_path, max_power_w = [], 0.0
        rf_sent_energy_j, rf_delivered_energy_j = None, 0.0
        if donation_state['donor'] is not None and donation_state['receiver'] is not None and t_step < donation_state['period_end']:
            donor = donation_state['donor']
            receiver = donation_state['receiver']
            # donor 本步的发射能耗
            rf_sent_energy_j = donor.rf_tx_power_w * SimConfig.TIME_STEP_S
            if donor.current_energy >= rf_sent_energy_j:
                best_path, max_power_w = routing_algorithm.find_optimal_energy_path(wsn, donor, receiver)
                if max_power_w > 0:
                    rf_delivered_energy_j = max_power_w * SimConfig.TIME_STEP_S
                    receiver.receive_rf_power(max_power_w, SimConfig.TIME_STEP_S)
                # 扣除 donor 的发射能耗
                donor.current_energy = max(0.0, donor.current_energy - rf_sent_energy_j)

        # d) 簇内 MRC 传输（排除 donor）
        mrc_entries = []
        donor_ch = donation_state['donor'] if (donation_state['donor'] is not None and t_step < donation_state['period_end']) else None
        for ch in mrc_transmitting_chs:
            if donor_ch is not None and ch is donor_ch:
                continue
            target_nodes = [c.sensor_nodes for c in wsn.clusters if c.cluster_head == ch][0]
            delivered_j, sent_j = ch.transmit_mrc_power(target_nodes, SimConfig.TIME_STEP_S, mrc_model)
            mrc_entries.append({'ch_id': ch.node_id, 'sent_j': sent_j, 'delivered_j': delivered_j})

        # e) 记录能量流
        logger.log_scheduling({'rf_target': rf_target_ch, 'mrc_transmitters': [ch for ch in mrc_transmitting_chs if (donor_ch is None or ch is not donor_ch)]})
        logger.log_routing(best_path, max_power_w)
        logger.log_energy_transfer(donation_state.get('receiver'), rf_sent_energy_j, rf_delivered_energy_j, mrc_entries)

        # e) 太阳能采集 + 自然衰减（按节点 update_energy）
        current_time_min = (current_time % (24 * 3600)) / 60.0
        for node in all_nodes:
            if hasattr(node, 'enable_energy_harvesting'):
                node.enable_energy_harvesting = getattr(node, 'has_solar', False) and WSNConfig.ENABLE_SOLAR
            node.update_energy(current_time_min)

        # f) 记录能量曲线 & 检查死亡
        for node in all_nodes:
            energy_history[node.node_id][t_step] = node.current_energy
            if node.current_energy < SensorNodeConfig.MIN_ENERGY_J:
                print(f"!!! Node {node.node_id} has died at {current_time}s !!!")

        # g) 记录各簇能量状态
        logger.log_cluster_energy(wsn)

        # h) 进度条
        if (t_step + 1) % 100 == 0:
            print(f"... Step {t_step + 1}/{num_steps} completed.")

    print("Simulation finished.")
    logger.close()
    return energy_history, node_ids


if __name__ == "__main__":
    energy_data, node_ids = run_simulation()

    print("\n--- Final Energy Status ---")
    for node_id in node_ids:
        final_energy = energy_data[node_id][-1]
        print(f"Node {node_id}: {final_energy:.4f} J")

    if SimConfig.ENABLE_PLOT_RESULTS:
        plot_energy_history(energy_data, node_ids)
