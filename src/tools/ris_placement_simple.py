"""
升级版 RIS 选址脚本（可直接在 PyCharm 运行，交互输入两个簇编号）。
流程（自动模式）：
1) 视域交集 单RIS：同时被两端看见的区域中择优（优先山顶/脊线，最稳健）
2) 走廊 单RIS：以两簇连线为走廊进行候选搜索（兼容性强）
3) 双RIS回退：A→RIS1→RIS2→B（在两端视域中选点并检查 RIS1↔RIS2 LoS）
4) 多RIS（3~5 个）：基于视域候选的束搜索（Beam Search），在 LoS 约束下扩展链路，择优功率最大链

注意：
- 工作目录请设为项目 src/；数据默认来自 data/ 目录（S3.csv/LAKE.csv/sink.csv）
- 参数可在 main() 调用处调整：走廊半宽、步长系数、安装高度、Top-K、束宽等
"""
import sys
import os
import math
import numpy as np
from typing import Tuple, List, Dict

# 允许 from src.xxx 导入
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.network.WSN import WSN
from src.core.RIS import RIS
from src.utils import rf_propagation_model


# ------------------------------
# 工具函数
# ------------------------------

def line_bounding_box(p1: np.ndarray, p2: np.ndarray, margin: float) -> Tuple[float, float, float, float]:
    xmin = min(p1[0], p2[0]) - margin
    xmax = max(p1[0], p2[0]) + margin
    ymin = min(p1[1], p2[1]) - margin
    ymax = max(p1[1], p2[1]) + margin
    return xmin, xmax, ymin, ymax


def point_to_segment_distance_xy(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ap = p[:2] - a[:2]
    ab = b[:2] - a[:2]
    ab2 = float(np.dot(ab, ab))
    if ab2 <= 1e-12:
        return float(np.linalg.norm(ap))
    t = max(0.0, min(1.0, float(np.dot(ap, ab) / ab2)))
    proj = a[:2] + t * ab
    return float(np.linalg.norm(p[:2] - proj))


def build_grid(xmin: float, xmax: float, ymin: float, ymax: float, step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi = np.arange(xmin, xmax + step, step, dtype=float)
    yi = np.arange(ymin, ymax + step, step, dtype=float)
    XI, YI = np.meshgrid(xi, yi)
    return xi, yi, XI, YI


def compute_viewshed_mask(env, origin: np.ndarray, XI: np.ndarray, YI: np.ndarray, install_height: float) -> np.ndarray:
    H, W = XI.shape
    mask = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            x = float(XI[i, j]); y = float(YI[i, j])
            z = float(env.get_elevation(x, y)) + install_height
            p = np.array([x, y, z], dtype=float)
            if env.check_los(origin, p):
                mask[i, j] = True
    return mask


def _get_ch(wsn: WSN, cid: int):
    for cl in wsn.clusters:
        if cl.cluster_id == cid:
            return cl.cluster_head
    if 0 <= cid < len(wsn.clusters):
        return wsn.clusters[cid].cluster_head
    return None


# ------------------------------
# 单 RIS：视域交集
# ------------------------------

def choose_single_ris_viewshed(wsn: WSN, src_cluster_id: int, dst_cluster_id: int,
                               corridor_half_width: float,
                               install_height: float,
                               coarse_step_factor: float) -> Dict:
    env = wsn.environment
    step = max(float(env.resolution) * coarse_step_factor, 20.0)

    ch_src = _get_ch(wsn, src_cluster_id)
    ch_dst = _get_ch(wsn, dst_cluster_id)
    if ch_src is None or ch_dst is None:
        return {'ok': False}

    p1 = np.array(ch_src.position, dtype=float)
    p2 = np.array(ch_dst.position, dtype=float)
    xmin, xmax, ymin, ymax = line_bounding_box(p1, p2, corridor_half_width)

    xi, yi, XI, YI = build_grid(xmin, xmax, ymin, ymax, step)

    # 两端视域
    mask_a = compute_viewshed_mask(env, p1, XI, YI, install_height)
    mask_b = compute_viewshed_mask(env, p2, XI, YI, install_height)
    mask = np.logical_and(mask_a, mask_b)

    best_power = 0.0
    best_pos = None
    feasible = int(mask.sum())
    evaluated = feasible

    # 在交集内评估功率
    it = np.argwhere(mask)
    for (i, j) in it:
        x = float(XI[i, j]); y = float(YI[i, j])
        z = float(env.get_elevation(x, y)) + install_height
        candidate = np.array([x, y, z], dtype=float)
        ris = RIS(panel_id=-1, position=candidate)
        pr_w = rf_propagation_model.calculate_ris_assisted_power(ch_src, ris, ch_dst, env)
        if pr_w > best_power:
            best_power = pr_w
            best_pos = candidate

    return {
        'ok': best_pos is not None,
        'best_pos': best_pos,
        'best_power_w': float(best_power),
        'evaluated': int(evaluated),
        'feasible': int(feasible),
        'mode': 'viewshed_single',
        'step': float(step),
    }


# ------------------------------
# 单 RIS：走廊模式（原方案）
# ------------------------------

def choose_single_ris_corridor(wsn: WSN, src_cluster_id: int, dst_cluster_id: int,
                               corridor_half_width: float,
                               install_height: float,
                               coarse_step_factor: float) -> Dict:
    env = wsn.environment

    ch_src = _get_ch(wsn, src_cluster_id)
    ch_dst = _get_ch(wsn, dst_cluster_id)
    if ch_src is None or ch_dst is None:
        return {'ok': False}

    p1 = np.array(ch_src.position, dtype=float)
    p2 = np.array(ch_dst.position, dtype=float)

    margin = corridor_half_width
    xmin, xmax, ymin, ymax = line_bounding_box(p1, p2, margin)
    step = max(float(env.resolution) * coarse_step_factor, 20.0)

    best_power = 0.0
    best_pos = None
    evaluated = 0
    feasible = 0

    xi, yi, XI, YI = build_grid(xmin, xmax, ymin, ymax, step)
    max_d = corridor_half_width

    for i in range(XI.shape[0]):
        for j in range(XI.shape[1]):
            x = float(XI[i, j]); y = float(YI[i, j])
            candidate_xy = np.array([x, y, 0.0], dtype=float)
            d = point_to_segment_distance_xy(candidate_xy, p1, p2)
            if d > max_d:
                continue
            z = float(env.get_elevation(x, y)) + install_height
            candidate = np.array([x, y, z], dtype=float)

            evaluated += 1
            if not env.check_los(p1, candidate):
                continue
            if not env.check_los(candidate, p2):
                continue
            feasible += 1

            ris = RIS(panel_id=-1, position=candidate)
            pr_w = rf_propagation_model.calculate_ris_assisted_power(ch_src, ris, ch_dst, env)
            if pr_w > best_power:
                best_power = pr_w
                best_pos = candidate

    return {
        'ok': best_pos is not None,
        'best_pos': best_pos,
        'best_power_w': float(best_power),
        'evaluated': int(evaluated),
        'feasible': int(feasible),
        'mode': 'corridor_single',
        'step': float(step),
    }


# ------------------------------
# 双 RIS：两端视域 + RIS间LoS
# ------------------------------

def choose_two_ris_viewsheds(wsn: WSN, src_cluster_id: int, dst_cluster_id: int,
                             corridor_half_width: float,
                             install_height: float,
                             coarse_step_factor: float,
                             topk_each: int = 200) -> Dict:
    env = wsn.environment
    step = max(float(env.resolution) * coarse_step_factor, 25.0)

    ch_src = _get_ch(wsn, src_cluster_id)
    ch_dst = _get_ch(wsn, dst_cluster_id)
    if ch_src is None or ch_dst is None:
        return {'ok': False}

    p1 = np.array(ch_src.position, dtype=float)
    p2 = np.array(ch_dst.position, dtype=float)

    xmin, xmax, ymin, ymax = line_bounding_box(p1, p2, corridor_half_width)
    xi, yi, XI, YI = build_grid(xmin, xmax, ymin, ymax, step)

    # A / B 视域
    mask_a = compute_viewshed_mask(env, p1, XI, YI, install_height)
    mask_b = compute_viewshed_mask(env, p2, XI, YI, install_height)

    cand_a = _select_top_candidates(env, XI, YI, mask_a, p1, p2, install_height, topk_each)
    cand_b = _select_top_candidates(env, XI, YI, mask_b, p2, p1, install_height, topk_each)

    best_power = 0.0
    best_pair = None
    pairs_checked = 0

    for pa in cand_a:
        pos_a = pa['pos']
        ris1 = RIS(panel_id=-1, position=pos_a)
        for pb in cand_b:
            pos_b = pb['pos']
            if not env.check_los(pos_a, pos_b):
                continue
            pairs_checked += 1
            ris2 = RIS(panel_id=-2, position=pos_b)
            power_at_ris2 = rf_propagation_model.calculate_ris_assisted_power(ch_src, ris1, ris2, env)
            if power_at_ris2 <= 0:
                continue
            if not env.check_los(pos_b, p2):
                continue
            # RIS2 作为源，近似最终一跳
            ris2_as_src = type('RISSource', (), {})
            ris2_as_src.position = pos_b
            ris2_as_src.get_tx_power_dbm = lambda p=power_at_ris2: 10 * np.log10(p * 1000.0)
            ris2_as_src.get_reflection_gain = ris2.get_reflection_gain
            freq = getattr(ch_src, 'frequency_hz', None)
            dist = float(np.linalg.norm(pos_b - p2))
            final_power_dbm = rf_propagation_model._log_distance_path_loss(
                ris2_as_src.get_tx_power_dbm(),
                ris2.get_reflection_gain(),
                ch_dst.rf_rx_gain_dbi,
                freq,
                dist,
                True,
            )
            final_power_w = 10 ** ((final_power_dbm - 30) / 10)
            if final_power_w > best_power:
                best_power = final_power_w
                best_pair = (pos_a, pos_b)

    return {
        'ok': best_pair is not None,
        'best_pos_pair': best_pair,
        'best_power_w': float(best_power),
        'pairs_checked': int(pairs_checked),
        'mode': 'viewshed_double',
        'step': float(step),
        'topk_each': int(topk_each),
    }


def _select_top_candidates(env, XI, YI, mask, endpoint: np.ndarray, other: np.ndarray,
                            install_height: float, topk: int) -> List[Dict]:
    H, W = XI.shape
    items = []
    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue
            x = float(XI[i, j]); y = float(YI[i, j])
            z_ground = float(env.get_elevation(x, y))
            z = z_ground + install_height
            pos = np.array([x, y, z], dtype=float)
            # 启发式评分：海拔越高越好，越靠近线段越好，离端点不过近/不过远
            elev = z_ground
            d_line = point_to_segment_distance_xy(np.array([x, y, 0.0]), endpoint, other)
            d_end = float(np.linalg.norm(pos[:2] - endpoint[:2]))
            score = elev - 0.001 * d_line - 0.0005 * d_end
            items.append({'pos': pos, 'score': score})
    if not items:
        return []
    items.sort(key=lambda t: t['score'], reverse=True)
    return items[:min(topk, len(items))]


# ------------------------------
# 多 RIS（3~5 个）：束搜索（LoS 约束 + 级联功率评估）
# ------------------------------

def evaluate_chain_final_power(ch_src, chain_positions: List[np.ndarray], ch_dst, env) -> float:
    """评估 A→R1→R2→...→Rm→B 的最终到达功率（W）。要求相邻段 LoS。"""
    if not chain_positions:
        return 0.0
    # 检查相邻 LoS
    prev_pos = np.array(ch_src.position, dtype=float)
    for pos in chain_positions:
        if not env.check_los(prev_pos, pos):
            return 0.0
        prev_pos = pos
    if not env.check_los(chain_positions[-1], np.array(ch_dst.position, dtype=float)):
        return 0.0

    source_obj = ch_src
    # 若只有1个 RIS，最后一跳在下面处理
    for i in range(len(chain_positions) - 1):
        ris_i = RIS(panel_id=-(i+1), position=chain_positions[i])
        ris_next = RIS(panel_id=-(i+2), position=chain_positions[i+1])
        p_next = rf_propagation_model.calculate_ris_assisted_power(source_obj, ris_i, ris_next, env)
        if p_next <= 0:
            return 0.0
        # ris_next 作为新的源，功率为 p_next
        src2 = type('RISSource', (), {})
        src2.position = chain_positions[i+1]
        src2.get_tx_power_dbm = (lambda p=p_next: 10 * np.log10(p * 1000.0))
        src2.get_reflection_gain = ris_next.get_reflection_gain
        src2.frequency_hz = getattr(ch_src, 'frequency_hz', None)
        source_obj = src2

    # 最后一跳：RIS_m -> B
    last_ris = RIS(panel_id=-999, position=chain_positions[-1])
    dist = float(np.linalg.norm(chain_positions[-1] - np.array(ch_dst.position, dtype=float)))
    freq = getattr(ch_src, 'frequency_hz', None)
    final_dbm = rf_propagation_model._log_distance_path_loss(
        source_obj.get_tx_power_dbm(),
        last_ris.get_reflection_gain(),
        ch_dst.rf_rx_gain_dbi,
        freq,
        dist,
        True,
    )
    return 10 ** ((final_dbm - 30) / 10)


def choose_k_ris_chain(wsn: WSN, src_cluster_id: int, dst_cluster_id: int,
                       k: int,
                       corridor_half_width: float,
                       install_height: float,
                       coarse_step_factor: float,
                       topk_each: int = 200,
                       beam_width: int = 40) -> Dict:
    """
    束搜索寻找最多 k 个 RIS 的链路（k ∈ {3,4,5}）：
    - 候选节点来自两端视域 Top-K 的并集（去重）
    - 路径扩展时强制相邻 LoS；在深度≥2时用级联到达“末端RIS”的功率作为评分保留束宽
    - 终止条件：找到可达 B（末端 RIS 与 B LoS），并更新全局最优功率
    """
    assert k >= 3
    env = wsn.environment
    step = max(float(env.resolution) * coarse_step_factor, 25.0)

    ch_src = _get_ch(wsn, src_cluster_id)
    ch_dst = _get_ch(wsn, dst_cluster_id)
    if ch_src is None or ch_dst is None:
        return {'ok': False}

    p1 = np.array(ch_src.position, dtype=float)
    p2 = np.array(ch_dst.position, dtype=float)
    xmin, xmax, ymin, ymax = line_bounding_box(p1, p2, corridor_half_width)
    xi, yi, XI, YI = build_grid(xmin, xmax, ymin, ymax, step)

    # 视域候选（两端 Top-K）
    mask_a = compute_viewshed_mask(env, p1, XI, YI, install_height)
    mask_b = compute_viewshed_mask(env, p2, XI, YI, install_height)
    cand_a = _select_top_candidates(env, XI, YI, mask_a, p1, p2, install_height, topk_each)
    cand_b = _select_top_candidates(env, XI, YI, mask_b, p2, p1, install_height, topk_each)

    # 合并去重
    def keypos(pos):
        return (round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3))
    pool_map = {}
    for item in cand_a + cand_b:
        pool_map[keypos(item['pos'])] = item['pos']
    pool = list(pool_map.values())
    if not pool:
        return {'ok': False}

    # 初始前沿：与 A LoS 的点
    frontier = []  # list of (chain_positions: List[np.ndarray], score: float, last_power_w: float or None)
    for pos in pool:
        if env.check_los(p1, pos):
            # 深度1时暂用启发式评分（高程 - 到连线距离）
            z_ground = float(env.get_elevation(pos[0], pos[1]))
            d_line = point_to_segment_distance_xy(np.array([pos[0], pos[1], 0.0]), p1, p2)
            score = z_ground - 0.001 * d_line
            frontier.append(([pos], score, None))
    if not frontier:
        return {'ok': False}

    best_power = 0.0
    best_chain = None
    expansions = 0

    max_depth = min(k, 5)
    for depth in range(2, max_depth + 1):
        next_frontier = []
        # 扩展每条前沿链
        for chain, score, last_pw in frontier:
            last = chain[-1]
            for pos in pool:
                if any(np.allclose(pos, c, atol=1e-6) for c in chain):
                    continue
                if not env.check_los(last, pos):
                    continue
                new_chain = chain + [pos]
                # 评分：从深度>=2开始，计算到达“末端RIS”的级联功率作为评分
                pw_score = 0.0
                try:
                    if len(new_chain) >= 2:
                        pw_score = rfpower_at_last = _power_at_last_ris(ch_src, new_chain, env)
                    else:
                        z_ground = float(env.get_elevation(pos[0], pos[1]))
                        d_line = point_to_segment_distance_xy(np.array([pos[0], pos[1], 0.0]), p1, p2)
                        pw_score = z_ground - 0.001 * d_line
                except Exception:
                    pw_score = 0.0
                next_frontier.append((new_chain, pw_score, None))
                expansions += 1

        # 束宽裁剪
        next_frontier.sort(key=lambda t: t[1], reverse=True)
        next_frontier = next_frontier[:beam_width]

        # 在当前深度检查能否直达 B
        for chain, sc, _ in next_frontier:
            if env.check_los(chain[-1], p2):
                fp = evaluate_chain_final_power(ch_src, chain, ch_dst, env)
                if fp > best_power:
                    best_power = fp
                    best_chain = chain
        frontier = next_frontier

    return {
        'ok': best_chain is not None,
        'best_chain': best_chain,
        'best_power_w': float(best_power),
        'mode': f'beam_multi_{max_depth}',
        'beam_width': int(beam_width),
        'expansions': int(expansions),
        'step': float(step),
        'topk_each': int(topk_each),
    }


def _power_at_last_ris(ch_src, chain_positions: List[np.ndarray], env) -> float:
    """返回到达链路最后一个 RIS 的级联功率（W）。用于束搜索评分。"""
    if len(chain_positions) < 2:
        return 0.0
    source_obj = ch_src
    for i in range(len(chain_positions) - 1):
        ris_i = RIS(panel_id=-(i+1), position=chain_positions[i])
        ris_next = RIS(panel_id=-(i+2), position=chain_positions[i+1])
        if not env.check_los(ris_i.position, ris_next.position):
            return 0.0
        p_next = rf_propagation_model.calculate_ris_assisted_power(source_obj, ris_i, ris_next, env)
        if p_next <= 0:
            return 0.0
        src2 = type('RISSource', (), {})
        src2.position = chain_positions[i+1]
        src2.get_tx_power_dbm = (lambda p=p_next: 10 * np.log10(p * 1000.0))
        src2.get_reflection_gain = ris_next.get_reflection_gain
        src2.frequency_hz = getattr(ch_src, 'frequency_hz', None)
        source_obj = src2
    return 10 ** ((source_obj.get_tx_power_dbm() - 30) / 10)


# ------------------------------
# 主入口（自动模式）
# ------------------------------

def main():
    print("构建场景并加载 DEM（请确保 Working Directory= 项目/src，数据在 data/ 目录）...")
    wsn = WSN()
    print(f"集群数量: {len(wsn.clusters)}，RIS数量(文件中已有): {len(wsn.ris_panels)}")

    try:
        a = int(input("请输入源簇编号 (e.g., 0): ").strip())
        b = int(input("请输入目标簇编号 (e.g., 3): ").strip())
    except Exception:
        print("输入无效。")
        return

    # 参数（可按需调整）
    corridor_half_width = 1000.0   # 走廊半宽（米），覆盖山顶/脊线
    coarse_step_factor = 3.0       # 采样步长 = max(env.res * factor, 20~25m)
    install_height = 6.0           # 安装高度（地表以上）
    topk_each = 200                # 双/多 RIS 候选各端 Top-K
    beam_width = 40                # 多 RIS 束搜索束宽

    # 1) 视域交集 单RIS
    print("尝试：视域交集 单RIS ...")
    res = choose_single_ris_viewshed(wsn, a, b, corridor_half_width, install_height, coarse_step_factor)
    if res.get('ok'):
        pos = res['best_pos']
        print("\n— 最优 RIS 位置（视域交集 单RIS）—")
        print(f"坐标: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m")
        print(f"预估接收功率（A→RIS→B）: {res['best_power_w']:.6e} W")
        print(f"模式: {res['mode']}；步长={res['step']:.1f} m；评估={res['evaluated']}；可行={res['feasible']}")
        return
    else:
        print(f"视域交集 单RIS 未找到（可行={res.get('feasible',0)}）。")

    # 2) 走廊 单RIS（原方案）
    print("尝试：走廊 单RIS ...")
    res2 = choose_single_ris_corridor(wsn, a, b, corridor_half_width, install_height, coarse_step_factor)
    if res2.get('ok'):
        pos = res2['best_pos']
        print("\n— 最优 RIS 位置（走廊 单RIS）—")
        print(f"坐标: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m")
        print(f"预估接收功率（A→RIS→B）: {res2['best_power_w']:.6e} W")
        print(f"模式: {res2['mode']}；步长={res2['step']:.1f} m；评估={res2['evaluated']}；可行={res2['feasible']}")
        return
    else:
        print(f"走廊 单RIS 未找到（评估={res2.get('evaluated',0)}；可行={res2.get('feasible',0)}）。")

    # 3) 双RIS 回退
    print("尝试：视域 双RIS 回退 ...")
    res3 = choose_two_ris_viewsheds(wsn, a, b, corridor_half_width, install_height, coarse_step_factor, topk_each)
    if res3.get('ok'):
        p1, p2 = res3['best_pos_pair']
        print("\n— 最优 RIS 位置对（双RIS）—")
        print(f"RIS1: [{p1[0]:.2f}, {p1[1]:.2f}, {p1[2]:.2f}] m")
        print(f"RIS2: [{p2[0]:.2f}, {p2[1]:.2f}, {p2[2]:.2f}] m")
        print(f"预估接收功率（A→RIS1→RIS2→B）: {res3['best_power_w']:.6e} W")
        print(f"模式: {res3['mode']}；步长={res3['step']:.1f} m；配对检查={res3['pairs_checked']}；TopK={res3['topk_each']}")
        return
    else:
        print(f"双RIS 未找到可行配对（检查组合={res3.get('pairs_checked',0)}）。")

    # 4) 多RIS（3~5）束搜索
    for k in [3, 4, 5]:
        print(f"尝试：多RIS 束搜索（最多 {k} 个 RIS）...")
        resk = choose_k_ris_chain(wsn, a, b, k, corridor_half_width, install_height, coarse_step_factor, topk_each, beam_width)
        if resk.get('ok'):
            chain = resk['best_chain']
            print(f"\n— 最优 RIS 链（{len(chain)} 个）—")
            for idx, p in enumerate(chain, 1):
                print(f"RIS{idx}: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}] m")
            print(f"预估接收功率（A→...→B）: {resk['best_power_w']:.6e} W")
            print(f"模式: {resk['mode']}；束宽={resk['beam_width']}；扩展数={resk['expansions']}；步长={resk['step']:.1f} m；TopK={resk['topk_each']}")
            return
        else:
            print("该层数未找到可行链。继续提升层数...")

    print("仍未找到可行方案。建议增大走廊半宽、提高安装高度、减小步长，或放宽束宽/TopK。")


if __name__ == '__main__':
    main()
