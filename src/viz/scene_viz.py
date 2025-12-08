"""
三维/二维场景可视化：
- 读取配置中的 Sink、RIS、簇头坐标
- 若已加载真实高度热力图，叠加作为背景；否则使用简化 DEM
- 简单标注视距（LoS）/非视距（NLoS）链路示意
"""

import numpy as np
import matplotlib.pyplot as plt

from src.config.simulation_config import SinkConfig, RISConfig, WSNConfig
from src.core.Environment import Environment


def _as_np(arr):
    return np.array(arr, dtype=float)


def plot_scene(show=True, save_path=None):
    """
    绘制场景的二维俯视与三维散点图，用于快速核对坐标与视距关系。
    """
    env = Environment()

    sink = _as_np(SinkConfig.POSITION)
    ris_list = [_as_np(p) for p in RISConfig.POSITIONS]
    ch_list = [_as_np(p) for p in WSNConfig.CLUSTER_HEAD_POSITIONS]

    # ---------- 二维俯视图 ----------
    fig = plt.figure(figsize=(14, 6))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    # 背景 DEM
    if env.dem is not None:
        extent = [0, env.width, 0, env.height]
        ax2d.imshow(
            np.flipud(env.dem),
            extent=extent,
            origin="lower",
            cmap="terrain",
            alpha=0.85,
        )
    ax2d.set_title("二维俯视（带海拔底图）")
    ax2d.set_xlabel("X / m")
    ax2d.set_ylabel("Y / m")

    # 绘制节点
    ax2d.scatter(sink[0], sink[1], c="red", marker="s", s=80, label="Sink")
    ax2d.scatter([p[0] for p in ris_list], [p[1] for p in ris_list], c="orange", marker="^", s=80, label="RIS")
    ax2d.scatter([p[0] for p in ch_list], [p[1] for p in ch_list], c="royalblue", marker="o", s=60, label="ClusterHead")

    # 简单连线：CH -> Sink 与 CH -> RIS1（可根据 LoS 着色）
    for ch in ch_list:
        los = env.check_los(sink, ch)
        ax2d.plot([sink[0], ch[0]], [sink[1], ch[1]], color="lime" if los else "gray", alpha=0.6, linewidth=1)
        for ris in ris_list:
            los_ris = env.check_los(ch, ris)
            ax2d.plot([ch[0], ris[0]], [ch[1], ris[1]], color="cyan" if los_ris else "silver", alpha=0.4, linewidth=0.8)

    ax2d.legend(loc="upper right")
    ax2d.set_aspect("equal")

    # ---------- 三维散点 ----------
    ax3d.set_title("三维散点示意")
    ax3d.set_xlabel("X / m")
    ax3d.set_ylabel("Y / m")
    ax3d.set_zlabel("Z / m")

    ax3d.scatter(sink[0], sink[1], sink[2], c="red", marker="s", s=80, label="Sink")
    ax3d.scatter([p[0] for p in ris_list], [p[1] for p in ris_list], [p[2] for p in ris_list],
                 c="orange", marker="^", s=80, label="RIS")
    ax3d.scatter([p[0] for p in ch_list], [p[1] for p in ch_list], [p[2] for p in ch_list],
                 c="royalblue", marker="o", s=60, label="ClusterHead")

    ax3d.legend(loc="upper left")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_scene(show=True, save_path=None)

