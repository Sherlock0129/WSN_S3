import numpy as np
import matplotlib.pyplot as plt

from src.utils.scenario_loader import build_dem_from_S3

# 一键运行参数（可在 PyCharm 中直接点运行本文件，无需命令行）
S3_CSV_PATH = 'data/S3.csv'
LAKE_CSV_PATH = 'data/LAKE.csv'
GRID_RESOLUTION_M = None   # 例如 5.0；None 表示自动根据点云密度估计（2~10m）
SMOOTH_SIGMA = None        # 例如 1.0；None 表示不平滑


def plot_dem_and_points(s3_csv: str, lake_csv: str, res: float | None, smooth: float | None):
    meta = build_dem_from_S3(
        s3_csv_path=s3_csv,
        lake_csv_path=lake_csv,
        grid_resolution_m=res,
        method='linear',
        fill='nearest',
        smooth_sigma=smooth,
    )
    Z = meta['dem']
    xi = meta['x_coords']
    yi = meta['y_coords']
    pts = meta['points']  # (N,3) in local meters

    # 2D 俯视图
    fig = plt.figure(figsize=(14, 6))
    ax2d = fig.add_subplot(1, 2, 1)
    im = ax2d.imshow(
        np.flipud(Z),
        extent=[xi.min(), xi.max(), yi.min(), yi.max()],
        origin='lower',
        cmap='terrain',
        aspect='equal'
    )
    plt.colorbar(im, ax=ax2d, shrink=0.8, label='Elevation (m)')
    ax2d.scatter(pts[:, 0], pts[:, 1], c='k', s=3, alpha=0.5, label='S3 points')
    ax2d.set_title('DEM (linear interp + nearest fill) with S3 points overlay')
    ax2d.set_xlabel('X (m)')
    ax2d.set_ylabel('Y (m)')
    ax2d.legend(loc='upper right')

    # 3D 视图（为加速可下采样）
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    Xg, Yg = np.meshgrid(xi, yi)
    step = max(1, int(max(len(xi), len(yi)) / 200))  # 控制到 ~200 采样/轴
    Xd = Xg[::step, ::step]
    Yd = Yg[::step, ::step]
    Zd = Z[::step, ::step]
    ax3d.plot_surface(Xd, Yd, Zd, cmap='terrain', rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.9)
    if pts.shape[0] > 0:
        sel = slice(None, None, max(1, pts.shape[0] // 5000))  # 最多 ~5k 点
        ax3d.scatter(pts[sel, 0], pts[sel, 1], pts[sel, 2], c='k', s=1, alpha=0.4)
    ax3d.set_title('DEM surface (with point cloud)')
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 直接运行，无需命令行参数
    plot_dem_and_points(S3_CSV_PATH, LAKE_CSV_PATH, GRID_RESOLUTION_M, SMOOTH_SIGMA)
