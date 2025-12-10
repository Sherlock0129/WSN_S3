"""
交互式 3D 地形可视化（PyVista）
- 从 data/S3.csv 构建 DEM（线性插值 + 凸包外最近邻填补 + 湖面拉平）
- 以可旋转/缩放的 3D 方式显示地形（terrain 颜色）
- 叠加场景实体：Sink、各簇头（CH）、RIS 的位置点与标签

使用：
- 在 PyCharm 直接运行本文件（Working directory=项目/src）
- 数据默认路径：data/S3.csv、data/LAKE.csv、data/sink.csv
- 依赖：pyvista（若首次使用，请 pip install pyvista）
"""
import numpy as np
import pyvista as pv

from src.utils.scenario_loader import build_dem_from_S3, load_scenario

# 一键运行参数
S3_CSV_PATH = 'data/S3.csv'
LAKE_CSV_PATH = 'data/LAKE.csv'
GRID_RESOLUTION_M = None   # None 自动；或手动指定（例如 5.0）
SMOOTH_SIGMA = None        # None 不平滑；或 1.0/2.0 做轻度平滑

# 可视化参数
WARP_FACTOR = 1.0          # 地形夸张系数（>1 更夸张）
POINT_SIZE = 12            # 点大小
LABEL_FONT_SIZE = 12       # 标签字号


def build_structured_grid(xi: np.ndarray, yi: np.ndarray, Z: np.ndarray) -> pv.StructuredGrid:
    """基于规则网格坐标构建 PyVista StructuredGrid，并使用标量 'elevation' 进行 warping。"""
    Xg, Yg = np.meshgrid(xi, yi)
    # 构造一个平面网格（Z0=0），并把真实地形 Z 作为标量进行 warp_by_scalar
    Z0 = np.zeros_like(Z, dtype=float)
    grid = pv.StructuredGrid(Xg, Yg, Z0)
    # 注意 StructuredGrid 的点标量按 Fortran 顺序
    grid['elevation'] = Z.astype(float).ravel(order='F')
    return grid


def add_points_with_labels(plotter: pv.Plotter, pts: np.ndarray, labels: list, color: str):
    if pts.size == 0:
        return
    plotter.add_points(pts, color=color, point_size=POINT_SIZE, render_points_as_spheres=True)
    # 为避免过密导致遮挡，按需显示标签（数量多时可仅标部分）
    try:
        plotter.add_point_labels(pts, labels, point_size=POINT_SIZE, font_size=LABEL_FONT_SIZE, always_visible=False)
    except Exception:
        pass


def main():
    # 1) 构建 DEM
    dem_meta = build_dem_from_S3(
        s3_csv_path=S3_CSV_PATH,
        lake_csv_path=LAKE_CSV_PATH,
        grid_resolution_m=GRID_RESOLUTION_M,
        method='linear',
        fill='nearest',
        smooth_sigma=SMOOTH_SIGMA,
    )
    Z = dem_meta['dem']
    xi = dem_meta['x_coords']
    yi = dem_meta['y_coords']

    # 2) 构建 StructuredGrid 并进行 warping
    grid = build_structured_grid(xi, yi, Z)
    warped = grid.warp_by_scalar('elevation', factor=WARP_FACTOR)

    # 3) 载入场景实体（Sink / RF(=CH) / RIS）并组装坐标
    scn = load_scenario(s3_csv_path=S3_CSV_PATH, sink_csv_path='data/sink.csv')
    sink = np.array(scn['sink_pos'], dtype=float).reshape(1, 3)
    rfs = np.array(scn['rf_positions'], dtype=float) if scn['rf_positions'] else np.zeros((0, 3))
    riss = np.array(scn['ris_positions'], dtype=float) if scn['ris_positions'] else np.zeros((0, 3))

    # 将簇头（CH）高度对齐到地表（避免出现“低于地面”的情况）
    def sample_elevation(x: float, y: float) -> float:
        # 使用最近邻在规则网格上采样 DEM 高程
        ix = int(np.argmin(np.abs(xi - x)))
        iy = int(np.argmin(np.abs(yi - y)))
        ix = max(0, min(ix, len(xi) - 1))
        iy = max(0, min(iy, len(yi) - 1))
        return float(Z[iy, ix])

    if rfs.size > 0:
        for k in range(rfs.shape[0]):
            rfs[k, 2] = sample_elevation(rfs[k, 0], rfs[k, 1])

    # 4) 绘制
    pv.set_plot_theme('document')
    plotter = pv.Plotter()
    plotter.add_mesh(warped, cmap='terrain', show_edges=False, scalar_bar_args={'title': 'Elevation (m)'})

    # Sink
    add_points_with_labels(plotter, sink, ['SINK'], color='red')
    # 簇头（RF）
    rf_labels = [f'CH_{i}' for i in range(len(rfs))]
    add_points_with_labels(plotter, rfs, rf_labels, color='royalblue')
    # RIS
    ris_labels = [f'RIS_{i}' for i in range(len(riss))]
    add_points_with_labels(plotter, riss, ris_labels, color='orange')

    plotter.add_axes()
    plotter.show_bounds(grid='front', location='outer', color='black')
    plotter.add_text('Interactive 3D DEM with Sink / CH / RIS', position='upper_left', font_size=14)
    plotter.show()


if __name__ == '__main__':
    main()
