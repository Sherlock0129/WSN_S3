"""
Environment class to manage the simulation area and terrain.
"""

import os

import numpy as np
from PIL import Image

from src.config.simulation_config import EnvConfig

class Environment:
    def __init__(self):
        """
        Initializes the simulation environment, including terrain model.
        """
        self.width = EnvConfig.AREA_WIDTH
        self.height = EnvConfig.AREA_HEIGHT
        self.use_terrain = EnvConfig.ENABLE_TERRAIN_MODEL
        self.resolution = EnvConfig.TERRAIN_RESOLUTION
        self.max_elevation = EnvConfig.TERRAIN_MAX_ELEVATION
        self.m_per_pixel = None  # 若使用热力图，将在加载时覆盖
        
        self.dem = None
        if self.use_terrain:
            loaded = self._load_heightmap_if_available()
            if not loaded:
                self._generate_terrain()

    def _generate_terrain(self):
        """
        生成简化 DEM（若未提供真实高度图时的回退方案）。
        """
        grid_width = int(self.width / self.resolution)
        grid_height = int(self.height / self.resolution)
        self.dem = np.zeros((grid_height, grid_width))

        peaks = [
            (int(grid_width * 0.25), int(grid_height * 0.4), self.max_elevation * 0.9),
            (int(grid_width * 0.7), int(grid_height * 0.6), self.max_elevation),
            (int(grid_width * 0.5), int(grid_height * 0.2), self.max_elevation * 0.75),
        ]

        x = np.arange(0, grid_width)
        y = np.arange(0, grid_height)
        xx, yy = np.meshgrid(x, y)

        for px, py, pz in peaks:
            dist_sq = (xx - px) ** 2 + (yy - py) ** 2
            sigma_sq = (grid_width * 0.1) ** 2
            self.dem += pz * np.exp(-dist_sq / (2 * sigma_sq))
        
        print("使用内置简化 DEM。")

    def _load_heightmap_if_available(self):
        """
        如果提供了高度热力图，则将其转换为 DEM：
        - 使用图中 500m 比例尺换算像素与米。
        - 将亮度线性映射到海拔范围 [HEIGHTMAP_MIN_ELEV, HEIGHTMAP_MAX_ELEV]。
        """
        path = EnvConfig.HEIGHTMAP_PATH
        if not path or not os.path.exists(path):
            return False
        if not EnvConfig.HEIGHTMAP_SCALE_BAR_PIXELS:
            print("HEIGHTMAP_SCALE_BAR_PIXELS 未设置，无法依据比例尺计算分辨率，回退内置 DEM。")
            return False

        # 计算米/像素并覆盖场景尺寸
        self.m_per_pixel = EnvConfig.HEIGHTMAP_SCALE_BAR_METERS / EnvConfig.HEIGHTMAP_SCALE_BAR_PIXELS
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"高度图加载失败: {exc}，回退内置 DEM。")
            return False

        arr = np.asarray(img).astype(np.float32) / 255.0
        # 简单亮度提取（RGB 取平均）并线性映射到海拔
        gray = arr.mean(axis=2)
        elev_min, elev_max = EnvConfig.HEIGHTMAP_MIN_ELEV, EnvConfig.HEIGHTMAP_MAX_ELEV
        dem = elev_min + gray * (elev_max - elev_min)

        # 保存 DEM，调整区域尺寸与分辨率
        self.dem = dem
        self.height, self.width = dem.shape[0] * self.m_per_pixel, dem.shape[1] * self.m_per_pixel
        self.resolution = self.m_per_pixel

        print(f"已加载高度热力图 DEM：分辨率约 {self.m_per_pixel:.2f} m/px，区域大小 {self.width:.1f}x{self.height:.1f} m。")
        return True

    def get_elevation(self, x, y):
        """
        Get the terrain elevation at a given (x, y) coordinate.
        """
        if not self.use_terrain or self.dem is None:
            return 0
        
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)

        grid_height, grid_width = self.dem.shape
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            return self.dem[grid_y, grid_x]
        return 0

    def check_los(self, point1, point2):
        """
        Checks for Line-of-Sight (LoS) between two 3D points, considering terrain.

        Args:
            point1 (np.array): 3D coordinates of the first point [x, y, z].
            point2 (np.array): 3D coordinates of the second point [x, y, z].

        Returns:
            bool: True if LoS exists, False otherwise.
        """
        if not self.use_terrain:
            return True

        p1 = np.array(point1)
        p2 = np.array(point2)

        # Ensure points are above terrain
        if p1[2] < self.get_elevation(p1[0], p1[1]) or p2[2] < self.get_elevation(p2[0], p2[1]):
            return False

        num_steps = int(np.linalg.norm(p2 - p1) / self.resolution)
        if num_steps < 2:
            return True

        path_vector = p2 - p1
        for i in range(1, num_steps):
            t = i / num_steps
            path_point = p1 + t * path_vector
            terrain_h = self.get_elevation(path_point[0], path_point[1])
            if path_point[2] < terrain_h:
                return False  # Path is obstructed by terrain
        
        return True

