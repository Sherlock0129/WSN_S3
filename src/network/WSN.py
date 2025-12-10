"""
Main WSN class to initialize and manage all simulation components.
"""

import numpy as np
import csv, re, math

from src.core.Environment import Environment
from src.core.RFTransmitter import RFTransmitter
from src.core.RIS import RIS
from src.network.Cluster import Cluster
from src.config.simulation_config import WSNConfig, RISConfig, SimConfig

def _parse_wkt_polygon_z(wkt: str):
    m = re.search(r"\(\((.*)\)\)", wkt)
    if not m:
        return []
    coords_str = m.group(1)
    pts = []
    for token in coords_str.split(','):
        parts = token.strip().split()
        if len(parts) >= 2:
            # WKT here seems to be "lon lat [z]"
            lon = float(parts[0]) if parts[0] != 'POLYGON' else float(parts[1])
            lat = float(parts[1]) if parts[0] != 'POLYGON' else float(parts[2])
            pts.append((lon, lat))
    return pts

def _polygon_area_m2_equirect(points_lonlat):
    if not points_lonlat or len(points_lonlat) < 3:
        return 0.0
    # remove duplicate last point if equal to first
    if points_lonlat[0] == points_lonlat[-1]:
        pts = points_lonlat[:-1]
    else:
        pts = points_lonlat
    # Equirectangular projection around reference latitude
    R = 6371000.0
    lon0, lat0 = pts[0]
    lat0_rad = math.radians(lat0)
    xy = []
    lon0_rad = math.radians(lon0)
    for lon, lat in pts:
        lon_rad = math.radians(lon)
        lat_rad = math.radians(lat)
        x = R * (lon_rad - lon0_rad) * math.cos(lat0_rad)
        y = R * (lat_rad - math.radians(lat0))
        xy.append((x, y))
    # Shoelace formula
    area2 = 0.0
    n = len(xy)
    for i in range(n):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5

def _load_lake_areas(csv_path: str):
    areas = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('name')
                wkt = row.get('WKT', '')
                pts = _parse_wkt_polygon_z(wkt)
                area = _polygon_area_m2_equirect(pts)
                if name:
                    areas[name.strip().strip('"')] = area
    except Exception as e:
        print(f"[WSN] Warning: failed to load lake areas from {csv_path}: {e}")
    return areas

def _allocate_nodes_by_area(area_map: dict, cluster_names: list, total_nodes: int):
    # Build area list in the order of cluster_names
    areas = [max(0.0, area_map.get(name, 0.0)) for name in cluster_names]
    total_area = sum(areas)
    n_clusters = len(cluster_names)
    if total_area <= 0:
        # fallback: equal allocation
        base = total_nodes // n_clusters
        rem = total_nodes - base * n_clusters
        alloc = [base] * n_clusters
        for i in range(rem):
            alloc[i] += 1
        return alloc
    # Largest remainder method
    shares = [a * total_nodes / total_area for a in areas]
    floors = [int(math.floor(s)) for s in shares]
    allocated = sum(floors)
    rem = total_nodes - allocated
    remainders = [(i, shares[i] - floors[i]) for i in range(n_clusters)]
    remainders.sort(key=lambda x: x[1], reverse=True)
    for i in range(rem):
        floors[remainders[i][0]] += 1
    return floors

class WSN:
    def __init__(self):
        """
        Initializes the entire Wireless Sensor Network simulation environment.
        """
        # Set random seed for reproducibility
        np.random.seed(SimConfig.RANDOM_SEED)
        
        # Create the environment
        self.environment = Environment()
        
        # Create the main RF power transmitter
        self.rf_transmitter = RFTransmitter()
        
        # Create the RIS panels
        self.ris_panels = []
        for i in range(RISConfig.NUM_RIS_PANELS):
            ris = RIS(panel_id=i, position=RISConfig.POSITIONS[i])
            self.ris_panels.append(ris)
            
        # Create the clusters（按是否采集太阳能分两类创建）
        self.clusters = []
        # Precompute allocation by lake area if enabled
        solar_positions = list(getattr(WSNConfig, 'SOLAR_CLUSTER_HEAD_POSITIONS', []))
        nonsolar_positions = list(getattr(WSNConfig, 'NON_SOLAR_CLUSTER_HEAD_POSITIONS', []))
        n_clusters = len(solar_positions) + len(nonsolar_positions)

        alloc_by_area = [None] * n_clusters
        if getattr(WSNConfig, 'ALLOCATE_BY_LAKE_AREA', False) and \
           len(getattr(WSNConfig, 'LAKE_NAME_PER_CLUSTER', [])) == n_clusters:
            area_map = _load_lake_areas(getattr(WSNConfig, 'LAKE_CSV_PATH', 'src/data/LAKE.csv'))
            total_nodes = getattr(WSNConfig, 'TOTAL_SENSOR_NODES', None)
            if total_nodes is None:
                total_nodes = n_clusters * getattr(WSNConfig, 'NODES_PER_CLUSTER', 10)
            alloc_by_area = _allocate_nodes_by_area(
                area_map,
                getattr(WSNConfig, 'LAKE_NAME_PER_CLUSTER'),
                total_nodes
            )
            print(f"[WSN] Lake-area-based allocation (total {total_nodes} sensors): {alloc_by_area}")
        else:
            # Fallback: equal count per cluster
            equal_n = getattr(WSNConfig, 'NODES_PER_CLUSTER', 10)
            alloc_by_area = [equal_n] * n_clusters
            print(f"[WSN] Equal allocation per cluster: {equal_n} sensors each")

        cluster_id = 0
        alloc_idx = 0
        # 先创建“太阳能簇”
        if solar_positions:
            for pos in solar_positions:
                nodes_count = alloc_by_area[alloc_idx] if alloc_idx < len(alloc_by_area) else None
                cluster = Cluster(cluster_id=cluster_id, center_position=pos, has_solar_nodes=True, nodes_count=nodes_count)
                self.clusters.append(cluster)
                cluster_id += 1
                alloc_idx += 1
        # 再创建“非太阳能簇”
        if nonsolar_positions:
            for pos in nonsolar_positions:
                nodes_count = alloc_by_area[alloc_idx] if alloc_idx < len(alloc_by_area) else None
                cluster = Cluster(cluster_id=cluster_id, center_position=pos, has_solar_nodes=False, nodes_count=nodes_count)
                self.clusters.append(cluster)
                cluster_id += 1
                alloc_idx += 1
        
        # Adjust node z-coordinates to be on the terrain surface
        self._place_nodes_on_terrain()
        
        print("WSN initialized with:")
        print(f"- {len(self.clusters)} clusters")
        print(f"- {len(self.ris_panels)} RIS panels")
        print(f"- RF Transmitter at {self.rf_transmitter.position}")
        for c in self.clusters:
            print(f"  > Cluster {c.cluster_id} sensors: {len(c.sensor_nodes)} (solar={c.has_solar_nodes})")

    def _place_nodes_on_terrain(self):
        """
        Adjusts the z-coordinate of all nodes (CHs and sensors) to sit on the terrain.
        """
        if not self.environment.use_terrain:
            return
            
        for cluster in self.clusters:
            # Place cluster head
            ch_pos = cluster.cluster_head.position
            ch_pos[2] = self.environment.get_elevation(ch_pos[0], ch_pos[1]) + 1.5 # 1.5m above ground
            cluster.cluster_head.position = ch_pos
            
            # Place sensor nodes
            for node in cluster.sensor_nodes:
                node_pos = node.position
                node_pos[2] = self.environment.get_elevation(node_pos[0], node_pos[1]) + 1.0 # 1m above ground
                node.position = node_pos

    def get_all_nodes(self):
        """
        Returns a flat list of all sensor nodes and cluster heads in the network.
        """
        all_nodes = []
        for cluster in self.clusters:
            all_nodes.append(cluster.cluster_head)
            all_nodes.extend(cluster.sensor_nodes)
        return all_nodes

