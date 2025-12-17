"""
Main WSN class to initialize and manage all simulation components.
"""

import numpy as np

from src.core.Environment import Environment
from src.core.RFTransmitter import RFTransmitter
from src.core.RIS import RIS
from src.network.Cluster import Cluster
from src.config.simulation_config import WSNConfig, SimConfig
from src.utils.scenario_loader import load_scenario, build_dem_from_S3, load_lakes

class WSN:
    def __init__(self):
        """
        Initializes the entire Wireless Sensor Network simulation environment.
        Now loads all coordinates from CSV files via scenario_loader:
          - S3.csv defines the local metric coordinate frame (0 as origin; 0→-1 as X; 0→-2 as Y)
          - sink.csv provides positions for sink (TX), RF (cluster heads), RIS panels
        """
        # Set random seed for reproducibility
        np.random.seed(SimConfig.RANDOM_SEED)
        
        # Build DEM from S3.csv and create Environment with external DEM
        dem_meta = build_dem_from_S3()
        self.environment = Environment(
            dem=dem_meta['dem'],
            origin_xy=dem_meta['origin_xy'],
            resolution=dem_meta['resolution']
        )

        # Load scenario (positions in meters, local XYZ frame)
        scenario = load_scenario()
        # Persist scenario for allocation and mapping
        self.scenario = scenario
        
        # Create the main RF power transmitter at sink position
        self.rf_transmitter = RFTransmitter(position=np.array(scenario['sink_pos']))
        
        # Create the RIS panels from file (one RIS per entry)
        self.ris_panels = []
        for i, pos in enumerate(scenario['ris_positions']):
            self.ris_panels.append(RIS(panel_id=i, position=np.array(pos)))

        # Allocate sensor nodes per cluster based on configuration
        nodes_per_cluster_list = self._allocate_nodes_per_cluster()
            
        # Create the clusters using RF positions (each RF represents one cluster head location)
        self.clusters = []
        # Prefer explicit indexes if provided; otherwise, mark the first N clusters as solar
        solar_indexes = set(getattr(WSNConfig, 'SOLAR_CLUSTER_INDEXES', []))
        fallback_num_solar = len(getattr(WSNConfig, 'SOLAR_CLUSTER_HEAD_POSITIONS', []))
        for i, ch_pos in enumerate(scenario['rf_positions']):
            has_solar = (i in solar_indexes) if solar_indexes else (i < fallback_num_solar)
            node_count = nodes_per_cluster_list[i]
            self.clusters.append(Cluster(cluster_id=i,
                                         center_position=np.array(ch_pos),
                                         has_solar_nodes=has_solar,
                                         nodes_count=node_count))
        
        # Adjust node z-coordinates to be on the terrain surface (if terrain enabled)
        self._place_nodes_on_terrain()
        
        print("WSN initialized with (from CSVs):")
        print(f"- {len(self.clusters)} clusters (RF entries)")
        print(f"- {len(self.ris_panels)} RIS panels")
        print(f"- RF Transmitter (sink) at {self.rf_transmitter.position}")

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

    def _allocate_nodes_per_cluster(self):
        """
        Calculates the number of sensor nodes for each cluster based on configuration.
        - If MANUAL_NODES_PER_CLUSTER is set, it uses the specified list.
        - Else if ALLOCATE_BY_LAKE_AREA is True, it distributes nodes based on lake area.
        - Otherwise, it assigns a fixed number of nodes to each cluster.
        """
        if hasattr(WSNConfig, 'MANUAL_NODES_PER_CLUSTER') and WSNConfig.MANUAL_NODES_PER_CLUSTER is not None:
            print(f"[WSN] Using manual node allocation: {WSNConfig.MANUAL_NODES_PER_CLUSTER}")
            return WSNConfig.MANUAL_NODES_PER_CLUSTER

        if not WSNConfig.ALLOCATE_BY_LAKE_AREA:
            return [WSNConfig.NODES_PER_CLUSTER] * WSNConfig.NUM_CLUSTERS

        lakes = load_lakes()
        if not lakes:
            print("[WSN] Warning: load_lakes() returned empty. Fallback to uniform allocation.")
            return [WSNConfig.NODES_PER_CLUSTER] * WSNConfig.NUM_CLUSTERS

        # Map each CH position to a lake by point-in-polygon; fallback to nearest centroid
        ch_xy = [np.array(self.scenario['rf_positions'][i])[:2] for i in range(len(self.scenario['rf_positions']))]
        mapped_names = []
        mapped_areas = []
        for xy in ch_xy:
            name = None
            area = None
            for lk in lakes:
                if lk['path'].contains_point((xy[0], xy[1])):
                    name = lk['name']
                    area = lk['area']
                    break
            if name is None:
                # nearest centroid
                dists = [np.linalg.norm(xy - lk['centroid']) for lk in lakes]
                j = int(np.argmin(dists))
                name = lakes[j]['name']
                area = lakes[j]['area']
            mapped_names.append(name)
            mapped_areas.append(area)

        total_nodes = WSNConfig.TOTAL_SENSOR_NODES or (WSNConfig.NUM_CLUSTERS * WSNConfig.NODES_PER_CLUSTER)
        weights = np.array(mapped_areas, dtype=float)

        # Solar weighting
        solar_weight_factor = getattr(WSNConfig, 'SOLAR_WEIGHT_FACTOR', 1.5)
        solar_indexes = getattr(WSNConfig, 'SOLAR_CLUSTER_INDEXES', [])
        for idx in solar_indexes:
            if 0 <= idx < len(weights):
                weights[idx] *= solar_weight_factor

        total_weight = float(np.sum(weights))
        if total_weight <= 0:
            print("[WSN] Warning: total lake area weight is zero. Fallback to uniform allocation.")
            return [WSNConfig.NODES_PER_CLUSTER] * WSNConfig.NUM_CLUSTERS

        proportions = weights / total_weight
        raw = proportions * total_nodes
        allocated_nodes = np.floor(raw).astype(int)
        # Distribute remainder by largest fractional part
        remainder = int(total_nodes - np.sum(allocated_nodes))
        if remainder > 0:
            frac_order = np.argsort(-(raw - allocated_nodes))
            for k in range(remainder):
                allocated_nodes[frac_order[k]] += 1

        # Ensure minimum 1 per cluster
        for i in range(len(allocated_nodes)):
            if allocated_nodes[i] == 0:
                j = int(np.argmax(allocated_nodes))
                if allocated_nodes[j] > 1:
                    allocated_nodes[j] -= 1
                    allocated_nodes[i] += 1

        # Log mapping
        print("[WSN] Lake-based allocation (CH_i -> lake, area[m^2], solar, nodes):")
        solar_set = set(getattr(WSNConfig, 'SOLAR_CLUSTER_INDEXES', []))
        for i, (nm, ar, cnt) in enumerate(zip(mapped_names, mapped_areas, allocated_nodes.tolist())):
            print(f"  - CH_{i} -> {nm}, area={ar:.1f}, solar={i in solar_set}, nodes={cnt}")

        return allocated_nodes.tolist()
