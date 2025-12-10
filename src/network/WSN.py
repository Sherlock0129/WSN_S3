"""
Main WSN class to initialize and manage all simulation components.
"""

import numpy as np

from src.core.Environment import Environment
from src.core.RFTransmitter import RFTransmitter
from src.core.RIS import RIS
from src.network.Cluster import Cluster
from src.config.simulation_config import WSNConfig, SimConfig
from src.utils.scenario_loader import load_scenario, build_dem_from_S3

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
        
        # Create the main RF power transmitter at sink position
        self.rf_transmitter = RFTransmitter(position=np.array(scenario['sink_pos']))
        
        # Create the RIS panels from file (one RIS per entry)
        self.ris_panels = []
        for i, pos in enumerate(scenario['ris_positions']):
            self.ris_panels.append(RIS(panel_id=i, position=np.array(pos)))
            
        # Create the clusters using RF positions (each RF represents one cluster head location)
        self.clusters = []
        for i, ch_pos in enumerate(scenario['rf_positions']):
            self.clusters.append(Cluster(cluster_id=i, center_position=np.array(ch_pos)))
        
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
