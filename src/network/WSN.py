"""
Main WSN class to initialize and manage all simulation components.
"""

import numpy as np

from src.core.Environment import Environment
from src.core.RFTransmitter import RFTransmitter
from src.core.RIS import RIS
from src.network.Cluster import Cluster
from src.config.simulation_config import WSNConfig, RISConfig, SimConfig

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
        cluster_id = 0
        # 先创建“太阳能簇”
        if hasattr(WSNConfig, 'SOLAR_CLUSTER_HEAD_POSITIONS'):
            for pos in WSNConfig.SOLAR_CLUSTER_HEAD_POSITIONS:
                cluster = Cluster(cluster_id=cluster_id, center_position=pos, has_solar_nodes=True)
                self.clusters.append(cluster)
                cluster_id += 1
        # 再创建“非太阳能簇”
        if hasattr(WSNConfig, 'NON_SOLAR_CLUSTER_HEAD_POSITIONS'):
            for pos in WSNConfig.NON_SOLAR_CLUSTER_HEAD_POSITIONS:
                cluster = Cluster(cluster_id=cluster_id, center_position=pos, has_solar_nodes=False)
                self.clusters.append(cluster)
                cluster_id += 1
        
        # Adjust node z-coordinates to be on the terrain surface
        self._place_nodes_on_terrain()
        
        print("WSN initialized with:")
        print(f"- {len(self.clusters)} clusters")
        print(f"- {len(self.ris_panels)} RIS panels")
        print(f"- RF Transmitter at {self.rf_transmitter.position}")

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

