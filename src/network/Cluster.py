"""
Cluster class to manage a cluster head and its sensor nodes.
"""

import numpy as np

from src.core.ClusterHead import ClusterHead
from src.core.SensorNode import SensorNode
from src.config.simulation_config import WSNConfig, SensorNodeConfig

class Cluster:
    def __init__(self, cluster_id, center_position, has_solar_nodes: bool = False):
        """
        Initializes a single WSN cluster.

        Args:
            cluster_id (int): Unique ID for the cluster.
            center_position (np.array): 3D coordinates for the cluster's center.
            has_solar_nodes (bool): Whether sensor nodes in this cluster harvest solar energy.
        """
        self.cluster_id = cluster_id
        self.center_position = center_position
        self.has_solar_nodes = has_solar_nodes
        
        # Create the cluster head
        self.cluster_head = ClusterHead(node_id=f"CH_{cluster_id}", position=center_position)
        
        # Create the sensor nodes within the cluster radius
        self.sensor_nodes = []
        for i in range(WSNConfig.NODES_PER_CLUSTER):
            node_id = f"{cluster_id}-{i}"
            # Generate a random position within the cluster radius in 3D
            radius = WSNConfig.CLUSTER_RADIUS * np.sqrt(np.random.rand())
            theta = 2 * np.pi * np.random.rand()
            phi = np.arccos(2 * np.random.rand() - 1) # For uniform spherical distribution
            
            x = center_position[0] + radius * np.sin(phi) * np.cos(theta)
            y = center_position[1] + radius * np.sin(phi) * np.sin(theta)
            z = center_position[2] + radius * np.cos(phi)
            
            node_position = np.array([x, y, z])
            
            node = SensorNode(
                node_id=node_id,
                initial_energy=SensorNodeConfig.INITIAL_ENERGY_J,
                position=node_position,
                low_threshold=0.1, # Example
                high_threshold=0.9, # Example
                has_solar=self.has_solar_nodes
            )
            self.sensor_nodes.append(node)
            
    def __repr__(self):
        return f"Cluster(ID={self.cluster_id}, CH={self.cluster_head.node_id}, Nodes={len(self.sensor_nodes)}, SolarNodes={self.has_solar_nodes})"

