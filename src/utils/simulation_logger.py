import os
from datetime import datetime
import numpy as np

class SimulationLogger:
    def __init__(self, log_dir="logs"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"log_{timestamp}.txt"
        self.log_file_path = os.path.join(log_dir, filename)
        
        # In-memory buffer to store log strings
        self.log_buffer = []
        
        self._write_header()

    def _write_header(self):
        self.log_buffer.append("# WSN_S3 Simulation Log\n\n")
        self.log_buffer.append("This log details the step-by-step events of the simulation, including scheduling, routing, and energy distribution.\n")

    def log_step(self, t_step, current_time):
        self.log_buffer.append(f"\n---\n\n## Step {t_step} (Time: {current_time:.2f}s)\n\n")

    def log_scheduling(self, actions):
        rf_target = actions.get('rf_target')
        mrc_transmitters = actions.get('mrc_transmitters', [])
        
        log_str = "### 1. Scheduling Decision\n"
        if rf_target:
            log_str += f"- **RF Target**: `{rf_target.node_id}` (Energy: {rf_target.current_energy:.4f} J)\n"
        else:
            log_str += "- **RF Target**: None\n"
            
        if mrc_transmitters:
            transmitter_ids = [ch.node_id for ch in mrc_transmitters]
            log_str += f"- **MRC Transmitters**: {', '.join(f'`{tid}`' for tid in transmitter_ids)}\n"
        else:
            log_str += "- **MRC Transmitters**: None\n"
            
        self.log_buffer.append(log_str)

    def log_routing(self, path, power):
        log_str = "### 2. Energy Routing\n"
        if power > 0 and path:
            path_str = ' -> '.join([p.node_id if hasattr(p, 'node_id') else f"RIS-{p.panel_id}" if hasattr(p, 'panel_id') else "RF_TX" for p in path])
            log_str += f"- **Optimal Path**: `{path_str}`\n"
            log_str += f"- **Delivered Power**: `{power * 1e6:.2f}` uW\n"
        else:
            log_str += "- **Result**: No viable energy path found or no power transferred.\n"
        self.log_buffer.append(log_str)

    def log_energy_transfer(self, rf_target, rf_sent_energy_j, rf_delivered_energy_j, mrc_entries, sensor_tx_consumption=None):
        log_str = "### 3. Energy & Information Flow\n"
        log_str += "#### Energy Flow\n"
        any_energy_flow = False
        # RF long-range
        if rf_target and (rf_sent_energy_j is not None):
            eta_rf = (rf_delivered_energy_j / rf_sent_energy_j) if rf_sent_energy_j > 0 else 0.0
            log_str += (
                f"- RF_TX -> `{rf_target.node_id}`: Sent `{rf_sent_energy_j:.6f}` J, "
                f"Delivered `{rf_delivered_energy_j:.6f}` J, Efficiency `{eta_rf:.6e}`.\n"
            )
            any_energy_flow = True
        
        # MRC local transfers
        if mrc_entries:
            for entry in mrc_entries:
                ch_id = entry.get('ch_id')
                sent_j = entry.get('sent_j', 0.0)
                delivered_j = entry.get('delivered_j', 0.0)
                eta = (delivered_j / sent_j) if sent_j > 0 else 0.0
                log_str += (
                    f"- `{ch_id}` (MRC) -> cluster nodes: Sent `{sent_j:.6f}` J, "
                    f"Delivered `{delivered_j:.6f}` J, Efficiency `{eta:.6e}`.\n"
                )
                any_energy_flow = True
        
        if not any_energy_flow:
            log_str += "- No energy transfer in this step.\n"

        log_str += "\n#### Information Flow & Consumption\n"
        if sensor_tx_consumption:
            for node_id, consumption in sensor_tx_consumption.items():
                log_str += f"- Node `{node_id}` -> CH: Sent data, consumed `{consumption:.6f}` J.\n"
        else:
            log_str += "- No information flow in this step.\n"

        self.log_buffer.append(log_str)

    def log_energy_balancing(self, cluster_id, average_energy, num_nodes):
        log_str = "### 3a. Energy Balancing Event\n"
        log_str += f"- **Cluster `{cluster_id}`**: Energy balanced across `{num_nodes}` nodes.\n"
        log_str += f"- **Result**: All nodes in cluster set to `{average_energy:.4f}` J.\n"
        self.log_buffer.append(log_str)

    def log_cluster_energy(self, wsn):
        log_str = "### 4. Cluster Energy Status\n"
        log_str += "| Cluster Head ID | CH Energy (J) | Avg. Node Energy (J) | Total Cluster Energy (J) |\n"
        log_str += "|:---:|:---:|:---:|:---:|\n"
        for cluster in wsn.clusters:
            ch = cluster.cluster_head
            node_energies = [node.current_energy for node in cluster.sensor_nodes]
            avg_node_energy = np.mean(node_energies) if node_energies else 0
            total_energy = ch.current_energy + np.sum(node_energies)
            log_str += f"| `{ch.node_id}` | {ch.current_energy:.4f} | {avg_node_energy:.4f} | {total_energy:.4f} |\n"
        self.log_buffer.append(log_str)

    def close(self):
        """Writes the entire log buffer to the file at once."""
        self.log_buffer.append("\n---\n\n**Simulation Finished.**")
        try:
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.log_buffer))
            print(f"Simulation log saved to {self.log_file_path}")
        except IOError as e:
            print(f"Error writing log file: {e}")
