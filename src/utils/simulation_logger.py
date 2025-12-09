import os
from datetime import datetime
import numpy as np

class SimulationLogger:
    def __init__(self, log_dir="logs"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"log_{timestamp}.md"

        self.log_file_path = os.path.join(log_dir, filename)
        # Start with an empty file
        self.log_file = open(self.log_file_path, "w", encoding="utf-8")
        self._write_header()
        self.log_file.close() # Close and reopen in append mode
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")

    def _write_header(self):
        self.log_file.write("# WSN_S3 Simulation Log\n\n")
        self.log_file.write("This log details the step-by-step events of the simulation, including scheduling, routing, and energy distribution.\n")

    def log_step(self, t_step, current_time):
        self.log_file.write(f"\n---\n\n## Step {t_step} (Time: {current_time:.2f}s)\n\n")

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
            
        self.log_file.write(log_str)

    def log_routing(self, path, power):
        log_str = "### 2. Energy Routing\n"
        if power > 0 and path:
            path_str = ' -> '.join([p.node_id if hasattr(p, 'node_id') else f"RIS-{p.panel_id}" if hasattr(p, 'panel_id') else "RF_TX" for p in path])
            log_str += f"- **Optimal Path**: `{path_str}`\n"
            log_str += f"- **Delivered Power**: `{power * 1e6:.2f}` uW\n"
        else:
            log_str += "- **Result**: No viable energy path found or no power transferred.\n"
        self.log_file.write(log_str)

    def log_energy_transfer(self, rf_target, max_power_w, mrc_transmitting_chs, time_step):
        log_str = "### 3. Energy & Information Flow\n"
        log_str += "#### Energy Flow\n"
        if rf_target and max_power_w > 0:
            energy_gained = max_power_w * time_step
            log_str += f"- `RF_TX` -> `{rf_target.node_id}`: Transferred `{energy_gained:.6f}` J.\n"
        
        if mrc_transmitting_chs:
            for ch in mrc_transmitting_chs:
                log_str += f"- `{ch.node_id}` -> its cluster nodes: MRC local transfer initiated.\n"
        
        if not (rf_target and max_power_w > 0) and not mrc_transmitting_chs:
             log_str += "- No energy transfer in this step.\n"

        log_str += "\n#### Information Flow (Conceptual)\n"
        log_str += "- Sensor Nodes -> Cluster Heads: Data packets (e.g., sensor readings).\n"
        log_str += "- Cluster Heads -> Sink: Aggregated data.\n"

        self.log_file.write(log_str)

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
        self.log_file.write(log_str)
        self.log_file.flush()

    def close(self):
        self.log_file.write("\n---\n\n**Simulation Finished.**")
        self.log_file.close()

