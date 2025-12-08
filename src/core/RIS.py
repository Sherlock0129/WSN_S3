"""
Reconfigurable Intelligent Surface (RIS) class.
"""

import numpy as np

from src.config.simulation_config import RISConfig, SinkConfig

class RIS:
    def __init__(self, panel_id, position):
        """
        Initializes a single RIS panel.

        Args:
            panel_id (int): Unique ID for the RIS panel.
            position (np.array): 3D coordinates of the RIS panel's center.
        """
        self.panel_id = panel_id
        self.position = position
        self.num_elements_h = RISConfig.NUM_ELEMENTS_H
        self.num_elements_v = RISConfig.NUM_ELEMENTS_V
        self.total_elements = self.num_elements_h * self.num_elements_v
        
        # Phase shifts for each element, initialized to 0
        self.phase_shifts = np.zeros((self.num_elements_h, self.num_elements_v))
        
        # Wavelength calculation
        c = 3e8  # Speed of light
        self.wavelength = c / SinkConfig.FREQUENCY_HZ
        
        # Element spacing
        self.element_spacing = RISConfig.ELEMENT_SPACING_FACTOR * self.wavelength

    def configure_phases(self, incident_point, reflection_point):
        """
        Configures the phase shifts of the RIS elements to steer the beam from an
        incident point to a reflection point (beamforming).

        Args:
            incident_point (np.array): 3D coordinates of the source (e.g., RF Transmitter).
            reflection_point (np.array): 3D coordinates of the target (e.g., another RIS or Cluster Head).
        """
        for m in range(self.num_elements_h):
            for n in range(self.num_elements_v):
                # Position of element (m, n) relative to the RIS center, in 3D
                element_pos_3d = self.position + np.array([
                    (m - (self.num_elements_h - 1) / 2) * self.element_spacing,
                    (n - (self.num_elements_v - 1) / 2) * self.element_spacing,
                    0
                ])
                
                # Distances from the element to the incident and reflection points
                d_in = np.linalg.norm(element_pos_3d - incident_point)
                d_out = np.linalg.norm(element_pos_3d - reflection_point)
                
                # Required phase shift to align signals at the reflection point
                # The phase shift compensates for the path length difference
                phi_mn = (2 * np.pi / self.wavelength) * (d_in + d_out)
                
                # Quantize the phase shift
                num_levels = 2**RISConfig.PHASE_RESOLUTION_BITS
                quantized_phi = np.round(phi_mn * num_levels / (2 * np.pi)) * (2 * np.pi / num_levels)
                
                self.phase_shifts[m, n] = quantized_phi % (2 * np.pi)

    def get_reflection_gain(self):
        """
        Calculates the theoretical maximum gain of the RIS panel.
        This is proportional to the number of elements squared.
        A simplified model for anomalous reflection gain.
        """
        # Ideal gain is proportional to N^2, where N is total elements
        # The gain also depends on the element area and frequency.
        element_area = self.element_spacing**2
        aperture_area = self.total_elements * element_area
        gain = (4 * np.pi * aperture_area) / (self.wavelength**2)
        return 10 * np.log10(gain) # Return gain in dBi

    def __repr__(self):
        return f"RIS(ID={self.panel_id}, Position={self.position}, Elements={self.total_elements})"

