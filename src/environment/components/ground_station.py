# -*- coding: utf-8 -*-
"""
This module defines the GroundStation class.
"""

from typing import Dict, Any
import numpy as np

from src.physics.orbits import geodetic_to_ecef

class GroundStation:
    """Represents a single ground station in the simulation."""
    def __init__(self, gs_id: int, lat_deg: float, lon_deg: float, config: Dict[str, Any]):
        """
        Initializes a GroundStation object.

        Args:
            gs_id (int): A unique identifier for the ground station.
            lat_deg (float): The latitude of the station in degrees.
            lon_deg (float): The longitude of the station in degrees.
            config (Dict[str, Any]): A dictionary containing configuration parameters,
                                     such as 'compute_gflops'.
        """
        self.id = gs_id
        self.name = f"GS-{gs_id}"
        self.latitude = lat_deg
        self.longitude = lon_deg
        self.compute_gflops = config.get('compute_gflops', 1000.0)
        
        # Convert geodetic coordinates to ECEF for distance calculations
        self.position_ecef = geodetic_to_ecef(lat_deg, lon_deg, 0.0) # Assume altitude is 0

    def __repr__(self) -> str:
        return f"GroundStation(id={self.id}, lat={self.latitude:.2f}, lon={self.longitude:.2f})"

