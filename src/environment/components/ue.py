# -*- coding: utf-8 -*-
"""
This module defines the UserEquipment class, which represents a ground-based user.
"""

import numpy as np
from src.physics.orbits import geodetic_to_ecef

class UserEquipment:
    """Represents a single ground-based user equipment (UE)."""
    def __init__(self, ue_id: int, lat_deg: float, lon_deg: float, alt_km: float = 0.0):
        """
        Initializes a UserEquipment object.

        Args:
            ue_id (int): A unique identifier for the UE.
            lat_deg (float): The latitude of the UE in degrees.
            lon_deg (float): The longitude of the UE in degrees.
            alt_km (float, optional): The altitude of the UE in kilometers. Defaults to 0.0.
        """
        self.id = ue_id
        self.latitude = lat_deg
        self.longitude = lon_deg
        self.altitude = alt_km
        # Convert geodetic coordinates to ECEF for distance calculations
        self.position_ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_km)

    def __repr__(self) -> str:
        return f"UE(id={self.id}, lat={self.latitude:.2f}, lon={self.longitude:.2f})"

