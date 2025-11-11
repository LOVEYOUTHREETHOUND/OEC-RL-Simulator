# -*- coding: utf-8 -*-
"""
This module handles the orbit propagation for satellites using TLE data.
It uses the sgp4 library to calculate satellite positions at given times.
"""

from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
from sgp4.api import Satrec, jday

from src.physics.constants import EARTH_EQUATORIAL_RADIUS_KM, EARTH_FLATTENING_F

def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    """
    Converts geodetic coordinates (latitude, longitude, altitude) to
    Earth-Centered, Earth-Fixed (ECEF) coordinates.

    Args:
        lat_deg (float): Latitude in degrees.
        lon_deg (float): Longitude in degrees.
        alt_km (float): Altitude in kilometers.

    Returns:
        np.ndarray: A 3-element array representing the ECEF coordinates (x, y, z) in km.
    """
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    
    e_sq = EARTH_FLATTENING_F * (2 - EARTH_FLATTENING_F)
    n = EARTH_EQUATORIAL_RADIUS_KM / np.sqrt(1 - e_sq * np.sin(lat_rad)**2)
    
    x = (n + alt_km) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (n + alt_km) * np.cos(lat_rad) * np.sin(lon_rad)
    z = ((1 - e_sq) * n + alt_km) * np.sin(lat_rad)
    
    return np.array([x, y, z])

def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Converts Earth-Centered, Earth-Fixed (ECEF) coordinates to
    geodetic coordinates (latitude, longitude, altitude).

    Args:
        x (float): ECEF X coordinate in km.
        y (float): ECEF Y coordinate in km.
        z (float): ECEF Z coordinate in km.

    Returns:
        Tuple[float, float, float]: (latitude_deg, longitude_deg, altitude_km).
    """
    a = EARTH_EQUATORIAL_RADIUS_KM
    f = EARTH_FLATTENING_F
    b = a * (1 - f)
    e_sq = f * (2 - f)
    e_prime_sq = e_sq / (1 - e_sq)
    
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)
    
    lon_rad = np.arctan2(y, x)
    lat_rad = np.arctan2(z + e_prime_sq * b * np.sin(theta)**3, p - e_sq * a * np.cos(theta)**3)
    
    n = a / np.sqrt(1 - e_sq * np.sin(lat_rad)**2)
    alt_km = (p / np.cos(lat_rad)) - n
    
    return np.rad2deg(lat_rad), np.rad2deg(lon_rad), alt_km

class OrbitPropagator:
    """
    Manages the simulation time and propagates satellite orbits.
    """

    def __init__(self, start_time: datetime):
        """
        Initializes the OrbitPropagator.

        Args:
            start_time (datetime): The simulation's start time in UTC.
        """
        self.simulation_time: datetime = start_time

    def get_positions_ecef(self, satellites: List['Satellite'], time_offset_seconds: float = 0.0) -> List[np.ndarray]:
        """
        Calculates the ECEF positions of a list of satellites at a specific time offset
        from the current simulation time.

        Args:
            satellites (List['Satellite']): The list of Satellite objects to propagate.
            time_offset_seconds (float): The time offset in seconds from the current
                                         simulation time. Defaults to 0.0.

        Returns:
            List[np.ndarray]: A list of 3-element numpy arrays, each representing the
                              ECEF coordinates (x, y, z) in km for a satellite.
        """
        target_time = self.simulation_time + timedelta(seconds=time_offset_seconds)
        jd, fr = jday(target_time.year, target_time.month, target_time.day, 
                      target_time.hour, target_time.minute, target_time.second + target_time.microsecond / 1e6)

        positions = []
        for sat in satellites:
            try:
                satrec = Satrec.twoline2rv(sat.tle[0], sat.tle[1])
                error, r, v = satrec.sgp4(jd, fr)
                if error == 0:
                    positions.append(np.array(r))
                else:
                    # TLE data is invalid or has expired
                    positions.append(np.array([np.nan, np.nan, np.nan]))
            except Exception:
                # Catch any other errors during propagation
                positions.append(np.array([np.nan, np.nan, np.nan]))
        
        return positions

    def update_satellite_positions(self, satellites: List['Satellite']):
        """
        Updates the position of each satellite in the list to the current simulation time.
        """
        positions = self.get_positions_ecef(satellites)
        for sat, pos in zip(satellites, positions):
            sat.update_position(pos)

    def advance_simulation_time(self, seconds: float):
        """
        Advances the simulation time by a given number of seconds.

        Args:
            seconds (float): The number of seconds to advance the time.
        """
        if not np.isinf(seconds) and seconds > 0:
            self.simulation_time += timedelta(seconds=seconds)
