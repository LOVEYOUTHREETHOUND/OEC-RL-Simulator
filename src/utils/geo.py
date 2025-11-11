# -*- coding: utf-8 -*-
"""
This module provides geospatial utility functions, including the generation
of hexagonal grid centers over a defined geographical area.
"""

import numpy as np
from typing import List, Tuple

# Approximate conversion factor for latitude, assuming a spherical Earth
KM_PER_DEG_LAT = 111.0

def km_per_deg_lon(latitude: float) -> float:
    """Calculates the approximate distance in km for one degree of longitude at a given latitude."""
    return KM_PER_DEG_LAT * np.cos(np.deg2rad(latitude))

def generate_hex_grid_centers(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, radius_km: float
) -> List[Tuple[float, float]]:
    """
    Generates a grid of hexagonal centers covering a given geographic area.

    Args:
        min_lon: Minimum longitude of the area.
        max_lon: Maximum longitude of the area.
        min_lat: Minimum latitude of the area.
        max_lat: Maximum latitude of the area.
        radius_km: The radius of the hexagon (distance from center to vertex).

    Returns:
        A list of (latitude, longitude) tuples for the center of each hexagon.
    """
    centers = []
    
    # Use the center latitude for a stable longitude scaling approximation across the area
    center_lat = (min_lat + max_lat) / 2.0
    deg_lon_per_km = 1.0 / km_per_deg_lon(center_lat)
    deg_lat_per_km = 1.0 / KM_PER_DEG_LAT

    # Hexagon geometry for tiling the plane
    h_dist_km = radius_km * np.sqrt(3)  # Horizontal distance between centers
    v_dist_km = radius_km * 1.5       # Vertical distance between rows

    h_dist_deg = h_dist_km * deg_lon_per_km
    v_dist_deg = v_dist_km * deg_lat_per_km

    # Start tiling from the bottom-left corner of the bounding box
    y = min_lat
    row_index = 0
    while y < max_lat + v_dist_deg: # Add a buffer to ensure the top edge is covered
        x = min_lon
        # Odd-numbered rows are shifted horizontally to create the honeycomb pattern
        if row_index % 2 != 0:
            x -= h_dist_deg / 2.0
        
        while x < max_lon + h_dist_deg: # Add a buffer for the right edge
            # Only add the center if it falls within the actual target area
            if min_lat <= y <= max_lat and min_lon <= x <= max_lon:
                centers.append((y, x)) # (latitude, longitude)
            x += h_dist_deg
        
        y += v_dist_deg
        row_index += 1
        
    return centers

