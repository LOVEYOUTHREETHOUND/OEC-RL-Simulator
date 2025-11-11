# -*- coding: utf-8 -*-
"""
This module defines the Satellite class, which represents a single satellite
in the simulation environment.
"""

from collections import deque
from typing import Dict, Any, Tuple

import numpy as np

from src.physics.orbits import ecef_to_geodetic


class Satellite:
    """
    Represents a single satellite, encapsulating its state and properties.

    This class acts as a data container for a satellite's static attributes
    (like ID and computing power) and its dynamic state (like position and
    the current task queue).
    """

    def __init__(self, sat_id: int, config: Dict[str, Any]):
        """
        Initializes a Satellite object from a configuration dictionary.

        Args:
            sat_id (int): A unique identifier for the satellite.
            config (Dict[str, Any]): A dictionary containing the satellite's
                                     configuration parameters, expected to contain:
                                     - 'compute_gflops' (float): Onboard computing capacity in GFLOPS.
                                     - 'tle' (Tuple[str, str]): The two lines of the TLE data.
        """
        self.id: int = sat_id
        self.name: str = config.get('name', f"SAT-{sat_id}")

        # --- Static Properties ---
        self.compute_gflops: float = config.get('compute_gflops', 100.0)  # Default to 100 GFLOPS
        self.tle: Tuple[str, str] = config['tle']

        # --- Dynamic State Variables ---

        # Current position in Earth-Centered, Earth-Fixed (ECEF) coordinates (km)
        self.position_ecef: np.ndarray = np.zeros(3)

        # Onboard task queue. We can store task objects or just their computational size.
        self.task_queue: deque = deque()

        # Total computational load in the queue (in FLOPs) for easy state access
        self.queue_load_flops: float = 0.0

    def update_position(self, new_position_ecef: np.ndarray):
        """
        Updates the satellite's current position.

        Args:
            new_position_ecef (np.ndarray): A 3-element numpy array representing
                                            the new ECEF coordinates (x, y, z) in km.
        """
        self.position_ecef = new_position_ecef

    def get_geodetic_coords(self) -> Tuple[float, float, float]:
        """
        Calculates and returns the satellite's current geodetic coordinates.

        Returns:
            Tuple[float, float, float]: A tuple containing (latitude_deg, longitude_deg, altitude_km).
        """
        if np.isnan(self.position_ecef).any() or np.isinf(self.position_ecef).any():
            return (np.nan, np.nan, np.nan)
        return ecef_to_geodetic(self.position_ecef[0], self.position_ecef[1], self.position_ecef[2])

    def add_task_to_queue(self, task_flops: float):
        """
        Adds a task's computational load to the satellite's processing queue.

        Args:
            task_flops (float): The computational amount of the task in FLOPs.
        """
        self.task_queue.append(task_flops)
        self.queue_load_flops += task_flops

    def __repr__(self) -> str:
        """
        Provides a developer-friendly string representation of the satellite.
        """
        return f"Satellite(id={self.id}, queue_load={self.queue_load_flops:.2e} FLOPs)"
