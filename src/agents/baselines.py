# -*- coding: utf-8 -*-
"""
This module defines simple, non-learning baseline policies for comparison.

These policies are adapted to generate hierarchical actions compatible with
the updated SatelliteEnv.
"""

from typing import Dict
import numpy as np

from src.environment.satellite_env import SatelliteEnv


class BasePolicy:
    """Abstract base class for all policies."""
    def __init__(self, env: SatelliteEnv):
        self.env = env

    def predict(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Predicts a hierarchical action based on the observation."""
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """A policy that selects hierarchical actions uniformly at random."""
    def predict(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Returns a random sample from the environment's hierarchical action space.

        Args:
            obs (Dict[str, np.ndarray]): The current observation (ignored).

        Returns:
            Dict[str, np.ndarray]: A random hierarchical action.
        """
        return self.env.action_space.sample()


class GreedyPolicy(BasePolicy):
    """
    A simple heuristic-based policy for the hierarchical action space.

    - High-level: Always chooses the simplest slicing strategy (the one with the fewest slices).
    - Low-level: Assigns all slices to the satellite/ground station with the lowest queue load.
    """
    def predict(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Predicts a hierarchical action using a greedy heuristic.

        Args:
            obs (Dict[str, np.ndarray]): The current observation, containing 'sat_queues'.

        Returns:
            Dict[str, np.ndarray]: The calculated greedy hierarchical action.
        """
        # --- High-level Decision ---
        # Greedily choose the simplest strategy (fewest slices), which is the first one.
        slice_strategy_idx = 0

        # --- Low-level Decision ---
        ground_station_load = 0.0
        satellite_loads = obs['sat_queues']
        all_loads = np.insert(satellite_loads, 0, ground_station_load)
        least_loaded_id = np.argmin(all_loads)
        
        # Get the maximum possible number of slices to create a correctly sized vector
        max_k = self.env.action_space['assignment'].shape[0]
        assignment_vector = np.full(max_k, least_loaded_id, dtype=int)

        return {
            'slice_strategy': np.array(slice_strategy_idx),
            'assignment': assignment_vector
        }
