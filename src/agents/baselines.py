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
        # cache convenience
        self.num_compute = env.num_nearest_compute
        self.num_dest = self.num_compute + 1  # + ground
        self.max_k = env.action_space['assignment'].shape[0]

    def predict(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _pack(self, size_idx: int, ov_idx: int, dest_vector: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            'slice_strategy': np.array([size_idx, ov_idx], dtype=np.int64),
            'assignment': dest_vector.astype(np.int64)
        }


class RandomPolicy(BasePolicy):
    """Uniform random hierarchical action."""
    def predict(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.env.action_space.sample()


class GreedyMinQueuePolicy(BasePolicy):
    """
    Heuristic:
    - Use a moderate-large slice (prefer larger context) and small overlap (0.15)
    - Assign all slices to the least-loaded destination among 10 computes + ground
    """
    def predict(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # choose slice size index = largest available; overlap index ~ 0.15 if exists else 0
        size_idx = len(self.env.slicing_strategies) - 1
        overlaps = self.env.slicing_strategies[0]['overlap_ratios']
        if 0.15 in overlaps:
            ov_idx = overlaps.index(0.15)
        else:
            ov_idx = 0
        # queues: compute + ground
        compute_q = obs.get('compute_queues', np.zeros(self.num_compute))
        gs_q = obs.get('ground_station_queue', np.zeros(1))[0]
        all_loads = np.concatenate([compute_q, np.array([gs_q])])
        dest = int(np.argmin(all_loads))
        assignment = np.full(self.max_k, dest, dtype=np.int64)
        return self._pack(size_idx, ov_idx, assignment)


class RoundRobinPolicy(BasePolicy):
    """Round-robin assignment over compute nodes, fixed slice strategy (512, 0.15)."""
    def predict(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # choose slice size 512 if exists else middle
        sizes = [s['slice_size'] for s in self.env.slicing_strategies]
        if 512 in sizes:
            size_idx = sizes.index(512)
        else:
            size_idx = len(sizes)//2
        overlaps = self.env.slicing_strategies[0]['overlap_ratios']
        ov_idx = overlaps.index(0.15) if 0.15 in overlaps else 0
        # round-robin among compute nodes only
        assign = np.arange(self.max_k) % self.num_compute
        return self._pack(size_idx, ov_idx, assign)


class GroundOnlyPolicy(BasePolicy):
    """All slices to ground, large slice and moderate overlap."""
    def predict(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        size_idx = len(self.env.slicing_strategies) - 1
        overlaps = self.env.slicing_strategies[0]['overlap_ratios']
        ov_idx = overlaps.index(0.25) if 0.25 in overlaps else 0
        dest_ground = self.num_dest - 1
        assignment = np.full(self.max_k, dest_ground, dtype=np.int64)
        return self._pack(size_idx, ov_idx, assignment)


class ComputeOnlyMinQueuePolicy(BasePolicy):
    """All slices to the compute node with the least queue (ignore ground)."""
    def predict(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        sizes = [s['slice_size'] for s in self.env.slicing_strategies]
        size_idx = sizes.index(1024) if 1024 in sizes else len(sizes)//2
        overlaps = self.env.slicing_strategies[0]['overlap_ratios']
        ov_idx = overlaps.index(0.15) if 0.15 in overlaps else 0
        compute_q = obs.get('compute_queues', np.zeros(self.num_compute))
        dest_compute = int(np.argmin(compute_q))
        assignment = np.full(self.max_k, dest_compute, dtype=np.int64)
        return self._pack(size_idx, ov_idx, assignment)
