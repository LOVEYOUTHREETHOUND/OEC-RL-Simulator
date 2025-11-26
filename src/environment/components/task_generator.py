# -*- coding: utf-8 -*-
"""
Unified Task and TaskGenerator with Poisson arrivals per source satellite.

A Task represents a single image-processing job that is assumed to already be
available on a source (remote sensing) satellite. There is only one task type.

Task attributes include:
- width (pixels)
- height (pixels)
- max_latency_sec (the maximum acceptable end-to-end latency for this task)
- data_size_bits (input data volume in bits)
- required_flops (total compute required in FLOPs)

TaskGenerator now models task arrivals as independent Poisson processes for each
source satellite. Inter-arrival times are exponentially distributed.

Configuration (task_config) supports:
- image_size_range: [min, max] for width/height (pixels)
- max_latency_sec_range: [min, max] (optional; default [60, 600])
- bits_per_pixel: default 8 if not provided (used to compute data_size_bits)
- poisson_rate_per_sec | poisson_rate_per_min | poisson_rate_per_hour
  (optional) Arrival rate. If none provided, falls back to 1/mean(interval_range_sec)
  if interval_range_sec exists, otherwise defaults to 1/300 s^-1 (one task per 5 min).

Additional parameter:
- flops_per_pixel: multiplier to compute required_flops from width*height
"""

from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timedelta
import numpy as np

# Forward-declare Satellite to avoid circular import issues
if False:
    from .satellite import Satellite


class Task:
    """
    Unified task model. Origin is optional and is used internally by the
    environment for link-distance calculations (e.g., ISL, downlink).
    """

    def __init__(
        self,
        task_id: int,
        width: int,
        height: int,
        max_latency_sec: float,
        data_size_bits: int,
        required_flops: float,
        origin: Optional['Satellite'] = None,
    ):
        self.id = task_id
        self.width = int(width)
        self.height = int(height)
        self.max_latency_sec = float(max_latency_sec)
        self.data_size_bits = int(data_size_bits)
        self.required_flops = float(required_flops)
        self.origin = origin  # Optional source satellite reference

    def __repr__(self) -> str:
        mb = self.data_size_bits / 8e6 if self.data_size_bits else 0.0
        return (
            f"Task(id={self.id}, w={self.width}, h={self.height}, "
            f"maxLat={self.max_latency_sec:.1f}s, data={mb:.2f} MB, "
            f"flops={self.required_flops:.2e})"
        )


class TaskGenerator:
    """
    Generates unified tasks for satellites with Poisson arrivals per satellite.
    """

    def __init__(self, task_config: Dict[str, Any], area_config: Dict[str, Any], flops_per_pixel: float):
        self.task_config = task_config
        self.area_config = area_config  # kept for compatibility; not used in Poisson arrivals by default
        self.flops_per_pixel = float(flops_per_pixel)
        self._rng = np.random.default_rng()
        self._next_task_id = 0

        # Determine Poisson arrival rate (per second)
        self.poisson_rate_per_sec = self._determine_rate(task_config)
        # State: per-satellite next arrival time
        self._next_arrival_time_by_sat: Dict[int, datetime] = {}

    # -----------------------------
    # Sampling helpers
    # -----------------------------
    def _determine_rate(self, cfg: Dict[str, Any]) -> float:
        if 'poisson_rate_per_sec' in cfg:
            rate = float(cfg['poisson_rate_per_sec'])
        elif 'poisson_rate_per_min' in cfg:
            rate = float(cfg['poisson_rate_per_min']) / 60.0
        elif 'poisson_rate_per_hour' in cfg:
            rate = float(cfg['poisson_rate_per_hour']) / 3600.0
        elif 'interval_range_sec' in cfg and isinstance(cfg['interval_range_sec'], (list, tuple)) and len(cfg['interval_range_sec']) >= 2:
            lo, hi = cfg['interval_range_sec'][0], cfg['interval_range_sec'][1]
            mean_interval = (float(lo) + float(hi)) / 2.0
            rate = 1.0 / max(mean_interval, 1e-6)
        else:
            # default: one task every 5 minutes
            rate = 1.0 / 300.0
        return max(rate, 1e-9)

    def _sample_size(self) -> Tuple[int, int]:
        size_range = self.task_config.get('image_size_range', (1024, 1024))
        width = int(self._rng.integers(size_range[0], size_range[1], endpoint=True))
        height = int(self._rng.integers(size_range[0], size_range[1], endpoint=True))
        return width, height

    def _sample_max_latency(self) -> float:
        lat_range = self.task_config.get('max_latency_sec_range', (60.0, 600.0))
        return float(self._rng.uniform(lat_range[0], lat_range[1]))

    def _compute_data_bits(self, width: int, height: int) -> int:
        bpp = int(self.task_config.get('bits_per_pixel', 8))
        return int(width * height * bpp)

    def _compute_flops(self, width: int, height: int) -> float:
        return float(width * height * self.flops_per_pixel)

    def _next_interarrival(self) -> timedelta:
        # Exponential with mean 1/lambda
        delta_sec = float(self._rng.exponential(1.0 / self.poisson_rate_per_sec))
        return timedelta(seconds=delta_sec)

    # -----------------------------
    # Public API
    # -----------------------------
    def reset_arrivals(self, sim_time: datetime, satellites: List['Satellite']):
        """Initialize next arrival time for each satellite from sim_time."""
        self._next_arrival_time_by_sat.clear()
        for sat in satellites:
            self._next_arrival_time_by_sat[sat.id] = sim_time + self._next_interarrival()

    def generate_unified_tasks(self, satellites: List['Satellite'], sim_time: datetime) -> List[Task]:
        """
        Generate tasks up to the current simulation time according to per-satellite
        Poisson processes. Multiple tasks may be generated per satellite if the
        simulation time advances by a large amount.
        """
        new_tasks: List[Task] = []
        for sat in satellites:
            # Initialize next arrival lazily if needed
            if sat.id not in self._next_arrival_time_by_sat:
                self._next_arrival_time_by_sat[sat.id] = sim_time + self._next_interarrival()

            # Generate all arrivals that occurred up to sim_time
            while self._next_arrival_time_by_sat[sat.id] <= sim_time:
                w, h = self._sample_size()
                max_lat = self._sample_max_latency()
                data_bits = self._compute_data_bits(w, h)
                flops = self._compute_flops(w, h)
                task = Task(
                    task_id=self._next_task_id,
                    width=w,
                    height=h,
                    max_latency_sec=max_lat,
                    data_size_bits=data_bits,
                    required_flops=flops,
                    origin=sat,
                )
                self._next_task_id += 1
                new_tasks.append(task)

                # Schedule next arrival for this satellite
                self._next_arrival_time_by_sat[sat.id] += self._next_interarrival()

        return new_tasks
