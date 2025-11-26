# -*- coding: utf-8 -*-
"""
This module defines the main reinforcement learning environment, SatelliteEnv,
updated to a unified-task setting: there is only one type of task, which is
assumed to already be available on a source (remote-sensing) satellite when it
is over the target area. We do not model UE-originated tasks or uplinks anymore.
"""

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from collections import deque
import itertools
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.environment.components.satellite import Satellite
from src.environment.components.ground_station import GroundStation
from src.environment.components.task_generator import Task, TaskGenerator
from src.physics.orbits import OrbitPropagator
from src.physics import links, constants
from src.utils.geo import generate_hex_grid_centers


class SatelliteEnv(gym.Env):
    """Custom Environment for Satellite Edge Computing Simulation with unified task generation."""
    metadata = {'render_modes': ['human']}

    def __init__(self, sim_config: Dict[str, Any], sat_configs: Dict[str, List[Dict[str, Any]]]):
        super().__init__()

        self.sim_config = sim_config
        self._rng = np.random.default_rng()

        # --- Load simulation parameters ---
        self.start_time = datetime.fromisoformat(sim_config['start_time'].replace('Z', '+00:00'))
        self.isl_frequency_ghz = sim_config['isl_frequency_ghz']
        self.downlink_frequency_ghz = sim_config['downlink_frequency_ghz']
        self.slicing_strategies = sim_config.get('slicing_strategies', [])
        self.max_episode_steps = sim_config.get('max_episode_steps', 500)

        # --- Initialize Environment Components ---
        self.orbit_propagator = OrbitPropagator(self.start_time)
        # TaskGenerator now unified; derive required FLOPs using flops_per_pixel
        flops_per_pixel = float(sim_config.get('dl_model', {}).get('flops_per_pixel', 1000.0))
        self.task_generator = TaskGenerator(
            task_config=sim_config['task_generation'],
            area_config=sim_config['target_area'],
            flops_per_pixel=flops_per_pixel,
        )

        self.source_satellites = [Satellite(sat_id=cfg['sat_id'], config=cfg) for cfg in sat_configs['source_satellites']]
        self.compute_satellites = [Satellite(sat_id=cfg['sat_id'], config=cfg) for cfg in sat_configs['compute_satellites']]
        # Optional MEO leader satellites
        self.leader_satellites = [Satellite(sat_id=cfg['sat_id'], config=cfg) for cfg in sat_configs.get('leader_satellites', [])]
        self.num_leader_satellites = len(self.leader_satellites)

        self.all_satellites = self.source_satellites + self.compute_satellites + self.leader_satellites
        self.num_compute_satellites = len(self.compute_satellites)

        # There are no UEs anymore; keep attribute for compatibility with scripts
        self.ground_ues: List = []

        self.ground_stations = self._init_ground_stations()
        self.num_ground_stations = len(self.ground_stations)

        # --- Dynamic State Containers ---
        self.task_queue: deque[Task] = deque()
        self.completed_tasks: deque[Task] = deque()
        self.current_task: Optional[Task] = None
        self.steps_taken = 0

        # --- Define Action and Observation Spaces ---
        self._define_spaces()

    def _define_spaces(self):
        """Defines the action and observation spaces for the environment."""
        num_slice_sizes = len(self.slicing_strategies)
        num_overlap_ratios = len(self.slicing_strategies[0]['overlap_ratios']) if num_slice_sizes > 0 else 0

        min_slice_size = self.slicing_strategies[0]['slice_size'] if num_slice_sizes > 0 else 512
        max_k = self._calculate_k(min_slice_size, 0.0)

        num_destinations = self.num_ground_stations + self.num_compute_satellites
        self.action_space = spaces.Dict({
            'slice_strategy': spaces.MultiDiscrete([num_slice_sizes, num_overlap_ratios]),
            'assignment': spaces.MultiDiscrete([num_destinations] * max_k)
        })

        self.observation_space = spaces.Dict({
            "task_origin_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            # New unified task info: [width, height, max_latency_sec, data_size_bits, required_flops]
            "task_info": spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32),
            "compute_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_compute_satellites, 3), dtype=np.float32),
            "compute_queues": spaces.Box(low=0, high=np.inf, shape=(self.num_compute_satellites,), dtype=np.float32),
            "ground_station_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_ground_stations, 3), dtype=np.float32),
            "queue_depth": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })

    def _init_ground_stations(self) -> List[GroundStation]:
        """Initializes ground stations based on the specified generation method."""
        gs_config = self.sim_config['ground_station']
        if gs_config.get('generation_method') == 'hex_grid':
            area = self.sim_config['target_area']
            radius_km = gs_config['grid_radius_km']
            centers = generate_hex_grid_centers(
                area['min_lon_deg'], area['max_lon_deg'],
                area['min_lat_deg'], area['max_lat_deg'],
                radius_km
            )
            return [GroundStation(gs_id=i, lat_deg=lat, lon_deg=lon, config=gs_config) for i, (lat, lon) in enumerate(centers)]
        else:
            # Fallback to a single, manually defined ground station if method is not hex_grid
            return [GroundStation(gs_id=0, lat_deg=0, lon_deg=0, config=gs_config)]

    def _calculate_k(self, slice_size: int, overlap_ratio: float) -> int:
        """Helper to calculate number of slices based on task properties."""
        if not self.current_task:
            max_size = self.sim_config['task_generation']['image_size_range'][1]
            img_w, img_h = max_size, max_size
        else:
            img_w, img_h = self.current_task.width, self.current_task.height

        stride = slice_size * (1 - overlap_ratio)
        if stride == 0:
            return 1
        slices_w = math.ceil((img_w - slice_size) / stride) + 1 if img_w > slice_size else 1
        slices_h = math.ceil((img_h - slice_size) / stride) + 1 if img_h > slice_size else 1
        return slices_w * slices_h

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Constructs the observation dictionary from the current environment state."""
        if not self.current_task:
            return {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.observation_space.spaces.items()}

        task_origin_pos = self.current_task.origin.position_ecef.astype(np.float32)
        task_info = np.array([
            self.current_task.width,
            self.current_task.height,
            self.current_task.max_latency_sec,
            self.current_task.data_size_bits,
            self.current_task.required_flops,
        ], dtype=np.float32)
        compute_pos = np.array([sat.position_ecef for sat in self.compute_satellites], dtype=np.float32)
        compute_queues = np.array([sat.queue_load_flops for sat in self.compute_satellites], dtype=np.float32)
        gs_pos = np.array([gs.position_ecef for gs in self.ground_stations], dtype=np.float32)
        queue_depth = np.array([len(self.task_queue)], dtype=np.float32)

        return {
            "task_origin_pos": task_origin_pos,
            "task_info": task_info,
            "compute_pos": compute_pos,
            "compute_queues": compute_queues,
            "ground_station_pos": gs_pos,
            "queue_depth": queue_depth
        }

    def _generate_new_tasks(self) -> List[Task]:
        """Generates new unified tasks (Poisson per satellite) up to current sim time and pushes to queue."""
        sim_time = self.orbit_propagator.simulation_time
        new_tasks = self.task_generator.generate_unified_tasks(self.source_satellites, sim_time)
        self.task_queue.extend(new_tasks)
        return new_tasks

    def _get_next_task(self) -> Dict[str, List]:
        """Advances simulation until a task becomes available, logging events."""
        log = {'newly_generated_rs': []}
        while not self.task_queue:
            if self.steps_taken >= self.max_episode_steps:
                self.current_task = None
                return log

            self.orbit_propagator.advance_simulation_time(60)
            self.orbit_propagator.update_satellite_positions(self.all_satellites)
            self.steps_taken += 1

            new_tasks = self._generate_new_tasks()
            log['newly_generated_rs'].extend(new_tasks)

        self.current_task = self.task_queue.popleft() if self.task_queue else None
        return log

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.orbit_propagator.simulation_time = self.start_time
        self.steps_taken = 0
        self.task_queue.clear()
        self.completed_tasks.clear()

        for sat in self.compute_satellites:
            sat.task_queue.clear()
            sat.queue_load_flops = 0.0

        self.orbit_propagator.update_satellite_positions(self.all_satellites)
        # Initialize Poisson arrivals per satellite starting from current sim time
        self.task_generator.reset_arrivals(self.orbit_propagator.simulation_time, self.source_satellites)
        self._get_next_task()

        return self._get_obs(), {}

    def _calculate_latency_and_reward(self, k: int, assignment: np.ndarray) -> Tuple[float, float]:
        if not self.current_task:
            return 0.0, 0.0

        source_sat = self.current_task.origin
        # Use unified task attributes
        slice_data_size_bits = self.current_task.data_size_bits / k
        slice_flops = self.current_task.required_flops / k
        # No explicit result size provided in the new model; assume negligible
        slice_result_size_bits = 0.0

        # Determine leader satellite (nearest MEO leader). If none configured, fall back to source-as-leader (legacy path)
        if self.num_leader_satellites > 0:
            dists = [links.get_distance_km(source_sat.position_ecef, ldr.position_ecef) for ldr in self.leader_satellites]
            leader_idx = int(np.argmin(dists))
            leader_sat = self.leader_satellites[leader_idx]
            # Initial hop: source RS -> leader via ISL, send full data once
            dist_rs_leader = dists[leader_idx]
            isl_cap_rs_leader = links.get_isl_capacity_bps(dist_rs_leader, self.isl_frequency_ghz)
            t_initial = (self.current_task.data_size_bits) / isl_cap_rs_leader if isl_cap_rs_leader > 0 else np.inf
        else:
            leader_sat = source_sat
            t_initial = 0.0

        latencies: List[float] = []
        for dest_id, group in itertools.groupby(sorted(enumerate(assignment[:k]), key=lambda x: x[1]), key=lambda x: x[1]):
            num_slices_for_dest = len(list(group))

            if 0 <= dest_id < self.num_ground_stations:
                # Destination is a Ground Station (cloud): leader downlinks raw slices, then compute on ground
                gs = self.ground_stations[dest_id]
                dist = links.get_distance_km(leader_sat.position_ecef, gs.position_ecef)
                capacity = links.get_downlink_capacity_bps(dist, self.downlink_frequency_ghz)
                t_trans = (num_slices_for_dest * slice_data_size_bits) / capacity if capacity > 0 else np.inf
                t_comp = (num_slices_for_dest * slice_flops) / (gs.compute_gflops * 1e9)
                latencies.append(t_initial + t_trans + t_comp)
            else:
                # Destination is a LEO Compute Satellite: leader sends raw slices via ISL to LEO, then queue+compute
                sat_idx = dest_id - self.num_ground_stations
                if 0 <= sat_idx < self.num_compute_satellites:
                    target_sat = self.compute_satellites[sat_idx]
                    dist_isl = links.get_distance_km(leader_sat.position_ecef, target_sat.position_ecef)
                    isl_capacity = links.get_isl_capacity_bps(dist_isl, self.isl_frequency_ghz)
                    t_isl = (num_slices_for_dest * slice_data_size_bits) / isl_capacity if isl_capacity > 0 else np.inf
                    t_queue = target_sat.queue_load_flops / (target_sat.compute_gflops * 1e9)
                    t_comp = (num_slices_for_dest * slice_flops) / (target_sat.compute_gflops * 1e9)
                    # Add the task to the satellite's queue
                    task_flops = num_slices_for_dest * slice_flops
                    target_sat.add_task_to_queue(task_flops)
                    # Result downlink assumed negligible
                    t_down = 0.0
                    latencies.append(t_initial + t_isl + t_queue + t_comp + t_down)

        total_latency_sec = max(latencies) if latencies else 0.0
        # Enforce max latency constraint: zero reward if violated
        if total_latency_sec > self.current_task.max_latency_sec:
            reward = 0.0
        else:
            aoi = np.exp(-constants.AOI_DECAY_RATE_LAMBDA * total_latency_sec)
            reward = float(aoi)
        return total_latency_sec, reward

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}
        if self.current_task:
            self.completed_tasks.append(self.current_task)

        if not self.current_task:
            wait_log = self._get_next_task()
            obs = self._get_obs()
            truncated = self.steps_taken >= self.max_episode_steps
            terminated = (not self.task_queue) and truncated
            info.update(wait_log)
            info['info'] = 'Agent was idle, waited for next task.'
            return obs, 0.0, terminated, truncated, info

        slice_size_idx, overlap_ratio_idx = action['slice_strategy']
        assignment = action['assignment']
        strategy = self.slicing_strategies[slice_size_idx]
        slice_size = strategy['slice_size']
        overlap_ratio = strategy['overlap_ratios'][overlap_ratio_idx]
        k = self._calculate_k(slice_size, overlap_ratio)

        total_latency_sec, reward = self._calculate_latency_and_reward(k, assignment)

        time_to_advance = total_latency_sec
        if np.isinf(total_latency_sec) or total_latency_sec <= 0:
            # Advance by a default interval if infeasible or zero
            time_to_advance = self.sim_config['task_generation'].get('interval_range_sec', [60, 300])[1]

        self.orbit_propagator.advance_simulation_time(time_to_advance)
        self.orbit_propagator.update_satellite_positions(self.all_satellites)
        self.steps_taken += 1

        new_tasks = self._generate_new_tasks()

        self._get_next_task()

        truncated = self.steps_taken >= self.max_episode_steps
        terminated = (not self.task_queue) and truncated

        obs = self._get_obs()
        info = {
            'total_latency': total_latency_sec,
            'chosen_slice_size': slice_size,
            'chosen_overlap_ratio': overlap_ratio,
            'calculated_k': k,
            'newly_generated_rs_tasks': new_tasks,  # keep key name for backward compatibility
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"--- Sim Time: {self.orbit_propagator.simulation_time}, Step: {self.steps_taken} ---")
            print(f"Current Task: {self.current_task}")
            print(f"Tasks in Queue: {len(self.task_queue)}")
