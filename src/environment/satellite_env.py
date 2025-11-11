# -*- coding: utf-8 -*-
"""
This module defines the main reinforcement learning environment, SatelliteEnv,
updated to support a more complex, multi-tasking simulation environment with
a grid of ground stations.
"""

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque
import itertools
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.environment.components.satellite import Satellite
from src.environment.components.ue import UserEquipment
from src.environment.components.ground_station import GroundStation
from src.environment.components.task_generator import Task, TaskGenerator
from src.physics.orbits import OrbitPropagator
from src.physics import links, constants
from src.utils.geo import generate_hex_grid_centers

class SatelliteEnv(gym.Env):
    """Custom Environment for Satellite Edge Computing Simulation with dynamic task generation."""
    metadata = {'render_modes': ['human']}

    def __init__(self, sim_config: Dict[str, Any], sat_configs: Dict[str, List[Dict[str, Any]]]):
        super().__init__()

        self.sim_config = sim_config
        self._rng = np.random.default_rng()

        # --- Load simulation parameters ---
        self.start_time = datetime.fromisoformat(sim_config['start_time'].replace('Z', '+00:00'))
        self.isl_frequency_ghz = sim_config['isl_frequency_ghz']
        self.downlink_frequency_ghz = sim_config['downlink_frequency_ghz']
        self.ue_uplink_frequency_ghz = sim_config['user_equipment']['uplink_frequency_ghz']
        self.dl_model_flops_per_pixel = sim_config['dl_model']['flops_per_pixel']
        self.dl_model_output_classes = sim_config['dl_model']['output_classes']
        self.slicing_strategies = sim_config.get('slicing_strategies', [])
        self.max_episode_steps = sim_config.get('max_episode_steps', 500)

        # --- Initialize Environment Components ---
        self.orbit_propagator = OrbitPropagator(self.start_time)
        self.task_generator = TaskGenerator(
            rs_config=sim_config['task_generation'],
            ue_config=sim_config['user_equipment']['task_generation'],
            area_config=sim_config['target_area']
        )

        self.source_satellites = [Satellite(sat_id=cfg['sat_id'], config=cfg) for cfg in sat_configs['source_satellites']]
        self.compute_satellites = [Satellite(sat_id=cfg['sat_id'], config=cfg) for cfg in sat_configs['compute_satellites']]
        self.all_satellites = self.source_satellites + self.compute_satellites
        self.num_compute_satellites = len(self.compute_satellites)
        
        self.ground_ues = self._init_ues()
        self.ground_stations = self._init_ground_stations()
        self.num_ground_stations = len(self.ground_stations)
        
        # --- Dynamic State Containers ---
        self.task_queue: deque[Task] = deque()
        self.uplink_buffer: List[Tuple[datetime, Task, int]] = []
        self.pending_ue_tasks: deque[Tuple[UserEquipment, Task]] = deque()
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
            "task_info": spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32),
            "compute_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_compute_satellites, 3), dtype=np.float32),
            "compute_queues": spaces.Box(low=0, high=np.inf, shape=(self.num_compute_satellites,), dtype=np.float32),
            "ground_station_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_ground_stations, 3), dtype=np.float32),
            "queue_depth": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })

    def _init_ues(self) -> List[UserEquipment]:
        """Initializes a fixed set of UEs within the target area."""
        num_ues = self.sim_config['user_equipment']['num_ues']
        area = self.sim_config['target_area']
        ues = []
        for i in range(num_ues):
            lat = self._rng.uniform(area['min_lat_deg'], area['max_lat_deg'])
            lon = self._rng.uniform(area['min_lon_deg'], area['max_lon_deg'])
            ues.append(UserEquipment(ue_id=i, lat_deg=lat, lon_deg=lon))
        return ues

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
        if stride == 0: return 1
        slices_w = math.ceil((img_w - slice_size) / stride) + 1 if img_w > slice_size else 1
        slices_h = math.ceil((img_h - slice_size) / stride) + 1 if img_h > slice_size else 1
        return slices_w * slices_h

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Constructs the observation dictionary from the current environment state."""
        if not self.current_task:
            return {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.observation_space.spaces.items()}

        task_origin_pos = self.current_task.origin.position_ecef.astype(np.float32)
        task_info = np.array([self.current_task.width, self.current_task.height, self.current_task.bands, self.current_task.data_size_bits], dtype=np.float32)
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

    def _process_uplink_buffer(self) -> List[Task]:
        """Check for tasks that have completed their uplink and return them."""
        arrived_tasks = []
        remaining_uplinks = []
        for arrival_time, task, dest_sat_id in self.uplink_buffer:
            if self.orbit_propagator.simulation_time >= arrival_time:
                dest_sat = next((sat for sat in self.source_satellites if sat.id == dest_sat_id), None)
                if dest_sat:
                    task.origin = dest_sat
                    arrived_tasks.append(task)
            else:
                remaining_uplinks.append((arrival_time, task, dest_sat_id))
        self.uplink_buffer = remaining_uplinks
        return arrived_tasks

    def _generate_new_tasks(self) -> Tuple[List[Task], List[Task]]:
        """Generates new RS tasks and queues a new UE task."""
        new_rs_tasks = self.task_generator.check_and_generate_remote_sensing_tasks(self.source_satellites)
        self.task_queue.extend(new_rs_tasks)

        if self.ground_ues:
            ue = self._rng.choice(self.ground_ues)
            ue_task = self.task_generator.generate_ue_task(ue)
            self.pending_ue_tasks.append((ue, ue_task))
            return new_rs_tasks, [ue_task]
        
        return new_rs_tasks, []

    def _process_pending_ue_tasks(self) -> List[Task]:
        """Try to find an uplink for pending UE tasks."""
        still_pending = deque()
        newly_uplinking = []
        while self.pending_ue_tasks:
            ue, task = self.pending_ue_tasks.popleft()
            
            min_dist = np.inf
            closest_sat = None
            for sat in self.source_satellites:
                if np.isnan(sat.position_ecef).any(): continue
                dist = links.get_distance_km(ue.position_ecef, sat.position_ecef)
                if dist < min_dist:
                    min_dist = dist
                    closest_sat = sat
            
            if closest_sat:
                uplink_capacity = links.get_uplink_capacity_bps(min_dist, self.ue_uplink_frequency_ghz)
                t_uplink = task.data_size_bits / uplink_capacity if uplink_capacity > 0 else np.inf
                
                if not np.isinf(t_uplink):
                    arrival_time = self.orbit_propagator.simulation_time + timedelta(seconds=t_uplink)
                    self.uplink_buffer.append((arrival_time, task, closest_sat.id))
                    newly_uplinking.append(task)
                else:
                    still_pending.append((ue, task))
            else:
                still_pending.append((ue, task))
        
        self.pending_ue_tasks = still_pending
        return newly_uplinking

    def _get_next_task(self) -> Dict[str, List]:
        """Advances simulation until a task becomes available, logging events."""
        log = {'newly_arrived': [], 'newly_generated_rs': [], 'newly_generated_ue': [], 'newly_uplinking_ue': []}
        while not self.task_queue:
            if self.steps_taken >= self.max_episode_steps:
                self.current_task = None
                return log

            self.orbit_propagator.advance_simulation_time(60)
            self.orbit_propagator.update_satellite_positions(self.all_satellites)
            self.steps_taken += 1
            
            arrived = self._process_uplink_buffer()
            self.task_queue.extend(arrived)
            log['newly_arrived'].extend(arrived)
            
            new_rs, new_ue_for_pending = self._generate_new_tasks()
            log['newly_generated_rs'].extend(new_rs)
            log['newly_generated_ue'].extend(new_ue_for_pending)
            
            newly_uplinking = self._process_pending_ue_tasks()
            log['newly_uplinking_ue'].extend(newly_uplinking)

        self.current_task = self.task_queue.popleft() if self.task_queue else None
        return log

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.orbit_propagator.simulation_time = self.start_time
        self.steps_taken = 0
        self.task_queue.clear()
        self.uplink_buffer.clear()
        self.pending_ue_tasks.clear()

        for sat in self.compute_satellites:
            sat.task_queue.clear()
            sat.queue_load_flops = 0.0
        
        self.orbit_propagator.update_satellite_positions(self.all_satellites)
        self._get_next_task()
        
        return self._get_obs(), {}

    def _calculate_latency_and_reward(self, k: int, assignment: np.ndarray) -> Tuple[float, float]:
        if not self.current_task: return 0.0, 0.0

        source_sat = self.current_task.origin
        slice_data_size_bits = self.current_task.data_size_bits / k
        slice_flops = (self.current_task.width * self.current_task.height / k) * self.current_task.bands * self.dl_model_flops_per_pixel
        output_bits_per_pixel = np.ceil(np.log2(self.dl_model_output_classes))
        slice_result_size_bits = (self.current_task.width * self.current_task.height / k) * self.current_task.bands * output_bits_per_pixel

        latencies = []
        for dest_id, group in itertools.groupby(sorted(enumerate(assignment[:k]), key=lambda x: x[1]), key=lambda x: x[1]):
            num_slices_for_dest = len(list(group))
            
            # New logic: Check if destination is a Ground Station or a Compute Satellite
            if 0 <= dest_id < self.num_ground_stations:
                # Destination is a Ground Station
                gs = self.ground_stations[dest_id]
                dist = links.get_distance_km(source_sat.position_ecef, gs.position_ecef)
                capacity = links.get_downlink_capacity_bps(dist, self.downlink_frequency_ghz)
                t_trans = (num_slices_for_dest * slice_data_size_bits) / capacity if capacity > 0 else np.inf
                t_comp = (num_slices_for_dest * slice_flops) / (gs.compute_gflops * 1e9)
                latencies.append(t_trans + t_comp)
            else:
                # Destination is a Compute Satellite
                sat_idx = dest_id - self.num_ground_stations
                if 0 <= sat_idx < self.num_compute_satellites:
                    target_sat = self.compute_satellites[sat_idx]
                    dist_isl = links.get_distance_km(source_sat.position_ecef, target_sat.position_ecef)
                    isl_capacity = links.get_isl_capacity_bps(dist_isl, self.isl_frequency_ghz)
                    t_isl = (num_slices_for_dest * slice_data_size_bits) / isl_capacity if isl_capacity > 0 else np.inf
                    t_queue = target_sat.queue_load_flops / (target_sat.compute_gflops * 1e9)
                    t_comp = (num_slices_for_dest * slice_flops) / (target_sat.compute_gflops * 1e9)
                    dist_down = links.get_distance_km(target_sat.position_ecef, self.ground_stations[0].position_ecef) # Simplified: all results to first GS
                    down_capacity = links.get_downlink_capacity_bps(dist_down, self.downlink_frequency_ghz)
                    t_down = (num_slices_for_dest * slice_result_size_bits) / down_capacity if down_capacity > 0 else np.inf
                    latencies.append(t_isl + t_queue + t_comp + t_down)

        total_latency_sec = max(latencies) if latencies else 0
        accuracy_factor = 1.0 # Placeholder
        aoi = np.exp(-constants.AOI_DECAY_RATE_LAMBDA * total_latency_sec)
        reward = accuracy_factor * aoi
        return total_latency_sec, float(reward)

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        info = {}
        if not self.current_task:
            wait_log = self._get_next_task()
            obs = self._get_obs()
            truncated = self.steps_taken >= self.max_episode_steps
            terminated = (not self.task_queue and not self.uplink_buffer) and truncated
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
        if np.isinf(total_latency_sec):
            time_to_advance = self.sim_config['task_generation']['interval_range_sec'][1]
        
        self.orbit_propagator.advance_simulation_time(time_to_advance)
        self.orbit_propagator.update_satellite_positions(self.all_satellites)
        self.steps_taken += 1
        
        arrived_tasks = self._process_uplink_buffer()
        self.task_queue.extend(arrived_tasks)
        new_rs_tasks, new_ue_tasks_for_pending = self._generate_new_tasks()
        newly_uplinking_tasks = self._process_pending_ue_tasks()

        self._get_next_task()

        truncated = self.steps_taken >= self.max_episode_steps
        terminated = (not self.task_queue and not self.uplink_buffer) and truncated

        obs = self._get_obs()
        info = {
            'total_latency': total_latency_sec,
            'chosen_slice_size': slice_size,
            'chosen_overlap_ratio': overlap_ratio,
            'calculated_k': k,
            'newly_generated_rs_tasks': new_rs_tasks,
            'newly_generated_ue_tasks': new_ue_tasks_for_pending,
            'newly_uplinking_ue_tasks': newly_uplinking_tasks,
            'newly_arrived_ue_tasks': arrived_tasks
        }
        
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"--- Sim Time: {self.orbit_propagator.simulation_time}, Step: {self.steps_taken} ---")
            print(f"Current Task: {self.current_task}")
            print(f"Tasks in Queue: {len(self.task_queue)}, Uplinks: {len(self.uplink_buffer)}, Pending UEs: {len(self.pending_ue_tasks)}")
