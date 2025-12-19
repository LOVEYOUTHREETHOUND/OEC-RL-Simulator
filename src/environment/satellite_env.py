# -*- coding: utf-8 -*-
"""
Refactored SatelliteEnv for single source satellite training.

Key changes:
1. Each episode focuses on ONE randomly selected source satellite
2. Only the nearest visible MEO leader satellite is selected per task
3. Observation includes only the 10 nearest compute satellites + nearest ground station
4. Task queues are tracked globally for all compute nodes (LEO + MEO + GS)
5. Episode terminates after a fixed number of tasks (decisions)
6. Invalid steps (no visible MEO/compute satellites) are skipped without reward/penalty
"""

from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
import itertools
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import os
from src.environment.components.satellite import Satellite
from src.environment.components.ground_station import GroundStation
from src.environment.components.compute_node import (
    ComputeNode, reset_compute_node_registry, get_all_compute_nodes
)
from src.environment.components.task_generator import Task, TaskGenerator
from src.physics.orbits import OrbitPropagator
from src.physics import links, constants
from src.utils.geo import generate_hex_grid_centers
from src.utils.tle_loader import preprocess_satellite_configs


class SatelliteEnv(gym.Env):
    # ---------- Helper: mIoU surrogate & table ----------
    def _miou_surrogate(self, slice_size: int, overlap_ratio: float) -> float:
        # Monotone, saturating functions
        sp = self.surrogate_params or {}
        s0 = float(sp.get('s0', 1024.0))
        alpha = float(sp.get('alpha', 1.0))
        tau = float(sp.get('tau', 0.2))
        beta = float(sp.get('beta', 1.0))
        w_size = float(sp.get('w_size', 0.7))
        w_ov = float(sp.get('w_overlap', 0.3))
        # Normalize slice_size relative to s0
        f_size = 1.0 - np.exp(- (slice_size / max(s0, 1e-6)) ** alpha)
        f_ov = 1.0 - np.exp(- (overlap_ratio / max(tau, 1e-6)) ** beta)
        xm_norm = (w_size * f_size + w_ov * f_ov) / max(w_size + w_ov, 1e-6)
        xm_norm = float(np.clip(xm_norm, 0.0, 1.0))
        # Scale and gamma
        xm_scaled = self.scale_min + (self.scale_max - self.scale_min) * (xm_norm ** self.gamma_miou)
        return float(np.clip(xm_scaled, 0.0, 1.0))

    def _miou_from_table(self, slice_size: int, overlap_ratio: float) -> float:
        if not self.miou_table or not self.miou_overlaps:
            return self._miou_surrogate(slice_size, overlap_ratio)
        # Find nearest slice_size key
        sizes = [s['slice_size'] for s in self.slicing_strategies]
        nearest_size = min(sizes, key=lambda s: abs(s - slice_size))
        key = str(nearest_size)
        arr = self.miou_table.get(key)
        if not arr:
            return self._miou_surrogate(slice_size, overlap_ratio)
        # Linear interp over overlaps
        xs = np.array(self.miou_overlaps, dtype=float)
        ys = np.array(arr, dtype=float)
        ov = float(np.clip(overlap_ratio, float(xs.min()), float(xs.max())))
        miou_raw = float(np.interp(ov, xs, ys))
        # Normalize and scale
        if self.miou_norm_range and len(self.miou_norm_range) == 2:
            mn, mx = float(self.miou_norm_range[0]), float(self.miou_norm_range[1])
        else:
            mn = self._miou_min if self._miou_min is not None else float(ys.min())
            mx = self._miou_max if self._miou_max is not None else float(ys.max())
        if mx <= mn:  # fallback
            xm = self._miou_surrogate(slice_size, overlap_ratio)
            return xm
        miou_norm = (miou_raw - mn) / (mx - mn)
        miou_norm = float(np.clip(miou_norm, 0.0, 1.0))
        xm_scaled = self.scale_min + (self.scale_max - self.scale_min) * (miou_norm ** self.gamma_miou)
        return float(np.clip(xm_scaled, 0.0, 1.0))

    """
    Refactored environment for single-source satellite RL training.
    
    Each episode:
    - Randomly selects one source satellite for task generation
    - Selects the nearest visible MEO leader per task
    - Observes only the 10 nearest compute satellites + nearest ground station
    - Terminates after a fixed number of task decisions
    """
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
        
        # Episode termination: number of tasks to process per episode
        self.max_tasks_per_episode = sim_config.get('max_tasks_per_episode', 100)
        
        # Observation parameters
        self.num_nearest_compute = sim_config.get('num_nearest_compute_satellites', 10)
        self.isl_visibility_range_km = sim_config.get('isl_visibility_range_km', 5000.0)

        # --- Initialize Environment Components ---
        self.orbit_propagator = OrbitPropagator(self.start_time)
        
        # TaskGenerator for unified tasks
        flops_per_pixel = float(sim_config.get('dl_model', {}).get('flops_per_pixel', 1000.0))
        self.task_generator = TaskGenerator(
            task_config=sim_config['task_generation'],
            area_config=sim_config['target_area'],
            flops_per_pixel=flops_per_pixel,
        )

        # Ensure TLEs are present; if not, preprocess configs to inject TLEs from cache/remote
        def _needs_tle(cfgs: List[Dict[str, Any]]) -> bool:
            return any('tle' not in c or not c.get('tle') for c in (cfgs or []))
        if (_needs_tle(sat_configs.get('source_satellites', [])) or
            _needs_tle(sat_configs.get('compute_satellites', [])) or
            _needs_tle(sat_configs.get('leader_satellites', []))):
            try:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                sat_configs = preprocess_satellite_configs(project_root, sat_configs)
            except Exception:
                pass

        # Initialize satellites
        self.source_satellites = [Satellite(sat_id=cfg['sat_id'], config=cfg) 
                                 for cfg in sat_configs.get('source_satellites', [])]
        self.compute_satellites = [Satellite(sat_id=cfg['sat_id'], config=cfg) 
                                  for cfg in sat_configs.get('compute_satellites', [])]
        self.leader_satellites = [Satellite(sat_id=cfg['sat_id'], config=cfg) 
                                 for cfg in sat_configs.get('leader_satellites', [])]
        
        self.all_satellites = self.source_satellites + self.compute_satellites + self.leader_satellites
        self.num_source_satellites = len(self.source_satellites)
        self.num_compute_satellites = len(self.compute_satellites)
        self.num_leader_satellites = len(self.leader_satellites)

        # Initialize ground stations
        self.ground_stations = self._init_ground_stations()
        self.num_ground_stations = len(self.ground_stations)

        # --- Initialize Compute Node Registry ---
        reset_compute_node_registry()
        self._init_compute_nodes()

        # --- Episode State ---
        self.current_source_satellite: Optional[Satellite] = None
        self.current_leader_satellite: Optional[ComputeNode] = None
        self.current_task: Optional[Task] = None
        self.task_queue: deque[Task] = deque()
        self.completed_tasks: deque[Task] = deque()
        
        # Episode tracking
        self.tasks_processed = 0  # Number of tasks processed in this episode
        self.steps_taken = 0  # Total steps (including invalid steps)
        self.valid_steps_taken = 0  # Valid steps only
        self.successful_tasks_count = 0  # Number of tasks that met latency constraint

        # Debug / diagnostics
        self.debug_latency: bool = bool(sim_config.get('debug_latency', True))
        self._last_latency_debug: Optional[Dict[str, Any]] = None

        # --- Model performance / reward config ---
        self.performance_cfg: Dict[str, Any] = sim_config.get('model_performance', {})
        self.reward_mode: str = str(self.performance_cfg.get('reward_mode', 'miou_soft_penalty'))
        self.use_monotone_surrogate: bool = bool(self.performance_cfg.get('use_monotone_surrogate', True))
        self.penalty_lambda: float = float(self.performance_cfg.get('penalty_lambda', 1.0))
        self.voi_beta: float = float(self.performance_cfg.get('beta', 0.7))
        self.epsilon_latency_bonus: float = float(self.performance_cfg.get('epsilon_latency_bonus', 0.0))
        self.miou_overlaps: List[float] = self.performance_cfg.get('overlaps', [])
        self.miou_table: Dict[str, List[float]] = self.performance_cfg.get('miou_table', {})
        self.miou_norm_range: List[float] = self.performance_cfg.get('normalize', [])
        self.surrogate_params: Dict[str, Any] = self.performance_cfg.get('surrogate', {})
        self.scale_min: float = float(self.surrogate_params.get('scale', [0.8, 1.0])[0] if self.surrogate_params.get('scale') else 0.8)
        self.scale_max: float = float(self.surrogate_params.get('scale', [0.8, 1.0])[1] if self.surrogate_params.get('scale') else 1.0)
        self.gamma_miou: float = float(self.surrogate_params.get('gamma', 1.0))

        # Precompute table min/max if provided
        self._miou_min: Optional[float] = None
        self._miou_max: Optional[float] = None
        if self.miou_table:
            vals = [v for arr in self.miou_table.values() for v in arr]
            if vals:
                self._miou_min = float(min(vals))
                self._miou_max = float(max(vals))

        # --- Define Action and Observation Spaces ---
        self._define_spaces()

    def _init_compute_nodes(self):
        """Initialize compute nodes for all LEO satellites, MEO satellites, and ground stations."""
        # LEO compute satellites
        for sat in self.compute_satellites:
            node = ComputeNode(
                node_type="LEO",
                node_id=sat.id,
                compute_gflops=sat.compute_gflops,
                position_ecef=sat.position_ecef,
                name=sat.name
            )
        
        # MEO leader satellites (can also compute)
        for sat in self.leader_satellites:
            node = ComputeNode(
                node_type="MEO",
                node_id=sat.id,
                compute_gflops=sat.compute_gflops,
                position_ecef=sat.position_ecef,
                name=sat.name
            )
        
        # Ground stations
        for gs in self.ground_stations:
            node = ComputeNode(
                node_type="GROUND_STATION",
                node_id=gs.id,
                compute_gflops=gs.compute_gflops,
                position_ecef=gs.position_ecef,
                name=gs.name
            )

    def _define_spaces(self):
        """Define action and observation spaces."""
        num_slice_sizes = len(self.slicing_strategies)

        # Use the smallest slice size to estimate the worst-case k (no overlap)
        if num_slice_sizes > 0:
            min_slice_size = min(s['slice_size'] for s in self.slicing_strategies)
        else:
            min_slice_size = 512
        max_k = self._calculate_k(min_slice_size)

        # Destinations: num_nearest_compute satellites + 1 ground station
        num_destinations = self.num_nearest_compute + 1
        
        self.action_space = spaces.Dict({
            'slice_strategy': spaces.MultiDiscrete([num_slice_sizes]),
            'assignment': spaces.MultiDiscrete([num_destinations] * max_k)
        })

        self.observation_space = spaces.Dict({
            "task_origin_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "task_info": spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32),
            "leader_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "compute_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_nearest_compute, 3), dtype=np.float32),
            "compute_queues": spaces.Box(low=0, high=np.inf, shape=(self.num_nearest_compute,), dtype=np.float32),
            "ground_station_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "ground_station_queue": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        })

    def _init_ground_stations(self) -> List[GroundStation]:
        """Initialize ground stations based on configuration."""
        gs_config = self.sim_config['ground_station']
        if gs_config.get('generation_method') == 'hex_grid':
            area = self.sim_config['target_area']
            radius_km = gs_config['grid_radius_km']
            centers = generate_hex_grid_centers(
                area['min_lon_deg'], area['max_lon_deg'],
                area['min_lat_deg'], area['max_lat_deg'],
                radius_km
            )
            return [GroundStation(gs_id=i, lat_deg=lat, lon_deg=lon, config=gs_config) 
                   for i, (lat, lon) in enumerate(centers)]
        else:
            return [GroundStation(gs_id=0, lat_deg=0, lon_deg=0, config=gs_config)]

    def _calculate_k(self, slice_size: int) -> int:
        """Calculate number of slices based on task properties.
        If current_task is not yet set, fall back to configured fixed_size (if fixed_mode)
        or to image_size_range max.
        """
        if not self.current_task:
            tg = self.sim_config.get('task_generation', {})
            if bool(tg.get('fixed_mode', True)):
                fs = tg.get('fixed_size', [6000, 6000])
                img_w = int(fs[0])
                img_h = int(fs[1])
            else:
                max_size = tg.get('image_size_range', [1024, 1024])[1]
                img_w, img_h = int(max_size), int(max_size)
        else:
            img_w, img_h = self.current_task.width, self.current_task.height

        stride = float(slice_size)
        if stride <= 0:
            return 1
        slices_w = math.ceil((img_w - slice_size) / stride) + 1 if img_w > slice_size else 1
        slices_h = math.ceil((img_h - slice_size) / stride) + 1 if img_h > slice_size else 1
        return slices_w * slices_h

    def _find_nearest_visible_leader(self, source_sat: Satellite) -> Optional[ComputeNode]:
        """
        Find the nearest MEO leader satellite that is visible (within ISL range).
        
        Returns:
            ComputeNode wrapping the MEO satellite, or None if no visible leader exists.
        """
        if self.num_leader_satellites == 0:
            return None
        
        source_pos = source_sat.position_ecef
        min_distance = float('inf')
        nearest_leader = None
        
        for leader_sat in self.leader_satellites:
            distance = links.get_distance_km(source_pos, leader_sat.position_ecef)
            
            # Check if within visibility range
            if distance <= self.isl_visibility_range_km and distance < min_distance:
                min_distance = distance
                nearest_leader = leader_sat
        
        if nearest_leader is None:
            return None
        
        # Find the corresponding ComputeNode
        all_nodes = get_all_compute_nodes()
        for node in all_nodes:
            if node.node_type == "MEO" and node.node_id == nearest_leader.id:
                return node
        
        return None

    def _find_nearest_compute_satellites(self, leader_pos: np.ndarray, 
                                        num_nearest: int) -> List[ComputeNode]:
        """
        Find the num_nearest compute satellites closest to the leader.
        Includes both LEO and MEO satellites (the leader itself can be selected as a compute node).
        
        Returns:
            List of ComputeNode objects (LEO and MEO)
        """
        all_nodes = get_all_compute_nodes()
        compute_nodes = [n for n in all_nodes if n.node_type in ["LEO", "MEO"]]
        
        # Sort by distance to leader (leader will have distance ~0 and thus be included)
        distances = [(node, links.get_distance_km(leader_pos, node.position_ecef)) 
                    for node in compute_nodes]
        distances.sort(key=lambda x: x[1])
        
        # Return the nearest num_nearest satellites
        return [node for node, _ in distances[:num_nearest]]

    def _find_nearest_ground_station(self, leader_pos: np.ndarray) -> Optional[ComputeNode]:
        """Find the nearest ground station to the leader."""
        all_nodes = get_all_compute_nodes()
        gs_nodes = [n for n in all_nodes if n.node_type == "GROUND_STATION"]
        
        if not gs_nodes:
            return None
        
        distances = [(node, links.get_distance_km(leader_pos, node.position_ecef)) 
                    for node in gs_nodes]
        distances.sort(key=lambda x: x[1])
        
        return distances[0][0]

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Construct observation from current state."""
        if not self.current_task:
            return {key: np.zeros(space.shape, dtype=space.dtype) 
                   for key, space in self.observation_space.spaces.items()}

        # If leader not chosen yet (e.g., right after reset), try to find it now
        if not self.current_leader_satellite:
            self.current_leader_satellite = self._find_nearest_visible_leader(self.current_source_satellite)

        task_origin_pos = self.current_task.origin.position_ecef.astype(np.float32)
        # Build task_info with extra difficulty features
        data_bits = float(self.current_task.data_size_bits)
        max_lat = float(self.current_task.max_latency_sec)
        ratio_sec_per_bit = max_lat / max(data_bits, 1.0)
        required_rate_bps = data_bits / max(max_lat, 1e-6)
        task_info = np.array([
            self.current_task.width,
            self.current_task.height,
            max_lat,
            data_bits,
            self.current_task.required_flops,
            ratio_sec_per_bit,
            required_rate_bps,
        ], dtype=np.float32)
        
        # Leader position (may be zeros if leader not found yet)
        leader_pos = (self.current_leader_satellite.position_ecef.astype(np.float32)
                      if self.current_leader_satellite else np.zeros(3, dtype=np.float32))
        
        # Get nearest compute satellites
        if self.current_leader_satellite is not None:
            nearest_compute = self._find_nearest_compute_satellites(
                self.current_leader_satellite.position_ecef, 
                self.num_nearest_compute
            )
        else:
            nearest_compute = []
        
        # Pad with zeros if fewer than num_nearest_compute available
        compute_pos_list = [node.position_ecef for node in nearest_compute]
        while len(compute_pos_list) < self.num_nearest_compute:
            compute_pos_list.append(np.zeros(3))
        compute_pos = np.array(compute_pos_list, dtype=np.float32)
        
        compute_queues = np.array([node.get_queue_load_flops() for node in nearest_compute] + 
                                 [0.0] * (self.num_nearest_compute - len(nearest_compute)), 
                                 dtype=np.float32)
        
        # Get nearest ground station
        if self.current_leader_satellite is not None:
            nearest_gs = self._find_nearest_ground_station(self.current_leader_satellite.position_ecef)
        else:
            nearest_gs = None
        gs_pos = nearest_gs.position_ecef.astype(np.float32) if nearest_gs else np.zeros(3, dtype=np.float32)
        gs_queue = np.array([nearest_gs.get_queue_load_flops()], dtype=np.float32) if nearest_gs else np.array([0.0], dtype=np.float32)

        return {
            "task_origin_pos": task_origin_pos,
            "task_info": task_info,
            "leader_pos": leader_pos,
            "compute_pos": compute_pos,
            "compute_queues": compute_queues,
            "ground_station_pos": gs_pos,
            "ground_station_queue": gs_queue,
        }

    def _generate_new_tasks(self) -> List[Task]:
        """Generate new tasks from the current source satellite."""
        if not self.current_source_satellite:
            return []
        
        sim_time = self.orbit_propagator.simulation_time
        new_tasks = self.task_generator.generate_unified_tasks([self.current_source_satellite], sim_time)
        self.task_queue.extend(new_tasks)
        return new_tasks

    def _get_next_task(self) -> bool:
        """
        Get the next task from the queue.
        
        Returns:
            True if a task was obtained, False if queue is empty
        """
        if self.task_queue:
            self.current_task = self.task_queue.popleft()
            return True
        return False

    def _update_compute_node_queues(self, current_time: float):
        """Update all compute node queues (process completed tasks)."""
        all_nodes = get_all_compute_nodes()
        for node in all_nodes:
            node.update_queue(current_time)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Reset time and counters
        self.orbit_propagator.simulation_time = self.start_time
        self.steps_taken = 0
        self.valid_steps_taken = 0
        self.tasks_processed = 0
        self.successful_tasks_count = 0
        self.task_queue.clear()
        self.completed_tasks.clear()

        # Clear all compute node queues
        all_nodes = get_all_compute_nodes()
        for node in all_nodes:
            node.clear_queue()
            node.last_update_time = 0.0

        # Update satellite positions
        self.orbit_propagator.update_satellite_positions(self.all_satellites)
        
        # Update compute node positions
        for node in all_nodes:
            if node.node_type == "LEO":
                for sat in self.compute_satellites:
                    if sat.id == node.node_id:
                        node.update_position(sat.position_ecef)
                        break
            elif node.node_type == "MEO":
                for sat in self.leader_satellites:
                    if sat.id == node.node_id:
                        node.update_position(sat.position_ecef)
                        break
            elif node.node_type == "GROUND_STATION":
                for gs in self.ground_stations:
                    if gs.id == node.node_id:
                        node.update_position(gs.position_ecef)
                        break

        # Randomly select a source satellite for this episode
        self.current_source_satellite = self._rng.choice(self.source_satellites)
        
        # Initialize task generation for the selected source satellite
        self.task_generator.reset_arrivals(self.orbit_propagator.simulation_time, [self.current_source_satellite])
        
        # Generate initial tasks and get the first one
        self._generate_new_tasks()
        if not self._get_next_task():
            # If no task available, advance time until we get one
            while not self.task_queue and self.steps_taken < 1000:
                self.orbit_propagator.advance_simulation_time(60)
                self.orbit_propagator.update_satellite_positions(self.all_satellites)
                self._generate_new_tasks()
                self.steps_taken += 1
            self._get_next_task()

        return self._get_obs(), {}

    def _calculate_latency_and_reward(self, k: int, assignment: np.ndarray, slice_size: int, overlap_ratio: float) -> Tuple[float, float, float, bool, Dict[str, Any]]:
        """
        Calculate latency and reward for the current task decision.
        Modified logic per user's requirement:
        - Fixed task size (handled by generator), per-slice data bits = slice_size*slice_size*24
        - Compute per-slice FLOPs from empirical table by slice_size
        - Network transmission uses per-slice bits; source->leader sends ALL k slices; leader->dest sends
          assigned slices per destination.
        - Reward uses hard constraint: reward=xm if total_latency<=max_latency else 0
        """
        if not self.current_task or not self.current_leader_satellite:
            return 0.0, 0.0, 0.0, False, {}

        source_sat = self.current_task.origin
        leader_node = self.current_leader_satellite

        # Per-slice bits based on 24 bpp and chosen slice_size
        slice_pixels = float(slice_size * slice_size)
        slice_data_size_bits = float(slice_pixels * 24.0)

        # FLOPs per slice based on empirical table (GFLOPs)
        flops_table_gflops = {
            128: 21.73,
            256: 86.90,
            512: 347.60,
            1024: 1390.0,  # 1.39 TFLOPs
            2048: 5560.0,  
        }
        # nearest key
        nearest_key = min(flops_table_gflops.keys(), key=lambda s: abs(s - slice_size))
        slice_flops = float(flops_table_gflops[nearest_key]) * 1e9  # convert to FLOPs

        debug: Dict[str, Any] = {
            'task_id': self.current_task.id,
            'width': int(self.current_task.width),
            'height': int(self.current_task.height),
            'max_latency_sec': float(self.current_task.max_latency_sec),
            'k_slices': int(k),
            'per_slice_bits': float(slice_data_size_bits),
            'per_slice_flops': float(slice_flops),
        }

        # Initial hop: source RS -> leader via ISL (send ALL k slices)
        dist_rs_leader = links.get_distance_km(source_sat.position_ecef, leader_node.position_ecef)
        isl_cap_rs_leader = links.get_isl_capacity_bps(dist_rs_leader, self.isl_frequency_ghz)
        t_initial_tx = (k * slice_data_size_bits) / isl_cap_rs_leader if isl_cap_rs_leader > 0 else np.inf
        t_initial_prop = (dist_rs_leader * 1000.0) / constants.SPEED_OF_LIGHT_M_S
        t_initial = t_initial_tx + t_initial_prop
        debug['source_to_leader'] = {
            'distance_km': float(dist_rs_leader),
            'isl_capacity_bps': float(isl_cap_rs_leader),
            't_initial_sec': float(t_initial),           # tx + propagation
            't_initial_tx_sec': float(t_initial_tx),     # tx only (for diagnostics)
            't_prop_sec': float(t_initial_prop),
        }

        latencies: List[float] = []
        per_dest: List[Dict[str, Any]] = []

        # Get the nearest compute satellites and ground station for this observation
        nearest_compute = self._find_nearest_compute_satellites(
            leader_node.position_ecef,
            self.num_nearest_compute
        )
        nearest_gs = self._find_nearest_ground_station(leader_node.position_ecef)

        # Build destination list: nearest_compute + nearest_gs
        destinations = nearest_compute + ([nearest_gs] if nearest_gs else [])

        # Process assignment
        for dest_idx, group in itertools.groupby(sorted(enumerate(assignment[:k]), key=lambda x: x[1]), key=lambda x: x[1]):
            indices = list(group)
            num_slices_for_dest = len(indices)
            entry: Dict[str, Any] = {
                'dest_index': int(dest_idx),
                'num_slices': int(num_slices_for_dest),
            }

            if dest_idx < len(destinations):
                dest_node = destinations[dest_idx]
                entry['node_type'] = dest_node.node_type
                entry['node_compute_gflops'] = float(dest_node.compute_gflops)

                if dest_node.node_type == "GROUND_STATION":
                    # Downlink to ground station
                    dist = links.get_distance_km(leader_node.position_ecef, dest_node.position_ecef)
                    capacity = links.get_downlink_capacity_bps(dist, self.downlink_frequency_ghz)
                    t_down_tx = (num_slices_for_dest * slice_data_size_bits) / capacity if capacity > 0 else np.inf
                    t_down_prop = (dist * 1000.0) / constants.SPEED_OF_LIGHT_M_S
                    t_trans = t_down_tx + t_down_prop
                    t_comp = (num_slices_for_dest * slice_flops) / (dest_node.compute_gflops * 1e9)
                    total = t_initial + t_trans + t_comp

                    entry.update({
                        'distance_km': float(dist),
                        'downlink_capacity_bps': float(capacity),
                        't_trans_sec': float(t_trans),            # tx + propagation
                        't_trans_tx_sec': float(t_down_tx),       # tx only
                        't_prop_sec': float(t_down_prop),
                        't_comp_sec': float(t_comp),
                        'path_latency_sec': float(total),
                    })
                    latencies.append(total)
                else:
                    # ISL to compute satellite (LEO or MEO)
                    dist_isl = links.get_distance_km(leader_node.position_ecef, dest_node.position_ecef)
                    isl_capacity = links.get_isl_capacity_bps(dist_isl, self.isl_frequency_ghz)
                    t_isl_tx = (num_slices_for_dest * slice_data_size_bits) / isl_capacity if isl_capacity > 0 else np.inf
                    t_isl_prop = (dist_isl * 1000.0) / constants.SPEED_OF_LIGHT_M_S
                    t_isl = t_isl_tx + t_isl_prop
                    queue_before = dest_node.get_queue_load_flops()
                    t_queue = queue_before / (dest_node.compute_gflops * 1e9)
                    t_comp = (num_slices_for_dest * slice_flops) / (dest_node.compute_gflops * 1e9)

                    # Add task to queue (in FLOPs)
                    task_flops = num_slices_for_dest * slice_flops
                    dest_node.add_task_slice(self.current_task.id, dest_idx, task_flops,
                                           self.orbit_propagator.simulation_time.timestamp())

                    total = t_initial + t_isl + t_queue + t_comp
                    entry.update({
                        'distance_km': float(dist_isl),
                        'isl_capacity_bps': float(isl_capacity),
                        'queue_load_flops_before': float(queue_before),
                        't_isl_sec': float(t_isl),               # tx + propagation
                        't_isl_tx_sec': float(t_isl_tx),         # tx only
                        't_prop_sec': float(t_isl_prop),
                        't_queue_sec': float(t_queue),
                        't_comp_sec': float(t_comp),
                        'path_latency_sec': float(total),
                    })
                    latencies.append(total)
            else:
                entry['error'] = 'dest_index_out_of_range'
                latencies.append(np.inf)
            per_dest.append(entry)

        total_latency_sec = max(latencies) if latencies else 0.0

        # Reward: hard constraint on latency, reward equals mIoU otherwise 0
        feasible = bool((total_latency_sec > 0) and (not np.isinf(total_latency_sec)) and (total_latency_sec <= float(self.current_task.max_latency_sec)))

        # mIoU from explicit mapping by slice_size (nearest-key fallback)
        miou_map = {
            128: 0.60,
            256: 0.65,
            512: 0.70,
            1024: 0.76,
            2048: 0.82,
        }
        nearest_key_m = min(miou_map.keys(), key=lambda s: abs(s - slice_size))
        xm = float(miou_map[nearest_key_m])
        xm_source = 'discrete_map'

        reward = float(xm) if feasible else 0.0

        debug.update({
            'destinations': per_dest,
            'total_latency_sec': float(total_latency_sec),
            'max_latency_sec': float(self.current_task.max_latency_sec),
            'reward': float(reward),
            'feasible': feasible,
        })
        if self.debug_latency:
            self._last_latency_debug = debug

        # return total_latency_sec, reward, xm, feasible, debug

        debug: Dict[str, Any] = {
            'task_id': self.current_task.id,
            'width': int(self.current_task.width),
            'height': int(self.current_task.height),
            'max_latency_sec': float(self.current_task.max_latency_sec),
            'data_size_bits': int(self.current_task.data_size_bits),
            'required_flops': float(self.current_task.required_flops),
            'k_slices': int(k),
        }

        # Initial hop: source RS -> leader via ISL
        dist_rs_leader = links.get_distance_km(source_sat.position_ecef, leader_node.position_ecef)
        isl_cap_rs_leader = links.get_isl_capacity_bps(dist_rs_leader, self.isl_frequency_ghz)
        t_initial = (self.current_task.data_size_bits) / isl_cap_rs_leader if isl_cap_rs_leader > 0 else np.inf
        debug['source_to_leader'] = {
            'distance_km': float(dist_rs_leader),
            'isl_capacity_bps': float(isl_cap_rs_leader),
            't_initial_sec': float(t_initial),
        }

        latencies: List[float] = []
        per_dest: List[Dict[str, Any]] = []
        
        # Get the nearest compute satellites and ground station for this observation
        nearest_compute = self._find_nearest_compute_satellites(
            leader_node.position_ecef, 
            self.num_nearest_compute
        )
        nearest_gs = self._find_nearest_ground_station(leader_node.position_ecef)
        
        # Build destination list: nearest_compute + nearest_gs
        destinations = nearest_compute + ([nearest_gs] if nearest_gs else [])

        # Process assignment
        for dest_idx, group in itertools.groupby(sorted(enumerate(assignment[:k]), key=lambda x: x[1]), key=lambda x: x[1]):
            indices = list(group)
            num_slices_for_dest = len(indices)
            entry: Dict[str, Any] = {
                'dest_index': int(dest_idx),
                'num_slices': int(num_slices_for_dest),
            }
            
            if dest_idx < len(destinations):
                dest_node = destinations[dest_idx]
                entry['node_type'] = dest_node.node_type
                entry['node_compute_gflops'] = float(dest_node.compute_gflops)

                if dest_node.node_type == "GROUND_STATION":
                    # Downlink to ground station
                    dist = links.get_distance_km(leader_node.position_ecef, dest_node.position_ecef)
                    capacity = links.get_downlink_capacity_bps(dist, self.downlink_frequency_ghz)
                    t_trans = (num_slices_for_dest * slice_data_size_bits) / capacity if capacity > 0 else np.inf
                    t_comp = (num_slices_for_dest * slice_flops) / (dest_node.compute_gflops * 1e9)
                    total = t_initial + t_trans + t_comp

                    entry.update({
                        'distance_km': float(dist),
                        'downlink_capacity_bps': float(capacity),
                        't_trans_sec': float(t_trans),
                        't_comp_sec': float(t_comp),
                        'path_latency_sec': float(total),
                    })
                    latencies.append(total)
                else:
                    # ISL to compute satellite (LEO or MEO)
                    dist_isl = links.get_distance_km(leader_node.position_ecef, dest_node.position_ecef)
                    isl_capacity = links.get_isl_capacity_bps(dist_isl, self.isl_frequency_ghz)
                    t_isl = (num_slices_for_dest * slice_data_size_bits) / isl_capacity if isl_capacity > 0 else np.inf
                    queue_before = dest_node.get_queue_load_flops()
                    t_queue = queue_before / (dest_node.compute_gflops * 1e9)
                    t_comp = (num_slices_for_dest * slice_flops) / (dest_node.compute_gflops * 1e9)
                    
                    # Add task to queue
                    task_flops = num_slices_for_dest * slice_flops
                    dest_node.add_task_slice(self.current_task.id, dest_idx, task_flops, 
                                           self.orbit_propagator.simulation_time.timestamp())
                    
                    total = t_initial + t_isl + t_queue + t_comp
                    entry.update({
                        'distance_km': float(dist_isl),
                        'isl_capacity_bps': float(isl_capacity),
                        'queue_load_flops_before': float(queue_before),
                        't_isl_sec': float(t_isl),
                        't_queue_sec': float(t_queue),
                        't_comp_sec': float(t_comp),
                        'path_latency_sec': float(total),
                    })
                    latencies.append(total)
            else:
                entry['error'] = 'dest_index_out_of_range'
                latencies.append(np.inf)
            per_dest.append(entry)

        total_latency_sec = max(latencies) if latencies else 0.0
        
        # Feasibility and mIoU-hard-constraint reward
        feasible = bool((total_latency_sec > 0) and (not np.isinf(total_latency_sec)) and (total_latency_sec <= float(self.current_task.max_latency_sec)))

        miou_map = {
            128: 0.40,
            256: 0.55,
            512: 0.70,
            1024: 0.82,
            2048: 0.90,
        }
        nearest_key_m = min(miou_map.keys(), key=lambda s: abs(s - slice_size))
        xm = float(miou_map[nearest_key_m])

        reward = float(xm) if feasible else 0.0
        
        debug.update({
            'destinations': per_dest,
            'total_latency_sec': float(total_latency_sec),
            'max_latency_sec': float(self.current_task.max_latency_sec),
            'reward': float(reward),
            'feasible': bool(feasible),
        })
        if self.debug_latency:
            self._last_latency_debug = debug

        return total_latency_sec, float(reward), float(xm), bool(feasible), debug

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.
        
        Returns:
            (obs, reward, terminated, truncated, info)
        """
        info: Dict[str, Any] = {}
        self.steps_taken += 1
        
        # Update compute node queues based on time elapsed since last step
        current_time_sec = (self.orbit_propagator.simulation_time - self.start_time).total_seconds()
        self._update_compute_node_queues(current_time_sec)
        
        # Check if we have a current task
        if not self.current_task:
            # Try to get next task
            if not self._get_next_task():
                # No task available - episode ends
                obs = self._get_obs()
                truncated = self.tasks_processed >= self.max_tasks_per_episode
                terminated = True
                info['reason'] = 'No more tasks available'
                return obs, 0.0, terminated, truncated, info
        
        # Find nearest visible MEO leader
        self.current_leader_satellite = self._find_nearest_visible_leader(self.current_source_satellite)
        
        if not self.current_leader_satellite:
            # No visible leader - skip this step (invalid)
            info['reason'] = 'No visible MEO leader'
            info['is_valid_step'] = False
            obs = self._get_obs()
            return obs, 0.0, False, False, info
        
        # Check if we have enough compute satellites
        nearest_compute = self._find_nearest_compute_satellites(
            self.current_leader_satellite.position_ecef, 
            self.num_nearest_compute
        )
        
        if len(nearest_compute) < self.num_nearest_compute:
            # Not enough compute satellites - skip this step (invalid)
            info['reason'] = f'Only {len(nearest_compute)} compute satellites visible (need {self.num_nearest_compute})'
            info['is_valid_step'] = False
            obs = self._get_obs()
            return obs, 0.0, False, False, info
        
        # Valid step - process the action
        info['is_valid_step'] = True
        self.valid_steps_taken += 1
        
        # Parse action (overlap removed; keep backward-compat if provided)
        ss = action['slice_strategy']
        if isinstance(ss, (list, tuple, np.ndarray)):
            ss = list(ss)
        else:
            ss = [int(ss)]
        slice_size_idx = int(ss[0])
        assignment = action['assignment']
        strategy = self.slicing_strategies[slice_size_idx]
        slice_size = strategy['slice_size']
        k = self._calculate_k(slice_size)

        # Calculate latency and reward using new logic
        total_latency_sec, reward, xm, feasible, debug = self._calculate_latency_and_reward(
            k, assignment, slice_size, 0.0
        )

        # Advance simulation time by the task latency (or fallback)
        time_to_advance = total_latency_sec
        if np.isinf(total_latency_sec) or total_latency_sec <= 0:
            time_to_advance = max(float(self.current_task.max_latency_sec), 1.0) if self.current_task else 60.0

        self.orbit_propagator.advance_simulation_time(time_to_advance)
        self.orbit_propagator.update_satellite_positions(self.all_satellites)
        
        # Update compute node positions
        all_nodes = get_all_compute_nodes()
        for node in all_nodes:
            if node.node_type == "LEO":
                for sat in self.compute_satellites:
                    if sat.id == node.node_id:
                        node.update_position(sat.position_ecef)
                        break
            elif node.node_type == "MEO":
                for sat in self.leader_satellites:
                    if sat.id == node.node_id:
                        node.update_position(sat.position_ecef)
                        break

        # Generate next task (fixed-mode: one per step)
        _ = self._generate_new_tasks()

        # Mark current task as completed and fetch next
        processed_task = self.current_task
        _ = self._get_next_task()
        self.tasks_processed += 1
        if processed_task is not None:
            self.completed_tasks.append(processed_task)

        # Track success count
        if bool(feasible):
            self.successful_tasks_count += 1

        # Check termination
        truncated = self.tasks_processed >= self.max_tasks_per_episode
        terminated = truncated

        obs = self._get_obs()
        info.update({
            'total_latency': float(total_latency_sec),
            'chosen_slice_size': int(slice_size),
            
            'calculated_k': int(k),
            'tasks_processed': int(self.tasks_processed),
            'valid_steps_taken': int(self.valid_steps_taken),
            'successful_tasks_count': int(self.successful_tasks_count),
            'miou': float(xm),
            'feasible': bool(feasible),
            'reward_mode': 'miou_hard_constraint',
            'reward': float(reward),
            'latency_debug': debug if self.debug_latency else None,
            'assignment': [int(x) for x in assignment[:k]],
            'leader_node_id': int(self.current_leader_satellite.node_id) if self.current_leader_satellite else -1,
            'sim_time': self.orbit_propagator.simulation_time.isoformat() + 'Z',
        })

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"--- Sim Time: {self.orbit_propagator.simulation_time}, Step: {self.steps_taken} ---")
            print(f"Current Source Satellite: {self.current_source_satellite}")
            print(f"Current Leader: {self.current_leader_satellite}")
            print(f"Current Task: {self.current_task}")
            print(f"Tasks Processed: {self.tasks_processed}/{self.max_tasks_per_episode}")
