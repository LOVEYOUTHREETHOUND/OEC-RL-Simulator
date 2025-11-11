# -*- coding: utf-8 -*-
"""
This module defines the Task and TaskGenerator classes.

A Task represents a single remote sensing image to be processed, and the
TaskGenerator is responsible for creating these tasks based on different
triggering conditions (e.g., satellite overflight, UE request).
"""

from typing import Dict, Any, Tuple, Union, List
import numpy as np

# Forward-declare Satellite to avoid circular import issues
if False:
    from .satellite import Satellite
from .ue import UserEquipment


class Task:
    """
    Represents a single data processing task, which can originate from a satellite
    or a ground-based User Equipment (UE).
    """
    def __init__(self, task_id: int, image_size_wh: Tuple[int, int], bands: int, bits_per_pixel: int, origin: Union['Satellite', UserEquipment, None] = None):
        self.id = task_id
        self.width = image_size_wh[0]
        self.height = image_size_wh[1]
        self.bands = bands
        self.bits_per_pixel = bits_per_pixel
        self.origin = origin

    @property
    def task_type(self) -> str:
        from .satellite import Satellite
        if isinstance(self.origin, Satellite):
            return 'remote_sensing'
        elif isinstance(self.origin, UserEquipment):
            return 'ue_generated'
        return 'unknown'

    @property
    def data_size_bits(self) -> int:
        return self.width * self.height * self.bands * self.bits_per_pixel

    def __repr__(self) -> str:
        return f"Task(id={self.id}, type={self.task_type}, origin={self.origin}, data_size={self.data_size_bits / 8e6:.2f} MB)"


class TaskGenerator:
    """
    Generates tasks based on satellite positions and UE requests.
    """
    def __init__(self, rs_config: Dict[str, Any], ue_config: Dict[str, Any], area_config: Dict[str, Any]):
        """
        Initializes the TaskGenerator.
        Args:
            rs_config: Configuration for remote sensing tasks.
            ue_config: Configuration for User Equipment tasks.
            area_config: Configuration for the target geographical area.
        """
        self.rs_task_config = rs_config
        self.ue_task_config = ue_config
        self.area_config = area_config
        self._rng = np.random.default_rng()
        self._next_task_id = 0

    def _generate_task_properties(self, config_group: Dict[str, Any]) -> Tuple[Tuple[int, int], int, int]:
        """Helper function to generate random properties for a task."""
        size_range = config_group.get('image_size_range', (1024, 1024))
        bands_range = config_group.get('bands_range', (3, 3))
        bits_per_pixel = config_group.get('bits_per_pixel', 8)

        width = self._rng.integers(size_range[0], size_range[1], endpoint=True)
        height = self._rng.integers(size_range[0], size_range[1], endpoint=True)
        bands = self._rng.integers(bands_range[0], bands_range[1], endpoint=True)
        
        return (width, height), bands, bits_per_pixel

    def generate_ue_task(self, ue: UserEquipment) -> Task:
        """Generates a new task originating from a specific UE."""
        size, bands, bpp = self._generate_task_properties(self.ue_task_config)
        task = Task(task_id=self._next_task_id, image_size_wh=size, bands=bands, bits_per_pixel=bpp, origin=ue)
        self._next_task_id += 1
        return task

    def check_and_generate_remote_sensing_tasks(self, satellites: List['Satellite']) -> List[Task]:
        """
        Checks which source satellites are over the target area and generates tasks for them.
        """
        new_tasks = []
        for sat in satellites:
            lat, lon, _ = sat.get_geodetic_coords()
            if not np.isnan(lat) and (self.area_config['min_lat_deg'] <= lat <= self.area_config['max_lat_deg'] and
                self.area_config['min_lon_deg'] <= lon <= self.area_config['max_lon_deg']):
                
                # Simple check to avoid generating tasks too frequently for the same satellite can be added here
                
                size, bands, bpp = self._generate_task_properties(self.rs_task_config)
                task = Task(task_id=self._next_task_id, image_size_wh=size, bands=bands, bits_per_pixel=bpp, origin=sat)
                self._next_task_id += 1
                new_tasks.append(task)
        return new_tasks
