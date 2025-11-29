# -*- coding: utf-8 -*-
"""
This module defines the ComputeNode class and global compute node registry.

A ComputeNode represents any entity that can process tasks:
- LEO satellites (compute_satellites)
- MEO satellites (leader_satellites, which can also compute)
- Ground stations

All compute nodes are registered globally and can be queried by index.
"""

from collections import deque
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

# Global registry of all compute nodes
_COMPUTE_NODE_REGISTRY: Dict[int, 'ComputeNode'] = {}
_NEXT_NODE_ID = 0


def reset_compute_node_registry():
    """Clear the global compute node registry."""
    global _COMPUTE_NODE_REGISTRY, _NEXT_NODE_ID
    _COMPUTE_NODE_REGISTRY.clear()
    _NEXT_NODE_ID = 0


def register_compute_node(node: 'ComputeNode') -> int:
    """Register a compute node and return its global index."""
    global _NEXT_NODE_ID
    node_id = _NEXT_NODE_ID
    _COMPUTE_NODE_REGISTRY[node_id] = node
    _NEXT_NODE_ID += 1
    return node_id


def get_compute_node(node_id: int) -> Optional['ComputeNode']:
    """Retrieve a compute node by its global index."""
    return _COMPUTE_NODE_REGISTRY.get(node_id)


def get_all_compute_nodes() -> List['ComputeNode']:
    """Get all registered compute nodes in order."""
    return [_COMPUTE_NODE_REGISTRY[i] for i in sorted(_COMPUTE_NODE_REGISTRY.keys())]


class ComputeTask:
    """Represents a task assigned to a compute node."""
    
    def __init__(self, task_id: int, slice_id: int, flops: float, arrival_time: float):
        """
        Args:
            task_id: Original task ID
            slice_id: Slice index within the task
            flops: Computational requirement in FLOPs
            arrival_time: Time when this slice arrived at the compute node (in seconds)
        """
        self.task_id = task_id
        self.slice_id = slice_id
        self.flops = float(flops)
        self.arrival_time = float(arrival_time)
        self.completion_time: Optional[float] = None
    
    def __repr__(self) -> str:
        return f"ComputeTask(task_id={self.task_id}, slice_id={self.slice_id}, flops={self.flops:.2e})"


class ComputeNode:
    """
    Represents a compute node (LEO satellite, MEO satellite, or ground station).
    
    Each node has:
    - A unique global index (assigned by the registry)
    - Position in ECEF coordinates
    - Compute capacity (GFLOPS)
    - A task queue
    - Task completion tracking
    """
    
    def __init__(self, node_type: str, node_id: int, compute_gflops: float, 
                 position_ecef: np.ndarray = None, name: str = ""):
        """
        Args:
            node_type: "LEO", "MEO", or "GROUND_STATION"
            node_id: Original ID within its type
            compute_gflops: Compute capacity in GFLOPS
            position_ecef: Current position in ECEF coordinates (km)
            name: Human-readable name
        """
        self.node_type = node_type
        self.node_id = node_id
        self.name = name or f"{node_type}-{node_id}"
        self.compute_gflops = float(compute_gflops)
        self.position_ecef = position_ecef if position_ecef is not None else np.zeros(3)
        
        # Global index assigned by registry
        self.global_index = register_compute_node(self)
        
        # Task queue and processing state
        self.task_queue: deque[ComputeTask] = deque()
        self.total_queue_flops = 0.0  # Total FLOPs in queue
        self.last_update_time = 0.0  # Last time queue was updated
    
    def update_position(self, new_position_ecef: np.ndarray):
        """Update the node's position."""
        self.position_ecef = np.array(new_position_ecef, dtype=np.float32)
    
    def add_task_slice(self, task_id: int, slice_id: int, flops: float, arrival_time: float):
        """Add a task slice to the queue."""
        task = ComputeTask(task_id, slice_id, flops, arrival_time)
        self.task_queue.append(task)
        self.total_queue_flops += flops
    
    def update_queue(self, current_time: float) -> List[int]:
        """
        Update the queue by checking which tasks have completed.
        
        Returns:
            List of completed task IDs
        """
        time_elapsed = current_time - self.last_update_time
        if time_elapsed <= 0:
            return []
        
        # Compute capacity available in this time interval (in FLOPs)
        compute_capacity_flops = self.compute_gflops * 1e9 * time_elapsed
        remaining_capacity = compute_capacity_flops
        
        completed_task_ids = []
        
        while self.task_queue and remaining_capacity > 0:
            task = self.task_queue[0]
            
            if task.flops <= remaining_capacity:
                # Task completes
                remaining_capacity -= task.flops
                task.completion_time = self.last_update_time + (compute_capacity_flops - remaining_capacity) / (self.compute_gflops * 1e9)
                self.task_queue.popleft()
                self.total_queue_flops -= task.flops
                completed_task_ids.append(task.task_id)
            else:
                # Task is partially completed, but not finished
                task.flops -= remaining_capacity
                self.total_queue_flops -= remaining_capacity
                remaining_capacity = 0
        
        self.last_update_time = current_time
        return completed_task_ids
    
    def get_queue_load_flops(self) -> float:
        """Get the total FLOPs in the queue."""
        return self.total_queue_flops
    
    def get_queue_depth(self) -> int:
        """Get the number of tasks in the queue."""
        return len(self.task_queue)
    
    def clear_queue(self):
        """Clear the task queue."""
        self.task_queue.clear()
        self.total_queue_flops = 0.0
    
    def __repr__(self) -> str:
        return (f"ComputeNode(type={self.node_type}, id={self.node_id}, "
                f"global_idx={self.global_index}, queue_flops={self.total_queue_flops:.2e})")

