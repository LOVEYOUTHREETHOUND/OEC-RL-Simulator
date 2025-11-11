# -*- coding: utf-8 -*-
"""
Script for testing the environment's validity without a Jupyter Notebook.

This script runs a single episode using a baseline policy and provides both
text-based and graphical visualization of the outcome.
"""

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.environment.satellite_env import SatelliteEnv
from src.agents.baselines import GreedyPolicy

# Attempt to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def load_config(path):
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_test():
    """Runs the environment test."""
    print("--- Starting Environment Validity Test ---")

    # --- 1. Load Configurations ---
    print("\n[1/4] Loading configurations...")
    sim_config = load_config(os.path.join(project_root, 'configs/environment/simulation_v1.yaml'))
    sat_configs = load_config(os.path.join(project_root, 'configs/satellites.yaml'))

    # --- 2. Initialize Environment and Policy ---
    print("\n[2/4] Initializing environment and greedy policy...")
    env = SatelliteEnv(sim_config=sim_config, sat_configs=sat_configs)
    policy = GreedyPolicy(env)

    # --- 3. Run a Single Episode ---
    print("\n[3/4] Running a single episode with GreedyPolicy...")
    obs, info = env.reset()
    terminated = False
    
    # Store satellite trajectories for plotting
    # Shape: (num_steps, num_satellites, 3)
    trajectory = [obs['sat_positions']]
    
    # We'll run for a fixed number of steps to simulate a longer orbit
    num_test_steps = 100 
    for i in tqdm(range(num_test_steps), desc="Simulating Steps"):
        # The environment currently terminates after one step.
        # To see the trajectory, we manually call reset() if terminated.
        if terminated:
            obs, info = env.reset()

        # Get action from the policy
        action = policy.predict(obs)
        chosen_k = env.slice_strategies[action['slice_strategy']]
        assignment = action['assignment'][:chosen_k]

        print(f"\n{'='*10} Step {i+1} {'='*10}")
        print(f"> Policy chose to slice into {chosen_k} pieces.")
        print(f"> Assignment decision: {assignment}")

        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"> Environment responded with:")
        print(f"  - Total Latency: {info.get('total_latency', 'N/A'):.4f} seconds")
        print(f"  - Reward (VoI): {reward:.6f}")

        # Store the new positions for the trajectory plot
        trajectory.append(obs['sat_positions'])

    # --- 4. Visualize Trajectory ---
    print("\n[4/4] Generating trajectory visualization...")
    if not MATPLOTLIB_AVAILABLE:
        print("\n[WARNING] matplotlib is not installed. Skipping graphical visualization.")
        print("Please run 'pip install matplotlib' to enable plotting.")
        return

    trajectory = np.array(trajectory)
    num_sats = trajectory.shape[1]

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Earth's equator
    earth_circle = plt.Circle((0, 0), 6371, color='blue', alpha=0.3, label='Earth (Equatorial Projection)')
    ax.add_artist(earth_circle)

    # Plot satellite trajectories (XY projection)
    for i in range(num_sats):
        x_coords = trajectory[:, i, 0]
        y_coords = trajectory[:, i, 1]
        ax.plot(x_coords, y_coords, marker='.', markersize=2, label=f'Satellite {i+1}')
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=8, label=f'Sat {i+1} Start') # Start point
        ax.plot(x_coords[-1], y_coords[-1], 'rs', markersize=8, label=f'Sat {i+1} End') # End point

    ax.set_title('Satellite Trajectories (XY ECEF Projection)')
    ax.set_xlabel('X coordinate (km)')
    ax.set_ylabel('Y coordinate (km)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=num_sats)
    
    plt.show()
    print("\n--- Test script finished ---")

if __name__ == '__main__':
    run_test()

