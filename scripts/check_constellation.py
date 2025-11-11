# -*- coding: utf-8 -*-
"""
A script to calculate and visualize the full orbits of the satellite constellation.
This version plots the 3D trajectory over one orbital period.
"""

import os
import sys
import yaml
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm

# Attempt to import requests for fetching TLE data
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.environment.components.satellite import Satellite
from src.physics.orbits import OrbitPropagator
from src.physics.constants import EARTH_RADIUS_KM

# Attempt to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def load_config(path):
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_tle_data(norad_id, cache_dir, cache_duration_hours=24):
    """Fetches TLE data, using a local cache to avoid redundant downloads."""
    cache_file = os.path.join(cache_dir, f"{norad_id}.tle")

    # 1. Check for a valid cache file
    if os.path.exists(cache_file):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - file_mod_time) < timedelta(hours=cache_duration_hours):
            with open(cache_file, 'r') as f:
                return [line.strip() for line in f.readlines()]

    # 2. If cache is invalid or missing, fetch from CelesTrak
    if not REQUESTS_AVAILABLE:
        print("\n[WARNING] 'requests' library not installed. Cannot fetch new TLE data.")
        return None

    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        tle_text = response.text.strip()

        if "No TLE found" in tle_text or not tle_text:
            print(f"\n[WARNING] No TLE found for NORAD ID {norad_id}.")
            return None

        tle_lines = tle_text.splitlines()
        # The name is the first line, TLE is the next two. We only need the TLE.
        tle_to_cache = []
        if len(tle_lines) >= 3:
            tle_to_cache = tle_lines[1:]
        elif len(tle_lines) == 2:
            tle_to_cache = tle_lines
        else:
            print(f"\n[WARNING] Incomplete TLE data for NORAD ID {norad_id}: {tle_text}")
            return None
        
        # 3. Save the new TLE data to the cache
        with open(cache_file, 'w') as f:
            f.write('\n'.join(tle_to_cache))
        
        return tle_to_cache

    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Failed to fetch TLE for NORAD ID {norad_id}: {e}")
        return None

def check_orbits():
    """Calculates and visualizes the full orbits of the constellation."""
    print("--- Starting Constellation Orbit Visualization --- ")

    # --- 1. Load Configurations ---
    print("\n[1/5] Loading configurations...")
    try:
        sat_configs = load_config(os.path.join(project_root, 'configs/satellites.yaml'))
        sim_config = load_config(os.path.join(project_root, 'configs/environment/simulation_v1.yaml'))
        start_time_str = sim_config['start_time']
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        source_configs = sat_configs.get('source_satellites', [])
        compute_configs = sat_configs.get('compute_satellites', [])
        print(f"Found {len(source_configs)} source and {len(compute_configs)} compute satellite configurations.")
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        return

    # --- 2. Fetch TLE Data ---
    print("\n[2/5] Fetching or loading TLE data...")
    tle_cache_dir = os.path.join(project_root, 'data', 'tle')
    os.makedirs(tle_cache_dir, exist_ok=True)

    all_configs = source_configs + compute_configs
    successful_configs = []
    for cfg in tqdm(all_configs, desc="Getting TLEs"):
        tle_data = get_tle_data(cfg['sat_id'], tle_cache_dir)
        if tle_data:
            cfg['tle'] = tle_data
            successful_configs.append(cfg)
        else:
            print(f"Could not process satellite {cfg['sat_id']}. It will be excluded.")

    # Filter out satellites for which TLE data could not be obtained
    source_configs = [cfg for cfg in source_configs if cfg in successful_configs]
    compute_configs = [cfg for cfg in compute_configs if cfg in successful_configs]

    print("Finished getting TLE data.")

    # --- 3. Create Satellite Objects ---
    print("\n[3/5] Creating Satellite objects...")
    source_satellites = [Satellite(sat_id=cfg['sat_id'], config=cfg) for cfg in source_configs]
    compute_satellites = [Satellite(sat_id=cfg['sat_id'], config=cfg) for cfg in compute_configs]
    all_satellites = source_satellites + compute_satellites
    if not all_satellites:
        print("\n[ERROR] No satellites could be processed. Aborting.")
        return
    print("Successfully created Satellite objects.")

    # --- 3. Calculate Full Orbits ---
    print("\n[4/5] Calculating full orbits for all satellites...")
    orbit_duration_minutes = 100 # Approx. one LEO period
    time_step_seconds = 60
    num_steps = int((orbit_duration_minutes * 60) / time_step_seconds)
    
    trajectories = {sat.id: [] for sat in all_satellites}
    propagator = OrbitPropagator(start_time)

    for i in tqdm(range(num_steps), desc="Propagating Orbits"):
        time_offset = i * time_step_seconds
        positions = propagator.get_positions_ecef(all_satellites, time_offset_seconds=time_offset)
        for sat, pos in zip(all_satellites, positions):
            trajectories[sat.id].append(pos)
    
    print("Successfully calculated orbit trajectories.")

    # --- 4. Visualize 3D Orbits ---
    print("\n[5/5] Generating 3D orbit plot...")
    if not MATPLOTLIB_AVAILABLE:
        print("\n[WARNING] matplotlib not installed. Skipping visualization.")
        return

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = EARTH_RADIUS_KM * np.cos(u) * np.sin(v)
    y = EARTH_RADIUS_KM * np.sin(u) * np.sin(v)
    z = EARTH_RADIUS_KM * np.cos(v)
    ax.plot_surface(x, y, z, color='royalblue', alpha=0.3)

    # Plot source satellite orbits
    for sat in source_satellites:
        traj = np.array(trajectories[sat.id])
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='yellow', label=f'Source: {sat.name}')
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='yellow', marker='*', s=200, edgecolor='black') # Start point

    # Plot compute satellite orbits
    # Use a color cycle for better visibility
    colors = plt.cm.cool(np.linspace(0, 1, len(compute_satellites)))
    for i, sat in enumerate(compute_satellites):
        traj = np.array(trajectories[sat.id])
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[i], label=f'Compute: {sat.name}')
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=colors[i], marker='o', s=50, edgecolor='black') # Start point

    ax.set_title(f'Constellation Orbits for ~100 minutes from {start_time_str}')
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
    max_range = EARTH_RADIUS_KM * 1.5
    ax.set_xlim([-max_range, max_range]); ax.set_ylim([-max_range, max_range]); ax.set_zlim([-max_range, max_range])
    
    # Create a single legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(1.15, 1.0))

    output_dir = os.path.join(project_root, 'results', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'constellation_orbits_3d.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"\n[SUCCESS] 3D orbit plot saved to: {output_path}")
    print("\n--- Orbit check finished ---")

if __name__ == '__main__':
    check_orbits()
