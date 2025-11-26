# -*- coding: utf-8 -*-
"""
This script is designed to test the fully refactored SatelliteEnv by interacting
with it using random actions in a multi-step episode.

Key Features:
- Initializes the new multi-tasking SatelliteEnv with a hex grid of ground stations.
- Runs a loop for a configurable number of steps within a single episode.
- In each step, it samples a random action from the expanded action space.
- Prints detailed information, including the new observation space and task queue status.
- Visualizes all satellites, UEs, and the grid of ground stations in a 3D plot.

Enhancement:
- Group-based TLE fetching: prefer matching by name from CelesTrak groups
  (starlink, galileo, active), with local caching to avoid frequent downloads.
"""

import os
import sys
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import re
import numpy as np

# Attempt to import requests for fetching TLE data
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.environment.satellite_env import SatelliteEnv
from src.utils.config_loader import load_config

# -----------------------------
# TLE utilities (ID-based)
# -----------------------------

def get_tle_data(norad_id: int, cache_dir: str, cache_duration_hours: int = 24) -> Optional[List[str]]:
    """Fetches TLE by NORAD ID, using a local cache to avoid redundant downloads."""
    cache_file = os.path.join(cache_dir, f"{norad_id}.tle")

    if os.path.exists(cache_file):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - file_mod_time) < timedelta(hours=cache_duration_hours):
            with open(cache_file, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]

    if not REQUESTS_AVAILABLE:
        print(f"\n[WARNING] 'requests' library not installed. Cannot fetch TLE for {norad_id}.")
        return None

    print(f"\n[INFO] TLE for {norad_id} not found/expired. Fetching from CelesTrak (CATNR)...")
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
    try:
        response = requests.get(url, timeout=20, verify=False)
        response.raise_for_status()
        tle_text = response.text.strip()

        if "No TLE found" in tle_text or not tle_text:
            return None

        tle_lines = [line.strip() for line in tle_text.splitlines() if line.strip()]
        # Save exactly what we got (can be 2 or 3 lines). We'll later pick the last two lines as L1/L2
        with open(cache_file, 'w') as f:
            f.write('\n'.join(tle_lines))
        return tle_lines

    except Exception:
        return None

# -----------------------------
# TLE utilities (Group-based by name)
# -----------------------------

_GROUP_URLS = {
    'starlink': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle',
    'galileo': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle',
    # Fallback big list for miscellaneous like GAOFEN
    'active': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
}

def _normalize_name(s: str) -> str:
    s = s.upper().strip()
    # Remove extra spaces and punctuation for robust matching
    return re.sub(r'[^A-Z0-9]', '', s)


def _fetch_group_text(group: str, cache_dir: str, cache_hours: int = 24) -> Optional[str]:
    if not REQUESTS_AVAILABLE:
        return None
    groups_dir = os.path.join(cache_dir, 'groups')
    os.makedirs(groups_dir, exist_ok=True)
    cache_file = os.path.join(groups_dir, f'{group}.tle')

    if os.path.exists(cache_file):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - file_mod_time) < timedelta(hours=cache_hours):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                pass

    url = _GROUP_URLS.get(group)
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=30, verify=False)
        resp.raise_for_status()
        text = resp.text
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)
        return text
    except Exception:
        return None


def _parse_group_tle(text: str) -> List[Tuple[str, str, str]]:
    """Parse a group TLE text into list of (name_line, line1, line2)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    triples: List[Tuple[str, str, str]] = []
    i = 0
    while i < len(lines) - 2:
        name_line = lines[i]
        l1 = lines[i + 1]
        l2 = lines[i + 2]
        if l1.startswith('1 ') and l2.startswith('2 '):
            triples.append((name_line, l1, l2))
            i += 3
        else:
            # Try to resync: find next '1 ' line
            i += 1
    return triples


def _match_tle_by_name(target_name: str, groups_texts: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """Try to find a TLE (l1,l2) whose name_line matches target_name, using multiple groups.
    Matching rules:
      - For STARLINK-XXXX: exact token containment after normalization
      - For Galileo: prefer GSAT\d{4} match; else 'GALILEO <num>' match
      - For GAOFEN: normalized containment match
    """
    norm_target = _normalize_name(target_name)

    # Extract tokens for better matches
    gsat_match = re.search(r'GSAT\d{4}', norm_target)
    gal_num_match = re.search(r'GALILEO(\d+)', norm_target)
    star_match = re.search(r'STARLINK-?(\d+)', target_name.upper())
    gaofen_match = re.search(r'GAOFEN-?\s*(\d+\s*\d*)', target_name.upper())

    for group, text in groups_texts.items():
        if not text:
            continue
        for name_line, l1, l2 in _parse_group_tle(text):
            norm_name_line = _normalize_name(name_line)
            ok = False
            # Priority: Starlink exact code
            if star_match and 'STARLINK' in norm_name_line:
                code = re.sub(r'\D', '', star_match.group(1))
                ok = _normalize_name(f'STARLINK{code}') in norm_name_line
            # GSAT id preferred for Galileo
            if not ok and gsat_match and 'GSAT' in norm_name_line:
                ok = gsat_match.group(0) in norm_name_line
            # Galileo number fallback
            if not ok and gal_num_match and 'GALILEO' in norm_name_line:
                ok = _normalize_name(f'GALILEO {gal_num_match.group(1)}') in norm_name_line
            # Gaofen generic containment
            if not ok and 'GAOFEN' in norm_target and 'GAOFEN' in norm_name_line:
                # Compare compact forms (remove non-alnum)
                ok = norm_target in norm_name_line or norm_name_line in norm_target
            # Generic fallback: normalized containment
            if not ok:
                ok = norm_target in norm_name_line

            if ok:
                return (l1, l2)
    return None


def get_tle_by_groups_and_name(name: str, cache_dir: str) -> Optional[Tuple[str, str]]:
    """Fetch TLE by searching CelesTrak groups using satellite name."""
    # Choose candidate groups based on name hints
    uname = name.upper()
    groups = []
    if 'STARLINK' in uname:
        groups.append('starlink')
    if 'GALILEO' in uname or 'GSAT' in uname:
        groups.append('galileo')
    if 'GAOFEN' in uname:
        groups.append('active')
    # Always consider 'active' as last resort for unknown types
    if 'active' not in groups:
        groups.append('active')

    texts: Dict[str, str] = {}
    for g in groups:
        texts[g] = _fetch_group_text(g, cache_dir) or ''

    return _match_tle_by_name(name, texts)


# -----------------------------
# Preprocess configs (inject TLE)
# -----------------------------

def preprocess_satellite_configs(sat_configs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Injects TLE data into the satellite configs.
    Order of attempts per satellite:
      1) Use cached file if present and fresh
      2) Try group-based lookup (by name) from starlink/galileo/active
      3) Fallback to ID-based fetch by CATNR
    """
    tle_cache_dir = os.path.join(project_root, 'data', 'tle')
    os.makedirs(tle_cache_dir, exist_ok=True)

    processed_configs = {'source_satellites': [], 'compute_satellites': [], 'leader_satellites': []}

    for group_name, sat_list in sat_configs.items():
        if group_name not in processed_configs:
            continue
        for sat_config in sat_list:
            sat_id = sat_config.get('sat_id')
            sat_name = sat_config.get('name', '')

            # 1) Use cached file if available
            cached = get_tle_data(sat_id, tle_cache_dir)
            if cached and len(cached) >= 2:
                sat_config['tle'] = (cached[-2], cached[-1])
                processed_configs[group_name].append(sat_config)
                continue

            # 2) Try group-based fetch by name and save to cache file
            tle_pair = get_tle_by_groups_and_name(sat_name, tle_cache_dir)
            if tle_pair:
                l1, l2 = tle_pair
                try:
                    with open(os.path.join(tle_cache_dir, f"{sat_id}.tle"), 'w') as f:
                        f.write(f"{l1}\n{l2}\n")
                except Exception:
                    pass
                sat_config['tle'] = (l1, l2)
                processed_configs[group_name].append(sat_config)
                continue

            # 3) Fallback to ID-based
            tle_data = get_tle_data(sat_id, tle_cache_dir)
            if tle_data and len(tle_data) >= 2:
                sat_config['tle'] = (tle_data[-2], tle_data[-1])
                processed_configs[group_name].append(sat_config)
            else:
                print(f"\n[WARNING] Skipping satellite {sat_id} ({sat_name}) as its TLE data could not be obtained.")

    return processed_configs


# -----------------------------
# Printing & Visualization
# -----------------------------

def print_observation(obs):
    """Prints the observation dictionary in a readable format."""
    print("\n--- Observation (State) ---")
    for key, value in obs.items():
        print(f"> {key}:")
        if isinstance(value, np.ndarray):
            print(np.array2string(value, prefix="  ", precision=2))
        else:
            print(f"  {value}")


def print_action(action):
    """Prints the action dictionary in a readable format."""
    print("\n--- Action (Randomly Sampled) ---")
    for key, value in action.items():
        print(f"> {key}:")
        if isinstance(value, np.ndarray):
            print(np.array2string(value, prefix="  ", max_line_width=120, threshold=1000))
        else:
            print(f"  {value}")


def print_task_dynamics(env, info):
    """Prints a detailed breakdown of task generation and queue status."""
    print("\n--- Task Dynamics ---")
    print(f"> Tasks in Queue ({len(env.task_queue)}): {list(env.task_queue)}")
    print(f"> Completed Tasks ({len(env.completed_tasks)}): {list(env.completed_tasks)}")

    for key, label in [
        ('newly_generated_rs_tasks', 'Newly Generated RS Tasks'),
        ('newly_generated_ue_tasks', 'Newly Generated UE Tasks (Pending)'),
        ('newly_uplinking_ue_tasks', 'Newly Uplinking UE Tasks'),
        ('newly_arrived_ue_tasks', 'Newly Arrived UE Tasks (in Queue)')
    ]:
        tasks = info.get(key, [])
        if tasks:
            print(f"> {label}: {tasks}")


def visualize_and_save_constellation(env, fig, ax, filepath):
    """Renders the current state of the constellation, including UEs and Ground Stations, and saves it."""
    ax.cla()
    ax.scatter([0], [0], [0], s=200, c='blue', marker='o', label='Earth Center')

    # Plot Ground Stations
    if env.ground_stations:
        gs_positions = np.array([gs.position_ecef for gs in env.ground_stations])
        ax.scatter(gs_positions[:, 0], gs_positions[:, 1], gs_positions[:, 2], s=15, c='black', marker='.', label='Ground Stations')

    # Plot source satellites
    source_positions = np.array([sat.position_ecef for sat in env.source_satellites])
    valid_source_pos = source_positions[~np.isnan(source_positions).any(axis=1)]
    if valid_source_pos.size > 0:
        ax.scatter(valid_source_pos[:, 0], valid_source_pos[:, 1], valid_source_pos[:, 2], s=80, c='green', marker='s', label='Source Satellites')

    # Plot compute satellites
    compute_positions = np.array([sat.position_ecef for sat in env.compute_satellites])
    valid_compute_pos = compute_positions[~np.isnan(compute_positions).any(axis=1)]
    if valid_compute_pos.size > 0:
        ax.scatter(valid_compute_pos[:, 0], valid_compute_pos[:, 1], valid_compute_pos[:, 2], s=50, c='red', marker='o', label='Compute Satellites')

    # Plot MEO leader satellites (if present)
    valid_leader_pos = np.empty((0, 3))
    if hasattr(env, 'leader_satellites') and env.leader_satellites:
        leader_positions = np.array([sat.position_ecef for sat in env.leader_satellites])
        valid_leader_pos = leader_positions[~np.isnan(leader_positions).any(axis=1)]
        if valid_leader_pos.size > 0:
            ax.scatter(valid_leader_pos[:, 0], valid_leader_pos[:, 1], valid_leader_pos[:, 2], s=70, c='orange', marker='^', label='MEO Leaders')

    # Plot User Equipments (UEs)
    if env.ground_ues:
        ue_positions = np.array([ue.position_ecef for ue in env.ground_ues])
        ax.scatter(ue_positions[:, 0], ue_positions[:, 1], ue_positions[:, 2], s=20, c='purple', marker='x', label='UEs')

    all_valid_positions = []
    if env.ground_stations: all_valid_positions.extend([gs.position_ecef for gs in env.ground_stations])
    if valid_source_pos.size > 0: all_valid_positions.extend(valid_source_pos)
    if valid_compute_pos.size > 0: all_valid_positions.extend(valid_compute_pos)
    if valid_leader_pos.size > 0: all_valid_positions.extend(valid_leader_pos)

    if not all_valid_positions:
        return

    max_range = np.max(np.abs(np.vstack(all_valid_positions))) * 1.2
    ax.set_xlim([-max_range, max_range]); ax.set_ylim([-max_range, max_range]); ax.set_zlim([-max_range, max_range])
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
    ax.set_title(f"State at Time: {env.orbit_propagator.simulation_time}")
    ax.legend()
    fig.savefig(filepath)
    print(f"\n[+] Saved visualization to {filepath}")


# -----------------------------
# Main
# -----------------------------

def main():
    """Main function to run the environment test."""
    print("Loading configurations...")
    sim_config = load_config('configs/environment/simulation.yaml')
    sat_configs_raw = load_config('configs/satellites.yaml')

    print("Preprocessing satellite configs to load or fetch TLE data (group-based)...")
    sat_configs = preprocess_satellite_configs(sat_configs_raw)

    if not sat_configs['source_satellites'] or not sat_configs['compute_satellites']:
        print("\n[ERROR] Not enough satellites with valid TLE data. Aborting.")
        return

    print("TLE data successfully prepared for all available satellites.")

    print("Initializing Satellite Environment...")
    env = SatelliteEnv(sim_config=sim_config, sat_configs=sat_configs)
    print(f"Initialized {env.num_ground_stations} ground stations in a hex grid.")

    output_dir = os.path.join(project_root, 'results', 'plots', 'interaction_test')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualization images to: {os.path.abspath(output_dir)}")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    print("\n========================================")
    print("      Environment Interaction Test      ")
    print("========================================")

    obs, info = env.reset()
    filepath = os.path.join(output_dir, "step_000_initial.png")
    visualize_and_save_constellation(env, fig, ax, filepath)

    for step in range(1, env.max_episode_steps + 1):
        print(f"\n{'='*20} Step {step}/{env.max_episode_steps} {'='*20}")
        env.render(mode='human')

        if env.current_task is None:
            print("\n--- No task to process. Agent is idle. ---")
        else:
            print_observation(obs)

        action = env.action_space.sample()
        print_action(action)

        next_obs, reward, terminated, truncated, info = env.step(action)

        print("\n--- Environment Response ---")
        print(f"> Reward: {reward:.4f}")
        print(f"> Info: {info.get('info', '')}")
        print(f"> Terminated: {terminated}, Truncated: {truncated}")

        print_task_dynamics(env, info)

        obs = next_obs

        filepath = os.path.join(output_dir, f"step_{step:03d}.png")
        visualize_and_save_constellation(env, fig, ax, filepath)

        if terminated or truncated:
            print("\n--- Episode Finished. ---")
            break

    print(f"\nTest finished after {env.steps_taken} steps. Images saved in {output_dir}.")
    plt.close(fig)


if __name__ == '__main__':
    main()
