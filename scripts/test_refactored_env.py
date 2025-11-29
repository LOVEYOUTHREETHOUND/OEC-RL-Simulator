# -*- coding: utf-8 -*-
"""
Test script for the refactored SatelliteEnv with single-source training.

This script tests:
1. Environment initialization
2. Reset and observation shape
3. Step execution with valid and invalid steps
4. Episode termination after max_tasks_per_episode
"""

import os
import sys
import argparse
import numpy as np

# Ensure project root on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config_loader import load_config
from src.environment.satellite_env import SatelliteEnv


def test_env_initialization():
    """Test environment initialization."""
    print("=" * 80)
    print("TEST 1: Environment Initialization")
    print("=" * 80)
    
    sim_config = load_config("configs/environment/simulation.yaml")
    raw_sat_configs = load_config("configs/satellites.yaml")

    # Inject TLEs into satellite configs (from cache or CelesTrak)
    from src.utils.tle_loader import preprocess_satellite_configs
    sat_configs = preprocess_satellite_configs(project_root, raw_sat_configs)
    
    env = SatelliteEnv(sim_config, sat_configs)
    
    print(f"✓ Environment created successfully")
    print(f"  - Source satellites: {env.num_source_satellites}")
    print(f"  - Compute satellites: {env.num_compute_satellites}")
    print(f"  - Leader satellites: {env.num_leader_satellites}")
    print(f"  - Ground stations: {env.num_ground_stations}")
    print(f"  - Max tasks per episode: {env.max_tasks_per_episode}")
    print(f"  - Num nearest compute to observe: {env.num_nearest_compute}")
    
    return env


def test_env_reset(env):
    """Test environment reset."""
    print("\n" + "=" * 80)
    print("TEST 2: Environment Reset")
    print("=" * 80)
    
    obs, info = env.reset(seed=42)
    
    print(f"✓ Environment reset successfully")
    print(f"  - Selected source satellite: {env.current_source_satellite.name}")
    print(f"  - Current task: {env.current_task}")
    print(f"  - Tasks in queue: {len(env.task_queue)}")
    
    # Check observation shape
    print(f"\n✓ Observation shapes:")
    for key, value in obs.items():
        print(f"  - {key}: {value.shape}")
    
    # Verify observation space
    for key, space in env.observation_space.spaces.items():
        assert obs[key].shape == space.shape, f"Shape mismatch for {key}"
    
    print(f"\n✓ All observation shapes match the observation space")
    
    return obs


def _fmt_vec(arr, precision=3):
    return np.array2string(np.array(arr, dtype=float), precision=precision, suppress_small=True)


def _print_observation(obs):
    print("  Observation snapshot:")
    # Task info unpack
    width, height, max_lat, data_bits, req_flops = obs["task_info"].tolist()
    print(f"    - task_origin_pos (ECEF km): {_fmt_vec(obs['task_origin_pos'])}")
    print(f"    - task_info: width={int(width)}, height={int(height)}, max_latency={max_lat:.2f}s, "
          f"data={data_bits/8e6:.2f} MB, flops={req_flops/1e9:.3f} GFLOPs")
    print(f"    - leader_pos (ECEF km): {_fmt_vec(obs['leader_pos'])}")
    # Compute nodes table
    compute_pos = obs["compute_pos"]
    compute_queues = obs["compute_queues"]
    print("    - nearest compute nodes (idx | queue [GFLOPs] | pos [km]):")
    for i in range(len(compute_queues)):
        q_gflops = compute_queues[i] / 1e9
        pos = compute_pos[i]
        print(f"      [{i:02d}] queue={q_gflops:8.3f} | pos={_fmt_vec(pos)}")
    print(f"    - ground_station_pos (ECEF km): {_fmt_vec(obs['ground_station_pos'])}")
    print(f"    - ground_station_queue [GFLOPs]: {obs['ground_station_queue'][0]/1e9:.3f}")


essential_keys = ["task_origin_pos","task_info","leader_pos","compute_pos","compute_queues","ground_station_pos","ground_station_queue"]


def _explain_action(env, action, info_before=None):
    # Strategy decode
    slice_idx, overlap_idx = action['slice_strategy']
    strategy = env.slicing_strategies[slice_idx]
    slice_size = strategy['slice_size']
    overlap = strategy['overlap_ratios'][overlap_idx]
    print("  Action decision:")
    print(f"    - slice_strategy -> idx=({slice_idx},{overlap_idx}) => slice_size={slice_size}, overlap={overlap}")
    # Assignment decode
    assign = action['assignment']
    n_dest = env.num_nearest_compute + 1
    counts = {i: int((assign == i).sum()) for i in range(n_dest)}
    print(f"    - assignment (per-destination slice counts; 0..{env.num_nearest_compute-1}=compute, {env.num_nearest_compute}=ground):")
    for i in range(n_dest):
        tag = ("compute" if i < env.num_nearest_compute else "ground")
        print(f"      dest {i:02d} ({tag}): {counts.get(i,0)} slices")
    # Preview first 20 mapping
    head = min(20, len(assign))
    print(f"    - assignment head[{head}]: {assign[:head]}")


def test_env_step(env, obs, steps_to_run: int = 5):
    """Run multiple steps with detailed observation and action printing."""
    print("\n" + "=" * 80)
    print("TEST 3: Environment Step Execution (multi-step)")
    print("=" * 80)

    step_idx = 0
    terminated = truncated = False
    while step_idx < steps_to_run and not (terminated or truncated):
        print(f"\n-- Step {step_idx+1} / {steps_to_run} --")
        # Print observation before action
        _print_observation(obs)
        # Sample action and explain
        action = env.action_space.sample()
        _explain_action(env, action)
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        # Outcome
        valid = info.get('is_valid_step', True)
        print("  Outcome:")
        print(f"    - valid_step: {valid}")
        if not valid:
            print(f"    - reason: {info.get('reason','N/A')}")
        print(f"    - reward: {reward:.6f}")
        print(f"    - latency: {info.get('total_latency', 0.0):.3f} s")
        print(f"    - K (slices): {info.get('calculated_k', 0)}")
        print(f"    - tasks_processed: {env.tasks_processed}  |  valid_steps: {env.valid_steps_taken}")

        # Detailed latency breakdown (from env debug)
        dbg = getattr(env, "_last_latency_debug", None)
        if dbg:
            print("  Latency breakdown:")
            print(f"    - task_id={dbg.get('task_id')}  k_slices={dbg.get('k_slices')}  max_latency={dbg.get('max_latency_sec'):.3f}s")
            sl = dbg.get('source_to_leader', {})
            if sl:
                cap = sl.get('isl_capacity_bps', 0.0)
                print(f"    - source→leader: dist={sl.get('distance_km', 0.0):.1f} km, ISL cap={cap/1e9:.3f} Gbps, t_initial={sl.get('t_initial_sec', 0.0):.3f}s")
            dests = dbg.get('destinations', [])
            for d in dests:
                idx = d.get('dest_index')
                nsl = d.get('num_slices')
                ntype = d.get('node_type', 'N/A')
                if 'error' in d:
                    print(f"      - dest[{idx}] ERROR: {d['error']}")
                    continue
                if ntype == 'GROUND_STATION':
                    cap = d.get('downlink_capacity_bps', 0.0)
                    print(f"      - dest[{idx}] GS: slices={nsl:4d}, dist={d.get('distance_km',0.0):7.1f} km, down cap={cap/1e9:.3f} Gbps, "
                          f"t_trans={d.get('t_trans_sec',0.0):.3f}s, t_comp={d.get('t_comp_sec',0.0):.3f}s, path={d.get('path_latency_sec',0.0):.3f}s")
                else:
                    cap = d.get('isl_capacity_bps', 0.0)
                    qbf = d.get('queue_load_flops_before', 0.0)
                    print(f"      - dest[{idx}] {ntype}: slices={nsl:4d}, dist={d.get('distance_km',0.0):7.1f} km, ISL cap={cap/1e9:.3f} Gbps, "
                          f"queue_before={qbf/1e9:.3f} GFLOPs, t_isl={d.get('t_isl_sec',0.0):.3f}s, t_queue={d.get('t_queue_sec',0.0):.3f}s, "
                          f"t_comp={d.get('t_comp_sec',0.0):.3f}s, path={d.get('path_latency_sec',0.0):.3f}s")
            print(f"    - total_latency={dbg.get('total_latency_sec',0.0):.3f}s  (constraint={dbg.get('max_latency_sec',0.0):.3f}s)  -> reward={dbg.get('reward',0.0):.6f}")

        step_idx += 1

    return obs, reward, terminated, truncated, info


def test_episode_run(env):
    """Test a full episode run."""
    print("\n" + "=" * 80)
    print("TEST 4: Full Episode Run")
    print("=" * 80)
    
    obs, info = env.reset(seed=42)
    
    episode_rewards = []
    valid_steps = 0
    invalid_steps = 0
    
    print(f"Starting episode with max_tasks_per_episode={env.max_tasks_per_episode}")
    print(f"Selected source satellite: {env.current_source_satellite.name}\n")
    
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get('is_valid_step', True):
            valid_steps += 1
            episode_rewards.append(reward)
            print(f"Step {valid_steps:3d}: Reward={reward:.4f}, Latency={info.get('total_latency', 0):.2f}s, "
                  f"K={info.get('calculated_k', 0)}, Tasks={env.tasks_processed}/{env.max_tasks_per_episode}")
        else:
            invalid_steps += 1
            print(f"  [INVALID] {info.get('reason', 'Unknown reason')}")
        
        if terminated or truncated:
            break
    
    print(f"\n✓ Episode completed")
    print(f"  - Valid steps: {valid_steps}")
    print(f"  - Invalid steps: {invalid_steps}")
    print(f"  - Tasks processed: {env.tasks_processed}")
    print(f"  - Average reward: {np.mean(episode_rewards):.4f}")
    print(f"  - Max reward: {np.max(episode_rewards):.4f}")
    print(f"  - Min reward: {np.min(episode_rewards):.4f}")


def main():
    parser = argparse.ArgumentParser("Test refactored SatelliteEnv")
    parser.add_argument("--sim_config", default=os.path.join("configs", "environment", "simulation.yaml"))
    parser.add_argument("--sats_config", default=os.path.join("configs", "satellites.yaml"))
    parser.add_argument("--full_episode", action="store_true", help="Run a full episode")
    parser.add_argument("--steps", type=int, default=5, help="Number of detailed steps to run in TEST 3")
    args = parser.parse_args()
    
    # Test 1: Initialization
    env = test_env_initialization()
    
    # Test 2: Reset
    obs = test_env_reset(env)
    
    # Test 3: Step (multi-step with detailed prints)
    obs, reward, terminated, truncated, info = test_env_step(env, obs, steps_to_run=args.steps)
    
    # Test 4: Full episode (optional)
    if args.full_episode:
        test_episode_run(env)
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()

