# -*- coding: utf-8 -*-
"""
Collect trajectories from a trained A2C model for Decision Transformer training.

Usage:
    python scripts/collect_trajectories_for_dt.py \
        --model_path results/models/a2c/a2c_20251202-235950/final_model.zip \
        --num_episodes 10000 \
        --output_path data/dt_trajectories.pkl
"""
import os
import sys
import argparse
import pickle
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from stable_baselines3 import A2C
from src.training.utils import make_vec_envs


def compute_returns_to_go(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Compute returns-to-go (RTG) from a reward sequence.
    
    RTG[t] = r[t] + gamma * r[t+1] + gamma^2 * r[t+2] + ...
    
    Args:
        rewards: List of rewards [r_0, r_1, ..., r_T]
        gamma: Discount factor
    
    Returns:
        List of returns-to-go [RTG_0, RTG_1, ..., RTG_T]
    """
    returns_to_go = []
    running_return = 0.0
    
    # Compute from back to front
    for r in reversed(rewards):
        running_return = r + gamma * running_return
        returns_to_go.insert(0, running_return)
    
    return returns_to_go


def flatten_dict_observation(obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Flatten a dict observation from VecEnv to a single-env format.
    
    Args:
        obs_dict: Dict observation from VecEnv with shape (1, ...)
    
    Returns:
        Flattened dict with shape (...)
    """
    return {k: v[0] for k, v in obs_dict.items()}


def collect_trajectories(
    model_path: str,
    num_episodes: int,
    sim_config: str = "configs/environment/simulation.yaml",
    sats_config: str = "configs/satellites.yaml",
    seed: int = 42,
    deterministic: bool = False,
    gamma: float = 0.99,
    n_parallel_envs: int = 4,
    save_freq: int = 1000,
    output_path: str = None,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    """
    Collect trajectories by running a trained A2C model.
    
    Args:
        model_path: Path to the trained A2C model (.zip file)
        num_episodes: Number of episodes to collect
        sim_config: Path to simulation config
        sats_config: Path to satellites config
        seed: Random seed
        deterministic: Whether to use deterministic policy
        gamma: Discount factor for RTG computation
        n_parallel_envs: Number of parallel environments (default: 4)
        save_freq: Save frequency (episodes per batch file)
        output_path: Output file path
        resume: Whether to resume from existing batch files
    
    Returns:
        List of trajectory dictionaries
    """
    # Check for existing batch files if resume is enabled
    start_episode = 0
    existing_trajectories = []
    
    if resume and output_path and save_freq > 0:
        base_dir = os.path.dirname(output_path) or '.'
        base_name = os.path.basename(output_path).replace('.pkl', '')
        
        # Find existing batch files
        import glob
        pattern = os.path.join(base_dir, f'{base_name}_batch_*.pkl')
        existing_batches = sorted(glob.glob(pattern))
        
        if existing_batches:
            print(f"\n{'='*60}")
            print(f"RESUME MODE: Found {len(existing_batches)} existing batch files")
            print(f"{'='*60}")
            
            for batch_file in existing_batches:
                try:
                    with open(batch_file, 'rb') as f:
                        batch = pickle.load(f)
                    existing_trajectories.extend(batch)
                    print(f"  ✓ Loaded {os.path.basename(batch_file)}: {len(batch)} trajectories")
                except Exception as e:
                    print(f"  ❌ Failed to load {batch_file}: {e}")
            
            start_episode = len(existing_trajectories)
            print(f"\nResuming from episode {start_episode}")
            print(f"Remaining to collect: {num_episodes - start_episode}")
            print(f"{'='*60}\n")
            
            if start_episode >= num_episodes:
                print(f"✓ Already collected {start_episode} episodes (target: {num_episodes})")
                print(f"No additional collection needed.")
                return existing_trajectories[:num_episodes]
    
    print(f"Loading model from {model_path}...")
    model = A2C.load(model_path)
    
    import time
    print(f"Creating {n_parallel_envs} parallel environments...")
    print("[collect_dt] NOTE: On Windows, SubprocVecEnv startup can be slow (TLE/config init per process).")
    print("[collect_dt] If this seems stuck for several minutes, try smaller --n_parallel (e.g., 4/8/12).")

    _t0 = time.time()
    env = make_vec_envs(
        sim_config_path=sim_config,
        sats_config_path=sats_config,
        n_envs=n_parallel_envs,  # Use multiple parallel environments
        seed=seed,
        use_subproc=True,  # Use subprocess for true parallelism
        use_action_wrapper=True,
        monitor_log_dir=None,
    )
    _t1 = time.time()
    print(f"[collect_dt] Environments created in {_t1 - _t0:.1f}s")
    
    # Initialize with existing trajectories if resuming
    trajectories = existing_trajectories.copy() if existing_trajectories else []
    episode_returns = [sum(t['rewards']) for t in existing_trajectories] if existing_trajectories else []
    
    # Track ongoing trajectories for each parallel environment
    current_trajectories = [
        {
            'observations': [],
            'actions': [],
            'rewards': [],
            'timesteps': [],
            # Extra per-step info for correct DT masking / env-alignment
            'calculated_ks': [],
            'num_destinations': None,
            'max_k': None,
            'num_slice_sizes': None,
        }
        for _ in range(n_parallel_envs)
    ]
    current_timesteps = [0] * n_parallel_envs
    
    # Adjust collection target if resuming
    episodes_to_collect = num_episodes - start_episode
    print(f"Collecting {episodes_to_collect} episodes (total target: {num_episodes})...")
    obs = env.reset()
    episodes_collected = start_episode
    
    with tqdm(total=num_episodes) as pbar:
        while episodes_collected < num_episodes:
            # Record observations for all environments
            for env_idx in range(n_parallel_envs):
                if episodes_collected + env_idx < num_episodes:
                    obs_flat = {k: v[env_idx] for k, v in obs.items()}
                    current_trajectories[env_idx]['observations'].append({
                        k: v.copy() for k, v in obs_flat.items()
                    })
                    current_trajectories[env_idx]['timesteps'].append(current_timesteps[env_idx])
            
            # Predict actions for all environments
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Record actions
            for env_idx in range(n_parallel_envs):
                if episodes_collected + env_idx < num_episodes:
                    current_trajectories[env_idx]['actions'].append(action[env_idx].copy())
            
            # Step all environments
            obs, reward, done, info = env.step(action)

            # Record info needed for DT masking and action-space alignment
            # VecEnv returns list[dict] infos
            if info is not None:
                for env_idx in range(n_parallel_envs):
                    if episodes_collected + env_idx < num_episodes:
                        try:
                            info_i = info[env_idx]
                        except Exception:
                            info_i = None
                        if isinstance(info_i, dict):
                            # per-step calculated k
                            ck = info_i.get('calculated_k', None)
                            if ck is not None:
                                current_trajectories[env_idx]['calculated_ks'].append(int(ck))
                            else:
                                current_trajectories[env_idx]['calculated_ks'].append(0)

                            # set per-episode static config (first time only)
                            if current_trajectories[env_idx].get('max_k', None) is None:
                                try:
                                    # max_k equals len of env action assignment vector in flattened action
                                    if isinstance(action, np.ndarray):
                                        # action[env_idx] = [slice_strategy_dims + max_k]
                                        a0 = action[env_idx]
                                        if hasattr(a0, 'shape'):
                                            # slice dims is 1 in this env
                                            current_trajectories[env_idx]['max_k'] = int(len(a0) - 1)
                                except Exception:
                                    pass
                            if current_trajectories[env_idx].get('num_destinations', None) is None:
                                # try infer from info assignment values
                                try:
                                    assign_list = info_i.get('assignment', None)
                                    if isinstance(assign_list, list) and len(assign_list) > 0:
                                        current_trajectories[env_idx]['num_destinations'] = int(max(assign_list)) + 1
                                except Exception:
                                    pass
                            if current_trajectories[env_idx].get('num_slice_sizes', None) is None:
                                # best-effort: infer from flattened action space first dim by reading obs? not available here
                                # will be inferred during training if missing
                                current_trajectories[env_idx]['num_slice_sizes'] = None
            
            # Process results for each environment
            for env_idx in range(n_parallel_envs):
                if episodes_collected >= num_episodes:
                    break
                    
                current_trajectories[env_idx]['rewards'].append(float(reward[env_idx]))
                current_timesteps[env_idx] += 1
                
                # Check if episode is done
                if done[env_idx]:
                    # Compute returns-to-go
                    current_trajectories[env_idx]['returns_to_go'] = compute_returns_to_go(
                        current_trajectories[env_idx]['rewards'], gamma=gamma
                    )
                    
                    # Store trajectory
                    episode_return = sum(current_trajectories[env_idx]['rewards'])
                    episode_returns.append(episode_return)
                    trajectories.append(current_trajectories[env_idx])
                    
                    # Reset for next episode
                    current_trajectories[env_idx] = {
                        'observations': [],
                        'actions': [],
                        'rewards': [],
                        'timesteps': [],
                        'calculated_ks': [],
                        'num_destinations': None,
                        'max_k': None,
                        'num_slice_sizes': None,
                    }
                    current_timesteps[env_idx] = 0
                    
                    episodes_collected += 1
                    pbar.update(1)
                    
                    # Update progress bar stats
                    if len(episode_returns) >= 100:
                        recent_returns = episode_returns[-100:]
                        pbar.set_postfix({
                            'avg_return': f"{np.mean(recent_returns):.2f}",
                            'std_return': f"{np.std(recent_returns):.2f}",
                        })
                    
                    # Periodic batch saving - save every save_freq episodes
                    if save_freq > 0 and episodes_collected % save_freq == 0 and output_path:
                        batch_num = episodes_collected // save_freq
                        batch_start = (batch_num - 1) * save_freq
                        batch_end = episodes_collected
                        
                        # Extract trajectories for this batch
                        batch_trajectories = trajectories[batch_start:batch_end]
                        
                        # Generate batch filename
                        base_dir = os.path.dirname(output_path)
                        base_name = os.path.basename(output_path).replace('.pkl', '')
                        batch_path = os.path.join(base_dir, f'{base_name}_batch_{batch_num:03d}.pkl')
                        
                        try:
                            os.makedirs(base_dir, exist_ok=True)
                            with open(batch_path, 'wb') as f:
                                pickle.dump(batch_trajectories, f)
                            print(f"\n✓ Saved batch {batch_num}: {batch_path} ({len(batch_trajectories)} trajectories)")
                        except Exception as e:
                            print(f"\n⚠️  Failed to save batch {batch_num}: {e}")
    
    env.close()
    
    # Save any remaining trajectories as final batch
    if save_freq > 0 and output_path and len(trajectories) % save_freq != 0:
        batch_num = (len(trajectories) // save_freq) + 1
        batch_start = (batch_num - 1) * save_freq
        batch_trajectories = trajectories[batch_start:]
        
        base_dir = os.path.dirname(output_path)
        base_name = os.path.basename(output_path).replace('.pkl', '')
        batch_path = os.path.join(base_dir, f'{base_name}_batch_{batch_num:03d}.pkl')
        
        try:
            with open(batch_path, 'wb') as f:
                pickle.dump(batch_trajectories, f)
            print(f"\n✓ Saved final batch {batch_num}: {batch_path} ({len(batch_trajectories)} trajectories)")
        except Exception as e:
            print(f"\n⚠️  Failed to save final batch: {e}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(f"Total episodes: {len(trajectories)}")
    print(f"Total steps: {sum(len(t['rewards']) for t in trajectories)}")
    print(f"Average episode length: {np.mean([len(t['rewards']) for t in trajectories]):.1f}")
    print(f"Return statistics:")
    print(f"  Mean: {np.mean(episode_returns):.2f}")
    print(f"  Std:  {np.std(episode_returns):.2f}")
    print(f"  Min:  {np.min(episode_returns):.2f}")
    print(f"  Max:  {np.max(episode_returns):.2f}")
    print(f"  Median: {np.median(episode_returns):.2f}")
    print("="*60)
    
    return trajectories


def save_trajectories(trajectories: List[Dict], output_path: str):
    """Save trajectories to a pickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"\nSaved {len(trajectories)} trajectories to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Collect trajectories for Decision Transformer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained A2C model")
    parser.add_argument("--num_episodes", type=int, default=10000, help="Number of episodes to collect")
    parser.add_argument("--output_path", type=str, default="data/dt_trajectories.pkl", help="Output file path")
    parser.add_argument("--sim_config", type=str, default="configs/environment/simulation.yaml")
    parser.add_argument("--sats_config", type=str, default="configs/satellites.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for RTG")
    parser.add_argument("--n_parallel", type=int, default=8, help="Number of parallel environments (default: 8)")
    parser.add_argument("--save_freq", type=int, default=1000, help="Save checkpoint every N episodes (default: 1000)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing batch files")
    
    args = parser.parse_args()
    
    # Collect trajectories
    trajectories = collect_trajectories(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        sim_config=args.sim_config,
        sats_config=args.sats_config,
        seed=args.seed,
        deterministic=args.deterministic,
        gamma=args.gamma,
        n_parallel_envs=args.n_parallel,
        save_freq=args.save_freq,
        output_path=args.output_path,
        resume=args.resume,
    )
    
    # Save to disk
    save_trajectories(trajectories, args.output_path)


if __name__ == "__main__":
    main()

