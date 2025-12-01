# -*- coding: utf-8 -*-
"""
Test baseline scheduling policies on the SatelliteEnv.

This script evaluates three baseline strategies with different slice and assignment choices:

1. Random Policy:
   - Slice Strategy: Randomly select from available slicing strategies
   - Assignment: Randomly assign to compute nodes
   
2. Greedy Policy:
   - Slice Strategy: Randomly select from available slicing strategies
   - Assignment: Always assign to the least busy (smallest queue) nodes
   
3. Round-Robin Policy:
   - Slice Strategy: Randomly select from available slicing strategies
   - Assignment: Cycle through compute nodes in order for load balancing

Each baseline is tested under the same conditions as training:
- Same environment configuration
- 100 steps per episode (100 task decisions)
- Same task arrival pattern (Poisson distribution)
- Results logged to TensorBoard in the same directory structure
"""
from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, Any, Tuple

import numpy as np
from tensorboardX import SummaryWriter

# Ensure project root on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training.utils import (
    make_vec_envs,
    prepare_run_dirs,
    save_run_metadata,
    _build_single_env,
)
from src.utils.config_loader import load_config


class BaselinePolicy:
    """Base class for baseline scheduling policies."""
    
    def __init__(self, num_compute_nodes: int, num_slice_strategies: int, max_k: int):
        """
        Args:
            num_compute_nodes: Total number of compute nodes (including ground station)
            num_slice_strategies: Number of available slice strategies
            max_k: Maximum number of slices (k value)
        """
        self.num_compute_nodes = num_compute_nodes
        self.num_slice_strategies = num_slice_strategies
        self.max_k = max_k
        self.step_count = 0
    
    def get_action(self, obs: Dict[str, np.ndarray], k: int) -> Dict[str, np.ndarray]:  # noqa: F841
        """
        Generate action (slice_strategy, assignment) based on observation.
        
        Args:
            obs: observation dict from environment
            k: number of compute nodes to assign to
            
        Returns:
            action dict with 'slice_strategy' and 'assignment'
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset policy state."""
        self.step_count = 0


class RandomPolicy(BaselinePolicy):
    """
    Random Policy: Randomly assign tasks to compute nodes.
    
    Rationale: Serves as a baseline to show the benefit of intelligent assignment strategies.
    - Slice Strategy: Randomly select from available slicing strategies (like A2C)
    - Assignment: Randomly assign to compute nodes
    """
    
    def get_action(self, obs: Dict[str, np.ndarray], k: int) -> Dict[str, np.ndarray]:
        """Randomly select slice strategy and compute nodes."""
        # Randomly select a slice strategy from available options (same as A2C)
        slice_strategy = np.array([np.random.randint(0, self.num_slice_strategies)], dtype=np.int32)
        
        # Randomly permute and select first k nodes
        assignment = np.random.permutation(self.num_compute_nodes)[:k].astype(np.int32)
        
        # Pad to max_k
        if len(assignment) < self.max_k:
            assignment = np.pad(assignment, (0, self.max_k - len(assignment)), 
                              mode='constant', constant_values=0)
        
        self.step_count += 1
        return {
            'slice_strategy': slice_strategy,
            'assignment': assignment
        }


class GreedyPolicy(BaselinePolicy):
    """
    Greedy Policy: Maximize parallelism with smallest slices, assign to least busy nodes.
    
    Rationale:
    - Slice Strategy: Use smallest slice size (index 0, typically 128 pixels) to maximize
      the number of slices (k), enabling maximum parallelism across compute nodes.
    - Assignment: Always select the k nodes with smallest queues to minimize latency.
    
    This is a reasonable heuristic that balances:
    - Fine-grained parallelism (many small slices)
    - Load balancing (assign to idle nodes)
    """
    
    def get_action(self, obs: Dict[str, np.ndarray], k: int) -> Dict[str, np.ndarray]:
        """Assign to k nodes with smallest queues using smallest slice size."""
        # Slice strategy: randomly select from available strategies
        slice_strategy = np.array([np.random.randint(0, self.num_slice_strategies)], dtype=np.int32)
        
        # Get compute node queues from observation (shape: num_nearest_compute,)
        compute_queues = obs['compute_queues']
        
        # Preferred order: nodes with smallest queues first
        preferred_order = np.argsort(compute_queues).astype(np.int32)
        
        # Build assignment of length max_k by repeating the preferred order
        repeats = int(np.ceil(self.max_k / max(len(preferred_order), 1)))
        assignment = np.tile(preferred_order, repeats)[:self.max_k]
        
        self.step_count += 1
        return {
            'slice_strategy': slice_strategy,
            'assignment': assignment
        }


class RoundRobinPolicy(BaselinePolicy):
    """
    Round-Robin Policy: Use medium slice size, distribute tasks evenly across nodes.
    
    Rationale:
    - Slice Strategy: Use medium slice size (index 2, typically 512 pixels) for a balance
      between parallelism and per-slice overhead.
    - Assignment: Cycle through compute nodes in order, ensuring each node receives
      roughly equal load over time.
    
    This is a classic load-balancing strategy that works well when:
    - Node capacities are similar
    - We want to avoid hotspots
    - Fairness across nodes is important
    """
    
    def __init__(self, num_compute_nodes: int, num_slice_strategies: int, max_k: int):
        super().__init__(num_compute_nodes, num_slice_strategies, max_k)
        self.current_index = 0
    
    def get_action(self, obs: Dict[str, np.ndarray], k: int) -> Dict[str, np.ndarray]:
        """Assign to k nodes in round-robin fashion using medium slice size."""
        # Slice strategy: randomly select from available strategies
        slice_strategy = np.array([np.random.randint(0, self.num_slice_strategies)], dtype=np.int32)
        
        # Generate round-robin assignment
        assignment = []
        for i in range(k):
            assignment.append((self.current_index + i) % self.num_compute_nodes)
        
        assignment = np.array(assignment, dtype=np.int32)
        
        # Update current index for next call
        self.current_index = (self.current_index + k) % self.num_compute_nodes
        
        # Pad to max_k
        if len(assignment) < self.max_k:
            assignment = np.pad(assignment, (0, self.max_k - len(assignment)), 
                              mode='constant', constant_values=0)
        
        self.step_count += 1
        return {
            'slice_strategy': slice_strategy,
            'assignment': assignment
        }
    
    def reset(self):
        """Reset policy state."""
        super().reset()
        self.current_index = 0


def parse_args():
    p = argparse.ArgumentParser("Test baseline scheduling policies on SatelliteEnv")
    p.add_argument("--sim_config", default=os.path.join("configs", "environment", "simulation.yaml"))
    p.add_argument("--sats_config", default=os.path.join("configs", "satellites.yaml"))
    p.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to test per baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default=None, help="Run name for baselines")
    p.add_argument("--override_max_tasks_per_episode", type=int, default=100, help="Tasks per episode (default 100)")
    return p.parse_args()


def run_baseline_evaluation(
    policy: BaselinePolicy,
    policy_name: str,
    env,  # single Gymnasium env (not VecEnv)
    num_episodes: int,
    writer: SummaryWriter,
    sim_config: Dict[str, Any],
    global_step: int = 0,
) -> Tuple[float, float, float, float]:
    """
    Run baseline policy evaluation.
    
    Args:
        policy: BaselinePolicy instance
        policy_name: name of policy for logging
        env: vectorized environment
        num_episodes: number of episodes to run
        writer: TensorBoard writer
        sim_config: simulation configuration (for extracting k)
        global_step: starting step for TensorBoard
        
    Returns:
        (mean_reward, mean_latency, mean_feasible_rate, std_reward)
    """
    episode_rewards = []
    episode_latencies = []
    episode_feasible_counts = []
    episode_lengths = []
    episode_slice_sizes = []
    
    obs, _ = env.reset()
    
    for episode in range(num_episodes):
        policy.reset()
        episode_reward = 0.0
        episode_latency = 0.0
        episode_feasible = 0
        episode_length = 0
        
        done = False
        
        while not done:
            # Get k from environment info (number of slices)
            # We need to extract this from the environment's calculation
            # For now, use a heuristic based on task info
            # task_info format: [W, H, max_lat, bits, flops, ratio, rate]
            # We'll let the policy decide k based on its slice strategy
            # The environment will calculate k internally
            
            # Get baseline action
            action = policy.get_action(obs, k=6)  # Use num_nearest_compute as default k
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Accumulate metrics
            episode_reward += float(reward)
            episode_latency += float(info.get('total_latency', 0.0))
            if info.get('feasible', False):
                episode_feasible += 1
            episode_length += 1
            
            # Track slice size used
            if 'chosen_slice_size' in info:
                episode_slice_sizes.append(float(info['chosen_slice_size']))
        
        # Log episode metrics
        episode_rewards.append(episode_reward)
        episode_latencies.append(episode_latency)
        episode_feasible_counts.append(episode_feasible)
        episode_lengths.append(episode_length)
        
        # Write to TensorBoard - per episode
        step = global_step + episode
        writer.add_scalar(f'baselines/{policy_name}/episode_reward', episode_reward, step)
        writer.add_scalar(f'baselines/{policy_name}/episode_latency', episode_latency, step)
        feasible_rate = episode_feasible / max(episode_length, 1)
        writer.add_scalar(f'baselines/{policy_name}/feasible_rate', feasible_rate, step)
        writer.add_scalar(f'baselines/{policy_name}/episode_length', episode_length, step)
        
        print(f"[{policy_name}] Episode {episode+1}/{num_episodes}: "
              f"reward={episode_reward:.4f}, latency={episode_latency:.2f}s, "
              f"feasible_rate={feasible_rate:.2%}, length={episode_length}")
    
    # Compute statistics
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_latency = float(np.mean(episode_latencies))
    mean_feasible_rate = float(np.mean([c / max(l, 1) for c, l in zip(episode_feasible_counts, episode_lengths)]))
    
    # Write summary statistics to TensorBoard (at step 0 for easy comparison)
    writer.add_scalar(f'baselines/summary/mean_reward_{policy_name}', mean_reward, 0)
    writer.add_scalar(f'baselines/summary/std_reward_{policy_name}', std_reward, 0)
    writer.add_scalar(f'baselines/summary/mean_latency_{policy_name}', mean_latency, 0)
    writer.add_scalar(f'baselines/summary/mean_feasible_rate_{policy_name}', mean_feasible_rate, 0)
    
    print(f"\n[{policy_name}] Summary Statistics:")
    print(f"  Mean Episode Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"  Mean Episode Latency: {mean_latency:.2f}s")
    print(f"  Mean Feasible Rate: {mean_feasible_rate:.2%}\n")
    
    return mean_reward, mean_latency, mean_feasible_rate, std_reward


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load simulation config to get slicing strategies
    sim_config = load_config(args.sim_config)
    num_slice_strategies = len(sim_config.get('slicing_strategies', []))
    
    # Prepare run directories (reuse training structure)
    paths = prepare_run_dirs(algo="baselines", run_name=args.run_name)
    
    # Save baseline metadata
    save_run_metadata(paths["meta"], {
        "algo": "Baselines",
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "sim_config": args.sim_config,
        "sats_config": args.sats_config,
        "tb_dir": paths["tb"],
        "policies": ["random", "greedy", "round_robin"],
        "override_max_tasks_per_episode": args.override_max_tasks_per_episode,
        "num_slice_strategies": num_slice_strategies,
        "slicing_strategies": [s['slice_size'] for s in sim_config.get('slicing_strategies', [])],
    })
    
    # Create a single Gymnasium env (non-VecEnv) to simplify baseline rollout
    env_thunk = _build_single_env(
        sim_config_path=args.sim_config,
        sats_config_path=args.sats_config,
        seed=args.seed,
        use_action_wrapper=False,          # keep Dict action interface (Dict actions expected)
        monitor_log_dir=None,
        override_max_tasks_per_episode=args.override_max_tasks_per_episode,
    )
    env = env_thunk()

    # Get environment parameters
    obs, info = env.reset()
    num_compute_nodes = obs['compute_pos'].shape[0]

    # Derive max_k directly from env action space for correctness
    assign_space = env.action_space.spaces['assignment']
    max_k = int(len(assign_space.nvec))

    print(f"Environment Configuration:")
    print(f"  Number of compute nodes (nearest): {num_compute_nodes}")
    print(f"  Number of slice strategies: {num_slice_strategies}")
    print(f"  Slice sizes: {[s['slice_size'] for s in sim_config.get('slicing_strategies', [])]}")
    print(f"  max_k (from env action space): {max_k}\n")
    
    # Create TensorBoard writer
    writer = SummaryWriter(paths["tb"])
    
    # Test each baseline policy
    policies = [
        (RandomPolicy(num_compute_nodes, num_slice_strategies, max_k), "random"),
        (GreedyPolicy(num_compute_nodes, num_slice_strategies, max_k), "greedy"),
        (RoundRobinPolicy(num_compute_nodes, num_slice_strategies, max_k), "round_robin"),
    ]
    
    results = {}
    for policy, policy_name in policies:
        print(f"=" * 70)
        print(f"Testing {policy_name.upper()} Policy")
        print(f"=" * 70)
        
        mean_reward, mean_latency, mean_feasible_rate, std_reward = run_baseline_evaluation(
            policy=policy,
            policy_name=policy_name,
            env=env,
            num_episodes=args.num_episodes,
            writer=writer,
            sim_config=sim_config,
            global_step=0,
        )
        
        results[policy_name] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_latency': mean_latency,
            'mean_feasible_rate': mean_feasible_rate,
        }
    
    # Write comparison summary
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 70)
    
    # Sort by mean reward (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
    
    for rank, (policy_name, metrics) in enumerate(sorted_results, 1):
        print(f"\n#{rank} {policy_name.upper()}:")
        print(f"  Mean Reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
        print(f"  Mean Latency: {metrics['mean_latency']:.2f}s")
        print(f"  Mean Feasible Rate: {metrics['mean_feasible_rate']:.2%}")
    
    # Write detailed summary text to TensorBoard
    summary_text = "BASELINE COMPARISON\n\n"
    for policy_name, metrics in sorted_results:
        summary_text += f"{policy_name.upper()}:\n"
        summary_text += f"  Reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}\n"
        summary_text += f"  Latency: {metrics['mean_latency']:.2f}s\n"
        summary_text += f"  Feasible Rate: {metrics['mean_feasible_rate']:.2%}\n\n"
    
    writer.add_text('baselines/comparison_summary', summary_text)
    
    # Write individual policy descriptions
    writer.add_text('baselines/policies/random', 
                   "Random Policy:\n"
                   "- Slice Strategy: Randomly select from available slicing strategies\n"
                   "- Assignment: Randomly assign to compute nodes\n"
                   "- Rationale: Baseline to show benefit of intelligent scheduling")
    
    writer.add_text('baselines/policies/greedy',
                   "Greedy Policy:\n"
                   "- Slice Strategy: Use smallest slice size (128) for maximum parallelism\n"
                   "- Assignment: Always assign to k nodes with smallest queues\n"
                   "- Rationale: Balances fine-grained parallelism with load balancing")
    
    writer.add_text('baselines/policies/round_robin',
                   "Round-Robin Policy:\n"
                   "- Slice Strategy: Use medium slice size (512) for balanced throughput\n"
                   "- Assignment: Cycle through compute nodes in order\n"
                   "- Rationale: Classic load-balancing strategy for fairness")
    
    writer.flush()
    writer.close()
    
    # Cleanup
    env.close()
    
    print(f"\n" + "=" * 70)
    print(f"Results saved to: {paths['tb']}")
    print(f"View with: tensorboard --logdir {paths['tb']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

