# -*- coding: utf-8 -*-
"""
Main script for evaluating trained agents and comparing them with baselines.
"""

import os
import sys
import yaml
import numpy as np
from pprint import pprint
from tqdm import tqdm

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.environment.satellite_env import SatelliteEnv
# from src.agents.hrl_agent import HierarchicalAgent # Keep for future use
from src.agents.baselines import (
    RandomPolicy,
    GreedyMinQueuePolicy,
    RoundRobinPolicy,
    GroundOnlyPolicy,
    ComputeOnlyMinQueuePolicy,
    BasePolicy,
)

def load_config(path):
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_policy(policy: BasePolicy, env: SatelliteEnv, n_episodes: int) -> Dict[str, float]:
    """
    Runs a given policy for a number of episodes and returns performance metrics.
    """
    total_rewards = []
    total_latencies = []

    for _ in tqdm(range(n_episodes), desc=f'Evaluating {policy.__class__.__name__}'):
        obs, info = env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            # All policies now return a hierarchical action dictionary
            action = policy.predict(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if 'total_latency' in info:
                total_latencies.append(info['total_latency'])

        total_rewards.append(episode_reward)

    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_latency': np.mean(total_latencies),
    }

def main():
    """Main evaluation loop."""
    print("--- Starting Policy Evaluation ---")
    n_eval_episodes = 100

    # --- 1. Load Configurations ---
    print("\n[1/3] Loading configurations...")
    sim_config = load_config(os.path.join(project_root, 'configs/environment/simulation.yaml'))
    raw_sat_configs = load_config(os.path.join(project_root, 'configs/satellites.yaml'))
    # agent_config = load_config(os.path.join(project_root, 'configs/agent/hrl_agent.yaml'))

    # --- 2. Initialize Environment ---
    print("\n[2/3] Initializing the environment...")
    from src.utils.tle_loader import preprocess_satellite_configs
    sat_configs = preprocess_satellite_configs(project_root, raw_sat_configs)
    env = SatelliteEnv(sim_config=sim_config, sat_configs=sat_configs)

    # --- 3. Evaluate Policies ---
    print(f"\n[3/3] Evaluating policies for {n_eval_episodes} episodes...")
    policies_to_evaluate = {
        'Random': RandomPolicy(env),
        'GreedyMinQueue': GreedyMinQueuePolicy(env),
        'RoundRobin': RoundRobinPolicy(env),
        'GroundOnly': GroundOnlyPolicy(env),
        'ComputeOnlyMinQueue': ComputeOnlyMinQueuePolicy(env),
        # 'HRL_Agent': HierarchicalAgent(env, config=agent_config)
    }

    results = {}
    for name, policy in policies_to_evaluate.items():
        results[name] = evaluate_policy(policy, env, n_eval_episodes)
    
    print("\n--- Evaluation Results ---")
    pprint(results)
    print("\n--- Evaluation script finished ---")

if __name__ == '__main__':
    main()
