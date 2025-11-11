# -*- coding: utf-8 -*-
"""
Main script for training the Hierarchical Reinforcement Learning agent.
"""

import os
import sys
import yaml
from pprint import pprint

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.environment.satellite_env import SatelliteEnv
from src.agents.hrl_agent import HierarchicalAgent

def load_config(path):
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main training loop."""
    print("--- Starting HRL Agent Training ---")

    # --- 1. Load Configurations ---
    print("\n[1/4] Loading configurations...")
    sim_config = load_config(os.path.join(project_root, 'configs/environment/simulation_v1.yaml'))
    sat_configs = load_config(os.path.join(project_root, 'configs/satellites.yaml'))
    agent_config = load_config(os.path.join(project_root, 'configs/agent/hrl_agent.yaml'))
    
    # A simple training config can also be useful
    training_config = {'total_timesteps': 100000}
    
    print("Configurations loaded successfully.")
    pprint(agent_config)

    # --- 2. Initialize Environment ---
    print("\n[2/4] Initializing the environment...")
    env = SatelliteEnv(sim_config=sim_config, sat_configs=sat_configs)
    print("Environment initialized.")

    # --- 3. Initialize Agent ---
    print("\n[3/4] Initializing the Hierarchical Agent...")
    agent = HierarchicalAgent(env, config=agent_config)
    print("Agent initialized.")

    # --- 4. Start Training ---
    print(f"\n[4/4] Starting training for {training_config['total_timesteps']} timesteps...")
    # Note: The learn method is a placeholder. A full HRL training loop is required.
    agent.learn(total_timesteps=training_config['total_timesteps'])
    print("\n--- Training script finished ---")

if __name__ == '__main__':
    main()

