# -*- coding: utf-8 -*-
"""
Train PPO on the SatelliteEnv with Dict observation and Dict action (flattened).

- Uses group-based TLE fetching and caching
- Flattens Dict action into a single MultiDiscrete for SB3
- Vectorized environments (SubprocVecEnv)
- TensorBoard logging and organized directories
- Periodic evaluation and checkpoints
"""
from __future__ import annotations

import os
import sys
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

# Ensure project root on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training.utils import (
    make_vec_envs,
    prepare_run_dirs,
    save_run_metadata,
    build_callbacks,
)


def parse_args():
    p = argparse.ArgumentParser("PPO training for SatelliteEnv")
    p.add_argument("--sim_config", default=os.path.join("configs", "environment", "simulation.yaml"))
    p.add_argument("--sats_config", default=os.path.join("configs", "satellites.yaml"))
    p.add_argument("--total_timesteps", type=int, default=2_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--use_subproc", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    # Prepare run directories
    paths = prepare_run_dirs(algo="ppo", run_name=args.run_name)
    save_run_metadata(paths["meta"], {
        "algo": "PPO",
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "sim_config": args.sim_config,
        "sats_config": args.sats_config,
        "tb_dir": paths["tb"],
        "models_dir": paths["models"],
        "monitor_dir": paths["monitor"],
        "evals_dir": paths["evals"],
    })

    # Build training and evaluation envs
    train_env = make_vec_envs(
        sim_config_path=args.sim_config,
        sats_config_path=args.sats_config,
        n_envs=args.n_envs,
        seed=args.seed,
        use_subproc=args.use_subproc,
        use_action_wrapper=True,
        monitor_log_dir=paths["monitor"],
    )
    eval_env = make_vec_envs(
        sim_config_path=args.sim_config,
        sats_config_path=args.sats_config,
        n_envs=1,
        seed=args.seed + 10_000,
        use_subproc=False,
        use_action_wrapper=True,
        monitor_log_dir=None,
    )

    # Model
    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        verbose=1,
        tensorboard_log=paths["tb"],
        device="cuda",
    )

    # Callbacks
    callbacks = build_callbacks(eval_env=eval_env,
                                models_dir=paths["models"],
                                evals_dir=paths["evals"],
                                best_model_name="best_model",
                                save_freq_steps=100_000)

    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)

    # Final save
    model.save(os.path.join(paths["models"], "final_model"))

    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

