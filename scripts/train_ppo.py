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

from stable_baselines3 import PPO, A2C
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
    # Fast/clear-convergence knobs
    p.add_argument("--n_steps", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--override_max_tasks_per_episode", type=int, default=None)
    p.add_argument("--fast_preset", action="store_true", help="Use faster/more responsive settings for clearer convergence curves")
    p.add_argument("--online_step_update", action="store_true", help="Train in online mode (step-by-step updates). Uses A2C with n_steps=1 and n_envs=1.")
    return p.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    # Apply fast preset if requested
    if args.fast_preset:
        args.n_steps = 1024
        args.n_epochs = 5
        args.learning_rate = 1e-4
        args.ent_coef = 0.005
        args.batch_size = 2048
        if args.override_max_tasks_per_episode is None:
            args.override_max_tasks_per_episode = 30

    # If online step-by-step update requested, override some settings
    if args.online_step_update:
        args.n_envs = 8
        args.use_subproc = False

    # Prepare run directories
    paths = prepare_run_dirs(algo="ppo", run_name=args.run_name)
    save_run_metadata(paths["meta"], {
        "algo": "PPO" if not args.online_step_update else "A2C-online",
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "sim_config": args.sim_config,
        "sats_config": args.sats_config,
        "tb_dir": paths["tb"],
        "models_dir": paths["models"],
        "monitor_dir": paths["monitor"],
        "evals_dir": paths["evals"],
        "hyperparams": {
            "online_step_update": bool(args.online_step_update),
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "ent_coef": args.ent_coef,
            "override_max_tasks_per_episode": args.override_max_tasks_per_episode,
            "fast_preset": bool(args.fast_preset),
        }
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
        override_max_tasks_per_episode=args.override_max_tasks_per_episode,
    )
    # Make eval env type consistent with train env to avoid warnings
    eval_env_n = args.n_envs if (args.use_subproc and args.n_envs > 1) else 1
    eval_env = make_vec_envs(
        sim_config_path=args.sim_config,
        sats_config_path=args.sats_config,
        n_envs=eval_env_n,
        seed=args.seed + 10_000,
        use_subproc=args.use_subproc,
        use_action_wrapper=True,
        monitor_log_dir=None,
        override_max_tasks_per_episode=args.override_max_tasks_per_episode,
    )

    # Model
    if args.online_step_update:
        model = A2C(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=1,  # update every step
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            seed=args.seed,
            verbose=1,
            tensorboard_log=paths["tb"],
            device="cuda",
        )
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=args.seed,
            verbose=1,
            tensorboard_log=paths["tb"],
            device="cuda",
        )

    # Callbacks
    info_keys = [
        "total_latency",
        "miou",
        "feasible",
        "chosen_slice_size",
        "calculated_k",
    ]
    callbacks = build_callbacks(eval_env=eval_env,
                                models_dir=paths["models"],
                                evals_dir=paths["evals"],
                                best_model_name="best_model",
                                save_freq_steps=100_000,
                                info_keys=info_keys,
                                info_log_every=200,
                                train_logs_dir=paths["train_logs"],
                                train_plain_logs_dir=paths["train_plain_logs"],
                                log_rotate_every_episodes=100)

    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)

    # Ensure TensorBoard logs are flushed to disk
    try:
        if hasattr(model, "logger"):
            model.logger.dump(model.num_timesteps)
    except Exception:
        pass

    # Final save
    model.save(os.path.join(paths["models"], "final_model"))

    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

