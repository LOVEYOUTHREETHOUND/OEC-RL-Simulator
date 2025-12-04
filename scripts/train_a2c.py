# -*- coding: utf-8 -*-
"""
Train A2C on the SatelliteEnv with Dict observation and Dict action (flattened).

- Reuses TLE preprocessing and action wrapper
- Vectorized environments (SubprocVecEnv/DummyVecEnv)
- TensorBoard logging and organized directories
- Periodic evaluation and checkpoints
"""
from __future__ import annotations

import os
import sys
import argparse

from stable_baselines3 import A2C
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
    p = argparse.ArgumentParser("A2C training for SatelliteEnv")
    p.add_argument("--sim_config", default=os.path.join("configs", "environment", "simulation.yaml"))
    p.add_argument("--sats_config", default=os.path.join("configs", "satellites.yaml"))
    p.add_argument("--total_timesteps", type=int, default=2_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--use_subproc", action="store_true")
    # Fast/clear-convergence knobs
    p.add_argument("--n_steps", type=int, default=4096)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--override_max_tasks_per_episode", type=int, default=None)
    p.add_argument("--fast_preset", action="store_true", help="Use faster/more responsive settings for clearer convergence curves")
    # Evaluation / checkpoint / early-stop knobs
    p.add_argument("--eval_freq_steps", type=int, default=50_000, help="Eval frequency in env calls (VecEnv calls).")
    p.add_argument("--save_ckpt_every_steps_env", type=int, default=100_000, help="Checkpoint frequency in effective env timesteps. Will be divided by n_envs to get callback calls.")
    p.add_argument("--early_stop", action="store_true", help="Enable early stopping based on eval mean reward.")
    p.add_argument("--early_stop_patience_evals", type=int, default=10, help="Number of consecutive non-improving evals to stop training.")
    p.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Absolute improvement required to count as better.")
    p.add_argument("--early_stop_min_delta_rel", type=float, default=0.01, help="Relative improvement (fraction of best) required to count as better.")
    return p.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    # Apply fast preset if requested
    if args.fast_preset:
        args.n_steps = 1024
        args.learning_rate = 1e-4
        args.ent_coef = 0.005
        if args.override_max_tasks_per_episode is None:
            args.override_max_tasks_per_episode = 30

    # Prepare run directories
    paths = prepare_run_dirs(algo="a2c", run_name=args.run_name)
    save_run_metadata(paths["meta"], {
        "algo": "A2C",
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
            "n_steps": args.n_steps,
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
    model = A2C(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        seed=args.seed,
        verbose=1,
        tensorboard_log=paths["tb"],
        device="cuda",
    )

    # Two-column episode reward TXT alongside TB dir
    episode_reward_txt = os.path.join(os.path.dirname(paths["tb"]), "by_episode_reward.txt")

    # Callbacks
    # scale checkpoint call-frequency by n_envs so that effective timesteps ~= save_ckpt_every_steps_env
    ckpt_calls = max(int(args.save_ckpt_every_steps_env // max(1, args.n_envs)), 1)
    callbacks = build_callbacks(eval_env=eval_env,
                                models_dir=paths["models"],
                                evals_dir=paths["evals"],
                                best_model_name="best_model",
                                save_freq_steps=ckpt_calls,
                                train_logs_dir=paths["train_logs"],
                                train_plain_logs_dir=paths["train_plain_logs"],
                                episode_reward_txt_path=episode_reward_txt,
                                eval_freq_steps=args.eval_freq_steps,
                                early_stop=bool(args.early_stop),
                                early_stop_patience_evals=args.early_stop_patience_evals,
                                early_stop_min_delta=args.early_stop_min_delta,
                                early_stop_min_delta_rel=args.early_stop_min_delta_rel,
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
