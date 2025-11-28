# -*- coding: utf-8 -*-
"""
Evaluate a trained agent (PPO/A2C) on SatelliteEnv and export metrics.

Metrics:
- Non-zero reward rate (overall and per-episode; only counting steps that processed a task)
- Timeout rate (reward==0 with a processed task)
- Average total latency (seconds)

Outputs:
- CSV with per-episode metrics
- Plots: latency histogram; per-episode avg latency curve; per-episode rates

Usage examples (PowerShell):
  # Evaluate PPO best model for 50 episodes, auto infer algo from path
  python -u scripts/eval_agent.py --model results/models/ppo/<run>/best_model.zip --episodes 50 --use_subproc

  # Only keep concise logs
  python -u scripts/eval_agent.py --model results/models/ppo/<run>/best_model.zip *>&1 \
    | Select-String -Pattern 'Saved metrics|\[ERROR\]|\[WARNING\]' | Tee-Object -FilePath results/evals/last_eval.log
"""
from __future__ import annotations

import os
import sys
import argparse
import time
from typing import Dict, Any, List

import numpy as np

# Optional deps
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # type: ignore

# Stable-Baselines3
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.utils import set_random_seed

# Ensure project root on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training.utils import make_vec_envs


def infer_algo_from_model_path(model_path: str) -> str:
    lower = model_path.lower()
    if os.sep + "ppo" + os.sep in lower:
        return "ppo"
    if os.sep + "a2c" + os.sep in lower:
        return "a2c"
    # Fallback: try by filename
    if "ppo" in lower:
        return "ppo"
    if "a2c" in lower:
        return "a2c"
    return "ppo"  # default


essential = (
    "[ERROR] pandas/matplotlib not installed. Install with: pip install pandas matplotlib",
)


def parse_args():
    p = argparse.ArgumentParser("Evaluate trained agent on SatelliteEnv")
    p.add_argument("--model", type=str, required=True, help="Path to model zip (best_model.zip or final_model.zip)")
    p.add_argument("--algo", type=str, choices=["ppo", "a2c"], default=None, help="Override algorithm (auto infer from path if omitted)")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--sim_config", default=os.path.join("configs", "environment", "simulation.yaml"))
    p.add_argument("--sats_config", default=os.path.join("configs", "satellites.yaml"))
    p.add_argument("--use_subproc", action="store_true")
    p.add_argument("--out_dir", default=None, help="Output directory for CSV/plots (default under results/evals/{algo}/{run}")
    return p.parse_args()


def build_out_dir(model_path: str, algo: str, out_dir: str | None) -> str:
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    # Derive a run name from model parent dir
    parent = os.path.basename(os.path.dirname(os.path.abspath(model_path)))
    ts = time.strftime("%Y%m%d-%H%M%S")
    eval_dir = os.path.join(project_root, "results", "evals", algo, f"eval_{parent}_{ts}")
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def load_model(algo: str, model_path: str, env) -> Any:
    if algo == "ppo":
        return PPO.load(model_path, env=env, print_system_info=False)
    elif algo == "a2c":
        return A2C.load(model_path, env=env, print_system_info=False)
    else:
        raise ValueError(f"Unsupported algo: {algo}")


def main():
    args = parse_args()
    set_random_seed(args.seed)

    algo = args.algo or infer_algo_from_model_path(args.model)

    # Vectorized single env for evaluation
    vec_env = make_vec_envs(
        sim_config_path=args.sim_config,
        sats_config_path=args.sats_config,
        n_envs=1,
        seed=args.seed,
        use_subproc=args.use_subproc,
        use_action_wrapper=True,
        monitor_log_dir=None,
    )

    model = load_model(algo, args.model, vec_env)

    # Metrics containers
    ep_records: List[Dict[str, Any]] = []
    all_latencies: List[float] = []
    total_task_steps = 0
    total_nonzero_steps = 0
    total_timeout_steps = 0

    for ep in range(args.episodes):
        obs = vec_env.reset()
        done = np.array([False])

        ep_task_steps = 0
        ep_nonzero_steps = 0
        ep_timeout_steps = 0
        ep_latencies: List[float] = []

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, infos = vec_env.step(action)
            info = infos[0]
            r = float(rewards[0])

            # Only count steps that processed a task (info contains total_latency)
            if isinstance(info, dict) and ('total_latency' in info):
                lat = float(info.get('total_latency', 0.0))
                ep_latencies.append(lat)
                all_latencies.append(lat)
                ep_task_steps += 1
                total_task_steps += 1

                if r > 0:
                    ep_nonzero_steps += 1
                    total_nonzero_steps += 1
                else:
                    ep_timeout_steps += 1
                    total_timeout_steps += 1

        # Episode summary
        ep_avg_latency = float(np.mean(ep_latencies)) if ep_latencies else float('nan')
        ep_nonzero_rate = (ep_nonzero_steps / ep_task_steps) if ep_task_steps > 0 else float('nan')
        ep_timeout_rate = (ep_timeout_steps / ep_task_steps) if ep_task_steps > 0 else float('nan')
        ep_records.append({
            'episode': ep + 1,
            'task_steps': ep_task_steps,
            'nonzero_steps': ep_nonzero_steps,
            'timeout_steps': ep_timeout_steps,
            'nonzero_rate': ep_nonzero_rate,
            'timeout_rate': ep_timeout_rate,
            'avg_latency_sec': ep_avg_latency,
        })
        print(f"Episode {ep+1:03d}: tasks={ep_task_steps}, nonzero_rate={ep_nonzero_rate:.3f}, timeout_rate={ep_timeout_rate:.3f}, avg_latency={ep_avg_latency:.2f}s")

    # Overall summary
    overall_nonzero_rate = (total_nonzero_steps / total_task_steps) if total_task_steps > 0 else float('nan')
    overall_timeout_rate = (total_timeout_steps / total_task_steps) if total_task_steps > 0 else float('nan')
    overall_avg_latency = float(np.mean(all_latencies)) if all_latencies else float('nan')

    print("\n=== Overall ===")
    print(f"Task steps: {total_task_steps}")
    print(f"Non-zero reward rate: {overall_nonzero_rate:.4f}")
    print(f"Timeout rate: {overall_timeout_rate:.4f}")
    print(f"Average latency (sec): {overall_avg_latency:.2f}")

    # Output directory
    out_dir = build_out_dir(args.model, algo, args.out_dir)

    # Save CSV
    csv_path = os.path.join(out_dir, "metrics.csv")
    try:
        if pd is None:
            raise RuntimeError(essential[0])
        df = pd.DataFrame(ep_records)
        df.to_csv(csv_path, index=False)
        print(f"Saved metrics CSV: {csv_path}")
    except Exception as e:
        print(f"[WARNING] Could not save CSV ({e}). Install pandas to enable CSV export.")

    # Plots
    try:
        if plt is None:
            raise RuntimeError(essential[0])

        # 1) Per-episode average latency
        plt.figure(figsize=(8, 4))
        xs = [rec['episode'] for rec in ep_records]
        ys = [rec['avg_latency_sec'] for rec in ep_records]
        plt.plot(xs, ys, marker='o')
        plt.xlabel('Episode')
        plt.ylabel('Avg Latency (sec)')
        plt.title('Per-episode Avg Latency')
        plt.grid(True, alpha=0.3)
        fig1_path = os.path.join(out_dir, 'avg_latency_per_episode.png')
        plt.tight_layout(); plt.savefig(fig1_path); plt.close()

        # 2) Per-episode rates
        plt.figure(figsize=(8, 4))
        nz = [rec['nonzero_rate'] for rec in ep_records]
        to = [rec['timeout_rate'] for rec in ep_records]
        plt.plot(xs, nz, label='Non-zero rate', marker='o')
        plt.plot(xs, to, label='Timeout rate', marker='s')
        plt.xlabel('Episode')
        plt.ylabel('Rate')
        plt.ylim(0, 1)
        plt.title('Per-episode Rates')
        plt.legend(); plt.grid(True, alpha=0.3)
        fig2_path = os.path.join(out_dir, 'rates_per_episode.png')
        plt.tight_layout(); plt.savefig(fig2_path); plt.close()

        # 3) Latency histogram
        if all_latencies:
            plt.figure(figsize=(6, 4))
            plt.hist(all_latencies, bins=30, color='steelblue', alpha=0.9)
            plt.xlabel('Latency (sec)')
            plt.ylabel('Count')
            plt.title('Latency Distribution (all task steps)')
            plt.grid(True, alpha=0.3)
            fig3_path = os.path.join(out_dir, 'latency_histogram.png')
            plt.tight_layout(); plt.savefig(fig3_path); plt.close()
        else:
            fig3_path = None

        print("Saved plots:")
        print(f"  {fig1_path}")
        print(f"  {fig2_path}")
        if fig3_path:
            print(f"  {fig3_path}")

    except Exception as e:
        print(f"[WARNING] Could not save plots ({e}). Install matplotlib to enable plotting.")

    # Write overall summary
    summary_path = os.path.join(out_dir, "summary.txt")
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== Overall ===\n")
            f.write(f"Task steps: {total_task_steps}\n")
            f.write(f"Non-zero reward rate: {overall_nonzero_rate:.4f}\n")
            f.write(f"Timeout rate: {overall_timeout_rate:.4f}\n")
            f.write(f"Average latency (sec): {overall_avg_latency:.2f}\n")
        print(f"Saved summary: {summary_path}")
    except Exception as e:
        print(f"[WARNING] Could not save summary ({e}).")

    vec_env.close()


if __name__ == "__main__":
    main()

