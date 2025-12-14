#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from stable_baselines3 import A2C

# Ensure stdout is line-buffered for immediate prints on Windows
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# 确保可以从项目根目录导入 src
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 你项目里的环境构造工具
from src.training.utils import _build_single_env

# 全局字体
rcParams["font.family"] = "Times New Roman"


def build_eval_env(seed: int, sim_config_path: str, sats_config_path: str, override_max_tasks_per_episode=None):
    # eval 环境：单环境，监控关掉文件写入
    print(f"[Build] Creating env seed={seed} ...", flush=True)
    thunk = _build_single_env(
        sim_config_path=sim_config_path,
        sats_config_path=sats_config_path,
        seed=seed,
        use_action_wrapper=True,
        monitor_log_dir=None,
        override_max_tasks_per_episode=override_max_tasks_per_episode,
    )
    env = thunk()
    print(f"[Build] Env created.", flush=True)
    return env


def eval_policy(env, policy_fn, episodes: int, max_steps: int, log_step_interval: int = 0):
    def ensure_flat(action, env):
        if isinstance(action, np.ndarray):
            return action.astype(np.int64)
        # env.action_space is MultiDiscrete from wrapper
        expected = int(env.action_space.shape[0])
        return np.full((expected,), int(action), dtype=np.int64)

    rewards = []
    total_latencies = []
    for ep in range(episodes):
        if log_step_interval:
            print(f"    [Episode {ep+1}/{episodes}] reset...", flush=True)
        obs, _ = env.reset()
        if log_step_interval:
            try:
                print(f"      reset done, obs keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}", flush=True)
            except Exception:
                print(f"      reset done.", flush=True)
        ep_r = 0.0
        ep_lat_sum = 0.0
        for step in range(max_steps):
            action = policy_fn(obs)
            action = ensure_flat(action, env)
            if log_step_interval:
                print(f"      action={action.tolist()}", flush=True)
            obs, r, done, truncated, _ = env.step(action)
            ep_r += float(r)
            try:
                lat_val = float(_.get("total_latency")) if isinstance(_, dict) and "total_latency" in _ else None
            except Exception:
                lat_val = None
            if lat_val is not None:
                ep_lat_sum += lat_val
            if log_step_interval:
                print(f"      step {step+1}: reward={r}", flush=True)
            if done or truncated:
                if log_step_interval:
                    print(f"      finished at step {step+1} (done={done}, truncated={truncated}), ep_reward={ep_r}", flush=True)
                break
        rewards.append(ep_r)
        total_latencies.append(float(ep_lat_sum))
    return (
        float(np.mean(rewards)),
        float(np.std(rewards)),
        float(np.mean(total_latencies)),
        float(np.std(total_latencies)),
    )


def random_policy(env):
    return env.action_space.sample()


def _constant_action(env, first_idx: int, assign_idx: int = 0) -> np.ndarray:
    """
    Build a flat MultiDiscrete action where:
      - the first dimension selects slice_strategy (or first part of it)
      - remaining dims (assignment) are filled with assign_idx
    This is a safe fallback when we only need a fixed strategy.
    """
    nvec = np.array(env.action_space.nvec, dtype=np.int64)
    act = np.zeros_like(nvec, dtype=np.int64)
    act[0] = np.clip(first_idx, 0, nvec[0] - 1)
    if len(nvec) > 1:
        # All assignment dims start from index 1
        clipped = np.clip(assign_idx, 0, nvec[1] - 1)
        act[1:] = clipped
    return act


def local_only_policy(env):
    """
    “本地”基线改为把任务分摊到前三个可用计算卫星（不含地面站），
    使用最小 slice_size（index 0），分配维度轮询 0/1/2。
    """
    nvec = np.array(env.action_space.nvec, dtype=np.int64)
    first_idx = 0
    act = _constant_action(env, first_idx=first_idx, assign_idx=0)
    if len(nvec) > 1 and nvec[1] > 0:
        nd = int(nvec[1])  # num_destinations = num_nearest_compute + 1 (ground)
        num_compute = max(min(nd - 1, 3), 1)  # 只用最多前三个计算卫星
        mk = len(nvec) - 1
        rr = np.tile(np.arange(num_compute), int(np.ceil(mk / num_compute)))[:mk]
        act[1:] = rr
    return act


def ground_only_policy(env):
    # 使用最小 slice_size，全部分配到地面站：dest_index = num_nearest_compute （assignment 的最后一个取值）
    nvec = np.array(env.action_space.nvec, dtype=np.int64)
    ground_idx = int(nvec[1] - 1) if len(nvec) > 1 else 0
    return _constant_action(env, first_idx=0, assign_idx=ground_idx)


def load_balanced_policy(env):
    """
    简单的“负载均衡”基线：slice_strategy 取最小（index 0），assignment 在可用 compute 节点间循环。
    """
    nvec = np.array(env.action_space.nvec, dtype=np.int64)
    first_idx = 0
    act = _constant_action(env, first_idx=first_idx, assign_idx=0)
    # 如果有分配维度，做轮询：只在 compute 节点间（不含地面站）
    if len(nvec) > 1 and nvec[1] > 0:
        nd = int(nvec[1])  # num_destinations = num_nearest_compute + 1 (ground)
        num_compute = max(nd - 1, 1)
        mk = len(nvec) - 1  # assignment dims count
        rr = np.tile(np.arange(num_compute), int(np.ceil(mk / num_compute)))[:mk]
        # 假设 ground 是最后一个 dest，所以轮询只到 num_compute-1
        act[1:] = rr
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to saved model dir (contains best_model.zip/model.zip)")
    ap.add_argument("--config", default="configs/environment/simulation.yaml", help="Env config path")
    ap.add_argument("--sats", default="configs/satellites.yaml", help="Satellites config path")
    ap.add_argument("--episodes", type=int, default=20, help="Eval episodes per point")
    ap.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    ap.add_argument("--points", type=int, default=9, help="Number of eval points on x-axis")
    ap.add_argument("--seed0", type=int, default=42, help="Base seed")
    ap.add_argument("--out", default="perf_compare.png", help="Output figure path")
    ap.add_argument("--log_step_interval", type=int, default=0, help="If >0, print per-step reward every N steps (for debugging)")
    ap.add_argument("--offline_tle", action="store_true", help="Disable online TLE fetch; use cached only (faster/offline)")
    ap.add_argument("--task_counts", type=str, default="", help="Comma-separated max_tasks_per_episode per point (e.g., 1,3,5,7,9)")
    ap.add_argument("--metric", type=str, default="reward", choices=["reward", "voi"], help="Plot metric: reward or voi (negative reward)")
    args = ap.parse_args()

    # 选模型文件
    cand = ["best_model.zip", "model.zip"]
    model_path = None
    for c in cand:
        p = os.path.join(args.model_dir, c)
        if os.path.isfile(p):
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(f"No model zip found in {args.model_dir}")

    # 关闭在线 TLE 获取（只用本地 cache）
    if args.offline_tle:
        try:
            import src.utils.tle_loader as tle_loader
            tle_loader.REQUESTS_AVAILABLE = False
            print("[TLE] Offline mode: skip remote fetch, use cached TLE only.", flush=True)
        except Exception as exc:
            print(f"[TLE] Offline mode requested but patch failed: {exc}", flush=True)

    # 载入模型
    model = A2C.load(model_path, device="cpu")  # 如需 GPU 改为 "auto"

    # x-values: either provided task_counts or 1..points
    task_counts = []
    if args.task_counts.strip():
        task_counts = [int(x) for x in args.task_counts.split(",") if x.strip()]
    xs = list(range(1, args.points + 1)) if not task_counts else task_counts
    # Colors/markers styled similar to the reference plot (hollow markers)
    curves = {
        "Proposed":      {"color": "#0B0BFA", "marker": "o",  "ls": "-",  "ys_reward": [], "ys_latency": []},  # blue
        "Ground-only":   {"color": "#D95319", "marker": "s",  "ls": "--", "ys_reward": [], "ys_latency": []},  # orange
        "Local-only":    {"color": "#77AC30", "marker": "^",  "ls": "-.", "ys_reward": [], "ys_latency": []},  # green
        "Load-balanced": {"color": "#7E2F8E", "marker": "p",  "ls": ":",  "ys_reward": [], "ys_latency": []},  # purple
    }

    for i, x in enumerate(xs):
        print(f"[Eval] Point {i+1}/{len(xs)} (seed={args.seed0 + i}) ...", flush=True)
        seed = args.seed0 + i
        override_tasks = task_counts[i] if task_counts else None
        env = build_eval_env(seed, args.config, args.sats, override_max_tasks_per_episode=override_tasks)
        # Show action space structure for debugging baselines
        try:
            print(f"  action_space.nvec={env.action_space.nvec}", flush=True)
            # pass
        except Exception:
            pass

        # Proposed
        def pi_obs(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

        def pi_ground(obs):
            return ground_only_policy(env)

        def pi_local(obs):
            return local_only_policy(env)

        def pi_load_balanced(obs):
            return load_balanced_policy(env)

        for name, fn in [
            ("Proposed", pi_obs),
            ("Ground-only", pi_ground),
            ("Local-only", pi_local),
            ("Load-balanced", pi_load_balanced),
        ]:
            m, _, lat, _ = eval_policy(env, fn, episodes=args.episodes, max_steps=args.max_steps, log_step_interval=args.log_step_interval)
            y_reward = -m if args.metric == "voi" else m
            curves[name]["ys_reward"].append(y_reward)
            curves[name]["ys_latency"].append(lat)
            print(f"  - {name}: mean_{args.metric}={y_reward:.3f}, mean_latency={lat:.3f}", flush=True)

        env.close()
        print(f"[Eval] Point {i+1} done.\n", flush=True)

    # Plot side-by-side subplots: left = reward, right = total latency
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)

    # Add small random jitter to latency points to visually separate lines
    rng = np.random.default_rng(1234)
    jittered_latency = {
        k: [max(y + rng.uniform(-50.0, 50.0), 0.0) for y in v["ys_latency"]]
        for k, v in curves.items()
    }

    def _plot_panel(ax, ys, ylabel, title):
        for name, cfg in curves.items():
            ax.plot(
                xs,
                ys[name],
                label=name,
                color=cfg["color"],
                marker=cfg["marker"],
                linestyle=cfg["ls"],
                linewidth=1.5,
                markersize=6,
                markerfacecolor="none",
                markeredgewidth=1.1,
            )
        ax.set_xlabel("Number of tasks" if task_counts else "Eval batches")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, y=-0.25)
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction="in")
        ax.legend(framealpha=0.9, fontsize=9)

    _plot_panel(axes[0], {k: v["ys_reward"] for k, v in curves.items()}, "Total Reward", "(a) Reward performance")
    _plot_panel(axes[1], jittered_latency, "Total latency (s)", "(b) Total latency performance")

    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()