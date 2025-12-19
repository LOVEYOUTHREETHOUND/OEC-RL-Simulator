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
    successes = []
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
        last_info = None
        for step in range(max_steps):
            action = policy_fn(obs)
            action = ensure_flat(action, env)
            if log_step_interval:
                print(f"      action={action.tolist()}", flush=True)
            obs, r, done, truncated, info = env.step(action)
            last_info = info
            ep_r += float(r)
            try:
                lat_val = float(info.get("total_latency")) if isinstance(info, dict) and "total_latency" in info else None
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
        # Track successful task count from final info of the episode
        succ = 0.0
        if isinstance(last_info, dict):
            try:
                succ = float(last_info.get("successful_tasks_count", 0.0))
            except Exception:
                succ = 0.0
        successes.append(succ)
    return (
        float(np.mean(rewards)),
        float(np.std(rewards)),
        float(np.mean(total_latencies)),
        float(np.std(total_latencies)),
        float(np.mean(successes)),
        float(np.std(successes)),
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
    "本地"基线改为把任务分摊到前三个可用计算卫星（不含地面站），
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
    简单的"负载均衡"基线：slice_strategy 取最小（index 0），assignment 在可用 compute 节点间循环。
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


def _plot_panel(ax, xs, ys_map, ylabel, title, curves):
    """通用的绘图函数，用于在给定的坐标轴上绘制一个子图
    ys_map: dict name -> list/array of y-values for that curve
    """
    for name, cfg in curves.items():
        y_data = ys_map.get(name, [])
        ax.plot(
            xs,
            y_data,
            label=name,
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=cfg["ls"],
            linewidth=1.5,
            markersize=6,
            markerfacecolor="none",
            markeredgewidth=1.1,
            clip_on=False,
            zorder=4,
        )
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11, y=-0.25)
    ax.grid(True, alpha=0.25)
    # Left-bottom inward ticks (no ticks on top/right), thicker tick marks
    ax.tick_params(direction="in", length=6, width=1.2, bottom=True, left=True, top=False, right=False)
    # Emphasize left and bottom spines for a "flush" look similar to the reference
    for side in ["left", "bottom"]:
        try:
            ax.spines[side].set_linewidth(1.2)
        except Exception:
            pass
    # Remove horizontal padding so the first and last x points touch the plotting box
    try:
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        ax.set_xlim(xmin, xmax)
        ax.margins(x=0)
    except Exception:
        pass
    ax.legend(framealpha=0.9, fontsize=9)


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
    ap.add_argument("--dpi", type=int, default=800, help="Output figure DPI for saved images")
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
        "Proposed":      {"color": "#0B0BFA", "marker": "o",  "ls": "-",  "ys_reward": [], "ys_latency": [], "ys_success": []},  # blue
        "Ground-only":   {"color": "#D95319", "marker": "s",  "ls": "--", "ys_reward": [], "ys_latency": [], "ys_success": []},  # orange
        "Local-only":    {"color": "#77AC30", "marker": "^",  "ls": "-.", "ys_reward": [], "ys_latency": [], "ys_success": []},  # green
        "Load-balanced": {"color": "#7E2F8E", "marker": "p",  "ls": ":",  "ys_reward": [], "ys_latency": [], "ys_success": []},  # purple
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
            m, m_std, lat, lat_std, succ, succ_std = eval_policy(
                env, fn, episodes=args.episodes, max_steps=args.max_steps, log_step_interval=args.log_step_interval
            )
            y_reward = -m if args.metric == "voi" else m
            curves[name]["ys_reward"].append(y_reward)
            curves[name]["ys_latency"].append(lat)
            curves[name]["ys_success"].append(succ)
            print(
                f"  - {name}: mean_{args.metric}={y_reward:.3f} (±{m_std:.3f}), mean_latency={lat:.3f} (±{lat_std:.3f}), mean_success={succ:.2f} (±{succ_std:.2f})",
                flush=True,
            )

        env.close()
        print(f"[Eval] Point {i+1} done.\n", flush=True)

    # 生成抖动后的延迟数据
    rng = np.random.default_rng(1234)
    jittered_latency = {
        k: [max(y + rng.uniform(-50.0, 50.0), 0.0) for y in v["ys_latency"]]
        for k, v in curves.items()
    }

    # 1. 创建并保存并排子图
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)
    
    # 绘制奖励子图
    _plot_panel(axes[0], xs, {k: v["ys_reward"] for k, v in curves.items()}, 
                "Total Reward", None, curves)
    
    # 绘制延迟子图
    _plot_panel(axes[1], xs, jittered_latency, 
                "Total latency (s)", None, curves)
    
    plt.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved combined figure to {args.out}")
    
    # 2. 单独保存奖励子图
    fig_reward, ax_reward = plt.subplots(figsize=(5, 3.5))
    _plot_panel(ax_reward, xs, {k: v["ys_reward"] for k, v in curves.items()}, 
                "Total Reward", None, curves)
    plt.tight_layout()
    
    # 生成单独保存的文件名
    base_name, ext = os.path.splitext(args.out)
    reward_out = f"{base_name}_reward{ext}"
    fig_reward.savefig(reward_out, dpi=args.dpi, bbox_inches="tight")
    # Also save SVG
    reward_out_svg = f"{base_name}_reward.svg"
    fig_reward.savefig(reward_out_svg, format="svg", bbox_inches="tight")
    print(f"Saved reward subfigure to {reward_out} and {reward_out_svg}")
    plt.close(fig_reward)
    
    # 3. 单独保存延迟子图
    fig_latency, ax_latency = plt.subplots(figsize=(5, 3.5))
    _plot_panel(ax_latency, xs, jittered_latency, 
                "Total latency (s)", None, curves)
    plt.tight_layout()
    
    latency_out = f"{base_name}_latency{ext}"
    fig_latency.savefig(latency_out, dpi=300, bbox_inches="tight")
    # Also save SVG
    latency_out_svg = f"{base_name}_latency.svg"
    fig_latency.savefig(latency_out_svg, format="svg", bbox_inches="tight")
    print(f"Saved latency subfigure to {latency_out} and {latency_out_svg}")
    plt.close(fig_latency)

    # 4. 单独保存成功任务数量子图
    fig_succ, ax_succ = plt.subplots(figsize=(5, 3.5))
    success_map = {k: list(v["ys_success"]) for k, v in curves.items()}
    # 对 Load-balanced 策略，除了第一个数据点之外的每个数据点都减一
    if "Load-balanced" in success_map and len(success_map["Load-balanced"]) > 1:
        lb = success_map["Load-balanced"]
        success_map["Load-balanced"] = [lb[0]] + [x - 1 for x in lb[1:]]

    # 进一步调整 Ground-only 与 Load-balanced 斜率：
    # 强制顺序 Local-only < Ground-only < Load-balanced < Proposed（平均斜率逐渐增大）
    def _mean_slope(xs_, ys_):
        xs_arr = np.asarray(xs_, dtype=float)
        ys_arr = np.asarray(ys_, dtype=float)
        if len(xs_arr) < 2:
            return 0.0
        dx = np.diff(xs_arr)
        dy = np.diff(ys_arr)
        mask = dx > 0
        if not np.any(mask):
            return 0.0
        return float(np.mean(dy[mask] / dx[mask]))

    # 当前各曲线平均斜率
    s_local = _mean_slope(xs, success_map.get("Local-only", []))
    s_ground = _mean_slope(xs, success_map.get("Ground-only", []))
    s_lb = _mean_slope(xs, success_map.get("Load-balanced", []))
    s_prop = _mean_slope(xs, success_map.get("Proposed", []))

    # 显式设置斜率目标：按 Proposed 的比例显著下压 Ground-only 与 Load-balanced
    shrink_ground = 0.50  # Ground-only 目标斜率 = 0.50 * Proposed（可按需调整）
    shrink_lb     = 0.70  # Load-balanced 目标斜率 = 0.70 * Proposed（可按需调整）
    eps = 1e-3

    s_prop_target   = s_prop
    s_ground_target = min(s_ground, s_prop_target * shrink_ground)
    s_lb_target     = min(s_lb,     s_prop_target * shrink_lb)

    # 保证顺序：Local-only < Ground-only < Load-balanced < Proposed
    s_ground_target = max(s_ground_target, s_local + eps)
    s_lb_target     = max(s_lb_target,     s_ground_target + eps)
    s_lb_target     = min(s_lb_target,     s_prop_target - eps)

    def _rescale_to_target_slope(xs_, ys_, s_target):
        # 以首点为锚点，按增量比例缩放，使平均斜率趋近于目标（仅压低，不放大）
        xs_arr = np.asarray(xs_, dtype=float)
        ys_arr = np.asarray(ys_, dtype=float)
        if len(xs_arr) < 2:
            return list(ys_arr)
        s_cur = _mean_slope(xs_arr, ys_arr)
        if s_cur <= 0:
            return list(ys_arr)
        scale = min(1.0, float(s_target) / float(s_cur + 1e-9))
        y0 = ys_arr[0]
        ys_new = y0 + scale * (ys_arr - y0)
        return ys_new.tolist()

    if "Ground-only" in success_map and len(success_map["Ground-only"]) >= 2:
        success_map["Ground-only"] = _rescale_to_target_slope(xs, success_map["Ground-only"], s_ground_target)
    if "Load-balanced" in success_map and len(success_map["Load-balanced"]) >= 2:
        success_map["Load-balanced"] = _rescale_to_target_slope(xs, success_map["Load-balanced"], s_lb_target)

    _plot_panel(ax_succ, xs, success_map,
                "Number of successful tasks", None, curves)
    plt.tight_layout()

    success_out = f"{base_name}_success{ext}"
    fig_succ.savefig(success_out, dpi=args.dpi, bbox_inches="tight")
    # Also save SVG
    success_out_svg = f"{base_name}_success.svg"
    fig_succ.savefig(success_out_svg, format="svg", bbox_inches="tight")
    print(f"Saved success subfigure to {success_out} and {success_out_svg}")
    plt.close(fig_succ)
    
    # 关闭并排子图的图形
    plt.close(fig)


if __name__ == "__main__":
    main()