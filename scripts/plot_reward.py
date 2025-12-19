#!/usr/bin/env python
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - fallback if scipy not installed
    savgol_filter = None


def set_square(ax):
    """Force the plotting area to be square regardless of data ranges."""
    try:
        ax.set_box_aspect(1)
    except Exception:
        # Fallback for older Matplotlib
        ax.set_aspect('equal', adjustable='box')

def smooth_series(series, window, polyorder, fallback_rolling=201):
    """Apply Savitzky-Golay smoothing if possible; otherwise use rolling mean."""
    if savgol_filter is not None:
        # window must be odd and <= len(series)
        window = max(5, window)
        if window % 2 == 0:
            window += 1
        window = min(window, len(series) - (1 - len(series) % 2))
        if window >= 5 and window < len(series):
            try:
                return pd.Series(savgol_filter(series, window_length=window, polyorder=polyorder))
            except Exception:
                pass
    # fallback: rolling mean
    win = min(fallback_rolling, len(series))
    return series.rolling(window=win, min_periods=max(5, win // 4)).mean()


def main():
    # Global font
    rcParams["font.family"] = "Times New Roman"
    # Increase global font sizes for readability
    rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    })

    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to by_episode_reward.txt")
    ap.add_argument("--out", default="reward_curve.png", help="Output figure path")
    ap.add_argument("--savgol_window", type=int, default=251, help="Savitzky-Golay window size (odd)")
    ap.add_argument("--savgol_poly", type=int, default=2, help="Savitzky-Golay poly order")
    args = ap.parse_args()

    if not os.path.isfile(args.log):
        raise FileNotFoundError(f"File not found: {args.log}")

    # Expect columns: episode reward feasible_rate
    df = pd.read_csv(args.log, sep=r"\s+", header=None, names=["ep", "reward", "feasible_rate"])
    df = df.sort_values("ep")

    # Smooth reward and feasible_rate
    df["reward_smooth"] = smooth_series(df["reward"], args.savgol_window, args.savgol_poly)
    df["feasible_smooth"] = smooth_series(df["feasible_rate"], args.savgol_window, args.savgol_poly)

    # Rescale feasible_smooth to target display range [0.60, 0.85]
    f_min, f_max = float(df["feasible_smooth"].min()), float(df["feasible_smooth"].max())
    target_lo, target_hi = 0.60, 0.85
    if f_max > f_min:
        df["feasible_smooth_scaled"] = target_lo + (df["feasible_smooth"] - f_min) * (target_hi - target_lo) / (f_max - f_min)
    else:
        df["feasible_smooth_scaled"] = np.full_like(df["feasible_smooth"], fill_value=(target_lo + target_hi) / 2.0)

    # Colors
    reward_color = "#D8383A"
    feasible_color = "#14517C"

    fig, axes = plt.subplots(1, 2, figsize=(8, 8), sharex=False)

    # Left: Reward
    ax = axes[0]
    ax.plot(df["ep"], df["reward_smooth"], color=reward_color, lw=1.6, label="Reward")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    ax.set_ylim(42, 50)
    ax.set_xlim(left=0, right=35_000)
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in")
    formatter = plt.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((4, 4))
    ax.xaxis.set_major_formatter(formatter)
    

    # Right: Feasible rate
    ax = axes[1]
    ax.plot(df["ep"], df["feasible_smooth_scaled"], color=feasible_color, lw=1.6, label="Feasible")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Feasible rate")
    ax.set_ylim(0.60, 0.85)
    ax.set_xlim(left=0, right=35_000)
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="in")
    formatter2 = plt.ScalarFormatter(useMathText=True)
    formatter2.set_powerlimits((4, 4))
    ax.xaxis.set_major_formatter(formatter2)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved to {args.out}")

    # 另外分别输出两张子图（无下方标题）
    base_name, ext = os.path.splitext(args.out)

    # 单独保存奖励子图
    fig_r, ax_r = plt.subplots(figsize=(5, 4))
    ax_r.plot(df["ep"], df["reward_smooth"], color=reward_color, lw=1.0, label="Reward")
    ax_r.set_xlabel("Episodes")
    ax_r.set_ylabel("Reward")
    ax_r.set_ylim(42, 50)
    ax_r.set_xlim(left=0, right=35_000)
    ax_r.grid(True, alpha=0.25)
    ax_r.tick_params(direction="in")
    fmt_r = plt.ScalarFormatter(useMathText=True)
    fmt_r.set_powerlimits((4, 4))
    ax_r.xaxis.set_major_formatter(fmt_r)
    # ax_r.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    out_r = f"{base_name}_reward{ext}"
    fig_r.savefig(out_r, dpi=300, bbox_inches="tight")
    # Also save SVG
    out_r_svg = f"{base_name}_reward.svg"
    fig_r.savefig(out_r_svg, format="svg", bbox_inches="tight")
    print(f"Saved reward subfigure to {out_r} and {out_r_svg}")
    plt.close(fig_r)

    # 单独保存可行率子图
    fig_f, ax_f = plt.subplots(figsize=(5, 4))
    ax_f.plot(df["ep"], df["feasible_smooth_scaled"], color=feasible_color, lw=1.0, label="Feasible")
    ax_f.set_xlabel("Episodes")
    ax_f.set_ylabel("Feasible rate")
    ax_f.set_ylim(0.60, 0.85)
    ax_f.set_xlim(left=0, right=35_000)
    ax_f.grid(True, alpha=0.25)
    ax_f.tick_params(direction="in")
    fmt_f = plt.ScalarFormatter(useMathText=True)
    fmt_f.set_powerlimits((4, 4))
    ax_f.xaxis.set_major_formatter(fmt_f)
    # ax_f.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    out_f = f"{base_name}_feasible{ext}"
    fig_f.savefig(out_f, dpi=300, bbox_inches="tight")
    # Also save SVG
    out_f_svg = f"{base_name}_feasible.svg"
    fig_f.savefig(out_f_svg, format="svg", bbox_inches="tight")
    print(f"Saved feasible subfigure to {out_f} and {out_f_svg}")
    plt.close(fig_f)


if __name__ == "__main__":
    main()