# -*- coding: utf-8 -*-
"""Analyze collected trajectories for Decision Transformer training.

This script provides comprehensive statistics and visualizations to assess
quality/diversity of collected trajectory data.

Enhancements in this project fork:
- Support loading multiple batch files via glob pattern (--data_pattern)
- Shape consistency checks across trajectories:
  * observation key shapes
  * action vector length/shape
  * num_destinations / max_k / calculated_ks length

Usage examples:
  # single file
  python scripts/analyze_trajectories.py --data_path data/dt_trajectories.pkl

  # glob pattern (recommended for batch files)
  python scripts/analyze_trajectories.py --data_pattern "D:/.../dt_trajectories_batch_00[1-4].pkl" --output_dir analysis_results/batch_1_4
"""

import os
import sys
import argparse
import pickle
from typing import List, Dict, Any, Tuple
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def _load_one(path: str) -> List[Dict[str, Any]]:
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_trajectories(data_path: str | None = None, data_pattern: str | None = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load trajectories from a pickle file or a glob pattern.

    Returns:
        trajectories: list of trajectories
        sources: list of source file paths used
    """
    if (data_path is None) == (data_pattern is None):
        raise ValueError("Provide exactly one of --data_path or --data_pattern")

    if data_path is not None:
        print(f"Loading trajectories from {data_path}...")
        trajectories = _load_one(data_path)
        print(f"✓ Loaded {len(trajectories)} trajectories")
        return trajectories, [data_path]

    # pattern
    print(f"Loading trajectories from glob pattern: {data_pattern}...")
    files = sorted(glob.glob(data_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {data_pattern}")

    trajectories: List[Dict[str, Any]] = []
    for p in files:
        try:
            batch = _load_one(p)
            trajectories.extend(batch)
            print(f"  ✓ Loaded {os.path.basename(p)}: {len(batch)} trajectories")
        except Exception as e:
            print(f"  ❌ Failed to load {p}: {e}")
            raise

    print(f"✓ Total trajectories loaded: {len(trajectories)} from {len(files)} files")
    return trajectories, files


def compute_basic_statistics(trajectories: List[Dict]) -> Dict[str, Any]:
    """Compute basic statistics about the dataset."""
    print("\n" + "="*70)
    print("BASIC STATISTICS")
    print("="*70)

    returns = [sum(t.get('rewards', [])) for t in trajectories]
    lengths = [len(t.get('rewards', [])) for t in trajectories]
    total_steps = int(sum(lengths))

    stats = {
        'num_trajectories': len(trajectories),
        'total_steps': total_steps,
        'returns': returns,
        'lengths': lengths,
    }

    print(f"Total trajectories: {len(trajectories):,}")
    print(f"Total steps: {total_steps:,}")
    print(f"Average episode length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Episode length range: [{np.min(lengths)}, {np.max(lengths)}]")

    print(f"\nReturn Statistics:")
    print(f"  Mean:   {np.mean(returns):8.2f}")
    print(f"  Median: {np.median(returns):8.2f}")
    print(f"  Std:    {np.std(returns):8.2f}")
    print(f"  Min:    {np.min(returns):8.2f}")
    print(f"  Max:    {np.max(returns):8.2f}")

    print(f"\nReturn Percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p:2d}%: {np.percentile(returns, p):8.2f}")

    return stats


def check_data_quality(stats: Dict[str, Any]) -> None:
    """Check data quality and provide recommendations."""
    print("\n" + "="*70)
    print("DATA QUALITY ASSESSMENT")
    print("="*70)

    returns = stats['returns']
    lengths = stats['lengths']

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)
    avg_length = np.mean(lengths)

    issues = []
    warnings = []
    good_signs = []

    if mean_return < 10:
        issues.append("Mean return is very low (<10).")
    elif mean_return < 30:
        warnings.append("Mean return is low (<30).")
    else:
        good_signs.append("Mean return is reasonable (>=30).")

    if std_return < 5:
        warnings.append("Return std is low (<5). Data may lack diversity.")
    elif std_return > 30:
        warnings.append("Return std is high (>30). Data may be unstable.")
    else:
        good_signs.append("Return std is in a reasonable range (5~30).")

    if max_return - min_return < 20:
        warnings.append("Return range is narrow. Data may lack diversity.")
    else:
        good_signs.append("Return range looks diverse.")

    high_quality_ratio = np.sum(np.array(returns) > mean_return + 0.5 * std_return) / max(len(returns), 1)
    low_quality_ratio = np.sum(np.array(returns) < mean_return - 0.5 * std_return) / max(len(returns), 1)

    if high_quality_ratio < 0.2:
        warnings.append(f"Only {high_quality_ratio*100:.1f}% high-quality trajectories.")
    else:
        good_signs.append(f"{high_quality_ratio*100:.1f}% high-quality trajectories.")

    if low_quality_ratio < 0.1:
        warnings.append(f"Only {low_quality_ratio*100:.1f}% low-quality trajectories.")
    else:
        good_signs.append(f"{low_quality_ratio*100:.1f}% low-quality trajectories.")

    if avg_length < 50:
        issues.append(f"Average episode length is very short ({avg_length:.1f}).")
    elif avg_length < 80:
        warnings.append(f"Average episode length is short ({avg_length:.1f}).")
    else:
        good_signs.append(f"Average episode length is good ({avg_length:.1f}).")

    if good_signs:
        print("\nGood Signs:")
        for s in good_signs:
            print(f"  ✓ {s}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ⚠️  {w}")

    if issues:
        print("\nCritical Issues:")
        for i in issues:
            print(f"  ❌ {i}")

    print("\n" + "-"*70)
    if not issues and len(warnings) <= 2:
        print("✅ OVERALL: Data quality looks GOOD.")
    elif not issues:
        print("⚠️  OVERALL: Data quality looks ACCEPTABLE.")
    else:
        print("❌ OVERALL: Data quality has ISSUES.")
    print("-"*70)


def analyze_observation_space(trajectories: List[Dict]) -> None:
    """Analyze observation space structure and check NaN/Inf."""
    print("\n" + "="*70)
    print("OBSERVATION SPACE ANALYSIS")
    print("="*70)

    first_obs = trajectories[0]['observations'][0]
    obs_keys = sorted(first_obs.keys())

    print(f"\nObservation keys: {len(obs_keys)}")
    for key in obs_keys:
        shape = first_obs[key].shape
        dtype = first_obs[key].dtype
        print(f"  {key:30s}: shape={str(shape):15s} dtype={dtype}")

    print("\nChecking for invalid values (NaN/Inf) on first 100 trajectories...")
    has_issues = False
    for key in obs_keys:
        all_values = []
        for traj in trajectories[:100]:
            for obs in traj.get('observations', []):
                all_values.append(obs[key].flatten())
        if not all_values:
            continue
        all_values = np.concatenate(all_values)

        num_nan = int(np.sum(np.isnan(all_values)))
        num_inf = int(np.sum(np.isinf(all_values)))

        if num_nan > 0 or num_inf > 0:
            print(f"  ⚠️  {key}: NaN={num_nan}, Inf={num_inf}")
            has_issues = True

    if not has_issues:
        print("  ✓ No invalid values found")


def analyze_action_space(trajectories: List[Dict]) -> None:
    """Analyze action space distribution."""
    print("\n" + "="*70)
    print("ACTION SPACE ANALYSIS")
    print("="*70)

    all_actions = []
    for traj in trajectories:
        all_actions.extend(traj.get('actions', []))

    first_action = all_actions[0]
    is_multidim = isinstance(first_action, (list, np.ndarray))

    if is_multidim:
        all_actions_array = np.array([np.array(a).reshape(-1) for a in all_actions], dtype=object)
        # length distribution
        lengths = [len(np.array(a).reshape(-1)) for a in all_actions]
        print(f"\nTotal actions: {len(lengths):,}")
        print(f"Action length distribution (top 10): {Counter(lengths).most_common(10)}")
    else:
        all_actions_array = np.array(all_actions)
        print(f"\nTotal actions: {len(all_actions_array):,}")
        print(f"Action range: [{np.min(all_actions_array)}, {np.max(all_actions_array)}]")
        print(f"Unique actions: {len(np.unique(all_actions_array))}")


def check_shape_consistency(trajectories: List[Dict], max_report: int = 20) -> None:
    """Check shapes/metadata consistency across trajectories.

    This is designed to catch the typical reason for DataLoader collate crashes:
    mixing trajectories generated under different env configs.
    """
    print("\n" + "="*70)
    print("SHAPE CONSISTENCY CHECK")
    print("="*70)

    # Counters
    obs_shape_counts: dict[str, Counter] = defaultdict(Counter)
    num_dest_counts: Counter = Counter()
    max_k_counts: Counter = Counter()
    action_len_counts: Counter = Counter()
    calc_ks_len_counts: Counter = Counter()

    offenders: List[str] = []

    for ti, traj in enumerate(trajectories):
        # metadata
        nd = traj.get('num_destinations', None)
        mk = traj.get('max_k', None)
        if nd is not None:
            try:
                num_dest_counts[int(nd)] += 1
            except Exception:
                num_dest_counts[str(nd)] += 1
        if mk is not None:
            try:
                max_k_counts[int(mk)] += 1
            except Exception:
                max_k_counts[str(mk)] += 1

        # action length (first action only)
        acts = traj.get('actions', [])
        if acts:
            try:
                a0 = np.array(acts[0]).reshape(-1)
                action_len_counts[int(len(a0))] += 1
            except Exception:
                action_len_counts['unreadable'] += 1

        # calculated_ks length
        cks = traj.get('calculated_ks', None)
        if cks is not None:
            try:
                calc_ks_len_counts[int(len(cks))] += 1
            except Exception:
                calc_ks_len_counts['unreadable'] += 1

        # observation shapes (first obs only)
        obs_list = traj.get('observations', [])
        if obs_list:
            obs0 = obs_list[0]
            if isinstance(obs0, dict):
                for k, v in obs0.items():
                    if isinstance(v, np.ndarray):
                        obs_shape_counts[k][tuple(v.shape)] += 1
                    else:
                        obs_shape_counts[k][type(v).__name__] += 1

    print("\nnum_destinations distribution:")
    print(f"  {num_dest_counts.most_common(10)}")

    print("\nmax_k distribution:")
    print(f"  {max_k_counts.most_common(10)}")

    print("\nfirst-action length distribution:")
    print(f"  {action_len_counts.most_common(10)}")

    print("\ncalculated_ks length distribution:")
    print(f"  {calc_ks_len_counts.most_common(10)}")

    print("\nObservation shape distributions (top 5 per key):")
    for k in sorted(obs_shape_counts.keys()):
        top = obs_shape_counts[k].most_common(5)
        print(f"  {k:30s}: {top}")

    # Identify keys with multiple shapes
    problematic_keys = [k for k, c in obs_shape_counts.items() if len(c) > 1]
    problematic_meta = []
    if len(num_dest_counts) > 1:
        problematic_meta.append('num_destinations')
    if len(action_len_counts) > 1:
        problematic_meta.append('action_length')

    print("\nSummary:")
    if not problematic_keys and not problematic_meta:
        print("  ✓ No obvious shape/metadata inconsistencies detected.")
    else:
        if problematic_meta:
            print(f"  ⚠️  Metadata inconsistencies: {problematic_meta}")
        if problematic_keys:
            print(f"  ⚠️  Observation keys with inconsistent shapes: {problematic_keys}")

        # Report a few offending trajectories by scanning again
        reported = 0
        ref_obs_shapes = {k: obs_shape_counts[k].most_common(1)[0][0] for k in obs_shape_counts}
        ref_nd = num_dest_counts.most_common(1)[0][0] if len(num_dest_counts) > 0 else None
        ref_action_len = action_len_counts.most_common(1)[0][0] if len(action_len_counts) > 0 else None

        for ti, traj in enumerate(trajectories):
            if reported >= max_report:
                break
            bad = False

            # meta
            nd = traj.get('num_destinations', None)
            try:
                ndv = int(nd) if nd is not None else None
            except Exception:
                ndv = None
            if ref_nd is not None and ndv is not None and ndv != ref_nd:
                bad = True

            acts = traj.get('actions', [])
            a_len = None
            if acts:
                try:
                    a_len = int(len(np.array(acts[0]).reshape(-1)))
                except Exception:
                    pass
            if ref_action_len is not None and a_len is not None and a_len != ref_action_len:
                bad = True

            obs_list = traj.get('observations', [])
            if obs_list:
                obs0 = obs_list[0]
                if isinstance(obs0, dict):
                    for k, v in obs0.items():
                        if isinstance(v, np.ndarray):
                            if k in ref_obs_shapes and tuple(v.shape) != ref_obs_shapes[k]:
                                bad = True

            if bad:
                print(f"  Example inconsistent traj idx={ti}: num_destinations={traj.get('num_destinations')}, max_k={traj.get('max_k')}")
                if a_len is not None:
                    print(f"    first action length={a_len}")
                if obs_list and isinstance(obs_list[0], dict):
                    print("    obs0 shapes:")
                    for k, v in obs_list[0].items():
                        if isinstance(v, np.ndarray):
                            print(f"      {k}: {tuple(v.shape)}")
                reported += 1


def create_visualizations(stats: Dict[str, Any], output_dir: str) -> None:
    """Create comprehensive visualizations."""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)

    returns = np.array(stats['returns'])
    lengths = np.array(stats['lengths'])

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(returns):.2f}")
    ax1.axvline(np.median(returns), color='green', linestyle='--', linewidth=2, label=f"Median: {np.median(returns):.2f}")
    ax1.set_xlabel('Episode Return')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Episode Returns')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    bp = ax2.boxplot(returns, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    ax2.set_ylabel('Episode Return')
    ax2.set_title('Return Box Plot')
    ax2.grid(alpha=0.3, axis='y')

    ax3 = plt.subplot(2, 3, 3)
    sorted_returns = np.sort(returns)
    cumulative = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    ax3.plot(sorted_returns, cumulative, linewidth=2, color='darkblue')
    ax3.set_xlabel('Episode Return')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Function')
    ax3.grid(alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(lengths, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax4.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(lengths):.1f}")
    ax4.set_xlabel('Episode Length')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Episode Lengths')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(lengths, returns, alpha=0.5, c=returns, cmap='viridis', s=20)
    ax5.set_xlabel('Episode Length')
    ax5.set_ylabel('Episode Return')
    ax5.set_title('Return vs Episode Length')
    plt.colorbar(scatter, ax=ax5, label='Return')
    ax5.grid(alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    high_quality = np.sum(returns > mean_return + 0.5 * std_return)
    medium_quality = np.sum((returns >= mean_return - 0.5 * std_return) & (returns <= mean_return + 0.5 * std_return))
    low_quality = np.sum(returns < mean_return - 0.5 * std_return)
    sizes = [high_quality, medium_quality, low_quality]
    labels = [
        f'High\n({high_quality}, {high_quality/len(returns)*100:.1f}%)',
        f'Mid\n({medium_quality}, {medium_quality/len(returns)*100:.1f}%)',
        f'Low\n({low_quality}, {low_quality/len(returns)*100:.1f}%)'
    ]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax6.set_title('Trajectory Quality Distribution')

    plt.tight_layout()

    out = os.path.join(output_dir, 'trajectory_analysis.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {out}")
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory data quality")
    parser.add_argument("--data_path", type=str, default=None, help="Path to trajectory pickle file")
    parser.add_argument("--data_pattern", type=str, default=None, help="Glob pattern for batch pickle files")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Output directory for visualizations")
    parser.add_argument("--max_report", type=int, default=20, help="Max number of inconsistent trajectories to print")

    args = parser.parse_args()

    trajectories, sources = load_trajectories(data_path=args.data_path, data_pattern=args.data_pattern)

    stats = compute_basic_statistics(trajectories)
    check_data_quality(stats)
    analyze_observation_space(trajectories)
    analyze_action_space(trajectories)
    check_shape_consistency(trajectories, max_report=int(args.max_report))
    create_visualizations(stats, args.output_dir)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Sources: {len(sources)} files")
    for p in sources[:10]:
        print(f"  - {p}")
    if len(sources) > 10:
        print(f"  ... ({len(sources)-10} more)")
    print(f"Results saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
