# -*- coding: utf-8 -*-
"""
Training utilities for building environments, vectorization, and logging dirs.

This module centralizes:
- Environment factory with TLE preprocessing
- Optional Dict->MultiDiscrete action flattening wrapper
- Observation sanitizer to avoid NaNs/Infs reaching the policy
- Creation of vectorized envs for SB3
- Standard callbacks (Eval + Checkpoint)
- Run directory scaffolding for models/logs
"""
from __future__ import annotations

import os
import sys
import json
import time
from typing import Callable, Dict, Any, List, Optional

import numpy as np
import gymnasium as gym

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
import json as _json
from datetime import datetime as _dt

# Resolve project root and import local modules
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.append(_project_root)

from src.utils.config_loader import load_config
from src.utils.tle_loader import preprocess_satellite_configs
from src.environment.satellite_env import SatelliteEnv

class ObservationScaler(gym.ObservationWrapper):
    """Scale observations to comparable ranges.
    Injects raw (unscaled) observation into info['_raw_obs'] for logging.
    Constants can be provided via constructor; otherwise defaults used.
    """
    def __init__(self, env: gym.Env, *,
                 enabled: bool = True,
                 Re: float = 6_371_000.0,
                 Wmax: float = 6000.0,
                 Hmax: float = 6000.0,
                 LatMax: float = 500.0,
                 BitsTot: float = (6000.0*6000.0*24.0),
                 FLOPsRef: float = 1.39e12,
                 QueueRef: float = 1e12,
                 RateRef: float = 1e9,
                 clip_pos: float = 2.0,
                 clip_queue: float = 10.0) -> None:
        super().__init__(env)
        self.enabled = bool(enabled)
        self.Re = float(Re)
        self.Wmax = float(Wmax)
        self.Hmax = float(Hmax)
        self.LatMax = float(LatMax)
        self.BitsTot = float(BitsTot)
        self.FLOPsRef = float(FLOPsRef)
        self.QueueRef = float(QueueRef)
        self.RateRef = float(RateRef)
        self.clip_pos = float(clip_pos)
        self.clip_queue = float(clip_queue)
        self._last_raw_obs = None

    def _scale_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.enabled or obs is None:
            return obs
        o = dict(obs)
        # positions
        if 'task_origin_pos' in o:
            o['task_origin_pos'] = np.clip(o['task_origin_pos'] / self.Re, -self.clip_pos, self.clip_pos)
        if 'leader_pos' in o:
            o['leader_pos'] = np.clip(o['leader_pos'] / self.Re, -self.clip_pos, self.clip_pos)
        if 'compute_pos' in o:
            o['compute_pos'] = np.clip(o['compute_pos'] / self.Re, -self.clip_pos, self.clip_pos)
        if 'ground_station_pos' in o:
            o['ground_station_pos'] = np.clip(o['ground_station_pos'] / self.Re, -self.clip_pos, self.clip_pos)
        # queues
        if 'compute_queues' in o:
            o['compute_queues'] = np.clip(o['compute_queues'] / self.QueueRef, 0.0, self.clip_queue)
        if 'ground_station_queue' in o:
            o['ground_station_queue'] = np.clip(o['ground_station_queue'] / self.QueueRef, 0.0, self.clip_queue)
        # task_info
        if 'task_info' in o and isinstance(o['task_info'], np.ndarray):
            ti = o['task_info'].astype(np.float32)
            # expect [W,H,max_lat,data_bits,required_flops,ratio,rate]
            if ti.shape[0] >= 1:
                ti[0] = ti[0] / self.Wmax
            if ti.shape[0] >= 2:
                ti[1] = ti[1] / self.Hmax
            if ti.shape[0] >= 3:
                ti[2] = ti[2] / self.LatMax
            if ti.shape[0] >= 4:
                ti[3] = ti[3] / self.BitsTot
            if ti.shape[0] >= 5:
                ti[4] = ti[4] / self.FLOPsRef
            if ti.shape[0] >= 6:
                # ratio_sec_per_bit，常见取值很小，这里乘 RateRef 让量级接近 1
                ti[6-1] = np.clip(ti[6-1] * self.RateRef, 0.0, 10.0)
            if ti.shape[0] >= 7:
                # required_rate_bps
                ti[7-1] = np.clip(ti[7-1] / self.RateRef, 0.0, 10.0)
            o['task_info'] = ti
        return o

    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self._last_raw_obs = observation
        return self._scale_obs(observation)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = obs
        scaled = self._scale_obs(raw)
        # 将未归一化观测注入 info，供日志使用
        try:
            if isinstance(info, dict):
                info.setdefault('_raw_obs', raw)
            # 对 VecEnv 场景，Monitor/VecEnv 会聚合为 list[dict]，每个 env 自己的 wrapper 会注入
        except Exception:
            pass
        return scaled, reward, terminated, truncated, info

from src.training.action_wrappers import FlattenedDictActionWrapper
# HRL wrappers (optional)
try:
    from src.training.hrl_wrappers import MacroController, MacroFlatOverrideWrapper, GoalInjectionObsWrapper
except Exception:
    MacroController = None
    MacroFlatOverrideWrapper = None
    GoalInjectionObsWrapper = None




class InfoScalarCallback(BaseCallback):
    """Log episode-level custom info (from Monitor info_keywords) to TensorBoard.

    It aggregates the values stored in self.model.ep_info_buffer (a deque of
    dicts created by Monitor at episode end) and writes their mean to TB.

    Args:
        keys: list of info keys to aggregate, must be present in Monitor's
              info_keywords and in env's 'episode' info when done.
        log_every: log frequency in environment steps.
    """
    def __init__(self, keys: List[str], log_every: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.keys = keys
        self.log_every = log_every
        self._last_log_step = 0

    def _on_step(self) -> bool:
        # Throttle logging frequency
        if (self.num_timesteps - self._last_log_step) < self.log_every:
            return True
        self._last_log_step = self.num_timesteps

        buf = getattr(self.model, 'ep_info_buffer', None)
        if not buf or len(buf) == 0:
            return True

        # Compute means per key over the buffer
        any_recorded = False
        for k in self.keys:
            vals = [float(info[k]) for info in buf if isinstance(info, dict) and (k in info)]
            if len(vals) > 0:
                self.logger.record(f'custom/{k}_mean', float(np.mean(vals)))
                any_recorded = True

        # Force immediate write to TensorBoard to avoid long buffering
        if any_recorded and hasattr(self.logger, 'dump'):
            try:
                self.logger.dump(self.num_timesteps)
            except Exception:
                pass
        return True


from collections import deque

class StepRewardMovingAverageCallback(BaseCallback):
    """Log moving average of per-step reward over a sliding window without
    waiting for episode to finish. Works with VecEnv (averages over envs per step).
    Also logs an exponential moving average (EMA) to show smooth convergence.
    """
    def __init__(self, window_size: int = 1000, log_every: int = 200, ema_alpha: float = 0.1, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = int(max(1, window_size))
        self.log_every = int(max(1, log_every))
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.buf: deque[float] = deque(maxlen=self.window_size)
        self._since_last = 0
        self._ema: float | None = None

    def _on_step(self) -> bool:
        rewards = None
        try:
            rewards = self.locals.get('rewards', None)
        except Exception:
            rewards = None
        if rewards is not None:
            try:
                step_mean = float(np.mean(rewards))
                self.buf.append(step_mean)
                # update EMA every step
                if self._ema is None:
                    self._ema = step_mean
                else:
                    self._ema = self.ema_alpha * step_mean + (1.0 - self.ema_alpha) * self._ema

                self._since_last += 1
                if self._since_last >= self.log_every and len(self.buf) > 0:
                    ma = float(np.mean(self.buf))
                    self.logger.record('train/step_reward_inst', step_mean)
                    self.logger.record('train/step_reward_ma', ma)
                    if self._ema is not None:
                        self.logger.record('train/step_reward_ema', float(self._ema))
                    # dump at current timesteps so x-axis remains timesteps
                    if hasattr(self.logger, 'dump'):
                        try:
                            self.logger.dump(self.num_timesteps)
                        except Exception:
                            pass
                    self._since_last = 0
            except Exception:
                pass
        return True


class EpisodeRewardCallback(BaseCallback):
    """Log episode reward using episode index as the TensorBoard step.
    Also compute per-episode feasible rate (= feasible_steps / episode_steps)
    and append multiple columns to a txt file for comprehensive tracking.
    
    TXT file format: episode reward feasible_rate success_rate mean_miou mean_latency episode_length
    """
    def __init__(self, log_ep_length: bool = True, prefix: str = 'by_episode', txt_file_path: str | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.log_ep_length = log_ep_length
        self.prefix = prefix
        self.episode_idx = 0
        self.txt_file_path = txt_file_path
        self._txt_file = None
        # Per-env counters for feasible rate
        self._n_envs: int | None = None
        self._feasible_counts: list[int] | None = None
        self._step_counts: list[int] | None = None
        # Per-env accumulators for episode statistics
        self._success_counts: list[int] | None = None
        self._miou_sums: list[float] | None = None
        self._latency_sums: list[float] | None = None
        self._miou_counts: list[int] | None = None
        self._latency_counts: list[int] | None = None
        # HRL manager-specific accumulators (optional)
        self._feasible_inwin_sums: list[int] | None = None
        self._macro_h_sums: list[int] | None = None
        if self.txt_file_path:
            try:
                os.makedirs(os.path.dirname(self.txt_file_path), exist_ok=True)
                # append mode: keep history if continuing the same run
                self._txt_file = open(self.txt_file_path, 'a', encoding='utf-8')
                # Write header if file is new/empty
                if os.path.getsize(self.txt_file_path) == 0:
                    self._txt_file.write("# episode reward feasible_rate success_rate mean_miou mean_latency episode_length\n")
                    self._txt_file.flush()
            except Exception:
                self._txt_file = None

    def _on_training_start(self) -> None:
        # Initialize per-env counters
        try:
            env = getattr(self, 'training_env', None) or getattr(self.model, 'env', None)
            if env is not None and hasattr(env, 'num_envs'):
                self._n_envs = int(env.num_envs)
            else:
                self._n_envs = 1
        except Exception:
            self._n_envs = 1
        self._feasible_counts = [0 for _ in range(self._n_envs)]
        self._step_counts = [0 for _ in range(self._n_envs)]
        self._success_counts = [0 for _ in range(self._n_envs)]
        self._miou_sums = [0.0 for _ in range(self._n_envs)]
        self._latency_sums = [0.0 for _ in range(self._n_envs)]
        self._miou_counts = [0 for _ in range(self._n_envs)]
        self._latency_counts = [0 for _ in range(self._n_envs)]

    def _on_step(self) -> bool:
        infos = None
        try:
            infos = self.locals.get('infos', None)
        except Exception:
            infos = None
        ended_stats = []  # collect stats for all envs that ended this global step
        if infos is not None:
            # Accumulate per-env feasible/step counts and other metrics
            for i, info in enumerate(infos):
                if isinstance(info, dict) and i < self._n_envs:
                    # step count +1 for this env index
                    if self._step_counts is not None and i < len(self._step_counts):
                        self._step_counts[i] += 1
                    
                    # feasible flag or ratio
                    try:
                        if self._feasible_counts is not None:
                            if 'feasible_ratio' in info:
                                self._feasible_counts[i] += float(info.get('feasible_ratio', 0.0))
                            elif bool(info.get('feasible', False)):
                                self._feasible_counts[i] += 1
                            # success count kept for compatibility with boolean feasible
                            if self._success_counts is not None and bool(info.get('feasible', False)):
                                self._success_counts[i] += 1
                    except Exception:
                        pass
                    
                    # Accumulate mIoU
                    try:
                        miou = info.get('miou', None)
                        if miou is not None and self._miou_sums is not None and self._miou_counts is not None:
                            self._miou_sums[i] += float(miou)
                            self._miou_counts[i] += 1
                    except Exception:
                        pass
                    
                    # Accumulate total latency
                    try:
                        latency = info.get('total_latency', None)
                        if latency is not None and self._latency_sums is not None and self._latency_counts is not None:
                            self._latency_sums[i] += float(latency)
                            self._latency_counts[i] += 1
                    except Exception:
                        pass
                
                # Episode termination for this env
                ep_info = info.get('episode') if isinstance(info, dict) else None
                if ep_info is not None:
                    self.episode_idx += 1
                    r = float(ep_info.get('r', 0.0))
                    l = float(ep_info.get('l', 0.0)) if self.log_ep_length else 0.0
                    
                    # Compute episode statistics for this env
                    # Prefer metrics directly provided by ep_info (if available)
                    feasible_rate = float(ep_info.get('feasible_rate', 0.0))
                    success_rate = float(ep_info.get('success_rate', 0.0))
                    mean_miou = float(ep_info.get('mean_miou', 0.0))
                    mean_latency = float(ep_info.get('mean_latency', 0.0))
                    
                    # Fallback to self-computed metrics when ep_info does not contain them
                    if (feasible_rate == 0.0 and 'feasible_rate' not in ep_info) or (
                        success_rate == 0.0 and 'success_rate' not in ep_info):
                        if i < self._n_envs:
                            steps = max(1, self._step_counts[i])
                            feasible_rate = float(self._feasible_counts[i]) / float(steps)
                            success_rate = float(self._success_counts[i]) / float(steps)
                            if self._miou_counts is not None and self._miou_counts[i] > 0:
                                mean_miou = self._miou_sums[i] / self._miou_counts[i]
                            if self._latency_counts is not None and self._latency_counts[i] > 0:
                                mean_latency = self._latency_sums[i] / self._latency_counts[i]
                        
                        # Reset counters for this env
                        if self._feasible_counts is not None:
                            self._feasible_counts[i] = 0
                        if self._step_counts is not None:
                            self._step_counts[i] = 0
                        if self._success_counts is not None:
                            self._success_counts[i] = 0
                        if self._miou_sums is not None:
                            self._miou_sums[i] = 0.0
                        if self._latency_sums is not None:
                            self._latency_sums[i] = 0.0
                        if self._miou_counts is not None:
                            self._miou_counts[i] = 0
                        if self._latency_counts is not None:
                            self._latency_counts[i] = 0
                    
                    # Log to TensorBoard
                    self.logger.record(f'{self.prefix}/episode_reward', r)
                    if self.log_ep_length:
                        self.logger.record(f'{self.prefix}/episode_length', l)
                    self.logger.record(f'{self.prefix}/feasible_rate', feasible_rate)
                    self.logger.record(f'{self.prefix}/success_rate', success_rate)
                    self.logger.record(f'{self.prefix}/mean_miou', mean_miou)
                    self.logger.record(f'{self.prefix}/mean_latency', mean_latency)
                    
                    # dump with episode index as x-axis
                    if hasattr(self.logger, 'dump'):
                        try:
                            self.logger.dump(self.episode_idx)
                        except Exception:
                            pass
                    
                    # Write to txt file: episode reward feasible_rate success_rate mean_miou mean_latency episode_length
                    if self._txt_file is not None:
                        try:
                            self._txt_file.write(f"{self.episode_idx} {r:.6f} {feasible_rate:.6f} {success_rate:.6f} {mean_miou:.6f} {mean_latency:.6f} {int(l)}\n")
                            self._txt_file.flush()
                        except Exception:
                            pass
        return True

    def _on_training_end(self) -> None:
        try:
            if self._txt_file is not None:
                self._txt_file.flush()
                self._txt_file.close()
        except Exception:
            pass


class A2CTrainingMetricsCallback(BaseCallback):
    """Log A2C-specific training metrics to TensorBoard and optionally to a txt file.
    
    Records policy loss, value loss, entropy loss, explained variance,
    learning rate, and other A2C-specific metrics that are crucial for
    monitoring convergence.
    
    Note: SB3's A2C already logs these to TensorBoard under 'train/*' prefix.
    This callback provides additional functionality:
    1. Logs to a separate txt file for easy analysis
    2. Creates duplicate metrics under 'a2c/*' prefix for better organization
    """
    def __init__(self, log_every: int = 100, txt_file_path: str | None = None, tb_prefix: str = 'a2c', verbose: int = 0):
        super().__init__(verbose)
        self.log_every = int(max(1, log_every))
        self._last_log_step = 0
        self.txt_file_path = txt_file_path
        self._txt_file = None
        self.tb_prefix = str(tb_prefix)
        
        # Track metrics for txt logging
        self._metrics_buffer = []
        
        if self.txt_file_path:
            try:
                os.makedirs(os.path.dirname(self.txt_file_path), exist_ok=True)
                self._txt_file = open(self.txt_file_path, 'a', encoding='utf-8')
                # Write header if file is new/empty
                if os.path.getsize(self.txt_file_path) == 0:
                    self._txt_file.write("# timesteps n_updates policy_loss value_loss entropy_loss explained_variance learning_rate total_loss\n")
                    self._txt_file.flush()
            except Exception as e:
                if self.verbose > 0:
                    print(f"[A2CTrainingMetricsCallback] Failed to open txt file: {e}")
                self._txt_file = None

    def _on_step(self) -> bool:
        # Note: SB3's A2C logs metrics after each rollout (every n_steps), not every env step
        # We check if new metrics are available in the logger
        return True

    def _on_rollout_end(self) -> None:
        """Called after each rollout (every n_steps). This is when A2C updates and logs metrics."""
        try:
            # Access the logger's name_to_value dict which contains recently logged metrics
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                logger = self.model.logger
                
                if hasattr(logger, 'name_to_value') and len(logger.name_to_value) > 0:
                    name_to_value = logger.name_to_value
                    
                    # Extract A2C metrics
                    metrics = {
                        'timesteps': self.num_timesteps,
                        'n_updates': name_to_value.get('train/n_updates', None),
                        'policy_loss': name_to_value.get('train/policy_loss', None),
                        'value_loss': name_to_value.get('train/value_loss', None),
                        'entropy_loss': name_to_value.get('train/entropy_loss', None),
                        'explained_variance': name_to_value.get('train/explained_variance', None),
                        'learning_rate': name_to_value.get('train/learning_rate', None),
                        'total_loss': name_to_value.get('train/loss', None),
                    }
                    
                    # Log to TensorBoard under custom prefix for better organization (e.g., worker/* or manager/*)
                    for key, value in metrics.items():
                        if value is not None and key != 'timesteps':
                            self.logger.record(f'{self.tb_prefix}/{key}', float(value))
                    
                    # Write to txt file
                    if self._txt_file is not None:
                        try:
                            line_parts = []
                            for key in ['timesteps', 'n_updates', 'policy_loss', 'value_loss', 
                                       'entropy_loss', 'explained_variance', 'learning_rate', 'total_loss']:
                                val = metrics.get(key, None)
                                if val is not None:
                                    if key == 'timesteps' or key == 'n_updates':
                                        line_parts.append(f"{int(val)}")
                                    else:
                                        line_parts.append(f"{float(val):.6f}")
                                else:
                                    line_parts.append("NA")
                            
                            self._txt_file.write(" ".join(line_parts) + "\n")
                            self._txt_file.flush()
                        except Exception as e:
                            if self.verbose > 0:
                                print(f"[A2CTrainingMetricsCallback] Error writing to txt: {e}")
                    
                    if self.verbose > 0:
                        print(f"[A2CTrainingMetricsCallback] Logged metrics at timestep {self.num_timesteps}")
                        
        except Exception as e:
            if self.verbose > 0:
                print(f"[A2CTrainingMetricsCallback] Error in _on_rollout_end: {e}")

    def _on_training_end(self) -> None:
        """Close txt file when training ends."""
        try:
            if self._txt_file is not None:
                self._txt_file.flush()
                self._txt_file.close()
        except Exception:
            pass


class EarlyStopOnEvalReward(BaseCallback):
    """Early stop when eval mean reward does not improve for N evals.

    Args:
        eval_cb: the EvalCallback instance to read results from
        patience_evals: stop after this many consecutive non-improving evals
        min_delta: absolute minimum improvement to count as better
        min_delta_rel: relative improvement (fraction of best) to count as better
    """
    def __init__(self, eval_cb: EvalCallback, patience_evals: int = 10,
                 min_delta: float = 0.0, min_delta_rel: float = 0.01, verbose: int = 1):
        super().__init__(verbose)
        self.eval_cb = eval_cb
        self.patience = max(1, int(patience_evals))
        self.min_delta = float(min_delta)
        self.min_delta_rel = float(min_delta_rel)
        self.no_improve = 0
        self._last_seen_eval_calls = 0
        self._best = -np.inf

    def _on_training_start(self) -> None:
        try:
            self._best = float(getattr(self.eval_cb, "best_mean_reward", -np.inf))
            self._last_seen_eval_calls = int(getattr(self.eval_cb, "n_eval_calls", 0))
        except Exception:
            pass

    def _on_step(self) -> bool:
        # Only act right after a new eval has been performed
        try:
            n_calls = int(getattr(self.eval_cb, "n_eval_calls", 0))
        except Exception:
            return True
        if n_calls == self._last_seen_eval_calls:
            return True
        self._last_seen_eval_calls = n_calls

        last = getattr(self.eval_cb, "last_mean_reward", None)
        if last is None:
            return True
        last = float(last)
        best = float(self._best)

        improved = (last > best + self.min_delta) or (
            (best > -np.inf) and ((last - best) > abs(best) * self.min_delta_rel)
        )
        if improved:
            self._best = last
            self.no_improve = 0
            if self.verbose:
                print(f"[EarlyStop] New best eval mean_reward={last:.4f}")
        else:
            self.no_improve += 1
            if self.verbose:
                print(f"[EarlyStop] No improvement ({self.no_improve}/{self.patience}) last={last:.4f}, best={best:.4f}")
            if self.no_improve >= self.patience:
                if self.verbose:
                    print("[EarlyStop] Patience exhausted. Stopping training.")
                return False
        return True


class TrainingJSONLoggerCallback(BaseCallback):
    """Write detailed per-step logs for analysis.
    - Writes JSON lines into rotated files every `rotate_every_episodes` episodes.
    - Each line contains timesteps, episode index, reward, action, and `info` including latency_debug.
    """
    def __init__(self, logs_dir: str, rotate_every_episodes: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.logs_dir = logs_dir
        self.rotate_every_episodes = max(1, int(rotate_every_episodes))
        os.makedirs(self.logs_dir, exist_ok=True)
        self.episode_idx = 0
        self._file = None
        self._file_path = None

    def _open_file(self):
        start = (self.episode_idx // self.rotate_every_episodes) * self.rotate_every_episodes + 1
        end = start + self.rotate_every_episodes - 1
        fname = f"train_ep_{start:06d}-{end:06d}.jsonl"
        self._file_path = os.path.join(self.logs_dir, fname)
        # open append to continue across sessions in same range
        self._file = open(self._file_path, 'a', encoding='utf-8')

    def _close_file(self):
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def _maybe_rotate(self):
        if self._file is None:
            self._open_file()
            return
        start = (self.episode_idx // self.rotate_every_episodes) * self.rotate_every_episodes + 1
        expected_name = f"train_ep_{start:06d}-{start + self.rotate_every_episodes - 1:06d}.jsonl"
        if os.path.basename(self._file_path) != expected_name:
            self._close_file()
            self._open_file()

    def _on_step(self) -> bool:
        try:
            infos = self.locals.get('infos', None)
            actions = self.locals.get('actions', None)
            rewards = self.locals.get('rewards', None)
            # rotate when we pass episode boundary
            if infos is not None:
                for info in infos:
                    ep_info = info.get('episode') if isinstance(info, dict) else None
                    if ep_info is not None:
                        self.episode_idx += 1
                        self._maybe_rotate()
            # build log entry
            entry = {
                'time': _dt.utcnow().isoformat() + 'Z',
                'timesteps': int(self.num_timesteps),
                'episode_idx': int(self.episode_idx),
            }
            if rewards is not None:
                try:
                    entry['reward_mean'] = float(np.mean(rewards))
                except Exception:
                    pass
            if isinstance(actions, np.ndarray):
                entry['actions'] = actions.tolist()
            # we expect first info to have details
            if infos and len(infos) > 0 and isinstance(infos[0], dict):
                entry['info'] = infos[0]
            # write line
            self._maybe_rotate()
            if self._file is None:
                self._open_file()
            self._file.write(_json.dumps(entry, ensure_ascii=False) + "\n")
            self._file.flush()
        except Exception:
            pass
        return True

    def _on_training_end(self) -> None:
        self._close_file()


class PlainTextLoggerCallback(BaseCallback):
    """Human-readable plain text logger.
    Writes one line per step with raw (unscaled) observation task_info, action, and latency breakdown summary.
    Rotates every N episodes.
    """
    def __init__(self, logs_dir: str, rotate_every_episodes: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.logs_dir = logs_dir
        self.rotate_every_episodes = max(1, int(rotate_every_episodes))
        os.makedirs(self.logs_dir, exist_ok=True)
        self.episode_idx = 0
        self._file = None
        self._file_path = None

    def _open_file(self):
        start = (self.episode_idx // self.rotate_every_episodes) * self.rotate_every_episodes + 1
        end = start + self.rotate_every_episodes - 1
        fname = f"train_ep_{start:06d}-{end:06d}.log"
        self._file_path = os.path.join(self.logs_dir, fname)
        self._file = open(self._file_path, 'a', encoding='utf-8')

    def _close_file(self):
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def _maybe_rotate(self):
        if self._file is None:
            self._open_file()
            return
        start = (self.episode_idx // self.rotate_every_episodes) * self.rotate_every_episodes + 1
        expected = f"train_ep_{start:06d}-{start + self.rotate_every_episodes - 1:06d}.log"
        if os.path.basename(self._file_path) != expected:
            self._close_file()
            self._open_file()

    def _format_task_info_raw(self, raw_obs: Dict[str, Any]) -> str:
        try:
            ti = raw_obs.get('task_info', None)
            if isinstance(ti, np.ndarray) and ti.size >= 7:
                W, H, max_lat, bits, flops, ratio, rate = [float(x) for x in ti[:7]]
                return f"task_info_raw=[W={W:.0f},H={H:.0f},max_lat={max_lat:.3f},bits={bits:.0f},flops={flops:.2e},ratio={ratio:.3e},rate={rate:.3e}]"
        except Exception:
            pass
        return "task_info_raw=[]"

    def _format_latency_summary(self, lat_dbg: Dict[str, Any]) -> str:
        if not isinstance(lat_dbg, dict):
            return "latency={}"
        try:
            t_init = lat_dbg.get('source_to_leader', {}).get('t_initial_sec', None)
            dests = lat_dbg.get('destinations', []) or []
            parts = []
            if t_init is not None:
                parts.append(f"t_init={float(t_init):.3f}s")
            # summarize up to first 5 destinations
            for i, d in enumerate(dests[:5]):
                p = d.get('path_latency_sec', None)
                t_trans = d.get('t_trans_sec', None) or d.get('t_isl_sec', None)
                t_q = d.get('t_queue_sec', None)
                t_c = d.get('t_comp_sec', None)
                seg = []
                if p is not None:
                    seg.append(f"path={float(p):.3f}")
                if t_trans is not None:
                    seg.append(f"trans={float(t_trans):.3f}")
                if t_q is not None:
                    seg.append(f"queue={float(t_q):.3f}")
                if t_c is not None:
                    seg.append(f"comp={float(t_c):.3f}")
                parts.append(f"dest{i}({','.join(seg)})")
            return "latency={" + "; ".join(parts) + "}"
        except Exception:
            return "latency={}"

    def _on_step(self) -> bool:
        try:
            infos = self.locals.get('infos', None)
            actions = self.locals.get('actions', None)
            # rotate when we pass episode boundary
            if infos is not None:
                for info in infos:
                    ep_info = info.get('episode') if isinstance(info, dict) else None
                    if ep_info is not None:
                        self.episode_idx += 1
                        self._maybe_rotate()
            # choose first env's info for line output
            info0 = None
            if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
                info0 = infos[0]
            # build fields
            step = int(self.num_timesteps)
            env_id = info0.get('env_id', 0) if isinstance(info0, dict) else 0
            miou = info0.get('miou', None) if isinstance(info0, dict) else None
            reward = info0.get('reward', None) if isinstance(info0, dict) else None
            feasible = info0.get('feasible', None) if isinstance(info0, dict) else None
            total_lat = info0.get('total_latency', None) if isinstance(info0, dict) else None
            slice_size = info0.get('chosen_slice_size', None) if isinstance(info0, dict) else None
            k = info0.get('calculated_k', None) if isinstance(info0, dict) else None
            assign = info0.get('assignment', None) if isinstance(info0, dict) else None
            lat_dbg = info0.get('latency_debug', None) if isinstance(info0, dict) else None
            raw_obs = info0.get('_raw_obs', None) if isinstance(info0, dict) else None
            sim_time = info0.get('sim_time', 'NA') if isinstance(info0, dict) else 'NA'
            curr_max_lat = lat_dbg.get('max_latency_sec', None) if isinstance(lat_dbg, dict) else None

            # format line
            head = f"[{_dt.utcnow().isoformat()}Z] sim_time={sim_time} step={step} ep={self.episode_idx} env={env_id} slice={slice_size} k={k} feasible={int(bool(feasible)) if feasible is not None else 'NA'} miou={miou if miou is not None else 'NA'} reward={reward if reward is not None else 'NA'} total_latency={total_lat if total_lat is not None else 'NA'} curr_max_lat={curr_max_lat if curr_max_lat is not None else 'NA'}"
            ti_line = self._format_task_info_raw(raw_obs if isinstance(raw_obs, dict) else {})
            # assignment summary (print first 16)
            assign_line = "assignment=[]"
            try:
                if isinstance(assign, (list, tuple)):
                    first = assign[: min(len(assign), 16)]
                    more = len(assign) - len(first)
                    assign_line = f"assignment={first}{'...(+'+str(more)+')' if more>0 else ''}"
            except Exception:
                pass
            lat_line = self._format_latency_summary(lat_dbg)
            line = f"{head} | {ti_line} | {assign_line} | {lat_line}\n"
            self._maybe_rotate()
            if self._file is None:
                self._open_file()
            self._file.write(line)
            self._file.flush()
        except Exception:
            pass
        return True

    def _on_training_end(self) -> None:
        self._close_file()


class ObservationSanitizer(gym.ObservationWrapper):
    """Sanitize Dict observations by replacing NaN/Inf and optional clipping.

    Args:
        env: base env
        nan_value: value to replace NaN
        posinf: value to replace +Inf
        neginf: value to replace -Inf
        clip: optional tuple (min, max) to clip arrays after replacement
    """
    def __init__(self,
                 env: gym.Env,
                 nan_value: float = 0.0,
                 posinf: float = 1e9,
                 neginf: float = -1e9,
                 clip: Optional[tuple[float, float]] = None) -> None:
        super().__init__(env)
        self.nan_value = nan_value
        self.posinf = posinf
        self.neginf = neginf
        self.clip = clip

    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self._sanitize_dict(observation)

    def _sanitize_array(self, arr: np.ndarray) -> np.ndarray:
        out = np.nan_to_num(arr, nan=self.nan_value, posinf=self.posinf, neginf=self.neginf)
        if self.clip is not None:
            out = np.clip(out, self.clip[0], self.clip[1])
        return out.astype(np.float32, copy=False)

    def _sanitize_dict(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                clean[k] = self._sanitize_array(v)
            elif isinstance(v, dict):
                clean[k] = self._sanitize_dict(v)
            else:
                clean[k] = v
        return clean


def _build_single_env(sim_config_path: str,
                      sats_config_path: str,
                      seed: int,
                      use_action_wrapper: bool = True,
                      monitor_log_dir: Optional[str] = None,
                      override_max_tasks_per_episode: Optional[int] = None) -> Callable[[], Monitor]:
    """Return a thunk that builds one environment instance when called.

    Args:
        sim_config_path: path to configs/environment/simulation.yaml
        sats_config_path: path to configs/satellites.yaml
        seed: base seed for env
        use_action_wrapper: wrap Dict action -> MultiDiscrete flat
        monitor_log_dir: directory to write episode logs (Monitor)
    """
    def _thunk():
        sim_config = load_config(sim_config_path)
        # Optional override for faster/shorter episodes
        if override_max_tasks_per_episode is not None:
            try:
                sim_config = dict(sim_config)
                sim_config['max_tasks_per_episode'] = int(override_max_tasks_per_episode)
            except Exception:
                pass
        raw_sat_configs = load_config(sats_config_path)
        sat_configs = preprocess_satellite_configs(_project_root, raw_sat_configs)
        env = SatelliteEnv(sim_config=sim_config, sat_configs=sat_configs)
        if use_action_wrapper:
            env = FlattenedDictActionWrapper(env)
        # Ensure required info keys always exist to avoid Monitor KeyError
        required_keys = [
            "total_latency",
            "calculated_k",
            "miou",
            "feasible",
            "reward_mode",
            "chosen_slice_size",
            "reward",
        ]
        class InfoPaddingWrapper(gym.Wrapper):
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                if isinstance(info, dict):
                    for k in required_keys:
                        if k not in info:
                            # Provide sensible defaults
                            if k == "feasible":
                                info[k] = False
                            elif k == "reward_mode":
                                info[k] = "unknown"
                            else:
                                info[k] = 0.0
                return obs, reward, terminated, truncated, info
        env = InfoPaddingWrapper(env)
        # Sanitize observations to avoid NaN/Inf propagating to the policy
        env = ObservationSanitizer(env, nan_value=0.0, posinf=1e9, neginf=-1e9, clip=None)
        # Optional ObservationScaler (default enabled)
        try:
            scaler_cfg = sim_config.get('observation_scaler', {}) or {}
            enabled = scaler_cfg.get('enabled', True)
            env = ObservationScaler(
                env,
                enabled=enabled,
                Re=float(scaler_cfg.get('Re', 6_371_000.0)),
                Wmax=float(scaler_cfg.get('Wmax', 6000.0)),
                Hmax=float(scaler_cfg.get('Hmax', 6000.0)),
                LatMax=float(scaler_cfg.get('LatMax', 500.0)),
                BitsTot=float(scaler_cfg.get('BitsTot', 6000.0*6000.0*24.0)),
                FLOPsRef=float(scaler_cfg.get('FLOPsRef', 1.39e12)),
                QueueRef=float(scaler_cfg.get('QueueRef', 1e12)),
                RateRef=float(scaler_cfg.get('RateRef', 1e9)),
                clip_pos=float(scaler_cfg.get('clip_pos', 2.0)),
                clip_queue=float(scaler_cfg.get('clip_queue', 10.0)),
            )
        except Exception:
            pass
        # Monitor for episode stats
        # Always wrap with Monitor to ensure EvalCallback works correctly.
        # Only log to file if a directory is provided.
        log_path = os.path.join(monitor_log_dir, f"monitor_{seed}.csv") if monitor_log_dir else None
        env = Monitor(env, filename=log_path, info_keywords=(
            "total_latency",
            "calculated_k",
            "miou",
            "feasible",
            "reward_mode",
            "chosen_slice_size",
            "reward"
        ))
        # Seed
        env.reset(seed=seed)
        return env
    return _thunk


def make_vec_envs(sim_config_path: str,
                  sats_config_path: str,
                  n_envs: int = 8,
                  seed: int = 0,
                  use_subproc: bool = True,
                  use_action_wrapper: bool = True,
                  monitor_log_dir: Optional[str] = None,
                  override_max_tasks_per_episode: Optional[int] = None):
    """Create a vectorized env for SB3.

    Returns a VecEnv (SubprocVecEnv or DummyVecEnv).
    """
    thunks = [
        _build_single_env(
            sim_config_path=sim_config_path,
            sats_config_path=sats_config_path,
            seed=seed + i,
            use_action_wrapper=use_action_wrapper,
            monitor_log_dir=monitor_log_dir,
            override_max_tasks_per_episode=override_max_tasks_per_episode,
        )
        for i in range(n_envs)
    ]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(thunks)
    else:
        return DummyVecEnv(thunks)


def prepare_run_dirs(algo: str, run_name: Optional[str] = None) -> Dict[str, str]:
    """Prepare organized output directories for a training run.

    Structure:
      results/
        models/{algo}/{run_name}/
        logs/{algo}/{run_name}/tb/
        logs/{algo}/{run_name}/monitor/
        logs/{algo}/{run_name}/train_logs/ (JSONL)
        logs/{algo}/{run_name}/train_plain_logs/ (.log)
        evals/{algo}/{run_name}/
    Returns a dict with paths.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    run = run_name or f"{algo}_{ts}"
    base_results = os.path.join(_project_root, "results")
    paths = {
        "models": os.path.join(base_results, "models", algo, run),
        "tb": os.path.join(base_results, "logs", algo, run, "tb"),
        "monitor": os.path.join(base_results, "logs", algo, run, "monitor"),
        "train_logs": os.path.join(base_results, "logs", algo, run, "train_logs"),
        "train_plain_logs": os.path.join(base_results, "logs", algo, run, "train_plain_logs"),
        "evals": os.path.join(base_results, "evals", algo, run),
        "meta": os.path.join(base_results, "logs", algo, run, "metadata.json"),
    }
    for p in paths.values():
        d = p if p.endswith(".json") else p
        if not d.endswith(".json"):
            os.makedirs(d, exist_ok=True)
    return paths


def save_run_metadata(meta_path: str, cfg: Dict[str, Any]) -> None:
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def build_callbacks(eval_env,
                    models_dir: str,
                    evals_dir: str,
                    best_model_name: str = "best_model",
                    save_freq_steps: int = 100_000,
                    info_keys: Optional[List[str]] = None,
                    info_log_every: int = 200,
                    enable_step_reward_ma: bool = True,
                    step_ma_window: int = 1000,
                    step_ma_log_every: int = 200,
                    enable_episode_reward: bool = True,
                    episode_log_prefix: str = 'by_episode',
                    episode_reward_txt_path: Optional[str] = None,
                    # evaluation frequency in env steps (calls)
                    eval_freq_steps: int = 10_000,
                    # early stop knobs
                    early_stop: bool = False,
                    early_stop_patience_evals: int = 10,
                    early_stop_min_delta: float = 0.0,
                    early_stop_min_delta_rel: float = 0.01,
                    enable_episode_feasible: bool = True,
                    train_logs_dir: Optional[str] = None,
                    train_plain_logs_dir: Optional[str] = None,
                    log_rotate_every_episodes: int = 100,
                    # A2C-specific metrics
                    enable_a2c_metrics: bool = True,
                    a2c_metrics_log_every: int = 100,
                    a2c_metrics_txt_path: Optional[str] = None,
                    a2c_tb_prefix: str = 'a2c') -> List:
    """Create standard callbacks: Eval, periodic Checkpoint, and optional InfoScalar logging.

    Args:
        eval_env: environment for periodic evaluation
        models_dir: directory to save best/ckpt models
        evals_dir: directory to save eval logs
        best_model_name: unused (reserved)
        save_freq_steps: checkpoint frequency in env steps
        info_keys: list of Monitor info keys to write to TB scalars (mean over ep buffer)
        info_log_every: logging frequency (env steps)
    """
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=models_dir,
        log_path=evals_dir,
        eval_freq=1000,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=save_freq_steps,
        save_path=models_dir,
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    cbs = [eval_cb, ckpt_cb]
    if info_keys:
        cbs.append(InfoScalarCallback(keys=info_keys, log_every=info_log_every, verbose=1))

    # Early stopping based on EvalCallback results
    if early_stop:
        try:
            cbs.append(EarlyStopOnEvalReward(eval_cb,
                                             patience_evals=early_stop_patience_evals,
                                             min_delta=early_stop_min_delta,
                                             min_delta_rel=early_stop_min_delta_rel,
                                             verbose=1))
        except Exception:
            pass

    # Step-wise reward moving average (does not wait for episode end)
    if enable_step_reward_ma:
        cbs.append(StepRewardMovingAverageCallback(window_size=step_ma_window,
                                                   log_every=step_ma_log_every,
                                                   verbose=0))

    # Episode-wise reward using episode index as x-axis
    if enable_episode_reward:
        cbs.append(EpisodeRewardCallback(prefix=episode_log_prefix,
                                         txt_file_path=episode_reward_txt_path,
                                         verbose=0))

    # A2C training metrics
    if enable_a2c_metrics:
        cbs.append(A2CTrainingMetricsCallback(log_every=a2c_metrics_log_every, 
                                              txt_file_path=a2c_metrics_txt_path,
                                              tb_prefix=a2c_tb_prefix,
                                              verbose=1))

    # JSONL training logger
    if train_logs_dir:
        cbs.append(TrainingJSONLoggerCallback(logs_dir=train_logs_dir,
                                              rotate_every_episodes=log_rotate_every_episodes,
                                              verbose=0))

    # Plain text training logger
    if train_plain_logs_dir:
        cbs.append(PlainTextLoggerCallback(logs_dir=train_plain_logs_dir,
                                           rotate_every_episodes=log_rotate_every_episodes,
                                           verbose=0))

    return cbs
