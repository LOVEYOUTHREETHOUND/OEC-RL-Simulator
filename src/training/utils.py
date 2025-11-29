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

# Resolve project root and import local modules
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.append(_project_root)

from src.utils.config_loader import load_config
from src.utils.tle_loader import preprocess_satellite_configs
from src.environment.satellite_env import SatelliteEnv
from src.training.action_wrappers import FlattenedDictActionWrapper


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
    This gives a clear view of reward vs episode progression.
    """
    def __init__(self, log_ep_length: bool = True, prefix: str = 'by_episode', verbose: int = 0):
        super().__init__(verbose)
        self.log_ep_length = log_ep_length
        self.prefix = prefix
        self.episode_idx = 0

    def _on_step(self) -> bool:
        infos = None
        try:
            infos = self.locals.get('infos', None)
        except Exception:
            infos = None
        if infos is not None:
            for info in infos:
                ep_info = info.get('episode') if isinstance(info, dict) else None
                if ep_info is not None:
                    self.episode_idx += 1
                    r = float(ep_info.get('r', 0.0))
                    self.logger.record(f'{self.prefix}/episode_reward', r)
                    if self.log_ep_length:
                        l = float(ep_info.get('l', 0.0))
                        self.logger.record(f'{self.prefix}/episode_length', l)
                    # dump with episode index as x-axis
                    if hasattr(self.logger, 'dump'):
                        try:
                            self.logger.dump(self.episode_idx)
                        except Exception:
                            pass
        return True


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
        # Sanitize observations to avoid NaN/Inf propagating to the policy
        env = ObservationSanitizer(env, nan_value=0.0, posinf=1e9, neginf=-1e9, clip=None)
        # Monitor for episode stats
        if monitor_log_dir:
            os.makedirs(monitor_log_dir, exist_ok=True)
            env = Monitor(env, filename=None, info_keywords=(
                "total_latency",
                "calculated_k",
                "aoi",
                "miou",
                "violation_ratio",
                "feasible",
                "reward_mode",
                "chosen_slice_size",
                "chosen_overlap_ratio"
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
                    info_log_every: int = 1000,
                    enable_step_reward_ma: bool = True,
                    step_ma_window: int = 1000,
                    step_ma_log_every: int = 200,
                    enable_episode_reward: bool = True,
                    episode_log_prefix: str = 'by_episode') -> List:
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
        eval_freq=10_000,
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

    # Step-wise reward moving average (does not wait for episode end)
    if enable_step_reward_ma:
        cbs.append(StepRewardMovingAverageCallback(window_size=step_ma_window,
                                                   log_every=step_ma_log_every,
                                                   verbose=0))

    # Episode-wise reward using episode index as x-axis
    if enable_episode_reward:
        cbs.append(EpisodeRewardCallback(prefix=episode_log_prefix, verbose=0))

    return cbs
