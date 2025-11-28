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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Resolve project root and import local modules
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.append(_project_root)

from src.utils.config_loader import load_config
from src.utils.tle_loader import preprocess_satellite_configs
from src.environment.satellite_env import SatelliteEnv
from src.training.action_wrappers import FlattenedDictActionWrapper


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
                      monitor_log_dir: Optional[str] = None) -> Callable[[], Monitor]:
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
            env = Monitor(env, filename=None, info_keywords=("total_latency", "calculated_k"))
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
                  monitor_log_dir: Optional[str] = None):
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
                    save_freq_steps: int = 100_000) -> List:
    """Create standard callbacks: Eval and periodic Checkpoint."""
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
    return [eval_cb, ckpt_cb]
