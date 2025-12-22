# -*- coding: utf-8 -*-
"""
Hierarchical A2C training with Transformer feature extractor for SatelliteEnv.

Structure:
- Worker (low-level) A2C: acts every env step, decides micro actions (assignment etc.).
  The macro part of the action (slice strategy index) is overridden by a MacroController.
  The current macro decision is injected into observation as a 'goal' vector so the
  Worker policy is conditioned on it.
- Manager (high-level) A2C: acts every H steps, selects macro action = slice strategy index.
  One Manager step triggers H Worker-controlled env steps internally.

Notes:
- We use DummyVecEnv for Worker training to simplify shared controller usage.
- ManagerEnvWrapper is trained as a single-env (SB3 auto-wraps with DummyVecEnv).
- The same NodeTransformerExtractor is used; it now supports an optional 'goal' token.
"""
from __future__ import annotations

import os
import sys
import argparse
import re
from datetime import datetime
from typing import Callable, Dict, Any, Optional, Tuple

import torch

# Mitigate CUDA fragmentation (Windows 不支持 expandable_segments，关闭并设置更保守参数)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False,garbage_collection_threshold:0.8,max_split_size_mb:64")

import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


def _ensure_callbacklist(cb):
    """Normalize and flatten into a CallbackList with only BaseCallback items."""
    def _flatten(seq):
        out = []
        for c in seq:
            if isinstance(c, CallbackList):
                out.extend(_flatten(c.callbacks))
            elif isinstance(c, list):
                out.extend(_flatten(c))
            elif isinstance(c, BaseCallback):
                out.append(c)
        return out

    if isinstance(cb, CallbackList):
        return CallbackList(_flatten(cb.callbacks))
    if isinstance(cb, BaseCallback):
        return CallbackList([cb])
    if isinstance(cb, list):
        return CallbackList(_flatten(cb))
    return CallbackList([])

# Ensure project root on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training.feature_extractors import NodeTransformerExtractor
from src.training.utils import (
    prepare_run_dirs,
    save_run_metadata,
    ObservationSanitizer,
    ObservationScaler,
    build_callbacks,
)
from src.training.action_wrappers import FlattenedDictActionWrapper
from src.training.hrl_wrappers import (
    MacroController,
    MacroFlatOverrideWrapper,
    GoalInjectionObsWrapper,
    ManagerEnvWrapper,
)
from src.utils.config_loader import load_config
from src.utils.tle_loader import preprocess_satellite_configs
from src.environment.satellite_env import SatelliteEnv


def _resolve_device(dev_arg: str) -> str:
    dev_arg = (dev_arg or "cuda").strip().lower()
    try:
        if dev_arg.startswith("cuda"):
            m = re.match(r"cuda(?::(\d+))?$", dev_arg)
            idx = int(m.group(1)) if (m and m.group(1) is not None) else 0
            if not torch.cuda.is_available() or idx >= torch.cuda.device_count():
                print(f"[Warn] CUDA unavailable or invalid index '{dev_arg}', fallback to CPU.", file=sys.stderr, flush=True)
                return "cpu"
            return f"cuda:{idx}"
        if dev_arg in ("cpu", "auto"):
            if dev_arg == "auto" and torch.cuda.is_available():
                return "cuda:0"
            return "cpu"
    except Exception as e:
        print(f"[Warn] Device resolve error for '{dev_arg}': {e}, fallback to CPU.", file=sys.stderr, flush=True)
    return "cpu"


class CudaMemLogger(BaseCallback):
    """Periodically prints CUDA memory usage/peak. Works even if CPU fallback (no-op)."""
    def __init__(self, tag: str, device_str: str, every_steps: int = 0):
        super().__init__()
        self.tag = tag
        self.device_str = device_str
        self.every_steps = int(every_steps)
        self._last_print = 0

    def _on_training_start(self) -> None:
        if torch.cuda.is_available() and self.device_str.startswith("cuda"):
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def _print_now(self) -> None:
        if torch.cuda.is_available() and self.device_str.startswith("cuda"):
            try:
                alloc = torch.cuda.memory_allocated() / (1024**2)
                reserved = torch.cuda.memory_reserved() / (1024**2)
                peak = torch.cuda.max_memory_allocated() / (1024**2)
                print(f"[{datetime.utcnow().isoformat()}Z][MEM][{self.tag}] steps={self.num_timesteps} alloc={alloc:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB device={self.device_str}", flush=True)
            except Exception:
                pass

    def _on_rollout_end(self) -> None:
        self._print_now()

    def _on_step(self) -> bool:
        if self.every_steps > 0 and (self.num_timesteps - self._last_print) >= self.every_steps:
            self._print_now()
            self._last_print = self.num_timesteps
        return True


class AdvantageLogger(BaseCallback):
    """Log advantage mean/std to TensorBoard and a txt file after each rollout."""
    def __init__(self, tag: str, txt_path: Optional[str] = None):
        super().__init__()
        self.tag = tag
        self.txt_path = txt_path
        self._fh = None

    def _ensure_file(self):
        if self.txt_path and self._fh is None:
            os.makedirs(os.path.dirname(self.txt_path), exist_ok=True)
            new_file = not os.path.exists(self.txt_path)
            self._fh = open(self.txt_path, "a", encoding="utf-8")
            if new_file:
                self._fh.write("steps\tadv_mean\tadv_std\n")
                self._fh.flush()

    def _on_rollout_end(self) -> None:
        rb = getattr(self.model, "rollout_buffer", None)
        if rb is None:
            return
        adv = getattr(rb, "advantages", None)
        if adv is None:
            # fallback: returns - values
            returns = getattr(rb, "returns", None)
            values = getattr(rb, "values", None)
            if returns is not None and values is not None:
                try:
                    adv = returns - values
                except Exception:
                    adv = None
        if adv is None:
            return
        try:
            adv_np = adv.detach().cpu().float().numpy().reshape(-1)
        except Exception:
            return
        if adv_np.size == 0:
            return
        mean = float(np.mean(adv_np))
        std = float(np.std(adv_np))
        # TensorBoard
        try:
            self.logger.record(f"{self.tag}/advantage_mean", mean)
            self.logger.record(f"{self.tag}/advantage_std", std)
        except Exception:
            pass
        # TXT
        try:
            self._ensure_file()
            if self._fh is not None:
                self._fh.write(f"{self.num_timesteps}\t{mean:.6f}\t{std:.6f}\n")
                self._fh.flush()
        except Exception:
            pass

    def _on_training_end(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass


def parse_args():
    p = argparse.ArgumentParser("HRL (Manager/Worker) A2C training for SatelliteEnv")
    p.add_argument("--sim_config", default=os.path.join("configs", "environment", "simulation.yaml"))
    p.add_argument("--sats_config", default=os.path.join("configs", "satellites.yaml"))
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--run_name", type=str, default=None)

    # Worker training knobs
    p.add_argument("--worker_total_timesteps", type=int, default=1_000_000)
    p.add_argument("--worker_n_envs", type=int, default=8)
    p.add_argument("--worker_n_steps", type=int, default=2048)
    p.add_argument("--worker_lr", type=float, default=3e-4)
    p.add_argument("--worker_ent_coef", type=float, default=0.01)

    # Manager training knobs
    p.add_argument("--manager_total_timesteps", type=int, default=300_000)
    p.add_argument("--manager_n_steps", type=int, default=1024)
    p.add_argument("--manager_lr", type=float, default=3e-4)
    p.add_argument("--manager_ent_coef", type=float, default=0.01)
    p.add_argument("--macro_horizon", type=int, default=8, help="Manager step spans H worker steps")

    # Transformer size knobs
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=2)

    # Episode length override for faster iterations (optional)
    p.add_argument("--override_max_tasks_per_episode", type=int, default=None)

    # Device
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


def build_single_env_thunk(sim_config_path: str,
                           sats_config_path: str,
                           seed: int,
                           controller: MacroController,
                           monitor_dir: Optional[str] = None,
                           override_max_tasks_per_episode: Optional[int] = None) -> Callable[[], Monitor]:
    def _thunk():
        sim_config = load_config(sim_config_path)
        if override_max_tasks_per_episode is not None:
            try:
                sim_config = dict(sim_config)
                sim_config['max_tasks_per_episode'] = int(override_max_tasks_per_episode)
            except Exception:
                pass
        raw_sat_configs = load_config(sats_config_path)
        sat_configs = preprocess_satellite_configs(project_root, raw_sat_configs)

        env = SatelliteEnv(sim_config=sim_config, sat_configs=sat_configs)
        # Action flatten first
        env = FlattenedDictActionWrapper(env)
        # Sanitize + Scale (consistent with standard make_vec_envs)
        env = ObservationSanitizer(env, nan_value=0.0, posinf=1e9, neginf=-1e9, clip=None)
        try:
            scaler_cfg = sim_config.get('observation_scaler', {}) or {}
            env = ObservationScaler(
                env,
                enabled=bool(scaler_cfg.get('enabled', True)),
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
        # HRL wrappers: override macro action and inject goal
        env = MacroFlatOverrideWrapper(env, controller=controller, slice_dims=1)
        env = GoalInjectionObsWrapper(env, controller=controller, slicing_strategies=sim_config.get('slicing_strategies', []))
        # Monitor
        log_path = os.path.join(monitor_dir, f"monitor_{seed}.csv") if monitor_dir else None
        env = Monitor(env, filename=log_path, info_keywords=(
            "total_latency",
            "calculated_k",
            "miou",
            "feasible",
            "reward_mode",
            "chosen_slice_size",
            "reward",
        ))
        env.reset(seed=seed)
        return env
    return _thunk


def main():
    args = parse_args()
    set_random_seed(args.seed)

    # Resolve device with safe fallback
    device_str = _resolve_device(getattr(args, "device", "cuda"))
    print(f"[Device] Using device: {device_str}", flush=True)

    # Prepare dirs
    paths = prepare_run_dirs(algo="a2c_hrl", run_name=args.run_name)
    save_run_metadata(paths["meta"], {
        "algo": "A2C_HRL",
        "seed": args.seed,
        "sim_config": args.sim_config,
        "sats_config": args.sats_config,
        "macro_horizon": args.macro_horizon,
        "worker": {
            "total_timesteps": args.worker_total_timesteps,
            "n_envs": args.worker_n_envs,
            "n_steps": args.worker_n_steps,
            "lr": args.worker_lr,
            "ent_coef": args.worker_ent_coef,
        },
        "manager": {
            "total_timesteps": args.manager_total_timesteps,
            "n_steps": args.manager_n_steps,
            "lr": args.manager_lr,
            "ent_coef": args.manager_ent_coef,
        },
        "transformer": {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
        },
    })

    # Read slicing strategies count for controller
    sim_cfg = load_config(args.sim_config)
    num_slice_sizes = int(len(sim_cfg.get('slicing_strategies', [])))
    if num_slice_sizes <= 0:
        raise RuntimeError("slicing_strategies is empty in sim config; HRL requires at least 1 slice size")

    # Shared macro controller (random during worker training)
    controller = MacroController(num_slice_sizes=num_slice_sizes, macro_horizon=args.macro_horizon)
    controller.set_mode('random')

    # Build Worker env (DummyVecEnv of N thunks that share controller)
    worker_thunks = [
        build_single_env_thunk(
            sim_config_path=args.sim_config,
            sats_config_path=args.sats_config,
            seed=args.seed + i,
            controller=controller,
            monitor_dir=paths["monitor"],
            override_max_tasks_per_episode=args.override_max_tasks_per_episode,
        )
        for i in range(args.worker_n_envs)
    ]
    worker_env = DummyVecEnv(worker_thunks)

    # Worker model
    worker = A2C(
        policy="MultiInputPolicy",
        env=worker_env,
        learning_rate=args.worker_lr,
        n_steps=args.worker_n_steps,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=args.worker_ent_coef,
        vf_coef=0.5,
        seed=args.seed,
        verbose=1,
        tensorboard_log=paths["tb"],
        device=device_str,
        policy_kwargs=dict(
            features_extractor_class=NodeTransformerExtractor,
            features_extractor_kwargs=dict(
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dropout=0.1,
                pool="mean",
            ),
        ),
    )

    # Build eval env for worker
    eval_worker_env = DummyVecEnv([
        build_single_env_thunk(
            sim_config_path=args.sim_config,
            sats_config_path=args.sats_config,
            seed=args.seed + 7777,
            controller=controller,
            monitor_dir=None,
            override_max_tasks_per_episode=args.override_max_tasks_per_episode,
        )
    ])

    # Worker callbacks (txt + TB)
    worker_models_dir = os.path.join(paths["models"], "worker")
    worker_evals_dir = os.path.join(paths["evals"], "worker")
    os.makedirs(worker_models_dir, exist_ok=True)
    os.makedirs(worker_evals_dir, exist_ok=True)
    worker_episode_txt = os.path.join(os.path.dirname(paths["tb"]), "worker_by_episode.txt")
    worker_metrics_txt = os.path.join(os.path.dirname(paths["tb"]), "worker_a2c_metrics.txt")
    worker_adv_txt = os.path.join(os.path.dirname(paths["tb"]), "worker_advantage.txt")
    worker_callbacks = build_callbacks(
        eval_env=eval_worker_env,
        models_dir=worker_models_dir,
        evals_dir=worker_evals_dir,
        save_freq_steps=max(args.worker_n_steps // max(1, args.worker_n_envs), 1),
        train_logs_dir=os.path.join(os.path.dirname(paths["tb"]), "worker_train_logs"),
        train_plain_logs_dir=os.path.join(os.path.dirname(paths["tb"]), "worker_train_plain_logs"),
        episode_log_prefix='worker/by_episode',
        episode_reward_txt_path=worker_episode_txt,
        enable_a2c_metrics=True,
        a2c_metrics_txt_path=worker_metrics_txt,
        a2c_tb_prefix='worker',
        log_rotate_every_episodes=100,
    )
    # 规范化为 CallbackList 并附加显存与优势日志回调
    worker_callbacks = _ensure_callbacklist(worker_callbacks)
    try:
        mem_cb_worker = CudaMemLogger(tag='worker', device_str=device_str, every_steps=args.worker_n_envs * args.worker_n_steps)
        adv_cb_worker = AdvantageLogger(tag='worker', txt_path=worker_adv_txt)
        worker_callbacks.callbacks.append(mem_cb_worker)
        worker_callbacks.callbacks.append(adv_cb_worker)
    except Exception:
        pass

    # Train Worker
    worker.learn(total_timesteps=args.worker_total_timesteps, callback=worker_callbacks, progress_bar=True)

    # Move worker to CPU and eval mode for manager sampling to free GPU memory
    try:
        worker.policy.to("cpu")
        worker.policy.set_training_mode(False)
    except Exception:
        pass

    # Build a single base env instance for Manager (shares same controller)
    # Manager will control controller (fixed mode), Worker acts inside.
    base_env = build_single_env_thunk(
        sim_config_path=args.sim_config,
        sats_config_path=args.sats_config,
        seed=args.seed + 100_000,
        controller=controller,
        monitor_dir=None,
        override_max_tasks_per_episode=args.override_max_tasks_per_episode,
    )()

    # Manager env wrapper (single env; SB3 will wrap with DummyVecEnv internally)
    manager_env = ManagerEnvWrapper(
        base_env=base_env,
        controller=controller,
        worker_model=worker,
        macro_horizon=args.macro_horizon,
        deterministic_worker=True,
    )
    # Wrap manager env with Monitor to enable episode logging
    manager_env = Monitor(manager_env, filename=os.path.join(paths["monitor"], "manager_monitor.csv"), info_keywords=(
        "feasible_ratio",
        "miou",
        "total_latency",
        "macro_idx",
        "macro_h",
        "sum_reward",
    ))

    # Build eval env for manager (separate controller to avoid interference)
    eval_controller = MacroController(num_slice_sizes=num_slice_sizes, macro_horizon=args.macro_horizon)
    eval_controller.set_mode('fixed')
    eval_base_env = build_single_env_thunk(
        sim_config_path=args.sim_config,
        sats_config_path=args.sats_config,
        seed=args.seed + 200_000,
        controller=eval_controller,
        monitor_dir=None,
        override_max_tasks_per_episode=args.override_max_tasks_per_episode,
    )()
    eval_manager_env = ManagerEnvWrapper(
        base_env=eval_base_env,
        controller=eval_controller,
        worker_model=worker,
        macro_horizon=args.macro_horizon,
        deterministic_worker=True,
    )
    eval_manager_env = Monitor(eval_manager_env, filename=None, info_keywords=(
        "feasible_ratio",
        "miou",
        "total_latency",
        "macro_idx",
        "macro_h",
        "sum_reward",
    ))

    # Manager model
    # Manager sees same observation space (with 'goal'), action space = Discrete(num_slice_sizes)
    manager = A2C(
        policy="MultiInputPolicy",
        env=manager_env,
        learning_rate=args.manager_lr,
        n_steps=args.manager_n_steps,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=args.manager_ent_coef,
        vf_coef=0.5,
        seed=args.seed + 42,
        verbose=1,
        tensorboard_log=paths["tb"],
        device=device_str,
        policy_kwargs=dict(
            features_extractor_class=NodeTransformerExtractor,
            features_extractor_kwargs=dict(
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                dropout=0.1,
                pool="mean",
            ),
        ),
    )

    # Manager callbacks (txt + TB)
    manager_models_dir = os.path.join(paths["models"], "manager")
    manager_evals_dir = os.path.join(paths["evals"], "manager")
    os.makedirs(manager_models_dir, exist_ok=True)
    os.makedirs(manager_evals_dir, exist_ok=True)
    manager_episode_txt = os.path.join(os.path.dirname(paths["tb"]), "manager_by_episode.txt")
    manager_metrics_txt = os.path.join(os.path.dirname(paths["tb"]), "manager_a2c_metrics.txt")
    manager_callbacks = build_callbacks(
        eval_env=eval_manager_env,
        models_dir=manager_models_dir,
        evals_dir=manager_evals_dir,
        save_freq_steps=max(args.manager_n_steps, 1),
        train_logs_dir=os.path.join(os.path.dirname(paths["tb"]), "manager_train_logs"),
        train_plain_logs_dir=os.path.join(os.path.dirname(paths["tb"]), "manager_train_plain_logs"),
        episode_log_prefix='manager/by_episode',
        episode_reward_txt_path=manager_episode_txt,
        enable_a2c_metrics=True,
        a2c_metrics_txt_path=manager_metrics_txt,
        a2c_tb_prefix='manager',
        log_rotate_every_episodes=100,
    )
    # 规范化为 CallbackList 并附加显存与优势日志回调
    manager_callbacks = _ensure_callbacklist(manager_callbacks)
    try:
        manager_adv_txt = os.path.join(os.path.dirname(paths["tb"]), "manager_advantage.txt")
        mem_cb_manager = CudaMemLogger(tag='manager', device_str=device_str, every_steps=args.manager_n_steps)
        adv_cb_manager = AdvantageLogger(tag='manager', txt_path=manager_adv_txt)
        manager_callbacks.callbacks.append(mem_cb_manager)
        manager_callbacks.callbacks.append(adv_cb_manager)
    except Exception:
        pass

    # Train Manager
    # Important: Manager sets controller.set_macro(); Worker acts deterministically inside its window.
    controller.set_mode('fixed')
    manager.learn(total_timesteps=args.manager_total_timesteps, callback=manager_callbacks, progress_bar=True)

    # Save models
    os.makedirs(paths["models"], exist_ok=True)
    worker.save(os.path.join(paths["models"], "worker_final"))
    manager.save(os.path.join(paths["models"], "manager_final"))

    try:
        worker_env.close()
        base_env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()

