# -*- coding: utf-8 -*-
"""
Wrappers and utilities to enable hierarchical RL (HRL) on top of SatelliteEnv using SB3 A2C.

Key components:
- MacroController: keeps current macro action (slice strategy index) and macro-step scheduling (every H env steps)
- MacroFlatOverrideWrapper: sits on top of FlattenedDictActionWrapper and overrides the first action dim(s)
  with the currently active macro action from MacroController. This lets the Worker policy output the full
  flat action but the macro part is ignored/overridden.
- GoalInjectionObsWrapper: injects a 'goal' vector into observations so the Worker policy is conditioned on
  the current macro decision. The goal is [slice_size_norm, k_norm].
- ManagerEnvWrapper: exposes a higher-level env where one manager step triggers H worker steps. The manager
  chooses the macro action (slice strategy index). Internally calls Worker policy to compute micro actions.

Notes:
- To avoid multiprocessing/pickling complications, it's recommended to use DummyVecEnv for Manager training.
- These wrappers are minimally invasive and don't change environment internal logic.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MacroController:
    """Stateful controller that stores the current macro action and handles macro step scheduling."""
    def __init__(self, num_slice_sizes: int, macro_horizon: int = 8, rng: Optional[np.random.Generator] = None):
        self.num_slice_sizes = int(num_slice_sizes)
        self.H = int(max(1, macro_horizon))
        self.rng = rng or np.random.default_rng()
        self.mode: str = 'random'  # 'random' or 'fixed'
        self.macro_idx: int = 0
        self.remaining: int = 0

    def set_mode(self, mode: str):
        assert mode in ('random', 'fixed')
        self.mode = mode

    def reset(self):
        self.macro_idx = 0
        self.remaining = 0

    def set_macro(self, idx: int):
        self.macro_idx = int(np.clip(idx, 0, self.num_slice_sizes - 1))
        self.remaining = self.H

    def ensure_macro(self):
        if self.remaining <= 0:
            if self.mode == 'random':
                self.macro_idx = int(self.rng.integers(0, self.num_slice_sizes))
                self.remaining = self.H
            else:
                # fixed but not set yet -> default 0
                self.macro_idx = int(np.clip(self.macro_idx, 0, self.num_slice_sizes - 1))
                self.remaining = self.H

    def on_step(self):
        if self.remaining > 0:
            self.remaining -= 1


class MacroFlatOverrideWrapper(gym.Wrapper):
    """Override the macro part of the flattened action using MacroController.

    Assumes inner env is already a FlattenedDictActionWrapper where the first K flat dims correspond to
    slice_strategy (here K=1 in current env). We override that dim with controller.macro_idx.
    """
    def __init__(self, env: gym.Env, controller: MacroController, slice_dims: int = 1):
        super().__init__(env)
        self.controller = controller
        self.slice_dims = int(slice_dims)
        # proxy action/observation spaces
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        # ensure a macro is active
        self.controller.ensure_macro()
        # override the first slice_dims dims
        try:
            if isinstance(action, (list, tuple)):
                action = np.array(action, dtype=np.int64)
            if isinstance(action, np.ndarray):
                if action.ndim == 1 and action.shape[0] >= self.slice_dims:
                    action = action.copy()
                    action[: self.slice_dims] = np.array([self.controller.macro_idx] * self.slice_dims, dtype=np.int64)
        except Exception:
            pass
        obs, rew, term, trunc, info = self.env.step(action)
        # countdown macro horizon
        self.controller.on_step()
        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        self.controller.reset()
        return self.env.reset(**kwargs)


class GoalInjectionObsWrapper(gym.ObservationWrapper):
    """Inject a 'goal' vector into observations to condition the Worker policy on current macro.

    goal = [slice_size_norm, k_norm]
      - slice_size_norm = chosen_slice_size / max_slice_size
      - k_norm = k(current_slice_size) / k_max (where k_max uses smallest slice size)
    """
    def __init__(self, env: gym.Env, controller: MacroController, slicing_strategies: List[Dict[str, Any]]):
        super().__init__(env)
        self.controller = controller
        self.slicing_strategies = slicing_strategies or []
        self._max_slice_size = max([s.get('slice_size', 512) for s in self.slicing_strategies] or [512])
        # approximate max-k by using the smallest slice size (typically yields largest k)
        self._min_slice_size = min([s.get('slice_size', 512) for s in self.slicing_strategies] or [512])
        # extend observation space
        orig = self.env.observation_space
        assert isinstance(orig, spaces.Dict)
        new_spaces = dict(orig.spaces)
        new_spaces['goal'] = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict(new_spaces)

    def _calc_k(self, slice_size: int) -> int:
        # Use env's internal method if available
        try:
            k = int(getattr(self.env.unwrapped, '_calculate_k')(int(slice_size)))
            return max(1, k)
        except Exception:
            return 1

    def observation(self, observation):
        # determine current slice size from controller macro
        try:
            idx = int(np.clip(self.controller.macro_idx, 0, max(0, len(self.slicing_strategies) - 1)))
            slice_size = int(self.slicing_strategies[idx].get('slice_size', self._min_slice_size))
        except Exception:
            slice_size = int(self._min_slice_size)
        k_cur = self._calc_k(slice_size)
        k_max = self._calc_k(self._min_slice_size)
        # normalize to [0,1]
        ss_norm = float(slice_size) / float(max(1, self._max_slice_size))
        k_norm = float(k_cur) / float(max(1, k_max))
        goal = np.array([ss_norm, k_norm], dtype=np.float32)
        out = dict(observation)
        out['goal'] = goal
        return out


class ManagerEnvWrapper(gym.Env):
    """Higher-level env for Manager policy. One step here = H worker steps in the base env.

    - action_space: Discrete(num_slice_sizes) -> choose slice strategy index
    - observation_space: same as base env (Dict), including 'goal' (so manager also sees current macro)
    - reward: sum of worker rewards over H steps (可根据需要改为平均/带权)
    - done: if any inner step ends episode, propagate done
    - info: aggregate some statistics (sum reward, feasible count)
    """
    def __init__(self, base_env: gym.Env, controller: MacroController, worker_model, macro_horizon: int = 8,
                 deterministic_worker: bool = False, num_slice_sizes: int = 0, slicing_strategies: Optional[List[Dict[str, Any]]] = None):
        super().__init__()
        self.base_env = base_env
        self.controller = controller
        self.worker_model = worker_model
        self.H = int(max(1, macro_horizon))
        self.det_worker = bool(deterministic_worker)
        # action space for manager: choose slice size index
        nss = int(num_slice_sizes or 0)
        if nss <= 0 and hasattr(base_env.unwrapped, 'slicing_strategies'):
            nss = int(len(getattr(base_env.unwrapped, 'slicing_strategies') or []))
        self.action_space = spaces.Discrete(max(1, nss))
        # observation space = base env's observation space (must be Dict)
        self.observation_space = base_env.observation_space
        self._last_obs = None
        self._feasible_in_window = 0
        self._slicing_strategies = slicing_strategies or getattr(base_env.unwrapped, 'slicing_strategies', [])

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self.controller.reset()
        self._last_obs = obs
        self._feasible_in_window = 0
        return obs, info

    def step(self, action):
        # set macro from manager
        macro_idx = int(action)
        self.controller.set_mode('fixed')
        self.controller.set_macro(macro_idx)
        total_rew = 0.0
        any_term = False
        any_trunc = False
        info_agg: Dict[str, Any] = {}

        obs = self._last_obs
        steps_in_window = 0
        miou_sum = 0.0
        miou_cnt = 0
        lat_sum = 0.0
        lat_cnt = 0
        for t in range(self.H):
            # let worker act on current obs
            worker_action, _ = self.worker_model.predict(obs, deterministic=self.det_worker)
            # step base env
            obs, rew, term, trunc, info = self.base_env.step(worker_action)
            total_rew += float(rew)
            steps_in_window += 1
            any_term = any_term or bool(term)
            any_trunc = any_trunc or bool(trunc)
            # accumulate feasible, miou, latency if provided
            try:
                if isinstance(info, dict):
                    if bool(info.get('feasible', False)):
                        self._feasible_in_window += 1
                    if 'miou' in info:
                        miou_sum += float(info.get('miou', 0.0))
                        miou_cnt += 1
                    if 'total_latency' in info:
                        lat_sum += float(info.get('total_latency', 0.0))
                        lat_cnt += 1
            except Exception:
                pass
            if any_term or any_trunc:
                break
        self._last_obs = obs
        # aggregate info for manager-level logging
        info_agg['sum_reward'] = float(total_rew)
        info_agg['macro_idx'] = int(macro_idx)
        info_agg['feasible_in_window'] = int(self._feasible_in_window)
        info_agg['macro_h'] = int(self.H)
        if steps_in_window > 0:
            info_agg['feasible_ratio'] = float(self._feasible_in_window) / float(steps_in_window)
        if miou_cnt > 0:
            info_agg['miou'] = float(miou_sum / miou_cnt)
        if lat_cnt > 0:
            info_agg['total_latency'] = float(lat_sum / lat_cnt)
        return obs, float(total_rew), bool(any_term), bool(any_trunc), info_agg

    def render(self):
        return self.base_env.render()

