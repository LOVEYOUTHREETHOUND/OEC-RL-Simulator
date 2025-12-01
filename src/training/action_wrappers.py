# -*- coding: utf-8 -*-
"""
Action wrappers for training.

FlattenedDictActionWrapper:
- Converts the environment's Dict action space
  { 'slice_strategy': MultiDiscrete([ns, no]), 'assignment': MultiDiscrete([nd]*mk) }
  into a single MultiDiscrete([ns, no] + [nd]*mk).
- On step(), maps the flat action back into the Dict expected by the env.

Note:
- The env internally only consumes the first k entries of 'assignment', where k
  is computed from the chosen slice strategy. The remaining entries are ignored
  by the env. SB3 will still compute a log-prob for all action dimensions,
  which introduces some noise but works in practice. For more precise credit
  assignment, a custom policy/loss with k-masking would be required.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlattenedDictActionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Validate original action space
        assert isinstance(env.action_space, spaces.Dict), "Env action_space must be Dict"
        slice_space = env.action_space.spaces['slice_strategy']
        assign_space = env.action_space.spaces['assignment']
        assert isinstance(slice_space, spaces.MultiDiscrete)
        assert isinstance(assign_space, spaces.MultiDiscrete)

        nvec = list(map(int, slice_space.nvec.tolist()))
        if len(nvec) == 1:
            self._ns = nvec[0]
            self._no = 0
            self._slice_dims = 1
        else:
            self._ns = nvec[0]
            self._no = nvec[1]
            self._slice_dims = 2
        self._mk = int(len(assign_space.nvec))
        self._nd = int(assign_space.nvec[0]) if self._mk > 0 else 0

        # New flattened MultiDiscrete: [ns] (+[no] if present) + [nd]*mk
        base = [self._ns] if self._slice_dims == 1 else [self._ns, self._no]
        self.action_space = spaces.MultiDiscrete(np.array(base + [self._nd] * self._mk, dtype=np.int64))
        # Observation space unchanged
        self.observation_space = env.observation_space

    def action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert flat action to the Dict expected by the inner env."""
        if isinstance(action, (list, tuple)):
            action = np.array(action, dtype=np.int64)
        expected = (1 if self._slice_dims == 1 else 2) + self._mk
        assert action.shape[0] == expected, f"Flat action has unexpected length: {action.shape[0]} != {expected}"
        if self._slice_dims == 1:
            slice_strategy = action[:1]
            assignment = action[1:]
        else:
            slice_strategy = action[:2]
            assignment = action[2:]
        return {
            'slice_strategy': slice_strategy.astype(np.int64),
            'assignment': assignment.astype(np.int64)
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        dict_action = self.action(action)
        obs, rew, term, trunc, info = self.env.step(dict_action)
        return obs, float(rew), bool(term), bool(trunc), info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

