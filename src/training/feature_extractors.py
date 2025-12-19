# -*- coding: utf-8 -*-
"""
Custom features extractors for SB3 policies.

NodeTransformerExtractor:
- Treats environment observation as a set/sequence of tokens and applies a Transformer encoder
  to model relationships between task, leader, ground station and nearest compute nodes.
- Minimally invasive: drop-in replacement for SB3's default CombinedExtractor used by
  MultiInputPolicy. It only changes the state representation; the A2C algorithm and heads stay
  untouched.

Inputs (expected keys in observation dict, already converted to torch tensors by SB3):
- task_info:             (B, 7)
- leader_pos:            (B, 3)
- compute_pos:           (B, N, 3)
- compute_queues:        (B, N)
- ground_station_pos:    (B, 3)
- ground_station_queue:  (B, 1)

Output:
- A single feature vector of size `d_model` per batch item.

Notes:
- Assumes observations have been pre-scaled by ObservationScaler; otherwise consider adding
  normalizations inside the projector layers.
- N (num_nearest_compute) is inferred from observation space shape at construction time.
"""
from __future__ import annotations

from typing import Dict

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NodeTransformerExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        pool: str = "mean",  # or "cls" (use task token)
    ) -> None:
        # features_dim must be set before super().__init__ returns
        super().__init__(observation_space, features_dim=d_model)

        # Infer N from observation space
        assert isinstance(observation_space, spaces.Dict), "Observation space must be Dict"
        comp_pos_space = observation_space.spaces.get("compute_pos")
        assert isinstance(comp_pos_space, spaces.Box) and comp_pos_space.shape[-1] == 3, (
            "compute_pos must be (N, 3)"
        )
        self.N = int(comp_pos_space.shape[0])

        # Projections from raw features to model dimension
        # compute: concat pos(3) + queue(1) -> d_model
        self.proj_compute = nn.Sequential(
            nn.Linear(3 + 1, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        # leader: pos(3) -> d_model
        self.proj_leader = nn.Sequential(
            nn.Linear(3, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        # ground station: pos(3) + queue(1) -> d_model
        self.proj_gs = nn.Sequential(
            nn.Linear(3 + 1, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        # task: (7,) -> d_model
        self.proj_task = nn.Sequential(
            nn.Linear(7, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        # optional goal: (2,) -> d_model
        self.proj_goal = nn.Sequential(
            nn.Linear(2, d_model), nn.ReLU(), nn.Dropout(dropout)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._pool_mode = str(pool)

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        # Expect tensors with shapes described in the module docstring
        # Ensure float dtype
        def _f(x: th.Tensor) -> th.Tensor:
            return x.to(dtype=th.float32)

        B = obs["task_info"].shape[0]

        # Compute tokens: (B, N, 3) + (B, N, 1) -> (B, N, d_model)
        cp = _f(obs.get("compute_pos"))  # (B, N, 3)
        cq = _f(obs.get("compute_queues")).unsqueeze(-1)  # (B, N, 1)
        cfeat = th.cat([cp, cq], dim=-1)
        ctok = self.proj_compute(cfeat)

        # Leader token: (B, 1, d_model)
        ltok = self.proj_leader(_f(obs.get("leader_pos"))).unsqueeze(1)

        # Ground station token: (B, 1, d_model)
        gs_pos = _f(obs.get("ground_station_pos"))  # (B, 3)
        gs_q = _f(obs.get("ground_station_queue"))  # (B, 1)
        gstok = self.proj_gs(th.cat([gs_pos, gs_q], dim=-1)).unsqueeze(1)

        # Task token: (B, 1, d_model)
        ttok = self.proj_task(_f(obs.get("task_info"))).unsqueeze(1)

        tokens_list = [ttok, ltok, gstok, ctok]
        # Optional goal token: shape (B, 1, d_model)
        if "goal" in obs:
            gtok = self.proj_goal(_f(obs.get("goal"))).unsqueeze(1)
            # put goal in front to act as conditioning token
            tokens_list = [gtok] + tokens_list

        # Token order: [GOAL? , TASK, LEADER, GS, COMPUTE_0..N-1]
        tokens = th.cat(tokens_list, dim=1)

        # Transformer encoder
        h = self.encoder(tokens)

        if self._pool_mode == "cls":
            pooled = h[:, 0, :]  # use first token as CLS (goal if present else task)
        else:
            pooled = h.mean(dim=1)

        return pooled

