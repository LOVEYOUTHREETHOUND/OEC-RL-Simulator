# -*- coding: utf-8 -*-
"""
Decision Transformer implementation for satellite task scheduling.

Based on "Decision Transformer: Reinforcement Learning via Sequence Modeling"
(Chen et al., NeurIPS 2021)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Any


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for Dict observation spaces.
    
    Predicts actions conditioned on:
    - Returns-to-go (desired future return)
    - State observations (Dict format)
    - Previous actions
    """
    
    def __init__(
        self,
        observation_space_shapes: Dict[str, tuple],
        action_space_config: Dict[str, Any],
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_ep_len: int = 200,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_ep_len = max_ep_len
        
        # Extract action space configuration
        self.num_slice_sizes = action_space_config['num_slice_sizes']
        self.num_destinations = action_space_config['num_destinations']
        self.max_k = action_space_config['max_k']
        
        # Embedding layers for each observation component
        self.obs_embeddings = nn.ModuleDict()
        for key, shape in observation_space_shapes.items():
            input_dim = int(np.prod(shape))
            self.obs_embeddings[key] = nn.Linear(input_dim, hidden_dim)
        
        # Combine all observation embeddings
        self.obs_combiner = nn.Linear(hidden_dim * len(observation_space_shapes), hidden_dim)
        
        # Return-to-go embedding
        self.rtg_embed = nn.Linear(1, hidden_dim)
        
        # Action embedding
        # For ratio action representation, the action vector is:
        #   [slice_strategy_one_hot(num_slice_sizes), dest_ratio(num_destinations)]
        # so action_dim = num_slice_sizes + num_destinations.
        action_dim = self.num_slice_sizes + self.num_destinations
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        
        # Timestep embedding
        self.timestep_embed = nn.Embedding(max_ep_len, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Action prediction heads
        self.slice_strategy_head = nn.Linear(hidden_dim, self.num_slice_sizes)
        # For 'ratio' action representation, the head predicts logits for each destination.
        self.assignment_head = nn.Linear(hidden_dim, self.num_destinations)
        
        # Layer normalization
        self.ln = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        returns_to_go: torch.Tensor,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of Decision Transformer.
        
        Args:
            returns_to_go: (batch, seq_len, 1)
            observations: Dict of (batch, seq_len, obs_dim)
            actions: (batch, seq_len, action_dim) - previous actions
            timesteps: (batch, seq_len)
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding
        
        Returns:
            action_logits: (batch, seq_len, action_dim)
        """
        batch_size, seq_len = returns_to_go.shape[0], returns_to_go.shape[1]
        
        # Embed observations
        obs_embeds = []
        for key in sorted(observations.keys()):
            obs_flat = observations[key].reshape(batch_size * seq_len, -1)
            obs_embed = self.obs_embeddings[key](obs_flat)
            obs_embeds.append(obs_embed)
        
        # Combine observation embeddings
        obs_combined = torch.cat(obs_embeds, dim=-1)
        obs_embed = self.obs_combiner(obs_combined)
        obs_embed = obs_embed.view(batch_size, seq_len, -1)
        
        # Embed returns-to-go
        rtg_embed = self.rtg_embed(returns_to_go)
        
        # Embed actions
        # Ensure actions have the correct shape
        if actions.dim() == 3:
            actions = actions.view(batch_size * seq_len, -1)
        
        expected_dim = self.num_slice_sizes + self.num_destinations
        if actions.size(-1) != expected_dim:
            # In ratio-mode we expect exact match
            raise ValueError(f"Expected actions dimension {expected_dim}, but got {actions.size(-1)}")
        
        action_embed = self.action_embed(actions.float())
        action_embed = action_embed.view(batch_size, seq_len, -1)
        
        # Embed timesteps
        timestep_embed = self.timestep_embed(timesteps)
        
        # Combine: interleave [RTG, State, Action] tokens
        # Reshape all tensors to have the same number of dimensions
        rtg_embed = rtg_embed.unsqueeze(2)  # [batch, seq_len, 1, hidden_dim]
        obs_embed = obs_embed.unsqueeze(2)  # [batch, seq_len, 1, hidden_dim]
        
        # Stack along the sequence dimension
        # Resulting shape: [batch, seq_len, 3, hidden_dim]
        tokens = torch.cat([rtg_embed, obs_embed, action_embed.unsqueeze(2)], dim=2)
        
        # Reshape to [batch, seq_len * 3, hidden_dim]
        tokens = tokens.reshape(batch_size, seq_len * 3, self.hidden_dim)
        
        # Add timestep embeddings (broadcast to all 3 tokens per timestep)
        timestep_embed_expanded = timestep_embed.repeat_interleave(3, dim=1)
        tokens = tokens + timestep_embed_expanded
        
        # Layer norm
        tokens = self.ln(tokens)
        
        # Create attention mask for causal modeling
        # For PyTorch TransformerEncoder:
        # - mask: (S, S) square mask where True indicates positions to mask
        # - src_key_padding_mask: (N, S) where True indicates padding tokens
        
        if attention_mask is not None:
            # Convert attention_mask to bool if it's float
            if attention_mask.dtype == torch.float:
                attention_mask = attention_mask.bool()
            # Expand mask to match token sequence
            attention_mask = attention_mask.repeat_interleave(3, dim=1)
            # Create key_padding_mask (True where padding is present)
            key_padding_mask = ~attention_mask
        else:
            key_padding_mask = None
        
        # Create causal mask (square mask where True indicates positions to mask)
        causal_mask = torch.triu(
            torch.ones(seq_len * 3, seq_len * 3, device=tokens.device),
            diagonal=1
        ).bool()
        
        # Transformer
        hidden_states = self.transformer(tokens, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        
        # Extract action prediction positions (every 3rd token, offset by 2)
        # Positions: 2, 5, 8, ... (after RTG and State)
        action_hidden = hidden_states[:, 2::3, :]
        
        # Predict actions
        # Split into slice strategy and assignment logits
        slice_strategy_logits = self.slice_strategy_head(action_hidden)  # [batch, seq_len, num_slice_sizes]
        
        # Assignment logits over destinations (ratio/mixture-style action)
        assignment_logits = self.assignment_head(action_hidden)  # [batch, seq_len, num_destinations]
        
        return {
            'slice_strategy_logits': slice_strategy_logits,
            'assignment_logits': assignment_logits
        }
        
    def get_action(
        self,
        returns_to_go: torch.Tensor,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Get action for inference (single timestep) in ratio-mode.

        Returns:
            Dict containing:
                - slice_strategy: (batch,) selected slice strategy index
                - dest_probs: (batch, num_destinations) probability distribution over destinations
        """
        with torch.no_grad():
            outputs = self.forward(returns_to_go, observations, actions, timesteps)

            slice_logits = outputs['slice_strategy_logits'][:, -1, :] / max(temperature, 1e-6)
            slice_probs = torch.softmax(slice_logits, dim=-1)
            slice_strategy = torch.multinomial(slice_probs, num_samples=1).squeeze(-1)

            dest_logits = outputs['assignment_logits'][:, -1, :] / max(temperature, 1e-6)
            dest_probs = torch.softmax(dest_logits, dim=-1)

            return {
                'slice_strategy': slice_strategy,
                'dest_probs': dest_probs,
            }

