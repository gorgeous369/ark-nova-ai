"""Recurrent actor-critic model with MLP+LSTM backbone."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class MaskedActorCritic(nn.Module):
    """State-conditioned action scorer for variable-sized legal action sets."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        lstm_size: int = 128,
        action_hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.lstm_size = int(lstm_size)
        self.action_hidden_size = int(action_hidden_size)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.state_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.lstm_size,
            batch_first=True,
        )
        state_latent_dim = self.lstm_size

        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.action_hidden_size),
            nn.ReLU(),
            nn.Linear(self.action_hidden_size, self.action_hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(state_latent_dim + self.action_hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(state_latent_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def init_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, int(batch_size), self.lstm_size, device=device)
        c = torch.zeros(1, int(batch_size), self.lstm_size, device=device)
        return h, c

    def forward_step(
        self,
        *,
        state_vec: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward one environment step.

        Args:
            state_vec: [B, state_dim]
            action_features: [B, N, action_dim]
            action_mask: [B, N] (`bool` or 1/0 numeric; true for legal)
            hidden: optional LSTM hidden state
        """
        if state_vec.ndim != 2:
            raise ValueError("state_vec must be [B, state_dim].")
        if action_features.ndim != 3:
            raise ValueError("action_features must be [B, N, action_dim].")
        if action_mask.ndim != 2:
            raise ValueError("action_mask must be [B, N].")
        if action_features.shape[0] != state_vec.shape[0]:
            raise ValueError("Batch size mismatch between state_vec and action_features.")
        if action_mask.shape[:2] != action_features.shape[:2]:
            raise ValueError("action_mask shape must match first 2 dims of action_features.")

        batch_size = state_vec.shape[0]
        action_count = action_features.shape[1]

        state_latent = self.state_encoder(state_vec)
        if hidden is None:
            hidden = self.init_hidden(batch_size, device=state_vec.device)
        lstm_input = state_latent.unsqueeze(1)  # [B,1,H]
        lstm_output, hidden = self.state_lstm(lstm_input, hidden)
        state_latent = lstm_output[:, 0, :]

        action_flat = action_features.reshape(batch_size * action_count, self.action_dim)
        action_latent = self.action_encoder(action_flat).reshape(
            batch_size,
            action_count,
            self.action_hidden_size,
        )
        expanded_state = state_latent.unsqueeze(1).expand(-1, action_count, -1)
        policy_input = torch.cat([expanded_state, action_latent], dim=-1)
        logits = self.policy_head(policy_input).squeeze(-1)

        legal_mask = action_mask if action_mask.dtype == torch.bool else action_mask > 0.5
        logits = logits.masked_fill(~legal_mask, -1e9)
        values = self.value_head(state_latent).squeeze(-1)
        return logits, values, hidden

    def forward_sequence(
        self,
        *,
        state_vec: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward an entire rollout sequence for recurrent PPO updates.

        Args:
            state_vec: [T, state_dim]
            action_features: [T, N, action_dim]
            action_mask: [T, N]
            hidden: optional initial LSTM hidden state for batch size 1
        """
        if state_vec.ndim != 2:
            raise ValueError("state_vec must be [T, state_dim].")
        if action_features.ndim != 3:
            raise ValueError("action_features must be [T, N, action_dim].")
        if action_mask.ndim != 2:
            raise ValueError("action_mask must be [T, N].")
        if action_features.shape[0] != state_vec.shape[0]:
            raise ValueError("Sequence length mismatch between state_vec and action_features.")
        if action_mask.shape[:2] != action_features.shape[:2]:
            raise ValueError("action_mask shape must match first 2 dims of action_features.")

        step_count = state_vec.shape[0]
        action_count = action_features.shape[1]

        state_latent = self.state_encoder(state_vec)
        if hidden is None:
            hidden = self.init_hidden(1, device=state_vec.device)
        lstm_input = state_latent.unsqueeze(0)  # [1,T,H]
        lstm_output, hidden = self.state_lstm(lstm_input, hidden)
        state_latent = lstm_output.squeeze(0)

        action_flat = action_features.reshape(step_count * action_count, self.action_dim)
        action_latent = self.action_encoder(action_flat).reshape(
            step_count,
            action_count,
            self.action_hidden_size,
        )
        expanded_state = state_latent.unsqueeze(1).expand(-1, action_count, -1)
        policy_input = torch.cat([expanded_state, action_latent], dim=-1)
        logits = self.policy_head(policy_input).squeeze(-1)

        legal_mask = action_mask if action_mask.dtype == torch.bool else action_mask > 0.5
        logits = logits.masked_fill(~legal_mask, -1e9)
        values = self.value_head(state_latent).squeeze(-1)
        return logits, values, hidden
