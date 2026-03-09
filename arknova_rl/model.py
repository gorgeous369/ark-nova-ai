"""Masked actor-critic model with MLP+LSTM backbone."""

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
        global_state_dim: int,
        hidden_size: int = 256,
        lstm_size: int = 128,
        action_hidden_size: int = 128,
        use_lstm: bool = True,
        use_centralized_value: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.global_state_dim = int(global_state_dim)
        self.hidden_size = int(hidden_size)
        self.lstm_size = int(lstm_size)
        self.action_hidden_size = int(action_hidden_size)
        self.use_lstm = bool(use_lstm)
        self.use_centralized_value = bool(use_centralized_value)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        if self.use_lstm:
            self.state_lstm = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.lstm_size,
                batch_first=True,
            )
            state_latent_dim = self.lstm_size
        else:
            self.state_lstm = None
            state_latent_dim = self.hidden_size

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

        if self.use_centralized_value:
            self.global_encoder = nn.Sequential(
                nn.Linear(self.global_state_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
            )
            value_input_dim = state_latent_dim + self.hidden_size
        else:
            self.global_encoder = None
            value_input_dim = state_latent_dim

        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def init_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.use_lstm:
            return None
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
        global_state_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward one environment step.

        Args:
            state_vec: [B, state_dim]
            action_features: [B, N, action_dim]
            action_mask: [B, N] (1 for legal, 0 for invalid)
            hidden: optional LSTM hidden state
            global_state_vec: [B, global_state_dim] for centralized critic
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
        if self.use_lstm:
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

        legal_mask = action_mask > 0.5
        logits = logits.masked_fill(~legal_mask, -1e9)

        if self.use_centralized_value:
            if global_state_vec is None:
                raise ValueError("global_state_vec is required when use_centralized_value=True.")
            global_latent = self.global_encoder(global_state_vec)
            value_input = torch.cat([state_latent, global_latent], dim=-1)
        else:
            value_input = state_latent
        values = self.value_head(value_input).squeeze(-1)
        return logits, values, hidden

