"""Independent Transformer Actor-Critic baseline for single-station control."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from heron.core.observation import Observation


class TemporalTransformerActorCritic(nn.Module):
    """Temporal Transformer encoder with independent actor and critic heads."""

    def __init__(
        self,
        obs_dim: int = 8,
        seq_len: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        action_hi: float = 0.8,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.action_hi = action_hi

        self.input_proj = nn.Linear(obs_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.actor_mean = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.actor_log_std = nn.Parameter(torch.tensor([-1.0]))

        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, seq: torch.Tensor):
        """Run one forward pass.

        Args:
            seq: Tensor shaped [B, L, obs_dim].
        """
        x = self.input_proj(seq) + self.pos_embed[:, : seq.shape[1], :]
        h = self.encoder(x)
        last = h[:, -1, :]

        mean = torch.sigmoid(self.actor_mean(last)) * self.action_hi
        std = torch.exp(self.actor_log_std).clamp(1e-3, 0.3)

        value = self.critic(last).squeeze(-1)
        return mean.squeeze(-1), std.squeeze(-1), value


class IndependentTransA3CPolicy:
    """Independent temporal Transformer actor-critic policy."""

    observation_mode = "local"

    def __init__(
        self,
        obs_dim: int = 8,
        seq_len: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        actor_lr: float = 1e-4,
        critic_lr: float = 5e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        seed: int = 42,
        device: str | None = None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.action_range = (0.0, 0.8)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = TemporalTransformerActorCritic(
            obs_dim=obs_dim,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            action_hi=0.8,
        ).to(self.device)

        actor_params = (
            list(self.model.input_proj.parameters())
            + [self.model.pos_embed]
            + list(self.model.encoder.parameters())
            + list(self.model.actor_mean.parameters())
            + [self.model.actor_log_std]
        )
        critic_params = list(self.model.critic.parameters())

        self.optimizer = torch.optim.AdamW(
            [
                {"params": actor_params, "lr": actor_lr},
                {"params": critic_params, "lr": critic_lr},
            ],
            weight_decay=1e-4,
        )

    def extract_obs_vector(self, observation, obs_dim: int = 8) -> np.ndarray:
        """Flatten local observations into a fixed-width vector."""
        if isinstance(observation, Observation):
            local = observation.local
            if isinstance(local, dict):
                if "obs" in local:
                    arr = np.asarray(local["obs"], dtype=np.float32).reshape(-1)
                    return self._fit_dim(arr, obs_dim)
                parts = []
                for value in local.values():
                    parts.extend(np.asarray(value, dtype=np.float32).reshape(-1).tolist())
                return self._fit_dim(np.asarray(parts, dtype=np.float32), obs_dim)

        if isinstance(observation, np.ndarray):
            return self._fit_dim(observation.astype(np.float32).reshape(-1), obs_dim)

        return np.zeros(obs_dim, dtype=np.float32)

    @staticmethod
    def _fit_dim(arr: np.ndarray, dim: int) -> np.ndarray:
        out = np.zeros(dim, dtype=np.float32)
        n = min(dim, arr.size)
        out[:n] = arr[:n]
        return out

    def act(self, seq: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample or greedily choose a pricing action for one station."""
        self.model.eval()
        seq_t = torch.as_tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            mean, std, _value = self.model(seq_t)
            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()

        action = action.clamp(0.0, 0.8)
        return np.array([float(action.item())], dtype=np.float32)

    def update(self, seqs: np.ndarray, actions: np.ndarray, returns: np.ndarray):
        """Update actor and critic from one station's rollout batch."""
        self.model.train()

        seqs_t = torch.as_tensor(seqs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device).view(-1)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device).view(-1)

        mean, std, values = self.model(seqs_t)
        dist = torch.distributions.Normal(mean, std)

        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        advantages = returns_t - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        actor_loss = -(log_probs * advantages).mean() - self.entropy_coef * entropy
        critic_loss = F.mse_loss(values, returns_t)
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
        }
