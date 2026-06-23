"""MAPPO-MLP baseline for multi-station EV charging pricing."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from heron.core.observation import Observation


class MLPActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_stations: int,
        station_embed_dim: int = 8,
        hidden_dim: int = 128,
        action_hi: float = 0.8,
    ):
        super().__init__()
        self.action_hi = action_hi
        self.station_embed = nn.Embedding(num_stations, station_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + station_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.tensor([-1.0], dtype=torch.float32))

    def forward(self, obs: torch.Tensor, station_idx: torch.Tensor):
        emb = self.station_embed(station_idx)
        x = torch.cat([obs, emb], dim=-1)
        mean = torch.sigmoid(self.net(x)).squeeze(-1) * self.action_hi
        std = torch.exp(self.log_std).clamp(1e-3, 0.3).expand_as(mean)
        return mean, std


class GlobalMLPCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_stations: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * num_stations, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs: torch.Tensor):
        x = global_obs.reshape(global_obs.shape[0], -1)
        return self.net(x).squeeze(-1)


class MAPPOPolicy:
    """CTDE PPO baseline with an MLP actor and centralized critic."""

    observation_mode = "ctde_mlp"

    def __init__(
        self,
        obs_dim: int = 10,
        num_stations: int = 5,
        hidden_dim: int = 128,
        station_embed_dim: int = 8,
        actor_lr: float = 1e-4,
        critic_lr: float = 5e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        seed: int = 42,
        device: str | None = None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.obs_dim = obs_dim
        self.num_stations = num_stations
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.action_range = (0.0, 0.8)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.actor = MLPActor(
            obs_dim=obs_dim,
            num_stations=num_stations,
            station_embed_dim=station_embed_dim,
            hidden_dim=hidden_dim,
            action_hi=0.8,
        ).to(self.device)
        self.critic = GlobalMLPCritic(
            obs_dim=obs_dim,
            num_stations=num_stations,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=actor_lr,
            weight_decay=1e-4,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=critic_lr,
            weight_decay=1e-4,
        )

    def extract_obs_vector(self, observation, obs_dim: int = 8) -> np.ndarray:
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

    def act(
        self,
        obs_vec: np.ndarray,
        station_index: int,
        deterministic: bool = False,
        return_log_prob: bool = False,
    ):
        self.actor.eval()
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        idx_t = torch.as_tensor([station_index], dtype=torch.long, device=self.device)

        with torch.no_grad():
            mean, std = self.actor(obs_t, idx_t)
            dist = torch.distributions.Normal(mean, std)
            action = mean if deterministic else dist.sample()
            action = action.clamp(0.0, 0.8)
            log_prob = dist.log_prob(action)

        action_np = np.array([float(action.item())], dtype=np.float32)
        if return_log_prob:
            return action_np, float(log_prob.item())
        return action_np

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
    ):
        self.actor.train()
        self.critic.train()

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device).view(-1)

        steps, num_stations, obs_dim = obs_t.shape
        station_idx = torch.arange(num_stations, dtype=torch.long, device=self.device)
        station_idx = station_idx.view(1, num_stations).expand(steps, num_stations)

        stats = {}
        for _ in range(self.ppo_epochs):
            values = self.critic(obs_t)
            advantages = returns_t - values.detach()
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            flat_obs = obs_t.reshape(steps * num_stations, obs_dim)
            flat_idx = station_idx.reshape(steps * num_stations)
            mean, std = self.actor(flat_obs, flat_idx)
            dist = torch.distributions.Normal(mean, std)

            flat_actions = actions_t.reshape(steps * num_stations)
            new_log = dist.log_prob(flat_actions).view(steps, num_stations)
            entropy = dist.entropy().mean()

            adv_matrix = advantages.view(steps, 1).expand(steps, num_stations)
            ratio = torch.exp(new_log - old_log_t)
            surrogate_1 = ratio * adv_matrix
            surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_matrix

            actor_loss = -torch.min(surrogate_1, surrogate_2).mean() - self.entropy_coef * entropy
            critic_loss = F.mse_loss(values, returns_t)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            approx_kl = (old_log_t - new_log).mean().detach()
            stats = {
                "actor_loss": float(actor_loss.item()),
                "critic_loss": float(critic_loss.item()),
                "entropy": float(entropy.item()),
                "approx_kl": float(approx_kl.item()),
            }

        return stats
