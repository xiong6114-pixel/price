"""Multi-agent Transformer Actor-Critic with shared spatial actor and global critic."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from heron.core.observation import Observation


class SpatialTransformerActor(nn.Module):
    """Shared actor over one station plus its spatially ordered neighbors."""

    def __init__(
        self,
        obs_dim: int = 8,
        seq_len: int = 5,
        station_vocab_size: int = 5,
        station_embed_dim: int = 8,
        neighbor_rank_embed_dim: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        action_hi: float = 0.8,
    ):
        super().__init__()
        self.action_hi = action_hi
        self.station_embed = nn.Embedding(station_vocab_size, station_embed_dim)
        self.rank_embed = nn.Embedding(seq_len, neighbor_rank_embed_dim)
        self.input_proj = nn.Linear(obs_dim + station_embed_dim + neighbor_rank_embed_dim + 1, d_model)
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

    def forward(
        self,
        seq: torch.Tensor,
        station_indices: torch.Tensor,
        rank_indices: torch.Tensor,
        is_self_flags: torch.Tensor,
    ):
        station_emb = self.station_embed(station_indices)
        rank_emb = self.rank_embed(rank_indices)
        actor_input = torch.cat([seq, station_emb, rank_emb, is_self_flags.unsqueeze(-1)], dim=-1)
        x = self.input_proj(actor_input) + self.pos_embed[:, : seq.shape[1], :]
        h = self.encoder(x)
        center_token = h[:, 0, :]
        mean = torch.sigmoid(self.actor_mean(center_token)) * self.action_hi
        std = torch.exp(self.actor_log_std).clamp(1e-3, 0.3)
        return mean.squeeze(-1), std.squeeze(-1)


class GlobalTransformerCritic(nn.Module):
    """Centralized critic over the full station observation matrix."""

    def __init__(
        self,
        obs_dim: int = 8,
        num_stations: int = 5,
        station_vocab_size: int = 5,
        station_embed_dim: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.station_embed = nn.Embedding(station_vocab_size, station_embed_dim)
        self.input_proj = nn.Linear(obs_dim + station_embed_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_stations, d_model))

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
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, global_obs: torch.Tensor, station_indices: torch.Tensor):
        station_emb = self.station_embed(station_indices)
        critic_input = torch.cat([global_obs, station_emb], dim=-1)
        x = self.input_proj(critic_input) + self.pos_embed[:, : global_obs.shape[1], :]
        h = self.encoder(x)
        pooled = h.mean(dim=1)
        return self.value_head(pooled).squeeze(-1)


class MATransA3CPolicy:
    """Shared spatial actor with a centralized global critic."""

    observation_mode = "neighbor_spatial"

    def __init__(
        self,
        obs_dim: int = 8,
        seq_len: int = 5,
        num_stations: int = 5,
        station_ids: list[str] | None = None,
        station_embed_dim: int = 8,
        neighbor_rank_embed_dim: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        actor_lr: float = 5e-5,
        critic_lr: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        lambda_rank: float = 0.0,
        rank_margin: float = 0.02,
        rank_eps: float = 0.10,
        lambda_response: float = 0.0,
        response_margin: float = 0.01,
        response_threshold: float = 0.02,
        response_ema_alpha: float = 0.9,
        lambda_anchor: float = 0.0,
        price_anchor: float = 0.40,
        seed: int = 42,
        device: str | None = None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.num_stations = num_stations
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.lambda_rank = lambda_rank
        self.rank_margin = rank_margin
        self.rank_eps = rank_eps
        self.lambda_response = lambda_response
        self.response_margin = response_margin
        self.response_threshold = response_threshold
        self.response_ema_alpha = response_ema_alpha
        self.lambda_anchor = lambda_anchor
        self.price_anchor = price_anchor
        self.action_range = (0.0, 0.8)
        self.station_ids = station_ids or [f"station_{i}" for i in range(num_stations)]
        self.station_to_index = {
            station_id: index for index, station_id in enumerate(self.station_ids)
        }
        self.global_station_indices = np.asarray(
            [self.station_to_index[station_id] for station_id in self.station_ids],
            dtype=np.int64,
        )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = SpatialTransformerActor(
            obs_dim=obs_dim,
            seq_len=seq_len,
            station_vocab_size=num_stations,
            station_embed_dim=station_embed_dim,
            neighbor_rank_embed_dim=neighbor_rank_embed_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            action_hi=0.8,
        ).to(self.device)
        self.critic = GlobalTransformerCritic(
            obs_dim=obs_dim,
            num_stations=num_stations,
            station_vocab_size=num_stations,
            station_embed_dim=station_embed_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        ).to(self.device)

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=1e-4)

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
        neighbor_seq: np.ndarray,
        neighbor_station_indices: np.ndarray,
        center_station_index: int,
        deterministic: bool = False,
    ) -> np.ndarray:
        self.actor.eval()
        seq_t = torch.as_tensor(neighbor_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        station_idx_t = torch.as_tensor(neighbor_station_indices, dtype=torch.long, device=self.device).unsqueeze(0)
        rank_idx_t = torch.arange(seq_t.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)
        is_self_t = (station_idx_t == int(center_station_index)).to(torch.float32)
        with torch.no_grad():
            mean, std = self.actor(seq_t, station_idx_t, rank_idx_t, is_self_t)
            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
        action = action.clamp(0.0, 0.8)
        return np.array([float(action.item())], dtype=np.float32)

    def update(
        self,
        neighbor_seqs: np.ndarray,
        neighbor_station_indices: np.ndarray,
        center_station_indices: np.ndarray,
        global_obs: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        rank_global_obs: np.ndarray | None = None,
        rank_valid_mask: np.ndarray | None = None,
    ):
        self.actor.train()
        self.critic.train()

        neighbor_t = torch.as_tensor(neighbor_seqs, dtype=torch.float32, device=self.device)
        neighbor_station_t = torch.as_tensor(neighbor_station_indices, dtype=torch.long, device=self.device)
        center_station_t = torch.as_tensor(center_station_indices, dtype=torch.long, device=self.device)
        global_t = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device).view(-1)
        rank_global_t = (
            torch.as_tensor(rank_global_obs, dtype=torch.float32, device=self.device)
            if rank_global_obs is not None
            else global_t
        )
        rank_valid_t = (
            torch.as_tensor(rank_valid_mask, dtype=torch.bool, device=self.device).view(-1)
            if rank_valid_mask is not None
            else torch.ones(global_t.shape[0], dtype=torch.bool, device=self.device)
        )

        steps, num_stations, seq_len, obs_dim = neighbor_t.shape
        rank_idx_t = torch.arange(seq_len, dtype=torch.long, device=self.device).view(1, 1, seq_len).expand(steps, num_stations, seq_len)
        is_self_t = (neighbor_station_t == center_station_t.unsqueeze(-1)).to(torch.float32)
        mean, std = self.actor(
            neighbor_t.view(steps * num_stations, seq_len, obs_dim),
            neighbor_station_t.view(steps * num_stations, seq_len),
            rank_idx_t.reshape(steps * num_stations, seq_len),
            is_self_t.view(steps * num_stations, seq_len),
        )
        dist = torch.distributions.Normal(mean, std)
        action_flat = actions_t.view(-1)
        log_probs = dist.log_prob(action_flat).view(steps, num_stations)
        mean_matrix = mean.view(steps, num_stations)
        entropy = dist.entropy().mean()

        critic_station_idx = torch.as_tensor(self.global_station_indices, dtype=torch.long, device=self.device).view(1, self.num_stations).expand(steps, -1)
        values = self.critic(global_t, critic_station_idx)
        advantages = returns_t - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        global_adv = advantages.unsqueeze(1).expand(-1, num_stations)

        actor_loss_pg = -(log_probs * global_adv).mean() - self.entropy_coef * entropy
        rank_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        response_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        anchor_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        rank_pair_count = 0
        rank_violation_count = 0
        response_trigger_count = 0
        response_violation_count = 0
        mu_std_across_stations = mean_matrix.std(dim=1, unbiased=False).mean()
        score_std_across_stations = torch.zeros((), dtype=torch.float32, device=self.device)
        mean_price_mean = mean_matrix.mean()
        if self.lambda_rank > 0.0 and rank_global_t.shape[-1] >= 9 and torch.any(rank_valid_t):
            congestion_score = (
                0.5 * rank_global_t[:, :, 4]
                + 0.3 * rank_global_t[:, :, 5]
                + 0.2 * rank_global_t[:, :, 8]
            ).detach()
            valid_scores = congestion_score[rank_valid_t]
            valid_means = mean_matrix[rank_valid_t]
            score_std_across_stations = valid_scores.std(dim=1, unbiased=False).mean()
            pair_losses = []
            for i in range(num_stations):
                for j in range(num_stations):
                    if i == j:
                        continue
                    mask = valid_scores[:, i] > (valid_scores[:, j] + self.rank_eps)
                    if torch.any(mask):
                        diff = valid_means[mask, i] - valid_means[mask, j]
                        rank_pair_count += int(mask.sum().item())
                        rank_violation_count += int((diff < self.rank_margin).sum().item())
                        pair_losses.append(F.relu(self.rank_margin - diff))
            if pair_losses:
                rank_loss = torch.cat(pair_losses).mean()

        if self.lambda_response > 0.0 and global_t.shape[-1] >= 9 and steps > 1:
            response_score = (
                0.5 * global_t[:, :, 4]
                + 0.3 * global_t[:, :, 5]
                + 0.2 * global_t[:, :, 8]
            ).detach()
            ema_score = torch.zeros_like(response_score)
            ema_score[0] = response_score[0]
            for t in range(1, steps):
                ema_score[t] = (
                    self.response_ema_alpha * ema_score[t - 1]
                    + (1.0 - self.response_ema_alpha) * response_score[t]
                )
            congestion_gap_prev = response_score[:-1] - ema_score[:-1]
            mu_delta = mean_matrix[1:] - mean_matrix[:-1]
            up_mask = congestion_gap_prev > self.response_threshold
            if torch.any(up_mask):
                response_trigger_count = int(up_mask.sum().item())
                response_violation_count = int((mu_delta[up_mask] < self.response_margin).sum().item())
                response_loss = F.relu(self.response_margin - mu_delta[up_mask]).mean()

        if self.lambda_anchor > 0.0:
            anchor_target = torch.full(
                (steps,),
                float(self.price_anchor),
                dtype=torch.float32,
                device=self.device,
            )
            # Treat the anchor as a price floor so high-price steps are not pushed back down.
            anchor_gap = F.relu(anchor_target - mean_matrix.mean(dim=1))
            anchor_loss = anchor_gap.pow(2).mean()

        actor_loss = (
            actor_loss_pg
            + self.lambda_rank * rank_loss
            + self.lambda_response * response_loss
            + self.lambda_anchor * anchor_loss
        )
        critic_loss = F.mse_loss(values, returns_t)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "actor_loss_pg": float(actor_loss_pg.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "actor_mean_price": float(mean_matrix.detach().mean().item()),
            "rank_loss": float(rank_loss.item()),
            "rank_loss_weighted": float((self.lambda_rank * rank_loss).item()),
            "rank_pair_count": float(rank_pair_count),
            "rank_violation_count": float(rank_violation_count),
            "rank_violation_rate": (
                float(rank_violation_count) / float(rank_pair_count)
                if rank_pair_count > 0
                else 0.0
            ),
            "response_loss": float(response_loss.item()),
            "response_loss_weighted": float((self.lambda_response * response_loss).item()),
            "response_trigger_count": float(response_trigger_count),
            "response_violation_count": float(response_violation_count),
            "response_violation_rate": (
                float(response_violation_count) / float(response_trigger_count)
                if response_trigger_count > 0
                else 0.0
            ),
            "anchor_loss": float(anchor_loss.item()),
            "anchor_loss_weighted": float((self.lambda_anchor * anchor_loss).item()),
            "mu_std_across_stations": float(mu_std_across_stations.item()),
            "score_std_across_stations": float(score_std_across_stations.item()),
            "mean_price_mean": float(mean_price_mean.item()),
        }
