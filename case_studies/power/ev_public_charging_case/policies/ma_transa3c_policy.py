"""Multi-agent Transformer Actor-Critic with shared spatial actor and global critics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from heron.core.observation import Observation


ACTION_EPS = 1e-6


def _squash_to_action(raw_action: torch.Tensor, action_hi: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    squashed = torch.tanh(raw_action)
    action = 0.5 * action_hi * (squashed + 1.0)
    return action, squashed


def _unsquash_from_action(action: torch.Tensor, action_hi: float = 0.8) -> torch.Tensor:
    squashed = (2.0 * action / action_hi) - 1.0
    squashed = squashed.clamp(-1.0 + ACTION_EPS, 1.0 - ACTION_EPS)
    return torch.atanh(squashed)


def _squashed_log_prob(
    dist: torch.distributions.Normal,
    raw_action: torch.Tensor,
    action_hi: float = 0.8,
) -> torch.Tensor:
    _, squashed = _squash_to_action(raw_action, action_hi=action_hi)
    scale = 0.5 * action_hi
    correction = torch.log(scale * (1.0 - squashed.pow(2)) + ACTION_EPS)
    return dist.log_prob(raw_action) - correction


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
        mean = self.actor_mean(center_token)
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
    """Shared spatial actor with optional dual centralized critics."""

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
        use_dual_critic: bool = False,
        dual_critic_w_profit: float = 1.0,
        dual_critic_w_stability: float = 0.3,
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
        self.use_dual_critic = use_dual_critic
        self.dual_critic_w_profit = float(dual_critic_w_profit)
        self.dual_critic_w_stability = float(dual_critic_w_stability)
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
        self.critic_profit = GlobalTransformerCritic(
            obs_dim=obs_dim,
            num_stations=num_stations,
            station_vocab_size=num_stations,
            station_embed_dim=station_embed_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        ).to(self.device)
        self.critic = self.critic_profit
        self.critic_stability = (
            GlobalTransformerCritic(
                obs_dim=obs_dim,
                num_stations=num_stations,
                station_vocab_size=num_stations,
                station_embed_dim=station_embed_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
            ).to(self.device)
            if use_dual_critic
            else None
        )

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)
        self._critic_params = list(self.critic_profit.parameters())
        if self.critic_stability is not None:
            self._critic_params.extend(self.critic_stability.parameters())
        self.critic_optimizer = torch.optim.AdamW(self._critic_params, lr=critic_lr, weight_decay=1e-4)

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

    @staticmethod
    def _normalize_advantage(advantage: torch.Tensor) -> torch.Tensor:
        if advantage.numel() <= 1:
            return advantage
        return (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

    def critic_state_dict(self) -> dict:
        state = {"profit": self.critic_profit.state_dict()}
        if self.critic_stability is not None:
            state["stability"] = self.critic_stability.state_dict()
        return state

    def load_critic_state_dict(self, state: dict) -> None:
        if "profit" not in state:
            self.critic_profit.load_state_dict(state)
            return
        self.critic_profit.load_state_dict(state["profit"])
        if self.critic_stability is not None and "stability" in state:
            self.critic_stability.load_state_dict(state["stability"])

    def training_state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic_state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "dual_critic_w_profit": float(self.dual_critic_w_profit),
            "dual_critic_w_stability": float(self.dual_critic_w_stability),
        }

    def load_training_state_dict(self, state: dict, load_optimizers: bool = True) -> None:
        self.actor.load_state_dict(state["actor"])
        self.load_critic_state_dict(state["critic"])
        self.dual_critic_w_profit = float(state.get("dual_critic_w_profit", self.dual_critic_w_profit))
        self.dual_critic_w_stability = float(state.get("dual_critic_w_stability", self.dual_critic_w_stability))
        if load_optimizers:
            if "actor_optimizer" in state:
                self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            if "critic_optimizer" in state:
                self.critic_optimizer.load_state_dict(state["critic_optimizer"])

    def adjust_dual_critic_weights(
        self,
        val_qvol: float,
        val_network_viol: float,
        target_qvol: float = 4.2785,
        target_network_viol: float = 880.0,
    ) -> dict[str, float]:
        if not self.use_dual_critic:
            return {}

        old_w_profit = self.dual_critic_w_profit
        old_w_stability = self.dual_critic_w_stability
        stability_ok = val_qvol < target_qvol and val_network_viol < target_network_viol
        if stability_ok:
            self.dual_critic_w_profit = min(self.dual_critic_w_profit * 1.10, 2.0)
            self.dual_critic_w_stability = max(self.dual_critic_w_stability * 0.95, 0.15)
        else:
            self.dual_critic_w_profit = max(self.dual_critic_w_profit * 0.95, 0.8)
            self.dual_critic_w_stability = min(self.dual_critic_w_stability * 1.10, 1.0)

        return {
            "dual_weight_profit_old": float(old_w_profit),
            "dual_weight_stability_old": float(old_w_stability),
            "dual_weight_profit": float(self.dual_critic_w_profit),
            "dual_weight_stability": float(self.dual_critic_w_stability),
            "dual_weight_stability_ok": 1.0 if stability_ok else 0.0,
        }

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
                action, _ = _squash_to_action(mean, action_hi=self.actor.action_hi)
            else:
                dist = torch.distributions.Normal(mean, std)
                raw_action = dist.rsample()
                action, _ = _squash_to_action(raw_action, action_hi=self.actor.action_hi)
        return np.array([float(action.item())], dtype=np.float32)

    def update(
        self,
        neighbor_seqs: np.ndarray,
        neighbor_station_indices: np.ndarray,
        center_station_indices: np.ndarray,
        global_obs: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        profit_returns: np.ndarray | None = None,
        stability_returns: np.ndarray | None = None,
        rank_global_obs: np.ndarray | None = None,
        rank_valid_mask: np.ndarray | None = None,
    ):
        self.actor.train()
        self.critic_profit.train()
        if self.critic_stability is not None:
            self.critic_stability.train()

        neighbor_t = torch.as_tensor(neighbor_seqs, dtype=torch.float32, device=self.device)
        neighbor_station_t = torch.as_tensor(neighbor_station_indices, dtype=torch.long, device=self.device)
        center_station_t = torch.as_tensor(center_station_indices, dtype=torch.long, device=self.device)
        global_t = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device).view(-1)
        profit_returns_t = torch.as_tensor(
            profit_returns if profit_returns is not None else returns,
            dtype=torch.float32,
            device=self.device,
        ).view(-1)
        stability_returns_t = torch.as_tensor(
            stability_returns if stability_returns is not None else returns,
            dtype=torch.float32,
            device=self.device,
        ).view(-1)
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
        raw_action_flat = _unsquash_from_action(action_flat, action_hi=self.actor.action_hi)
        log_probs = _squashed_log_prob(dist, raw_action_flat, action_hi=self.actor.action_hi).view(steps, num_stations)
        mean_actions, _ = _squash_to_action(mean, action_hi=self.actor.action_hi)
        mean_matrix = mean_actions.view(steps, num_stations)
        entropy = dist.entropy().mean()

        critic_station_idx = torch.as_tensor(self.global_station_indices, dtype=torch.long, device=self.device).view(1, self.num_stations).expand(steps, -1)
        profit_values = self.critic_profit(global_t, critic_station_idx)
        critic_loss_stability = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.use_dual_critic and self.critic_stability is not None:
            stability_values = self.critic_stability(global_t, critic_station_idx)
            profit_advantages = self._normalize_advantage(profit_returns_t - profit_values.detach())
            stability_advantages = self._normalize_advantage(stability_returns_t - stability_values.detach())
            advantages = (
                self.dual_critic_w_profit * profit_advantages
                + self.dual_critic_w_stability * stability_advantages
            )
            advantages = self._normalize_advantage(advantages)
            critic_loss_profit = F.mse_loss(profit_values, profit_returns_t)
            critic_loss_stability = F.mse_loss(stability_values, stability_returns_t)
            critic_loss = critic_loss_profit + critic_loss_stability
        else:
            advantages = self._normalize_advantage(returns_t - profit_values.detach())
            critic_loss_profit = F.mse_loss(profit_values, returns_t)
            critic_loss = critic_loss_profit
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
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self._critic_params, 1.0)
        self.critic_optimizer.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "actor_loss_pg": float(actor_loss_pg.item()),
            "critic_loss": float(critic_loss.item()),
            "critic_profit_loss": float(critic_loss_profit.item()),
            "critic_stability_loss": float(critic_loss_stability.item()),
            "dual_weight_profit": float(self.dual_critic_w_profit),
            "dual_weight_stability": float(self.dual_critic_w_stability),
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
