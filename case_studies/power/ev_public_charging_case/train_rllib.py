
import argparse
import csv
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from heron.core.observation import Observation
from heron.core.action import Action

from case_studies.power.ev_public_charging_case.agents import ChargingSlot, StationCoordinator
from case_studies.power.ev_public_charging_case.envs.charging_env import ChargingEnv
from case_studies.power.ev_public_charging_case.policies import (
    IndependentTransA3CPolicy,
    MAPPOPolicy,
    MATransA3CPolicy,
    PricingPolicy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Environment Factory
# ============================================================================
def get_soc_calibrated_config() -> Dict[str, Any]:
    """Shared SOC-calibrated environment config for baseline evaluation."""
    return {
        "num_stations": 5,
        "num_chargers": 10,
        "p_max_kw": 75.0,
        "arrival_rate": 65.0,
        "dt": 900.0,
        "episode_length": 86400.0,
        "center_demand_prob": 0.92,
        "center_sigma": 0.42,
        "edge_sigma": 0.50,
        "travel_speed_kmph": 40.0,
        "omega_travel": 0.5,
        "omega_wait": 1.0,
        "omega_price": 1.0,
        "dsafe_km": 2.0,
        "mean_charge_time_min": 60.0,
        "max_queue_size": 12,
        "q_threshold": 4,
        "eta": 1.0,
        "generalized_cost_threshold": 90.0,
    }


def get_paper_aligned_config(
    q_threshold: float = 6.0,
    max_queue_size: int = 15,
    generalized_cost_threshold: float = 90.0,
    lmp_base: float = 0.45,
    lmp_amp: float = 0.10,
) -> Dict[str, Any]:
    """Paper-aligned environment config kept separate from the SOC mainline."""
    cfg = get_soc_calibrated_config().copy()
    cfg.update({
        "q_threshold": q_threshold,
        "max_queue_size": max_queue_size,
        "generalized_cost_threshold": generalized_cost_threshold,
        "lmp_base": lmp_base,
        "lmp_amp": lmp_amp,
    })
    return cfg


def iter_paper_aligned_grid():
    """Small environment grid for paper-aligned reproduction."""
    candidates = [
        ("A_q6_max15_gc90", 6.0, 15, 90.0),
        ("B_q8_max15_gc90", 8.0, 15, 90.0),
        ("C_q6_max15_gc80", 6.0, 15, 80.0),
        ("D_q8_max15_gc80", 8.0, 15, 80.0),
    ]
    for name, q_th, q_max, gc_th in candidates:
        yield name, get_paper_aligned_config(
            q_threshold=q_th,
            max_queue_size=q_max,
            generalized_cost_threshold=gc_th,
            lmp_base=0.45,
            lmp_amp=0.10,
        )


def get_paper_response_config(
    q_threshold: float = 4.0,
    max_queue_size: int = 12,
    generalized_cost_threshold: float = 120.0,
    omega_price: float = 0.35,
    choice_lmp_weight: float = 0.0,
    charge_lmp_weight: float = 0.0,
) -> Dict[str, Any]:
    """Response-calibrated config that preserves high-LMP accounting scale."""
    cfg = get_paper_aligned_config(
        q_threshold=q_threshold,
        max_queue_size=max_queue_size,
        generalized_cost_threshold=generalized_cost_threshold,
        lmp_base=0.45,
        lmp_amp=0.10,
    )
    cfg.update({
        "omega_price": omega_price,
        "choice_lmp_weight": choice_lmp_weight,
        "charge_lmp_weight": charge_lmp_weight,
        "enable_in_session_price_response": False,
        "min_charge_power_frac": 0.35,
    })
    return cfg


def get_paper_response_F_p30_arr50_config() -> Dict[str, Any]:
    """Primary response-calibrated config selected from smoke probes."""
    cfg = get_paper_response_config(
        q_threshold=5.0,
        max_queue_size=12,
        generalized_cost_threshold=120.0,
        omega_price=0.35,
        choice_lmp_weight=0.0,
        charge_lmp_weight=0.0,
    )
    cfg.update({
        "p_max_kw": 30.0,
        "arrival_rate": 50.0,
    })
    return cfg


def get_paper_response_F_p30_arr50_eta4_config() -> Dict[str, Any]:
    """Response-calibrated config with stronger queue penalty for training."""
    cfg = get_paper_response_F_p30_arr50_config()
    cfg.update({
        "eta": 4.0,
    })
    return cfg


# Backward-compatible alias for earlier helper name.
def get_paper_response_f_p30_arr50() -> Dict[str, Any]:
    return get_paper_response_F_p30_arr50_config()


def iter_paper_response_grid():
    """Focused response-calibrated candidate set after smoke probes."""
    candidates = [
        (
            "paper_response_F_p30_arr50_eta4",
            get_paper_response_F_p30_arr50_eta4_config(),
        ),
        (
            "paper_response_F_p30_arr45",
            {
                **get_paper_response_F_p30_arr50_config(),
                "arrival_rate": 45.0,
            },
        ),
        (
            "paper_response_F_p30_arr50_q4",
            {
                **get_paper_response_F_p30_arr50_config(),
                "q_threshold": 4.0,
            },
        ),
    ]
    for name, cfg in candidates:
        yield name, cfg


def create_charging_env(config: Dict[str, Any] = None) -> ChargingEnv:
  
    config = config or {}
    num_stations = config.get("num_stations", 5)
    num_chargers = config.get("num_chargers", 10)
    p_max_kw = config.get("p_max_kw", 60.0)
    arrival_rate = config.get("arrival_rate", 30.0)
    dt = config.get("dt", 900.0)
    episode_length = config.get("episode_length", 86400.0)
    q_threshold = config.get("q_threshold", 8.0)
    eta = config.get("eta", 1.0)
    env_kwargs = {
        "travel_speed_kmph": config.get("travel_speed_kmph", 40.0),
        "omega_travel": config.get("omega_travel", 0.5),
        "omega_wait": config.get("omega_wait", 1.0),
        "omega_price": config.get("omega_price", 1.0),
        "dsafe_km": config.get("dsafe_km", 2.0),
        "mean_charge_time_min": config.get("mean_charge_time_min", 30.0),
        "max_queue_size": config.get("max_queue_size", 20),
        "center_demand_prob": config.get("center_demand_prob", 0.8),
        "center_sigma": config.get("center_sigma", 0.55),
        "edge_sigma": config.get("edge_sigma", 0.5),
        "generalized_cost_threshold": config.get("generalized_cost_threshold", 60.0),
        "lmp_base": config.get("lmp_base", 0.20),
        "lmp_amp": config.get("lmp_amp", 0.10),
        "choice_lmp_weight": config.get("choice_lmp_weight", 1.0),
        "charge_lmp_weight": config.get("charge_lmp_weight", 1.0),
        "enable_in_session_price_response": config.get("enable_in_session_price_response", True),
        "min_charge_power_frac": config.get("min_charge_power_frac", 0.25),
        "station_positions": config.get("station_positions"),
    }

    coordinators: List[StationCoordinator] = []
    for i in range(num_stations):
        s_id = f"station_{i}"
        slots = {
            f"{s_id}_slot_{j}": ChargingSlot(agent_id=f"{s_id}_slot_{j}", p_max_kw=p_max_kw)
            for j in range(num_chargers)
        }
        coordinators.append(StationCoordinator(agent_id=s_id, subordinates=slots))

    return ChargingEnv(
        coordinator_agents=coordinators,
        arrival_rate=arrival_rate,
        dt=dt,
        episode_length=episode_length,
        q_threshold=q_threshold,
        eta=eta,
        **env_kwargs,
    )


def fixed_price_by_time(time_s: float) -> float:
    """Paper-style fixed pricing schedule."""
    hour = (time_s % 86400.0) / 3600.0

    if 0.0 <= hour < 8.0:
        return 0.25
    if 8.0 <= hour < 16.0:
        return 0.11
    return 0.08


# ============================================================================
# CTDE Training with PricingPolicy
# ============================================================================
def train_simple(
    num_episodes: int = 50,
    steps_per_episode: int = 96,
    seed: int = 42,
    gamma: float = 0.99,
    lr: float = 0.01,
    env_config: Dict[str, Any] = None,
    output_dir: str = "outputs",
    algo: str = "I-AC-MLP",
) -> Tuple[ChargingEnv, Dict[str, PricingPolicy], List[float]]:
   
    np.random.seed(seed)
    env = create_charging_env(env_config)

    station_ids = [
        aid for aid, agent in env.registered_agents.items()
        if isinstance(agent, StationCoordinator)
    ]
    logger.info(f"Station agents: {station_ids}")

    policies = {
        sid: PricingPolicy(obs_dim=8, action_dim=1, hidden_dim=32, seed=seed + i)
        for i, sid in enumerate(station_ids)
    }

    returns_history: List[float] = []
    episode_logs: List[Dict[str, Any]] = []
    step_logs: List[Dict[str, Any]] = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        trajectories = {sid: {"obs": [], "actions": [], "rewards": []} for sid in station_ids}
        episode_reward = {sid: 0.0 for sid in station_ids}

        for step in range(steps_per_episode):
            actions = {}
            for sid in station_ids:
                obs_value = obs[sid]
                if isinstance(obs_value, Observation):
                    observation = Observation(timestamp=step, local=obs_value.local)
                elif isinstance(obs_value, np.ndarray):
                    observation = Observation(timestamp=step, local={"obs": obs_value[:8]})
                else:
                    observation = Observation(timestamp=step, local={"obs": np.zeros(8, dtype=np.float32)})

                action = policies[sid].forward(observation)
                actions[sid] = action

                obs_vec = policies[sid].extract_obs_vector(observation, 8)
                reg_signal = float(obs_vec[5])
                headroom_up = float(obs_vec[6])
                headroom_down = float(obs_vec[7])
                if episode == 0 and step % 20 == 0:
                    logger.info(
                        f"[{sid}] t_step={step:03d} reg={reg_signal:+.3f} "
                        f"headroom_up={headroom_up:.3f} headroom_down={headroom_down:.3f}"
                    )
                trajectories[sid]["obs"].append(obs_vec)
                trajectories[sid]["actions"].append(action.c.copy())

            obs, rewards, terminated, truncated, infos = env.step(actions)
            station_metrics = infos.get("__all__", {}).get("station_metrics", {})

            for sid in station_ids:
                r = rewards.get(sid, 0.0)
                trajectories[sid]["rewards"].append(r)
                episode_reward[sid] += r

            _append_step_logs(step_logs, station_metrics, algo, episode, step)

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        # Policy gradient update with advantage estimation
        for sid, traj in trajectories.items():
            if not traj["rewards"]:
                continue
            returns = []
            G = 0.0
            for r in reversed(traj["rewards"]):
                G = r + gamma * G
                returns.insert(0, G)
            returns_arr = np.array(returns)

            for t in range(len(traj["obs"])):
                obs_t = traj["obs"][t]
                baseline = policies[sid].get_value(
                    Observation(timestamp=t, local={"obs": obs_t})
                )
                advantage = returns_arr[t] - baseline
                policies[sid].update(obs_t, traj["actions"][t], advantage, lr)
                policies[sid].update_critic(obs_t, returns_arr[t], lr)
            policies[sid].decay_noise()

        total = sum(episode_reward.values())
        returns_history.append(total)
        episode_logs.append({
            "algo": algo,
            "episode": episode,
            "total_reward": total,
            **{f"reward_{sid}": episode_reward[sid] for sid in station_ids},
        })
        if (episode + 1) % 10 == 0:
            logger.info(
                f"Episode {episode+1:3d} | "
                f"Total reward: {total:8.2f} | "
                f"Per-station: {dict((k, round(v, 2)) for k, v in episode_reward.items())}"
            )

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    _write_metrics(out_dir, step_logs, episode_logs)

    logger.info("Training completed.")
    return env, policies, returns_history


def evaluate_policies(
    policies,
    num_episodes: int = 5,
    steps_per_episode: int = 96,
    seed: int = 2026,
    env_config: Dict[str, Any] = None,
    output_dir: str = "outputs/eval_i_ac_mlp",
    algo: str = "I-AC-MLP",
):
    """Evaluate trained policies without exploration noise."""
    env = create_charging_env(env_config)
    station_ids = [
        aid for aid, agent in env.registered_agents.items()
        if isinstance(agent, StationCoordinator)
    ]

    old_noise = {}
    for sid, policy in policies.items():
        old_noise[sid] = policy.noise_scale
        policy.noise_scale = 0.0

    step_logs: List[Dict[str, Any]] = []
    episode_logs: List[Dict[str, Any]] = []

    try:
        for episode in range(num_episodes):
            obs, info = env.reset(seed=seed + episode)
            episode_reward = {sid: 0.0 for sid in station_ids}

            for step in range(steps_per_episode):
                actions = {}
                for sid in station_ids:
                    obs_value = obs[sid]
                    if isinstance(obs_value, Observation):
                        observation = Observation(timestamp=step, local=obs_value.local)
                    elif isinstance(obs_value, np.ndarray):
                        observation = Observation(timestamp=step, local={"obs": obs_value[:8]})
                    else:
                        observation = Observation(timestamp=step, local={"obs": np.zeros(8, dtype=np.float32)})
                    actions[sid] = policies[sid].forward_deterministic(observation)

                obs, rewards, terminated, truncated, infos = env.step(actions)
                station_metrics = infos.get("__all__", {}).get("station_metrics", {})

                for sid in station_ids:
                    episode_reward[sid] += float(rewards.get(sid, 0.0))

                _append_step_logs(step_logs, station_metrics, algo, episode, step)

                if terminated.get("__all__", False) or truncated.get("__all__", False):
                    break

            total = sum(episode_reward.values())
            episode_logs.append({
                "algo": algo,
                "episode": episode,
                "total_reward": total,
                **{f"reward_{sid}": episode_reward[sid] for sid in station_ids},
            })
            logger.info(
                f"[{algo}][Eval] Episode {episode + 1:3d} | "
                f"Total reward: {total:8.2f} | "
                f"Per-station: {dict((k, round(v, 2)) for k, v in episode_reward.items())}"
            )
    finally:
        for sid, policy in policies.items():
            policy.noise_scale = old_noise[sid]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(out_dir, step_logs, episode_logs)
    logger.info(f"[{algo}][Eval] Metrics saved to: {out_dir}")
    return episode_logs, step_logs


def run_fixed_pricing(
    num_episodes: int = 5,
    steps_per_episode: int = 20,
    seed: int = 42,
    env_config: Dict[str, Any] = None,
    output_dir: str = "outputs/fp",
):
    """Run the fixed-pricing baseline and export step/episode metrics."""
    env_config = env_config or {}
    env = create_charging_env(env_config)

    station_ids = [
        aid for aid, agent in env.registered_agents.items()
        if isinstance(agent, StationCoordinator)
    ]

    step_logs: List[Dict[str, Any]] = []
    episode_logs: List[Dict[str, Any]] = []
    dt = float(env_config.get("dt", 900.0))

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = {sid: 0.0 for sid in station_ids}

        for step in range(steps_per_episode):
            time_s = step * dt
            fixed_price = fixed_price_by_time(time_s)

            actions = {
                sid: _fixed_price_action(fixed_price)
                for sid in station_ids
            }

            obs, rewards, terminated, truncated, infos = env.step(actions)
            station_metrics = infos.get("__all__", {}).get("station_metrics", {})

            for sid in station_ids:
                episode_reward[sid] += float(rewards.get(sid, 0.0))

            _append_step_logs(step_logs, station_metrics, "FP", episode, step)

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        total = sum(episode_reward.values())
        episode_logs.append({
            "algo": "FP",
            "episode": episode,
            "total_reward": total,
            **{f"reward_{sid}": episode_reward[sid] for sid in station_ids},
        })

        logger.info(
            f"[FP] Episode {episode + 1:3d} | "
            f"Total reward: {total:8.2f} | "
            f"Per-station: {dict((k, round(v, 2)) for k, v in episode_reward.items())}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_metrics(out_dir, step_logs, episode_logs)

    logger.info(f"[FP] Metrics saved to: {out_dir}")
    return episode_logs, step_logs


def run_constant_pricing(
    price: float,
    num_episodes: int = 1,
    steps_per_episode: int = 96,
    seed: int = 2026,
    env_config: Dict[str, Any] = None,
    output_dir: str = "outputs/constant_price",
    algo: str = "CONST",
):
    """Run a constant service-fee policy for smoke diagnostics."""
    env_config = env_config or {}
    env = create_charging_env(env_config)

    station_ids = [
        aid for aid, agent in env.registered_agents.items()
        if isinstance(agent, StationCoordinator)
    ]

    step_logs: List[Dict[str, Any]] = []
    episode_logs: List[Dict[str, Any]] = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = {sid: 0.0 for sid in station_ids}

        actions = {
            sid: _fixed_price_action(price)
            for sid in station_ids
        }

        for step in range(steps_per_episode):
            obs, rewards, terminated, truncated, infos = env.step(actions)
            station_metrics = infos.get("__all__", {}).get("station_metrics", {})

            for sid in station_ids:
                episode_reward[sid] += float(rewards.get(sid, 0.0))

            _append_step_logs(step_logs, station_metrics, algo, episode, step)

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        total = sum(episode_reward.values())
        episode_logs.append({
            "algo": algo,
            "episode": episode,
            "total_reward": total,
            **{f"reward_{sid}": episode_reward[sid] for sid in station_ids},
        })
        logger.info(
            f"[{algo}] Episode {episode + 1:3d} | "
            f"Total reward: {total:8.2f} | "
            f"Per-station: {dict((k, round(v, 2)) for k, v in episode_reward.items())}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(out_dir, step_logs, episode_logs)
    logger.info(f"[{algo}] Metrics saved to: {out_dir}")
    return episode_logs, step_logs


def _fixed_price_action(price: float) -> Action:
    """Build an HERON-compatible continuous pricing action."""
    action = Action()
    action.set_specs(
        dim_c=1,
        dim_d=0,
        range=(np.array([0.0], dtype=np.float32), np.array([0.8], dtype=np.float32)),
    )
    action.set_values(c=np.array([price], dtype=np.float32))
    return action


def make_price_action(price: float) -> Action:
    """Public helper for constructing continuous pricing actions."""
    return _fixed_price_action(price)


def _append_step_logs(
    step_logs: List[Dict[str, Any]],
    station_metrics: Dict[str, Dict[str, Any]],
    algo: str,
    episode: int,
    step: int,
) -> None:
    for sid, metrics in station_metrics.items():
        step_logs.append({
            "algo": algo,
            "episode": episode,
            "step": step,
            "station": sid,
            "price": metrics["price"],
            "service_fee": metrics.get("service_fee", metrics["price"]),
            "retail_price": metrics.get("retail_price", metrics["price"]),
            "lmp": metrics["lmp"],
            "served_kwh": metrics["served_kwh"],
            "revenue": metrics["revenue"],
            "grid_cost": metrics["grid_cost"],
            "profit": metrics["profit"],
            "queue_len": metrics["queue_len"],
            "utilization": metrics["utilization"],
            "reward": metrics["reward"],
            "arrivals": metrics["arrivals"],
            "abandoned": metrics["abandoned"],
            "abandoned_soc": metrics.get("abandoned_soc", 0),
            "abandoned_cost": metrics.get("abandoned_cost", 0),
            "abandoned_full": metrics.get("abandoned_full", 0),
            "abandoned_timeout": metrics.get("abandoned_timeout", 0),
            "congestion_penalty": metrics["congestion_penalty"],
            "occupied_slots": metrics.get("occupied_slots"),
            "open_slots": metrics.get("open_slots"),
        })


def _extract_obs_vector(obs_value, step: int, obs_dim: int = 8) -> Observation:
    if isinstance(obs_value, Observation):
        return Observation(timestamp=step, local=obs_value.local)
    if isinstance(obs_value, np.ndarray):
        return Observation(timestamp=step, local={"obs": obs_value[:obs_dim]})
    return Observation(timestamp=step, local={"obs": np.zeros(obs_dim, dtype=np.float32)})


def build_neighbor_index(station_ids, station_positions, k_neighbors: int = 4):
    neighbor_index = {}
    for sid in station_ids:
        x, y = station_positions[sid]
        others = []
        for sj in station_ids:
            xj, yj = station_positions[sj]
            dist = ((x - xj) ** 2 + (y - yj) ** 2) ** 0.5
            others.append((dist, sj))
        others.sort(key=lambda item: item[0])
        neighbor_index[sid] = [sj for _, sj in others[: min(k_neighbors + 1, len(others))]]
    return neighbor_index


def build_neighbor_station_index_arrays(station_ids, neighbor_index):
    station_to_index = {sid: idx for idx, sid in enumerate(station_ids)}
    arrays = {
        sid: np.asarray([station_to_index[nid] for nid in neighbor_index[sid]], dtype=np.int64)
        for sid in station_ids
    }
    center_indices = {
        sid: int(station_to_index[sid])
        for sid in station_ids
    }
    return arrays, center_indices


def build_ma_station_obs_matrix(
    station_ids: List[str],
    raw_obs_vecs: Dict[str, np.ndarray],
    station_metrics: Dict[str, Dict[str, Any]],
    env_config: Dict[str, Any] | None = None,
) -> Dict[str, np.ndarray]:
    """Build MA-specific station observations with explicit congestion features."""
    env_config = env_config or {}
    max_queue_size = float(env_config.get("max_queue_size", 12))
    q_threshold = float(env_config.get("q_threshold", 4.0))
    eta = float(env_config.get("eta", 1.0))
    num_chargers = float(env_config.get("num_chargers", 10))
    p_max_kw = float(env_config.get("p_max_kw", 75.0))
    dt = float(env_config.get("dt", 900.0))
    arrival_rate = float(env_config.get("arrival_rate", 65.0))

    served_kwh_cap = max(num_chargers * p_max_kw * dt / 3600.0, 1.0)
    arrivals_cap = max(arrival_rate * dt / 3600.0, 1.0)
    congestion_cap = max(eta * max(max_queue_size - q_threshold, 1.0), 1.0)

    ma_obs = {}
    for sid in station_ids:
        raw = raw_obs_vecs[sid]
        metrics = station_metrics.get(sid, {})

        service_fee = float(metrics.get("service_fee", float(raw[1]) * 0.8 if raw.size > 1 else 0.25))
        lmp = float(metrics.get("lmp", float(raw[2]) if raw.size > 2 else 0.2))
        t_sin = float(raw[3]) if raw.size > 3 else 0.0
        t_cos = float(raw[4]) if raw.size > 4 else 1.0
        queue_len_norm = float(metrics.get("queue_len", 0.0)) / max(max_queue_size, 1.0)
        utilization = float(metrics.get("utilization", 0.0))
        arrivals_norm = float(metrics.get("arrivals", 0.0)) / arrivals_cap
        served_kwh_norm = float(metrics.get("served_kwh", 0.0)) / served_kwh_cap
        abandoned_norm = float(metrics.get("abandoned", 0.0)) / arrivals_cap
        congestion_penalty_norm = float(metrics.get("congestion_penalty", 0.0)) / congestion_cap

        ma_obs[sid] = np.asarray(
            [
                service_fee,
                lmp,
                t_sin,
                t_cos,
                queue_len_norm,
                utilization,
                arrivals_norm,
                served_kwh_norm,
                abandoned_norm,
                congestion_penalty_norm,
            ],
            dtype=np.float32,
        )
    return ma_obs


def train_independent_transa3c(
    num_episodes: int = 50,
    steps_per_episode: int = 96,
    seed: int = 42,
    gamma: float = 0.99,
    seq_len: int = 8,
    env_config: Dict[str, Any] = None,
    output_dir: str = "outputs/i_transa3c_train",
    algo: str = "I-TransA3C",
):
    """Train independent temporal Transformer actor-critic policies per station."""
    if IndependentTransA3CPolicy is None:
        raise ImportError("IndependentTransA3CPolicy requires PyTorch. Install torch in the active environment.")

    np.random.seed(seed)
    env = create_charging_env(env_config)

    station_ids = [
        aid for aid, agent in env.registered_agents.items()
        if isinstance(agent, StationCoordinator)
    ]

    policies = {
        sid: IndependentTransA3CPolicy(
            obs_dim=8,
            seq_len=seq_len,
            d_model=128,
            nhead=8,
            num_layers=2,
            actor_lr=1e-4,
            critic_lr=5e-4,
            gamma=gamma,
            entropy_coef=0.01,
            seed=seed + i,
        )
        for i, sid in enumerate(station_ids)
    }

    step_logs: List[Dict[str, Any]] = []
    episode_logs: List[Dict[str, Any]] = []
    returns_history: List[float] = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)

        histories = {
            sid: deque([np.zeros(8, dtype=np.float32) for _ in range(seq_len)], maxlen=seq_len)
            for sid in station_ids
        }
        trajectories = {
            sid: {"seqs": [], "actions": [], "rewards": []}
            for sid in station_ids
        }
        episode_reward = {sid: 0.0 for sid in station_ids}

        for step in range(steps_per_episode):
            actions = {}
            for sid in station_ids:
                obs_value = obs[sid]
                if isinstance(obs_value, Observation):
                    observation = Observation(timestamp=step, local=obs_value.local)
                elif isinstance(obs_value, np.ndarray):
                    observation = Observation(timestamp=step, local={"obs": obs_value[:8]})
                else:
                    observation = Observation(timestamp=step, local={"obs": np.zeros(8, dtype=np.float32)})

                obs_vec = policies[sid].extract_obs_vector(observation, 8)
                histories[sid].append(obs_vec)

                seq = np.stack(list(histories[sid]), axis=0).astype(np.float32)
                action_arr = policies[sid].act(seq, deterministic=False)
                price = float(action_arr[0])

                actions[sid] = make_price_action(price)
                trajectories[sid]["seqs"].append(seq.copy())
                trajectories[sid]["actions"].append(price)

            obs, rewards, terminated, truncated, infos = env.step(actions)
            station_metrics = infos.get("__all__", {}).get("station_metrics", {})

            for sid in station_ids:
                reward_value = float(rewards.get(sid, 0.0))
                trajectories[sid]["rewards"].append(reward_value)
                episode_reward[sid] += reward_value

            _append_step_logs(step_logs, station_metrics, algo, episode, step)

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        for sid, traj in trajectories.items():
            if not traj["rewards"]:
                continue

            returns = []
            G = 0.0
            for reward_value in reversed(traj["rewards"]):
                G = reward_value + gamma * G
                returns.insert(0, G)

            seqs = np.asarray(traj["seqs"], dtype=np.float32)
            acts = np.asarray(traj["actions"], dtype=np.float32)
            rets = np.asarray(returns, dtype=np.float32)
            policies[sid].update(seqs, acts, rets)

        total = sum(episode_reward.values())
        returns_history.append(total)
        episode_logs.append({
            "algo": algo,
            "episode": episode,
            "total_reward": total,
            **{f"reward_{sid}": episode_reward[sid] for sid in station_ids},
        })

        logger.info(
            f"[{algo}] Episode {episode + 1:3d} | "
            f"Total reward: {total:8.2f} | "
            f"Per-station: {dict((k, round(v, 2)) for k, v in episode_reward.items())}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(out_dir, step_logs, episode_logs)

    logger.info(f"[{algo}] Training metrics saved to: {out_dir}")
    return env, policies, returns_history


def evaluate_independent_transa3c(
    policies: Dict[str, "IndependentTransA3CPolicy"],
    num_episodes: int = 1,
    steps_per_episode: int = 96,
    seed: int = 2026,
    seq_len: int = 8,
    env_config: Dict[str, Any] = None,
    output_dir: str = "outputs/i_transa3c_day",
    algo: str = "I-TransA3C",
):
    """Evaluate trained I-TransA3C policies deterministically."""
    env = create_charging_env(env_config)
    station_ids = [
        aid for aid, agent in env.registered_agents.items()
        if isinstance(agent, StationCoordinator)
    ]

    step_logs: List[Dict[str, Any]] = []
    episode_logs: List[Dict[str, Any]] = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        histories = {
            sid: deque([np.zeros(8, dtype=np.float32) for _ in range(seq_len)], maxlen=seq_len)
            for sid in station_ids
        }
        episode_reward = {sid: 0.0 for sid in station_ids}

        for step in range(steps_per_episode):
            actions = {}
            for sid in station_ids:
                obs_value = obs[sid]
                if isinstance(obs_value, Observation):
                    observation = Observation(timestamp=step, local=obs_value.local)
                elif isinstance(obs_value, np.ndarray):
                    observation = Observation(timestamp=step, local={"obs": obs_value[:8]})
                else:
                    observation = Observation(timestamp=step, local={"obs": np.zeros(8, dtype=np.float32)})

                obs_vec = policies[sid].extract_obs_vector(observation, 8)
                histories[sid].append(obs_vec)

                seq = np.stack(list(histories[sid]), axis=0).astype(np.float32)
                action_arr = policies[sid].act(seq, deterministic=True)
                actions[sid] = make_price_action(float(action_arr[0]))

            obs, rewards, terminated, truncated, infos = env.step(actions)
            station_metrics = infos.get("__all__", {}).get("station_metrics", {})

            for sid in station_ids:
                episode_reward[sid] += float(rewards.get(sid, 0.0))

            _append_step_logs(step_logs, station_metrics, algo, episode, step)

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        total = sum(episode_reward.values())
        episode_logs.append({
            "algo": algo,
            "episode": episode,
            "total_reward": total,
            **{f"reward_{sid}": episode_reward[sid] for sid in station_ids},
        })

        logger.info(
            f"[EVAL {algo}] Episode {episode + 1:3d} | "
            f"Total reward: {total:8.2f}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(out_dir, step_logs, episode_logs)

    logger.info(f"[EVAL {algo}] Metrics saved to: {out_dir}")
    return episode_logs, step_logs


def train_ma_transa3c(
    num_episodes: int = 10,
    steps_per_episode: int = 96,
    seed: int = 42,
    gamma: float = 0.99,
    k_neighbors: int = 4,
    use_ma_station_obs: bool = False,
    use_lagged_rank_loss: bool = False,
    lambda_rank: float = 0.0,
    rank_margin: float = 0.02,
    rank_eps: float = 0.10,
    lambda_response: float = 0.0,
    response_margin: float = 0.01,
    response_threshold: float = 0.02,
    response_ema_alpha: float = 0.9,
    lambda_anchor: float = 0.0,
    price_anchor: float = 0.40,
    env_config: Dict[str, Any] = None,
    output_dir: str = "outputs/MA_TransA3C_soc_train",
    algo: str = "MA-TransA3C",
):
    """Train shared spatial actor with centralized global critic."""
    if MATransA3CPolicy is None:
        raise ImportError("MATransA3CPolicy requires PyTorch. Install torch in the active environment.")

    np.random.seed(seed)
    env = create_charging_env(env_config)
    station_ids = [
        aid for aid, agent in env.registered_agents.items()
        if isinstance(agent, StationCoordinator)
    ]
    neighbor_index = build_neighbor_index(station_ids, env.station_positions, k_neighbors=k_neighbors)
    neighbor_station_arrays, center_station_indices = build_neighbor_station_index_arrays(station_ids, neighbor_index)

    policy = MATransA3CPolicy(
        obs_dim=10 if use_ma_station_obs else 8,
        seq_len=min(k_neighbors + 1, len(station_ids)),
        num_stations=len(station_ids),
        station_ids=station_ids,
        d_model=128,
        nhead=8,
        num_layers=2,
        actor_lr=5e-5,
        critic_lr=1e-4,
        gamma=gamma,
        entropy_coef=0.01,
        lambda_rank=lambda_rank,
        rank_margin=rank_margin,
        rank_eps=rank_eps,
        lambda_response=lambda_response,
        response_margin=response_margin,
        response_threshold=response_threshold,
        response_ema_alpha=response_ema_alpha,
        lambda_anchor=lambda_anchor,
        price_anchor=price_anchor,
        seed=seed,
    )

    step_logs: List[Dict[str, Any]] = []
    episode_logs: List[Dict[str, Any]] = []
    returns_history: List[float] = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = {sid: 0.0 for sid in station_ids}
        prev_station_metrics: Dict[str, Dict[str, Any]] = {}
        update_stats: Dict[str, float] = {}
        traj_neighbor_seqs = []
        traj_neighbor_station_indices = []
        traj_center_station_indices = []
        traj_global_obs = []
        traj_rank_global_obs = []
        traj_rank_valid_mask = []
        traj_actions = []
        traj_global_rewards = []
        prev_rank_global_obs = None

        for step in range(steps_per_episode):
            raw_obs_vecs = {}
            for sid in station_ids:
                observation = _extract_obs_vector(obs[sid], step, obs_dim=8)
                raw_obs_vecs[sid] = policy.extract_obs_vector(observation, 8)

            obs_vecs = (
                build_ma_station_obs_matrix(station_ids, raw_obs_vecs, prev_station_metrics, env_config)
                if use_ma_station_obs
                else raw_obs_vecs
            )

            global_obs = np.stack([obs_vecs[sid] for sid in station_ids], axis=0).astype(np.float32)
            if use_lagged_rank_loss and prev_rank_global_obs is not None:
                rank_global_obs = prev_rank_global_obs.copy()
                rank_valid = True
            else:
                rank_global_obs = global_obs.copy()
                rank_valid = not use_lagged_rank_loss
            neighbor_seqs = []
            neighbor_station_indices = []
            center_indices = []
            action_values = []
            actions = {}

            for sid in station_ids:
                neighbor_seq = np.stack([obs_vecs[nid] for nid in neighbor_index[sid]], axis=0).astype(np.float32)
                neighbor_station_idx = neighbor_station_arrays[sid]
                center_idx = center_station_indices[sid]
                action_arr = policy.act(
                    neighbor_seq,
                    neighbor_station_indices=neighbor_station_idx,
                    center_station_index=center_idx,
                    deterministic=False,
                )
                price = float(action_arr[0])
                actions[sid] = make_price_action(price)
                neighbor_seqs.append(neighbor_seq)
                neighbor_station_indices.append(neighbor_station_idx.copy())
                center_indices.append(center_idx)
                action_values.append(price)

            obs, rewards, terminated, truncated, infos = env.step(actions)
            station_metrics = infos.get("__all__", {}).get("station_metrics", {})
            prev_station_metrics = station_metrics
            global_reward = 0.0

            for sid in station_ids:
                reward_value = float(rewards.get(sid, 0.0))
                episode_reward[sid] += reward_value
                global_reward += reward_value

            traj_neighbor_seqs.append(np.stack(neighbor_seqs, axis=0))
            traj_neighbor_station_indices.append(np.stack(neighbor_station_indices, axis=0))
            traj_center_station_indices.append(np.asarray(center_indices, dtype=np.int64))
            traj_global_obs.append(global_obs)
            traj_rank_global_obs.append(rank_global_obs)
            traj_rank_valid_mask.append(rank_valid)
            traj_actions.append(np.asarray(action_values, dtype=np.float32))
            traj_global_rewards.append(global_reward)
            prev_rank_global_obs = global_obs

            _append_step_logs(step_logs, station_metrics, algo, episode, step)

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        returns = []
        G = 0.0
        for reward_value in reversed(traj_global_rewards):
            G = reward_value + gamma * G
            returns.insert(0, G)

        if returns:
            update_stats = policy.update(
                neighbor_seqs=np.asarray(traj_neighbor_seqs, dtype=np.float32),
                neighbor_station_indices=np.asarray(traj_neighbor_station_indices, dtype=np.int64),
                center_station_indices=np.asarray(traj_center_station_indices, dtype=np.int64),
                global_obs=np.asarray(traj_global_obs, dtype=np.float32),
                actions=np.asarray(traj_actions, dtype=np.float32),
                returns=np.asarray(returns, dtype=np.float32),
                rank_global_obs=np.asarray(traj_rank_global_obs, dtype=np.float32),
                rank_valid_mask=np.asarray(traj_rank_valid_mask, dtype=bool),
            )

        total = sum(episode_reward.values())
        returns_history.append(total)
        episode_logs.append({
            "algo": algo,
            "episode": episode,
            "total_reward": total,
            "actor_loss": update_stats.get("actor_loss"),
            "actor_loss_pg": update_stats.get("actor_loss_pg"),
            "critic_loss": update_stats.get("critic_loss"),
            "entropy": update_stats.get("entropy"),
            "rank_loss": update_stats.get("rank_loss"),
            "rank_loss_weighted": update_stats.get("rank_loss_weighted"),
            "rank_pair_count": update_stats.get("rank_pair_count"),
            "rank_violation_count": update_stats.get("rank_violation_count"),
            "rank_violation_rate": update_stats.get("rank_violation_rate"),
            "response_loss": update_stats.get("response_loss"),
            "response_loss_weighted": update_stats.get("response_loss_weighted"),
            "response_trigger_count": update_stats.get("response_trigger_count"),
            "response_violation_count": update_stats.get("response_violation_count"),
            "response_violation_rate": update_stats.get("response_violation_rate"),
            "anchor_loss": update_stats.get("anchor_loss"),
            "anchor_loss_weighted": update_stats.get("anchor_loss_weighted"),
            "mu_std_across_stations": update_stats.get("mu_std_across_stations"),
            "score_std_across_stations": update_stats.get("score_std_across_stations"),
            "mean_price_mean": update_stats.get("mean_price_mean"),
            **{f"reward_{sid}": episode_reward[sid] for sid in station_ids},
        })
        logger.info(
            f"[{algo}] Episode {episode + 1:3d} | "
            f"Total reward: {total:8.2f} | "
            f"Rank loss: {update_stats.get('rank_loss', 0.0):.4f} | "
            f"Response loss: {update_stats.get('response_loss', 0.0):.4f} | "
            f"Per-station: {dict((k, round(v, 2)) for k, v in episode_reward.items())}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(out_dir, step_logs, episode_logs)
    logger.info(f"[{algo}] Training metrics saved to: {out_dir}")
    return env, policy, returns_history


def evaluate_ma_transa3c(
    policy: "MATransA3CPolicy",
    num_episodes: int = 1,
    steps_per_episode: int = 96,
    seed: int = 2026,
    k_neighbors: int = 4,
    use_ma_station_obs: bool = False,
    env_config: Dict[str, Any] = None,
    output_dir: str = "outputs/MA_TransA3C_soc_calibrated",
    algo: str = "MA-TransA3C",
):
    """Evaluate MA-TransA3C deterministically."""
    env = create_charging_env(env_config)
    station_ids = [
        aid for aid, agent in env.registered_agents.items()
        if isinstance(agent, StationCoordinator)
    ]
    neighbor_index = build_neighbor_index(station_ids, env.station_positions, k_neighbors=k_neighbors)
    neighbor_station_arrays, center_station_indices = build_neighbor_station_index_arrays(station_ids, neighbor_index)

    step_logs: List[Dict[str, Any]] = []
    episode_logs: List[Dict[str, Any]] = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = {sid: 0.0 for sid in station_ids}
        prev_station_metrics: Dict[str, Dict[str, Any]] = {}

        for step in range(steps_per_episode):
            raw_obs_vecs = {}
            for sid in station_ids:
                observation = _extract_obs_vector(obs[sid], step, obs_dim=8)
                raw_obs_vecs[sid] = policy.extract_obs_vector(observation, 8)

            obs_vecs = (
                build_ma_station_obs_matrix(station_ids, raw_obs_vecs, prev_station_metrics, env_config)
                if use_ma_station_obs
                else raw_obs_vecs
            )

            actions = {}
            for sid in station_ids:
                neighbor_seq = np.stack([obs_vecs[nid] for nid in neighbor_index[sid]], axis=0).astype(np.float32)
                action_arr = policy.act(
                    neighbor_seq,
                    neighbor_station_indices=neighbor_station_arrays[sid],
                    center_station_index=center_station_indices[sid],
                    deterministic=True,
                )
                actions[sid] = make_price_action(float(action_arr[0]))

            obs, rewards, terminated, truncated, infos = env.step(actions)
            station_metrics = infos.get("__all__", {}).get("station_metrics", {})
            prev_station_metrics = station_metrics

            for sid in station_ids:
                episode_reward[sid] += float(rewards.get(sid, 0.0))

            _append_step_logs(step_logs, station_metrics, algo, episode, step)

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        total = sum(episode_reward.values())
        episode_logs.append({
            "algo": algo,
            "episode": episode,
            "total_reward": total,
            **{f"reward_{sid}": episode_reward[sid] for sid in station_ids},
        })
        logger.info(
            f"[EVAL {algo}] Episode {episode + 1:3d} | "
            f"Total reward: {total:8.2f}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(out_dir, step_logs, episode_logs)
    logger.info(f"[EVAL {algo}] Metrics saved to: {out_dir}")
    return episode_logs, step_logs


def train_mappo_mlp(
    num_episodes: int = 30,
    steps_per_episode: int = 96,
    seed: int = 42,
    gamma: float = 0.99,
    use_ma_station_obs: bool = True,
    env_config: Dict[str, Any] = None,
    output_dir: str = "outputs/MAPPO_MLP_train",
    algo: str = "MAPPO-MLP",
):
    """Train MAPPO-MLP baseline with a centralized critic."""
    if MAPPOPolicy is None:
        raise ImportError("MAPPOPolicy requires PyTorch. Install torch in the active environment.")

    np.random.seed(seed)
    env = create_charging_env(env_config)
    station_ids = [
        aid for aid, agent in env.registered_agents.items()
        if isinstance(agent, StationCoordinator)
    ]
    station_to_index = {sid: i for i, sid in enumerate(station_ids)}

    policy = MAPPOPolicy(
        obs_dim=10 if use_ma_station_obs else 8,
        num_stations=len(station_ids),
        hidden_dim=128,
        actor_lr=1e-4,
        critic_lr=5e-4,
        gamma=gamma,
        entropy_coef=0.01,
        clip_eps=0.2,
        ppo_epochs=4,
        seed=seed,
    )

    step_logs: List[Dict[str, Any]] = []
    episode_logs: List[Dict[str, Any]] = []
    returns_history: List[float] = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = {sid: 0.0 for sid in station_ids}
        prev_station_metrics: Dict[str, Dict[str, Any]] = {}

        traj_obs = []
        traj_actions = []
        traj_old_log_probs = []
        traj_global_rewards = []
        update_stats: Dict[str, float] = {}

        for step in range(steps_per_episode):
            raw_obs_vecs = {}
            for sid in station_ids:
                observation = _extract_obs_vector(obs[sid], step, obs_dim=8)
                raw_obs_vecs[sid] = policy.extract_obs_vector(observation, 8)

            obs_vecs = (
                build_ma_station_obs_matrix(station_ids, raw_obs_vecs, prev_station_metrics, env_config)
                if use_ma_station_obs
                else raw_obs_vecs
            )

            obs_matrix = np.stack([obs_vecs[sid] for sid in station_ids], axis=0).astype(np.float32)
            actions = {}
            action_values = []
            old_log_values = []

            for sid in station_ids:
                action_arr, old_log_prob = policy.act(
                    obs_vecs[sid],
                    station_index=station_to_index[sid],
                    deterministic=False,
                    return_log_prob=True,
                )
                price = float(action_arr[0])
                actions[sid] = make_price_action(price)
                action_values.append(price)
                old_log_values.append(old_log_prob)

            obs, rewards, terminated, truncated, infos = env.step(actions)
            station_metrics = infos.get("__all__", {}).get("station_metrics", {})
            prev_station_metrics = station_metrics
            global_reward = 0.0

            for sid in station_ids:
                reward_value = float(rewards.get(sid, 0.0))
                episode_reward[sid] += reward_value
                global_reward += reward_value

            traj_obs.append(obs_matrix)
            traj_actions.append(np.asarray(action_values, dtype=np.float32))
            traj_old_log_probs.append(np.asarray(old_log_values, dtype=np.float32))
            traj_global_rewards.append(global_reward)

            _append_step_logs(step_logs, station_metrics, algo, episode, step)

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        returns = []
        G = 0.0
        for reward_value in reversed(traj_global_rewards):
            G = reward_value + gamma * G
            returns.insert(0, G)

        if traj_obs:
            update_stats = policy.update(
                obs=np.asarray(traj_obs, dtype=np.float32),
                actions=np.asarray(traj_actions, dtype=np.float32),
                old_log_probs=np.asarray(traj_old_log_probs, dtype=np.float32),
                returns=np.asarray(returns, dtype=np.float32),
            )

        total = sum(episode_reward.values())
        returns_history.append(total)
        episode_logs.append({
            "algo": algo,
            "episode": episode,
            "total_reward": total,
            **{f"reward_{sid}": episode_reward[sid] for sid in station_ids},
            **update_stats,
        })
        logger.info(
            f"[{algo}] Episode {episode + 1:3d} | "
            f"Total reward: {total:8.2f} | "
            f"actor_loss={update_stats.get('actor_loss', 0.0):.4f} | "
            f"critic_loss={update_stats.get('critic_loss', 0.0):.4f}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(out_dir, step_logs, episode_logs)
    logger.info(f"[{algo}] Training metrics saved to: {out_dir}")
    return env, policy, returns_history


def evaluate_mappo_mlp(
    policy: "MAPPOPolicy",
    num_episodes: int = 1,
    steps_per_episode: int = 96,
    seed: int = 2026,
    use_ma_station_obs: bool = True,
    env_config: Dict[str, Any] = None,
    output_dir: str = "outputs/MAPPO_MLP_eval",
    algo: str = "MAPPO-MLP",
):
    """Evaluate MAPPO-MLP deterministically."""
    env = create_charging_env(env_config)
    station_ids = [
        aid for aid, agent in env.registered_agents.items()
        if isinstance(agent, StationCoordinator)
    ]
    station_to_index = {sid: i for i, sid in enumerate(station_ids)}

    step_logs: List[Dict[str, Any]] = []
    episode_logs: List[Dict[str, Any]] = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = {sid: 0.0 for sid in station_ids}
        prev_station_metrics: Dict[str, Dict[str, Any]] = {}

        for step in range(steps_per_episode):
            raw_obs_vecs = {}
            for sid in station_ids:
                observation = _extract_obs_vector(obs[sid], step, obs_dim=8)
                raw_obs_vecs[sid] = policy.extract_obs_vector(observation, 8)

            obs_vecs = (
                build_ma_station_obs_matrix(station_ids, raw_obs_vecs, prev_station_metrics, env_config)
                if use_ma_station_obs
                else raw_obs_vecs
            )

            actions = {}
            for sid in station_ids:
                action_arr = policy.act(
                    obs_vecs[sid],
                    station_index=station_to_index[sid],
                    deterministic=True,
                    return_log_prob=False,
                )
                actions[sid] = make_price_action(float(action_arr[0]))

            obs, rewards, terminated, truncated, infos = env.step(actions)
            station_metrics = infos.get("__all__", {}).get("station_metrics", {})
            prev_station_metrics = station_metrics

            for sid in station_ids:
                episode_reward[sid] += float(rewards.get(sid, 0.0))

            _append_step_logs(step_logs, station_metrics, algo, episode, step)

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        total = sum(episode_reward.values())
        episode_logs.append({
            "algo": algo,
            "episode": episode,
            "total_reward": total,
            **{f"reward_{sid}": episode_reward[sid] for sid in station_ids},
        })
        logger.info(f"[EVAL {algo}] Episode {episode + 1:3d} | Total reward: {total:8.2f}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(out_dir, step_logs, episode_logs)
    logger.info(f"[EVAL {algo}] Metrics saved to: {out_dir}")
    return episode_logs, step_logs


def _write_metrics(out_dir: Path, step_logs: List[Dict[str, Any]], episode_logs: List[Dict[str, Any]]) -> None:
    if step_logs:
        with (out_dir / "step_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=step_logs[0].keys())
            writer.writeheader()
            writer.writerows(step_logs)

    if episode_logs:
        with (out_dir / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=episode_logs[0].keys())
            writer.writeheader()
            writer.writerows(episode_logs)


# ============================================================================
# Ray RLlib Training (optional)
# ============================================================================
def train_rllib(num_iterations: int = 50):
    
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from heron.adaptors.rllib import RLlibBasedHeronEnv
    except ImportError:
        logger.error("Ray RLlib not installed. Run: pip install 'ray[rllib]'")
        return

    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)

    steps_per_episode = 288  # 288 * 300s = 86400s = 1 day

    num_stations = 2
    num_chargers = 5

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=RLlibBasedHeronEnv,
            env_config={
                "agents": [
                    {"agent_id": f"station_{i}_slot_{j}",
                     "agent_cls": ChargingSlot,
                     "p_max_kw": 150.0,
                     "coordinator": f"station_{i}"}
                    for i in range(num_stations)
                    for j in range(num_chargers)
                ],
                "coordinators": [
                    {"coordinator_id": f"station_{i}",
                     "agent_cls": StationCoordinator}
                    for i in range(num_stations)
                ],
                "env_class": ChargingEnv,
                "env_kwargs": {
                    "arrival_rate": 10.0,
                    "dt": 300.0,
                    "episode_length": 86400.0,
                },
                "agent_ids": [f"station_{i}" for i in range(num_stations)],
                "max_steps": steps_per_episode,
            },
        )
        .framework("torch")
        .training(
            lr=1e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_clip_param=10.0,
            train_batch_size=4000,
        )
        .env_runners(
            num_env_runners=2,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
        )
        .resources(num_gpus=0, num_cpus_for_main_process=2)
        .multi_agent(
            policies={"station_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kw: "station_policy",
        )
    )

    logger.info("Building PPO algorithm...")
    algo = config.build()

    try:
        for i in range(num_iterations):
            result = algo.train()
            reward_mean = (
                result.get("env_runners", {}).get("episode_reward_mean")
                or result.get("episode_reward_mean", 0)
            )
            episodes = (
                result.get("num_episodes_done")
                or result.get("episodes_total", 0)
            )
            logger.info(f"Iter {i:3d} | Reward mean: {reward_mean:7.2f} | Episodes: {episodes}")

            if i % 10 == 0 and i > 0:
                checkpoint = algo.save()
                logger.info(f"Checkpoint: {checkpoint.checkpoint.path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted")
    finally:
        algo.stop()
        ray.shutdown()


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--rllib":
        train_rllib(num_iterations=50)
    elif len(sys.argv) > 1 and sys.argv[1] == "--event-driven":
        from case_studies.power.ev_public_charging_case.run_event_driven import main as run_ed
        run_ed()
    else:
        train_simple(
            num_episodes=1,
            steps_per_episode=5,
            env_config={
                "num_stations": 5,
                "num_chargers": 10,
                "arrival_rate": 30.0,
                "dt": 900.0,
                "episode_length": 86400.0,
            },
        )
