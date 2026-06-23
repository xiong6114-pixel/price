"""Minimal multi-agent environment runner for the local smoke test."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.observation import Observation


class _Proxy:
    """Read-only view used by reward helpers."""

    def __init__(self, env: "HeronEnv"):
        self._env = env

    def get_local_state(self, agent_id: str) -> Dict[str, Any]:
        agent = self._env.registered_agents[agent_id]
        state = self._env.global_state["agent_states"].get(agent_id, {}).get("features", {})
        local_state = {}
        for name, values in state.items():
            feature = deepcopy(agent.state.features[name])
            feature.set_values(**values)
            local_state[name] = feature.vector()
        return local_state


class HeronEnv:
    """Small subset of HERON env orchestration used by this project."""

    def __init__(self, coordinator_agents, env_id: str = "heron_env", **kwargs):
        self.env_id = env_id
        self.coordinator_agents = list(coordinator_agents)
        self.registered_agents: Dict[str, object] = {}
        for coordinator in self.coordinator_agents:
            self.registered_agents[coordinator.agent_id] = coordinator
            for slot_id, slot in coordinator.subordinates.items():
                self.registered_agents[str(slot_id)] = slot
        self.global_state = self._build_global_state_from_agents()

    def _build_global_state_from_agents(self) -> Dict[str, Any]:
        agent_states = {}
        for agent_id, agent in self.registered_agents.items():
            feature_state = {
                name: feature.to_dict()
                for name, feature in agent.state.features.items()
            }
            agent_states[agent_id] = {"features": feature_state}
        return {"agent_states": agent_states}

    def _sync_agents_from_global_state(self) -> None:
        for agent_id, state_dict in self.global_state.get("agent_states", {}).items():
            agent = self.registered_agents.get(agent_id)
            if agent is None:
                continue
            for name, values in state_dict.get("features", {}).items():
                if name in agent.state.features:
                    agent.state.features[name].set_values(**values)

    def _build_observations(self) -> Dict[str, Observation]:
        observations = {}
        for agent_id, agent in self.registered_agents.items():
            if not isinstance(agent, CoordinatorAgent):
                continue
            local = {}
            for name, feature in agent.state.features.items():
                local[name] = feature.vector()
            observations[agent_id] = Observation(timestamp=0, local=local)
        return observations

    def reset(self, *, seed=None, **kwargs):
        self.global_state = self._build_global_state_from_agents()
        self._sync_agents_from_global_state()
        obs = self._build_observations()
        infos = {agent_id: {} for agent_id in obs}
        return obs, infos

    def step(self, actions):
        for agent_id, action in actions.items():
            if agent_id not in self.registered_agents:
                continue
            agent = self.registered_agents[agent_id]
            if hasattr(agent, "set_action"):
                agent.set_action(action)
            if "ChargingStationFeature" in agent.state.features:
                price = float(getattr(action, "c", [0.25])[0])
                agent.state.update_feature("ChargingStationFeature", charging_price=price)

        self.global_state = self._build_global_state_from_agents()
        env_state = self.global_state_to_env_state(self.global_state)
        env_state = self.run_simulation(env_state)
        self.global_state = self.env_state_to_global_state(env_state)
        self._sync_agents_from_global_state()

        proxy = _Proxy(self)
        rewards = {}
        for coordinator in self.coordinator_agents:
            rewards.update(coordinator.compute_rewards(proxy))

        observations = self._build_observations()
        terminated = {agent_id: False for agent_id in observations}
        truncated = {agent_id: False for agent_id in observations}
        infos = {agent_id: {} for agent_id in observations}
        return observations, rewards, terminated, truncated, infos
