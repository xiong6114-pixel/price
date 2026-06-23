"""Minimal agent abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from heron.core.action import Action


@dataclass
class AgentState:
    """Feature state bundle owned by an agent."""

    features: Dict[str, object]

    def update_feature(self, name: str, **kwargs):
        feature = self.features[name]
        feature.set_values(**kwargs)


class Agent:
    """Base agent with feature state and reward helpers."""

    def __init__(
        self,
        agent_id,
        features: Optional[Iterable[object]] = None,
        upstream_id=None,
        env_id=None,
        schedule_config=None,
        policy=None,
        protocol=None,
    ):
        self.agent_id = str(agent_id)
        self.upstream_id = upstream_id
        self.env_id = env_id
        self.schedule_config = schedule_config
        self.policy = policy
        self.protocol = protocol
        feature_list = list(features or [])
        self.state = AgentState({feature.__class__.__name__: feature for feature in feature_list})
        self.action = self.init_action(feature_list)

    def init_action(self, features: List[object] | None = None) -> Action:
        return Action()

    def compute_local_reward(self, local_state: dict) -> float:
        return 0.0

    def compute_rewards(self, proxy) -> Dict[str, float]:
        return {self.agent_id: float(self.compute_local_reward(proxy.get_local_state(self.agent_id)))}

    @classmethod
    def handler(cls, _event_name):
        def decorator(func):
            return func
        return decorator
