

from typing import Dict, List, Optional

import numpy as np
from gymnasium.spaces import Box

from heron.agents.coordinator_agent import CoordinatorAgent
from heron.core.feature import Feature
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.protocols.vertical import BroadcastActionProtocol, VerticalProtocol
from heron.scheduling.schedule_config import ScheduleConfig
from heron.utils.typing import AgentID

from case_studies.power.ev_public_charging_case.features import ChargingStationFeature, MarketFeature, RegulationFeature
from .charging_slot import ChargingSlot


class StationCoordinator(CoordinatorAgent):
   

    def __init__(
        self,
        agent_id: AgentID,
        subordinates: Dict[AgentID, ChargingSlot],
        features: Optional[List[Feature]] = None,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        if not subordinates:
            raise ValueError(
                "StationCoordinator requires subordinates (ChargingSlot agents). "
                "Create slots externally and pass as subordinates dict."
            )

        default_features = [
            ChargingStationFeature(max_chargers=len(subordinates), open_chargers=len(subordinates)),
            MarketFeature(),
            RegulationFeature()
        ]
        all_features = (features or []) + default_features

        super().__init__(
            agent_id=agent_id,
            features=all_features,
            subordinates=subordinates,
            upstream_id=upstream_id,
            env_id=env_id,
            schedule_config=schedule_config,
            policy=policy,
            protocol=protocol or VerticalProtocol(
                action_protocol=BroadcastActionProtocol(),
            ),
        )

        # Observation: ChargingStationFeature(2) + MarketFeature(3) + RegulationFeature(3) = 8
        self.observation_space = Box(-np.inf, np.inf, (8,), np.float32)
        # Action: pricing in [0, 0.8] $/kWh
        self.action_space = Box(0.0, 0.8, (1,), np.float32)

    @property
    def charging_slots(self) -> Dict[AgentID, ChargingSlot]:
        """Alias for subordinates."""
        return self.subordinates

    def compute_rewards(self, proxy) -> Dict[AgentID, float]:
       
        # First compute all subordinate rewards
        sub_rewards: Dict[AgentID, float] = {}
        for subordinate in self.subordinates.values():
            sub_rewards.update(subordinate.compute_rewards(proxy))

        # Coordinator reward = sum of subordinate rewards
        coordinator_reward = sum(sub_rewards.values())

        rewards = {self.agent_id: coordinator_reward}
        rewards.update(sub_rewards)
        return rewards

    def compute_local_reward(self, local_state: dict) -> float:
       
        # Use subordinate rewards if available (event-driven mode post-simulation)
        subordinate_rewards = local_state.get("subordinate_rewards", {})
        if subordinate_rewards:
            return sum(subordinate_rewards.values())

        # Fallback: compute from own features (synchronous mode or no subordinate rewards yet)
        csf = local_state.get("ChargingStationFeature")
        if csf is None:
            return 0.0
        # ChargingStationFeature.vector(): [open_norm, price_norm]
        open_norm = float(csf[0])
        price_norm = float(csf[1])
        occupied_fraction = 1.0 - open_norm

        # Denormalize price: price_norm is in [0, 1], actual price in [0, 0.8]
        price = price_norm * 0.8

        mf = local_state.get("MarketFeature")
        if mf is not None:
            # MarketFeature.vector(): [lmp, sin(theta), cos(theta)]
            lmp = float(mf[0])
        else:
            lmp = 0.2

        margin = max(0.0, price - lmp)
        return occupied_fraction * margin
