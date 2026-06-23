"""Charging slot agent — one charger port with an optional EV.

Combines the physical charger and its EV occupancy into a single FieldAgent.
When a slot is occupied (EVSlotFeature.occupied == 1), the EV charges.
When empty, the slot is idle.
"""

from typing import Any, List, Optional

import numpy as np

from heron.agents.base import Agent
from heron.agents.field_agent import FieldAgent
from heron.core.action import Action
from heron.core.feature import Feature
from heron.core.policies import Policy
from heron.protocols.base import Protocol
from heron.scheduling.scheduler import Event, EventScheduler
from heron.scheduling.schedule_config import ScheduleConfig
from heron.utils.typing import AgentID

from case_studies.power.ev_public_charging_case.features import ChargerFeature, EVSlotFeature


class ChargingSlot(FieldAgent):
    """A single charging port that may or may not have an EV plugged in.

    Features:
        ChargerFeature — physical charger state (power, max power, open/closed)
        EVSlotFeature  — EV occupancy (occupied flag, SOC, price sensitivity)

    Action:
        1D continuous [0, 0.8] — station price broadcast from coordinator ($/kWh).
        The slot uses the received price + its own EV state to determine charging power.
    """

    def __init__(
        self,
        agent_id: AgentID,
        p_max_kw: float = 150.0,
        upstream_id: Optional[AgentID] = None,
        env_id: Optional[str] = None,
        schedule_config: Optional[ScheduleConfig] = None,
        policy: Optional[Policy] = None,
        protocol: Optional[Protocol] = None,
    ):
        self._p_max_kw = p_max_kw

        features: List[Feature] = [
            ChargerFeature(p_max_kw=p_max_kw),
            EVSlotFeature(),
        ]

        super().__init__(
            agent_id=agent_id,
            features=features,
            upstream_id=upstream_id,
            env_id=env_id,
            schedule_config=schedule_config,
            policy=policy,
            protocol=protocol,
        )

    def init_action(self, features: List[Feature] = []) -> Action:
        action = Action()
        action.set_specs(
            dim_c=1,
            dim_d=0,
            range=(np.array([0.0]), np.array([0.8])),  # price range from coordinator
        )
        action.set_values(c=np.array([0.25], dtype=np.float32))  # default price
        return action

    def set_action(self, action: Any, *args, **kwargs) -> None:
        if isinstance(action, Action):
            if action.c.size > 0:
                self.action.set_values(c=action.c)
        elif isinstance(action, np.ndarray):
            self.action.set_values(c=action.flatten()[:1])
        elif isinstance(action, (int, float)):
            self.action.set_values(c=np.array([float(action)], dtype=np.float32))

    def set_state(self, **kwargs) -> None:
        if 'p_kw' in kwargs:
            self.state.update_feature("ChargerFeature", p_kw=kwargs['p_kw'])
        if 'open_or_not' in kwargs:
            self.state.update_feature("ChargerFeature", open_or_not=kwargs['open_or_not'])
        if 'occupied' in kwargs:
            self.state.update_feature("EVSlotFeature", occupied=kwargs['occupied'])
        if 'soc' in kwargs:
            self.state.update_feature("EVSlotFeature", soc=kwargs['soc'])
        if 'soc_target' in kwargs:
            self.state.update_feature("EVSlotFeature", soc_target=kwargs['soc_target'])
        if 'arrival_time' in kwargs:
            self.state.update_feature("EVSlotFeature", arrival_time=kwargs['arrival_time'])
        if 'price_sensitivity' in kwargs:
            self.state.update_feature("EVSlotFeature", price_sensitivity=kwargs['price_sensitivity'])

    def apply_action(self) -> None:
    
        pass

    @Agent.handler("action_effect")
    def action_effect_handler(self, event: Event, scheduler: EventScheduler) -> None:
        """No-op: simulation handles all state updates for charging slots.

        Skips the default FieldAgent handler (which calls apply_action + sends
        state to proxy) since apply_action is a no-op and the state would be
        redundant — simulation fully recomputes slot states.
        """
        pass

    def compute_local_reward(self, local_state: dict) -> float:
        """Per-slot reward: 0 for empty slots, positive for charging revenue.

        The local_state dict maps feature names to numpy vectors (from proxy
        via state.observed_by()). Reward = normalized power for occupied slots.
        """
        ev_vec = local_state.get("EVSlotFeature")
        if ev_vec is None:
            return 0.0
        # EVSlotFeature.vector(): [occupied, soc, soc_target, price_sensitivity]
        occupied = float(ev_vec[0])
        if occupied < 0.5:
            return 0.0

        charger_vec = local_state.get("ChargerFeature")
        if charger_vec is None:
            return 0.0
        # ChargerFeature.vector(): [p_norm, open_or_not]
        p_norm = float(charger_vec[0])
        return p_norm
