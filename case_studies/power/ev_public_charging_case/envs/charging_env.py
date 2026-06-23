

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from heron.envs.base import HeronEnv
from heron.agents.coordinator_agent import CoordinatorAgent
from heron.utils.typing import AgentID, MultiAgentDict

from case_studies.power.ev_public_charging_case.envs.common import EVRequest, EnvState, SlotState, StationStepMetrics
from case_studies.power.ev_public_charging_case.envs.market_scenario import MarketScenario
from case_studies.power.ev_public_charging_case.envs.regulation_scenario import RegulationScenario


class ChargingEnv(HeronEnv):
    """Multi-station EV public charging environment."""

    def __init__(
        self,
        coordinator_agents: List[CoordinatorAgent],
        arrival_rate: float = 10.0,
        dt: float = 300.0,
        episode_length: float = 86400.0,
        env_id: str = "ev_charging_env",
        # Regulation scenario params (Route A: metrics only)
        reg_freq: float = 4.0,
        reg_alpha: float = 0.2,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.dt = float(dt)
        self.episode_length = float(episode_length)
        self._arrival_rate = float(arrival_rate)
        self.q_threshold = float(kwargs.get("q_threshold", 8.0))
        self.eta = float(kwargs.get("eta", 1.0))
        self.station_positions = kwargs.get("station_positions", None)
        self.travel_speed_kmph = float(kwargs.get("travel_speed_kmph", 40.0))
        self.omega_travel = float(kwargs.get("omega_travel", 0.5))
        self.omega_wait = float(kwargs.get("omega_wait", 1.0))
        self.omega_price = float(kwargs.get("omega_price", 1.0))
        self.dsafe_km = float(kwargs.get("dsafe_km", 2.0))
        self.mean_charge_time_min = float(kwargs.get("mean_charge_time_min", 30.0))
        self.max_queue_size = int(kwargs.get("max_queue_size", 20))
        self.center_demand_prob = float(kwargs.get("center_demand_prob", 0.80))
        self.center_sigma = float(kwargs.get("center_sigma", 0.55))
        self.edge_sigma = float(kwargs.get("edge_sigma", 0.50))
        self.generalized_cost_threshold = float(kwargs.get("generalized_cost_threshold", 60.0))
        self.lmp_base = float(kwargs.get("lmp_base", 0.20))
        self.lmp_amp = float(kwargs.get("lmp_amp", 0.10))
        self.choice_lmp_weight = float(kwargs.get("choice_lmp_weight", 1.0))
        self.charge_lmp_weight = float(kwargs.get("charge_lmp_weight", 1.0))
        self.enable_in_session_price_response = bool(kwargs.get("enable_in_session_price_response", True))
        self.min_charge_power_frac = float(kwargs.get("min_charge_power_frac", 0.25))

        self._time_s = 0.0

        # RNG for reproducibility (affects arrivals assignment + SOC sampling)
        self._rng = np.random.default_rng(seed)

        # External scenarios
        self.scenario = MarketScenario(
            self._arrival_rate,
            3600.0,
            lmp_base=self.lmp_base,
            lmp_amp=self.lmp_amp,
            rng=self._rng,
        )
        self.reg_scenario = RegulationScenario(reg_freq=reg_freq, alpha=reg_alpha, seed=seed or 0)
        self._latest_station_metrics: Dict[str, Dict[str, Any]] = {}
        self._latest_reg_metrics: Dict[str, Any] = {}

        # Build slot → station mapping from coordinator subordinates
        self._slot_to_station: Dict[str, str] = {}
        for coord in coordinator_agents:
            for slot_id in coord.subordinates:
                self._slot_to_station[str(slot_id)] = str(coord.agent_id)

        if self.station_positions is None:
            ordered_station_ids = list(dict.fromkeys(self._slot_to_station.values()))
            default_positions = [
                (0.0, 0.0),
                (1.5, 0.0),
                (-1.5, 0.0),
                (0.0, 1.5),
                (0.0, -1.5),
            ]
            self.station_positions = {}
            for index, station_id in enumerate(ordered_station_ids):
                if index < len(default_positions):
                    self.station_positions[str(station_id)] = default_positions[index]
                else:
                    self.station_positions[str(station_id)] = (1.5 + 0.5 * (index - len(default_positions) + 1), 0.0)
        else:
            self.station_positions = {
                str(station_id): (float(pos[0]), float(pos[1]))
                for station_id, pos in self.station_positions.items()
            }

        self._station_queues: Dict[str, List[EVRequest]] = {
            str(station_id): [] for station_id in self.station_positions
        }
        self._station_arrivals: Dict[str, int] = {
            str(station_id): 0 for station_id in self.station_positions
        }
        self._station_abandoned: Dict[str, int] = {
            str(station_id): 0 for station_id in self.station_positions
        }
        self._station_abandoned_soc: Dict[str, int] = {
            str(station_id): 0 for station_id in self.station_positions
        }
        self._station_abandoned_cost: Dict[str, int] = {
            str(station_id): 0 for station_id in self.station_positions
        }
        self._station_abandoned_full: Dict[str, int] = {
            str(station_id): 0 for station_id in self.station_positions
        }
        self._station_abandoned_timeout: Dict[str, int] = {
            str(station_id): 0 for station_id in self.station_positions
        }

        super().__init__(
            coordinator_agents=coordinator_agents,
            env_id=env_id,
            **kwargs,
        )

    # ============================================
    # Lifecycle overrides
    # ============================================

    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Reset environment state for a new episode."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.scenario = MarketScenario(
            self._arrival_rate,
            3600.0,
            lmp_base=self.lmp_base,
            lmp_amp=self.lmp_amp,
            rng=self._rng,
        )
        # reset regulation scenario clock too
        self.reg_scenario = RegulationScenario(
            reg_freq=self.reg_scenario.reg_freq,
            alpha=self.reg_scenario.alpha,
            seed=(seed or 0),
        )

        self._time_s = 0.0
        self._latest_station_metrics = {}
        self._latest_reg_metrics = {}
        self._station_queues = {
            str(station_id): [] for station_id in self.station_positions
        }
        self._station_arrivals = {
            str(station_id): 0 for station_id in self.station_positions
        }
        self._station_abandoned = {
            str(station_id): 0 for station_id in self.station_positions
        }
        self._station_abandoned_soc = {
            str(station_id): 0 for station_id in self.station_positions
        }
        self._station_abandoned_cost = {
            str(station_id): 0 for station_id in self.station_positions
        }
        self._station_abandoned_full = {
            str(station_id): 0 for station_id in self.station_positions
        }
        self._station_abandoned_timeout = {
            str(station_id): 0 for station_id in self.station_positions
        }
        return super().reset(seed=seed, **kwargs)

    def step(self, actions: Dict[AgentID, Any]) -> Tuple[
        Dict[AgentID, Any],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, Dict],
    ]:
        """Execute one step and add __all__ + episode truncation."""
        obs, rewards, terminated, truncated, infos = super().step(actions)

        any_terminated = any(v for k, v in terminated.items() if k != "__all__")
        terminated["__all__"] = any_terminated

        time_up = self._time_s >= self.episode_length
        truncated["__all__"] = time_up

        # Route A: attach regulation metrics to system info (if present)
        # We store them under infos["__all__"] as a convenient place.
        if "__all__" not in infos:
            infos["__all__"] = {}
        infos["__all__"].update(getattr(self, "_latest_reg_metrics", {}))
        infos["__all__"]["station_metrics"] = getattr(self, "_latest_station_metrics", {})

        for sid, metrics in getattr(self, "_latest_station_metrics", {}).items():
            rewards[sid] = float(metrics["reward"])
            infos.setdefault(sid, {})
            infos[sid]["metrics"] = metrics

        return obs, rewards, terminated, truncated, infos

    # ============================================
    # Abstract simulation methods (required by BaseEnv)
    # ============================================

    def pre_step(self) -> None:
        """Advance market scenario clock (called at start of each step)."""
        # Market update is done inside run_simulation
        return

    def global_state_to_env_state(self, global_state: Dict[str, Any]) -> EnvState:
        """Extract simulation inputs from proxy global state."""
        agent_states = global_state.get("agent_states", {})
        env_state = EnvState(
            slot_to_station=dict(self._slot_to_station),
            dt=self.dt,
            time_s=self._time_s,
        )

        for agent_id, state_dict in agent_states.items():
            features = state_dict.get("features", state_dict)

            # Coordinator agent → extract pricing
            if "ChargingStationFeature" in features:
                csf = features["ChargingStationFeature"]
                env_state.station_prices[str(agent_id)] = float(csf.get("charging_price", 0.25))

            # Charging slot → extract charger + EV slot state
            if "ChargerFeature" in features and "EVSlotFeature" in features:
                cf = features["ChargerFeature"]
                ef = features["EVSlotFeature"]
                env_state.slot_states[str(agent_id)] = SlotState(
                    p_kw=float(cf.get("p_kw", 0.0)),
                    p_max_kw=float(cf.get("p_max_kw", 150.0)),
                    open_or_not=int(cf.get("open_or_not", 1)),
                    occupied=int(ef.get("occupied", 0)),
                    soc=float(ef.get("soc", 0.0)),
                    soc_target=float(ef.get("soc_target", 0.8)),
                    battery_kwh=float(ef.get("battery_kwh", 75.0)),
                    km_per_kwh=float(ef.get("km_per_kwh", 6.0)),
                    arrival_time=float(ef.get("arrival_time", 0.0)),
                    max_wait_time=float(ef.get("max_wait_time", 3600.0)),
                    price_sensitivity=float(ef.get("price_sensitivity", 0.5)),
                )

        return env_state

    def _distance_km(self, origin: Tuple[float, float], station_id: str) -> float:
        station_x, station_y = self.station_positions.get(str(station_id), (0.0, 0.0))
        dx = float(origin[0]) - float(station_x)
        dy = float(origin[1]) - float(station_y)
        return float(np.sqrt(dx * dx + dy * dy))

    def _sample_ev_request(self, time_s: float) -> EVRequest:
        if self._rng.random() < self.center_demand_prob:
            origin_x = float(self._rng.normal(0.0, self.center_sigma))
            origin_y = float(self._rng.normal(0.0, self.center_sigma))
        else:
            station_id = str(self._rng.choice(list(self.station_positions.keys())))
            base_x, base_y = self.station_positions.get(station_id, (0.0, 0.0))
            origin_x = float(base_x + self._rng.normal(0.0, self.edge_sigma))
            origin_y = float(base_y + self._rng.normal(0.0, self.edge_sigma))

        return EVRequest(
            origin_x=origin_x,
            origin_y=origin_y,
            soc=float(self._rng.uniform(0.1, 0.35)),
            soc_target=float(self._rng.uniform(0.75, 0.95)),
            battery_kwh=float(self._rng.uniform(55.0, 90.0)),
            km_per_kwh=float(self._rng.uniform(5.0, 7.0)),
            price_sensitivity=float(self._rng.uniform(0.2, 0.8)),
            arrival_time=float(time_s),
            max_wait_time=float(self._rng.uniform(1800.0, 5400.0)),
        )

    def _station_empty_slots(self, env_state: EnvState, station_id: str) -> List[str]:
        return [
            slot_id
            for slot_id, mapped_station_id in env_state.slot_to_station.items()
            if mapped_station_id == station_id
            and env_state.slot_states[slot_id].occupied == 0
            and env_state.slot_states[slot_id].open_or_not == 1
        ]

    def _estimate_wait_min(self, env_state: EnvState, station_id: str) -> float:
        queue_len = len(self._station_queues.get(station_id, []))
        open_slots = sum(
            1
            for slot_id, mapped_station_id in env_state.slot_to_station.items()
            if mapped_station_id == station_id and env_state.slot_states[slot_id].open_or_not == 1
        )
        if open_slots <= 0:
            return float(self.mean_charge_time_min * max(queue_len + 1, 1))
        if self._station_empty_slots(env_state, station_id):
            return 0.0
        effective_position = queue_len + 1
        batches = np.ceil(effective_position / max(open_slots, 1))
        return float(batches * self.mean_charge_time_min)

    def _choose_station_for_request(self, req: EVRequest, env_state: EnvState) -> Tuple[Optional[str], float]:
        best_station_id: Optional[str] = None
        best_cost = float("inf")
        energy_req_kwh = max(0.0, (req.soc_target - req.soc) * req.battery_kwh)

        for station_id in self.station_positions:
            dist_km = self._distance_km((req.origin_x, req.origin_y), station_id)
            reachable_km = req.soc * req.battery_kwh * req.km_per_kwh
            if reachable_km < dist_km + self.dsafe_km:
                continue

            service_fee = float(env_state.station_prices.get(station_id, 0.25))
            choice_price = service_fee + self.choice_lmp_weight * float(env_state.lmp)
            travel_time_min = 60.0 * dist_km / max(self.travel_speed_kmph, 1e-6)
            wait_time_min = self._estimate_wait_min(env_state, station_id)
            generalized_cost = (
                self.omega_travel * travel_time_min
                + self.omega_wait * wait_time_min
                + self.omega_price * choice_price * energy_req_kwh
            )

            if generalized_cost < best_cost:
                best_cost = generalized_cost
                best_station_id = str(station_id)

        return best_station_id, best_cost

    def _assign_request_to_slot(self, req: EVRequest, slot_state: SlotState, time_s: float) -> None:
        slot_state.occupied = 1
        slot_state.soc = float(req.soc)
        slot_state.soc_target = float(req.soc_target)
        slot_state.battery_kwh = float(req.battery_kwh)
        slot_state.km_per_kwh = float(req.km_per_kwh)
        slot_state.price_sensitivity = float(req.price_sensitivity)
        slot_state.arrival_time = float(time_s)
        slot_state.max_wait_time = float(req.max_wait_time)
        slot_state.p_kw = 0.0
        slot_state.revenue = 0.0

    def _serve_station_queue(self, env_state: EnvState, station_id: str) -> None:
        queue = self._station_queues.get(station_id, [])
        empty_slots = self._station_empty_slots(env_state, station_id)
        while queue and empty_slots:
            req = queue.pop(0)
            slot_id = empty_slots.pop(0)
            self._assign_request_to_slot(req, env_state.slot_states[slot_id], env_state.time_s)

    def _handle_new_arrivals_with_choice(self, env_state: EnvState) -> None:
        for _ in range(env_state.new_arrivals):
            req = self._sample_ev_request(env_state.time_s)
            station_id, best_cost = self._choose_station_for_request(req, env_state)
            if station_id is None:
                nearest_station_id = min(
                    self.station_positions.keys(),
                    key=lambda sid: self._distance_km((req.origin_x, req.origin_y), sid),
                )
                self._station_abandoned[nearest_station_id] = self._station_abandoned.get(nearest_station_id, 0) + 1
                self._station_abandoned_soc[nearest_station_id] = self._station_abandoned_soc.get(nearest_station_id, 0) + 1
                continue

            self._station_arrivals[station_id] = self._station_arrivals.get(station_id, 0) + 1
            if best_cost > self.generalized_cost_threshold:
                self._station_abandoned[station_id] = self._station_abandoned.get(station_id, 0) + 1
                self._station_abandoned_cost[station_id] = self._station_abandoned_cost.get(station_id, 0) + 1
                continue

            empty_slots = self._station_empty_slots(env_state, station_id)
            if empty_slots:
                self._assign_request_to_slot(req, env_state.slot_states[empty_slots[0]], env_state.time_s)
                continue

            queue = self._station_queues.setdefault(station_id, [])
            if len(queue) >= self.max_queue_size:
                self._station_abandoned[station_id] = self._station_abandoned.get(station_id, 0) + 1
                self._station_abandoned_full[station_id] = self._station_abandoned_full.get(station_id, 0) + 1
                continue
            queue.append(req)

    def _drop_timeout_queue_requests(self, env_state: EnvState) -> None:
        for station_id, queue in self._station_queues.items():
            kept_requests: List[EVRequest] = []
            dropped = 0
            for req in queue:
                waited_time = env_state.time_s - req.arrival_time
                if waited_time > req.max_wait_time:
                    dropped += 1
                else:
                    kept_requests.append(req)
            self._station_queues[station_id] = kept_requests
            if dropped:
                self._station_abandoned[station_id] = self._station_abandoned.get(station_id, 0) + dropped
                self._station_abandoned_timeout[station_id] = self._station_abandoned_timeout.get(station_id, 0) + dropped

    def run_simulation(self, env_state: EnvState, *args, **kwargs) -> EnvState:
        """Run one step of EV charging simulation."""
        # 1) Advance market scenario
        scenario_data = self.scenario.step(self.dt)
        self._time_s = float(scenario_data["t"])
        env_state.lmp = float(scenario_data["lmp"])
        env_state.time_s = float(scenario_data["t"])
        env_state.new_arrivals = int(scenario_data["arrivals"])

        # 1b) Advance regulation scenario (Route A: metrics only)
        reg_data = self.reg_scenario.step(self.dt)
        env_state.reg_signal = float(reg_data["reg_signal"])

        # 2) EV arrivals — assign to random empty slots
        self._station_arrivals = {
            str(station_id): 0 for station_id in self.station_positions
        }
        self._station_abandoned = {
            str(station_id): 0 for station_id in self.station_positions
        }

        # 3) Charging physics — compute p_kw from price + occupancy, update SOC
        for slot_id, ss in env_state.slot_states.items():
            ss.revenue = 0.0
            ss.last_step_energy_kwh = 0.0
            ss.last_step_revenue = 0.0
            ss.last_step_grid_cost = 0.0
            ss.last_step_profit = 0.0
            if ss.occupied == 0 or ss.open_or_not == 0:
                ss.p_kw = 0.0
                continue

            station_id = env_state.slot_to_station.get(slot_id)
            service_fee = float(env_state.station_prices.get(station_id, 0.25))
            retail_price = float(env_state.lmp) + service_fee
            charge_decision_price = service_fee + self.charge_lmp_weight * float(env_state.lmp)

            if not self.enable_in_session_price_response:
                ss.p_kw = ss.p_max_kw
            else:
                willingness = max(ss.price_sensitivity * 1.2, 1e-6)
                if charge_decision_price <= willingness:
                    frac = 1.0
                else:
                    frac = max(
                        self.min_charge_power_frac,
                        willingness / max(charge_decision_price, 1e-6),
                    )
                ss.p_kw = ss.p_max_kw * frac

            energy_kwh = ss.p_kw * self.dt / 3600.0
            if energy_kwh > 0 and ss.soc < ss.soc_target:
                battery_kwh = max(float(ss.battery_kwh), 1.0)
                delta_soc = energy_kwh / battery_kwh
                revenue = retail_price * energy_kwh
                grid_cost = float(env_state.lmp) * energy_kwh
                profit = revenue - grid_cost
                ss.last_step_energy_kwh = energy_kwh
                ss.last_step_revenue = revenue
                ss.last_step_grid_cost = grid_cost
                ss.last_step_profit = profit
                ss.soc = min(1.0, ss.soc + delta_soc)
                ss.revenue = revenue

        # 4) EV departures — slots where SOC >= target or max wait exceeded
        for slot_id, ss in env_state.slot_states.items():
            if ss.occupied == 0:
                continue
            time_connected = env_state.time_s - ss.arrival_time
            if ss.soc >= ss.soc_target or time_connected > ss.max_wait_time:
                ss.occupied = 0
                ss.soc = 0.0
                ss.p_kw = 0.0

        self._drop_timeout_queue_requests(env_state)
        for station_id in self.station_positions:
            self._serve_station_queue(env_state, station_id)

        self._handle_new_arrivals_with_choice(env_state)
        for station_id in self.station_positions:
            self._serve_station_queue(env_state, station_id)

        # 5) Aggregate station power/capacity (open chargers only)
        station_power: Dict[str, float] = {}
        station_capacity: Dict[str, float] = {}
        for slot_id, ss in env_state.slot_states.items():
            st = env_state.slot_to_station.get(slot_id)
            if st is None:
                continue
            station_power.setdefault(st, 0.0)
            station_capacity.setdefault(st, 0.0)

            if ss.open_or_not == 1:
                station_capacity[st] += float(ss.p_max_kw)
                # actual power only if occupied/open (we already set p_kw=0 if not)
                station_power[st] += float(ss.p_kw)

        env_state.station_power = station_power
        env_state.station_capacity = station_capacity

        # 6) Route A metrics: compute target/error/violation at station level
        # Target: request a fraction of open capacity
        alpha = float(self.reg_scenario.alpha)
        errors = []
        violation_seconds = 0.0
        max_abs_error = 0.0

        for st, cap in station_capacity.items():
            p_act = station_power.get(st, 0.0)
            # regulation target around 0 baseline (metrics-only)
            p_tgt = alpha * env_state.reg_signal * cap

            err = p_act - p_tgt
            errors.append(err)
            max_abs_error = max(max_abs_error, abs(err))

            # violation: requesting nonzero when cap is zero, or request exceeds cap (rare here)
            if cap <= 1e-9 and abs(p_tgt) > 1e-6:
                violation_seconds += self.dt

        rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0

        self._latest_reg_metrics = {
            "reg_signal": float(env_state.reg_signal),
            "reg_rmse": rmse,
            "reg_max_abs_error": float(max_abs_error),
            "reg_violation_seconds": float(violation_seconds),
        }

        station_metrics: Dict[str, StationStepMetrics] = {}
        for st, cap in station_capacity.items():
            slots_of_station = [
                sid for sid, station_id in env_state.slot_to_station.items()
                if station_id == st
            ]

            occupied = sum(env_state.slot_states[sid].occupied for sid in slots_of_station)
            open_slots = sum(env_state.slot_states[sid].open_or_not for sid in slots_of_station)

            served_kwh = 0.0
            revenue = 0.0
            grid_cost = 0.0
            profit = 0.0

            service_fee = float(env_state.station_prices.get(st, 0.25))
            lmp = float(env_state.lmp)
            retail_price = lmp + service_fee

            for sid in slots_of_station:
                ss = env_state.slot_states[sid]
                served_kwh += float(ss.last_step_energy_kwh)
                revenue += float(ss.last_step_revenue)
                grid_cost += float(ss.last_step_grid_cost)
                profit += float(ss.last_step_profit)
            queue_len = int(len(self._station_queues.get(st, [])))
            congestion_penalty = self.eta * max(0.0, queue_len - self.q_threshold)
            reward = profit - congestion_penalty
            utilization = occupied / max(open_slots, 1)

            station_metrics[st] = StationStepMetrics(
                price=service_fee,
                service_fee=service_fee,
                retail_price=retail_price,
                lmp=lmp,
                served_kwh=served_kwh,
                revenue=revenue,
                grid_cost=grid_cost,
                profit=profit,
                queue_len=queue_len,
                occupied_slots=int(occupied),
                open_slots=int(open_slots),
                utilization=float(utilization),
                congestion_penalty=float(congestion_penalty),
                reward=float(reward),
                arrivals=int(self._station_arrivals.get(st, 0)),
                abandoned=int(self._station_abandoned.get(st, 0)),
                abandoned_soc=int(self._station_abandoned_soc.get(st, 0)),
                abandoned_cost=int(self._station_abandoned_cost.get(st, 0)),
                abandoned_full=int(self._station_abandoned_full.get(st, 0)),
                abandoned_timeout=int(self._station_abandoned_timeout.get(st, 0)),
            )

        env_state.station_metrics = station_metrics
        self._latest_station_metrics = {
            sid: metrics.__dict__.copy()
            for sid, metrics in station_metrics.items()
        }

        return env_state

    def env_state_to_global_state(self, env_state: EnvState) -> Dict[str, Any]:
        """Convert simulation results back to proxy global state format."""
        agent_states: Dict[str, Any] = {}

        # Update slot agent states (FieldAgent, level 1)
        for slot_id, ss in env_state.slot_states.items():
            agent_states[slot_id] = {
                "_owner_id": slot_id,
                "_owner_level": 1,
                "_state_type": "FieldAgentState",
                "features": {
                    "ChargerFeature": {
                        "p_kw": ss.p_kw,
                        "p_max_kw": ss.p_max_kw,
                        "open_or_not": ss.open_or_not,
                    },
                    "EVSlotFeature": {
                        "occupied": ss.occupied,
                        "soc": ss.soc,
                        "soc_target": ss.soc_target,
                        "battery_kwh": ss.battery_kwh,
                        "km_per_kwh": ss.km_per_kwh,
                        "arrival_time": ss.arrival_time,
                        "max_wait_time": ss.max_wait_time,
                        "price_sensitivity": ss.price_sensitivity,
                    },
                },
            }

        # Update coordinator agent states (level 2)
        for station_id, price in env_state.station_prices.items():
            # Count open chargers for this station (occupied==0 means open/available)
            station_slots = [sid for sid, st in env_state.slot_to_station.items() if st == station_id]
            open_count = sum(
                1 for sid in station_slots
                if sid in env_state.slot_states and env_state.slot_states[sid].occupied == 0 and env_state.slot_states[sid].open_or_not == 1
            )

            # RegulationFeature write-back (so StationCoordinator obs last 3 dims are non-zero)
            cap = float(env_state.station_capacity.get(station_id, 0.0))
            p_act = float(env_state.station_power.get(station_id, 0.0))
            if cap > 1e-9:
                headroom_up = (cap - p_act) / cap
                headroom_down = p_act / cap
            else:
                headroom_up = 0.0
                headroom_down = 0.0

            agent_states[station_id] = {
                "_owner_id": station_id,
                "_owner_level": 2,
                "_state_type": "CoordinatorAgentState",
                "features": {
                    "ChargingStationFeature": {
                        "charging_price": float(price),
                        "open_chargers": int(open_count),
                    },
                    "MarketFeature": {
                        "lmp": float(env_state.lmp),
                        "t_day_s": float(env_state.time_s),
                    },
                    "RegulationFeature": {
                        "reg_signal": float(env_state.reg_signal),
                        "headroom_up": float(headroom_up),
                        "headroom_down": float(headroom_down),
                    },
                },
            }

        return {"agent_states": agent_states}
