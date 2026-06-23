"""Common data structures for EV charging environment simulation."""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SlotState:
    """State of a single charging slot during simulation."""
    p_kw: float = 0.0
    p_max_kw: float = 150.0
    open_or_not: int = 1
    occupied: int = 0
    soc: float = 0.0
    soc_target: float = 0.8
    battery_kwh: float = 75.0
    km_per_kwh: float = 6.0
    arrival_time: float = 0.0
    max_wait_time: float = 3600.0
    price_sensitivity: float = 0.5
    # Revenue accumulated during this step
    revenue: float = 0.0
    # Per-step accounting captured before departure/reset clears slot state.
    last_step_energy_kwh: float = 0.0
    last_step_revenue: float = 0.0
    last_step_grid_cost: float = 0.0
    last_step_profit: float = 0.0


@dataclass
class EVRequest:
    """Incoming EV charging request before being assigned to a charger."""

    origin_x: float = 0.0
    origin_y: float = 0.0
    soc: float = 0.2
    soc_target: float = 0.8
    battery_kwh: float = 75.0
    km_per_kwh: float = 6.0
    price_sensitivity: float = 0.5
    arrival_time: float = 0.0
    max_wait_time: float = 3600.0


@dataclass
class StationStepMetrics:
    """Per-station metrics collected at each simulation step."""

    price: float = 0.0
    service_fee: float = 0.0
    retail_price: float = 0.0
    lmp: float = 0.0

    served_kwh: float = 0.0
    revenue: float = 0.0
    grid_cost: float = 0.0
    profit: float = 0.0

    queue_len: int = 0
    occupied_slots: int = 0
    open_slots: int = 0
    utilization: float = 0.0

    congestion_penalty: float = 0.0
    reward: float = 0.0

    arrivals: int = 0
    abandoned: int = 0
    abandoned_soc: int = 0
    abandoned_cost: int = 0
    abandoned_full: int = 0
    abandoned_timeout: int = 0


@dataclass
class EnvState:
    """Simulation state exchanged between global_state ↔ env ↔ run_simulation."""
    slot_states: Dict[str, SlotState] = field(default_factory=dict)
    station_prices: Dict[str, float] = field(default_factory=dict)
    # Map slot_id → station_id for reverse lookup
    slot_to_station: Dict[str, str] = field(default_factory=dict)
    # Market info
    lmp: float = 0.20
    time_s: float = 0.0
    dt: float = 300.0
    new_arrivals: int = 0
    # frequency regulation info
    reg_signal: float = 0.0
    station_power: Dict[str, float] = field(default_factory=dict)
    station_capacity: Dict[str, float] = field(default_factory=dict)
    station_metrics: Dict[str, StationStepMetrics] = field(default_factory=dict)
