"""Market scenario simulation."""

import numpy as np


class MarketScenario:
    def __init__(
        self,
        arrival_rate: float,
        price_freq: float,
        lmp_base: float = 0.20,
        lmp_amp: float = 0.10,
    ):
        self.arrival_rate = arrival_rate
        self.price_freq = price_freq
        self.lmp_base = float(lmp_base)
        self.lmp_amp = float(lmp_amp)
        self.time_seconds = 0.0
        self.last_price_update = -price_freq
        self.current_lmp = self.lmp_base

    def step(self, dt: float):
        self.time_seconds += dt
        if self.time_seconds - self.last_price_update >= self.price_freq:
            self.current_lmp = self.lmp_base + self.lmp_amp * np.sin(
                2 * np.pi * self.time_seconds / 86400
            )
            self.last_price_update = self.time_seconds
        return {"lmp": self.current_lmp, "t": self.time_seconds,
                "arrivals": np.random.poisson(self.arrival_rate * dt / 3600.0)}
