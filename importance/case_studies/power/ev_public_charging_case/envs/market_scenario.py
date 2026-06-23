"""Market scenario simulation."""

import numpy as np


class MarketScenario:
    def __init__(self, arrival_rate: float, price_freq: float):
        self.arrival_rate, self.price_freq = arrival_rate, price_freq
        self.time_seconds, self.last_price_update = 0.0, -price_freq
        self.current_lmp = 0.20

    def step(self, dt: float):
        self.time_seconds += dt
        if self.time_seconds - self.last_price_update >= self.price_freq:
            self.current_lmp = 0.2 + 0.1 * np.sin(2 * np.pi * self.time_seconds / 86400)
            self.last_price_update = self.time_seconds
        return {"lmp": self.current_lmp, "t": self.time_seconds,
                "arrivals": np.random.poisson(self.arrival_rate * dt / 3600.0)}
