import numpy as np

class RegulationScenario:
    def __init__(self, reg_freq: float, alpha: float, process: str = "ou", seed: int = 0):
        self.reg_freq = reg_freq          # e.g., 4.0 seconds per update
        self.alpha = alpha                # scale of requested regulation, e.g., 0.2 of capacity
        self.time_seconds = 0.0
        self.last_reg_update = -reg_freq
        self.reg_signal = 0.0
        self.rng = np.random.default_rng(seed)

    def step(self, dt: float):
        self.time_seconds += dt
        if self.time_seconds - self.last_reg_update >= self.reg_freq:
            # example: bounded OU-like update (simple)
            noise = self.rng.normal(0.0, 0.3)
            self.reg_signal = float(np.clip(0.9 * self.reg_signal + 0.1 * noise, -1.0, 1.0))
            self.last_reg_update = self.time_seconds
        return {"reg_signal": self.reg_signal, "t": self.time_seconds}