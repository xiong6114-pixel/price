"""Minimal continuous action container."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Action:
    """Compatibility action object with continuous and discrete parts."""

    c: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    d: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    low: np.ndarray | None = None
    high: np.ndarray | None = None

    def set_specs(self, dim_c: int = 0, dim_d: int = 0, range=None):
        self.c = np.zeros(dim_c, dtype=np.float32)
        self.d = np.zeros(dim_d, dtype=np.int32)
        if range is not None:
            self.low, self.high = range

    def set_values(self, c=None, d=None):
        if c is not None:
            self.c = np.asarray(c, dtype=np.float32)
        if d is not None:
            self.d = np.asarray(d, dtype=np.int32)
