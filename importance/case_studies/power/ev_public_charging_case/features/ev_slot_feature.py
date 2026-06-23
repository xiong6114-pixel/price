"""EV slot feature provider.

Represents the EV occupancy state of a charging slot.
When occupied=1, the slot has an EV with the given SOC and parameters.
When occupied=0, the slot is empty (SOC fields are meaningless).
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Sequence

from heron.core.feature import Feature


@dataclass(slots=True)
class EVSlotFeature(Feature):
    visibility: ClassVar[Sequence[str]] = ['owner']
    occupied: int = 0
    soc: float = 0.0
    soc_target: float = 0.8
    arrival_time: float = 0.0
    max_wait_time: float = 3600.0
    price_sensitivity: float = 0.5

    def vector(self) -> np.ndarray:
        return np.array(
            [float(self.occupied), self.soc, self.soc_target, self.price_sensitivity],
            dtype=np.float32,
        )

    def names(self):
        return ['occupied', 'soc', 'soc_target', 'price_sensitivity']

    def to_dict(self):
        return {
            'occupied': self.occupied,
            'soc': self.soc,
            'soc_target': self.soc_target,
            'arrival_time': self.arrival_time,
            'max_wait_time': self.max_wait_time,
            'price_sensitivity': self.price_sensitivity,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw):
        allowed = {'occupied', 'soc', 'soc_target', 'arrival_time', 'max_wait_time', 'price_sensitivity'}
        for k, v in kw.items():
            if k not in allowed:
                continue
            if k == 'occupied':
                self.occupied = int(np.clip(v, 0, 1))
            elif k == 'soc':
                self.soc = float(np.clip(v, 0.0, 1.0))
            elif k == 'soc_target':
                self.soc_target = float(np.clip(v, 0.0, 1.0))
            elif k == 'arrival_time':
                self.arrival_time = float(v)
            elif k == 'max_wait_time':
                self.max_wait_time = float(v)
            elif k == 'price_sensitivity':
                self.price_sensitivity = float(np.clip(v, 0.0, 1.0))
