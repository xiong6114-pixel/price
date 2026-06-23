from dataclasses import dataclass
import numpy as np
from typing import ClassVar, Sequence
from heron.core.feature import Feature


@dataclass(slots=True)
class RegulationFeature(Feature):

    visibility: ClassVar[Sequence[str]] = ['owner', 'system']

    reg_signal: float = 0.0
    headroom_up: float = 0.0
    headroom_down: float = 0.0

    def vector(self):
        return np.array(
            [self.reg_signal, self.headroom_up, self.headroom_down],
            dtype=np.float32
        )

    def names(self):
        return [
            "reg_signal_norm",
            "headroom_up_norm",
            "headroom_down_norm",
        ]

    def to_dict(self):
        return {
            "reg_signal": self.reg_signal,
            "headroom_up": self.headroom_up,
            "headroom_down": self.headroom_down,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw):
        if "reg_signal" in kw:
            self.reg_signal = float(np.clip(kw["reg_signal"], -1.0, 1.0))

        if "headroom_up" in kw:
            self.headroom_up = float(np.clip(kw["headroom_up"], 0.0, 1.0))

        if "headroom_down" in kw:
            self.headroom_down = float(np.clip(kw["headroom_down"], 0.0, 1.0))