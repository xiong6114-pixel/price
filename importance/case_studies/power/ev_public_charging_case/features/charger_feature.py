"""Charger feature provider."""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Sequence

from heron.core.feature import Feature
from case_studies.power.ev_public_charging_case.utils import safe_div


@dataclass(slots=True)
class ChargerFeature(Feature):
    visibility: ClassVar[Sequence[str]] = ['owner']
    p_kw: float = 0.0
    p_max_kw: float = 150.0
    open_or_not: int = 1

    def vector(self) -> np.ndarray:
        return np.array(
            [safe_div(self.p_kw, self.p_max_kw), float(self.open_or_not)],
            dtype=np.float32,
        )

    def names(self):
        return ['p_norm', 'open']

    def to_dict(self):
        return {
            'p_kw': self.p_kw,
            'p_max_kw': self.p_max_kw,
            'open_or_not': self.open_or_not,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw):
        allowed = {'p_kw', 'p_max_kw', 'open_or_not'}
        for k, v in kw.items():
            if k not in allowed:
                continue
            if k == 'p_kw':
                self.p_kw = float(v)
            elif k == 'p_max_kw':
                self.p_max_kw = float(v)
            elif k == 'open_or_not':
                self.open_or_not = int(np.clip(v, 0, 1))
