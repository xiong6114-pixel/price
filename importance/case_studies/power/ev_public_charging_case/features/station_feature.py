"""Charging station feature provider."""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Sequence

from heron.core.feature import Feature
from case_studies.power.ev_public_charging_case.utils import safe_div, norm01

PRICE_LO = 0.0
PRICE_HI = 0.8


@dataclass(slots=True)
class ChargingStationFeature(Feature):
    visibility: ClassVar[Sequence[str]] = ['owner', 'upper_level', 'system']
    open_chargers: int = 5
    max_chargers: int = 5
    charging_price: float = 0.25

    def vector(self) -> np.ndarray:
        return np.array(
            [safe_div(self.open_chargers, self.max_chargers), norm01(self.charging_price, PRICE_LO, PRICE_HI)],
            dtype=np.float32,
        )

    def names(self):
        return ['open_norm', 'price_norm']

    def to_dict(self):
        return {
            'open_chargers': self.open_chargers,
            'max_chargers': self.max_chargers,
            'charging_price': self.charging_price,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def set_values(self, **kw):
        allowed = {'open_chargers', 'max_chargers', 'charging_price'}
        for k, v in kw.items():
            if k not in allowed:
                continue
            if k == 'charging_price':
                self.charging_price = float(v)
            elif k == 'open_chargers':
                self.open_chargers = int(v)
            elif k == 'max_chargers':
                self.max_chargers = int(v)
