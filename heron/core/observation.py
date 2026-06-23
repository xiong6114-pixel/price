"""Observation container."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Observation:
    """Simple observation wrapper."""

    timestamp: int | float = 0
    local: Dict[str, Any] = field(default_factory=dict)
