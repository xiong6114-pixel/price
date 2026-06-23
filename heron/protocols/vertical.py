"""Vertical protocol placeholders."""

from dataclasses import dataclass


@dataclass
class BroadcastActionProtocol:
    """Placeholder action broadcast protocol."""


@dataclass
class VerticalProtocol:
    """Placeholder coordinator protocol."""

    action_protocol: object | None = None
