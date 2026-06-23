"""Coordinator-agent compatibility type."""

from .base import Agent


class CoordinatorAgent(Agent):
    """Agent that manages subordinate agents."""

    def __init__(self, *args, subordinates=None, **kwargs):
        self.subordinates = {str(k): v for k, v in (subordinates or {}).items()}
        super().__init__(*args, **kwargs)
