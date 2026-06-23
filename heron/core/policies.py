"""Policy helpers used by the local training script."""

from functools import wraps

import numpy as np

from .action import Action
from .observation import Observation


class Policy:
    """Minimal policy base with observation vectorization helpers."""

    observation_mode = "local"

    def extract_obs_vector(self, observation, expected_dim: int | None = None) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            vec = observation.astype(np.float32).flatten()
        elif isinstance(observation, Observation):
            if "obs" in observation.local:
                vec = np.asarray(observation.local["obs"], dtype=np.float32).flatten()
            else:
                parts = []
                for key in sorted(observation.local):
                    parts.append(np.asarray(observation.local[key], dtype=np.float32).flatten())
                vec = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        else:
            vec = np.asarray(observation, dtype=np.float32).flatten()

        if expected_dim is not None:
            if vec.size < expected_dim:
                vec = np.pad(vec, (0, expected_dim - vec.size))
            elif vec.size > expected_dim:
                vec = vec[:expected_dim]
        return vec.astype(np.float32)


def obs_to_vector(func):
    """Convert HERON Observation inputs into flat numpy arrays."""

    @wraps(func)
    def wrapper(self, observation, *args, **kwargs):
        obs_vec = self.extract_obs_vector(observation)
        return func(self, obs_vec, *args, **kwargs)

    return wrapper


def vector_to_action(func):
    """Convert action vectors into Action objects."""

    @wraps(func)
    def wrapper(self, observation, *args, **kwargs):
        result = func(self, observation, *args, **kwargs)
        if isinstance(result, Action):
            return result
        action = Action()
        action.set_specs(dim_c=np.asarray(result).size, dim_d=0)
        action.set_values(c=np.asarray(result, dtype=np.float32).flatten())
        return action

    return wrapper
