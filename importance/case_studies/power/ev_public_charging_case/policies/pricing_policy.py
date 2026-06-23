"""Neural pricing policy for StationCoordinator.

Follows the NeuralPolicy pattern from tests/integration/test_e2e.py.
Uses actor-critic architecture with policy gradient updates.

Observation: 8D local (ChargingStationFeature 2D + MarketFeature 3D + RegulationFeature 3D)
Action: 1D pricing in [0.0, 0.8] $/kWh
"""

import numpy as np

from heron.core.policies import Policy, obs_to_vector, vector_to_action
from heron.core.observation import Observation


class SimpleMLP:
    """Simple MLP with ReLU hidden layer and tanh output."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.W1 + self.b1)
        return np.tanh(h @ self.W2 + self.b2)

    def update(self, x: np.ndarray, target: np.ndarray, lr: float = 0.01) -> None:
        h = np.maximum(0, x @ self.W1 + self.b1)
        out = np.tanh(h @ self.W2 + self.b2)
        d_out = (out - target) * (1 - out**2)
        self.W2 -= lr * np.outer(h, d_out)
        self.b2 -= lr * d_out.flatten()
        d_h = d_out @ self.W2.T
        d_h[h <= 0] = 0
        self.W1 -= lr * np.outer(x, d_h)
        self.b1 -= lr * d_h.flatten()


class PricingActorMLP(SimpleMLP):
    """Actor MLP for pricing actions.

    Output is sigmoid-scaled to [0, 0.8] instead of tanh [-1, 1],
    matching the StationCoordinator's action_space.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        super().__init__(input_dim, hidden_dim, output_dim, seed)
        # Smaller initial weights for stable exploration
        rng = np.random.RandomState(seed + 100)
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.1
        # Bias toward mid-range pricing (~0.25 $/kWh)
        self.b2 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.W1 + self.b1)
        raw = h @ self.W2 + self.b2
        # Sigmoid scaled to [0, 0.8]
        return 0.8 / (1.0 + np.exp(-raw))

    def update(self, x: np.ndarray, action_taken: np.ndarray, advantage: float, lr: float = 0.01) -> None:
        """Policy gradient update."""
        h = np.maximum(0, x @ self.W1 + self.b1)
        raw = h @ self.W2 + self.b2
        current_action = 0.8 / (1.0 + np.exp(-raw))

        # Sigmoid derivative: sigmoid * (1 - sigmoid) * 0.8
        sig = 1.0 / (1.0 + np.exp(-raw))
        d_sigmoid = sig * (1 - sig) * 0.8

        error = current_action - action_taken
        grad_scale = advantage * d_sigmoid

        d_W2 = np.outer(h, grad_scale * error)
        d_b2 = (grad_scale * error).flatten()
        d_h = (grad_scale * error) @ self.W2.T
        d_h[h <= 0] = 0
        d_W1 = np.outer(x, d_h)
        d_b1 = d_h.flatten()

        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1


class PricingPolicy(Policy):
    """Neural pricing policy for StationCoordinator.

    Architecture: obs (8D local) -> hidden (32 ReLU) -> action (1D, sigmoid * 0.8)

    Uses LOCAL-ONLY observations via observation_mode for decentralized execution.
    The policy learns to set prices that maximize charging revenue minus energy cost.
    """

    observation_mode = "local"

    def __init__(self, obs_dim: int = 8, action_dim: int = 1, hidden_dim: int = 32, seed: int = 42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = (0.0, 0.8)
        self.hidden_dim = hidden_dim

        self.actor = PricingActorMLP(obs_dim, hidden_dim, action_dim, seed)
        self.critic = SimpleMLP(obs_dim, hidden_dim, 1, seed + 1)

        self.noise_scale = 0.08

    @obs_to_vector
    @vector_to_action
    def forward(self, obs_vec: np.ndarray) -> np.ndarray:
        """Compute pricing action with exploration noise."""
        action_mean = self.actor.forward(obs_vec)
        action_vec = action_mean + np.random.normal(0, self.noise_scale, self.action_dim)
        return np.clip(action_vec, 0.0, 0.8)

    @obs_to_vector
    @vector_to_action
    def forward_deterministic(self, obs_vec: np.ndarray) -> np.ndarray:
        """Compute pricing action without exploration noise."""
        return self.actor.forward(obs_vec)

    @obs_to_vector
    def get_value(self, obs_vec: np.ndarray) -> float:
        """Estimate value of current state."""
        return float(self.critic.forward(obs_vec)[0])

    def update(self, obs: np.ndarray, action_taken: np.ndarray, advantage: float, lr: float = 0.01) -> None:
        """Update actor using policy gradient with advantage."""
        self.actor.update(obs, action_taken, advantage, lr)

    def update_critic(self, obs: np.ndarray, target: float, lr: float = 0.01) -> None:
        """Update critic to estimate state values."""
        self.critic.update(obs, np.array([target]), lr)

    def decay_noise(self, decay_rate: float = 0.995, min_noise: float = 0.02) -> None:
        """Decay exploration noise over training."""
        self.noise_scale = max(min_noise, self.noise_scale * decay_rate)
