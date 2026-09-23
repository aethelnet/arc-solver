from dataclasses import dataclass, field
import numpy as np
from typing import Optional, List, Dict, Any

@dataclass
class LatentState:
    """
    Continuous latent representation s_t in R^d with estimated uncertainty.
    """
    mean: np.ndarray          # Shape (d,)
    variance: np.ndarray      # Shape (d,) - epistemic uncertainty per dimension
    step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.variance = np.asarray(self.variance, dtype=np.float64)
        if self.mean.shape != self.variance.shape:
            raise ValueError(f"Shape mismatch: mean {self.mean.shape} vs variance {self.variance.shape}")

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    def distance_to(self, target: np.ndarray) -> float:
        """Euclidean distance to a target state vector."""
        target = np.asarray(target, dtype=np.float64)
        return float(np.linalg.norm(self.mean - target))


class StateEncoder:
    """
    Encodes raw observations o_t into normalized latent space s_t.
    Tracks running statistics for zero-mean, unit-variance normalization.
    """
    def __init__(self, obs_dim: int, latent_dim: int, momentum: float = 0.05):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.momentum = momentum
        
        self.running_mean = np.zeros(obs_dim, dtype=np.float64)
        self.running_var = np.ones(obs_dim, dtype=np.float64)
        self.projection = np.random.randn(obs_dim, latent_dim) / np.sqrt(obs_dim)
        self.is_initialized = False

    def encode(self, obs: np.ndarray, step: int = 0) -> LatentState:
        obs = np.asarray(obs, dtype=np.float64).flatten()
        if len(obs) != self.obs_dim:
            raise ValueError(f"Expected obs_dim {self.obs_dim}, got {len(obs)}")

        if not self.is_initialized:
            self.running_mean = obs.copy()
            self.running_var = np.ones_like(obs)
            self.is_initialized = True
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * obs
            diff = obs - self.running_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * (diff ** 2)

        std = np.sqrt(self.running_var + 1e-8)
        norm_obs = (obs - self.running_mean) / std

        # Project into latent space: s = norm_obs @ W
        if self.obs_dim == self.latent_dim:
            latent_mean = norm_obs
        else:
            latent_mean = norm_obs @ self.projection

        # Baseline measurement uncertainty
        latent_var = np.full_like(latent_mean, 0.05)

        return LatentState(mean=latent_mean, variance=latent_var, step=step)
