import numpy as np
from typing import Tuple, List, Optional
from .state_space import LatentState

class CausalTransitionModel:
    """
    Forward causal dynamics model: s_{t+1} = f(s_t, a_t)
    Maintains recursive posterior distribution over transition parameters to quantify
    both expected state transition and epistemic model uncertainty.
    """
    def __init__(self, latent_dim: int, action_dim: int, ridge_alpha: float = 1.0, lr: float = 0.05):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.in_dim = latent_dim + action_dim
        self.out_dim = latent_dim
        self.lr = lr

        # Ensemble of weights to capture model disagreement / epistemic uncertainty
        self.n_ensemble = 5
        self.weights: List[np.ndarray] = [
            np.random.randn(self.in_dim, self.out_dim) * 0.1
            for _ in range(self.n_ensemble)
        ]
        
        # Precision matrix for Bayesian uncertainty estimation
        self.precision = np.eye(self.in_dim, dtype=np.float64) * ridge_alpha
        self.precision_inv = np.linalg.inv(self.precision)
        self.sample_count = 0

    def predict(self, state: LatentState, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts next state mean and epistemic variance for given (state, action).
        Returns:
            pred_mean: Shape (latent_dim,)
            pred_var: Shape (latent_dim,)
        """
        action = np.asarray(action, dtype=np.float64).flatten()
        x = np.concatenate([state.mean, action])  # Shape (in_dim,)

        # Ensemble predictions
        preds = np.array([x @ w for w in self.weights])  # Shape (n_ensemble, out_dim)
        pred_mean = np.mean(preds, axis=0)

        # Disagreement across ensemble measures epistemic uncertainty
        ensemble_var = np.var(preds, axis=0)

        # Theoretical Bayesian epistemic variance: x^T * (X^T X + alpha I)^-1 * x
        data_uncertainty = float(x @ self.precision_inv @ x)
        total_var = ensemble_var + data_uncertainty * 0.1 + 1e-4

        return pred_mean, total_var

    def update(self, state: LatentState, action: np.ndarray, next_state: LatentState):
        """
        Observes real transition (s_t, a_t -> s_{t+1}) and updates model weights
        and precision matrix via online recursive Bayesian update.
        """
        action = np.asarray(action, dtype=np.float64).flatten()
        x = np.concatenate([state.mean, action])
        y = next_state.mean

        # Update ensemble using Normalized Least Mean Squares (NLMS):
        # Step size normalized by ||x||^2 to guarantee numerical convergence:
        # eta_eff = lr / (eps + ||x||^2), ensuring |1 - eta_eff * ||x||^2| < 1.
        norm_sq = float(np.sum(x ** 2))
        eta_eff = self.lr / (1.0 + norm_sq)

        for w in self.weights:
            pred = x @ w
            err = pred - y
            # Gradient step with L2 regularization and clipping
            grad = np.outer(x, err) + 1e-4 * w
            grad = np.clip(grad, -50.0, 50.0)
            w -= eta_eff * grad

        # Recursive update of precision matrix using Sherman-Morrison formula:
        # (A + u v^T)^-1 = A^-1 - (A^-1 u v^T A^-1) / (1 + v^T A^-1 u)
        v = x.reshape(-1, 1)
        A_inv_v = self.precision_inv @ v
        denom = float(1.0 + (v.T @ A_inv_v).item())
        self.precision_inv -= (A_inv_v @ A_inv_v.T) / denom
        self.sample_count += 1
