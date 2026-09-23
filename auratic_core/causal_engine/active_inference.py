import numpy as np
from typing import List, Tuple, Optional
from .state_space import LatentState
from .forward_model import CausalTransitionModel

class ActiveInferenceController:
    """
    Active Inference Controller based on the Free Energy Principle.
    Selects actions by minimizing Expected Free Energy (EFE):
        G(a) = Pragmatic Cost (Distance to Target) - Epistemic Value (Information Gain)
    """
    def __init__(
        self,
        target_state: Optional[np.ndarray] = None,
        action_candidates: Optional[List[np.ndarray]] = None,
        utility_weights: Optional[np.ndarray] = None,
        risk_weight: float = 0.0,
        epistemic_weight: float = 1.0,
        pragmatic_weight: float = 1.0,
        temperature: float = 0.2,
    ):
        if target_state is not None:
            self.target_state = np.asarray(target_state, dtype=np.float64)
        else:
            self.target_state = None
        self.action_candidates = [np.asarray(a, dtype=np.float64) for a in (action_candidates or [])]
        self.utility_weights = np.asarray(utility_weights, dtype=np.float64) if utility_weights is not None else None
        self.risk_weight = risk_weight
        self.epistemic_weight = epistemic_weight
        self.pragmatic_weight = pragmatic_weight
        self.temperature = temperature

    def evaluate_actions(
        self,
        current_state: LatentState,
        model: CausalTransitionModel
    ) -> List[Tuple[np.ndarray, float, float, float]]:
        """
        Evaluates all candidate actions under Expected Free Energy.
        Returns list of tuples: (action, total_G, pragmatic_cost, epistemic_value)
        """
        results = []

        for a in self.action_candidates:
            pred_mean, pred_var = model.predict(current_state, a)

            # 1. Pragmatic Cost:
            # If utility_weights are provided: Maximizing utility means minimizing -Utility
            # Otherwise: Squared divergence from preferred target
            if self.utility_weights is not None:
                pragmatic_cost = float(-np.dot(pred_mean, self.utility_weights))
            elif self.target_state is not None:
                pragmatic_cost = float(np.sum((pred_mean - self.target_state) ** 2))
            else:
                pragmatic_cost = 0.0

            # Risk penalty (variance aversion)
            risk_penalty = float(self.risk_weight * np.sum(pred_var))

            # 2. Epistemic Value (Information Gain / Exploration Incentive):
            # Proportional to the entropy / variance of the prediction.
            # High variance = High potential to learn causal structure.
            epistemic_value = float(0.5 * np.sum(np.log(pred_var + 1e-4)))

            # Expected Free Energy: We want LOW pragmatic cost and HIGH epistemic value
            total_g = (self.pragmatic_weight * pragmatic_cost) + risk_penalty - (self.epistemic_weight * epistemic_value)

            results.append((a, total_g, pragmatic_cost, epistemic_value))

        return results

    def select_action(
        self,
        current_state: LatentState,
        model: CausalTransitionModel,
        greedy: bool = False
    ) -> np.ndarray:
        """
        Selects best action either greedily (arg min G) or probabilistically via Softmax.
        """
        evals = self.evaluate_actions(current_state, model)
        actions = [e[0] for e in evals]
        g_values = np.array([e[1] for e in evals])

        if greedy:
            best_idx = int(np.argmin(g_values))
            return actions[best_idx]

        # Softmax sampling over negative G (Boltzmann distribution)
        neg_g = -g_values / max(self.temperature, 1e-6)
        # Shift for numerical stability
        neg_g -= np.max(neg_g)
        probs = np.exp(neg_g)
        probs /= np.sum(probs)

        chosen_idx = int(np.random.choice(len(actions), p=probs))
        return actions[chosen_idx]
