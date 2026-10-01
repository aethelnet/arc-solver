import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import unittest
import numpy as np
from auratic_core.causal_engine import (
    StateEncoder,
    LatentState,
    CausalTransitionModel,
    ActiveInferenceController
)

class SimulatedEcologyEnvironment:
    """
    Non-linear dynamic environment with tipping dynamics.
    State: [Resource Density, Flow Rate]
    Action: Intervention in [-1.0, 1.0]
    """
    def __init__(self):
        self.state = np.array([0.2, 0.0], dtype=np.float64)

    def step(self, action: float) -> np.ndarray:
        r, v = self.state
        # Non-linear dynamics: Logistic growth + quadratic friction + action effect
        # dr/dt = v
        # dv/dt = 0.5 * r * (1 - r) - 0.2 * v + 0.8 * action
        v_next = v + (0.5 * r * (1.0 - r) - 0.2 * v + 0.8 * action) * 0.2
        r_next = max(0.01, r + v_next * 0.2)
        self.state = np.array([r_next, v_next], dtype=np.float64)
        return self.state.copy()


class TestCausalEngine(unittest.TestCase):
    def test_encoder_and_latent_state(self):
        encoder = StateEncoder(obs_dim=2, latent_dim=2)
        raw_obs = np.array([1.5, -0.3])
        s0 = encoder.encode(raw_obs, step=0)

        self.assertEqual(s0.dim, 2)
        self.assertEqual(s0.step, 0)
        self.assertAlmostEqual(s0.distance_to(s0.mean), 0.0, places=6)

    def test_forward_model_learning(self):
        model = CausalTransitionModel(latent_dim=2, action_dim=1, lr=0.1)
        s0 = LatentState(mean=np.array([0.5, 0.1]), variance=np.array([0.05, 0.05]))
        action = np.array([0.5])
        s1 = LatentState(mean=np.array([0.6, 0.2]), variance=np.array([0.05, 0.05]))

        # Initial prediction error
        pred_before, var_before = model.predict(s0, action)
        err_before = np.linalg.norm(pred_before - s1.mean)

        # Update model across 10 repeats
        for _ in range(10):
            model.update(s0, action, s1)

        pred_after, var_after = model.predict(s0, action)
        err_after = np.linalg.norm(pred_after - s1.mean)

        # The error must decrease as causal evidence is incorporated
        self.assertLess(err_after, err_before)

    def test_active_inference_closed_loop(self):
        env = SimulatedEcologyEnvironment()
        encoder = StateEncoder(obs_dim=2, latent_dim=2)
        model = CausalTransitionModel(latent_dim=2, action_dim=1, lr=0.08)

        # Target: Stabilize resource density at 1.0, zero velocity
        # Let's see target in observation space: [1.0, 0.0]
        # We can set target state directly
        target_state = np.array([0.0, 0.0]) # Target in normalized latent space

        actions = [np.array([a]) for a in [-1.0, -0.5, 0.0, 0.5, 1.0]]
        controller = ActiveInferenceController(
            target_state=target_state,
            action_candidates=actions,
            epistemic_weight=0.5,
            pragmatic_weight=2.0,
            temperature=0.1
        )

        obs = env.state.copy()
        current_state = encoder.encode(obs, step=0)

        distances = []
        for step in range(1, 51):
            action = controller.select_action(current_state, model, greedy=(step > 30))
            next_obs = env.step(float(action[0]))
            next_state = encoder.encode(next_obs, step=step)

            # Online causal model update
            model.update(current_state, action, next_state)

            dist = current_state.distance_to(target_state)
            distances.append(dist)
            current_state = next_state

        # Verify that the controller maintained numerical stability throughout
        self.assertEqual(len(distances), 50)
        self.assertFalse(np.isnan(distances).any())
        self.assertFalse(np.isinf(distances).any())
        print(f"\n✅ Active Inference Loop completed: Initial dist = {distances[0]:.3f}, Final dist = {distances[-1]:.3f}")

    def test_active_inference_utility_weights(self):
        # Test that utility_weights drive the agent towards growth/utility instead of 0
        model = CausalTransitionModel(latent_dim=2, action_dim=1, lr=0.1)
        s0 = LatentState(mean=np.array([0.0, 0.0]), variance=np.array([0.01, 0.01]))
        
        # Action 0 leads to positive return [0.0, +2.0]
        # Action 1 leads to negative return [0.0, -2.0]
        a_gain = np.array([0.0])
        a_loss = np.array([1.0])
        s_gain = LatentState(mean=np.array([0.0, 2.0]), variance=np.array([0.01, 0.01]))
        s_loss = LatentState(mean=np.array([0.0, -2.0]), variance=np.array([0.01, 0.01]))

        for _ in range(15):
            model.update(s0, a_gain, s_gain)
            model.update(s0, a_loss, s_loss)

        # Controller with utility on dimension 1 (Account Growth)
        controller = ActiveInferenceController(
            action_candidates=[a_gain, a_loss],
            utility_weights=np.array([0.0, 5.0]), # Reward dim 1
            epistemic_weight=0.0,
            pragmatic_weight=1.0,
            temperature=0.05
        )

        evals = controller.evaluate_actions(s0, model)
        # Gain action must have lower expected free energy G (higher utility)
        g_gain = [e[1] for e in evals if e[0][0] == 0.0][0]
        g_loss = [e[1] for e in evals if e[0][0] == 1.0][0]
        self.assertLess(g_gain, g_loss)
        
        chosen = controller.select_action(s0, model, greedy=True)
        self.assertEqual(chosen[0], 0.0)
        print(f"✅ Utility Weights Test: G(Gain) = {g_gain:.2f} vs G(Loss) = {g_loss:.2f} -> Selected Gain!")


if __name__ == "__main__":
    unittest.main()
