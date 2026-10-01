import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from kaggle_arc.aethelnet_dsl import (
    is_corner_deadlock,
    evaluate_push_deadlock,
    veto_deadlock_move,
    AethelnetVault,
    NogoodClause,
    RealActiveInferenceSolver,
    AethelnetGameOntology
)


class TestArc3DeadlockNogood(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_corner_deadlock_detection(self):
        # 10x10 room with boundary walls
        grid = np.zeros((10, 10), dtype=int)
        grid[0, :] = 1  # Top wall
        grid[:, 0] = 1  # Left wall
        grid[9, :] = 1  # Bottom wall
        grid[:, 9] = 1  # Right wall

        # Top-left corner (1, 1) bordered by (0, 1) and (1, 0)
        self.assertTrue(is_corner_deadlock(grid, (1, 1)))

        # Top-right corner (1, 8) bordered by (0, 8) and (1, 9)
        self.assertTrue(is_corner_deadlock(grid, (1, 8)))

        # Open space at (5, 5)
        self.assertFalse(is_corner_deadlock(grid, (5, 5)))

        # Next to single top wall at (1, 5) - only top is blocked, left and right open
        self.assertFalse(is_corner_deadlock(grid, (1, 5)))

        # Goal coordinate exemption
        goals = {(1, 1)}
        self.assertFalse(is_corner_deadlock(grid, (1, 1), goal_coords=goals))

    def test_evaluate_push_deadlock(self):
        grid = np.zeros((10, 10), dtype=int)
        grid[0, :] = 1  # Top wall
        grid[:, 0] = 1  # Left wall
        grid[1, 2] = 2  # Box

        # Pushing LEFT (action 3) moves box from (1, 2) to (1, 1) -> corner deadlock!
        deadlock, reason = evaluate_push_deadlock(grid, (1, 2), 3)
        self.assertTrue(deadlock)
        self.assertIn("corner deadlock", reason.lower())

        # Pushing RIGHT (action 4) moves box from (1, 2) to (1, 3) -> slides along single wall, not a corner!
        deadlock, reason = evaluate_push_deadlock(grid, (1, 2), 4)
        self.assertFalse(deadlock)
        self.assertIsNone(reason)

        # Pushing DOWN (action 2) into open room (2, 2)
        deadlock, reason = evaluate_push_deadlock(grid, (1, 2), 2)
        self.assertFalse(deadlock)
        self.assertIsNone(reason)

    def test_nogood_vault_memory_and_prompt_injection(self):
        vault = AethelnetVault("test_game", storage_dir=self.temp_dir)
        self.assertEqual(len(vault.get_nogoods_for_level(1)), 0)

        # Remember a nogood clause
        vault.remember_nogood(
            level=1,
            state_signature="lvl1_box(1,1)",
            forbidden_action="ACTION3",
            reason="Pushed into corner",
            turn=12
        )

        # Query nogood
        is_ng, reason = vault.is_action_nogood(1, "lvl1_box(1,1)", "ACTION3")
        self.assertTrue(is_ng)
        self.assertEqual(reason, "Pushed into corner")

        # Different action is not nogood
        is_ng, _ = vault.is_action_nogood(1, "lvl1_box(1,1)", "ACTION4")
        self.assertFalse(is_ng)

        # Prompt header injection for level 1
        hdr = vault.format_prompt_header(current_level=1)
        self.assertIn("=== POST-RESET LESSONS (DO NOT REPEAT FAILURE) ===", hdr)
        self.assertIn("FORBIDDEN: At state 'lvl1_box(1,1)', DO NOT execute ACTION3!", hdr)
        self.assertIn("MANDATE: At this state, explore an alternate viable action branch.", hdr)

        # Different level has no nogoods
        hdr_lvl2 = vault.format_prompt_header(current_level=2)
        self.assertNotIn("POST-RESET LESSONS", hdr_lvl2)

    def test_active_inference_controlled_reset(self):
        # Construct grid where cargo (color 2) is already in a corner deadlock
        grid = np.zeros((10, 10), dtype=int)
        grid[0, :] = 1  # Top wall
        grid[:, 0] = 1  # Left wall
        # Multi-cell pushable cargo (color 2) in corner (1, 1) to (2, 2)
        grid[1:3, 1:3] = 2

        solver = RealActiveInferenceSolver(game_id="test_ai_game")
        solver.calibrated = True
        solver.avatar_pos = (5, 5)

        # Choose action with RESET (0) available
        act, data, reason = solver.choose_action(grid, available_actions=[0, 1, 2, 3, 4], levels_completed=0, state_name="RUNNING")
        self.assertEqual(act, 0)
        self.assertIn("ControlledReset", reason)
        self.assertEqual(solver.level_resets[0], 1)

        # Reset count cap: if it reaches 2 resets, it must not reset a third time
        solver.level_resets[0] = 2
        act2, data2, reason2 = solver.choose_action(grid, available_actions=[0, 1, 2, 3, 4], levels_completed=0, state_name="RUNNING")
        self.assertNotEqual(act2, 0)


if __name__ == "__main__":
    unittest.main()
