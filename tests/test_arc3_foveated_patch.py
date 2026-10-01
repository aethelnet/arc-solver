import unittest
import numpy as np

from kaggle_arc.aethelnet_dsl import (
    extract_foveated_patch,
    format_macro_radar,
    track_entity_velocities,
    simulate_step,
    AethelnetGameOntology,
    OntologyEntity
)


class TestArc3FoveatedPatch(unittest.TestCase):
    def test_extract_foveated_patch(self):
        # 20x20 grid
        grid = np.zeros((20, 20), dtype=int)
        grid[10, 10] = 5  # Center entity

        patch = extract_foveated_patch(grid, center=(10, 10), radius=2)
        self.assertEqual(len(patch["lines"]), 5)  # 2*2 + 1 = 5
        self.assertEqual(len(patch["lines"][0]), 5)
        # Center of patch should be '5'
        self.assertEqual(patch["lines"][2][2], "5")

        # Corner test (center at 0, 0) - should pad out of bounds with '#'
        patch_corner = extract_foveated_patch(grid, center=(0, 0), radius=2)
        self.assertEqual(len(patch_corner["lines"]), 5)
        self.assertEqual(patch_corner["lines"][0][0], "#")  # Out of bounds padded with '#'

    def test_format_macro_radar(self):
        entities = [
            OntologyEntity(id=1, color="1", archetype="GOAL", bbox=(5, 15, 6, 16), area=4, center=(5, 15)),
            OntologyEntity(id=2, color="2", archetype="PUSHABLE", bbox=(5, 7, 6, 8), area=4, center=(5, 7)),
        ]
        radar = format_macro_radar(entities, avatar_pos=(5, 5))
        self.assertIn("=== AETHELNET MACRO-RADAR ===", radar)
        self.assertIn("AVATAR: at (5, 5)", radar)
        self.assertIn("PUSHABLE #2 (col '2'): at (5, 7) [Offset: (+0, +2), Dist: 2]", radar)
        self.assertIn("GOAL #1 (col '1'): at (5, 15) [Offset: (+0, +10), Dist: 10]", radar)

    def test_track_entity_velocities(self):
        prev = [
            OntologyEntity(id=1, color="3", archetype="HAZARD", bbox=(10, 10, 10, 10), area=1, center=(10, 10)),
        ]
        curr = [
            OntologyEntity(id=1, color="3", archetype="HAZARD", bbox=(11, 10, 11, 10), area=1, center=(11, 10)),
        ]
        vels = track_entity_velocities(prev, curr)
        self.assertEqual(len(vels), 1)
        self.assertEqual(vels[0]["velocity"], (1, 0))
        self.assertEqual(vels[0]["direction"], "DOWN")

    def test_simulate_step_move_and_push(self):
        grid = np.zeros((10, 10), dtype=int)
        grid[5, 5] = 2  # Avatar at (5, 5)
        grid[5, 6] = 3  # Pushable cargo at (5, 6)
        # Note: 3x3 room around them is open

        # Action 4 is RIGHT
        pred_lines, is_deadlock, events = simulate_step(grid, action=4, avatar_pos=(5, 5))
        self.assertFalse(is_deadlock)
        self.assertTrue(any("Pushed cargo" in e or "Avatar moved" in e for e in events))


if __name__ == "__main__":
    unittest.main()
