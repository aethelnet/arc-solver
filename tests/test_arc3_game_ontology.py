"""
Unit tests for ARC-3 Inductive Game Ontology, Morphological Congruence, and Actuator Affordances.
"""

import sys
from pathlib import Path
import unittest

# Ensure kaggle_arc is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "kaggle_arc"))

import aethelnet_dsl as aethel


class TestARC3GameOntology(unittest.TestCase):

    def setUp(self):
        # 16x16 test board:
        # - Top/bottom border walls (color '#')
        # - Hollow 4x4 slot (color '8', inner 2x2 empty)
        # - Solid 2x2 cargo block (color '4')
        # - Switch button (color '2', 1x1)
        # - Matching gate barrier (color '2', 1x4)
        # - Blueprint template in TOP_RIGHT corner with colors '1', '3'
        self.lines = [
            "################",
            "#..............#",
            "#...8888.......#",
            "#...8..8...13..#",
            "#...8..8...31..#",
            "#...8888.......#",
            "#..............#",
            "#...44.........#",
            "#...44.........#",
            "#..............#",
            "#...2..........#",
            "#..............#",
            "#...2222.......#",
            "#..............#",
            "#..............#",
            "################",
        ]

        class MockFrame:
            def __init__(self, lines):
                self.ascii = "\n".join(lines)

        self.frame = MockFrame(self.lines)

    def test_extract_entities_and_archetypes(self):
        bg, entities = aethel.AethelnetGameOntology.extract_entities(self.frame)
        self.assertEqual(bg, ".")
        self.assertTrue(len(entities) >= 5)

        # Verify archetypes are assigned properly
        archetypes = {e.archetype for e in entities}
        self.assertIn("GOAL", archetypes)
        self.assertIn("ACTUATOR", archetypes)
        self.assertIn("GATE", archetypes)

        # Check hollow slot properties
        slots = [e for e in entities if e.archetype == "GOAL" and e.color == "8"]
        self.assertEqual(len(slots), 1)
        slot = slots[0]
        self.assertTrue(slot.is_hollow)
        self.assertEqual(slot.inner_area, 4)
        self.assertEqual(slot.dimensions, (4, 4))

        # Check gate properties
        gates = [e for e in entities if e.archetype == "GATE" and e.color == "2"]
        self.assertEqual(len(gates), 1)
        gate = gates[0]
        self.assertEqual(gate.dimensions, (1, 4))

    def test_match_slot_cargo(self):
        _, entities = aethel.AethelnetGameOntology.extract_entities(self.frame)
        matches = aethel.AethelnetGameOntology.match_slot_cargo(entities)
        self.assertTrue(len(matches) >= 1)

        # Entity 7 (cargo color 4, 2x2 area 4) should match slot color 8 (inner area 4)
        top_match = matches[0]
        self.assertEqual(top_match["cargo_color"], "4")
        self.assertEqual(top_match["slot_color"], "8")
        self.assertGreaterEqual(top_match["score"], 0.7)

    def test_detect_actuator_affordances(self):
        _, entities = aethel.AethelnetGameOntology.extract_entities(self.frame)
        affordances = aethel.AethelnetGameOntology.detect_actuator_affordances(entities)
        self.assertTrue(len(affordances) >= 1)

        # Switch of color '2' should match Gate of color '2'
        aff = affordances[0]
        self.assertEqual(aff["color"], "2")
        self.assertEqual(aff["target_archetype"], "GATE")

    def test_detect_blueprints(self):
        _, entities = aethel.AethelnetGameOntology.extract_entities(self.frame)
        blueprints = aethel.AethelnetGameOntology.detect_blueprints(self.frame, entities)
        self.assertTrue(len(blueprints) >= 1)

        tr_bp = [b for b in blueprints if b["location"] == "TOP_RIGHT"]
        self.assertEqual(len(tr_bp), 1)
        self.assertIn("1", tr_bp[0]["colors"])
        self.assertIn("3", tr_bp[0]["colors"])

    def test_prompt_header_injection(self):
        vault = aethel.get_vault("test_spec_game", storage_dir="/tmp")
        hdr = vault.format_prompt_header(max_facts=5, current_frame=self.frame)
        self.assertIn("=== INDUCTIVE HYPOTHESES (GAME ONTOLOGY) ===", hdr)
        self.assertIn("[SLOT-CARGO]", hdr)
        self.assertIn("[ACTUATOR]", hdr)
        self.assertIn("[BLUEPRINT]", hdr)


if __name__ == "__main__":
    unittest.main()
