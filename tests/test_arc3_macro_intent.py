import pytest
import numpy as np
from kaggle_arc.aethelnet_dsl import (
    AethelnetMacroCompiler,
    MacroIntent,
    MacroChainResult
)

def test_macro_intent_navigate_corridor():
    # 10x10 grid with background '.', walls 'X', avatar at (5, 2), target at (5, 6)
    lines = [
        "..........",
        "..........",
        "..........",
        "XXXXXXXXXX",
        "X..@...G.X", # r=4: avatar @ at (4,3), goal G at (4,7)
        "XXXXXXXXXX",
        "..........",
        "..........",
        "..........",
        ".........."
    ]
    # Solid mask: row 3 and 5 are walls
    solid_mask = np.zeros((10, 10), dtype=bool)
    solid_mask[3, :] = True
    solid_mask[5, :] = True
    solid_mask[4, 0] = True
    solid_mask[4, 9] = True

    intent = MacroIntent(name="NAVIGATE_TO", target=(4, 7))
    res = AethelnetMacroCompiler.compile_intent(
        intent,
        lines,
        solid_mask=solid_mask,
        avatar_pos=(4, 3)
    )

    assert res.success is True
    assert res.step_count == 4
    assert res.action_names == ["RIGHT", "RIGHT", "RIGHT", "RIGHT"]
    assert len(res.actions) == 4
    assert all(a["action"] == "ACTION4" for a in res.actions)

def test_macro_intent_push_box_safe():
    # 10x10 open room, avatar at (5, 3), box at (5, 5), push RIGHT
    # Expected: avatar routes to push_pose (5, 4) behind box, then executes RIGHT push
    grid = np.full((10, 10), '.', dtype=str)
    solid_mask = np.zeros((10, 10), dtype=bool)

    intent = MacroIntent(name="PUSH_BOX", box_pos=(5, 5), direction="RIGHT")
    res = AethelnetMacroCompiler.compile_intent(
        intent,
        grid,
        solid_mask=solid_mask,
        avatar_pos=(5, 3)
    )

    assert res.success is True
    # Step from (5,3) to (5,4) is RIGHT (1 move), then push is RIGHT (1 move)
    assert res.step_count == 2
    assert res.action_names == ["RIGHT", "RIGHT"]
    assert res.actions[-1]["action"] == "ACTION4"

def test_macro_intent_push_box_deadlock_veto():
    # 10x10 room, box at (1, 1), walls at (0, :) and (:, 0) -> top-left corner
    # Pushing LEFT into (1, 0) is wall, pushing UP into (0, 1) is wall
    # Pushing into (1, 1) from (2, 1) UP is moving into a corner deadlock!
    grid = np.full((10, 10), '.', dtype=str)
    solid_mask = np.zeros((10, 10), dtype=bool)
    solid_mask[0, :] = True
    solid_mask[:, 0] = True

    # Pushing box at (2, 1) UP into (1, 1) creates 90-degree wall deadlock with top wall and left wall
    intent = MacroIntent(name="PUSH_BOX", box_pos=(2, 1), direction="UP")
    res = AethelnetMacroCompiler.compile_intent(
        intent,
        grid,
        solid_mask=solid_mask,
        avatar_pos=(3, 1)
    )

    assert res.success is False
    assert "DeadlockVeto" in res.reason
    assert res.step_count == 0

def test_macro_intent_mouse_click():
    grid = np.full((10, 10), '.', dtype=str)
    intent = MacroIntent(name="INTERACT", target=(3, 7))
    res = AethelnetMacroCompiler.compile_intent(intent, grid)

    assert res.success is True
    assert res.step_count == 1
    assert res.actions == [{"action": "MOUSE", "row": 3, "col": 7}]
    assert res.action_names == ["MOUSE"]

def test_macro_intent_with_stride():
    # 20x20 grid with avatar at (4, 4), target at (12, 4), stride=4
    # Expected: 2 steps DOWN (stride 4 each) -> 2 actions DOWN
    grid = np.zeros((20, 20), dtype=int)
    solid_mask = np.zeros((20, 20), dtype=bool)
    intent = MacroIntent(name="NAVIGATE_TO", target=(12, 4))
    res = AethelnetMacroCompiler.compile_intent(
        intent,
        grid,
        solid_mask=solid_mask,
        avatar_pos=(4, 4),
        stride=4
    )
    assert res.success is True
    assert res.step_count == 2
    assert res.action_names == ["DOWN", "DOWN"]
    assert all(a["action"] == "ACTION2" for a in res.actions)

def test_macro_intent_actuate_switch_with_action5():
    # Avatar at (5, 4), switch at (5, 5), available_actions includes ACTION5 (5)
    grid = np.zeros((10, 10), dtype=int)
    solid_mask = np.zeros((10, 10), dtype=bool)
    intent = MacroIntent(name="ACTUATE_SWITCH", target=(5, 5))
    res = AethelnetMacroCompiler.compile_intent(
        intent,
        grid,
        solid_mask=solid_mask,
        avatar_pos=(5, 4),
        available_actions=[1, 2, 3, 4, 5]
    )
    assert res.success is True
    assert res.actions[-1] == {"action": "ACTION5"}
    assert res.action_names[-1] == "ACTION5"

def test_to_lines_3d_array():
    from kaggle_arc.aethelnet_dsl import _to_lines
    f3d = np.zeros((1, 8, 8), dtype=int)
    f3d[0, 2, 3] = 4
    lines = _to_lines(f3d)
    assert len(lines) == 8
    assert len(lines[0]) == 8
    assert lines[2][3] == '4'

