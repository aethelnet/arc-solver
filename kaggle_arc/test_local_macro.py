"""
=============================================================================
LOCAL ARC-AGI-3 MACRO-ACTION TEST RUNNER (AETHELNET v42)
=============================================================================
Runs local ARC-AGI-3 offline games with:
1. High-level Macro-Intents (NAVIGATE_TO, PUSH_BOX, ACTUATE_SWITCH)
2. AethelnetMacroCompiler (A* pathfinding + Sokoban pre-position + Deadlock Veto)
3. Dual Policy Engine:
   - Mode 'ollama': Local Ollama Llama-3 (8B) emits JSON macro-intents (~10-15 calls/game)
   - Mode 'symbolic': Pure Active Inference / Ontological Goal Pursuit (0 calls, sub-second)
4. Dynamic Kinematics & Avatar Calibration from RealActiveInferenceSolver
=============================================================================
"""

import os
import sys
import json
import time
import requests
import collections
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
ARC_DIR = os.path.abspath(os.path.dirname(__file__))
if ARC_DIR not in sys.path:
    sys.path.insert(0, ARC_DIR)

os.environ['OPERATION_MODE'] = 'offline'
from arc_agi import Arcade, OperationMode
from arcengine import GameAction
from kaggle_arc.aethelnet_dsl import (
    AethelnetGameOntology,
    AethelnetMacroCompiler,
    RealActiveInferenceSolver,
    locate_avatar_on_grid,
    MacroIntent,
    MacroChainResult,
    OntologyEntity,
    get_background_color,
    _to_lines
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:latest"


def ask_ollama_for_intent(game_id: str, level: int, entities: List[OntologyEntity], avatar_pos: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    """Query local Ollama for the next high-level macro intent."""
    entity_desc = []
    for e in entities[:10]:
        entity_desc.append(f"Entity_{e.id}: {e.archetype} at {e.center} (color {e.color}, area {e.area})")
    
    prompt = (
        f"You are the high-level navigator for ARC-AGI game {game_id} (Level {level}).\n"
        f"Player avatar is at: {avatar_pos}.\n"
        f"Detected entities on board:\n" + "\n".join(entity_desc) + "\n\n"
        "Choose the next macro intent to solve this level.\n"
        "Allowed intents:\n"
        "- {\"name\": \"NAVIGATE_TO\", \"target\": [row, col]}\n"
        "- {\"name\": \"PUSH_BOX\", \"box_pos\": [row, col], \"direction\": \"UP\"|\"DOWN\"|\"LEFT\"|\"RIGHT\"}\n"
        "- {\"name\": \"ACTUATE_SWITCH\", \"target\": [row, col]}\n\n"
        "Return ONLY the raw JSON object, nothing else."
    )
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 60
        }
    }
    
    try:
        t0 = time.time()
        resp = requests.post(OLLAMA_URL, json=payload, timeout=25.0)
        dt = time.time() - t0
        if resp.status_code == 200:
            raw = resp.json().get("response", "{}")
            data = json.loads(raw)
            print(f"  [Ollama Llama-3] ({dt:.2f}s) -> Intent: {data}")
            return data
    except Exception as exc:
        print(f"  [Ollama Error] {exc} -> Falling back to symbolic policy")
    return None


def symbolic_intent_selector(entities: List[OntologyEntity], avatar_pos: Tuple[int, int], completed_targets: Optional[Set[Tuple[int, int]]] = None) -> Optional[Dict[str, Any]]:
    """Deterministic fallback policy based on nearest ontological goal / collectible."""
    done = completed_targets or set()

    # Priority 1: Collectibles
    collectibles = [e for e in entities if e.archetype == "COLLECTIBLE" and e.center not in done]
    if collectibles:
        collectibles.sort(key=lambda e: abs(e.center[0] - avatar_pos[0]) + abs(e.center[1] - avatar_pos[1]))
        target = collectibles[0].center
        return {"name": "NAVIGATE_TO", "target": list(target)}
        
    # Priority 2: Actuators / Switches
    switches = [e for e in entities if e.archetype == "ACTUATOR" and e.center != avatar_pos and e.center not in done]
    if switches:
        switches.sort(key=lambda e: abs(e.center[0] - avatar_pos[0]) + abs(e.center[1] - avatar_pos[1]))
        target = switches[0].center
        return {"name": "ACTUATE_SWITCH", "target": list(target)}

    # Priority 3: Goals
    goals = [e for e in entities if e.archetype == "GOAL" and e.center != avatar_pos and e.center not in done]
    if goals:
        goals.sort(key=lambda e: abs(e.center[0] - avatar_pos[0]) + abs(e.center[1] - avatar_pos[1]))
        target = goals[0].center
        return {"name": "NAVIGATE_TO", "target": list(target)}

    # Priority 4: Pushables
    pushables = [e for e in entities if e.archetype == "PUSHABLE" and e.center not in done]
    if pushables:
        pushables.sort(key=lambda e: abs(e.center[0] - avatar_pos[0]) + abs(e.center[1] - avatar_pos[1]))
        target = pushables[0].center
        return {"name": "PUSH_BOX", "box_pos": list(target), "direction": "RIGHT"}

    return None


class LocalMacroAgent:
    """Agent that plans with Macro-Action Intents and executes via AethelnetMacroCompiler."""
    def __init__(self, game_id: str, mode: str = "symbolic", stride: int = 1):
        self.game_id = game_id
        self.mode = mode
        self.stride = stride
        self.solver = RealActiveInferenceSolver(game_id=game_id)
        self.action_queue: List[Dict[str, Any]] = []
        self.current_intent_desc = "None"
        self.last_level = 0
        self.macro_call_count = 0
        self.failed_intents = set()
        self.completed_targets = set()
        self.prev_avatar_pos = None
        self.current_target = None
        self.active_target = None
        self.active_intent_name = None
        self.target_step_count = 0

    def choose_action(self, obs, legal_actions) -> Tuple[Any, Dict[str, Any]]:
        grid = np.array(obs.frame)
        while grid.ndim > 2:
            grid = grid[0]

        current_level = obs.levels_completed
        state_name = getattr(obs.state, 'name', str(obs.state))

        if current_level != self.last_level or state_name in ('GAME_OVER', 'NOT_PLAYED'):
            self.action_queue.clear()
            self.failed_intents.clear()
            self.last_level = current_level

        # 1. Let perception solver handle calibration probe
        if not self.solver.calibrated:
            act_int, act_data, reason = self.solver.choose_action(grid, legal_actions, current_level, state_name)
            name_map = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5", 6: "ACTION6"}
            act_name = name_map.get(act_int, "ACTION1")
            return getattr(GameAction, act_name, GameAction.ACTION1), (act_data or {})

        # 2. If queue has pending actions, pop next action
        if self.action_queue:
            next_act = self.action_queue.pop(0)
            act_name = next_act.get("action", "ACTION1")
            act_obj = getattr(GameAction, act_name, GameAction.ACTION1)
            data = {}
            if act_name == "MOUSE":
                data = {"x": next_act.get("col", 32), "y": next_act.get("row", 32)}
            return act_obj, data

        # 3. Locate avatar from calibrated solver
        self.solver.avatar_pos = locate_avatar_on_grid(grid, self.solver.profile, self.solver.hud_rows, self.solver.hud_cols)
        avatar_pos = self.solver.avatar_pos
        if avatar_pos is None:
            # Fallback to solver reactive step
            act_int, act_data, _ = self.solver.choose_action(grid, legal_actions, current_level, state_name)
            name_map = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5", 6: "ACTION6"}
            return getattr(GameAction, name_map.get(act_int, "ACTION1"), GameAction.ACTION1), (act_data or {})

        # 3b. Anti-Stagnation / Wall-Bump Check
        if self.prev_avatar_pos is not None and avatar_pos == self.prev_avatar_pos:
            if self.current_target:
                self.completed_targets.add(self.current_target)
            self.action_queue.clear()
        self.prev_avatar_pos = avatar_pos

        # 4. Resolve solid mask from solver perception
        bg = get_background_color(grid)
        counts = collections.Counter(grid.flatten())
        non_bg = [c for c, _ in counts.most_common() if c != bg]
        wall_color = non_bg[0] if non_bg else bg
        solid_mask = (grid == wall_color)
        if 0 <= avatar_pos[0] < grid.shape[0] and 0 <= avatar_pos[1] < grid.shape[1]:
            solid_mask[avatar_pos[0], avatar_pos[1]] = False

        # 5. Extract entities and select next macro intent
        self.macro_call_count += 1
        _, entities = AethelnetGameOntology.extract_entities(grid)

        stride = self.solver.profile.stride or self.stride

        intent_dict = None
        # Sticky Intent: stay focused on active target to prevent greedy 2-target ping-pong
        if self.active_target is not None:
            if self.active_target not in self.completed_targets and self.target_step_count < 25:
                intent_dict = {"name": self.active_intent_name or "NAVIGATE_TO", "target": list(self.active_target)}
                self.target_step_count += 1
            else:
                self.active_target = None
                self.target_step_count = 0

        if not intent_dict and self.mode == "ollama":
            intent_dict = ask_ollama_for_intent(self.game_id, current_level, entities, avatar_pos)

        if not intent_dict:
            intent_dict = symbolic_intent_selector(entities, avatar_pos, completed_targets=self.completed_targets)
            if intent_dict and intent_dict.get('target'):
                self.active_target = tuple(intent_dict['target'])
                self.active_intent_name = intent_dict.get('name')
                self.target_step_count = 0

        if not intent_dict:
            return GameAction.ACTION1, {}

        self.current_intent_desc = f"{intent_dict.get('name')} {intent_dict.get('target', intent_dict.get('box_pos', ''))}"
        self.current_target = tuple(intent_dict['target']) if intent_dict.get('target') else None

        # Compile intent into atomic actions
        res = AethelnetMacroCompiler.compile_intent(
            intent=intent_dict,
            frame_or_grid=grid,
            solid_mask=solid_mask,
            avatar_pos=avatar_pos,
            stride=stride,
            available_actions=legal_actions
        )

        if res.reason == "AlreadyAtTarget" and intent_dict.get("target"):
            self.completed_targets.add(tuple(intent_dict["target"]))

        if res.success and res.actions:
            print(f"  [Macro-Compiler] Compiled {intent_dict['name']} -> {len(res.actions)} actions ({res.action_names[:5]}...)")
            self.action_queue = list(res.actions)
            first_act = self.action_queue.pop(0)
            act_name = first_act.get("action", "ACTION1")
            act_obj = getattr(GameAction, act_name, GameAction.ACTION1)
            data = {}
            if act_name == "MOUSE":
                data = {"x": first_act.get("col", 32), "y": first_act.get("row", 32)}
            return act_obj, data
        else:
            if not res.success and intent_dict.get("target"):
                self.completed_targets.add(tuple(intent_dict["target"]))
            # Fallback to solver step
            act_int, act_data, _ = self.solver.choose_action(grid, legal_actions, current_level, state_name)
            name_map = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5", 6: "ACTION6"}
            return getattr(GameAction, name_map.get(act_int, "ACTION1"), GameAction.ACTION1), (act_data or {})


def run_local_macro_test(target_game: str, mode: str = "symbolic", max_steps: int = 400):
    env_dir = './kaggle_arc/environment_files'
    subdirs = sorted(os.listdir(os.path.join(env_dir, target_game)))
    v = subdirs[0]
    game_id = f"{target_game}-{v}"

    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
    env = arcade.make(game_id)
    obs = env.observation_space

    # Check stride from metadata or known games
    stride = 4 if target_game == 'wa30' else 1

    agent = LocalMacroAgent(game_id=game_id, mode=mode, stride=stride)

    print(f"\n==================================================================")
    print(f" STARTING LOCAL TEST: {game_id} | Mode: {mode.upper()} | Stride: {stride}")
    print(f"==================================================================")

    t0 = time.time()
    step = 0
    while step < max_steps:
        step += 1
        available = getattr(obs, 'available_actions', None) or [1, 2, 3, 4]
        legal = [getattr(a, 'value', a) for a in available if getattr(a, 'value', a) != 0]

        act, data = agent.choose_action(obs, legal)
        obs = env.step(act, data=data)

        if step % 20 == 0 or obs.state.name in ('WIN', 'GAME_OVER'):
            print(f"Step {step:3d} | Level: {obs.levels_completed}/{obs.win_levels} | Solved: {obs.levels_completed} | State: {obs.state.name} | MacroCalls: {agent.macro_call_count}")

        if obs.state.name in ('WIN', 'GAME_OVER'):
            break

    elapsed = time.time() - t0
    print(f"------------------------------------------------------------------")
    print(f" RESULT {target_game}: Solved {obs.levels_completed}/{obs.win_levels} in {step} steps ({elapsed:.2f}s)")
    print(f" Total Macro-Intents: {agent.macro_call_count} | End State: {obs.state.name}")
    print(f"==================================================================\n")
    return obs.levels_completed


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'wa30'
    run_mode = sys.argv[2] if len(sys.argv) > 2 else 'symbolic'
    run_local_macro_test(target, mode=run_mode)
