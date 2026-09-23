"""
=============================================================================
ARC-AGI CAUSAL META-TOOL SELECTOR: LOCAL CELLULAR PHYSICS INDUCER
=============================================================================
Induces local transition dynamics and cellular automata laws online from
frame transitions (s_t, a_t -> s_{t+1}) in arbitrary unclassified ARC games:

1. Permeability Induction: Identifies walkable floor colors vs. impassable solid walls.
2. Displacement / Push Induction: Identifies pushable entity colors, mass/extent,
   and legal support surfaces (Sokoban mechanics, sliding blocks, ice curling).
3. Remote / Reactive Triggers: Identifies switch-gate relationships (Interact at A
   causes barrier removal or door transformation at B).
4. Point-and-Click Rewrites: Identifies stencil transformations (Galois GF(2)
   involutory toggles, cross-patterns, flood-fills).
5. Hazard Identification: Marks lethal colors and death zones causing GAME_OVER.
6. In-Memory Counterfactual Rollout: Accurately simulates the induced laws in
   MentalSandbox without hardcoded domain knowledge.
=============================================================================
"""

import collections
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np


class PhysicsRuleType(Enum):
    PERMEABILITY = "PERMEABILITY"      # Walkable vs. Solid
    DISPLACEMENT = "DISPLACEMENT"      # Pushable block / Matter shift
    REMOTE_TRIGGER = "REMOTE_TRIGGER"  # Button / Switch / Gate unlock
    CLICK_REWRITE = "CLICK_REWRITE"    # Involutory toggle / Click stencil
    HAZARD = "HAZARD"                  # Lethal death tile / enemy collision


@dataclass
class DisplacementRule:
    """Induced push / matter displacement rule for a specific entity color."""
    color: int
    pushable_directions: Set[int] = field(default_factory=set) # Legal action IDs (1, 2, 3, 4)
    passable_destinations: Set[int] = field(default_factory=set) # Colors a block can be pushed onto
    solid_barriers: Set[int] = field(default_factory=set) # Colors that block pushing
    observed_pushes: int = 0


@dataclass
class RemoteTriggerRule:
    """Induced causal dependency between an actuator and a distant modification."""
    trigger_pos: Tuple[int, int]
    trigger_action: int # 5 or 6
    affected_cells: List[Tuple[int, int, int, int]] # [(r, c, old_val, new_val), ...]
    confidence: float = 1.0


@dataclass
class GravityActuatorRule:
    """Induced causal dependency between an actuator action and gravity inversion."""
    actuator_pos: Tuple[int, int]
    trigger_action: int # 5 (interact), 6 (click), or 0 (step-on)
    from_gravity: Tuple[int, int]
    to_gravity: Tuple[int, int]
    confidence: float = 1.0


class LocalPhysicsInducer:
    """
    Online Scientist Engine:
    Learns the cellular transition function T(Neighborhood, a) -> Neighborhood'
    purely from online environmental observations.
    """
    def __init__(self):
        self.walkable_colors: Set[int] = set()
        self.solid_colors: Set[int] = set()
        self.hazard_colors: Set[int] = set()
        self.suspect_hazard_colors: Set[int] = set()
        self.swatch_colors: Set[int] = set()
        self.displacement_rules: Dict[int, DisplacementRule] = {} # color -> DisplacementRule
        self.remote_triggers: List[RemoteTriggerRule] = []
        self.gravity_actuators: List[GravityActuatorRule] = []
        self.last_actuator_event: Optional[Tuple[Tuple[int, int], int]] = None # (pos, action_id)
        self.click_rewrites: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
        self.avatar_stride: int = 1
        self.total_observations: int = 0
        self.gravity_vector: Optional[Tuple[int, int]] = None
        self.jump_height: int = 2

    def reset(self):
        """Resets inducer state across full game evaluations."""
        self.walkable_colors.clear()
        self.solid_colors.clear()
        self.hazard_colors.clear()
        self.suspect_hazard_colors.clear()
        self.swatch_colors.clear()
        self.displacement_rules.clear()
        self.remote_triggers.clear()
        self.gravity_actuators.clear()
        self.last_actuator_event = None
        self.click_rewrites.clear()
        self.avatar_stride = 1
        self.total_observations = 0
        self.gravity_vector = None
        self.jump_height = 2

    def observe_transition(
        self,
        prev_grid: Optional[np.ndarray],
        action_id: int,
        action_data: Optional[Dict[str, Any]],
        current_grid: np.ndarray,
        prev_avatar_pos: Optional[Tuple[int, int]] = None,
        current_avatar_pos: Optional[Tuple[int, int]] = None,
        game_over: bool = False,
        level_advanced: bool = False
    ):
        """
        Observes a single state transition (s_t, a_t -> s_{t+1}) and updates
        the cellular physics rule library.
        """
        if prev_grid is None:
            return

        p_grid = np.asarray(prev_grid)
        c_grid = np.asarray(current_grid)
        while p_grid.ndim > 2:
            p_grid = p_grid[-1]
        while c_grid.ndim > 2:
            c_grid = c_grid[-1]

        self.total_observations += 1
        H, W = p_grid.shape

        # Diff analysis
        diff_mask = (p_grid != c_grid)
        diff_count = int(np.sum(diff_mask))
        diff_coords = np.argwhere(diff_mask)

        # 1. Hazard Induction (Death tile)
        if game_over and prev_avatar_pos is not None:
            if action_id in (1, 2, 3, 4):
                dr, dc = self._action_to_delta(action_id)
                tr, tc = prev_avatar_pos[0] + dr, prev_avatar_pos[1] + dc
                if 0 <= tr < H and 0 <= tc < W:
                    death_val = int(p_grid[tr, tc])
                    self.hazard_colors.add(death_val)

        # 2. Navigation & Permeability Induction (Actions 1..4)
        if action_id in (1, 2, 3, 4) and prev_avatar_pos is not None:
            dr, dc = self._action_to_delta(action_id)
            target_r = prev_avatar_pos[0] + dr
            target_c = prev_avatar_pos[1] + dc

            # Did the avatar move?
            if current_avatar_pos is not None:
                avatar_moved = (current_avatar_pos != prev_avatar_pos)
                actual_dr = current_avatar_pos[0] - prev_avatar_pos[0]
                actual_dc = current_avatar_pos[1] - prev_avatar_pos[1]
                measured_stride = max(abs(actual_dr), abs(actual_dc))
                if measured_stride > 0:
                    self.avatar_stride = measured_stride

                # Directional Gravity Induction:
                # If a lateral action (3=LEFT, 4=RIGHT) or non-nav action produces downward vertical displacement
                if actual_dr > 0 and action_id in (3, 4, 5, 6):
                    self.gravity_vector = (1, 0)
                elif actual_dr < 0 and action_id in (3, 4, 5, 6):
                    self.gravity_vector = (-1, 0)
            else:
                avatar_moved = (diff_count > 0)

            if not avatar_moved and 0 <= target_r < H and 0 <= target_c < W:
                # Avatar collided with something solid
                solid_val = int(p_grid[target_r, target_c])
                self.solid_colors.add(solid_val)

            # 3. Push / Displacement Induction:
            # If avatar moved AND cells beyond the avatar also changed
            is_push = False
            if avatar_moved and diff_count >= 3 and prev_avatar_pos is not None:
                stride = self.avatar_stride
                s_dr = dr * stride
                s_dc = dc * stride
                push_source_r = prev_avatar_pos[0] + s_dr
                push_source_c = prev_avatar_pos[1] + s_dc
                push_dest_r = push_source_r + s_dr
                push_dest_c = push_source_c + s_dc

                if 0 <= push_source_r < H and 0 <= push_source_c < W:
                    pushed_color = int(p_grid[push_source_r, push_source_c])
                    if 0 <= push_dest_r < H and 0 <= push_dest_c < W:
                        dest_prev_color = int(p_grid[push_dest_r, push_dest_c])
                        dest_curr_color = int(c_grid[push_dest_r, push_dest_c])

                        # Did the block appear at the destination?
                        if dest_curr_color == pushed_color:
                            is_push = True
                            if pushed_color not in self.displacement_rules:
                                self.displacement_rules[pushed_color] = DisplacementRule(color=pushed_color)
                            rule = self.displacement_rules[pushed_color]
                            rule.pushable_directions.add(action_id)
                            rule.passable_destinations.add(dest_prev_color)
                            rule.observed_pushes += 1
                            # The floor color is the destination cell's previous color or avatar origin color
                            self.walkable_colors.add(dest_prev_color)

            if avatar_moved and not is_push and current_avatar_pos is not None:
                # Target cell is walkable floor
                dest_val = int(p_grid[current_avatar_pos[0], current_avatar_pos[1]])
                self.walkable_colors.add(dest_val)
                curr_at_val = int(c_grid[current_avatar_pos[0], current_avatar_pos[1]])
                if dest_val != 0 and curr_at_val == dest_val:
                    self.swatch_colors.add(dest_val)

        # 4. Point-and-Click / Actuator Rewrites (Action 6)
        if action_id == 6 and action_data and 'x' in action_data and 'y' in action_data:
            click_pt = (int(action_data['x']), int(action_data['y']))
            if 0 < len(diff_coords) <= 256:
                diffs = [(int(r), int(c), int(p_grid[r, c]), int(c_grid[r, c])) for r, c in diff_coords]
                self.click_rewrites[click_pt] = diffs

        # 5. Remote Trigger / Switch Gate Induction (Action 5 or Action 6)
        if action_id in (5, 6) and prev_avatar_pos is not None:
            # Check if mutations occurred away from the avatar / click position
            remote_diffs = []
            for r, c in diff_coords:
                dist_to_avatar = abs(r - prev_avatar_pos[0]) + abs(c - prev_avatar_pos[1])
                if dist_to_avatar > self.avatar_stride * 2:
                    remote_diffs.append((int(r), int(c), int(p_grid[r, c]), int(c_grid[r, c])))

            if remote_diffs:
                trigger_pos = prev_avatar_pos if action_id == 5 else (
                    (int(action_data.get('y', prev_avatar_pos[0])), int(action_data.get('x', prev_avatar_pos[1])))
                    if action_data else prev_avatar_pos
                )
                self.remote_triggers.append(RemoteTriggerRule(
                    trigger_pos=trigger_pos,
                    trigger_action=action_id,
                    affected_cells=remote_diffs
                ))

        # 6. Directional & Actuator-Driven Gravity Induction
        if prev_avatar_pos is not None and current_avatar_pos is not None:
            actual_dr = current_avatar_pos[0] - prev_avatar_pos[0]
            observed_g = None
            if actual_dr > 0 and action_id in (3, 4, 5, 6):
                observed_g = (1, 0)
            elif actual_dr < 0 and action_id in (3, 4, 5, 6):
                observed_g = (-1, 0)

            if observed_g is not None:
                if self.gravity_vector is not None and observed_g != self.gravity_vector:
                    # Inversion detected!
                    act_pos = prev_avatar_pos
                    act_id = action_id
                    if action_id not in (5, 6) and self.last_actuator_event is not None:
                        act_pos, act_id = self.last_actuator_event

                    rule = GravityActuatorRule(
                        actuator_pos=act_pos,
                        trigger_action=act_id if act_id in (5, 6) else 0,
                        from_gravity=self.gravity_vector,
                        to_gravity=observed_g
                    )
                    if not any(r.actuator_pos == rule.actuator_pos and r.to_gravity == rule.to_gravity for r in self.gravity_actuators):
                        self.gravity_actuators.append(rule)

                self.gravity_vector = observed_g

        if action_id in (5, 6) and prev_avatar_pos is not None:
            self.last_actuator_event = (prev_avatar_pos, action_id)
        else:
            self.last_actuator_event = None

    def register_gravity_actuator(
        self,
        pos: Tuple[int, int],
        trigger_action: int = 5,
        from_gravity: Tuple[int, int] = (1, 0),
        to_gravity: Tuple[int, int] = (-1, 0),
        confidence: float = 1.0
    ):
        """Explicitly registers an induced gravity actuator rule."""
        rule = GravityActuatorRule(
            actuator_pos=pos,
            trigger_action=trigger_action,
            from_gravity=from_gravity,
            to_gravity=to_gravity,
            confidence=confidence
        )
        self.gravity_actuators.append(rule)

    def get_gravity_actuators(self) -> List[Any]:
        """Returns induced gravity actuators formatted for platformer_a_star."""
        from .platformer_planner import GravityActuator
        actuators = []
        for rule in self.gravity_actuators:
            actuators.append(GravityActuator(
                pos=rule.actuator_pos,
                action_id=rule.trigger_action,
                target_gravity=rule.to_gravity,
                requires_adjacent=(rule.trigger_action != 6)
            ))
        return actuators

    def is_color_pushable(self, color: int) -> bool:
        """Returns True if the specified color is an induced pushable block."""
        return color in self.displacement_rules and self.displacement_rules[color].observed_pushes > 0

    def get_suspect_hazard_colors(
        self,
        grid: np.ndarray,
        bg_canvas: int = 0,
        avatar_pos: Optional[Tuple[int, int]] = None,
        verified_floors: Optional[Set[int]] = None
    ) -> Set[int]:
        """
        Pessimistic Safe Active Inference:
        Identifies unverified non-background colors that have not been proven safe,
        treating them as presumptively hazardous until positive evidence of safety is observed.
        """
        grid_2d = np.asarray(grid)
        while grid_2d.ndim > 2:
            grid_2d = grid_2d[-1]

        counts = collections.Counter(grid_2d.flatten())
        suspects = set()

        if verified_floors:
            self.walkable_colors.update(verified_floors)

        # Any cell the avatar is currently occupying without dying is verified safe
        if avatar_pos is not None and 0 <= avatar_pos[0] < grid_2d.shape[0] and 0 <= avatar_pos[1] < grid_2d.shape[1]:
            curr_avatar_color = int(grid_2d[avatar_pos[0], avatar_pos[1]])
            self.walkable_colors.add(curr_avatar_color)

        for col in counts.keys():
            c_int = int(col)
            if c_int == bg_canvas:
                continue
            if c_int in self.walkable_colors:
                continue
            if c_int in self.solid_colors:
                continue
            if c_int in self.swatch_colors:
                continue
            if c_int in self.displacement_rules:
                continue
            suspects.add(c_int)

        self.suspect_hazard_colors = set(suspects)
        return suspects

    def is_color_verified_safe(self, color: int, bg_canvas: int = 0) -> bool:
        """Returns True if the color has been positively confirmed as safe floor or swatch."""
        return color == bg_canvas or color in self.walkable_colors or color in self.swatch_colors

    def is_cell_passable(self, color: int, allow_unverified: bool = False) -> bool:
        """
        Returns True if the avatar can step onto this color without pushing.
        Enforces Safe-Set Inversion: unverified colors are rejected unless allow_unverified is True.
        """
        if color in self.solid_colors:
            return False
        if color in self.hazard_colors:
            return False
        if color in self.displacement_rules:
            return False
        if not allow_unverified and self.walkable_colors:
            # If we have verified walkable colors, reject unverified suspect colors
            if color not in self.walkable_colors and color not in self.swatch_colors and color != 0:
                return False
        return True

    def simulate_forward_step(
        self,
        grid: np.ndarray,
        avatar_pos: Tuple[int, int],
        action_id: int,
        action_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Tuple[int, int], bool, str]:
        """
        Virtual Forward Simulator:
        Simulates the physical result of applying action_id on grid from avatar_pos
        using purely induced local cellular laws.
        Returns:
            (next_grid, next_avatar_pos, success: bool, reason: str)
        """
        sim_grid = grid.copy()
        while sim_grid.ndim > 2:
            sim_grid = sim_grid[-1]
        H, W = sim_grid.shape

        # 1. Action 6 (Click Rewrite simulation)
        if action_id == 6:
            if action_data and 'x' in action_data and 'y' in action_data:
                click_pt = (int(action_data['x']), int(action_data['y']))
                if click_pt in self.click_rewrites:
                    for r, c, old_v, new_v in self.click_rewrites[click_pt]:
                        if 0 <= r < H and 0 <= c < W:
                            sim_grid[r, c] = new_v
                    return sim_grid, avatar_pos, True, "Click rewrite applied"
            return sim_grid, avatar_pos, True, "No effect click"

        # 2. Action 5 (Remote Trigger & Gravity Actuator simulation)
        if action_id == 5:
            if self.gravity_actuators:
                for rule in self.gravity_actuators:
                    if rule.trigger_action == 5:
                        dist = abs(rule.actuator_pos[0] - avatar_pos[0]) + abs(rule.actuator_pos[1] - avatar_pos[1])
                        if dist <= self.avatar_stride:
                            from .platformer_planner import simulate_gravity_fall, is_supported
                            nxt_g = rule.to_gravity
                            self.gravity_vector = nxt_g
                            if is_supported(sim_grid, avatar_pos, nxt_g, self.solid_colors, self.avatar_stride):
                                nxt_p = avatar_pos
                            else:
                                nxt_p, _, _ = simulate_gravity_fall(
                                    sim_grid, avatar_pos, nxt_g, self.solid_colors, self.hazard_colors, stride=self.avatar_stride
                                )
                            return sim_grid, nxt_p, True, f"Gravity inverted to {nxt_g}"

            for rule in self.remote_triggers:
                if rule.trigger_action == 5:
                    dist = abs(rule.trigger_pos[0] - avatar_pos[0]) + abs(rule.trigger_pos[1] - avatar_pos[1])
                    if dist <= self.avatar_stride * 2:
                        for r, c, old_v, new_v in rule.affected_cells:
                            if 0 <= r < H and 0 <= c < W:
                                sim_grid[r, c] = new_v
                        return sim_grid, avatar_pos, True, "Remote switch activated"
            return sim_grid, avatar_pos, True, "No-op interaction"

        # 3. Actions 1..4 (Navigation & Push simulation)
        if action_id not in (1, 2, 3, 4):
            return sim_grid, avatar_pos, False, f"Unknown action {action_id}"

        dr, dc = self._action_to_delta(action_id)
        stride = self.avatar_stride
        nr = avatar_pos[0] + dr * stride
        nc = avatar_pos[1] + dc * stride

        # Out of bounds
        if not (0 <= nr < H and 0 <= nc < W):
            return sim_grid, avatar_pos, False, "Motion out of bounds"

        target_val = int(sim_grid[nr, nc])

        # Solid barrier collision
        if target_val in self.solid_colors:
            return sim_grid, avatar_pos, False, f"Motion blocked by solid color {target_val}"

        # Hazard collision
        if target_val in self.hazard_colors:
            return sim_grid, (nr, nc), False, f"Collision with lethal hazard color {target_val}"

        avatar_color = int(sim_grid[avatar_pos[0], avatar_pos[1]])
        floor_color = list(self.walkable_colors)[0] if self.walkable_colors else 0

        # Displacement / Push Dynamics
        if self.is_color_pushable(target_val):
            dest_r = nr + dr * stride
            dest_c = nc + dc * stride

            if not (0 <= dest_r < H and 0 <= dest_c < W):
                return sim_grid, avatar_pos, False, "Block push out of bounds"

            dest_val = int(sim_grid[dest_r, dest_c])

            # Check if destination cell accepts the block
            if dest_val in self.solid_colors or self.is_color_pushable(dest_val):
                return sim_grid, avatar_pos, False, f"Block push blocked by {dest_val}"

            # Shift block to destination
            sim_grid[dest_r, dest_c] = target_val
            # Avatar steps into previous block position
            sim_grid[nr, nc] = avatar_color
            # Avatar former position becomes floor
            sim_grid[avatar_pos[0], avatar_pos[1]] = floor_color

            return sim_grid, (nr, nc), True, f"Pushed block {target_val} to ({dest_r}, {dest_c})"

        # Simple navigation step
        sim_grid[nr, nc] = avatar_color
        sim_grid[avatar_pos[0], avatar_pos[1]] = floor_color
        return sim_grid, (nr, nc), True, "Navigated to empty cell"

    def _action_to_delta(self, action_id: int) -> Tuple[int, int]:
        """Maps ARC action ID to (dr, dc) delta."""
        if action_id == 1:
            return (-1, 0) # UP
        elif action_id == 2:
            return (1, 0)  # DOWN
        elif action_id == 3:
            return (0, -1) # LEFT
        elif action_id == 4:
            return (0, 1)  # RIGHT
        return (0, 0)
