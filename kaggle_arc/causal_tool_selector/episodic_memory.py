"""
=============================================================================
EPISODIC CROSS-LEVEL CAUSAL MEMORY BANK & INVARIANT REPLAY
=============================================================================
Maintains, consolidates, and verifies causal invariants across level transitions
(k -> k+1) within an ARC-AGI puzzle game.

Warm-starts subsequent levels by retaining:
1. Spatial & Kinematic Invariants:
   - Walkable / floor colors
   - Solid wall / barrier colors
   - Hazard / lethal colors
   - Avatar color, dimensions, and motion stride
   - Directional gravity vector g and ballistic jump parameters
2. Dynamic & Mechanics Invariants:
   - Pushable block colors and pushability constraints
   - Swatch / paint dip interactive colors
   - Click / toggle transformation stencils
   - Learned patch rewrite rules
3. Grammar & Objective Invariants:
   - Winning hypothesis types (e.g. RECEPTOR_MATCHING, COLLECT_ITEMS, GOAL_TILE)
   - Winning options and successful subgoal patterns
4. Epistemic Bayesian Verification & Invalidation:
   - Continuously tests cached invariants against new observations.
   - If an invariant is violated (e.g. presumed floor is impassable or solid color changed),
     triggers adaptive epistemic invalidation without catastrophic failure.
=============================================================================
"""

from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import collections
import numpy as np


@dataclass
class PhysicalInvariants:
    """Consolidated physical and cellular mechanics invariants across levels."""
    walkable_colors: Set[int] = field(default_factory=set)
    solid_colors: Set[int] = field(default_factory=set)
    hazard_colors: Set[int] = field(default_factory=set)
    swatch_colors: Set[int] = field(default_factory=set)
    pushable_colors: Set[int] = field(default_factory=set)
    avatar_color: Optional[int] = None
    avatar_stride: int = 1
    gravity_vector: Optional[Tuple[int, int]] = None
    jump_height: int = 2
    gravity_actuators: List[Any] = field(default_factory=list)
    click_rewrites: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = field(default_factory=dict)
    displacement_rules: Dict[int, Any] = field(default_factory=dict)
    color_confidences: Dict[int, float] = field(default_factory=dict)


@dataclass
class GrammarInvariants:
    """Consolidated game grammar and objective invariants across levels."""
    winning_hypotheses: List[Any] = field(default_factory=list)
    winning_options: List[str] = field(default_factory=list)
    preferred_hypothesis_type: Optional[Any] = None
    level_solve_steps: List[int] = field(default_factory=list)


class EpisodicMemoryBank:
    """
    Episodic Cross-Level Memory & Invariant Replay Engine.
    Preserves inductive discoveries across level transitions within the same game.
    """
    def __init__(self):
        self.physical = PhysicalInvariants()
        self.grammar = GrammarInvariants()
        self.total_levels_completed: int = 0
        self.violation_log: List[Dict[str, Any]] = []

    def reset(self):
        """Resets all episodic memory across different games."""
        self.physical = PhysicalInvariants()
        self.grammar = GrammarInvariants()
        self.total_levels_completed = 0
        self.violation_log.clear()

    def record_level_completion(
        self,
        level_idx: int,
        explorer: Optional[Any] = None,
        physics_inducer: Optional[Any] = None,
        hypothesis_inducer: Optional[Any] = None,
        sandbox: Optional[Any] = None,
        active_option_name: Optional[str] = None,
        steps_taken: int = 0
    ):
        """
        Consolidates causal invariants upon advancing from level_idx -> level_idx + 1.
        """
        self.total_levels_completed = max(self.total_levels_completed, level_idx + 1)
        self.grammar.level_solve_steps.append(steps_taken)

        if active_option_name:
            self.grammar.winning_options.append(active_option_name)

        # 1. Consolidate Kinematic & Spatial Invariants from Explorer
        if explorer is not None:
            if getattr(explorer, 'avatar_color', None) is not None:
                self.physical.avatar_color = explorer.avatar_color
            if getattr(explorer, 'stride', 1) > 1:
                self.physical.avatar_stride = explorer.stride
            if getattr(explorer, 'floor_colors', None):
                for c in explorer.floor_colors:
                    self.physical.walkable_colors.add(c)
                    self.physical.color_confidences[c] = self.physical.color_confidences.get(c, 0.5) + 0.3

        # 2. Consolidate Cellular Physics Invariants
        if physics_inducer is not None:
            if getattr(physics_inducer, 'walkable_colors', None):
                for c in physics_inducer.walkable_colors:
                    self.physical.walkable_colors.add(c)
                    self.physical.color_confidences[c] = self.physical.color_confidences.get(c, 0.5) + 0.3
            if getattr(physics_inducer, 'solid_colors', None):
                for c in physics_inducer.solid_colors:
                    self.physical.solid_colors.add(c)
            if getattr(physics_inducer, 'hazard_colors', None):
                for c in physics_inducer.hazard_colors:
                    self.physical.hazard_colors.add(c)
            if getattr(physics_inducer, 'swatch_colors', None):
                for c in physics_inducer.swatch_colors:
                    self.physical.swatch_colors.add(c)
            if getattr(physics_inducer, 'gravity_vector', None) is not None:
                self.physical.gravity_vector = physics_inducer.gravity_vector
            if getattr(physics_inducer, 'jump_height', 2) != 2:
                self.physical.jump_height = physics_inducer.jump_height
            if getattr(physics_inducer, 'gravity_actuators', None):
                self.physical.gravity_actuators = list(physics_inducer.gravity_actuators)
            if getattr(physics_inducer, 'displacement_rules', None):
                for col, rule in physics_inducer.displacement_rules.items():
                    self.physical.displacement_rules[col] = rule
                    self.physical.pushable_colors.add(col)

        # 3. Consolidate Patch & Click Rules from Sandbox
        if sandbox is not None:
            if getattr(sandbox, 'learned_click_rules', None):
                self.physical.click_rewrites.update(sandbox.learned_click_rules)

        # 4. Consolidate Objective Hypothesis Grammar
        if hypothesis_inducer is not None:
            curr_hypo = getattr(hypothesis_inducer, 'current_hypothesis', None)
            if curr_hypo is not None:
                hypo_t = getattr(curr_hypo, 'hypo_type', getattr(curr_hypo, 'hypothesis_type', None))
                if hypo_t is not None:
                    self.grammar.winning_hypotheses.append(hypo_t)
                    self.grammar.preferred_hypothesis_type = hypo_t

    def warm_start_level(
        self,
        level_idx: int,
        explorer: Optional[Any] = None,
        physics_inducer: Optional[Any] = None,
        hypothesis_inducer: Optional[Any] = None,
        sandbox: Optional[Any] = None
    ):
        """
        Transfers learned cross-level invariants into the agent's perception and planning engines
        for the new level, while clearing transient coordinate-specific states.
        """
        # 1. Warm-start Explorer
        if explorer is not None:
            # Clear level-specific coordinate cache
            if hasattr(explorer, 'visited_cells'):
                explorer.visited_cells.clear()
            if hasattr(explorer, 'obstacle_cells'):
                explorer.obstacle_cells.clear()
            if hasattr(explorer, 'traversable_cells'):
                explorer.traversable_cells.clear()
            if hasattr(explorer, 'recent_positions'):
                explorer.recent_positions.clear()
            if hasattr(explorer, 'interacted_targets'):
                explorer.interacted_targets.clear()
            if hasattr(explorer, 'active_path'):
                explorer.active_path.clear()
            explorer.consecutive_stuck = 0
            explorer.avatar_pos = None

            # Inject cached physical invariants
            if self.physical.avatar_color is not None:
                explorer.avatar_color = self.physical.avatar_color
                explorer.avatar_confidence = 0.95
            if self.physical.avatar_stride > 1:
                explorer.stride = self.physical.avatar_stride
            if self.physical.walkable_colors:
                explorer.floor_colors = set(self.physical.walkable_colors)

        # 2. Warm-start Local Physics Inducer
        if physics_inducer is not None:
            if self.physical.walkable_colors:
                physics_inducer.walkable_colors.update(self.physical.walkable_colors)
            if self.physical.solid_colors:
                physics_inducer.solid_colors.update(self.physical.solid_colors)
            if self.physical.hazard_colors:
                physics_inducer.hazard_colors.update(self.physical.hazard_colors)
            if self.physical.swatch_colors:
                physics_inducer.swatch_colors.update(self.physical.swatch_colors)
            if self.physical.gravity_vector is not None:
                physics_inducer.gravity_vector = self.physical.gravity_vector
                physics_inducer.jump_height = self.physical.jump_height
            if self.physical.gravity_actuators:
                physics_inducer.gravity_actuators = list(self.physical.gravity_actuators)
            if self.physical.displacement_rules:
                physics_inducer.displacement_rules.update(self.physical.displacement_rules)
            if self.physical.avatar_stride > 1:
                physics_inducer.avatar_stride = self.physical.avatar_stride

        # 3. Warm-start Mental Sandbox
        if sandbox is not None:
            if self.physical.click_rewrites:
                sandbox.learned_click_rules.update(self.physical.click_rewrites)
            sandbox.state_hashes.clear()
            sandbox.action_history.clear()
            sandbox.wall_bump_count = 0
            sandbox.consecutive_intentional_waits = 0
            sandbox.last_grid_hash = None

        # 4. Warm-start Hypothesis Inducer Grammar Priors
        if hypothesis_inducer is not None and self.grammar.preferred_hypothesis_type is not None:
            # Boost prior of previously successful hypothesis type
            pref = self.grammar.preferred_hypothesis_type
            if hasattr(hypothesis_inducer, 'hypo_priors') and pref in hypothesis_inducer.hypo_priors:
                # Re-normalize with preference boost
                for k in hypothesis_inducer.hypo_priors:
                    hypothesis_inducer.hypo_priors[k] = 0.10
                hypothesis_inducer.hypo_priors[pref] = 0.60
            hypothesis_inducer.current_hypothesis = None
            hypothesis_inducer.prev_potential = None
            hypothesis_inducer.subgoal_counter = 0

    def validate_transition(
        self,
        prev_grid: Optional[np.ndarray],
        action_id: int,
        current_grid: np.ndarray,
        prev_avatar_pos: Optional[Tuple[int, int]],
        current_avatar_pos: Optional[Tuple[int, int]],
        game_over: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Epistemic Invariant Verification:
        Compares observed transition against cached invariants.
        If a contradiction is observed (e.g. wall bump on assumed floor),
        adjusts confidence or removes the faulty invariant.
        """
        if prev_grid is None or prev_avatar_pos is None or current_avatar_pos is None:
            return True, None

        # Check if directional movement failed unexpectedly into an assumed walkable cell
        if action_id in (1, 2, 3, 4) and prev_avatar_pos == current_avatar_pos:
            dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action_id, (0, 0))
            stride = self.physical.avatar_stride
            target_r = prev_avatar_pos[0] + dr * stride
            target_c = prev_avatar_pos[1] + dc * stride
            H, W = current_grid.shape
            if 0 <= target_r < H and 0 <= target_c < W:
                target_color = int(current_grid[target_r, target_c])
                if target_color in self.physical.walkable_colors:
                    # Contradiction! Assumed walkable color acted as solid obstacle
                    self.physical.color_confidences[target_color] = self.physical.color_confidences.get(target_color, 0.5) - 0.4
                    if self.physical.color_confidences[target_color] <= 0.2:
                        self.physical.walkable_colors.remove(target_color)
                        self.physical.solid_colors.add(target_color)
                        detail = f"Invariant Invalidation: Color {target_color} demoted from walkable to solid"
                        self.violation_log.append({'step': len(self.violation_log), 'detail': detail})
                        return False, detail

        # Check if avatar stepped on an assumed hazard color without game over
        if not game_over and current_avatar_pos != prev_avatar_pos:
            stepped_color = int(prev_grid[current_avatar_pos[0], current_avatar_pos[1]])
            if stepped_color in self.physical.hazard_colors:
                self.physical.hazard_colors.remove(stepped_color)
                detail = f"Invariant Invalidation: Color {stepped_color} removed from hazard_colors (avatar survived)"
                self.violation_log.append({'step': len(self.violation_log), 'detail': detail})
                return False, detail

        return True, None
