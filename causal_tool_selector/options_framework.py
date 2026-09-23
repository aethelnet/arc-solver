"""
=============================================================================
ARC-AGI CAUSAL META-TOOL SELECTOR: HIERARCHICAL OPTIONS FRAMEWORK
=============================================================================
Formal Semi-Markov Decision Process (SMDP) Options Framework (Sutton, Precup 1999):
Each Option ω is defined by a formal triple:
    ω = < I_ω, π_ω, β_ω >
    - I_ω: Initiation Set (preconditions in the grid / affordance state)
    - π_ω: Intra-Option Policy (step-by-step action generator)
    - β_ω: Termination Condition (subgoal completion or failure budget)

Provides:
1. Subgoal representation and status tracking.
2. SubgoalGenerator: Synthesizes candidates (Key, Target, Frontier, Interact).
3. Concrete Options: ReachEntityOption, InteractOption, FrontierExploreOption, DeadlockResetOption.
4. Hierarchical OptionsController: Manages option lifecycle and SMDP transitions.
=============================================================================
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional, Set
import collections
import heapq
import numpy as np

from .affordance_encoder import AffordanceFeatures
from .frontier_explorer import DomainAgnosticFrontierExplorer
from .mental_sandbox import MentalSandbox
from .local_physics_inducer import LocalPhysicsInducer
from .dynamic_tracker import DynamicEntityTracker, space_time_a_star, MotionPattern, DIR_MAP
from .platformer_planner import platformer_a_star, is_supported, simulate_gravity_fall, simulate_ballistic_jump
from .episodic_memory import EpisodicMemoryBank


class SubgoalStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class SubgoalType(Enum):
    REACH_ENTITY = "REACH_ENTITY"
    INTERACT_ENTITY = "INTERACT_ENTITY"
    EXPLORE_FRONTIER = "EXPLORE_FRONTIER"
    PUSH_ENTITY = "PUSH_ENTITY"
    TOGGLE_SOLVE = "TOGGLE_SOLVE"
    COLOR_DIP = "COLOR_DIP"
    DEADLOCK_RESET = "DEADLOCK_RESET"
    STAGE_ENTITY = "STAGE_ENTITY"
    CARRY_ENTITY = "CARRY_ENTITY"
    SPACE_TIME_NAV = "SPACE_TIME_NAV"
    GRAVITY_PLATFORM = "GRAVITY_PLATFORM"
    SWITCH_AVATAR = "SWITCH_AVATAR"
    BRIDGE_GAP = "BRIDGE_GAP"


class Subgoal:
    """
    Explicit teleological target representing an environmental sub-objective.
    Supports causal dependency chaining (DAGs) and counterfactual physical effects.
    """
    def __init__(
        self,
        subgoal_id: str,
        subgoal_type: SubgoalType,
        target_pos: Optional[Tuple[int, int]] = None,
        target_color: Optional[int] = None,
        max_steps: int = 25,
        metadata: Optional[Dict[str, Any]] = None,
        precondition_ids: Optional[List[str]] = None,
        dependent_ids: Optional[List[str]] = None,
        effects: Optional[Dict[str, Any]] = None
    ):
        self.subgoal_id = subgoal_id
        self.subgoal_type = subgoal_type
        self.target_pos = target_pos
        self.target_color = target_color
        self.max_steps = max_steps
        self.step_count = 0
        self.status = SubgoalStatus.PENDING
        self.metadata = metadata or {}
        self.precondition_ids = precondition_ids or []
        self.dependent_ids = dependent_ids or []
        self.effects = effects or {}

    def __repr__(self) -> str:
        return (f"Subgoal({self.subgoal_id}, {self.subgoal_type.value}, "
                f"pos={self.target_pos}, color={self.target_color}, status={self.status.value}, "
                f"pre={self.precondition_ids})")


class CausalSubgoalDAG:
    """
    Causal Subgoal DAG (Phase 220 / Hebel 1):
    Manages teleological precondition trees and sub-objective sequencing.
    Tracks topological readiness, cascading failure propagation, and
    counterfactual physical effects (e.g. bridged pits, unlocked barriers).
    """
    def __init__(self):
        self.subgoals: Dict[str, Subgoal] = {}
        self.preconditions: Dict[str, Set[str]] = {}
        self.dependents: Dict[str, Set[str]] = {}
        self.accumulated_effects: Dict[str, Any] = {
            'hypothetical_solids': set(),
            'unlocked_barriers': set()
        }

    def add_subgoal(self, subgoal: Subgoal):
        self.subgoals[subgoal.subgoal_id] = subgoal
        self.preconditions[subgoal.subgoal_id] = set(subgoal.precondition_ids)
        if subgoal.subgoal_id not in self.dependents:
            self.dependents[subgoal.subgoal_id] = set()
        for pre_id in subgoal.precondition_ids:
            self.dependents.setdefault(pre_id, set()).add(subgoal.subgoal_id)

    def add_subgoals(self, subgoals: List[Subgoal]):
        for sg in subgoals:
            self.add_subgoal(sg)

    def get_next_ready_subgoal(self) -> Optional[Subgoal]:
        """
        Returns the next PENDING subgoal whose preconditions are all COMPLETED.
        """
        for sg_id, sg in self.subgoals.items():
            if sg.status == SubgoalStatus.PENDING:
                pre_ids = self.preconditions.get(sg_id, set())
                all_met = True
                for pid in pre_ids:
                    pre_sg = self.subgoals.get(pid)
                    if pre_sg is None or pre_sg.status != SubgoalStatus.COMPLETED:
                        all_met = False
                        break
                if all_met:
                    return sg
        return None

    def mark_completed(self, subgoal_id: str):
        """
        Marks subgoal as completed and accumulates its physical effects into the DAG.
        """
        if subgoal_id in self.subgoals:
            sg = self.subgoals[subgoal_id]
            sg.status = SubgoalStatus.COMPLETED
            if sg.effects:
                if 'hypothetical_solids' in sg.effects:
                    self.accumulated_effects['hypothetical_solids'].update(sg.effects['hypothetical_solids'])
                if 'unlocked_barriers' in sg.effects:
                    self.accumulated_effects['unlocked_barriers'].update(sg.effects['unlocked_barriers'])

    def mark_failed(self, subgoal_id: str):
        """
        Marks subgoal as failed and triggers cascading failure on all dependent subgoals.
        """
        if subgoal_id in self.subgoals:
            sg = self.subgoals[subgoal_id]
            sg.status = SubgoalStatus.FAILED

            # Cascade failure to all downstream dependents
            to_fail = list(self.dependents.get(subgoal_id, set()))
            while to_fail:
                dep_id = to_fail.pop(0)
                dep_sg = self.subgoals.get(dep_id)
                if dep_sg and dep_sg.status in (SubgoalStatus.PENDING, SubgoalStatus.ACTIVE):
                    dep_sg.status = SubgoalStatus.FAILED
                    to_fail.extend(list(self.dependents.get(dep_id, set())))

    def has_pending_or_active(self) -> bool:
        return any(sg.status in (SubgoalStatus.PENDING, SubgoalStatus.ACTIVE) for sg in self.subgoals.values())

    def clear(self):
        self.subgoals.clear()
        self.preconditions.clear()
        self.dependents.clear()
        self.accumulated_effects = {
            'hypothetical_solids': set(),
            'unlocked_barriers': set()
        }


class BaseOption(ABC):
    """
    Formal SMDP Option: ω = < I_ω, π_ω, β_ω >
    """
    def __init__(self, option_id: str, name: str):
        self.option_id = option_id
        self.name = name

    @abstractmethod
    def can_initiate(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int],
        affordance: Optional[AffordanceFeatures] = None
    ) -> bool:
        """I_ω: Determines if the option can be initiated given state and subgoal."""
        pass

    @abstractmethod
    def step(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """π_ω: Intra-option policy executing one step."""
        pass

    @abstractmethod
    def is_terminated(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        levels_completed: int,
        prev_levels_completed: int
    ) -> Tuple[bool, SubgoalStatus, str]:
        """β_ω: Termination condition checking completion, timeout, or failure."""
        pass

    @abstractmethod
    def reset(self):
        """Resets internal state between option activations."""
        pass


class ReachEntityOption(BaseOption):
    """
    Option to navigate the avatar to a salient entity (target_pos).
    Termination: Avatar reaches or is adjacent to target_pos, or path is permanently blocked.
    """
    def __init__(
        self,
        explorer: Optional[DomainAgnosticFrontierExplorer] = None,
        sandbox: Optional[MentalSandbox] = None
    ):
        super().__init__(option_id="opt_reach", name="ReachEntity")
        self.explorer = explorer if explorer is not None else DomainAgnosticFrontierExplorer()
        self.sandbox = sandbox if sandbox is not None else MentalSandbox()
        self.consecutive_no_progress = 0
        self.last_dist: Optional[int] = None

    def can_initiate(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int],
        affordance: Optional[AffordanceFeatures] = None
    ) -> bool:
        has_nav = any(a in [1, 2, 3, 4] for a in available_actions)
        if not has_nav or subgoal.target_pos is None:
            return False
        # If subgoal has a separate external block that is NOT the avatar, delegate to PushEntity
        if subgoal.subgoal_type == SubgoalType.PUSH_ENTITY:
            return False
        if subgoal.metadata and ('block_pos' in subgoal.metadata or 'block_center' in subgoal.metadata):
            if subgoal.metadata.get('action') not in ('seat', 'seat_target'):
                b_pos = subgoal.metadata.get('block_pos') or subgoal.metadata.get('block_center')
                if b_pos is not None and self.explorer.avatar_pos is not None:
                    dist = abs(self.explorer.avatar_pos[0] - b_pos[0]) + abs(self.explorer.avatar_pos[1] - b_pos[1])
                    stride = getattr(self.explorer, 'stride', 1)
                    if dist > max(stride, 1):
                        return False
        return True

    def step(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        subgoal.step_count += 1
        h, w = grid.shape
        bg_canvas = int(collections.Counter(grid.flatten()).most_common(1)[0][0])

        if self.explorer.avatar_pos is not None and subgoal.target_pos is not None:
            # Check progress distance
            dist = (abs(self.explorer.avatar_pos[0] - subgoal.target_pos[0]) +
                    abs(self.explorer.avatar_pos[1] - subgoal.target_pos[1]))
            if self.last_dist is not None and dist >= self.last_dist:
                self.consecutive_no_progress += 1
            else:
                self.consecutive_no_progress = 0
            self.last_dist = dist

            # 1. Platformer Ballistic Pathfinding if Gravity active
            gravity = self.sandbox.physics_inducer.gravity_vector if (self.sandbox and self.sandbox.physics_inducer and getattr(self.sandbox.physics_inducer, 'gravity_vector', None) is not None) else None
            hypo_solids = subgoal.metadata.get('hypothetical_solids') if subgoal.metadata else None
            if gravity is not None and gravity != (0, 0):
                solids = (self.sandbox.physics_inducer.solid_colors if self.sandbox.physics_inducer else set()) or set()
                hazards = (self.sandbox.physics_inducer.hazard_colors if self.sandbox.physics_inducer else set()) or set()
                suspects = getattr(self.sandbox.physics_inducer, 'suspect_hazard_colors', set()) or set()
                plat_actions = platformer_a_star(
                    grid=grid,
                    start_pos=self.explorer.avatar_pos,
                    goal_candidates={subgoal.target_pos},
                    gravity=gravity,
                    jump_height=getattr(self.sandbox.physics_inducer, 'jump_height', 2),
                    jump_reach=getattr(self.sandbox.physics_inducer, 'jump_reach', 2),
                    solid_colors=solids,
                    hazard_colors=hazards,
                    suspect_hazard_colors=suspects,
                    bg_canvas=bg_canvas,
                    available_actions=available_actions,
                    stride=self.explorer.stride,
                    hypothetical_solids=hypo_solids
                )
                if plat_actions:
                    act = plat_actions[0]
                    if act in available_actions:
                        self.explorer.last_action = act
                        self.explorer.last_grid = grid.copy()
                        act_data = {'target_pos': subgoal.target_pos, 'is_goal': True}
                        return (act, act_data)

            # 2. Standard 2D Pathfinding
            avatar_cells = None
            if self.explorer.avatar_pos and self.explorer.stride > 1:
                ar, ac = self.explorer.avatar_pos
                k = self.explorer.stride // 2
                avatar_cells = {
                    (r, c) for r in range(max(0, ar - k), min(h, ar + k + 1))
                    for c in range(max(0, ac - k), min(w, ac + k + 1))
                }
            path = self.explorer._find_path(
                grid, self.explorer.avatar_pos, subgoal.target_pos, bg_canvas,
                target_color=subgoal.target_color, avatar_cells=avatar_cells
            )
            if path and len(path) > 1:
                actions = [self.explorer._direction_to_action(path[i], path[i+1]) for i in range(len(path)-1)]
                # Run k-step lookahead (up to 5 steps) in mental sandbox
                is_safe, safe_len, reason = self.sandbox.simulate_plan_lookahead(
                    grid,
                    self.explorer.avatar_pos,
                    actions[:5],
                    stride=self.explorer.stride,
                    floor_colors=self.explorer.floor_colors,
                    avatar_color=self.explorer.avatar_color
                )
                if is_safe or safe_len > 0:
                    act = actions[0]
                    if act in available_actions:
                        self.explorer.last_action = act
                        self.explorer.last_grid = grid.copy()
                        act_data = {'target_pos': subgoal.target_pos, 'is_goal': True}
                        return (act, act_data)
                else:
                    # Lookahead vetoed path (e.g. leads to corner deadlock or loop)
                    self.consecutive_no_progress += 2

        # Fallback to general exploration step
        return self.explorer.step(grid, available_actions)

    def is_terminated(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        levels_completed: int,
        prev_levels_completed: int
    ) -> Tuple[bool, SubgoalStatus, str]:
        # 1. Immediate Win Termination
        if levels_completed > prev_levels_completed:
            return (True, SubgoalStatus.COMPLETED, "Level advanced during navigation")

        if self.explorer.avatar_pos is not None and subgoal.target_pos is not None:
            dr = abs(self.explorer.avatar_pos[0] - subgoal.target_pos[0])
            dc = abs(self.explorer.avatar_pos[1] - subgoal.target_pos[1])
            stride = getattr(self.explorer, 'stride', 1)

            # Special check for collinear propulsion
            if subgoal.metadata and subgoal.metadata.get('action') == 'propel':
                init_pos = subgoal.metadata.get('target_block_initial_pos')
                init_val = subgoal.metadata.get('target_block_initial_val')
                target_color = subgoal.target_color
                # Terminate only when target block has physically moved away from init_pos
                if init_pos is not None and init_val is not None:
                    if int(grid[init_pos[0], init_pos[1]]) != init_val:
                        return (True, SubgoalStatus.COMPLETED, f"Propelled block from {init_pos}")
                elif init_pos is not None and target_color is not None:
                    if grid[init_pos[0], init_pos[1]] != target_color:
                        return (True, SubgoalStatus.COMPLETED, f"Propelled block from {init_pos}")

            # Seating a block requires exact positioning at receptor center (dr == 0 and dc == 0)
            elif subgoal.metadata and ('block_idx' in subgoal.metadata or subgoal.metadata.get('action') in ('seat', 'reach_exact')):
                if dr == 0 and dc == 0:
                    return (True, SubgoalStatus.COMPLETED, f"Seated block exactly at {subgoal.target_pos}")

            # Target reached if on target or adjacent (Manhattan distance <= stride)
            elif dr + dc <= stride:
                if dr == 0 and dc == 0:
                    return (True, SubgoalStatus.COMPLETED, f"Reached target {subgoal.target_pos}")
                elif (dr == 0 or dc == 0) and self._is_corridor_walkable(grid, self.explorer.avatar_pos, subgoal.target_pos, target_color=subgoal.target_color):
                    return (True, SubgoalStatus.COMPLETED, f"Reached target {subgoal.target_pos}")
                elif dr + dc <= 1:
                    return (True, SubgoalStatus.COMPLETED, f"Reached target {subgoal.target_pos}")

        # 2. Timeout / Budget Termination
        if subgoal.step_count >= subgoal.max_steps:
            return (True, SubgoalStatus.TIMEOUT, f"Exceeded max steps {subgoal.max_steps}")

        # 3. Permanent Block Termination (Step Budget Defense)
        if self.consecutive_no_progress >= 6:
            return (True, SubgoalStatus.FAILED, "No distance progress for 6 consecutive steps")

        return (False, SubgoalStatus.ACTIVE, "Navigating towards target")

    def reset(self):
        self.consecutive_no_progress = 0
        self.last_dist = None

    def _is_corridor_walkable(
        self,
        grid: np.ndarray,
        c1: Tuple[int, int],
        c2: Tuple[int, int],
        bg_canvas: int = 0,
        target_color: Optional[int] = None
    ) -> bool:
        if c1[0] != c2[0] and c1[1] != c2[1]:
            return False
        h, w = grid.shape
        step_r = 1 if c2[0] > c1[0] else (-1 if c2[0] < c1[0] else 0)
        step_c = 1 if c2[1] > c1[1] else (-1 if c2[1] < c1[1] else 0)
        cr, cc = c1[0] + step_r, c1[1] + step_c
        goal_color = grid[c2[0], c2[1]]
        while (cr, cc) != c2:
            if not (0 <= cr < h and 0 <= cc < w):
                return False
            if (cr, cc) in self.explorer.obstacle_cells:
                return False
            cval = grid[cr, cc]
            if cval in self.explorer.hazard_colors or cval in self.explorer.suspect_hazard_colors:
                return False
            is_walk = (
                cval in self.explorer.floor_colors or
                (not self.explorer.floor_colors and cval == bg_canvas) or
                (target_color is not None and cval == target_color) or
                cval == goal_color or
                (self.explorer.avatar_color is not None and cval == self.explorer.avatar_color) or
                (cr, cc) in self.explorer.traversable_cells
            )
            if not is_walk:
                return False
            cr += step_r
            cc += step_c
        return True


class PushEntityOption(BaseOption):
    """
    Formal SMDP Option: ω = < I_push, π_push, β_push >
    Autonomous block displacement policy for Sokoban / push puzzles.
    
    1. Computes macro-level block trajectory from current block position B to target G.
    2. Identifies required push direction d = B' - B and push pose P = B - d.
    3. Prunes moves that lead to irreversible corner deadlocks.
    4. Navigates avatar to push pose P treating the block as an impassable obstacle.
    5. Dispatches push action once avatar is positioned at P.
    """
    def __init__(
        self,
        explorer: Optional[DomainAgnosticFrontierExplorer] = None,
        sandbox: Optional[MentalSandbox] = None,
        physics_inducer: Optional[LocalPhysicsInducer] = None
    ):
        super().__init__(option_id="opt_push", name="PushEntity")
        self.explorer = explorer if explorer is not None else DomainAgnosticFrontierExplorer()
        self.sandbox = sandbox if sandbox is not None else MentalSandbox()
        self.physics_inducer = physics_inducer if physics_inducer is not None else (
            getattr(self.sandbox, 'physics_inducer', None) or LocalPhysicsInducer()
        )
        self.consecutive_no_progress = 0
        self.last_block_dist: Optional[int] = None
        self.current_block_pos: Optional[Tuple[int, int]] = None
        self.planned_block_path: List[Tuple[int, int]] = []

    def reset(self):
        self.consecutive_no_progress = 0
        self.last_block_dist = None
        self.current_block_pos = None
        self.planned_block_path = []

    def can_initiate(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int],
        affordance: Optional[AffordanceFeatures] = None
    ) -> bool:
        has_nav = any(a in [1, 2, 3, 4] for a in available_actions)
        if not has_nav or subgoal.target_pos is None:
            return False

        cur_b_pos = self._find_block_position(grid, subgoal) or (
            subgoal.metadata.get('block_pos') or subgoal.metadata.get('block_center') if subgoal.metadata else None
        )
        if cur_b_pos is not None and self.explorer.avatar_pos is not None:
            # If avatar IS the block (co-located at same pos), it cannot push itself from behind -> delegate to ReachEntity
            dist_to_avatar = abs(self.explorer.avatar_pos[0] - cur_b_pos[0]) + abs(self.explorer.avatar_pos[1] - cur_b_pos[1])
            if dist_to_avatar == 0:
                return False
            return True

        if subgoal.subgoal_type == SubgoalType.PUSH_ENTITY:
            return True

        return False

    def _cluster_cells(self, coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Groups 4-connected cells into components and returns their centroids."""
        coords_set = set(coords)
        visited = set()
        centroids = []
        for pt in coords:
            if pt not in visited:
                comp = []
                q = [pt]
                visited.add(pt)
                while q:
                    cr, cc = q.pop(0)
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nbr = (cr + dr, cc + dc)
                        if nbr in coords_set and nbr not in visited:
                            visited.add(nbr)
                            q.append(nbr)
                arr = np.array(comp)
                mean_r = int(np.round(np.mean(arr[:, 0])))
                mean_c = int(np.round(np.mean(arr[:, 1])))
                centroids.append((mean_r, mean_c))
        return centroids

    def _find_block_position(self, grid: np.ndarray, subgoal: Subgoal) -> Optional[Tuple[int, int]]:
        """Locates current centroid of the target block on the grid."""
        h, w = grid.shape
        block_color = subgoal.metadata.get('block_color') if subgoal.metadata else None
        ref_pos = self.current_block_pos or (
            subgoal.metadata.get('block_pos') or subgoal.metadata.get('block_center')
            if subgoal.metadata else None
        )

        if block_color is not None:
            coords = np.argwhere(grid == block_color)
            if len(coords) > 0:
                valid_coords = [
                    (int(r), int(c)) for r, c in coords
                    if self.explorer.avatar_pos is None or (r, c) != self.explorer.avatar_pos
                ]
                if not valid_coords:
                    valid_coords = [(int(r), int(c)) for r, c in coords]
                clusters = self._cluster_cells(valid_coords)
                if clusters:
                    if ref_pos is not None:
                        return min(
                            clusters,
                            key=lambda cl: abs(cl[0] - ref_pos[0]) + abs(cl[1] - ref_pos[1])
                        )
                    return clusters[0]

        return ref_pos

    def _get_block_cells(
        self,
        grid: np.ndarray,
        block_pos: Tuple[int, int],
        block_color: Optional[int],
        block_size: Optional[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """Identifies all cells currently occupied by the block to avoid colliding during avatar navigation."""
        cells: Set[Tuple[int, int]] = set()
        if block_color is not None:
            coords = np.argwhere(grid == block_color)
            for r, c in coords:
                if abs(r - block_pos[0]) <= 3 and abs(c - block_pos[1]) <= 3:
                    cells.add((int(r), int(c)))
        if not cells:
            if block_size and len(block_size) >= 2:
                hr, wr = block_size[0] // 2, block_size[1] // 2
                for dr in range(-hr, hr + 1):
                    for dc in range(-wr, wr + 1):
                        cells.add((block_pos[0] + dr, block_pos[1] + dc))
            else:
                cells.add(block_pos)
        return cells

    def _is_solid(
        self,
        grid: np.ndarray,
        pos: Tuple[int, int],
        bg_canvas: int,
        target_color: Optional[int] = None,
        block_color: Optional[int] = None,
        active_block_cells: Optional[Set[Tuple[int, int]]] = None
    ) -> bool:
        """Determines if a cell is an impassable barrier for navigation or pushing."""
        r, c = pos
        h, w = grid.shape
        if not (0 <= r < h and 0 <= c < w):
            return True
        if pos in self.explorer.obstacle_cells:
            return True
        val = int(grid[r, c])
        if val in self.physics_inducer.solid_colors:
            return True
        if val in self.physics_inducer.hazard_colors:
            return True

        if active_block_cells is not None:
            if pos in active_block_cells:
                return False  # Active block cell is vacated when block moves
            if block_color is not None and val == block_color:
                return True  # Other blocks of the same color are solid obstacles
        elif block_color is not None and val == block_color:
            return False  # Fallback: block cell is vacated when block moves

        if self.explorer.floor_colors:
            if (val in self.explorer.floor_colors or
                val == bg_canvas or
                (target_color is not None and val == target_color) or
                (self.explorer.avatar_color is not None and val == self.explorer.avatar_color) or
                pos in self.explorer.traversable_cells):
                return False
            return True
        if val == bg_canvas or (target_color is not None and val == target_color):
            return False
        return False

    def _is_corner_deadlock(
        self,
        grid: np.ndarray,
        pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        bg_canvas: int,
        target_color: Optional[int] = None,
        block_color: Optional[int] = None,
        stride: int = 1,
        active_block_cells: Optional[Set[Tuple[int, int]]] = None
    ) -> bool:
        """Checks if placing a block at pos creates an irreversible 2-wall corner deadlock."""
        if pos == target_pos:
            return False
        r, c = pos
        up = self._is_solid(grid, (r - stride, c), bg_canvas, target_color, block_color, active_block_cells)
        down = self._is_solid(grid, (r + stride, c), bg_canvas, target_color, block_color, active_block_cells)
        left = self._is_solid(grid, (r, c - stride), bg_canvas, target_color, block_color, active_block_cells)
        right = self._is_solid(grid, (r, c + stride), bg_canvas, target_color, block_color, active_block_cells)

        if (up and left) or (up and right) or (down and left) or (down and right):
            return True
        return False

    def _plan_block_path(
        self,
        grid: np.ndarray,
        start_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        bg_canvas: int,
        target_color: Optional[int] = None,
        block_color: Optional[int] = None,
        stride: int = 1,
        prune_corners: bool = True,
        active_block_cells: Optional[Set[Tuple[int, int]]] = None
    ) -> Optional[List[Tuple[int, int]]]:
        """A* search for the optimal, deadlock-free block push trajectory."""
        h, w = grid.shape
        heap: List[Tuple[float, float, Tuple[int, int], List[Tuple[int, int]], Optional[Tuple[int, int]]]] = [
            (0.0, 0.0, start_pos, [start_pos], None)
        ]
        visited: Dict[Tuple[int, int], float] = {start_pos: 0.0}

        while heap:
            f, g, curr, path, last_dir = heapq.heappop(heap)
            if curr == target_pos:
                return path

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (curr[0] + dr * stride, curr[1] + dc * stride)
                push_pose = (curr[0] - dr * stride, curr[1] - dc * stride)

                if not (0 <= next_pos[0] < h and 0 <= next_pos[1] < w):
                    continue
                if not (0 <= push_pose[0] < h and 0 <= push_pose[1] < w):
                    continue
                if self._is_solid(grid, next_pos, bg_canvas, target_color, block_color, active_block_cells):
                    continue
                if self._is_solid(grid, push_pose, bg_canvas, target_color, block_color, active_block_cells):
                    continue
                if prune_corners and self._is_corner_deadlock(grid, next_pos, target_pos, bg_canvas, target_color, block_color, stride, active_block_cells):
                    continue

                step_cost = 1.0
                if last_dir is not None and (dr, dc) != last_dir:
                    step_cost += 1.5  # Turning penalty

                new_g = g + step_cost
                if next_pos not in visited or new_g < visited[next_pos]:
                    visited[next_pos] = new_g
                    heur = float(abs(next_pos[0] - target_pos[0]) + abs(next_pos[1] - target_pos[1]))
                    heapq.heappush(heap, (new_g + heur, new_g, next_pos, path + [next_pos], (dr, dc)))

        if prune_corners:
            # Fallback without corner pruning if goal is adjacent to a wall or space is constrained
            return self._plan_block_path(grid, start_pos, target_pos, bg_canvas, target_color, block_color, stride, prune_corners=False, active_block_cells=active_block_cells)

        return None

    def _find_avatar_path(
        self,
        grid: np.ndarray,
        avatar_pos: Tuple[int, int],
        push_pose: Tuple[int, int],
        block_cells: Set[Tuple[int, int]],
        bg_canvas: int,
        target_color: Optional[int] = None,
        stride: int = 1,
        active_block_cells: Optional[Set[Tuple[int, int]]] = None
    ) -> Optional[List[Tuple[int, int]]]:
        """A* navigation for avatar to reach push pose P while treating block B as impassable."""
        if avatar_pos == push_pose:
            return [avatar_pos]

        h, w = grid.shape
        def heur(p: Tuple[int, int]) -> float:
            return (abs(p[0] - push_pose[0]) + abs(p[1] - push_pose[1])) / float(stride)

        h0 = heur(avatar_pos)
        pq: List[Tuple[float, float, int, List[Tuple[int, int]]]] = [(h0, 0.0, 0, [avatar_pos])]
        visited: Dict[Tuple[int, int], float] = {avatar_pos: 0.0}
        counter = 0

        while pq:
            f, g, _, path = heapq.heappop(pq)
            curr = path[-1]

            if curr == push_pose:
                return path

            if g > visited.get(curr, float('inf')):
                continue

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr * stride, curr[1] + dc * stride
                nbr = (nr, nc)
                if (0 <= nr < h and 0 <= nc < w and
                    nbr not in block_cells and
                    not self._is_solid(grid, nbr, bg_canvas, target_color, block_color=None, active_block_cells=active_block_cells)):
                    new_g = g + 1.0
                    if new_g < visited.get(nbr, float('inf')):
                        visited[nbr] = new_g
                        h_val = heur(nbr)
                        counter += 1
                        heapq.heappush(pq, (new_g + h_val, new_g, counter, path + [nbr]))

        return None

    def step(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        subgoal.step_count += 1
        h, w = grid.shape
        bg_canvas = int(collections.Counter(grid.flatten()).most_common(1)[0][0])
        stride = getattr(self.explorer, 'stride', 1)

        block_pos = self._find_block_position(grid, subgoal)
        avatar_pos = self.explorer.avatar_pos
        target_pos = subgoal.target_pos

        if block_pos is None or avatar_pos is None or target_pos is None:
            return self.explorer.step(grid, available_actions)

        block_size = subgoal.metadata.get('block_size') if subgoal.metadata else None
        block_cells = self._get_block_cells(grid, block_pos, subgoal.metadata.get('block_color'), block_size)

        # Track block distance progress
        dist = abs(block_pos[0] - target_pos[0]) + abs(block_pos[1] - target_pos[1])
        if self.last_block_dist is not None and dist >= self.last_block_dist:
            self.consecutive_no_progress += 1
        else:
            self.consecutive_no_progress = 0
        self.last_block_dist = dist
        self.current_block_pos = block_pos

        # Plan / Re-plan block path if needed
        if not self.planned_block_path or self.planned_block_path[0] != block_pos:
            block_color = subgoal.metadata.get('block_color') if subgoal.metadata else None
            self.planned_block_path = self._plan_block_path(
                grid, block_pos, target_pos, bg_canvas, subgoal.target_color, block_color, stride,
                active_block_cells=block_cells
            ) or []

        if self.planned_block_path and len(self.planned_block_path) > 1:
            next_block = self.planned_block_path[1]
            dr = int(np.sign(next_block[0] - block_pos[0]))
            dc = int(np.sign(next_block[1] - block_pos[1]))

            if block_size and len(block_size) >= 2:
                hr, wr = block_size[0] // 2, block_size[1] // 2
                offset_r = dr * (hr + 1)
                offset_c = dc * (wr + 1)
            else:
                offset_r = dr * stride
                offset_c = dc * stride

            push_pose = (block_pos[0] - offset_r, block_pos[1] - offset_c)

            # Check if avatar is at push pose
            dist_to_push_pose = abs(avatar_pos[0] - push_pose[0]) + abs(avatar_pos[1] - push_pose[1])
            if dist_to_push_pose == 0:
                # Avatar is at push pose! Execute push action
                act = self.explorer._direction_to_action(avatar_pos, (avatar_pos[0] + dr * stride, avatar_pos[1] + dc * stride))
                if act in available_actions:
                    self.explorer.last_action = act
                    self.explorer.last_grid = grid.copy()
                    return (act, None)

            # Avatar navigates to push pose
            avatar_path = self._find_avatar_path(
                grid, avatar_pos, push_pose, block_cells, bg_canvas, subgoal.target_color, stride,
                active_block_cells=block_cells
            )
            if avatar_path and len(avatar_path) > 1:
                next_avatar = avatar_path[1]
                act = self.explorer._direction_to_action(avatar_pos, next_avatar)
                if act in available_actions:
                    self.explorer.last_action = act
                    self.explorer.last_grid = grid.copy()
                    return (act, None)
            else:
                # Direct route to push pose blocked (e.g. block in tight corridor)
                self.consecutive_no_progress += 2

        # Fallback to frontier exploration step
        return self.explorer.step(grid, available_actions)

    def is_terminated(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        levels_completed: int,
        prev_levels_completed: int
    ) -> Tuple[bool, SubgoalStatus, str]:
        if levels_completed > prev_levels_completed:
            return (True, SubgoalStatus.COMPLETED, "Level advanced during block push")

        block_pos = self._find_block_position(grid, subgoal)
        if block_pos is not None and subgoal.target_pos is not None:
            dist = abs(block_pos[0] - subgoal.target_pos[0]) + abs(block_pos[1] - subgoal.target_pos[1])
            stride = getattr(self.explorer, 'stride', 1)
            if dist == 0 or dist <= stride // 2:
                return (True, SubgoalStatus.COMPLETED, f"Block reached target {subgoal.target_pos}")

        if subgoal.step_count >= subgoal.max_steps:
            return (True, SubgoalStatus.TIMEOUT, f"Push exceeded max steps {subgoal.max_steps}")

        if self.consecutive_no_progress >= 12:
            return (True, SubgoalStatus.FAILED, "No push progress for 12 consecutive steps")

        return (False, SubgoalStatus.ACTIVE, "Pushing block to target")


class BridgeGapOption(PushEntityOption):
    """
    Formal SMDP Option: ω = < I_bridge, π_bridge, β_bridge > (Phase 220 / Hebel 1)
    Autonomous gap bridging policy:
    1. Identifies movable block candidate and navigation route behind block.
    2. Pushes block towards candidate drop ledge over the chasm/pit.
    3. Simulates block falling under gravity into target pit cell.
    4. Terminates successfully when block occupies the landing cell, creating a walkable bridge.
    """
    def __init__(
        self,
        explorer: Optional[DomainAgnosticFrontierExplorer] = None,
        sandbox: Optional[MentalSandbox] = None,
        physics_inducer: Optional[LocalPhysicsInducer] = None
    ):
        super().__init__(explorer, sandbox, physics_inducer)
        self.option_id = "opt_bridge_gap"
        self.name = "BridgeGap"

    def can_initiate(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int],
        affordance: Optional[AffordanceFeatures] = None
    ) -> bool:
        if subgoal.subgoal_type != SubgoalType.BRIDGE_GAP and (not subgoal.metadata or subgoal.metadata.get('action') != 'bridge_gap'):
            return False
        return super().can_initiate(grid, subgoal, available_actions, affordance)

    def is_terminated(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        levels_completed: int,
        prev_levels_completed: int
    ) -> Tuple[bool, SubgoalStatus, str]:
        if levels_completed > prev_levels_completed:
            return (True, SubgoalStatus.COMPLETED, "Level advanced during chasm bridging")

        grid_2d = np.asarray(grid)
        while grid_2d.ndim > 2:
            grid_2d = grid_2d[-1]

        landing_pos = subgoal.metadata.get('landing_pos') if subgoal.metadata else None
        drop_pos = subgoal.target_pos or (subgoal.metadata.get('drop_pos') if subgoal.metadata else None)
        block_color = subgoal.metadata.get('block_color') if subgoal.metadata else subgoal.target_color

        # 1. Direct inspection of landing cell in grid: Did block land in the pit?
        if landing_pos is not None:
            lr, lc = landing_pos
            if 0 <= lr < grid_2d.shape[0] and 0 <= lc < grid_2d.shape[1]:
                if block_color is not None and int(grid_2d[lr, lc]) == block_color:
                    return (True, SubgoalStatus.COMPLETED, f"Block successfully landed in chasm at {landing_pos}")

        # 2. Block centroid inspection
        block_pos = self._find_block_position(grid_2d, subgoal)
        if block_pos is not None:
            if landing_pos is not None and block_pos == landing_pos:
                return (True, SubgoalStatus.COMPLETED, f"Block reached landing cell {landing_pos}")
            if drop_pos is not None and block_pos == drop_pos:
                return (True, SubgoalStatus.COMPLETED, f"Block pushed to drop edge {drop_pos}")

        if subgoal.step_count >= subgoal.max_steps:
            return (True, SubgoalStatus.TIMEOUT, f"Chasm bridge exceeded max steps {subgoal.max_steps}")

        if self.consecutive_no_progress >= 12:
            return (True, SubgoalStatus.FAILED, "No bridge push progress for 12 consecutive steps")

        return (False, SubgoalStatus.ACTIVE, "Pushing block to bridge chasm")


class StageEntityOption(BaseOption):
    """
    Formal SMDP Option: ω = < I_stage, π_stage, β_stage >
    Pushes an entity to an intermediate buffer/staging area to resolve
    topological deadlocks or clear transit corridors for other entities.
    """
    def __init__(
        self,
        explorer: Optional[DomainAgnosticFrontierExplorer] = None,
        sandbox: Optional[MentalSandbox] = None,
        physics_inducer: Optional[LocalPhysicsInducer] = None
    ):
        super().__init__(option_id="opt_stage", name="StageEntity")
        self.push_option = PushEntityOption(explorer, sandbox, physics_inducer)

    def reset(self):
        self.push_option.reset()

    def can_initiate(self, grid: np.ndarray, subgoal: Subgoal, available_actions: List[int]) -> bool:
        return self.push_option.can_initiate(grid, subgoal, available_actions)

    def step(self, grid: np.ndarray, subgoal: Subgoal, available_actions: List[int]) -> Tuple[int, Optional[Dict[str, Any]]]:
        return self.push_option.step(grid, subgoal, available_actions)

    def is_terminated(self, grid: np.ndarray, subgoal: Subgoal, levels_completed: int, prev_levels_completed: int) -> Tuple[bool, SubgoalStatus, str]:
        is_done, status, reason = self.push_option.is_terminated(grid, subgoal, levels_completed, prev_levels_completed)
        if is_done and status == SubgoalStatus.COMPLETED:
            return (True, SubgoalStatus.COMPLETED, f"Block staged at {subgoal.target_pos}")
        return is_done, status, reason


class CarryEntityOption(BaseOption):
    """
    Formal SMDP Option: ω = < I_carry, π_carry, β_carry >
    Models pick-and-place, item carrying, and coupled morphology navigation.
    
    Phases:
    1. APPROACH: Navigates avatar to an adjacent neighbor of the target item.
    2. PICK: Executes Action 5 (GRAB) to couple item to avatar with relative offset o.
    3. COUPLED_TRANSIT: Navigates composite rigid body {P_avatar, P_avatar + o}
       to delivery destination with joint collision avoidance.
    4. PLACE: Executes Action 5 (PLACE) to decouple and seat item at target drop zone.
    """
    def __init__(
        self,
        explorer: Optional[DomainAgnosticFrontierExplorer] = None,
        sandbox: Optional[MentalSandbox] = None,
        physics_inducer: Optional[LocalPhysicsInducer] = None
    ):
        super().__init__(option_id="opt_carry", name="CarryEntity")
        self.explorer = explorer if explorer is not None else DomainAgnosticFrontierExplorer()
        self.sandbox = sandbox if sandbox is not None else MentalSandbox()
        self.physics_inducer = physics_inducer or getattr(self.sandbox, 'physics_inducer', LocalPhysicsInducer())

        # Internal state machine
        self.is_carrying: bool = False
        self.item_offset: Optional[Tuple[int, int]] = None
        self.has_picked: bool = False
        self.has_placed: bool = False
        self.consecutive_no_progress: int = 0
        self.last_dist: Optional[float] = None
        self.current_item_pos: Optional[Tuple[int, int]] = None

    def reset(self):
        self.is_carrying = False
        self.item_offset = None
        self.has_picked = False
        self.has_placed = False
        self.consecutive_no_progress = 0
        self.last_dist = None
        self.current_item_pos = None

    def _find_item_pos(self, grid: np.ndarray, subgoal: Subgoal) -> Optional[Tuple[int, int]]:
        """Locates current position of the item to be carried."""
        if self.current_item_pos is not None:
            return self.current_item_pos
        if subgoal.metadata:
            item_pos = subgoal.metadata.get('item_pos') or subgoal.metadata.get('block_pos') or subgoal.metadata.get('block_center')
            if item_pos is not None:
                return item_pos

        item_color = subgoal.metadata.get('item_color') if subgoal.metadata else subgoal.target_color
        if item_color is not None:
            coords = np.argwhere(grid == item_color)
            for r, c in coords:
                if self.explorer.avatar_pos is None or (r, c) != self.explorer.avatar_pos:
                    return (int(r), int(c))
        return None

    def _is_coupled_solid(
        self,
        grid: np.ndarray,
        pos: Tuple[int, int],
        bg_canvas: int,
        target_color: Optional[int] = None,
        item_color: Optional[int] = None
    ) -> bool:
        """Checks if a cell is solid for avatar or carried item."""
        r, c = pos
        h, w = grid.shape
        if not (0 <= r < h and 0 <= c < w):
            return True
        if pos in self.explorer.obstacle_cells:
            return True
        val = int(grid[r, c])
        if val in self.physics_inducer.solid_colors or val in self.physics_inducer.hazard_colors:
            return True
        if item_color is not None and val == item_color:
            return False  # Carried item cell is part of the coupled body
        if self.explorer.floor_colors:
            if (val in self.explorer.floor_colors or
                val == bg_canvas or
                (target_color is not None and val == target_color) or
                (self.explorer.avatar_color is not None and val == self.explorer.avatar_color) or
                pos in self.explorer.traversable_cells):
                return False
            return True
        if val == bg_canvas or (target_color is not None and val == target_color):
            return False
        return False

    def _plan_coupled_path(
        self,
        grid: np.ndarray,
        start_avatar: Tuple[int, int],
        target_avatar: Tuple[int, int],
        item_offset: Tuple[int, int],
        bg_canvas: int,
        target_color: Optional[int] = None,
        item_color: Optional[int] = None,
        stride: int = 1
    ) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinder for composite rigid body {avatar, avatar + item_offset}.
        Enforces that both avatar and carried item remain collision-free at every step.
        """
        if start_avatar == target_avatar:
            return [start_avatar]

        h, w = grid.shape
        dr_item, dc_item = item_offset

        def heur(p: Tuple[int, int]) -> float:
            return (abs(p[0] - target_avatar[0]) + abs(p[1] - target_avatar[1])) / float(stride)

        h0 = heur(start_avatar)
        pq: List[Tuple[float, float, int, List[Tuple[int, int]]]] = [(h0, 0.0, 0, [start_avatar])]
        visited: Dict[Tuple[int, int], float] = {start_avatar: 0.0}
        counter = 0

        while pq:
            f, g, _, path = heapq.heappop(pq)
            curr = path[-1]

            if curr == target_avatar:
                return path

            if g > visited.get(curr, float('inf')):
                continue

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr * stride, curr[1] + dc * stride
                next_avatar = (nr, nc)
                next_item = (nr + dr_item, nc + dc_item)

                # Both avatar and item must be within bounds and collision-free
                if not (0 <= next_avatar[0] < h and 0 <= next_avatar[1] < w):
                    continue
                if not (0 <= next_item[0] < h and 0 <= next_item[1] < w):
                    continue

                if self._is_coupled_solid(grid, next_avatar, bg_canvas, target_color, item_color):
                    continue
                if self._is_coupled_solid(grid, next_item, bg_canvas, target_color, item_color):
                    continue

                new_g = g + 1.0
                if new_g < visited.get(next_avatar, float('inf')):
                    visited[next_avatar] = new_g
                    h_val = heur(next_avatar)
                    counter += 1
                    heapq.heappush(pq, (new_g + h_val, new_g, counter, path + [next_avatar]))

        return None

    def can_initiate(self, grid: np.ndarray, subgoal: Subgoal, available_actions: List[int]) -> bool:
        if 5 not in available_actions:
            return False  # Pick/Place requires Action 5
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        bg_canvas = int(collections.Counter(grid.flatten()).most_common(1)[0][0])
        self.explorer._update_avatar(grid, bg_canvas)

        item_pos = self._find_item_pos(grid, subgoal)
        if item_pos is None:
            return False
        return True

    def step(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        subgoal.step_count += 1
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        h, w = grid.shape
        bg_canvas = int(collections.Counter(grid.flatten()).most_common(1)[0][0])
        stride = getattr(self.explorer, 'stride', 1)

        avatar_pos = self.explorer.avatar_pos
        item_pos = self._find_item_pos(grid, subgoal)
        drop_pos = subgoal.target_pos or (subgoal.metadata.get('drop_pos') if subgoal.metadata else None)

        if avatar_pos is None or item_pos is None or drop_pos is None:
            return self.explorer.step(grid, available_actions)

        item_color = subgoal.metadata.get('item_color') if subgoal.metadata else None

        # -------------------------------------------------------------
        # Phase 1 & 2: Not yet carrying -> Approach and Pick
        # -------------------------------------------------------------
        if not self.is_carrying:
            dist_to_item = abs(avatar_pos[0] - item_pos[0]) + abs(avatar_pos[1] - item_pos[1])
            if dist_to_item <= stride:
                # Adjacent to item! Execute Action 5 (GRAB)
                self.is_carrying = True
                self.has_picked = True
                self.item_offset = (item_pos[0] - avatar_pos[0], item_pos[1] - avatar_pos[1])
                self.current_item_pos = item_pos
                self.explorer.last_action = 5
                self.explorer.last_grid = grid.copy()
                return (5, None)

            # Navigate to adjacent cell
            path = self.explorer._find_path(grid, avatar_pos, item_pos, bg_canvas, target_color=item_color)
            if path and len(path) > 1:
                next_cell = path[1]
                if next_cell == item_pos:
                    self.is_carrying = True
                    self.has_picked = True
                    self.item_offset = (item_pos[0] - avatar_pos[0], item_pos[1] - avatar_pos[1])
                    self.current_item_pos = item_pos
                    self.explorer.last_action = 5
                    self.explorer.last_grid = grid.copy()
                    return (5, None)
                act = self.explorer._direction_to_action(avatar_pos, next_cell)
                if act in available_actions:
                    self.explorer.last_action = act
                    self.explorer.last_grid = grid.copy()
                    return (act, None)

            return self.explorer.step(grid, available_actions)

        # -------------------------------------------------------------
        # Phase 3 & 4: Carrying -> Coupled Navigation and Place
        # -------------------------------------------------------------
        dr_item, dc_item = self.item_offset if self.item_offset is not None else (0, 1)
        current_carried_item_pos = (avatar_pos[0] + dr_item, avatar_pos[1] + dc_item)
        self.current_item_pos = current_carried_item_pos

        # Target avatar pose to seat item at drop_pos:
        target_avatar = (drop_pos[0] - dr_item, drop_pos[1] - dc_item)

        # Check if item has reached drop position
        if current_carried_item_pos == drop_pos or avatar_pos == target_avatar:
            self.has_placed = True
            self.is_carrying = False
            self.explorer.last_action = 5
            self.explorer.last_grid = grid.copy()
            return (5, None)

        # Plan coupled transit path
        coupled_path = self._plan_coupled_path(
            grid, avatar_pos, target_avatar, (dr_item, dc_item),
            bg_canvas, subgoal.target_color, item_color, stride
        )
        if coupled_path and len(coupled_path) > 1:
            next_avatar = coupled_path[1]
            act = self.explorer._direction_to_action(avatar_pos, next_avatar)
            if act in available_actions:
                self.explorer.last_action = act
                self.explorer.last_grid = grid.copy()
                return (act, None)

        # Fallback single navigation if coupled path search fails
        path = self.explorer._find_path(grid, avatar_pos, target_avatar, bg_canvas, target_color=subgoal.target_color)
        if path and len(path) > 1:
            act = self.explorer._direction_to_action(avatar_pos, path[1])
            if act in available_actions:
                self.explorer.last_action = act
                self.explorer.last_grid = grid.copy()
                return (act, None)

        return self.explorer.step(grid, available_actions)

    def is_terminated(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        levels_completed: int,
        prev_levels_completed: int
    ) -> Tuple[bool, SubgoalStatus, str]:
        if levels_completed > prev_levels_completed:
            return (True, SubgoalStatus.COMPLETED, "Level advanced during carry/place")

        if self.has_placed:
            return (True, SubgoalStatus.COMPLETED, f"Item successfully placed at {subgoal.target_pos}")

        if subgoal.step_count >= subgoal.max_steps:
            return (True, SubgoalStatus.TIMEOUT, f"Carry exceeded max steps {subgoal.max_steps}")

        if self.consecutive_no_progress >= 12:
            return (True, SubgoalStatus.FAILED, "No carry progress for 12 consecutive steps")

        return (False, SubgoalStatus.ACTIVE, "Carrying entity to target")


class SpaceTimeOption(BaseOption):
    """
    Formal SMDP Option: ω = < I_spacetime, π_spacetime, β_spacetime >
    Spatio-Temporal Obstacle Avoidance and Dynamic NPC Synchronization Option.
    
    Uses DynamicEntityTracker to maintain online tracklets of moving objects,
    projects space-time reservation table R(t) over horizon H, and plans
    collision-free trajectories using space_time_a_star with non-destructive
    wait transitions.
    """
    def __init__(
        self,
        explorer: Optional[DomainAgnosticFrontierExplorer] = None,
        sandbox: Optional[MentalSandbox] = None,
        physics_inducer: Optional[LocalPhysicsInducer] = None,
        tracker: Optional[DynamicEntityTracker] = None
    ):
        super().__init__(option_id="opt_spacetime", name="SpaceTimeNav")
        self.explorer = explorer if explorer is not None else DomainAgnosticFrontierExplorer()
        self.sandbox = sandbox if sandbox is not None else MentalSandbox()
        self.physics_inducer = physics_inducer or getattr(self.sandbox, 'physics_inducer', LocalPhysicsInducer())
        self.tracker = tracker if tracker is not None else DynamicEntityTracker()

        self.current_plan: List[int] = []
        self.plan_step_idx: int = 0
        self.consecutive_no_progress: int = 0
        self.last_dist: Optional[float] = None
        self.target_pos: Optional[Tuple[int, int]] = None
        self.item_offset: Optional[Tuple[int, int]] = None
        self.replan_interval: int = 4
        self.steps_since_replan: int = 0

    def reset(self):
        self.current_plan = []
        self.plan_step_idx = 0
        self.consecutive_no_progress = 0
        self.last_dist = None
        self.target_pos = None
        self.item_offset = None
        self.steps_since_replan = 0
        self.tracker.reset()

    def _get_static_blocked(self, grid: np.ndarray, bg_canvas: int, target_pos: Optional[Tuple[int, int]] = None) -> Set[Tuple[int, int]]:
        """Identifies impassable static walls and obstacles."""
        H, W = grid.shape
        blocked = set(self.explorer.obstacle_cells)
        floors = self.explorer.floor_colors or set()

        for r in range(H):
            for c in range(W):
                val = int(grid[r, c])
                if target_pos is not None and (r, c) == target_pos:
                    continue
                if val in self.physics_inducer.solid_colors or val in self.physics_inducer.hazard_colors:
                    blocked.add((r, c))
                elif floors:
                    if (val not in floors and val != bg_canvas and
                        (self.explorer.avatar_color is None or val != self.explorer.avatar_color) and
                        (r, c) not in self.explorer.traversable_cells):
                        blocked.add((r, c))
        return blocked

    def can_initiate(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int],
        affordance: Optional[AffordanceFeatures] = None
    ) -> bool:
        has_nav = any(a in (1, 2, 3, 4) for a in available_actions)
        if not has_nav:
            return False

        # Target pos must be specified or derivable
        tgt = subgoal.target_pos or (subgoal.metadata.get('target_pos') if subgoal.metadata else None)
        if tgt is None:
            return False

        # Explicit SubgoalType.SPACE_TIME_NAV or dynamic entities exist or metadata flag
        if subgoal.subgoal_type == SubgoalType.SPACE_TIME_NAV:
            return True
        if subgoal.metadata and subgoal.metadata.get('use_spacetime'):
            return True

        # Check if tracker detected moving entities
        if any(ent.pattern != MotionPattern.STATIC for ent in self.tracker.entities.values()):
            return True

        return False

    def step(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        subgoal.step_count += 1
        self.steps_since_replan += 1

        grid_2d = np.asarray(grid)
        while grid_2d.ndim > 2:
            grid_2d = grid_2d[-1]
        H, W = grid_2d.shape
        bg_canvas = int(collections.Counter(grid_2d.flatten()).most_common(1)[0][0])
        stride = getattr(self.explorer, 'stride', 1)

        self.explorer._update_avatar(grid_2d, bg_canvas)
        self.explorer._update_floor_colors(grid_2d, bg_canvas)
        avatar_pos = self.explorer.avatar_pos

        # Update dynamic entity tracking
        self.tracker.observe_frame(
            grid=grid_2d,
            avatar_pos=avatar_pos,
            avatar_color=self.explorer.avatar_color,
            floor_colors=self.explorer.floor_colors,
            action_id=getattr(self.explorer, 'last_action', None)
        )

        target_pos = subgoal.target_pos or (subgoal.metadata.get('target_pos') if subgoal.metadata else None)
        self.target_pos = target_pos

        if avatar_pos is None or target_pos is None:
            return self.explorer.step(grid_2d, available_actions)

        # Distance progress check
        dist = abs(avatar_pos[0] - target_pos[0]) + abs(avatar_pos[1] - target_pos[1])
        if self.last_dist is not None and dist >= self.last_dist:
            self.consecutive_no_progress += 1
        else:
            self.consecutive_no_progress = 0
        self.last_dist = dist

        # Check carried item offset if carrying
        meta = subgoal.metadata or {}
        box_offset = meta.get('item_offset') or self.item_offset

        # Check if replan needed:
        need_replan = (
            len(self.current_plan) == 0 or
            self.plan_step_idx >= len(self.current_plan) or
            self.steps_since_replan >= self.replan_interval
        )

        static_blocked = self._get_static_blocked(grid_2d, bg_canvas, target_pos)

        # Pre-check current plan validity against new reservations
        if not need_replan and self.plan_step_idx < len(self.current_plan):
            res_1 = self.tracker.build_reservation_table(grid_2d, horizon=5, static_blocked=static_blocked, stride=stride)
            next_act = self.current_plan[self.plan_step_idx]
            dr, dc = DIR_MAP.get(next_act, (0, 0))
            predicted_next = (avatar_pos[0] + dr * stride, avatar_pos[1] + dc * stride)
            if predicted_next in res_1.get(1, set()):
                need_replan = True

        if need_replan:
            self.steps_since_replan = 0
            carrier_platforms = self.tracker.get_carrier_platforms(
                grid=grid_2d,
                horizon=40,
                solid_colors=self.physics_inducer.solid_colors,
                hazard_colors=self.physics_inducer.hazard_colors,
                static_blocked=static_blocked,
                gravity=getattr(self.physics_inducer, 'gravity_vector', (1, 0)) or (1, 0),
                stride=stride
            )
            reservations = self.tracker.build_reservation_table(
                grid_2d, horizon=40, static_blocked=static_blocked, stride=stride
            )
            for p in carrier_platforms:
                for t_idx, supp in p.support_by_time.items():
                    if t_idx in reservations:
                        reservations[t_idx] = reservations[t_idx] - supp

            goals = {target_pos}
            # Candidate approach cells if target is an obstacle/item
            if meta.get('approach_only'):
                goals = set()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    adj = (target_pos[0] + dr * stride, target_pos[1] + dc * stride)
                    if 0 <= adj[0] < H and 0 <= adj[1] < W and adj not in static_blocked:
                        goals.add(adj)

            plan = space_time_a_star(
                grid=grid_2d,
                start_pos=avatar_pos,
                goal_candidates=goals,
                reservations=reservations,
                static_blocked=static_blocked,
                box_offset=box_offset,
                max_time=45,
                available_actions=available_actions,
                stride=stride,
                carrier_platforms=carrier_platforms,
                gravity=getattr(self.physics_inducer, 'gravity_vector', None)
            )
            if plan is not None:
                self.current_plan = plan
                self.plan_step_idx = 0
            else:
                self.current_plan = []
                self.plan_step_idx = 0

        # Execute step from plan
        if self.current_plan and self.plan_step_idx < len(self.current_plan):
            act = self.current_plan[self.plan_step_idx]
            self.plan_step_idx += 1
            if act in available_actions:
                self.explorer.last_action = act
                self.explorer.last_grid = grid_2d.copy()

                act_data = None
                if act == 5:
                    act_data = {'intentional_wait': True}
                elif act in (1, 2, 3, 4) and avatar_pos is not None:
                    dr, dc = DIR_MAP[act]
                    nr, nc = avatar_pos[0] + dr * stride, avatar_pos[1] + dc * stride
                    is_avatar_blocked = (nr, nc) in static_blocked or not (0 <= nr < H and 0 <= nc < W)
                    is_box_blocked = False
                    if box_offset is not None:
                        bnr, bnc = nr + box_offset[0], nc + box_offset[1]
                        is_box_blocked = (bnr, bnc) in static_blocked or not (0 <= bnr < H and 0 <= bnc < W)
                    if is_avatar_blocked or is_box_blocked:
                        act_data = {'intentional_wait': True}

                return (act, act_data)

        # Fallback to standard explorer
        return self.explorer.step(grid_2d, available_actions)

    def is_terminated(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        levels_completed: int,
        prev_levels_completed: int
    ) -> Tuple[bool, SubgoalStatus, str]:
        if levels_completed > prev_levels_completed:
            return (True, SubgoalStatus.COMPLETED, "Level advanced during spacetime navigation")

        avatar_pos = self.explorer.avatar_pos
        target_pos = subgoal.target_pos or (subgoal.metadata.get('target_pos') if subgoal.metadata else None)
        stride = getattr(self.explorer, 'stride', 1)

        if avatar_pos is not None and target_pos is not None:
            dist = abs(avatar_pos[0] - target_pos[0]) + abs(avatar_pos[1] - target_pos[1])
            meta = subgoal.metadata or {}
            if meta.get('approach_only'):
                if dist <= stride:
                    return (True, SubgoalStatus.COMPLETED, f"Safely approached target {target_pos}")
            elif dist == 0 or (dist <= stride and target_pos in self.explorer.obstacle_cells):
                return (True, SubgoalStatus.COMPLETED, f"Safely reached target {target_pos}")

        if subgoal.step_count >= subgoal.max_steps:
            return (True, SubgoalStatus.TIMEOUT, f"SpaceTime navigation exceeded max steps {subgoal.max_steps}")

        if self.consecutive_no_progress >= 16:
            return (True, SubgoalStatus.FAILED, "No progress for 16 consecutive steps in spacetime nav")

        return (False, SubgoalStatus.ACTIVE, "Navigating spacetime trajectory")


class GravityPlatformOption(BaseOption):
    """
    Formal SMDP Option: ω = < I_plat, π_plat, β_plat >
    Ballistic Platformer Navigation under Directional Gravity.
    
    Plans and executes platformer locomotion:
    - Horizontal walks on solid platforms
    - Controlled ledge drops onto lower tiers
    - Parabolic jump arcs over chasms and hazards
    - Ledge climbs to elevated platforms
    """
    def __init__(
        self,
        explorer: Optional[DomainAgnosticFrontierExplorer] = None,
        sandbox: Optional[MentalSandbox] = None,
        physics_inducer: Optional[LocalPhysicsInducer] = None,
        dynamic_tracker: Optional[DynamicEntityTracker] = None
    ):
        super().__init__(option_id="opt_platform", name="GravityPlatform")
        self.explorer = explorer if explorer is not None else DomainAgnosticFrontierExplorer()
        self.sandbox = sandbox if sandbox is not None else MentalSandbox()
        self.physics_inducer = physics_inducer or getattr(self.sandbox, 'physics_inducer', LocalPhysicsInducer())
        self.dynamic_tracker = dynamic_tracker

        self.current_plan: List[int] = []
        self.plan_step_idx: int = 0
        self.consecutive_no_progress: int = 0
        self.last_dist: Optional[float] = None
        self.target_pos: Optional[Tuple[int, int]] = None
        self.replan_interval: int = 6
        self.steps_since_replan: int = 0

    def reset(self):
        self.current_plan = []
        self.plan_step_idx = 0
        self.consecutive_no_progress = 0
        self.last_dist = None
        self.target_pos = None
        self.steps_since_replan = 0

    def can_initiate(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int],
        affordance: Optional[AffordanceFeatures] = None
    ) -> bool:
        # Must have lateral movement
        if not (3 in available_actions or 4 in available_actions):
            return False

        tgt = subgoal.target_pos or (subgoal.metadata.get('target_pos') if subgoal.metadata else None)
        if tgt is None:
            return False

        if subgoal.subgoal_type == SubgoalType.GRAVITY_PLATFORM:
            return True
        if subgoal.metadata and (subgoal.metadata.get('has_gravity') or subgoal.metadata.get('use_platformer')):
            return True
        if self.physics_inducer.gravity_vector is not None:
            return True
        if affordance is not None and (affordance.has_1d_nav > 0 or affordance.has_gravity_platforms > 0 or affordance.gravity_flux > 0):
            return True

        return False

    def step(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        subgoal.step_count += 1
        self.steps_since_replan += 1

        grid_2d = np.asarray(grid)
        while grid_2d.ndim > 2:
            grid_2d = grid_2d[-1]
        H, W = grid_2d.shape
        bg_canvas = int(collections.Counter(grid_2d.flatten()).most_common(1)[0][0])
        stride = getattr(self.explorer, 'stride', 1)

        self.explorer._update_avatar(grid_2d, bg_canvas)
        avatar_pos = self.explorer.avatar_pos

        target_pos = subgoal.target_pos or (subgoal.metadata.get('target_pos') if subgoal.metadata else None)
        self.target_pos = target_pos

        if avatar_pos is None or target_pos is None:
            return self.explorer.step(grid_2d, available_actions)

        # Distance progress tracking
        dist = abs(avatar_pos[0] - target_pos[0]) + abs(avatar_pos[1] - target_pos[1])
        if self.last_dist is not None and dist >= self.last_dist:
            self.consecutive_no_progress += 1
        else:
            self.consecutive_no_progress = 0
        self.last_dist = dist

        # Replan if plan exhausted or replan interval reached
        need_replan = (
            len(self.current_plan) == 0 or
            self.plan_step_idx >= len(self.current_plan) or
            self.steps_since_replan >= self.replan_interval
        )

        gravity = self.physics_inducer.gravity_vector or (1, 0)
        solids = set(self.physics_inducer.solid_colors) | set(self.explorer.obstacle_cells)
        hazards = set(self.physics_inducer.hazard_colors)
        suspect_hazards = self.physics_inducer.get_suspect_hazard_colors(grid_2d, bg_canvas) if hasattr(self.physics_inducer, 'get_suspect_hazard_colors') else set()
        gravity_actuators = self.physics_inducer.get_gravity_actuators() if hasattr(self.physics_inducer, 'get_gravity_actuators') else []
        if subgoal.metadata and 'gravity_actuators' in subgoal.metadata:
            gravity_actuators = list(gravity_actuators) + list(subgoal.metadata['gravity_actuators'])
        carrier_platforms = getattr(subgoal, 'carrier_platforms', None) or (subgoal.metadata.get('carrier_platforms') if subgoal.metadata else None)

        if need_replan:
            self.steps_since_replan = 0
            reservations = None
            if self.dynamic_tracker is not None:
                reservations = self.dynamic_tracker.predict_spacetime_reservations(
                    grid=grid_2d,
                    horizon=40,
                    static_blocked=solids,
                    stride=stride
                )
            plan = platformer_a_star(
                grid=grid_2d,
                start_pos=avatar_pos,
                goal_candidates={target_pos},
                gravity=gravity,
                jump_height=self.physics_inducer.jump_height,
                jump_reach=2,
                solid_colors=solids,
                hazard_colors=hazards,
                suspect_hazard_colors=suspect_hazards,
                bg_canvas=bg_canvas,
                available_actions=available_actions,
                stride=stride,
                gravity_actuators=gravity_actuators,
                carrier_platforms=carrier_platforms,
                reservations=reservations
            )
            if plan is not None:
                self.current_plan = plan
                self.plan_step_idx = 0
            else:
                self.current_plan = []
                self.plan_step_idx = 0

        # Execute next action from plan
        if self.current_plan and self.plan_step_idx < len(self.current_plan):
            act = self.current_plan[self.plan_step_idx]
            self.plan_step_idx += 1
            if act in available_actions:
                self.explorer.last_action = act
                self.explorer.last_grid = grid_2d.copy()
                return (act, None)

        return self.explorer.step(grid_2d, available_actions)

    def is_terminated(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        levels_completed: int,
        prev_levels_completed: int
    ) -> Tuple[bool, SubgoalStatus, str]:
        if levels_completed > prev_levels_completed:
            return (True, SubgoalStatus.COMPLETED, "Level advanced during platform navigation")

        avatar_pos = self.explorer.avatar_pos
        target_pos = subgoal.target_pos or (subgoal.metadata.get('target_pos') if subgoal.metadata else None)
        stride = getattr(self.explorer, 'stride', 1)

        if avatar_pos is not None and target_pos is not None:
            dist = abs(avatar_pos[0] - target_pos[0]) + abs(avatar_pos[1] - target_pos[1])
            if dist == 0 or (dist <= stride and target_pos in self.explorer.obstacle_cells):
                return (True, SubgoalStatus.COMPLETED, f"Safely reached platform target {target_pos}")

        if subgoal.step_count >= subgoal.max_steps:
            return (True, SubgoalStatus.TIMEOUT, f"Platform navigation exceeded max steps {subgoal.max_steps}")

        if self.consecutive_no_progress >= 16:
            return (True, SubgoalStatus.FAILED, "No progress for 16 consecutive steps on platform")

        return (False, SubgoalStatus.ACTIVE, "Navigating platformer trajectory")


class InteractEntityOption(BaseOption):
    """
    Option to execute an interaction (Action 5 or Action 6) on an adjacent entity.
    Termination: Interaction executed or state change observed.
    """
    def __init__(self, explorer: Optional[DomainAgnosticFrontierExplorer] = None):
        super().__init__(option_id="opt_interact", name="InteractEntity")
        self.explorer = explorer
        self.has_acted = False

    def can_initiate(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int],
        affordance: Optional[AffordanceFeatures] = None
    ) -> bool:
        if 6 in available_actions and subgoal.target_pos is not None:
            return True
        if 5 in available_actions:
            if subgoal.target_pos is None:
                return True
            if self.explorer is not None and self.explorer.avatar_pos is not None:
                ax, ay = self.explorer.avatar_pos
                tx, ty = subgoal.target_pos
                s = self.explorer.stride or 1
                dist = abs(ax - tx) + abs(ay - ty)
                return dist <= s * 2
            return True
        return False

    def step(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        subgoal.step_count += 1
        self.has_acted = True
        if 5 in available_actions:
            return (5, None)
        elif 6 in available_actions and subgoal.target_pos is not None:
            if self.explorer is not None:
                self.explorer.avatar_pos = (int(subgoal.target_pos[0]), int(subgoal.target_pos[1]))
                self.explorer.avatar_confidence = 1.0
                self.explorer.last_action = 6
                self.explorer.last_grid = np.array(grid).copy()
            return (6, {'x': int(subgoal.target_pos[1]), 'y': int(subgoal.target_pos[0])})
        return (available_actions[0], None)

    def is_terminated(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        levels_completed: int,
        prev_levels_completed: int
    ) -> Tuple[bool, SubgoalStatus, str]:
        if self.has_acted:
            return (True, SubgoalStatus.COMPLETED, "Interaction action dispatched")
        if subgoal.step_count >= 2:
            return (True, SubgoalStatus.TIMEOUT, "Interact timed out")
        return (False, SubgoalStatus.ACTIVE, "Awaiting interaction")

    def reset(self):
        self.has_acted = False


class SwitchAvatarOption(BaseOption):
    """
    Formal SMDP Option: ω = < I_switch, π_switch, β_switch >
    Transfers active control to a different selectable entity (avatar/puck)
    via Action 6 point-and-click.
    """
    def __init__(self, explorer: Optional[DomainAgnosticFrontierExplorer] = None):
        super().__init__(option_id="opt_switch_avatar", name="SwitchAvatar")
        self.explorer = explorer
        self.has_acted = False

    def can_initiate(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int],
        affordance: Optional[AffordanceFeatures] = None
    ) -> bool:
        if 6 not in available_actions or subgoal.target_pos is None:
            return False
        return True

    def _find_target_pos(self, grid: np.ndarray, subgoal: Subgoal) -> Tuple[int, int]:
        """Locates current position of the target entity to click."""
        if subgoal.target_pos is None:
            return (0, 0)
        tr, tc = int(subgoal.target_pos[0]), int(subgoal.target_pos[1])
        target_color = subgoal.target_color or (subgoal.metadata.get('block_color') if subgoal.metadata else None)
        if target_color is not None:
            coords = np.argwhere(grid == target_color)
            if len(coords) > 0:
                coords_set = set((int(r), int(c)) for r, c in coords)
                visited = set()
                clusters = []
                for pt in coords_set:
                    if pt not in visited:
                        comp = []
                        q = [pt]
                        visited.add(pt)
                        while q:
                            cr, cc = q.pop(0)
                            comp.append((cr, cc))
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nbr = (cr + dr, cc + dc)
                                if nbr in coords_set and nbr not in visited:
                                    visited.add(nbr)
                                    q.append(nbr)
                        arr = np.array(comp)
                        clusters.append((int(np.round(np.mean(arr[:, 0]))), int(np.round(np.mean(arr[:, 1])))))
                if clusters:
                    return min(clusters, key=lambda cl: abs(cl[0] - tr) + abs(cl[1] - tc))
        return (tr, tc)

    def step(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        subgoal.step_count += 1
        self.has_acted = True
        target_r, target_c = self._find_target_pos(grid, subgoal)
        if self.explorer is not None:
            self.explorer.avatar_pos = (target_r, target_c)
            self.explorer.avatar_confidence = 1.0
            self.explorer.last_action = 6
            self.explorer.last_grid = np.array(grid).copy()
            bg_canvas = int(collections.Counter(grid.flatten()).most_common(1)[0][0])
            local_patch = grid[max(0, target_r - 1):target_r + 2, max(0, target_c - 1):target_c + 2]
            non_floor = [int(c) for c in local_patch.flatten() if c not in self.explorer.floor_colors and c != bg_canvas and c != 0]
            if self.explorer.avatar_color is not None and self.explorer.avatar_color in non_floor:
                pass # Preserve established confirmed avatar color
            elif non_floor:
                self.explorer.avatar_color = collections.Counter(non_floor).most_common(1)[0][0]
            elif 0 <= target_r < grid.shape[0] and 0 <= target_c < grid.shape[1]:
                self.explorer.avatar_color = int(grid[target_r, target_c])
            self.explorer.visited_cells.add((target_r, target_c))
        return (6, {'x': target_c, 'y': target_r})

    def is_terminated(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        levels_completed: int,
        prev_levels_completed: int
    ) -> Tuple[bool, SubgoalStatus, str]:
        if self.has_acted:
            return (True, SubgoalStatus.COMPLETED, "Avatar switch action dispatched")
        if subgoal.step_count >= 2:
            return (True, SubgoalStatus.TIMEOUT, "Avatar switch timed out")
        return (False, SubgoalStatus.ACTIVE, "Awaiting avatar switch")

    def reset(self):
        self.has_acted = False


class FrontierExploreOption(BaseOption):
    """
    Option to explore unknown boundaries and frontier clusters.
    Termination: Unvisited cell reached or state entropy increases.
    """
    def __init__(self, explorer: Optional[DomainAgnosticFrontierExplorer] = None):
        super().__init__(option_id="opt_frontier", name="FrontierExplore")
        self.explorer = explorer if explorer is not None else DomainAgnosticFrontierExplorer()
        self.visited_before = 0

    def can_initiate(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int],
        affordance: Optional[AffordanceFeatures] = None
    ) -> bool:
        return any(a in [1, 2, 3, 4] for a in available_actions)

    def step(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        if subgoal.step_count == 0:
            self.visited_before = len(self.explorer.visited_cells)
        subgoal.step_count += 1
        return self.explorer.step(grid, available_actions)

    def is_terminated(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        levels_completed: int,
        prev_levels_completed: int
    ) -> Tuple[bool, SubgoalStatus, str]:
        if levels_completed > prev_levels_completed:
            return (True, SubgoalStatus.COMPLETED, "Level advanced during frontier exploration")

        # Terminate if new unique cells have been explored
        new_visited = len(self.explorer.visited_cells) - self.visited_before
        if new_visited >= 3:
            return (True, SubgoalStatus.COMPLETED, f"Explored {new_visited} novel frontier cells")

        if subgoal.step_count >= subgoal.max_steps:
            return (True, SubgoalStatus.TIMEOUT, f"Frontier exploration reached max steps {subgoal.max_steps}")

        return (False, SubgoalStatus.ACTIVE, "Exploring frontier")

    def reset(self):
        self.visited_before = 0


class DeadlockResetOption(BaseOption):
    """
    Option to execute Action 7 (RESET) to clear deadlocks when stuck.
    Termination: Board reset executed.
    """
    def __init__(self):
        super().__init__(option_id="opt_reset", name="DeadlockReset")
        self.executed = False

    def can_initiate(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int],
        affordance: Optional[AffordanceFeatures] = None
    ) -> bool:
        return 7 in available_actions

    def step(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        subgoal.step_count += 1
        self.executed = True
        return (7, None)

    def is_terminated(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        levels_completed: int,
        prev_levels_completed: int
    ) -> Tuple[bool, SubgoalStatus, str]:
        if self.executed:
            return (True, SubgoalStatus.COMPLETED, "Deadlock reset executed")
        return (False, SubgoalStatus.ACTIVE, "Executing reset")

    def reset(self):
        self.executed = False


def gaussian_elimination_modulo_k(
    A: np.ndarray,
    b: np.ndarray,
    k: int = 2
) -> Optional[np.ndarray]:
    """
    Solves the modular linear system A x ≡ b (mod k) over finite field / modular ring Z_k.
    
    Args:
        A: (N, M) matrix of causal actuator coefficients in Z_k
        b: (N,) or (N, 1) vector of target state differences in Z_k
        k: Modulus (typically 2 for binary toggles, 3 for 3-state cycles)
        
    Returns:
        x: (M,) integer solution vector in {0, ..., k-1} minimizing total clicks sum(x),
           or None if the linear system is inconsistent.
    """
    A = np.asarray(A, dtype=int) % k
    b = np.asarray(b, dtype=int).flatten() % k
    n_rows, n_cols = A.shape
    M = np.hstack([A, b.reshape(-1, 1)]) % k

    row = 0
    pivots = []
    pivot_cols = set()

    for col in range(n_cols):
        p = None
        # Prefer a pivot coprime to k (guaranteed invertible)
        for r in range(row, n_rows):
            val = int(M[r, col] % k)
            if val != 0:
                import math
                if math.gcd(val, k) == 1:
                    p = r
                    break
                elif p is None:
                    p = r
        if p is None:
            continue

        M[[row, p]] = M[[p, row]]
        pivot_val = int(M[row, col])
        try:
            inv = pow(pivot_val, -1, k)
        except ValueError:
            continue

        M[row] = (M[row] * inv) % k
        for r in range(n_rows):
            if r != row and M[r, col] % k != 0:
                factor = M[r, col]
                M[r] = (M[r] - factor * M[row]) % k
        pivots.append((row, col))
        pivot_cols.add(col)
        row += 1

    # Inconsistency check: row has all zeros in A but non-zero target constant
    for r in range(row, n_rows):
        if M[r, -1] % k != 0:
            return None

    free_cols = [c for c in range(n_cols) if c not in pivot_cols]

    if not free_cols:
        x_sol = np.zeros(n_cols, dtype=int)
        for r, col in reversed(pivots):
            x_sol[col] = (M[r, -1] - np.dot(M[r, :n_cols], x_sol)) % k
        return x_sol

    # Multiple solutions exist due to free variables (nullspace > 0):
    # Search over free variables to minimize total actuator activations sum(x)
    import itertools
    best_x = None
    min_clicks = float('inf')
    max_search = min(len(free_cols), 6)  # Bound search to k^6 candidates

    for free_vals in itertools.product(range(k), repeat=max_search):
        x_cand = np.zeros(n_cols, dtype=int)
        for i, val in enumerate(free_vals):
            x_cand[free_cols[i]] = val
        for r, col in reversed(pivots):
            x_cand[col] = (M[r, -1] - np.dot(M[r, :n_cols], x_cand)) % k

        clicks = int(np.sum(x_cand))
        if clicks < min_clicks:
            min_clicks = clicks
            best_x = x_cand

    return best_x


class ToggleSolveOption(BaseOption):
    """
    Formal SMDP Option for Discrete Actuator / Galois Field Linear Systems:
    ω_toggle = < I_toggle, π_toggle, β_toggle >
    
    Operates 100% domain-agnostically:
    1. Actuator Discovery: Identifies clickable buttons, switches, or toggle lattice cells
       from Subgoal metadata or morphological grid analysis.
    2. Online System Identification (Active Probing):
       If effect matrix A is unknown, probes actuators online. On each transition,
       calculates Δs = s_{t+1} ⊖ s_t (mod k) to populate column A_{:, j} and induce stencils.
    3. Galois Field Solver: Solves A x ≡ b (mod k) via Gaussian Elimination over Z_k.
    4. Intra-Option Policy π_toggle: Dispatches Action 6 clicks or Action 5 interactions.
    5. Termination Condition β_toggle: Target reached, level advanced, plan finished, or timeout.
    """
    def __init__(
        self,
        explorer: Optional[DomainAgnosticFrontierExplorer] = None,
        sandbox: Optional[MentalSandbox] = None,
        physics_inducer: Optional[LocalPhysicsInducer] = None
    ):
        super().__init__(option_id="opt_toggle_solve", name="ToggleSolve")
        self.explorer = explorer
        self.sandbox = sandbox
        self.physics_inducer = physics_inducer

        # Actuators and controlled cells
        self.actuators: List[Tuple[int, int]] = []
        self.controlled_cells: List[Tuple[int, int]] = []
        self.actuator_to_idx: Dict[Tuple[int, int], int] = {}
        self.cell_to_idx: Dict[Tuple[int, int], int] = {}

        # Linear system parameters
        self.matrix: Optional[np.ndarray] = None
        self.modulus: int = 2
        self.target_state: Optional[np.ndarray] = None
        self.color_map: Dict[int, int] = {}
        self.rev_color_map: Dict[int, int] = {}

        # Probing & Online System Identification
        self.probed_actuators: Set[int] = set()
        self.last_clicked_actuator_idx: Optional[int] = None
        self.grid_before_action: Optional[np.ndarray] = None
        self.inferred_stencil: Optional[List[Tuple[int, int, int]]] = None

        # Plan queue & execution tracking
        self.planned_actions: collections.deque[Tuple[int, int]] = collections.deque()
        self.clicks_executed: int = 0
        self.consecutive_no_change: int = 0
        self.last_state_vec: Optional[np.ndarray] = None

    def reset(self):
        """Resets option internal state between episodes or subgoals."""
        self.actuators.clear()
        self.controlled_cells.clear()
        self.actuator_to_idx.clear()
        self.cell_to_idx.clear()
        self.matrix = None
        self.modulus = 2
        self.target_state = None
        self.color_map.clear()
        self.rev_color_map.clear()
        self.probed_actuators.clear()
        self.last_clicked_actuator_idx = None
        self.grid_before_action = None
        self.inferred_stencil = None
        self.planned_actions.clear()
        self.clicks_executed = 0
        self.consecutive_no_change = 0
        self.last_state_vec = None

    def _detect_actuators_and_cells(self, grid: np.ndarray, subgoal: Subgoal):
        """Initializes actuators, controlled cells, color mapping, and effect matrix."""
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        h, w = grid.shape
        bg_canvas = int(collections.Counter(grid.flatten()).most_common(1)[0][0])

        meta = subgoal.metadata or {}
        self.modulus = int(meta.get('modulus', 2))

        # 1. Actuators
        if 'actuators' in meta and meta['actuators']:
            self.actuators = [tuple(pt) for pt in meta['actuators']]
        elif 'switches' in meta and meta['switches']:
            self.actuators = [tuple(sw.center if hasattr(sw, 'center') else sw) for sw in meta['switches']]
        elif subgoal.target_pos is not None:
            self.actuators = [subgoal.target_pos]
        else:
            self.actuators = []

        # 2. Controlled Cells
        if 'controlled_cells' in meta and meta['controlled_cells']:
            self.controlled_cells = [tuple(pt) for pt in meta['controlled_cells']]
        elif self.actuators:
            self.controlled_cells = list(self.actuators)
        else:
            self.controlled_cells = []

        # Fallback actuator discovery from grid if empty
        if not self.actuators and subgoal.target_pos is not None:
            self.actuators = [subgoal.target_pos]
            self.controlled_cells = [subgoal.target_pos]

        # Ensure consistent order
        self.actuators.sort(key=lambda p: (p[0], p[1]))
        self.controlled_cells.sort(key=lambda p: (p[0], p[1]))
        self.actuator_to_idx = {pt: i for i, pt in enumerate(self.actuators)}
        self.cell_to_idx = {pt: i for i, pt in enumerate(self.controlled_cells)}

        # 3. Color Mapping
        if 'color_map' in meta and meta['color_map']:
            self.color_map = {int(k): int(v) for k, v in meta['color_map'].items()}
            self.rev_color_map = {v: k for k, v in self.color_map.items()}
        else:
            cell_vals = [int(grid[r, c]) for r, c in self.controlled_cells if 0 <= r < h and 0 <= c < w]
            counts = collections.Counter(cell_vals).most_common()
            distinct_colors = [c for c, _ in counts]
            if not distinct_colors:
                distinct_colors = [0, 1]
            elif len(distinct_colors) == 1:
                other_col = (distinct_colors[0] + 1) % 10
                distinct_colors.append(other_col)

            target_col = subgoal.target_color
            if target_col is not None and target_col in distinct_colors:
                distinct_colors.remove(target_col)
                distinct_colors.insert(0, target_col)
            elif bg_canvas in distinct_colors:
                distinct_colors.remove(bg_canvas)
                distinct_colors.insert(0, bg_canvas)

            self.color_map = {c: idx % self.modulus for idx, c in enumerate(distinct_colors)}
            self.rev_color_map = {idx % self.modulus: c for idx, c in enumerate(distinct_colors)}

        # 4. Target State
        n_cells = len(self.controlled_cells)
        if 'target_state' in meta and meta['target_state'] is not None:
            self.target_state = np.asarray(meta['target_state'], dtype=int) % self.modulus
        else:
            self.target_state = np.zeros(n_cells, dtype=int)

        # 5. Matrix initialization
        n_actuators = len(self.actuators)
        if 'effect_matrix' in meta and meta['effect_matrix'] is not None:
            self.matrix = np.asarray(meta['effect_matrix'], dtype=int) % self.modulus
            self.probed_actuators = set(range(n_actuators))
        else:
            self.matrix = np.zeros((n_cells, n_actuators), dtype=int)
            # Check if LocalPhysicsInducer already recorded click rewrites
            if self.physics_inducer is not None and self.physics_inducer.click_rewrites:
                for j, act_pos in enumerate(self.actuators):
                    click_pt = (act_pos[1], act_pos[0])
                    if click_pt in self.physics_inducer.click_rewrites:
                        diffs = self.physics_inducer.click_rewrites[click_pt]
                        for r, c, old_v, new_v in diffs:
                            if (r, c) in self.cell_to_idx:
                                i_cell = self.cell_to_idx[(r, c)]
                                old_s = self.color_map.get(old_v, 0)
                                new_s = self.color_map.get(new_v, 1)
                                self.matrix[i_cell, j] = (new_s - old_s) % self.modulus
                        self.probed_actuators.add(j)

    def _get_current_state_vector(self, grid: np.ndarray) -> np.ndarray:
        """Extracts the state vector in Z_k for all controlled cells."""
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        h, w = grid.shape
        states = np.zeros(len(self.controlled_cells), dtype=int)
        for i, (r, c) in enumerate(self.controlled_cells):
            if 0 <= r < h and 0 <= c < w:
                val = int(grid[r, c])
                states[i] = self.color_map.get(val, 1 if val != 0 else 0) % self.modulus
        return states

    def _observe_last_transition(self, grid: np.ndarray):
        """Observes the physical differential effect of the previous click on the controlled cells."""
        if self.last_clicked_actuator_idx is None or self.grid_before_action is None:
            return

        prev_s = self._get_current_state_vector(self.grid_before_action)
        curr_s = self._get_current_state_vector(grid)
        delta_s = (curr_s - prev_s) % self.modulus

        j = self.last_clicked_actuator_idx
        if self.matrix is not None and j < self.matrix.shape[1]:
            if np.any(delta_s != 0):
                self.matrix[:, j] = delta_s
                self.probed_actuators.add(j)

                # Induce translational stencil across regular lattices
                if self.inferred_stencil is None and len(self.actuators) > 1:
                    a_pos = self.actuators[j]
                    offsets = []
                    for i, d_val in enumerate(delta_s):
                        if d_val != 0:
                            c_pos = self.controlled_cells[i]
                            offsets.append((c_pos[0] - a_pos[0], c_pos[1] - a_pos[1], int(d_val)))
                    if offsets:
                        self.inferred_stencil = offsets
                        for j_other, a_other in enumerate(self.actuators):
                            if j_other not in self.probed_actuators:
                                cand_col = np.zeros(len(self.controlled_cells), dtype=int)
                                for dr, dc, d_val in offsets:
                                    target_cell = (a_other[0] + dr, a_other[1] + dc)
                                    if target_cell in self.cell_to_idx:
                                        cand_col[self.cell_to_idx[target_cell]] = d_val
                                self.matrix[:, j_other] = cand_col

        self.last_clicked_actuator_idx = None
        self.grid_before_action = None

    def can_initiate(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int],
        affordance: Optional[AffordanceFeatures] = None
    ) -> bool:
        if 6 not in available_actions and 5 not in available_actions:
            return False

        if subgoal.subgoal_type == SubgoalType.TOGGLE_SOLVE:
            return True

        meta = subgoal.metadata or {}
        if meta.get('is_toggle') or meta.get('actuators') or meta.get('switches'):
            return True

        if subgoal.subgoal_type == SubgoalType.INTERACT_ENTITY and subgoal.target_pos is not None:
            return True

        return False

    def step(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        subgoal.step_count += 1
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]

        # Initialize entities & system representation if needed
        if self.matrix is None or len(self.actuators) == 0:
            self._detect_actuators_and_cells(grid, subgoal)

        # Observe effect of previous action
        self._observe_last_transition(grid)

        curr_s = self._get_current_state_vector(grid)
        if self.last_state_vec is not None and np.array_equal(curr_s, self.last_state_vec):
            self.consecutive_no_change += 1
        else:
            self.consecutive_no_change = 0
        self.last_state_vec = curr_s.copy()

        # Check if already solved
        if self.target_state is not None and np.array_equal(curr_s, self.target_state):
            return (available_actions[0], None)

        # If execution queue is empty, attempt solve or continue probing
        if not self.planned_actions and self.matrix is not None and self.target_state is not None:
            b = (self.target_state - curr_s) % self.modulus
            if np.any(b != 0):
                # Try solving linear system over Z_k
                x_sol = gaussian_elimination_modulo_k(self.matrix, b, k=self.modulus)
                if x_sol is not None and np.sum(x_sol) > 0:
                    for j_idx, count in enumerate(x_sol):
                        for _ in range(int(count)):
                            self.planned_actions.append(self.actuators[j_idx])

            # If still empty (unsolvable with current matrix or matrix incomplete), probe unprobed actuators
            if not self.planned_actions:
                unprobed = [j for j in range(len(self.actuators)) if j not in self.probed_actuators]
                if unprobed:
                    probe_j = unprobed[0]
                    self.planned_actions.append(self.actuators[probe_j])

        # Execute next action from plan
        if self.planned_actions:
            act_pos = self.planned_actions.popleft()
            if act_pos in self.actuator_to_idx:
                self.last_clicked_actuator_idx = self.actuator_to_idx[act_pos]
            self.grid_before_action = grid.copy()
            self.clicks_executed += 1

            if 6 in available_actions:
                if self.explorer is not None:
                    self.explorer.last_action = 6
                    self.explorer.last_grid = grid.copy()
                return (6, {'x': int(act_pos[1]), 'y': int(act_pos[0])})
            elif 5 in available_actions:
                return (5, None)

        # Fallback
        if self.explorer is not None:
            return self.explorer.step(grid, available_actions)
        return (available_actions[0], None)

    def is_terminated(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        levels_completed: int,
        prev_levels_completed: int
    ) -> Tuple[bool, SubgoalStatus, str]:
        if levels_completed > prev_levels_completed:
            return (True, SubgoalStatus.COMPLETED, "Level advanced during toggle solve")

        if self.target_state is not None:
            curr_s = self._get_current_state_vector(grid)
            if np.array_equal(curr_s, self.target_state):
                return (True, SubgoalStatus.COMPLETED, "Target state reached across all toggles")

        if subgoal.step_count >= subgoal.max_steps:
            return (True, SubgoalStatus.TIMEOUT, f"Toggle solve reached max steps {subgoal.max_steps}")

        if self.clicks_executed > 0 and not self.planned_actions and len(self.probed_actuators) == len(self.actuators):
            if self.target_state is not None:
                curr_s = self._get_current_state_vector(grid)
                if np.array_equal(curr_s, self.target_state):
                    return (True, SubgoalStatus.COMPLETED, "Target state reached after plan completion")
                else:
                    return (True, SubgoalStatus.FAILED, "Planned clicks executed but target state not reached")

        if self.consecutive_no_change >= 8:
            return (True, SubgoalStatus.FAILED, "No state change detected for 8 consecutive steps")

        return (False, SubgoalStatus.ACTIVE, "Executing toggle solve policy")


class ColorDipOption(BaseOption):
    """
    Formal SMDP Option for State Transformation & Palette/Swatch Affordances:
    ω_dip = < I_dip, π_dip, β_dip >
    
    Operates 100% domain-agnostically:
    1. Swatch / Palette Localization:
       Identifies target color swatches in the grid (either spatial floor tiles,
       paint dispensers, or click palette buttons).
    2. Path Planning & Navigation / Palette Interaction:
       - Spatial Swatch: Computes BFS path from avatar to swatch position.
         Navigates avatar to step directly onto the swatch (or adjacent if Action 5 interact is needed).
       - Click Palette: If Action 6 is available and swatch is in a HUD/palette bar, dispatches Action 6 click.
    3. Property Transformation Detection:
       Detects when avatar has absorbed the target color (avatar_color == target_color
       or grid at avatar position equals target_color).
    4. Termination Condition β_dip:
       Terminates with COMPLETED upon acquiring target color, or TIMEOUT / FAILED on stagnation.
    """
    def __init__(
        self,
        explorer: Optional[DomainAgnosticFrontierExplorer] = None,
        sandbox: Optional[MentalSandbox] = None,
        physics_inducer: Optional[LocalPhysicsInducer] = None
    ):
        super().__init__(option_id="opt_color_dip", name="ColorDip")
        self.explorer = explorer if explorer is not None else DomainAgnosticFrontierExplorer()
        self.sandbox = sandbox if sandbox is not None else MentalSandbox()
        self.physics_inducer = physics_inducer

        self.target_color: Optional[int] = None
        self.swatch_pos: Optional[Tuple[int, int]] = None
        self.last_dist: Optional[int] = None
        self.consecutive_no_progress: int = 0
        self.visited_swatch: bool = False
        self.interacted_on_swatch: bool = False
        self.has_clicked_palette: bool = False

    def reset(self):
        """Resets option state between subgoals/episodes."""
        self.target_color = None
        self.swatch_pos = None
        self.last_dist = None
        self.consecutive_no_progress = 0
        self.visited_swatch = False
        self.interacted_on_swatch = False
        self.has_clicked_palette = False

    def can_initiate(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int],
        affordance: Optional[AffordanceFeatures] = None
    ) -> bool:
        if subgoal.subgoal_type == SubgoalType.COLOR_DIP:
            return True

        meta = subgoal.metadata or {}
        if meta.get('is_swatch') or meta.get('requires_color') or meta.get('action') == 'dip':
            return True

        return False

    def _locate_swatch(self, grid: np.ndarray, subgoal: Subgoal, bg_canvas: int) -> Optional[Tuple[int, int]]:
        """Finds coordinate of target color swatch."""
        if subgoal.target_pos is not None:
            return subgoal.target_pos

        meta = subgoal.metadata or {}
        if 'swatch_pos' in meta and meta['swatch_pos'] is not None:
            return tuple(meta['swatch_pos'])

        target_col = subgoal.target_color or meta.get('target_color')
        if target_col is None:
            return None

        h, w = grid.shape
        coords = np.argwhere(grid == target_col)
        if len(coords) == 0:
            return None

        avatar_pos = self.explorer.avatar_pos
        if avatar_pos is not None:
            # Sort by Manhattan distance to avatar
            sorted_coords = sorted(coords, key=lambda p: abs(p[0] - avatar_pos[0]) + abs(p[1] - avatar_pos[1]))
            return (int(sorted_coords[0][0]), int(sorted_coords[0][1]))

        return (int(coords[0][0]), int(coords[0][1]))

    def _find_swatch_path(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        bg_canvas: int,
        target_color: Optional[int] = None
    ) -> Optional[List[Tuple[int, int]]]:
        """A* path from start to swatch cell (or adjacent cell)."""
        if start == goal:
            return [start]
        h, w = grid.shape
        stride = getattr(self.explorer, 'stride', 1)
        floor_colors = self.explorer.floor_colors

        def heur(p: Tuple[int, int]) -> float:
            return (abs(p[0] - goal[0]) + abs(p[1] - goal[1])) / float(stride)

        h0 = heur(start)
        pq: List[Tuple[float, float, int, List[Tuple[int, int]]]] = [(h0, 0.0, 0, [start])]
        visited: Dict[Tuple[int, int], float] = {start: 0.0}
        counter = 0

        while pq:
            f, g, _, path = heapq.heappop(pq)
            curr = path[-1]

            if curr == goal:
                return path

            if g > visited.get(curr, float('inf')):
                continue

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr * stride, curr[1] + dc * stride
                neighbor = (nr, nc)

                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if neighbor in self.explorer.obstacle_cells and neighbor != goal:
                    continue

                val = int(grid[nr, nc])
                is_trav = (
                    val in floor_colors or
                    (not floor_colors and val == bg_canvas) or
                    (target_color is not None and val == target_color) or
                    neighbor in self.explorer.traversable_cells or
                    neighbor == goal
                )
                if is_trav:
                    new_g = g + 1.0
                    if new_g < visited.get(neighbor, float('inf')):
                        visited[neighbor] = new_g
                        h_val = heur(neighbor)
                        counter += 1
                        heapq.heappush(pq, (new_g + h_val, new_g, counter, path + [neighbor]))

        return None

    def step(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        available_actions: List[int]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        subgoal.step_count += 1
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        h, w = grid.shape
        bg_canvas = int(collections.Counter(grid.flatten()).most_common(1)[0][0])

        self.target_color = subgoal.target_color or (subgoal.metadata.get('target_color') if subgoal.metadata else None)
        self.swatch_pos = self._locate_swatch(grid, subgoal, bg_canvas)

        avatar_pos = self.explorer.avatar_pos
        meta = subgoal.metadata or {}

        # 1. Click Palette (Action 6)
        if 6 in available_actions and self.swatch_pos is not None:
            # Check if swatch is in top/side palette bar or explicitly designated click
            if meta.get('action') == 'click' or meta.get('palette_click') or (self.swatch_pos[0] <= 4 and (avatar_pos is None or abs(self.swatch_pos[0] - avatar_pos[0]) > 6)):
                self.has_clicked_palette = True
                if self.target_color is not None and self.explorer is not None:
                    self.explorer.avatar_color = self.target_color
                if self.explorer is not None:
                    self.explorer.last_action = 6
                    self.explorer.last_grid = grid.copy()
                return (6, {'x': int(self.swatch_pos[1]), 'y': int(self.swatch_pos[0])})

        # 2. Spatial Navigation to Swatch
        if avatar_pos is not None and self.swatch_pos is not None:
            dist = abs(avatar_pos[0] - self.swatch_pos[0]) + abs(avatar_pos[1] - self.swatch_pos[1])

            # Track progress
            if self.last_dist is not None and dist >= self.last_dist:
                self.consecutive_no_progress += 1
            else:
                self.consecutive_no_progress = 0
            self.last_dist = dist

            # If avatar is on swatch:
            if dist == 0:
                self.visited_swatch = True
                if self.target_color is not None:
                    self.explorer.avatar_color = self.target_color
                if 5 in available_actions and not self.interacted_on_swatch:
                    self.interacted_on_swatch = True
                    return (5, None)
                return (available_actions[0], None)

            # Plan path directly to swatch
            path = self._find_swatch_path(grid, avatar_pos, self.swatch_pos, bg_canvas, target_color=self.target_color)
            if path and len(path) > 1:
                next_pos = path[1]
                act = self.explorer._direction_to_action(avatar_pos, next_pos)
                if act in available_actions:
                    self.explorer.last_action = act
                    self.explorer.last_grid = grid.copy()
                    return (act, None)

            # If avatar is adjacent to swatch
            stride = getattr(self.explorer, 'stride', 1)
            if dist <= stride:
                act = self.explorer._direction_to_action(avatar_pos, self.swatch_pos)
                if act in available_actions:
                    self.explorer.last_action = act
                    self.explorer.last_grid = grid.copy()
                    return (act, None)
                elif 5 in available_actions:
                    self.interacted_on_swatch = True
                    return (5, None)

        # Fallback to explorer step
        return self.explorer.step(grid, available_actions)

    def is_terminated(
        self,
        grid: np.ndarray,
        subgoal: Subgoal,
        levels_completed: int,
        prev_levels_completed: int
    ) -> Tuple[bool, SubgoalStatus, str]:
        if levels_completed > prev_levels_completed:
            return (True, SubgoalStatus.COMPLETED, "Level advanced during color dip")

        target_col = self.target_color or subgoal.target_color or (subgoal.metadata.get('target_color') if subgoal.metadata else None)

        # 1. Check if avatar has acquired target color
        if target_col is not None:
            if self.explorer.avatar_color == target_col:
                return (True, SubgoalStatus.COMPLETED, f"Avatar acquired target color {target_col}")

            avatar_pos = self.explorer.avatar_pos
            if avatar_pos is not None and 0 <= avatar_pos[0] < grid.shape[0] and 0 <= avatar_pos[1] < grid.shape[1]:
                cell_val = int(grid[avatar_pos[0], avatar_pos[1]])
                if cell_val == target_col:
                    self.explorer.avatar_color = target_col
                    return (True, SubgoalStatus.COMPLETED, f"Avatar on target color cell {target_col}")

        # 2. Check if avatar clicked palette
        if self.has_clicked_palette:
            if target_col is not None:
                self.explorer.avatar_color = target_col
            return (True, SubgoalStatus.COMPLETED, f"Clicked palette swatch for color {target_col}")

        # 3. Check if avatar visited swatch
        if self.visited_swatch:
            if target_col is not None:
                self.explorer.avatar_color = target_col
            return (True, SubgoalStatus.COMPLETED, f"Visited swatch at {self.swatch_pos}")

        if subgoal.step_count >= subgoal.max_steps:
            return (True, SubgoalStatus.TIMEOUT, f"Color dip reached max steps {subgoal.max_steps}")

        if self.consecutive_no_progress >= 8:
            return (True, SubgoalStatus.FAILED, "No progress towards swatch for 8 consecutive steps")

        return (False, SubgoalStatus.ACTIVE, "Navigating to color swatch")


class SubgoalGenerator:
    """
    Analyzes spatial topology, affordances, and latent teleological hypotheses
    to generate a prioritized queue of Subgoals.
    """
    def __init__(self, explorer: Optional[DomainAgnosticFrontierExplorer] = None):
        self.explorer = explorer if explorer is not None else DomainAgnosticFrontierExplorer()
        self.subgoal_counter = 0
        from .hypothesis_inducer import ObjectiveHypothesisInducer
        self.inducer = ObjectiveHypothesisInducer()
        from .causal_precondition_inducer import CausalPreconditionInducer
        self.precondition_inducer = CausalPreconditionInducer()

    def generate_subgoals(
        self,
        grid: np.ndarray,
        available_actions: List[int],
        affordance: Optional[AffordanceFeatures] = None
    ) -> List[Subgoal]:
        """
        Synthesizes candidate subgoals:
        1. Teleological Hypothesis Induction (Receptors, Goals, Switches, Collectibles).
        2. Causal Precondition Chaining (Bridging Chasms, Unlocking Barriers via DAGs).
        3. Salient entities (Keys, Targets, Swatches) -> Reach & Interact.
        4. Unvisited Frontier regions -> FrontierExplore.
        """
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        h, w = grid.shape

        bg_canvas = int(collections.Counter(grid.flatten()).most_common(1)[0][0])
        self.explorer._update_avatar(grid, bg_canvas)
        self.explorer._update_floor_colors(grid, bg_canvas)

        subgoals: List[Subgoal] = []

        # 1. Teleological Hypothesis Induction
        try:
            hypo = self.inducer.induce(
                grid=grid,
                available_actions=available_actions,
                avatar_pos=self.explorer.avatar_pos,
                floor_colors=self.explorer.floor_colors
            )
            if hypo.confidence >= 0.30:
                induced_sgs = self.inducer.synthesize_subgoals(
                    hypo, grid, available_actions, self.explorer.avatar_pos,
                    floor_colors=self.explorer.floor_colors
                )
                if induced_sgs:
                    subgoals.extend(induced_sgs)
        except Exception:
            pass

        # If teleological induction generated actionable subgoals, return them directly
        if subgoals:
            return subgoals

        # 2. Salient Entities Fallback & Causal Precondition Chaining
        salient = self.explorer._extract_salient_entities(grid, bg_canvas)
        unvisited = [
            e for e in salient
            if e['centroid'] not in self.explorer.visited_cells and
               e['centroid'] not in self.explorer.interacted_targets and
               (self.explorer.avatar_pos is None or e['centroid'] != self.explorer.avatar_pos)
        ]

        if unvisited:
            # Sort by rarity and distance
            if self.explorer.avatar_pos is not None:
                unvisited.sort(key=lambda e: (
                    e['size'],
                    abs(e['centroid'][0] - self.explorer.avatar_pos[0]) + abs(e['centroid'][1] - self.explorer.avatar_pos[1])
                ))

                # Check for Causal Precondition Chaining (Phase 220 / Hebel 1)
                primary_target = unvisited[0]['centroid']
                gravity = getattr(self.explorer, 'gravity_vector', (1, 0))
                solids = getattr(self.explorer, 'solid_colors', None)
                hazards = getattr(self.explorer, 'hazard_colors', None)
                suspects = getattr(self.explorer, 'suspect_hazard_colors', None)
                try:
                    bridge_plan = self.precondition_inducer.induce_chasm_bridge_plan(
                        grid=grid,
                        avatar_pos=self.explorer.avatar_pos,
                        target_pos=primary_target,
                        avatar_color=self.explorer.avatar_color,
                        gravity=gravity,
                        solid_colors=solids,
                        hazard_colors=hazards,
                        suspect_hazard_colors=suspects,
                        bg_canvas=bg_canvas,
                        floor_colors=self.explorer.floor_colors,
                        jump_height=getattr(self.explorer, 'jump_height', 2),
                        jump_reach=getattr(self.explorer, 'jump_reach', 2),
                        available_actions=available_actions,
                        stride=getattr(self.explorer, 'stride', 1)
                    )
                    if bridge_plan:
                        return bridge_plan
                except Exception:
                    pass

                try:
                    unlock_plan = self.precondition_inducer.induce_barrier_unlock_plan(
                        grid=grid,
                        avatar_pos=self.explorer.avatar_pos,
                        target_pos=primary_target,
                        bg_canvas=bg_canvas,
                        floor_colors=self.explorer.floor_colors,
                        solid_colors=solids,
                        available_actions=available_actions
                    )
                    if unlock_plan:
                        return unlock_plan
                except Exception:
                    pass

            for ent in unvisited[:3]:
                self.subgoal_counter += 1
                reach_sg = Subgoal(
                    subgoal_id=f"sg_reach_{self.subgoal_counter}",
                    subgoal_type=SubgoalType.REACH_ENTITY,
                    target_pos=ent['centroid'],
                    target_color=ent['color'],
                    max_steps=20
                )
                subgoals.append(reach_sg)

                # Pair with an Interact subgoal if Action 5/6 is available
                if 5 in available_actions or 6 in available_actions:
                    self.subgoal_counter += 1
                    interact_sg = Subgoal(
                        subgoal_id=f"sg_interact_{self.subgoal_counter}",
                        subgoal_type=SubgoalType.INTERACT_ENTITY,
                        target_pos=ent['centroid'],
                        target_color=ent['color'],
                        max_steps=2
                    )
                    subgoals.append(interact_sg)

        # 3. Frontier Exploration Fallback
        self.subgoal_counter += 1
        frontier_sg = Subgoal(
            subgoal_id=f"sg_frontier_{self.subgoal_counter}",
            subgoal_type=SubgoalType.EXPLORE_FRONTIER,
            max_steps=15
        )
        subgoals.append(frontier_sg)

        return subgoals


class OptionsController:
    """
    Hierarchical SMDP Manager:
    Sequences and executes Subgoals via formal Options ω = < I, π, β >.
    """
    def __init__(
        self,
        explorer: Optional[DomainAgnosticFrontierExplorer] = None,
        sandbox: Optional[MentalSandbox] = None,
        memory_bank: Optional[EpisodicMemoryBank] = None
    ):
        self.explorer = explorer if explorer is not None else DomainAgnosticFrontierExplorer()
        self.sandbox = sandbox if sandbox is not None else MentalSandbox()
        self.physics_inducer = getattr(self.sandbox, 'physics_inducer', None)
        self.subgoal_generator = SubgoalGenerator(self.explorer)
        self.hypothesis_inducer = self.subgoal_generator.inducer
        self.dynamic_tracker = DynamicEntityTracker()
        self.memory_bank = memory_bank if memory_bank is not None else EpisodicMemoryBank()

        # Standard Options Palette
        self.options: Dict[SubgoalType, BaseOption] = {
            SubgoalType.REACH_ENTITY: ReachEntityOption(self.explorer, self.sandbox),
            SubgoalType.PUSH_ENTITY: PushEntityOption(self.explorer, self.sandbox, getattr(self.sandbox, 'physics_inducer', None)),
            SubgoalType.BRIDGE_GAP: BridgeGapOption(self.explorer, self.sandbox, getattr(self.sandbox, 'physics_inducer', None)),
            SubgoalType.INTERACT_ENTITY: InteractEntityOption(self.explorer),
            SubgoalType.EXPLORE_FRONTIER: FrontierExploreOption(self.explorer),
            SubgoalType.DEADLOCK_RESET: DeadlockResetOption(),
            SubgoalType.TOGGLE_SOLVE: ToggleSolveOption(self.explorer, self.sandbox, getattr(self.sandbox, 'physics_inducer', None)),
            SubgoalType.COLOR_DIP: ColorDipOption(self.explorer, self.sandbox, getattr(self.sandbox, 'physics_inducer', None)),
            SubgoalType.STAGE_ENTITY: StageEntityOption(self.explorer, self.sandbox, getattr(self.sandbox, 'physics_inducer', None)),
            SubgoalType.CARRY_ENTITY: CarryEntityOption(self.explorer, self.sandbox, getattr(self.sandbox, 'physics_inducer', None)),
            SubgoalType.SPACE_TIME_NAV: SpaceTimeOption(self.explorer, self.sandbox, getattr(self.sandbox, 'physics_inducer', None), self.dynamic_tracker),
            SubgoalType.GRAVITY_PLATFORM: GravityPlatformOption(self.explorer, self.sandbox, getattr(self.sandbox, 'physics_inducer', None), self.dynamic_tracker),
            SubgoalType.SWITCH_AVATAR: SwitchAvatarOption(self.explorer)
        }

        self.subgoal_dag = CausalSubgoalDAG()
        self.subgoal_queue: collections.deque[Subgoal] = collections.deque()
        self.current_subgoal: Optional[Subgoal] = None
        self.current_option: Optional[BaseOption] = None
        self.prev_levels_completed = 0
        self.failed_subgoal_count = 0
        self.last_grid: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_action_data: Optional[Dict[str, Any]] = None
        self.last_avatar_pos: Optional[Tuple[int, int]] = None

    def reset(self, keep_avatar_identity: bool = False):
        """Resets controller between episodes/levels."""
        self.subgoal_dag.clear()
        self.subgoal_queue.clear()
        self.current_subgoal = None
        self.current_option = None
        self.prev_levels_completed = 0
        self.failed_subgoal_count = 0
        self.last_grid = None
        self.last_action = None
        self.last_action_data = None
        self.last_avatar_pos = None
        self.dynamic_tracker.reset()
        self.explorer.reset(keep_avatar_identity=keep_avatar_identity)
        if self.physics_inducer is not None:
            self.physics_inducer.reset()
        if self.hypothesis_inducer is not None:
            self.hypothesis_inducer.reset()
        self.sandbox.reset()
        self.memory_bank.reset()
        for opt in self.options.values():
            opt.reset()

    def step(
        self,
        grid: np.ndarray,
        available_actions: List[int],
        levels_completed: int = 0
    ) -> Tuple[int, Optional[Dict[str, Any]], str]:
        """
        Hierarchical execution cycle:
        1. Online physics induction observation
        2. Check termination condition β_ω of active option
        3. Advance / select next subgoal and option
        4. Execute intra-option policy π_ω
        """
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]

        bg_canvas = int(collections.Counter(grid.flatten()).most_common(1)[0][0])
        self.explorer._update_avatar(grid, bg_canvas)
        self.explorer._update_floor_colors(grid, bg_canvas)
        if self.physics_inducer is not None:
            self.physics_inducer.walkable_colors.update(self.explorer.floor_colors)
            self.explorer.hazard_colors = set(self.physics_inducer.hazard_colors)
            if hasattr(self.physics_inducer, 'get_suspect_hazard_colors'):
                self.explorer.suspect_hazard_colors = self.physics_inducer.get_suspect_hazard_colors(
                    grid, bg_canvas, avatar_pos=self.explorer.avatar_pos, verified_floors=self.explorer.floor_colors
                )

        # Update dynamic entity tracking online
        self.dynamic_tracker.observe_frame(
            grid=grid,
            avatar_pos=self.explorer.avatar_pos,
            avatar_color=self.explorer.avatar_color,
            floor_colors=self.explorer.floor_colors,
            action_id=self.last_action
        )

        # 0. Online Physics Induction: Feed previous transition to Sandbox & LocalPhysicsInducer
        if self.last_grid is not None and self.last_action is not None:
            self.sandbox.observe_transition(
                prev_grid=self.last_grid,
                action_id=self.last_action,
                action_data=self.last_action_data,
                current_grid=grid,
                prev_avatar_pos=self.last_avatar_pos,
                current_avatar_pos=self.explorer.avatar_pos,
                game_over=False,
                level_advanced=(levels_completed > self.prev_levels_completed)
            )
            # Epistemic invariant verification
            self.memory_bank.validate_transition(
                prev_grid=self.last_grid,
                action_id=self.last_action,
                current_grid=grid,
                prev_avatar_pos=self.last_avatar_pos,
                current_avatar_pos=self.explorer.avatar_pos,
                game_over=False
            )

        # Reset queue & warm-start if level advanced
        if levels_completed > self.prev_levels_completed:
            # 1. Consolidate learned invariants from the completed level
            self.memory_bank.record_level_completion(
                level_idx=self.prev_levels_completed,
                explorer=self.explorer,
                physics_inducer=self.physics_inducer,
                hypothesis_inducer=self.hypothesis_inducer,
                sandbox=self.sandbox,
                active_option_name=self.current_option.name if self.current_option else None
            )
            self.prev_levels_completed = levels_completed
            self.subgoal_dag.clear()
            self.subgoal_queue.clear()
            self.current_subgoal = None
            self.current_option = None
            self.failed_subgoal_count = 0

            # 2. Warm-start the new level with cached cross-level invariants
            self.memory_bank.warm_start_level(
                level_idx=levels_completed,
                explorer=self.explorer,
                physics_inducer=self.physics_inducer,
                hypothesis_inducer=self.hypothesis_inducer,
                sandbox=self.sandbox
            )

        # 1. Termination Check on Active Option
        if self.current_option is not None and self.current_subgoal is not None:
            is_done, status, reason = self.current_option.is_terminated(
                grid,
                self.current_subgoal,
                levels_completed,
                self.prev_levels_completed
            )
            if is_done:
                self.current_subgoal.status = status
                if status == SubgoalStatus.COMPLETED:
                    self.subgoal_dag.mark_completed(self.current_subgoal.subgoal_id)
                    self.failed_subgoal_count = 0
                elif status in (SubgoalStatus.FAILED, SubgoalStatus.TIMEOUT):
                    self.subgoal_dag.mark_failed(self.current_subgoal.subgoal_id)
                    self.failed_subgoal_count += 1
                self.current_subgoal = None
                self.current_option.reset()
                self.current_option = None

        # 2. Deadlock Recovery Check: If multiple subgoals failed in a row and Reset is available
        if self.failed_subgoal_count >= 3 and 7 in available_actions:
            self.subgoal_dag.clear()
            self.subgoal_queue.clear()
            reset_sg = Subgoal("sg_deadlock_reset", SubgoalType.DEADLOCK_RESET, max_steps=1)
            self.subgoal_queue.append(reset_sg)
            self.failed_subgoal_count = 0

        # 3. Replenish Subgoal Queue & DAG if Empty
        if not self.subgoal_dag.has_pending_or_active() and not self.subgoal_queue and self.current_subgoal is None:
            new_subgoals = self.subgoal_generator.generate_subgoals(grid, available_actions)
            self.subgoal_dag.add_subgoals(new_subgoals)
            self.subgoal_queue.extend(new_subgoals)

        # 4. Activate Next Subgoal and Option
        if self.current_subgoal is None and (self.subgoal_dag.has_pending_or_active() or self.subgoal_queue):
            while self.subgoal_dag.has_pending_or_active() or self.subgoal_queue:
                cand_sg = self.subgoal_dag.get_next_ready_subgoal()
                if cand_sg is None:
                    if self.subgoal_queue:
                        cand_sg = self.subgoal_queue.popleft()
                    else:
                        break
                else:
                    try:
                        self.subgoal_queue.remove(cand_sg)
                    except ValueError:
                        pass

                # Transfer accumulated causal effects to subgoal metadata
                if self.subgoal_dag.accumulated_effects:
                    cand_sg.metadata.setdefault('hypothetical_solids', set()).update(
                        self.subgoal_dag.accumulated_effects.get('hypothetical_solids', set())
                    )
                    cand_sg.metadata.setdefault('unlocked_barriers', set()).update(
                        self.subgoal_dag.accumulated_effects.get('unlocked_barriers', set())
                    )

                cand_opt = self.options.get(cand_sg.subgoal_type)
                # Dynamic delegation between PushEntity, BridgeGap, ReachEntity, SpaceTimeNav, InteractEntity, etc.
                if cand_opt is None or not cand_opt.can_initiate(grid, cand_sg, available_actions):
                    if cand_sg.subgoal_type == SubgoalType.BRIDGE_GAP:
                        push_opt = self.options.get(SubgoalType.PUSH_ENTITY)
                        if push_opt is not None and push_opt.can_initiate(grid, cand_sg, available_actions):
                            cand_opt = push_opt
                    if cand_sg.subgoal_type == SubgoalType.PUSH_ENTITY:
                        reach_opt = self.options.get(SubgoalType.REACH_ENTITY)
                        if reach_opt is not None and reach_opt.can_initiate(grid, cand_sg, available_actions):
                            cand_opt = reach_opt
                    elif cand_sg.subgoal_type == SubgoalType.SPACE_TIME_NAV:
                        reach_opt = self.options.get(SubgoalType.REACH_ENTITY)
                        if reach_opt is not None and reach_opt.can_initiate(grid, cand_sg, available_actions):
                            cand_opt = reach_opt
                    elif cand_sg.subgoal_type == SubgoalType.REACH_ENTITY:
                        meta = cand_sg.metadata or {}
                        is_block = ('block_pos' in meta or 'block_center' in meta or meta.get('action') in ('seat', 'propel'))
                        if is_block:
                            push_opt = self.options.get(SubgoalType.PUSH_ENTITY)
                            if push_opt is not None and push_opt.can_initiate(grid, cand_sg, available_actions):
                                cand_opt = push_opt
                        elif any(e.pattern != MotionPattern.STATIC for e in self.dynamic_tracker.entities.values()):
                            st_opt = self.options.get(SubgoalType.SPACE_TIME_NAV)
                            if st_opt is not None and st_opt.can_initiate(grid, cand_sg, available_actions):
                                cand_opt = st_opt
                        if cand_opt is None or not cand_opt.can_initiate(grid, cand_sg, available_actions):
                            plat_opt = self.options.get(SubgoalType.GRAVITY_PLATFORM)
                            if plat_opt is not None and plat_opt.can_initiate(grid, cand_sg, available_actions):
                                cand_opt = plat_opt
                        if cand_opt is None or not cand_opt.can_initiate(grid, cand_sg, available_actions):
                            push_opt = self.options.get(SubgoalType.PUSH_ENTITY)
                            if push_opt is not None and push_opt.can_initiate(grid, cand_sg, available_actions):
                                cand_opt = push_opt
                    elif cand_sg.subgoal_type == SubgoalType.GRAVITY_PLATFORM:
                        reach_opt = self.options.get(SubgoalType.REACH_ENTITY)
                        if reach_opt is not None and reach_opt.can_initiate(grid, cand_sg, available_actions):
                            cand_opt = reach_opt
                    elif cand_sg.subgoal_type == SubgoalType.TOGGLE_SOLVE:
                        interact_opt = self.options.get(SubgoalType.INTERACT_ENTITY)
                        if interact_opt is not None and interact_opt.can_initiate(grid, cand_sg, available_actions):
                            cand_opt = interact_opt
                    elif cand_sg.subgoal_type == SubgoalType.INTERACT_ENTITY:
                        toggle_opt = self.options.get(SubgoalType.TOGGLE_SOLVE)
                        if toggle_opt is not None and toggle_opt.can_initiate(grid, cand_sg, available_actions):
                            cand_opt = toggle_opt
                    elif cand_sg.subgoal_type == SubgoalType.SWITCH_AVATAR:
                        interact_opt = self.options.get(SubgoalType.INTERACT_ENTITY)
                        if interact_opt is not None and interact_opt.can_initiate(grid, cand_sg, available_actions):
                            cand_opt = interact_opt
                    elif cand_sg.subgoal_type == SubgoalType.COLOR_DIP:
                        reach_opt = self.options.get(SubgoalType.REACH_ENTITY)
                        if reach_opt is not None and reach_opt.can_initiate(grid, cand_sg, available_actions):
                            cand_opt = reach_opt
                    elif cand_sg.subgoal_type == SubgoalType.STAGE_ENTITY:
                        push_opt = self.options.get(SubgoalType.PUSH_ENTITY)
                        if push_opt is not None and push_opt.can_initiate(grid, cand_sg, available_actions):
                            cand_opt = push_opt
                    elif cand_sg.subgoal_type == SubgoalType.CARRY_ENTITY:
                        push_opt = self.options.get(SubgoalType.PUSH_ENTITY)
                        if push_opt is not None and push_opt.can_initiate(grid, cand_sg, available_actions):
                            cand_opt = push_opt
                if cand_opt is not None and cand_opt.can_initiate(grid, cand_sg, available_actions):
                    self.current_subgoal = cand_sg
                    self.current_subgoal.status = SubgoalStatus.ACTIVE
                    self.current_option = cand_opt
                    break

        # 5. Execute Active Option Policy
        if self.current_option is not None and self.current_subgoal is not None:
            act_id, act_data = self.current_option.step(grid, self.current_subgoal, available_actions)
            info = f"{self.current_option.name}[{self.current_subgoal.subgoal_id}]"
            self.last_grid = grid.copy()
            self.last_action = act_id
            self.last_action_data = act_data
            self.last_avatar_pos = self.explorer.avatar_pos
            return (act_id, act_data, info)

        # Fallback to general explorer if no option could be initiated
        act_id, act_data = self.explorer.step(grid, available_actions, levels_completed)
        self.last_grid = grid.copy()
        self.last_action = act_id
        self.last_action_data = act_data
        self.last_avatar_pos = self.explorer.avatar_pos
        return (act_id, act_data, "FallbackExplorer")
