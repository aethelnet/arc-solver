"""
=============================================================================
ARC-AGI CAUSAL META-TOOL SELECTOR: MENTAL SANDBOX & K-STEP LOOKAHEAD
=============================================================================
Simulates candidate actions and multi-step plans in a fast, in-memory virtual
rollout before committing them to the physical environment:
1. Interface Legality Guard: Vetoes actions outside available_actions.
2. Coordinate Bounds Guard: Vetoes out-of-bounds clicks.
3. K-Step Forward Physical Rollout: Simulates push mechanics and avatar trajectories.
4. Irreversible Corner-Trap Guard (Dead-Square Detection):
   Detects if a push trajectory traps a block in an impassable 2-wall corner.
5. In-Memory Loop & Cyclic Attractor Detection:
   Prunes plans that cycle endlessly in imagination with 0 state progress.
=============================================================================
"""

from typing import Tuple, Optional, Dict, Any, List, Set
from collections import deque
import collections
import numpy as np


class MentalSandbox:
    """
    In-memory cognitive rollout sandbox.
    Acts as an epistemic safety filter and forward simulator between
    option/tool proposals and the physical environment.
    """
    def __init__(
        self,
        history_len: int = 16,
        lookahead_depth: int = 6,
        physics_inducer: Optional[Any] = None
    ):
        self.history_len = history_len
        self.lookahead_depth = lookahead_depth
        self.state_hashes: deque = deque(maxlen=history_len)
        self.action_history: deque = deque(maxlen=history_len)
        self.avatar_pos_history: deque = deque(maxlen=history_len)
        self.wall_bumped_actions: Dict[Tuple[int, int], Set[int]] = {}
        self.wall_bump_count: int = 0
        self.consecutive_intentional_waits: int = 0
        self.last_grid_hash: Optional[int] = None
        self.learned_click_rules: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
        if physics_inducer is None:
            try:
                from .local_physics_inducer import LocalPhysicsInducer
                self.physics_inducer = LocalPhysicsInducer()
            except ImportError:
                self.physics_inducer = None
        else:
            self.physics_inducer = physics_inducer

    def reset(self):
        """Resets sandbox tracking across games/episodes."""
        self.state_hashes.clear()
        self.action_history.clear()
        self.avatar_pos_history.clear()
        self.wall_bumped_actions.clear()
        self.wall_bump_count = 0
        self.consecutive_intentional_waits = 0
        self.last_grid_hash = None
        self.learned_click_rules.clear()
        if self.physics_inducer is not None:
            self.physics_inducer.reset()

    def record_step(
        self,
        grid: np.ndarray,
        action_id: int,
        action_data: Optional[Dict[str, Any]] = None,
        prev_grid: Optional[np.ndarray] = None,
        prev_avatar_pos: Optional[Tuple[int, int]] = None,
        current_avatar_pos: Optional[Tuple[int, int]] = None,
        game_over: bool = False,
        level_advanced: bool = False
    ):
        """Records executed transition hash and learns online patch rewrite rules."""
        grid_2d = np.asarray(grid)
        while grid_2d.ndim > 2:
            grid_2d = grid_2d[-1]
        h = hash(grid_2d.tobytes())

        if current_avatar_pos is not None:
            self.avatar_pos_history.append(current_avatar_pos)

        is_intentional_wait = bool(action_data and action_data.get('intentional_wait'))

        if is_intentional_wait:
            self.consecutive_intentional_waits += 1
            # Do not increment wall_bump_count on intentional waits
        else:
            self.consecutive_intentional_waits = 0
            # HUD-invariant wall-bump tracking via avatar coordinates
            if prev_avatar_pos is not None and current_avatar_pos is not None:
                if action_id in (1, 2, 3, 4):
                    if prev_avatar_pos == current_avatar_pos:
                        self.wall_bump_count += 1
                        self.wall_bumped_actions.setdefault(prev_avatar_pos, set()).add(action_id)
                    else:
                        self.wall_bump_count = 0
            elif self.last_grid_hash is not None and h == self.last_grid_hash:
                if action_id in (1, 2, 3, 4):
                    self.wall_bump_count += 1
            else:
                self.wall_bump_count = 0

        self.last_grid_hash = h
        self.state_hashes.append(h)
        self.action_history.append(action_id)

        # Update local cellular physics induction
        if self.physics_inducer is not None and prev_grid is not None:
            self.physics_inducer.observe_transition(
                prev_grid=prev_grid,
                action_id=action_id,
                action_data=action_data,
                current_grid=grid_2d,
                prev_avatar_pos=prev_avatar_pos,
                current_avatar_pos=current_avatar_pos,
                game_over=game_over,
                level_advanced=level_advanced
            )

        # Learn online click rewrite rules: (click_x, click_y) -> [pixels modified]
        if action_id == 6 and action_data and 'x' in action_data and 'y' in action_data and prev_grid is not None:
            prev_2d = np.asarray(prev_grid)
            while prev_2d.ndim > 2:
                prev_2d = prev_2d[-1]
            diff_coords = np.argwhere(prev_2d != grid_2d)
            if len(diff_coords) > 0 and len(diff_coords) <= 200:
                diffs = [(int(r), int(c), int(prev_2d[r, c]), int(grid_2d[r, c])) for r, c in diff_coords]
                key = (int(action_data['x']), int(action_data['y']))
                self.learned_click_rules[key] = diffs

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
        """Direct alias for observe_transition calling record_step."""
        self.record_step(
            grid=current_grid,
            action_id=action_id,
            action_data=action_data,
            prev_grid=prev_grid,
            prev_avatar_pos=prev_avatar_pos,
            current_avatar_pos=current_avatar_pos,
            game_over=game_over,
            level_advanced=level_advanced
        )

    def validate_action(
        self,
        grid: np.ndarray,
        action_id: int,
        action_data: Optional[Dict[str, Any]],
        available_actions: List[int],
        avatar_pos: Optional[Tuple[int, int]] = None,
        stride: int = 1,
        floor_colors: Optional[Set[int]] = None,
        is_tool_busy: bool = False,
        **kwargs
    ) -> Tuple[bool, str]:
        """
        Evaluates 1-step candidate action safety:
        Legality, bounds, zero-motion wall bumps, and cyclic loops.
        """
        grid_2d = np.asarray(grid)
        while grid_2d.ndim > 2:
            grid_2d = grid_2d[-1]
        H, W = grid_2d.shape

        # 1. Interface Legality Check
        if action_id not in available_actions:
            return False, f"Illegal action {action_id}; available: {available_actions}"

        # 2. Click Coordinate Bounds Check (Action 6)
        if action_id == 6:
            if not action_data or 'x' not in action_data or 'y' not in action_data:
                return False, "Action 6 missing required 'x' and 'y' coordinates"
            px, py = int(action_data['x']), int(action_data['y'])
            if px < 0 or px >= W or py < 0 or py >= H:
                return False, f"Click ({px}, {py}) out of bounds ({W}x{H})"

        is_intentional_wait = bool(action_data and action_data.get('intentional_wait'))

        # Safety threshold: prevent infinite wait deadlock if an option hangs
        if is_intentional_wait and self.consecutive_intentional_waits >= 15:
            return False, "Excessive intentional wait threshold reached (>=15 steps)"

        # 3. Wall Ramming & Confirmed Blocked Action Trap
        if action_id in (1, 2, 3, 4) and avatar_pos is not None:
            if action_id in self.wall_bumped_actions.get(avatar_pos, set()):
                return False, f"Action {action_id} already confirmed blocked at {avatar_pos}"

        if not is_intentional_wait and not is_tool_busy and self.wall_bump_count >= 1 and action_id in (1, 2, 3, 4):
            if len(self.action_history) > 0 and action_id == self.action_history[-1]:
                return False, f"Wall-bump deadlock: Repeated action {action_id} into wall with 0 motion"

        # 4. Cyclic Oscillation Check (A <-> B <-> A <-> B with zero net progress)
        if not is_intentional_wait and len(self.avatar_pos_history) >= 4 and avatar_pos is not None:
            pos_list = list(self.avatar_pos_history)
            if pos_list[-4] == pos_list[-2] and pos_list[-3] == pos_list[-1] and pos_list[-2] != pos_list[-1]:
                dr, dc = self._action_to_delta(action_id)
                next_pos = (avatar_pos[0] + dr * stride, avatar_pos[1] + dc * stride)
                if next_pos == pos_list[-2]:
                    return False, f"Cyclic avatar oscillation loop detected between {pos_list[-2]} and {pos_list[-1]}"
        elif not is_intentional_wait and len(self.action_history) >= 4:
            acts = list(self.action_history)
            if acts[-4] == acts[-2] and acts[-3] == acts[-1]:
                if action_id == acts[-2]:
                    hashes = list(self.state_hashes)
                    if len(hashes) >= 4 and hashes[-4] == hashes[-2] and hashes[-3] == hashes[-1]:
                        return False, f"Cyclic oscillation loop detected ({acts[-2]} <-> {acts[-1]}) with zero state change"

        # 5. Immediate 1-Step Corner Push Deadlock Check
        if action_id in (1, 2, 3, 4) and avatar_pos is not None:
            is_deadlock, reason = self.simulate_push_step(
                grid_2d, avatar_pos, action_id, stride=stride, floor_colors=floor_colors
            )
            if is_deadlock:
                return False, reason

        # 6. Lethal Hazard & Suspect Death-Tile Contact Prevention (Pessimistic Safe Active Inference)
        if action_id in (1, 2, 3, 4) and avatar_pos is not None:
            dr, dc = self._action_to_delta(action_id)
            tr, tc = avatar_pos[0] + dr * stride, avatar_pos[1] + dc * stride
            if 0 <= tr < H and 0 <= tc < W:
                target_val = int(grid_2d[tr, tc])
                # Direct confirmed hazard check
                hazards = getattr(self.physics_inducer, 'hazard_colors', set())
                if target_val in hazards:
                    return False, f"Lethal Hazard Veto: Action {action_id} steps into confirmed lethal color {target_val}"
                
                # Check verified safe floor colors first
                floors = floor_colors if floor_colors is not None else set()
                walkable = getattr(self.physics_inducer, 'walkable_colors', set()) or set()
                if target_val not in floors and target_val not in walkable:
                    # Suspect unverified hazard check
                    suspects = getattr(self.physics_inducer, 'suspect_hazard_colors', set())
                    if not suspects and hasattr(self.physics_inducer, 'get_suspect_hazard_colors'):
                        bg_canvas = int(collections.Counter(grid_2d.flatten()).most_common(1)[0][0])
                        suspects = self.physics_inducer.get_suspect_hazard_colors(
                            grid_2d, bg_canvas, avatar_pos=avatar_pos, verified_floors=floors
                        )
                    if target_val in suspects:
                        is_intentional = (
                            (action_data and (action_data.get('probe_target') or action_data.get('is_goal'))) or
                            (action_data and action_data.get('target_pos') == (tr, tc))
                        )
                        if not is_intentional:
                            return False, f"Pessimistic Hazard Veto: Action {action_id} steps into unverified suspect color {target_val}"

        # 7. Kinematic Irreversibility & Safe-Set Pit Trap Veto
        if action_id in (1, 2, 3, 4) and avatar_pos is not None:
            gravity = None
            if self.physics_inducer and getattr(self.physics_inducer, 'gravity_vector', None) is not None:
                gravity = self.physics_inducer.gravity_vector
            elif kwargs.get('gravity') is not None:
                gravity = kwargs.get('gravity')

            if gravity is not None and gravity != (0, 0):
                jump_h = getattr(self.physics_inducer, 'jump_height', 2) if self.physics_inducer else 2
                jump_r = 2
                solids = (self.physics_inducer.solid_colors if self.physics_inducer else set()) or set()
                hazards = (self.physics_inducer.hazard_colors if self.physics_inducer else set()) or set()
                suspects = getattr(self.physics_inducer, 'suspect_hazard_colors', set()) or set()
                bg_canvas = int(collections.Counter(grid_2d.flatten()).most_common(1)[0][0])

                is_intentional = bool(action_data and (action_data.get('is_goal') or action_data.get('probe_target')))
                if not is_intentional:
                    dr, dc = self._action_to_delta(action_id)
                    to_pos = (avatar_pos[0] + dr * stride, avatar_pos[1] + dc * stride)

                    try:
                        from .platformer_planner import is_kinematically_irreversible_drop
                    except ImportError:
                        from platformer_planner import is_kinematically_irreversible_drop

                    is_irrev, is_fatal, reason, land_p = is_kinematically_irreversible_drop(
                        grid=grid_2d,
                        from_pos=avatar_pos,
                        to_pos=to_pos,
                        gravity=gravity,
                        jump_height=jump_h,
                        jump_reach=jump_r,
                        solid_colors=solids,
                        hazard_colors=hazards,
                        suspect_hazard_colors=suspects,
                        bg_canvas=bg_canvas,
                        stride=stride,
                        goal_candidates=kwargs.get('goal_candidates')
                    )
                    if is_irrev:
                        return False, reason

        return True, "Safe"

    def simulate_push_step(
        self,
        grid: np.ndarray,
        avatar_pos: Tuple[int, int],
        action_id: int,
        stride: int = 1,
        floor_colors: Optional[Set[int]] = None
    ) -> Tuple[bool, str]:
        """
        Simulates one directional push step to detect immediate corner traps.
        """
        floors = set(floor_colors) if floor_colors else set()
        dr, dc = self._action_to_delta(action_id)
        target_r, target_c = avatar_pos[0] + dr * stride, avatar_pos[1] + dc * stride
        H, W = grid.shape

        if not (0 <= target_r < H and 0 <= target_c < W):
            return False, ""

        bg_canvas = int(collections.Counter(grid.flatten()).most_common(1)[0][0])
        target_val = grid[target_r, target_c]

        # If target cell is not empty / background / floor, check if it's a movable block
        if target_val != bg_canvas and target_val != 0 and target_val not in floors:
            dest_r, dest_c = target_r + dr * stride, target_c + dc * stride
            if 0 <= dest_r < H and 0 <= dest_c < W:
                # Is the destination in a 2-wall corner?
                if self._is_corner_deadlock(grid, dest_r, dest_c, bg_canvas, floor_colors=floors):
                    return True, f"Irreversible Push Deadlock: Pushing block into 2-wall corner ({dest_r}, {dest_c})"

        return False, ""

    def simulate_plan_lookahead(
        self,
        grid: np.ndarray,
        initial_avatar: Tuple[int, int],
        actions: List[Any],
        goals: Optional[Set[Tuple[int, int]]] = None,
        stride: int = 1,
        floor_colors: Optional[Set[int]] = None,
        avatar_color: Optional[int] = None
    ) -> Tuple[bool, int, str]:
        """
        Simulates a k-step sequence of actions forward in time:
        Rollout: s_0 -> a_1 -> s_1 -> a_2 -> ... -> a_k -> s_k.
        Supports multi-pixel strides, entity pushing, and learned click rewrites.
        Returns:
            (is_safe: bool, safe_prefix_len: int, veto_reason: str)
        """
        sim_grid = grid.copy()
        while sim_grid.ndim > 2:
            sim_grid = sim_grid[-1]
        H, W = sim_grid.shape

        bg_canvas = int(collections.Counter(sim_grid.flatten()).most_common(1)[0][0])
        floors = set(floor_colors) if floor_colors else set()
        avatar = initial_avatar
        visited_sim_states = set()
        visited_sim_states.add((avatar, hash(sim_grid.tobytes())))

        goals = goals or set()
        bridged_cells: Set[Tuple[int, int]] = set()

        for step_idx, act_entry in enumerate(actions):
            act_id = act_entry[0] if isinstance(act_entry, (tuple, list)) else act_entry
            act_data = act_entry[1] if isinstance(act_entry, (tuple, list)) and len(act_entry) > 1 else None

            # 1. Action 6 (Click) Virtual Simulation via Learned Rewrite Rules
            if act_id == 6:
                if act_data and 'x' in act_data and 'y' in act_data:
                    click_key = (int(act_data['x']), int(act_data['y']))
                    if click_key in self.learned_click_rules:
                        for r, c, old_v, new_v in self.learned_click_rules[click_key]:
                            if 0 <= r < H and 0 <= c < W:
                                sim_grid[r, c] = new_v
                continue

            if act_id not in (1, 2, 3, 4):
                continue

            dr, dc = self._action_to_delta(act_id)
            s_dr = dr * stride
            s_dc = dc * stride
            nr, nc = avatar[0] + s_dr, avatar[1] + s_dc

            # Out of bounds check
            if not (0 <= nr < H and 0 <= nc < W):
                return False, step_idx, f"Step {step_idx}: Action {act_id} moves avatar out of bounds ({nr}, {nc})"

            cell_val = sim_grid[nr, nc]

            # If cell matches avatar_color (color-matching door/barrier), it is permeable
            if avatar_color is not None and cell_val == avatar_color:
                avatar = (nr, nc)
                continue

            solids = (self.physics_inducer.solid_colors if self.physics_inducer else set()) or set()
            hazards = (self.physics_inducer.hazard_colors if self.physics_inducer else set()) or set()
            suspects = getattr(self.physics_inducer, 'suspect_hazard_colors', set()) or set()
            gravity = self.physics_inducer.gravity_vector if (self.physics_inducer and getattr(self.physics_inducer, 'gravity_vector', None) is not None) else None

            # If moving into another entity (Push dynamics)
            if cell_val != bg_canvas and cell_val != 0 and cell_val not in floors:
                push_r, push_c = nr + s_dr, nc + s_dc
                if not (0 <= push_r < H and 0 <= push_c < W):
                    return False, step_idx, f"Step {step_idx}: Pushing block out of bounds ({push_r}, {push_c})"

                dest_val = sim_grid[push_r, push_c]
                # Block blocked by static wall or another object
                if dest_val != bg_canvas and dest_val != 0 and dest_val not in floors:
                    return False, step_idx, f"Step {step_idx}: Motion blocked by static obstacle at ({push_r}, {push_c})"

                # Check if pushed into corner dead-square (only if not falling into pit under gravity)
                is_corner = (push_r, push_c) not in goals and self._is_corner_deadlock(sim_grid, push_r, push_c, bg_canvas, floor_colors=floors)
                if is_corner:
                    try:
                        from .platformer_planner import is_supported
                    except ImportError:
                        from platformer_planner import is_supported
                    if gravity is None or gravity == (0, 0) or is_supported(sim_grid, (push_r, push_c), gravity, solids, stride):
                        return False, step_idx, f"Step {step_idx}: Irreversible corner deadlock at ({push_r}, {push_c})"

                # Execute virtual push on sim_grid: Shift entire entity component
                # Find connected component of the target object
                visited_comp = set()
                comp_q = [(nr, nc)]
                visited_comp.add((nr, nc))
                target_col = sim_grid[nr, nc]
                while comp_q:
                    cr, cc = comp_q.pop(0)
                    for d_r, d_c in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        adj_r, adj_c = cr + d_r, cc + d_c
                        if 0 <= adj_r < H and 0 <= adj_c < W and (adj_r, adj_c) not in visited_comp:
                            if sim_grid[adj_r, adj_c] == target_col:
                                visited_comp.add((adj_r, adj_c))
                                comp_q.append((adj_r, adj_c))

                # Shift component if compact (<= 25 cells)
                if len(visited_comp) <= 25:
                    old_cells = list(visited_comp)
                    new_cells = [(r + s_dr, c + s_dc) for r, c in old_cells]
                    # Clear old
                    for r, c in old_cells:
                        sim_grid[r, c] = bg_canvas
                    # Place new
                    for r, c in new_cells:
                        if 0 <= r < H and 0 <= c < W:
                            sim_grid[r, c] = target_col
                else:
                    new_cells = [(push_r, push_c)]
                    sim_grid[push_r, push_c] = cell_val
                    sim_grid[nr, nc] = bg_canvas

                # If gravity active, simulate gravity fall for pushed block into pit/chasm
                if gravity is not None and gravity != (0, 0):
                    try:
                        from .platformer_planner import is_supported, simulate_gravity_fall
                    except ImportError:
                        from platformer_planner import is_supported, simulate_gravity_fall
                    for b_r, b_c in new_cells:
                        if 0 <= b_r < H and 0 <= b_c < W:
                            if not is_supported(sim_grid, (b_r, b_c), gravity, solids, stride):
                                land_b, _, b_hit_haz = simulate_gravity_fall(
                                    sim_grid, (b_r, b_c), gravity, solids, hazards, suspects, bg_canvas, stride
                                )
                                if not b_hit_haz and land_b is not None:
                                    sim_grid[b_r, b_c] = bg_canvas
                                    sim_grid[land_b[0], land_b[1]] = target_col
                                    bridged_cells.add(land_b)

            # Directional Gravity & Fall Rollout in Simulation
            if gravity is not None and gravity != (0, 0):
                try:
                    from .platformer_planner import is_supported, is_kinematically_irreversible_drop
                except ImportError:
                    from platformer_planner import is_supported, is_kinematically_irreversible_drop

                if not is_supported(sim_grid, (nr, nc), gravity, solids, stride, hypothetical_solids=bridged_cells):
                    is_irrev, is_fatal, irrev_reason, land_pos = is_kinematically_irreversible_drop(
                        grid=sim_grid,
                        from_pos=avatar,
                        to_pos=(nr, nc),
                        gravity=gravity,
                        jump_height=getattr(self.physics_inducer, 'jump_height', 2),
                        solid_colors=solids,
                        hazard_colors=hazards,
                        suspect_hazard_colors=suspects,
                        bg_canvas=bg_canvas,
                        stride=stride,
                        goal_candidates=goals,
                        hypothetical_solids=bridged_cells
                    )
                    if is_fatal:
                        return False, step_idx, f"Step {step_idx}: Sim rollout caused fatal fall into hazard/void"
                    if is_irrev and land_pos not in goals:
                        return False, step_idx, f"Step {step_idx}: Sim rollout trapped in irreversible pit at {land_pos}"
                    if land_pos is not None:
                        avatar = land_pos
                else:
                    avatar = (nr, nc)
            else:
                avatar = (nr, nc)

            # In-memory loop detection
            state_key = (avatar, hash(sim_grid.tobytes()))
            if state_key in visited_sim_states:
                return False, step_idx, f"Step {step_idx}: Sim rollout entered cyclic attractor at {avatar}"
            visited_sim_states.add(state_key)

        return True, len(actions), "Plan Verified Safe"

    def _is_corner_deadlock(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        bg_color: int,
        floor_colors: Optional[Set[int]] = None
    ) -> bool:
        """
        Determines if cell (r, c) forms an impassable 2-wall corner:
        - (Wall Above OR Wall Below) AND (Wall Left OR Wall Right)
        """
        H, W = grid.shape
        floors = set(floor_colors) if floor_colors else set()

        def is_wall(cr: int, cc: int) -> bool:
            if not (0 <= cr < H and 0 <= cc < W):
                return True # Grid boundary acts as wall
            val = grid[cr, cc]
            return val != bg_color and val != 0 and val not in floors

        wall_up = is_wall(r - 1, c)
        wall_down = is_wall(r + 1, c)
        wall_left = is_wall(r, c - 1)
        wall_right = is_wall(r, c + 1)

        # 4 corner configurations
        top_left = wall_up and wall_left
        top_right = wall_up and wall_right
        bottom_left = wall_down and wall_left
        bottom_right = wall_down and wall_right

        return bool(top_left or top_right or bottom_left or bottom_right)

    def _action_to_delta(self, action: int) -> Tuple[int, int]:
        mapping = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        return mapping.get(action, (0, 0))

    def simulate_gravity_fall(
        self,
        grid: np.ndarray,
        start_pos: Tuple[int, int],
        gravity: Tuple[int, int] = (1, 0),
        solid_colors: Optional[Set[int]] = None,
        hazard_colors: Optional[Set[int]] = None,
        bg_canvas: int = 0,
        stride: int = 1,
        max_fall: int = 35
    ) -> Tuple[Tuple[int, int], List[Tuple[int, int]], bool]:
        """Traces downward fall under gravity until landing or lethal hazard."""
        from .platformer_planner import simulate_gravity_fall
        solids = solid_colors or (self.physics_inducer.solid_colors if self.physics_inducer else set())
        hazards = hazard_colors or (self.physics_inducer.hazard_colors if self.physics_inducer else set())
        return simulate_gravity_fall(
            grid=grid,
            start_pos=start_pos,
            gravity=gravity,
            solid_colors=solids,
            hazard_colors=hazards,
            bg_canvas=bg_canvas,
            stride=stride,
            max_fall=max_fall
        )

    def simulate_ballistic_jump(
        self,
        grid: np.ndarray,
        start_pos: Tuple[int, int],
        lateral_dir: int,
        jump_height: int = 2,
        jump_reach: int = 2,
        gravity: Tuple[int, int] = (1, 0),
        solid_colors: Optional[Set[int]] = None,
        hazard_colors: Optional[Set[int]] = None,
        bg_canvas: int = 0,
        stride: int = 1
    ) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]], bool]:
        """Simulates parabolic jump trajectory and landing detection."""
        from .platformer_planner import simulate_ballistic_jump
        solids = solid_colors or (self.physics_inducer.solid_colors if self.physics_inducer else set())
        hazards = hazard_colors or (self.physics_inducer.hazard_colors if self.physics_inducer else set())
        return simulate_ballistic_jump(
            grid=grid,
            start_pos=start_pos,
            lateral_dir=lateral_dir,
            jump_height=jump_height,
            jump_reach=jump_reach,
            gravity=gravity,
            solid_colors=solids,
            hazard_colors=hazards,
            bg_canvas=bg_canvas,
            stride=stride
        )
