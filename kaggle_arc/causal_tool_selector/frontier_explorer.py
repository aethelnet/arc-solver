"""
=============================================================================
ARC-AGI CAUSAL META-TOOL SELECTOR: DOMAIN-AGNOSTIC FRONTIER EXPLORER
=============================================================================
Universal neurosymbolic explorer for unclassified ARC-AGI environments:
1. Online Avatar Localization & Tracking (motion-diff correlation)
2. Arena Floor & Canvas Separation (multi-scale traversability discovery)
3. Dynamic Obstacle & Wall Bump Learning
4. Salient Interactive Target & Epistemic Prioritization
5. 4-Connected BFS Shortest-Path Planning
6. Contextual Action Generation (Nav, Interact, or Click)
7. Anti-Stagnation & Cyclic Oscillation Pruning
=============================================================================
"""

import collections
import heapq
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np

try:
    from .kinematics_calibrator import OnlineKinematicsCalibrator, CARDINAL_DELTAS
except ImportError:
    from kinematics_calibrator import OnlineKinematicsCalibrator, CARDINAL_DELTAS


class DomainAgnosticFrontierExplorer:
    """
    Autonomous domain-agnostic explorer that replaces blind action loops
    with principled spatial perception, entity discovery, and pathfinding.
    """
    def __init__(
        self,
        calibrator: Optional[OnlineKinematicsCalibrator] = None,
        physics_inducer: Optional[Any] = None
    ):
        self.avatar_pos: Optional[Tuple[int, int]] = None
        self.avatar_color: Optional[int] = None
        self.avatar_confidence: float = 0.0
        self.stride: int = 1
        self.calibrator = calibrator if calibrator is not None else OnlineKinematicsCalibrator()
        self.physics_inducer = physics_inducer
        self._gravity_vector: Optional[Tuple[int, int]] = None
        self.jump_height: int = 2

        self.visited_cells: Set[Tuple[int, int]] = set()
        self.recent_positions: collections.deque = collections.deque(maxlen=24)
        self.obstacle_cells: Set[Tuple[int, int]] = set()
        self.traversable_cells: Set[Tuple[int, int]] = set()
        self.floor_colors: Set[int] = set()
        self.hazard_colors: Set[int] = set()
        self.suspect_hazard_colors: Set[int] = set()
        self.border_wall_colors: Set[int] = set()
        self.interacted_targets: Set[Tuple[int, int]] = set()

        # Epistemic & Stagnation State
        self.unreachable_cooldown: Dict[Tuple[int, int], int] = {}
        self.target_failure_counts: collections.Counter = collections.Counter()
        self.junction_history: List[Tuple[int, int]] = []
        self.retreat_target: Optional[Tuple[int, int]] = None
        self.cycle_trap_detected: int = 0
        self.current_cycle_orbit: Set[Tuple[int, int]] = set()
        self.epistemic_frontiers: List[Dict[str, Any]] = []

        self.last_grid: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.consecutive_stuck: int = 0
        self.active_path: List[Tuple[int, int]] = []

    @property
    def gravity_vector(self) -> Optional[Tuple[int, int]]:
        if self.physics_inducer and getattr(self.physics_inducer, 'gravity_vector', None) is not None:
            return self.physics_inducer.gravity_vector
        return self._gravity_vector

    @gravity_vector.setter
    def gravity_vector(self, val: Optional[Tuple[int, int]]):
        self._gravity_vector = val
        if self.physics_inducer is not None:
            self.physics_inducer.gravity_vector = val

    def reset(self, keep_avatar_identity: bool = False):
        """Resets explorer memory between episodes/levels."""
        self.avatar_pos = None
        self._gravity_vector = None
        if not keep_avatar_identity:
            self.avatar_color = None
            self.avatar_confidence = 0.0
            self.stride = 1
        else:
            self.avatar_confidence = 0.5
        self.visited_cells.clear()
        self.recent_positions.clear()
        self.obstacle_cells.clear()
        self.traversable_cells.clear()
        self.floor_colors.clear()
        self.hazard_colors.clear()
        self.suspect_hazard_colors.clear()
        self.border_wall_colors.clear()
        self.interacted_targets.clear()
        self.unreachable_cooldown.clear()
        self.target_failure_counts.clear()
        self.junction_history.clear()
        self.retreat_target = None
        self.cycle_trap_detected = 0
        self.current_cycle_orbit.clear()
        self.epistemic_frontiers.clear()
        self.last_grid = None
        self.last_action = None
        self.consecutive_stuck = 0
        self.active_path.clear()

    def step(
        self,
        grid: np.ndarray,
        available_actions: List[int],
        levels_completed: int = 0
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Executes one exploratory step:
        1. Updates spatial perception (avatar position, floor colors, traversability).
        2. Maintains unreachable target cooldowns & invalidates on global grid changes.
        3. Detects stagnation, oscillation cycles, and branches (junctions).
        4. Detects salient entities and unvisited frontiers via Epistemic Information Gain.
        5. Generates optimal navigation, hierarchical retreat, or interaction action.
        """
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        h, w = grid.shape

        bg_canvas = int(collections.Counter(grid.flatten()).most_common(1)[0][0])

        # 1. Cooldown decay for unreachable targets
        expired = [pos for pos, timer in self.unreachable_cooldown.items() if timer <= 1]
        for pos in expired:
            del self.unreachable_cooldown[pos]
        for pos in list(self.unreachable_cooldown.keys()):
            if pos not in expired:
                self.unreachable_cooldown[pos] -= 1

        # 2. Check for external grid changes away from avatar (e.g. opened door / switch state)
        if self.last_grid is not None and self.avatar_pos is not None:
            raw_diffs = np.argwhere(self.last_grid != grid)
            if len(raw_diffs) > 0:
                remote_diffs = [
                    (r, c) for r, c in raw_diffs
                    if (abs(r - self.avatar_pos[0]) > 2 or abs(c - self.avatar_pos[1]) > 2) and r < h - 1
                ]
                if len(remote_diffs) >= 1:
                    # Environment changed remotely; previously unreachable targets may now be accessible
                    self.unreachable_cooldown.clear()
                    self.target_failure_counts.clear()

        # 3. Avatar Localization & Tracking
        self._update_avatar(grid, bg_canvas)

        # 4. Update Floor Colors around avatar
        self._update_floor_colors(grid, bg_canvas)
        if self.floor_colors:
            self.suspect_hazard_colors.difference_update(self.floor_colors)

        if self.avatar_pos is not None:
            self.visited_cells.add(self.avatar_pos)
            self.traversable_cells.add(self.avatar_pos)
            self.recent_positions.append(self.avatar_pos)
            # Record branching junction
            self._record_junction_if_branching(grid, bg_canvas)

            # Online Gravity Detection from Motion Diffs:
            # If lateral or non-vertical action caused vertical downward displacement
            if len(self.recent_positions) >= 2 and self.last_action in [3, 4, 5, 6]:
                prev_p = self.recent_positions[-2]
                curr_p = self.recent_positions[-1]
                if curr_p[0] - prev_p[0] > 0:
                    self.gravity_vector = (1, 0)
                elif curr_p[0] - prev_p[0] < 0:
                    self.gravity_vector = (-1, 0)

        # 5. Check for Stagnation & Generalized Periodic Cyclic Oscillation (periods 2 to 8)
        is_static = (len(self.recent_positions) >= 2 and self.recent_positions[-1] == self.recent_positions[-2])
        is_cycle = False
        detected_period = 0
        recent_list = list(self.recent_positions)
        if not is_static and len(recent_list) >= 4:
            for K in range(2, min(9, len(recent_list) // 2 + 1)):
                if recent_list[-K:] == recent_list[-2*K:-K]:
                    is_cycle = True
                    detected_period = K
                    break

        if is_static:
            self.consecutive_stuck += 1
            if self.last_action in [1, 2, 3, 4] and self.avatar_pos is not None:
                dr, dc = self._action_to_delta(self.last_action)
                bump_cell = (self.avatar_pos[0] + dr * self.stride, self.avatar_pos[1] + dc * self.stride)
                if 0 <= bump_cell[0] < h and 0 <= bump_cell[1] < w:
                    self.obstacle_cells.add(bump_cell)
        elif is_cycle:
            self.consecutive_stuck += 1
            self.cycle_trap_detected += 1
            self.current_cycle_orbit = set(recent_list[-detected_period:])
            # Add all cells in the cyclic orbit to cooldown to force breakout
            for c_pos in self.current_cycle_orbit:
                self.unreachable_cooldown[c_pos] = max(self.unreachable_cooldown.get(c_pos, 0), 15)
        else:
            self.consecutive_stuck = max(0, self.consecutive_stuck - 1)
            self.cycle_trap_detected = max(0, self.cycle_trap_detected - 1)
            if self.cycle_trap_detected == 0:
                self.current_cycle_orbit.clear()

        # Trigger Hierarchical Retreat if stuck or oscillating
        if (self.consecutive_stuck >= 2 or self.cycle_trap_detected >= 1) and self.avatar_pos is not None:
            retreat_j = self._find_best_retreat_junction(grid, bg_canvas)
            if retreat_j is not None:
                self.retreat_target = retreat_j

        # If arrived at retreat target, clear retreat
        if self.retreat_target is not None and self.avatar_pos == self.retreat_target:
            self.retreat_target = None
            self.consecutive_stuck = 0
            self.cycle_trap_detected = 0

        # 6. If adjacent to an uninteracted salient entity and Action 5 (INTERACT) is available
        if 5 in available_actions and self.avatar_pos is not None:
            adj_entities = self._find_adjacent_uninteracted_entities(grid, bg_canvas)
            if adj_entities:
                target = adj_entities[0]
                self.interacted_targets.add(target)
                self.last_action = 5
                self.last_grid = grid.copy()
                return (5, None)

        # 7. Find Targets (Salient Entities, Epistemic Frontiers, or Retreat Target)
        target_pos = self._select_next_target(grid, bg_canvas)

        # 8. Plan Path to Target
        nav_action = None
        if self.avatar_pos is not None and target_pos is not None:
            path = self._find_path(grid, self.avatar_pos, target_pos, bg_canvas)
            if path and len(path) > 1:
                next_cell = path[1]
                nav_action = self._direction_to_action(self.avatar_pos, next_cell)
            else:
                # Path planning failed for selected target: cooldown
                self.target_failure_counts[target_pos] += 1
                if self.target_failure_counts[target_pos] >= 2:
                    self.unreachable_cooldown[target_pos] = 25

        # 9. Fallback if pathfinding fails or avatar unknown
        if nav_action is None or nav_action not in available_actions:
            nav_action = self._fallback_action(available_actions, grid, bg_canvas)

        # 10. Fallback to Click (Action 6) if navigation impossible
        action_data = None
        if nav_action not in available_actions and 6 in available_actions and target_pos is not None:
            nav_action = 6
            action_data = {'x': int(target_pos[1]), 'y': int(target_pos[0])}

        self.last_action = nav_action
        self.last_grid = grid.copy()
        return (nav_action, action_data)

    def _update_avatar(self, current_grid: np.ndarray, bg_canvas: int = 0, **kwargs):
        """Identifies avatar by correlating previous action displacement with grid delta across strides."""
        if 'bg_color' in kwargs:
            bg_canvas = kwargs['bg_color']
        h, w = current_grid.shape
        if self.last_grid is not None and self.last_action in [1, 2, 3, 4]:
            expected_dr, expected_dc = self._action_to_delta(self.last_action)
            raw_diff_coords = np.argwhere(self.last_grid != current_grid)

            # Filter out HUD step-counter row 63 and extreme margins
            diff_coords = [tuple(p) for p in raw_diff_coords if p[0] < h - 1]

            if len(diff_coords) > 0:
                disappeared = []
                appeared = []
                for r, c in diff_coords:
                    if self.last_grid[r, c] != current_grid[r, c]:
                        if current_grid[r, c] in self.floor_colors or current_grid[r, c] == bg_canvas:
                            disappeared.append((r, c, self.last_grid[r, c]))
                        else:
                            appeared.append((r, c, current_grid[r, c]))

                # Check multi-stride correlation: S in [1, 2, 3, 4, 5, 6, 8]
                candidate_strides = [1, 2, 3, 4, 5, 6, 8]
                test_dirs = [(expected_dr, expected_dc)] + [d for d in CARDINAL_DELTAS if d != (expected_dr, expected_dc)]

                # Case A: Avatar position was already established
                if self.avatar_pos is not None:
                    best_S = None
                    best_dir = None
                    best_score = -1.0
                    for dr, dc in test_dirs:
                        for S in candidate_strides:
                            k = max(S // 2, 0)
                            cand_r = self.avatar_pos[0] + dr * S
                            cand_c = self.avatar_pos[1] + dc * S
                            if (0 <= self.avatar_pos[0] - k and self.avatar_pos[0] + k + 1 <= h and
                                0 <= self.avatar_pos[1] - k and self.avatar_pos[1] + k + 1 <= w and
                                0 <= cand_r - k and cand_r + k + 1 <= h and
                                0 <= cand_c - k and cand_c + k + 1 <= w):
                                old_patch = self.last_grid[self.avatar_pos[0]-k : self.avatar_pos[0]+k+1, self.avatar_pos[1]-k : self.avatar_pos[1]+k+1]
                                new_patch = current_grid[cand_r-k : cand_r+k+1, cand_c-k : cand_c+k+1]
                                score = float(np.mean(old_patch == new_patch))
                                if dr == expected_dr and dc == expected_dc and S == getattr(self, 'stride', 1):
                                    if current_grid[self.avatar_pos[0], self.avatar_pos[1]] != self.last_grid[self.avatar_pos[0], self.avatar_pos[1]]:
                                        score = max(score, 0.85)
                                if S == getattr(self, 'stride', 1):
                                    score += 0.05
                                if score > best_score:
                                    best_score = score
                                    best_S = S
                                    best_dir = (dr, dc)
                        if best_score >= 0.70:
                            break

                    if best_S is not None and best_dir is not None and best_score >= 0.70:
                        self.stride = best_S
                        s_dr = best_dir[0] * best_S
                        s_dc = best_dir[1] * best_S
                        exp_target = (self.avatar_pos[0] + s_dr, self.avatar_pos[1] + s_dc)
                        if 0 <= exp_target[0] < h and 0 <= exp_target[1] < w:
                            self.avatar_pos = exp_target
                            self.avatar_confidence = 1.0
                            at_val = int(current_grid[exp_target[0], exp_target[1]])
                            if at_val != bg_canvas and at_val not in self.floor_colors and at_val != 0:
                                self.avatar_color = at_val
                            if hasattr(self, 'calibrator') and self.last_action in (1, 2, 3, 4):
                                self.calibrator.observe(self.last_action, best_dir)
                            self.last_action = None
                            self.last_grid = current_grid.copy()
                            return

                # Case B: Avatar discovery / re-localization via rigid cluster shift
                diff_set = set(diff_coords)
                best_S_b = None
                best_dir_b = None
                best_matching_appeared = []
                for dr, dc in test_dirs:
                    for S in candidate_strides:
                        s_dr = dr * S
                        s_dc = dc * S
                        matching = []
                        for r, c in diff_coords:
                            nr, nc = r + s_dr, c + s_dc
                            if (nr, nc) in diff_set:
                                if self.last_grid[r, c] == current_grid[nr, nc]:
                                    matching.append((nr, nc, int(current_grid[nr, nc])))
                        if len(matching) > len(best_matching_appeared):
                            best_matching_appeared = matching
                            best_S_b = S
                            best_dir_b = (dr, dc)
                    if len(best_matching_appeared) >= 2:
                        break

                if best_S_b is not None and best_dir_b is not None and len(best_matching_appeared) >= 1:
                    self.stride = best_S_b
                    avatar_cells = [m for m in best_matching_appeared if self.avatar_color is not None and m[2] == self.avatar_color]
                    chosen_match = avatar_cells if avatar_cells else best_matching_appeared
                    coords = np.array([[r, c] for r, c, _ in chosen_match])
                    mean_pos = np.round(np.mean(coords, axis=0)).astype(int)
                    self.avatar_pos = (int(mean_pos[0]), int(mean_pos[1]))
                    c_cand = chosen_match[0][2]
                    if c_cand != 0 or self.avatar_color is None:
                        self.avatar_color = c_cand
                    self.avatar_confidence = 1.0
                    if hasattr(self, 'calibrator') and self.last_action in (1, 2, 3, 4):
                        self.calibrator.observe(self.last_action, best_dir_b)
                    self.last_action = None
                    self.last_grid = current_grid.copy()
                    return

            # If motion was blocked (wall bump)
            if self.avatar_pos is not None:
                bump_r = self.avatar_pos[0] + expected_dr * self.stride
                bump_c = self.avatar_pos[1] + expected_dc * self.stride
                if 0 <= bump_r < h and 0 <= bump_c < w:
                    self.obstacle_cells.add((bump_r, bump_c))
            self.last_action = None
            self.last_grid = current_grid.copy()

        # If avatar is still unknown, search for candidate
        if self.avatar_pos is None:
            self._detect_avatar_candidate(current_grid, bg_canvas)

    def _detect_avatar_candidate(self, grid: np.ndarray, bg_canvas: int):
        """Finds isolated single-cell or small minority clusters that resemble an avatar."""
        h, w = grid.shape
        counts = collections.Counter(grid.flatten())
        # Prioritize confirmed avatar color from previous levels of this game
        if self.avatar_color is not None and counts.get(self.avatar_color, 0) in range(1, 10):
            coords = np.argwhere(grid == self.avatar_color)
            if len(coords) >= 1:
                centroid = np.mean(coords, axis=0).astype(int)
                self.avatar_pos = (int(centroid[0]), int(centroid[1]))
                self.avatar_confidence = 0.8
                return

        # Colors that are rare but not unique background
        rare_colors = [c for c, count in counts.items() if c != bg_canvas and 1 <= count <= 9]
        for color in rare_colors:
            coords = np.argwhere(grid == color)
            if len(coords) == 1:
                self.avatar_pos = (int(coords[0][0]), int(coords[0][1]))
                self.avatar_color = color
                self.avatar_confidence = 0.5
                return
        if rare_colors:
            coords = np.argwhere(grid == rare_colors[0])
            centroid = np.mean(coords, axis=0).astype(int)
            self.avatar_pos = (int(centroid[0]), int(centroid[1]))
            self.avatar_color = rare_colors[0]
            self.avatar_confidence = 0.3

    def _update_floor_colors(self, grid: np.ndarray, bg_canvas: int):
        """Identifies traversable floor colors based on global prevalence and connectivity."""
        if self.avatar_pos is None:
            return
        h, w = grid.shape
        grid_counts = collections.Counter(grid[0:h-1].flatten())

        # Exclude border wall / perimeter frame colors
        border_pixels = np.concatenate([grid[0, :], grid[h-2, :], grid[:, 0], grid[:, w-1]])
        total_border = len(border_pixels)
        border_wall_colors = {int(c) for c, cnt in collections.Counter(border_pixels).items() if cnt / total_border >= 0.15}
        self.border_wall_colors = border_wall_colors

        ar, ac = self.avatar_pos
        stride = getattr(self, 'stride', 1)
        radius = max(3, stride + 1)
        r_min, r_max = max(0, ar - radius), min(h, ar + radius + 1)
        c_min, c_max = max(0, ac - radius), min(w, ac + radius + 1)
        sub = grid[r_min:r_max, c_min:c_max]
        counts = collections.Counter(sub.flatten())

        for color, cnt in counts.items():
            c_int = int(color)
            if c_int != bg_canvas and c_int != self.avatar_color and c_int not in border_wall_colors and cnt >= 3:
                # A true floor must be globally prominent (>= 80 pixels)
                if grid_counts.get(color, 0) >= 80:
                    self.floor_colors.add(c_int)

    def _record_junction_if_branching(self, grid: np.ndarray, bg_canvas: int):
        """Identifies and records topological junction decision points (>=3 walkable branches)."""
        if self.avatar_pos is None:
            return
        r, c = self.avatar_pos
        h, w = grid.shape
        stride = getattr(self, 'stride', 1)
        walkable_branches = 0
        unvisited_branches = 0

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr * stride, c + dc * stride
            if 0 <= nr < h - 1 and 0 <= nc < w:
                if (nr, nc) not in self.obstacle_cells and int(grid[nr, nc]) not in self.hazard_colors:
                    cval = int(grid[nr, nc])
                    is_walk = (
                        cval in self.floor_colors or
                        (not self.floor_colors and cval == bg_canvas) or
                        (nr, nc) in self.traversable_cells or
                        (self.avatar_color is not None and cval == self.avatar_color)
                    )
                    if is_walk:
                        walkable_branches += 1
                        if (nr, nc) not in self.visited_cells:
                            unvisited_branches += 1

        if walkable_branches >= 3 and unvisited_branches >= 1:
            if self.avatar_pos not in self.junction_history:
                self.junction_history.append(self.avatar_pos)
                if len(self.junction_history) > 30:
                    self.junction_history.pop(0)

    def _find_best_retreat_junction(self, grid: np.ndarray, bg_canvas: int) -> Optional[Tuple[int, int]]:
        """Finds the most recent reachable junction with open unexplored branches outside cyclic orbits."""
        if not self.junction_history or self.avatar_pos is None:
            return None
        orbit = getattr(self, 'current_cycle_orbit', set())
        for j_pos in reversed(self.junction_history):
            if j_pos == self.avatar_pos or j_pos in orbit:
                continue
            dist = abs(j_pos[0] - self.avatar_pos[0]) + abs(j_pos[1] - self.avatar_pos[1])
            if dist >= 2:
                path = self._find_path(grid, self.avatar_pos, j_pos, bg_canvas=bg_canvas)
                if path and len(path) > 1:
                    return j_pos
        return None

    def _extract_yamauchi_frontiers(self, grid: np.ndarray, bg_canvas: int) -> List[Dict[str, Any]]:
        """
        Extracts contiguous frontier boundaries separating visited from unvisited territory,
        scored by Epistemic Information Gain.
        """
        h, w = grid.shape
        stride = getattr(self, 'stride', 1)
        frontier_cells = set()
        cell_unvisited_neighbors: Dict[Tuple[int, int], Set[Tuple[int, int]]] = collections.defaultdict(set)

        # Candidates are known walkable cells
        candidate_sources = self.visited_cells.union(self.traversable_cells)
        if not candidate_sources and self.avatar_pos is not None:
            candidate_sources = {self.avatar_pos}

        for r, c in candidate_sources:
            if not (0 <= r < h - 1 and 0 <= c < w):
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr * stride, c + dc * stride
                if 0 <= nr < h - 1 and 0 <= nc < w:
                    if (nr, nc) not in self.visited_cells and (nr, nc) not in self.obstacle_cells:
                        nval = int(grid[nr, nc])
                        if nval not in self.hazard_colors and nval not in self.suspect_hazard_colors:
                            frontier_cells.add((r, c))
                            cell_unvisited_neighbors[(r, c)].add((nr, nc))

        # Fallback if no visited neighbors border unvisited space
        if not frontier_cells:
            for r in range(0, h - 1, max(1, stride)):
                for c in range(0, w, max(1, stride)):
                    if (r, c) not in self.obstacle_cells:
                        cval = int(grid[r, c])
                        if cval in self.floor_colors or (not self.floor_colors and cval == bg_canvas) or (r, c) in self.traversable_cells:
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = r + dr * stride, c + dc * stride
                                if 0 <= nr < h - 1 and 0 <= nc < w:
                                    if (nr, nc) not in self.visited_cells and (nr, nc) not in self.obstacle_cells:
                                        if int(grid[nr, nc]) not in self.hazard_colors:
                                            frontier_cells.add((r, c))
                                            cell_unvisited_neighbors[(r, c)].add((nr, nc))

        if not frontier_cells:
            return []

        # Cluster adjacent frontier cells
        visited_f = set()
        clusters = []
        for f_cell in frontier_cells:
            if f_cell in visited_f:
                continue
            cluster = []
            q = collections.deque([f_cell])
            visited_f.add(f_cell)
            unvisited_zone = set()

            while q:
                curr = q.popleft()
                cluster.append(curr)
                unvisited_zone.update(cell_unvisited_neighbors.get(curr, set()))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    adj = (curr[0] + dr * stride, curr[1] + dc * stride)
                    if adj in frontier_cells and adj not in visited_f:
                        visited_f.add(adj)
                        q.append(adj)

            coords = np.array(cluster)
            mean_r, mean_c = np.mean(coords[:, 0]), np.mean(coords[:, 1])
            best_rep = min(cluster, key=lambda p: abs(p[0] - mean_r) + abs(p[1] - mean_c))

            n_frontier = len(cluster)
            n_unvisited = len(unvisited_zone)

            # Saliency bonus: unvisited neighbors with rare entity colors
            saliency_bonus = 0.0
            for ur, uc in unvisited_zone:
                u_val = int(grid[ur, uc])
                if u_val != bg_canvas and u_val not in self.floor_colors:
                    saliency_bonus += 3.0

            info_gain = float(n_frontier + 1.5 * n_unvisited + saliency_bonus)
            dist = float(abs(best_rep[0] - self.avatar_pos[0]) + abs(best_rep[1] - self.avatar_pos[1])) if self.avatar_pos else 10.0
            epistemic_score = info_gain / (1.0 + 0.08 * (dist / float(max(1, stride))))

            clusters.append({
                'centroid': best_rep,
                'cells': cluster,
                'size': n_frontier,
                'unvisited_volume': n_unvisited,
                'epistemic_score': epistemic_score
            })

        clusters.sort(key=lambda c: c['epistemic_score'], reverse=True)
        return clusters

    def _select_next_target(self, grid: np.ndarray, bg_canvas: int) -> Optional[Tuple[int, int]]:
        """
        Selects the highest-priority epistemic target:
        1. Backtracking retreat junction if trapped or oscillating.
        2. Reachable unvisited salient entity (Object, Key, Switch) not on cooldown.
        3. Yamauchi frontier cluster with highest Epistemic Information Gain.
        4. Geometric frontier fallback.
        """
        h, w = grid.shape
        if self.avatar_pos is None:
            return None

        # 1. Retreat Target if trapped/oscillating
        if self.retreat_target is not None and self.retreat_target != self.avatar_pos:
            path = self._find_path(grid, self.avatar_pos, self.retreat_target, bg_canvas=bg_canvas)
            if path and len(path) > 1:
                return self.retreat_target
            else:
                self.retreat_target = None

        # 2. Unvisited Salient Entities (Objects, Receptors, Keys)
        salient_entities = self._extract_salient_entities(grid, bg_canvas)
        unvisited = [
            e for e in salient_entities
            if e['centroid'] != self.avatar_pos
            and e['centroid'] not in self.visited_cells
            and e['centroid'] not in self.interacted_targets
            and e['centroid'] not in self.unreachable_cooldown
            and (abs(e['centroid'][0] - self.avatar_pos[0]) + abs(e['centroid'][1] - self.avatar_pos[1]) > 0)
        ]

        if unvisited:
            unvisited.sort(key=lambda e: (e['size'], abs(e['centroid'][0] - self.avatar_pos[0]) + abs(e['centroid'][1] - self.avatar_pos[1])))
            for cand in unvisited:
                pos = cand['centroid']
                path = self._find_path(grid, self.avatar_pos, pos, bg_canvas=bg_canvas)
                if path and len(path) > 1:
                    return pos
                else:
                    self.target_failure_counts[pos] += 1
                    if self.target_failure_counts[pos] >= 2:
                        self.unreachable_cooldown[pos] = 25

        # 3. Epistemic Yamauchi Frontier Clusters
        frontier_clusters = self._extract_yamauchi_frontiers(grid, bg_canvas)
        self.epistemic_frontiers = frontier_clusters

        for cluster in frontier_clusters:
            c_pos = cluster['centroid']
            if c_pos in self.unreachable_cooldown or c_pos == self.avatar_pos:
                continue
            path = self._find_path(grid, self.avatar_pos, c_pos, bg_canvas=bg_canvas)
            if path and len(path) > 1:
                return c_pos
            else:
                self.unreachable_cooldown[c_pos] = 20

        # 4. Fallback Frontier
        furthest = self._find_furthest_frontier(grid, bg_canvas)
        if furthest is not None and furthest != self.avatar_pos and furthest not in self.unreachable_cooldown:
            return furthest

        return None

    def _extract_salient_entities(self, grid: np.ndarray, bg_canvas: int) -> List[Dict[str, Any]]:
        """Finds distinct non-canvas, non-floor connected components."""
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        entities = []

        for r in range(h):
            for c in range(w):
                color = grid[r, c]
                if color != bg_canvas and color not in self.floor_colors and not visited[r, c]:
                    if self.avatar_pos == (r, c):
                        visited[r, c] = True
                        continue

                    cells = []
                    q = collections.deque([(r, c)])
                    visited[r, c] = True

                    while q:
                        cr, cc = q.popleft()
                        cells.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                q.append((nr, nc))

                    if self.avatar_pos is not None and self.avatar_pos in cells:
                        continue

                    # Filter out giant walls (> 20% of grid)
                    if 1 <= len(cells) <= int(h * w * 0.20):
                        coords = np.array(cells)
                        centroid = (int(np.round(np.mean(coords[:, 0]))), int(np.round(np.mean(coords[:, 1]))))
                        if centroid != self.avatar_pos:
                            entities.append({
                                'color': color,
                                'size': len(cells),
                                'cells': cells,
                                'centroid': centroid
                            })
        return entities

    def _find_furthest_frontier(self, grid: np.ndarray, bg_canvas: int) -> Optional[Tuple[int, int]]:
        """Yamauchi-style frontier exploration: Finds unvisited traversable cells."""
        h, w = grid.shape
        candidates = []
        for r in range(0, h, 2):
            for c in range(0, w, 2):
                if (r, c) not in self.visited_cells and (r, c) not in self.obstacle_cells:
                    if grid[r, c] == bg_canvas or grid[r, c] in self.floor_colors or (r, c) in self.traversable_cells:
                        candidates.append((r, c))

        if not candidates or self.avatar_pos is None:
            return None

        candidates.sort(key=lambda c: abs(c[0] - self.avatar_pos[0]) + abs(c[1] - self.avatar_pos[1]))
        return candidates[0]

    def _find_adjacent_uninteracted_entities(self, grid: np.ndarray, bg_canvas: int) -> List[Tuple[int, int]]:
        """Checks if avatar is currently adjacent to an uninteracted entity."""
        if self.avatar_pos is None:
            return []
        h, w = grid.shape
        r, c = self.avatar_pos
        adjacent_targets = []
        border_walls = getattr(self, 'border_wall_colors', set())
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if (nr, nc) in self.obstacle_cells:
                    continue
                cval = int(grid[nr, nc])
                if cval != bg_canvas and cval not in self.floor_colors and cval not in border_walls and (nr, nc) not in self.interacted_targets:
                    adjacent_targets.append((nr, nc))
        return adjacent_targets

    def _find_path(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        bg_canvas: int = 0,
        **kwargs
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Heuristic A* shortest path from start to goal (or adjacent to goal).
        Preserves turn budget by directing exploration greedily along minimal Manhattan distance,
        supporting multi-scale strides, composite entity corridor validation, and hazard evasion.
        """
        if 'bg_color' in kwargs:
            bg_canvas = kwargs['bg_color']
        h, w = grid.shape
        stride = getattr(self, 'stride', 1)

        goal_color = grid[goal[0], goal[1]]
        is_goal_solid = (goal_color not in self.floor_colors and goal_color != bg_canvas and goal not in self.traversable_cells)

        target_color = kwargs.get('target_color')
        reach_exact = kwargs.get('reach_exact', False)
        goal_cells = kwargs.get('goal_cells')
        avatar_cells = kwargs.get('avatar_cells')
        max_expansions = kwargs.get('max_expansions', 2000)

        def _is_direct_los_clear(c1: Tuple[int, int], c2: Tuple[int, int]) -> bool:
            """Checks if straight-line corridor between c1 and c2 is free of walls, obstacles, and hazards."""
            if c1[0] != c2[0] and c1[1] != c2[1]:
                return False
            step_r = 1 if c2[0] > c1[0] else (-1 if c2[0] < c1[0] else 0)
            step_c = 1 if c2[1] > c1[1] else (-1 if c2[1] < c1[1] else 0)
            cr, cc = c1[0] + step_r, c1[1] + step_c
            while (cr, cc) != c2:
                if not (0 <= cr < h and 0 <= cc < w):
                    return False
                if (cr, cc) in self.obstacle_cells:
                    return False
                cval = grid[cr, cc]
                if cval in self.hazard_colors or cval in self.suspect_hazard_colors:
                    return False
                is_walk = (
                    cval in self.floor_colors or
                    (not self.floor_colors and cval == bg_canvas) or
                    (target_color is not None and cval == target_color) or
                    cval == goal_color or
                    (self.avatar_color is not None and cval == self.avatar_color) or
                    (cr, cc) in self.traversable_cells or
                    (goal_cells is not None and (cr, cc) in goal_cells) or
                    (avatar_cells is not None and (cr, cc) in avatar_cells)
                )
                if not is_walk:
                    return False
                cr += step_r
                cc += step_c
            return True

        def heuristic(p: Tuple[int, int]) -> float:
            dist = abs(p[0] - goal[0]) + abs(p[1] - goal[1])
            if reach_exact or not is_goal_solid:
                return dist / float(stride)
            return max(0.0, float(dist - stride)) / float(stride)

        # Priority queue: (f_score, h_score, tie_breaker, path)
        h0 = heuristic(start)
        pq: List[Tuple[float, float, int, List[Tuple[int, int]]]] = [(h0, h0, 0, [start])]
        g_scores: Dict[Tuple[int, int], int] = {start: 0}
        counter = 0
        expansions = 0

        while pq and expansions < max_expansions:
            expansions += 1
            f, h_val, _, path = heapq.heappop(pq)
            curr = path[-1]
            curr_g = len(path) - 1

            if curr_g > g_scores.get(curr, float('inf')):
                continue

            dist_to_goal = abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])
            if reach_exact or not is_goal_solid:
                is_reached = (curr == goal)
            else:
                is_reached = (
                    curr == goal or
                    (dist_to_goal <= 1 and _is_direct_los_clear(curr, goal)) or
                    (dist_to_goal <= stride and (curr[0] == goal[0] or curr[1] == goal[1]) and _is_direct_los_clear(curr, goal))
                )

            if is_reached and len(path) > 1:
                return path

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr * stride, curr[1] + dc * stride
                neighbor = (nr, nc)

                if not (0 <= nr < h and 0 <= nc < w):
                    continue

                if neighbor in self.obstacle_cells:
                    continue

                cell_val = grid[nr, nc]
                if (cell_val in self.hazard_colors or cell_val in self.suspect_hazard_colors) and neighbor != goal:
                    continue

                is_traversable = (
                    cell_val in self.floor_colors or
                    (not self.floor_colors and cell_val == bg_canvas) or
                    (target_color is not None and cell_val == target_color) or
                    (self.avatar_color is not None and cell_val == self.avatar_color) or
                    cell_val == goal_color or
                    neighbor in self.traversable_cells or
                    neighbor == goal or
                    (goal_cells is not None and neighbor in goal_cells) or
                    (avatar_cells is not None and neighbor in avatar_cells)
                )

                if is_traversable and stride > 1:
                    for k in range(1, stride):
                        kr, kc = curr[0] + dr * k, curr[1] + dc * k
                        if not (0 <= kr < h and 0 <= kc < w):
                            is_traversable = False
                            break
                        kval = grid[kr, kc]
                        if (kr, kc) in self.obstacle_cells or kval in self.hazard_colors:
                            is_traversable = False
                            break
                        is_mid_walkable = (
                            kval in self.floor_colors or
                            (not self.floor_colors and kval == bg_canvas) or
                            (kr, kc) in self.traversable_cells or
                            kval == target_color or kval == goal_color or
                            (self.avatar_color is not None and kval == self.avatar_color) or
                            (goal_cells is not None and (kr, kc) in goal_cells) or
                            (avatar_cells is not None and (kr, kc) in avatar_cells)
                        )
                        if not is_mid_walkable:
                            is_traversable = False
                            break

                # Kinematic Irreversibility & Pit Trap Guard under Gravity
                if is_traversable and self.gravity_vector is not None and self.gravity_vector != (0, 0):
                    try:
                        from .platformer_planner import is_kinematically_irreversible_drop
                    except ImportError:
                        from platformer_planner import is_kinematically_irreversible_drop

                    solids_set = set(self.obstacle_cells)
                    if self.physics_inducer:
                        solids_set |= self.physics_inducer.solid_colors

                    is_irrev, is_fatal, _, land_p = is_kinematically_irreversible_drop(
                        grid=grid,
                        from_pos=curr,
                        to_pos=neighbor,
                        gravity=self.gravity_vector,
                        jump_height=getattr(self, 'jump_height', 2),
                        jump_reach=2,
                        solid_colors=solids_set,
                        hazard_colors=self.hazard_colors,
                        suspect_hazard_colors=self.suspect_hazard_colors,
                        bg_canvas=bg_canvas,
                        stride=stride,
                        goal_candidates={goal}
                    )
                    if is_fatal or (is_irrev and neighbor != goal and land_p != goal):
                        is_traversable = False

                if is_traversable:
                    new_g = curr_g + 1
                    if new_g < g_scores.get(neighbor, float('inf')):
                        g_scores[neighbor] = new_g
                        h_new = heuristic(neighbor)
                        f_new = new_g + h_new
                        counter += 1
                        heapq.heappush(pq, (f_new, h_new, counter, path + [neighbor]))

        return None

    def _direction_to_action(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        dr = int(np.sign(to_pos[0] - from_pos[0]))
        dc = int(np.sign(to_pos[1] - from_pos[1]))
        cardinal = (dr, dc)
        if hasattr(self, 'calibrator') and self.calibrator is not None:
            return self.calibrator.get_action_for_direction(cardinal)
        if dr < 0 and dc == 0:
            return 1 # UP
        elif dr > 0 and dc == 0:
            return 2 # DOWN
        elif dr == 0 and dc < 0:
            return 3 # LEFT
        elif dr == 0 and dc > 0:
            return 4 # RIGHT
        return 1

    def _action_to_delta(self, action: int) -> Tuple[int, int]:
        if hasattr(self, 'calibrator') and self.calibrator is not None:
            return self.calibrator.get_delta_for_action(action)
        mapping = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        return mapping.get(action, (0, 0))

    def _fallback_action(self, available_actions: List[int], grid: np.ndarray, bg_canvas: int) -> int:
        """Picks a safe legal navigation action that avoids immediate obstacles, traps, and cycles."""
        h, w = grid.shape
        nav_legal = [a for a in [1, 2, 3, 4] if a in available_actions]
        if not nav_legal:
            return available_actions[0]

        if self.avatar_pos is not None:
            r, c = self.avatar_pos
            stride = getattr(self, 'stride', 1)
            scored_moves: List[Tuple[float, int]] = []

            for a in nav_legal:
                dr, dc = self._action_to_delta(a)
                nr, nc = r + dr * stride, c + dc * stride
                if 0 <= nr < h and 0 <= nc < w:
                    val = int(grid[nr, nc])
                    if val in self.hazard_colors or val in self.suspect_hazard_colors:
                        continue
                    if (nr, nc) in self.obstacle_cells:
                        continue

                    solids_set = set(self.obstacle_cells) | getattr(self, 'border_wall_colors', set())
                    if self.physics_inducer:
                        solids_set |= self.physics_inducer.solid_colors

                    if val in solids_set:
                        continue

                    if self.floor_colors and val not in self.floor_colors and val != bg_canvas and (nr, nc) not in self.traversable_cells:
                        continue

                    # Kinematic Irreversibility & Pit Trap Guard for Fallback Navigation
                    if self.gravity_vector is not None and self.gravity_vector != (0, 0):
                        try:
                            from .platformer_planner import is_kinematically_irreversible_drop
                        except ImportError:
                            from platformer_planner import is_kinematically_irreversible_drop

                        solids_set = set(self.obstacle_cells)
                        if self.physics_inducer:
                            solids_set |= self.physics_inducer.solid_colors

                        is_irrev, is_fatal, _, land_pos = is_kinematically_irreversible_drop(
                            grid=grid,
                            from_pos=(r, c),
                            to_pos=(nr, nc),
                            gravity=self.gravity_vector,
                            jump_height=getattr(self, 'jump_height', 2),
                            jump_reach=2,
                            solid_colors=solids_set,
                            hazard_colors=self.hazard_colors,
                            suspect_hazard_colors=self.suspect_hazard_colors,
                            bg_canvas=bg_canvas,
                            stride=stride
                        )
                        if is_fatal or is_irrev:
                            continue
                        if land_pos == (r, c) and (nr, nc) != (r, c):
                            # Hop in place with 0 net displacement under gravity
                            continue

                    # Validate intermediate cells if stride > 1
                    is_stride_blocked = False
                    if stride > 1:
                        for k in range(1, stride):
                            kr, kc = r + dr * k, c + dc * k
                            if not (0 <= kr < h and 0 <= kc < w) or (kr, kc) in self.obstacle_cells or int(grid[kr, kc]) in self.hazard_colors:
                                is_stride_blocked = True
                                break
                    if is_stride_blocked:
                        continue

                    # Recency penalty: how often has (nr, nc) appeared in recent_positions?
                    recency_penalty = list(self.recent_positions).count((nr, nc))
                    is_visited = 1 if (nr, nc) in self.visited_cells else 0

                    # Count open neighbors from (nr, nc) to avoid 1-cell dead ends
                    open_neighbors = 0
                    for ndr, ndc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nnr, nnc = nr + ndr * stride, nc + ndc * stride
                        if 0 <= nnr < h and 0 <= nnc < w and (nnr, nnc) not in self.obstacle_cells:
                            if int(grid[nnr, nnc]) not in self.hazard_colors:
                                open_neighbors += 1

                    orbit = getattr(self, 'current_cycle_orbit', set())
                    orbit_penalty = 25.0 if (nr, nc) in orbit else 0.0
                    move_score = 10.0 - 5.0 * recency_penalty - 3.0 * is_visited - orbit_penalty + 1.0 * open_neighbors
                    scored_moves.append((move_score, a))

            if scored_moves:
                scored_moves.sort(key=lambda x: x[0], reverse=True)
                return scored_moves[0][1]

        # If stuck on last action, cycle through legal moves to prevent blind repetitive wall-bumping
        if self.last_action in nav_legal and self.consecutive_stuck > 0:
            idx = nav_legal.index(self.last_action)
            return nav_legal[(idx + 1) % len(nav_legal)]

        return nav_legal[0]
