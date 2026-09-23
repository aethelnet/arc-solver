"""
=============================================================================
ARC-AGI CAUSAL META-TOOL SELECTOR: OBJECTIVE HYPOTHESIS INDUCER
=============================================================================
Induces the latent teleological objective of unclassified ARC games online:
1. Perceptual morphological decomposition (Hollow Receptors, Movable Blocks,
   Goal Tiles, Interactive Switch Cards, Collectible Tokens).
2. Bayesian scoring across formal hypothesis space:
     H_Receptor   : Move all matching blocks into hollow receptor frames.
     H_GoalTile   : Navigate avatar to rare exit / destination tile.
     H_Collect    : Eliminate / collect all target tokens.
     H_SwitchGates: Toggle control switches to remove barriers to goal.
     H_Explore    : Unstructured frontier exploration fallback.
3. Computes objective potential Phi(s) and synthesizes tailored SMDP Subgoals.
=============================================================================
"""

from enum import Enum
from typing import List, Dict, Any, Tuple, Optional, Set
import collections
import numpy as np

from .options_framework import Subgoal, SubgoalType, SubgoalStatus


class HypothesisType(Enum):
    RECEPTOR_MATCHING = "RECEPTOR_MATCHING"
    GOAL_TILE = "GOAL_TILE"
    COLLECT_ITEMS = "COLLECT_ITEMS"
    SWITCH_GATES = "SWITCH_GATES"
    EXPLORATION = "EXPLORATION"


class ReceptorTarget:
    def __init__(
        self,
        color: int,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        inner_size: Tuple[int, int] = (1, 1),
        center: Tuple[int, int] = (0, 0)
    ):
        self.color = color
        self.bbox = bbox if bbox is not None else (center[0], center[1], center[0], center[1])
        self.inner_size = inner_size
        self.center = center  # Target destination inside the frame

    def __repr__(self) -> str:
        return f"Receptor(color={self.color}, inner_size={self.inner_size}, center={self.center})"


class MovableBlock:
    def __init__(
        self,
        color: int,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        size: Tuple[int, int] = (1, 1),
        center: Tuple[int, int] = (0, 0),
        is_active: bool = False
    ):
        self.color = color
        self.bbox = bbox if bbox is not None else (center[0], center[1], center[0], center[1])
        self.size = size
        self.center = center
        self.is_active = is_active

    def __repr__(self) -> str:
        return f"Block(color={self.color}, size={self.size}, center={self.center}, active={self.is_active})"


class GoalTarget:
    def __init__(self, color: int, bbox: Tuple[int, int, int, int], size: Tuple[int, int], center: Tuple[int, int]):
        self.color = color
        self.bbox = bbox
        self.size = size
        self.center = center

    def __repr__(self) -> str:
        return f"GoalTarget(color={self.color}, size={self.size}, center={self.center})"


class ClickWidget:
    def __init__(self, color: int, bbox: Tuple[int, int, int, int], size: Tuple[int, int], center: Tuple[int, int], widget_type: str = "button"):
        self.color = color
        self.bbox = bbox
        self.size = size
        self.center = center
        self.widget_type = widget_type

    def __repr__(self) -> str:
        return f"ClickWidget({self.widget_type}, color={self.color}, center={self.center})"


class ObjectiveHypothesis:
    """
    Representation of the inferred game objective.
    """
    def __init__(
        self,
        hypo_type: HypothesisType,
        confidence: float,
        potential: float,
        targets: List[Any],
        movable_units: List[Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.hypo_type = hypo_type
        self.confidence = confidence
        self.potential = potential  # Phi(s): distance to completion
        self.targets = targets
        self.movable_units = movable_units
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (f"ObjectiveHypothesis({self.hypo_type.value}, conf={self.confidence:.2f}, "
                f"Phi={self.potential:.1f}, targets={len(self.targets)}, units={len(self.movable_units)})")


from itertools import permutations

def optimal_bipartite_assignment(blocks: List[MovableBlock], receptors: List[ReceptorTarget]) -> Dict[int, int]:
    """
    Computes optimal 1-to-1 matching from blocks to receptors minimizing total Manhattan distance,
    giving strong priority to matching block.size == receptor.inner_size.
    """
    n_blocks = len(blocks)
    n_receptors = len(receptors)
    if n_blocks == 0 or n_receptors == 0:
        return {}

    if n_blocks <= n_receptors and n_receptors <= 8:
        best_cost = float('inf')
        best_perm = None
        for p in permutations(range(n_receptors), n_blocks):
            cost = 0.0
            for i in range(n_blocks):
                b = blocks[i]
                r = receptors[p[i]]
                dist = abs(b.center[0] - r.center[0]) + abs(b.center[1] - r.center[1])
                # Penalty if sizes don't match
                size_penalty = 0.0 if b.size == r.inner_size else 500.0
                cost += dist + size_penalty
            if cost < best_cost:
                best_cost = cost
                best_perm = p
        if best_perm is not None:
            return {i: best_perm[i] for i in range(n_blocks)}

    # Greedy assignment without replacement fallback
    assigned: Dict[int, int] = {}
    avail_receptors = set(range(n_receptors))
    for i, b in enumerate(blocks):
        if not avail_receptors:
            break
        # Filter matching sizes first
        same_size = [r_idx for r_idx in avail_receptors if receptors[r_idx].inner_size == b.size]
        cand_receptors = same_size if same_size else list(avail_receptors)
        best_r = min(
            cand_receptors,
            key=lambda r_idx: abs(b.center[0] - receptors[r_idx].center[0]) + abs(b.center[1] - receptors[r_idx].center[1])
        )
        assigned[i] = best_r
        avail_receptors.remove(best_r)

    return assigned


def is_reachable(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    bg_canvas: int,
    floor_colors: Set[int],
    target_color: Optional[int] = None,
    stride: int = 1
) -> bool:
    """Fast BFS reachability check to test if a path exists between start and goal."""
    grid = np.asarray(grid)
    while grid.ndim > 2:
        grid = grid[-1]
    h, w = grid.shape
    q = collections.deque([start])
    visited = {start}
    goal_color = int(grid[goal[0], goal[1]])
    while q:
        curr = q.popleft()
        if curr == goal or abs(curr[0] - goal[0]) + abs(curr[1] - goal[1]) <= stride:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = curr[0] + dr * stride, curr[1] + dc * stride
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                val = int(grid[nr, nc])
                is_trav = (
                    val in floor_colors or
                    (target_color is not None and val == target_color) or
                    val == goal_color or
                    (val == bg_canvas and (not floor_colors or bg_canvas in floor_colors))
                )
                if is_trav:
                    visited.add((nr, nc))
                    q.append((nr, nc))
    return False


def find_block_path_bfs(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    bg_canvas: int,
    floor_colors: Set[int],
    target_color: Optional[int] = None,
    obstacle_cells: Optional[Set[Tuple[int, int]]] = None,
    stride: int = 1
) -> Optional[List[Tuple[int, int]]]:
    """Finds shortest BFS path for a block between start and goal."""
    grid = np.asarray(grid)
    while grid.ndim > 2:
        grid = grid[-1]
    h, w = grid.shape
    obstacles = obstacle_cells or set()

    q = collections.deque([[start]])
    visited = {start}
    goal_color = int(grid[goal[0], goal[1]]) if 0 <= goal[0] < h and 0 <= goal[1] < w else None

    while q:
        path = q.popleft()
        curr = path[-1]
        if curr == goal or abs(curr[0] - goal[0]) + abs(curr[1] - goal[1]) <= stride // 2:
            return path

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = curr[0] + dr * stride, curr[1] + dc * stride
            nbr = (nr, nc)
            if 0 <= nr < h and 0 <= nc < w and nbr not in visited and nbr not in obstacles:
                val = int(grid[nr, nc])
                is_trav = (
                    val in floor_colors or
                    (target_color is not None and val == target_color) or
                    (goal_color is not None and val == goal_color) or
                    (val == bg_canvas and (not floor_colors or bg_canvas in floor_colors))
                )
                if is_trav:
                    visited.add(nbr)
                    q.append(path + [nbr])
    return None


def is_corner_deadlock_cell(
    grid: np.ndarray,
    pos: Tuple[int, int],
    bg_canvas: int,
    floor_colors: Set[int],
    stride: int = 1
) -> bool:
    """Checks if a position pos is in an irreversible 2-wall corner deadlock."""
    r, c = pos
    h, w = grid.shape

    def is_solid(cell):
        cr, cc = cell
        if not (0 <= cr < h and 0 <= cc < w):
            return True
        v = int(grid[cr, cc])
        if floor_colors and v in floor_colors:
            return False
        return v != bg_canvas

    up = is_solid((r - stride, c))
    down = is_solid((r + stride, c))
    left = is_solid((r, c - stride))
    right = is_solid((r, c + stride))

    return (up and left) or (up and right) or (down and left) or (down and right)


def find_staging_cell(
    grid: np.ndarray,
    start: Tuple[int, int],
    reserved_cells: Set[Tuple[int, int]],
    bg_canvas: int,
    floor_colors: Set[int],
    stride: int = 1,
    max_dist: int = 5
) -> Optional[Tuple[int, int]]:
    """Finds a safe buffer cell near start that is not in reserved_cells or a corner deadlock."""
    grid = np.asarray(grid)
    while grid.ndim > 2:
        grid = grid[-1]
    h, w = grid.shape

    q = collections.deque([(start, 0)])
    visited = {start}

    while q:
        curr, dist = q.popleft()
        if dist > 0 and curr not in reserved_cells:
            val = int(grid[curr[0], curr[1]])
            if val == bg_canvas or (floor_colors and val in floor_colors):
                if not is_corner_deadlock_cell(grid, curr, bg_canvas, floor_colors, stride):
                    return curr

        if dist < max_dist:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr * stride, curr[1] + dc * stride
                nbr = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w and nbr not in visited:
                    visited.add(nbr)
                    val = int(grid[nr, nc])
                    if val == bg_canvas or (floor_colors and val in floor_colors):
                        q.append((nbr, dist + 1))
    return None


def topological_block_ordering(
    blocks: List[MovableBlock],
    receptors: List[ReceptorTarget],
    assignment: Dict[int, int],
    unseated_indices: List[int],
    grid: np.ndarray,
    avatar_pos: Optional[Tuple[int, int]],
    bg_canvas: int,
    floor_colors: Set[int],
    stride: int = 1
) -> Tuple[List[int], Dict[int, Tuple[int, int]]]:
    """
    Computes a deadlock-free topological delivery order for multiple blocks to receptors.
    
    Returns:
        (ordered_block_indices: List[int], staging_targets: Dict[int, Tuple[int, int]])
        - ordered_block_indices: execution sequence of block indices.
        - staging_targets: mapping block_idx -> staging_cell if temporary buffering is required.
    """
    if len(unseated_indices) <= 1:
        return list(unseated_indices), {}

    grid = np.asarray(grid)
    while grid.ndim > 2:
        grid = grid[-1]
    h, w = grid.shape

    # 1. Compute nominal trajectories for each unseated block
    paths: Dict[int, List[Tuple[int, int]]] = {}
    path_cells: Dict[int, Set[Tuple[int, int]]] = {}
    target_positions: Dict[int, Tuple[int, int]] = {}
    start_positions: Dict[int, Tuple[int, int]] = {}

    for idx in unseated_indices:
        b = blocks[idx]
        r_idx = assignment.get(idx)
        if r_idx is None or r_idx >= len(receptors):
            continue
        rec = receptors[r_idx]
        start_positions[idx] = b.center
        target_positions[idx] = rec.center

        p = find_block_path_bfs(
            grid, b.center, rec.center, bg_canvas, floor_colors,
            target_color=rec.color, stride=stride
        )
        if p is not None:
            paths[idx] = p
            path_cells[idx] = set(p)
        else:
            paths[idx] = [b.center]
            path_cells[idx] = {b.center}

    # 2. Build Dependency Precedence Graph G = (V, E)
    # Edge j -> i means block j must precede block i (B_j ≺ B_i)
    edges: Set[Tuple[int, int]] = set()

    for i in unseated_indices:
        if i not in target_positions:
            continue
        Ti = target_positions[i]
        Si = start_positions[i]
        for j in unseated_indices:
            if i == j or j not in target_positions:
                continue
            Tj = target_positions[j]
            Sj = start_positions[j]
            Pj = path_cells.get(j, set())
            Pi = path_cells.get(i, set())

            # Rule A: Target Obstruction
            # If placing block i at Ti lies on block j's path Pj (excluding final arrival if shared),
            # then block j must be delivered before block i blocks the hallway/receptor!
            if Ti in Pj and Ti != Tj:
                edges.add((j, i))

            # Rule B: Start Position Obstruction
            # If block j's initial position Sj is on block i's path Pi:
            # Block j must move out of the way before block i can pass!
            if Sj in Pi and Sj != Si:
                edges.add((j, i))

            # Rule C: Alcove Depth Ordering (Deepest Target First)
            # If receptor Tj is deeper into a dead-end corridor than Ti
            direct_dist_to_Tj = abs(Si[0] - Tj[0]) + abs(Si[1] - Tj[1])
            dist_to_Ti = abs(Si[0] - Ti[0]) + abs(Si[1] - Ti[1])
            if (Ti in Pj) or (abs(Ti[0] - Tj[0]) + abs(Ti[1] - Tj[1]) <= 2 and direct_dist_to_Tj > dist_to_Ti):
                edges.add((j, i))

    # 3. Detect Cycles and Synthesize Staging Buffers
    staging_targets: Dict[int, Tuple[int, int]] = {}

    for i in unseated_indices:
        for j in unseated_indices:
            if i < j and (j, i) in edges and (i, j) in edges:
                # Mutual deadlock detected! One block must be staged in a buffer siding
                stage_idx = i
                other_idx = j
                if avatar_pos is not None:
                    d_i = abs(avatar_pos[0] - start_positions[i][0]) + abs(avatar_pos[1] - start_positions[i][1])
                    d_j = abs(avatar_pos[0] - start_positions[j][0]) + abs(avatar_pos[1] - start_positions[j][1])
                    if d_j < d_i:
                        stage_idx = j
                        other_idx = i

                U = find_staging_cell(
                    grid, start_positions[stage_idx], path_cells.get(other_idx, set()),
                    bg_canvas, floor_colors, stride
                )
                if U is not None:
                    staging_targets[stage_idx] = U
                    edges.discard((other_idx, stage_idx))
                    edges.add((stage_idx, other_idx))

    # 4. Topological Sort (Kahn's Algorithm)
    in_degrees: Dict[int, int] = {idx: 0 for idx in unseated_indices}
    for j, i in edges:
        if i in in_degrees and j in in_degrees:
            in_degrees[i] += 1

    ordered: List[int] = []
    candidates = [idx for idx, deg in in_degrees.items() if deg == 0]

    while candidates:
        if avatar_pos is not None:
            candidates.sort(key=lambda idx: abs(avatar_pos[0] - start_positions[idx][0]) + abs(avatar_pos[1] - start_positions[idx][1]))
        curr = candidates.pop(0)
        ordered.append(curr)

        for j, i in list(edges):
            if j == curr:
                edges.discard((j, i))
                if i in in_degrees:
                    in_degrees[i] -= 1
                    if in_degrees[i] == 0:
                        candidates.append(i)

    # Any remaining unseated blocks (e.g. from unresolved complex cycles) append at the end
    for idx in unseated_indices:
        if idx not in ordered:
            ordered.append(idx)

    return ordered, staging_targets


class ObjectiveHypothesisInducer:
    """
    Domain-agnostic Bayesian hypothesis engine for unseen ARC-AGI-3 puzzles.
    """
    def __init__(self):
        self.current_hypothesis: Optional[ObjectiveHypothesis] = None
        self.hypo_priors: Dict[HypothesisType, float] = {
            HypothesisType.RECEPTOR_MATCHING: 0.20,
            HypothesisType.GOAL_TILE: 0.25,
            HypothesisType.COLLECT_ITEMS: 0.20,
            HypothesisType.SWITCH_GATES: 0.15,
            HypothesisType.EXPLORATION: 0.20
        }
        self.prev_potential: Optional[float] = None
        self.subgoal_counter: int = 0

    def reset(self):
        self.current_hypothesis = None
        self.prev_potential = None
        self.subgoal_counter = 0

    def detect_receptors_and_blocks(
        self,
        grid: np.ndarray,
        bg_canvas: int,
        floor_colors: Set[int]
    ) -> Tuple[List[ReceptorTarget], List[MovableBlock]]:
        """
        Scans for hollow rectangular frames (receptors) and matching solid blocks.
        """
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        components = []

        # Exclude row 63 HUD
        for r in range(h - 1):
            for c in range(w):
                val = int(grid[r, c])
                if not visited[r, c] and val != bg_canvas and val not in floor_colors:
                    comp = []
                    q = [(r, c)]
                    visited[r, c] = True
                    while q:
                        cr, cc = q.pop(0)
                        comp.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h - 1 and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == val:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    components.append((val, comp))

        receptors: List[ReceptorTarget] = []
        candidate_blocks: List[MovableBlock] = []

        for color, comp in components:
            coords = np.array(comp)
            rmin, cmin = coords.min(axis=0)
            rmax, cmax = coords.max(axis=0)
            ch = int(rmax - rmin + 1)
            cw = int(cmax - cmin + 1)

            # Check if hollow outline frame (must not be outer bounding wall of entire grid):
            if ch >= 4 and cw >= 4 and (ch < h - 2 or cw < w - 2) and (ch * cw <= h * w * 0.40):
                inner = grid[rmin+1:rmax, cmin+1:cmax]
                # Interior must be mostly floor or canvas
                is_hollow = np.mean(np.isin(inner, list(floor_colors) + [bg_canvas])) >= 0.70
                if is_hollow:
                    center = (int((rmin + rmax) // 2), int((cmin + cmax) // 2))
                    receptors.append(ReceptorTarget(
                        color=color,
                        bbox=(int(rmin), int(cmin), int(rmax), int(cmax)),
                        inner_size=(ch - 2, cw - 2),
                        center=center
                    ))
                    continue

            # Check if compact movable block (size 2..8)
            if 2 <= ch <= 8 and 2 <= cw <= 8 and len(comp) <= 64:
                center = (int((rmin + rmax) // 2), int((cmin + cmax) // 2))
                candidate_blocks.append(MovableBlock(
                    color=color,
                    bbox=(int(rmin), int(cmin), int(rmax), int(cmax)),
                    size=(ch, cw),
                    center=center
                ))

        # Filter candidate blocks: Keep only blocks whose dimensions match receptor inner sizes
        if receptors:
            valid_sizes = {r.inner_size for r in receptors}
            matched_blocks = [b for b in candidate_blocks if b.size in valid_sizes]
            return receptors, matched_blocks

        return [], []

    def detect_goal_tiles(
        self,
        grid: np.ndarray,
        bg_canvas: int,
        floor_colors: Set[int],
        avatar_pos: Optional[Tuple[int, int]],
        blocks: Optional[List[MovableBlock]] = None
    ) -> List[GoalTarget]:
        """
        Scans for rare destination / exit tiles on the floor.
        Excludes interior pixels of movable blocks or avatar.
        """
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        h, w = grid.shape
        counts = collections.Counter(grid[0:h-1].flatten())
        # Target candidates are rare (<= 9 pixels) non-canvas, non-floor colors
        rare_colors = [c for c, count in counts.items() if c != bg_canvas and c not in floor_colors and 1 <= count <= 9]

        goals = []
        for color in rare_colors:
            coords = np.argwhere(grid[0:h-1] == color)
            if len(coords) == 0:
                continue
            rmin, cmin = coords.min(axis=0)
            rmax, cmax = coords.max(axis=0)
            ch = int(rmax - rmin + 1)
            cw = int(cmax - cmin + 1)

            # Check if it resembles an isolated goal tile (1x1 to 3x3)
            if 1 <= ch <= 3 and 1 <= cw <= 3:
                center = (int((rmin + rmax) // 2), int((cmin + cmax) // 2))
                # Exclude if center is inside any block of a different color
                if blocks and any(b.color != color and b.bbox[0] <= center[0] <= b.bbox[2] and b.bbox[1] <= center[1] <= b.bbox[3] for b in blocks):
                    continue
                if avatar_pos is None or (abs(center[0] - avatar_pos[0]) > 2 or abs(center[1] - avatar_pos[1]) > 2):
                    goals.append(GoalTarget(color=int(color), bbox=(int(rmin), int(cmin), int(rmax), int(cmax)), size=(ch, cw), center=center))

        return goals

    def detect_click_widgets(
        self,
        grid: np.ndarray,
        bg_canvas: int,
        available_actions: List[int],
        blocks: List[MovableBlock]
    ) -> List[ClickWidget]:
        """
        Detects clickable interactive cards, control switches, or selectable blocks.
        """
        if 6 not in available_actions:
            return []

        h, w = grid.shape
        widgets: List[ClickWidget] = []

        # 1. Blocks with Click action (multi-unit select)
        for b in blocks:
            widgets.append(ClickWidget(
                color=b.color,
                bbox=b.bbox,
                size=b.size,
                center=b.center,
                widget_type="unit"
            ))

        # 2. Side-panel control buttons (e.g. dc22 cards of size 10..100)
        visited = np.zeros((h, w), dtype=bool)
        for r in range(h - 1):
            for c in range(w):
                val = int(grid[r, c])
                if not visited[r, c] and val != bg_canvas:
                    comp = []
                    q = [(r, c)]
                    visited[r, c] = True
                    while q:
                        cr, cc = q.pop(0)
                        comp.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h - 1 and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == val:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    if 15 <= len(comp) <= 120:
                        coords = np.array(comp)
                        rmin, cmin = coords.min(axis=0)
                        rmax, cmax = coords.max(axis=0)
                        ch = int(rmax - rmin + 1)
                        cw = int(cmax - cmin + 1)
                        # Rectangular card
                        if (4 <= ch <= 15 and 8 <= cw <= 25) or (8 <= ch <= 25 and 4 <= cw <= 15):
                            center = (int((rmin + rmax) // 2), int((cmin + cmax) // 2))
                            widgets.append(ClickWidget(
                                color=val,
                                bbox=(int(rmin), int(cmin), int(rmax), int(cmax)),
                                size=(ch, cw),
                                center=center,
                                widget_type="switch"
                            ))
                    elif 1 <= len(comp) <= 12:
                        coords = np.array(comp)
                        rmin, cmin = coords.min(axis=0)
                        rmax, cmax = coords.max(axis=0)
                        ch = int(rmax - rmin + 1)
                        cw = int(cmax - cmin + 1)
                        if ch <= 4 and cw <= 4:
                            center = (int((rmin + rmax) // 2), int((cmin + cmax) // 2))
                            widgets.append(ClickWidget(
                                color=val,
                                bbox=(int(rmin), int(cmin), int(rmax), int(cmax)),
                                size=(ch, cw),
                                center=center,
                                widget_type="button"
                            ))

        return widgets

    def detect_collectibles(
        self,
        grid: np.ndarray,
        bg_canvas: int,
        floor_colors: Set[int]
    ) -> List[Dict[str, Any]]:
        """
        Scans for repeated identical tokens/dots scattered across floor.
        """
        h, w = grid.shape
        counts = collections.Counter(grid.flatten())
        collectible_candidates = []
        for color, count in counts.items():
            if color != bg_canvas and color not in floor_colors and 3 <= count <= 30:
                coords = np.argwhere(grid == color)
                coords = [pt for pt in coords if pt[0] < h - 1]
                if len(coords) >= 3:
                    collectible_candidates.append({
                        'color': int(color),
                        'count': len(coords),
                        'coords': [tuple(pt) for pt in coords]
                    })
        return collectible_candidates

    def induce(
        self,
        grid: np.ndarray,
        available_actions: List[int],
        avatar_pos: Optional[Tuple[int, int]],
        floor_colors: Optional[Set[int]] = None,
        levels_completed: int = 0
    ) -> ObjectiveHypothesis:
        """
        Synthesizes perceptual observations to induce the dominant teleological hypothesis.
        """
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        h, w = grid.shape
        bg_canvas = int(collections.Counter(grid.flatten()).most_common(1)[0][0])
        floors = set(floor_colors) if floor_colors else set()

        # Auto-infer prominent floor colors if not provided
        if not floors:
            counts = collections.Counter(grid[0:h-1].flatten()).most_common(5)
            for c_val, count in counts[1:]:
                if count >= 100:
                    floors.add(int(c_val))

        # 1. Morphological Extractions
        receptors, blocks = self.detect_receptors_and_blocks(grid, bg_canvas, floors)
        goals = self.detect_goal_tiles(grid, bg_canvas, floors, avatar_pos, blocks=blocks)
        click_widgets = self.detect_click_widgets(grid, bg_canvas, available_actions, blocks)
        collectibles = self.detect_collectibles(grid, bg_canvas, floors)

        scores: Dict[HypothesisType, float] = self.hypo_priors.copy()

        # 2. Hypothesis Likelihood Evaluation
        # A. Receptor Matching Evidence: Receptors exist and blocks match inner size
        if receptors and blocks:
            match_ratio = min(len(receptors), len(blocks)) / max(len(receptors), len(blocks), 1)
            scores[HypothesisType.RECEPTOR_MATCHING] = 0.85 + 0.10 * match_ratio

        # B. Goal Tile Evidence: Single unique rare goal
        if goals and not receptors:
            scores[HypothesisType.GOAL_TILE] = 0.80

        # C. Switch Gates Evidence: Control switches exist alongside maze/goal
        switches = [w for w in click_widgets if w.widget_type in ("switch", "button")]
        if switches and 6 in available_actions:
            scores[HypothesisType.SWITCH_GATES] = 0.75

        # D. Collectibles Evidence
        if collectibles and not receptors and not goals:
            scores[HypothesisType.COLLECT_ITEMS] = 0.75

        # Normalize posterior
        total_score = sum(scores.values())
        posteriors = {k: v / total_score for k, v in scores.items()}

        # Pick leading hypothesis
        leading_type = max(posteriors, key=posteriors.get)
        leading_conf = posteriors[leading_type]

        # 3. Compute Potential Function Phi(s)
        potential = 100.0
        targets: List[Any] = []
        units: List[Any] = blocks

        if leading_type == HypothesisType.RECEPTOR_MATCHING and receptors and blocks:
            # Sum of distances from each block to its nearest receptor
            pot = 0.0
            for b in blocks:
                min_d = min(abs(b.center[0] - r.center[0]) + abs(b.center[1] - r.center[1]) for r in receptors)
                pot += min_d
            potential = float(pot)
            targets = receptors

        elif leading_type == HypothesisType.GOAL_TILE and goals and avatar_pos is not None:
            g = goals[0]
            potential = float(abs(avatar_pos[0] - g.center[0]) + abs(avatar_pos[1] - g.center[1]))
            targets = goals

        elif leading_type == HypothesisType.SWITCH_GATES and switches:
            potential = float(len(switches) * 10.0)
            targets = switches

        elif leading_type == HypothesisType.COLLECT_ITEMS and collectibles:
            potential = float(collectibles[0]['count'])
            targets = collectibles[0]['coords']

        hypo = ObjectiveHypothesis(
            hypo_type=leading_type,
            confidence=leading_conf,
            potential=potential,
            targets=targets,
            movable_units=units,
            metadata={
                'switches': switches,
                'receptors': receptors,
                'blocks': blocks,
                'goals': goals
            }
        )
        self.current_hypothesis = hypo
        return hypo

    def synthesize_subgoals(
        self,
        hypo: ObjectiveHypothesis,
        grid: np.ndarray,
        available_actions: List[int],
        avatar_pos: Optional[Tuple[int, int]],
        floor_colors: Optional[Set[int]] = None
    ) -> List[Subgoal]:
        """
        Synthesizes macro-level SMDP Subgoals guided by the induced hypothesis.
        """
        subgoals: List[Subgoal] = []
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]

        if hypo.hypo_type == HypothesisType.RECEPTOR_MATCHING:
            receptors: List[ReceptorTarget] = hypo.metadata.get('receptors', [])
            blocks: List[MovableBlock] = hypo.metadata.get('blocks', [])

            # Compute optimal 1-to-1 bipartite assignment
            assignment = optimal_bipartite_assignment(blocks, receptors)

            # Determine active unit (block closest to avatar_pos)
            active_block_idx = None
            if avatar_pos is not None:
                for idx, b in enumerate(blocks):
                    if abs(avatar_pos[0] - b.center[0]) <= 2 and abs(avatar_pos[1] - b.center[1]) <= 2:
                        active_block_idx = idx
                        break

            # Find unseated blocks
            receptor_centers = {r.center for r in receptors}
            unseated_indices = [i for i, b in enumerate(blocks) if b.center not in receptor_centers]

            bg_canvas = int(collections.Counter(grid.flatten()).most_common(1)[0][0])
            floors = set(floor_colors) if floor_colors else set()
            if not floors:
                counts = collections.Counter(grid.flatten()).most_common(5)
                for c_val, count in counts[1:]:
                    if count >= 80:
                        floors.add(int(c_val))

            # 1. Check for Collinear Propulsion Dependency:
            # If an unseated block b_target CANNOT reach its receptor directly,
            # but another block b_pusher is aligned behind it pointing towards the receptor
            propel_found = False
            for target_idx in unseated_indices:
                b_target = blocks[target_idx]
                r_idx = assignment.get(target_idx)
                if r_idx is None or r_idx >= len(receptors):
                    continue
                target_rec = receptors[r_idx]

                # If b_target can already reach target_rec, it doesn't need propulsion
                stride_est = max(b_target.size[0], 1)
                can_reach = is_reachable(
                    grid, b_target.center, target_rec.center, bg_canvas, floors,
                    target_color=target_rec.color, stride=stride_est
                )
                if can_reach:
                    continue

                for pusher_idx in unseated_indices:
                    if pusher_idx == target_idx:
                        continue
                    b_pusher = blocks[pusher_idx]

                    same_row = (b_pusher.center[0] == b_target.center[0])
                    same_col = (b_pusher.center[1] == b_target.center[1])
                    if not (same_row or same_col):
                        continue

                    if same_row:
                        dir_push = 1 if b_target.center[1] > b_pusher.center[1] else -1
                        dir_rec = 1 if target_rec.center[1] > b_target.center[1] else -1
                        is_aligned = (dir_push == dir_rec)
                    else:
                        dir_push = 1 if b_target.center[0] > b_pusher.center[0] else -1
                        dir_rec = 1 if target_rec.center[0] > b_target.center[0] else -1
                        is_aligned = (dir_push == dir_rec)

                    if is_aligned:
                        # Ensure pusher is selected if Action 6 available
                        if (active_block_idx is None or active_block_idx != pusher_idx) and 6 in available_actions:
                            self.subgoal_counter += 1
                            subgoals.append(Subgoal(
                                subgoal_id=f"sg_select_pusher_{self.subgoal_counter}",
                                subgoal_type=SubgoalType.SWITCH_AVATAR,
                                target_pos=b_pusher.center,
                                target_color=b_pusher.color,
                                max_steps=2
                            ))

                        # Pusher propels target block by moving towards it
                        self.subgoal_counter += 1
                        subgoals.append(Subgoal(
                            subgoal_id=f"sg_propel_block_{self.subgoal_counter}",
                            subgoal_type=SubgoalType.REACH_ENTITY,
                            target_pos=b_target.center,
                            target_color=b_target.color,
                            max_steps=15,
                            metadata={
                                'action': 'propel',
                                'pusher_idx': pusher_idx,
                                'target_block_idx': target_idx,
                                'target_block_initial_pos': b_target.center,
                                'target_block_initial_val': int(grid[b_target.center[0], b_target.center[1]])
                            }
                        ))

                        # After propulsion, pusher seats in its assigned receptor
                        pusher_r_idx = assignment.get(pusher_idx)
                        if pusher_r_idx is not None and pusher_r_idx < len(receptors):
                            pusher_rec = receptors[pusher_r_idx]
                            self.subgoal_counter += 1
                            is_push = (avatar_pos is None or abs(avatar_pos[0] - b_pusher.center[0]) > 2 or abs(avatar_pos[1] - b_pusher.center[1]) > 2)
                            sg_type = SubgoalType.PUSH_ENTITY if is_push else SubgoalType.REACH_ENTITY
                            subgoals.append(Subgoal(
                                subgoal_id=f"sg_seat_block_{self.subgoal_counter}",
                                subgoal_type=sg_type,
                                target_pos=pusher_rec.center,
                                target_color=pusher_rec.color,
                                max_steps=50,
                                metadata={
                                    'block_idx': pusher_idx,
                                    'block_center': b_pusher.center,
                                    'block_pos': b_pusher.center,
                                    'block_color': b_pusher.color,
                                    'block_size': b_pusher.size,
                                    'action': 'seat'
                                }
                            ))

                        # After seating pusher, switch control to the propelled target block and seat it
                        target_r_idx = assignment.get(target_idx)
                        if target_r_idx is not None and target_r_idx < len(receptors):
                            target_rec = receptors[target_r_idx]
                            if 6 in available_actions:
                                self.subgoal_counter += 1
                                subgoals.append(Subgoal(
                                    subgoal_id=f"sg_switch_target_{self.subgoal_counter}",
                                    subgoal_type=SubgoalType.SWITCH_AVATAR,
                                    target_pos=b_target.center,
                                    target_color=b_target.color,
                                    max_steps=5,
                                    metadata={
                                        'block_idx': target_idx,
                                        'target_block_idx': target_idx,
                                        'block_color': b_target.color
                                    }
                                ))
                            self.subgoal_counter += 1
                            subgoals.append(Subgoal(
                                subgoal_id=f"sg_seat_target_{self.subgoal_counter}",
                                subgoal_type=SubgoalType.REACH_ENTITY,
                                target_pos=target_rec.center,
                                target_color=target_rec.color,
                                max_steps=50,
                                metadata={
                                    'block_idx': target_idx,
                                    'action': 'seat_target'
                                }
                            ))

                        propel_found = True
                        break
                if propel_found:
                    break

            if not propel_found:
                # Sequence blocks using topological deadlock ordering
                ordered_indices, staging_targets = topological_block_ordering(
                    blocks=blocks,
                    receptors=receptors,
                    assignment=assignment,
                    unseated_indices=unseated_indices,
                    grid=grid,
                    avatar_pos=avatar_pos,
                    bg_canvas=bg_canvas,
                    floor_colors=floors,
                    stride=1
                )

                # 1. Emit staging subgoals for blocks that must yield to clear transit corridors
                for b_idx in ordered_indices:
                    if b_idx in staging_targets:
                        b = blocks[b_idx]
                        staging_pos = staging_targets[b_idx]
                        if 6 in available_actions:
                            self.subgoal_counter += 1
                            subgoals.append(Subgoal(
                                subgoal_id=f"sg_select_stage_{self.subgoal_counter}",
                                subgoal_type=SubgoalType.SWITCH_AVATAR,
                                target_pos=b.center,
                                target_color=b.color,
                                max_steps=2,
                                metadata={'block_idx': b_idx, 'block_color': b.color}
                            ))
                        self.subgoal_counter += 1
                        subgoals.append(Subgoal(
                            subgoal_id=f"sg_stage_block_{self.subgoal_counter}",
                            subgoal_type=SubgoalType.STAGE_ENTITY,
                            target_pos=staging_pos,
                            target_color=b.color,
                            max_steps=30,
                            metadata={
                                'block_idx': b_idx,
                                'block_center': b.center,
                                'block_pos': b.center,
                                'block_color': b.color,
                                'block_size': b.size,
                                'action': 'stage',
                                'is_staging': True
                            }
                        ))

                # 2. Emit delivery subgoals in deadlock-free topological sequence
                for b_idx in ordered_indices:
                    b = blocks[b_idx]
                    r_idx = assignment.get(b_idx)
                    if r_idx is None or r_idx >= len(receptors):
                        continue
                    target_rec = receptors[r_idx]

                    # 1. If block is not currently active and Action 6 is available, select it
                    if (active_block_idx is None or active_block_idx != b_idx) and 6 in available_actions:
                        self.subgoal_counter += 1
                        subgoals.append(Subgoal(
                            subgoal_id=f"sg_select_block_{self.subgoal_counter}",
                            subgoal_type=SubgoalType.SWITCH_AVATAR,  # Dispatches Action 6
                            target_pos=b.center,
                            target_color=b.color,
                            max_steps=2,
                            metadata={'block_idx': b_idx, 'block_color': b.color}
                        ))

                    # 2. Navigate / Push block to receptor center
                    self.subgoal_counter += 1
                    is_push = (avatar_pos is None or abs(avatar_pos[0] - b.center[0]) > 2 or abs(avatar_pos[1] - b.center[1]) > 2)
                    sg_type = SubgoalType.PUSH_ENTITY if is_push else SubgoalType.REACH_ENTITY
                    subgoals.append(Subgoal(
                        subgoal_id=f"sg_seat_block_{self.subgoal_counter}",
                        subgoal_type=sg_type,
                        target_pos=target_rec.center,
                        target_color=target_rec.color,
                        max_steps=50,
                        metadata={
                            'block_idx': b_idx,
                            'block_center': b.center,
                            'block_pos': b.center,
                            'block_color': b.color,
                            'block_size': b.size,
                            'action': 'push_to_receptor'
                        }
                    ))

        elif hypo.hypo_type == HypothesisType.GOAL_TILE:
            goals: List[GoalTarget] = hypo.metadata.get('goals', [])
            switches: List[ClickWidget] = hypo.metadata.get('switches', [])
            if goals:
                g = goals[0]
                # Precondition: if switches exist and 6 in available_actions, click switches first
                if switches and 6 in available_actions:
                    for sw in switches:
                        self.subgoal_counter += 1
                        subgoals.append(Subgoal(
                            subgoal_id=f"sg_pre_switch_{self.subgoal_counter}",
                            subgoal_type=SubgoalType.INTERACT_ENTITY,
                            target_pos=sw.center,
                            target_color=sw.color,
                            max_steps=2
                        ))

                self.subgoal_counter += 1
                subgoals.append(Subgoal(
                    subgoal_id=f"sg_reach_goal_{self.subgoal_counter}",
                    subgoal_type=SubgoalType.REACH_ENTITY,
                    target_pos=g.center,
                    target_color=g.color,
                    max_steps=40
                ))
                if 5 in available_actions:
                    self.subgoal_counter += 1
                    subgoals.append(Subgoal(
                        subgoal_id=f"sg_interact_goal_{self.subgoal_counter}",
                        subgoal_type=SubgoalType.INTERACT_ENTITY,
                        target_pos=g.center,
                        target_color=g.color,
                        max_steps=2
                    ))

        elif hypo.hypo_type == HypothesisType.SWITCH_GATES:
            switches: List[ClickWidget] = hypo.metadata.get('switches', [])
            if len(switches) >= 2 and 6 in available_actions:
                self.subgoal_counter += 1
                subgoals.append(Subgoal(
                    subgoal_id=f"sg_toggle_solve_{self.subgoal_counter}",
                    subgoal_type=SubgoalType.TOGGLE_SOLVE,
                    target_pos=switches[0].center,
                    max_steps=max(len(switches) * 4, 30),
                    metadata={
                        'actuators': [sw.center for sw in switches],
                        'controlled_cells': [sw.center for sw in switches],
                        'switches': switches,
                        'is_toggle': True
                    }
                ))
            else:
                for sw in switches:
                    self.subgoal_counter += 1
                    subgoals.append(Subgoal(
                        subgoal_id=f"sg_click_switch_{self.subgoal_counter}",
                        subgoal_type=SubgoalType.INTERACT_ENTITY,
                        target_pos=sw.center,
                        target_color=sw.color,
                        max_steps=2
                    ))

        return subgoals
