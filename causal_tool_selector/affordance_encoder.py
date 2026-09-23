"""
=============================================================================
ARC-AGI CAUSAL META-TOOL SELECTOR: AFFORDANCE ENCODER
=============================================================================
Extracts a continuous semantic affordance state vector from raw ARC frames
and physical transition dynamics:
  1. Action Interface Constraints (Nav, Click, Interact)
  2. Spatial Geometry & Complexity (Entity counts, Color entropy, Symmetry)
  3. Physical & Dynamic Signatures (Motion response, Gravity, Involutory toggles, Pushing)
  4. Teleological Progress (Level completion delta, Survival status)
=============================================================================
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Set
import collections
from dataclasses import dataclass

from auratic_core.causal_engine import StateEncoder, LatentState

@dataclass
class AffordanceFeatures:
    has_nav: float                   # 0.0 or 1.0
    has_click: float                 # 0.0 or 1.0
    has_interact: float              # 0.0 or 1.0
    has_reset: float                 # 0.0 or 1.0 (action 7)
    has_1d_nav: float                # 0.0 or 1.0 (1D constrained nav e.g. actions 3, 4 without 1, 2)
    color_count_norm: float          # 0.0 to 1.0 (colors / 16)
    entity_count_norm: float         # 0.0 to 1.0 (clipped at 50)
    spatial_symmetry: float          # 0.0 to 1.0
    has_receptors: float             # 0.0 or 1.0 (hollow frames with matching blocks)
    has_goal_tile: float             # 0.0 or 1.0 (isolated destination exit tile)
    has_switch_widgets: float        # 0.0 or 1.0 (clickable control buttons/cards)
    has_gravity_platforms: float     # 0.0 or 1.0 (horizontal shelves with vertical drop)
    directional_motion_score: float  # -1.0 to 1.0
    gravity_flux: float              # 0.0 to 1.0
    involutory_toggle_score: float   # 0.0 to 1.0
    push_displacement_score: float   # 0.0 to 1.0
    color_flood_score: float         # 0.0 to 1.0
    level_win_delta: float           # +1.0 (win), -1.0 (game over), 0.0 (step)

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.has_nav,
            self.has_click,
            self.has_interact,
            self.has_reset,
            self.has_1d_nav,
            self.color_count_norm,
            self.entity_count_norm,
            self.spatial_symmetry,
            self.has_receptors,
            self.has_goal_tile,
            self.has_switch_widgets,
            self.has_gravity_platforms,
            self.directional_motion_score,
            self.gravity_flux,
            self.involutory_toggle_score,
            self.push_displacement_score,
            self.color_flood_score,
            self.level_win_delta
        ], dtype=np.float64)


class ARCAffordanceEncoder:
    """
    Transforms multi-frame ARC observations and actions into a normalized
    18-dimensional LatentState representation for Active Inference.
    """
    FEATURE_DIM = 18

    def __init__(self, momentum: float = 0.05):
        self.encoder = StateEncoder(obs_dim=self.FEATURE_DIM, latent_dim=self.FEATURE_DIM, momentum=momentum)

    def extract_raw_features(
        self,
        current_grid: Any,
        prev_grid: Optional[Any],
        prev_action: Optional[int],
        prev_action_data: Optional[Dict[str, Any]],
        available_actions: List[int],
        levels_completed: int,
        prev_levels_completed: int,
        game_over: bool = False
    ) -> AffordanceFeatures:
        current_grid = np.asarray(current_grid)
        while current_grid.ndim > 2:
            current_grid = current_grid[-1]
        h, w = current_grid.shape

        # 1. Action interface constraints
        has_nav = 1.0 if any(a in [1, 2, 3, 4] for a in available_actions) else 0.0
        has_click = 1.0 if 6 in available_actions else 0.0
        has_interact = 1.0 if 5 in available_actions else 0.0
        has_reset = 1.0 if 7 in available_actions else 0.0
        has_1d_nav = 1.0 if (has_nav > 0.0 and (1 not in available_actions or 2 not in available_actions)) else 0.0

        # 2. Color Palette & Entropy
        unique_colors = np.unique(current_grid)
        color_count_norm = float(len(unique_colors) / 16.0)

        # 3. Entity Count via connected components
        color_counts = collections.Counter(current_grid.flatten()).most_common(6)
        bg_color = int(color_counts[0][0])
        entities_count = self._count_entities(current_grid, bg_color)
        entity_count_norm = float(min(entities_count, 50) / 50.0)

        # 4. Spatial Symmetry (Horizontal and Vertical axis similarity)
        h_sym = float(np.mean(current_grid == np.flipud(current_grid)))
        v_sym = float(np.mean(current_grid == np.fliplr(current_grid)))
        spatial_symmetry = float((h_sym + v_sym) / 2.0)

        # 5. Static Morphological Features (Step 0 Prior Resonance)
        floor_colors = {int(c[0]) for c in color_counts[1:] if c[1] >= 100}
        has_receptors, has_goal_tile, has_switch_widgets, has_gravity_platforms = self._extract_static_morphology(
            current_grid, bg_color, floor_colors, available_actions
        )

        # 6. Dynamics across transitions (if prev_grid exists)
        directional_motion_score = 0.0
        gravity_flux = 0.0
        involutory_toggle_score = 0.0
        push_displacement_score = 0.0
        color_flood_score = 0.0

        if prev_grid is not None:
            prev_grid = np.asarray(prev_grid)
            while prev_grid.ndim > 2:
                prev_grid = prev_grid[-1]
            diff_mask = (prev_grid != current_grid)
            diff_count = int(np.sum(diff_mask))

            if diff_count > 0:
                # Test Directional Motion Response (Nav Actions 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT)
                if prev_action in [1, 2, 3, 4]:
                    directional_motion_score = self._compute_motion_alignment(prev_grid, current_grid, prev_action, bg_color)

                # Test Gravity Flux: do pixels move downward without DOWN input?
                if prev_action != 2:
                    downward_shift = self._detect_downward_flux(prev_grid, current_grid, bg_color)
                    gravity_flux = float(downward_shift)

                # Test Involutory Toggle (Action 6 Click)
                if prev_action == 6:
                    if 1 <= diff_count <= 9:
                        # Typical cross or localized cell toggle
                        involutory_toggle_score = 1.0
                    elif diff_count > 25:
                        # Large flood fill or color spill
                        color_flood_score = float(min(diff_count / 100.0, 1.0))

                # Test Push Displacement: did one object shift another?
                if prev_action in [1, 2, 3, 4] and diff_count >= 2:
                    push_displacement_score = self._detect_push_displacement(prev_grid, current_grid, bg_color)

        # 7. Teleological Progress
        if game_over:
            level_win_delta = -1.0
        elif levels_completed > prev_levels_completed:
            level_win_delta = 1.0
        else:
            level_win_delta = 0.0

        return AffordanceFeatures(
            has_nav=has_nav,
            has_click=has_click,
            has_interact=has_interact,
            has_reset=has_reset,
            has_1d_nav=has_1d_nav,
            color_count_norm=color_count_norm,
            entity_count_norm=entity_count_norm,
            spatial_symmetry=spatial_symmetry,
            has_receptors=has_receptors,
            has_goal_tile=has_goal_tile,
            has_switch_widgets=has_switch_widgets,
            has_gravity_platforms=has_gravity_platforms,
            directional_motion_score=directional_motion_score,
            gravity_flux=gravity_flux,
            involutory_toggle_score=involutory_toggle_score,
            push_displacement_score=push_displacement_score,
            color_flood_score=color_flood_score,
            level_win_delta=level_win_delta
        )

    def encode(
        self,
        current_grid: np.ndarray,
        prev_grid: Optional[np.ndarray],
        prev_action: Optional[int],
        prev_action_data: Optional[Dict[str, Any]],
        available_actions: List[int],
        levels_completed: int,
        prev_levels_completed: int,
        game_over: bool = False,
        step: int = 0
    ) -> LatentState:
        features = self.extract_raw_features(
            current_grid=current_grid,
            prev_grid=prev_grid,
            prev_action=prev_action,
            prev_action_data=prev_action_data,
            available_actions=available_actions,
            levels_completed=levels_completed,
            prev_levels_completed=prev_levels_completed,
            game_over=game_over
        )
        vec = features.to_vector()
        latent_state = self.encoder.encode(vec, step=step)
        latent_state.metadata['features'] = features
        return latent_state

    def _count_entities(self, grid: np.ndarray, bg_color: int) -> int:
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        count = 0
        for r in range(h):
            for c in range(w):
                if grid[r, c] != bg_color and not visited[r, c]:
                    count += 1
                    q = collections.deque([(r, c)])
                    visited[r, c] = True
                    color = grid[r, c]
                    while q:
                        cr, cc = q.popleft()
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                q.append((nr, nc))
        return count

    def _compute_motion_alignment(self, prev_grid: np.ndarray, curr_grid: np.ndarray, action: int, bg_color: int) -> float:
        # Expected displacement: 1=UP (-1, 0), 2=DOWN (+1, 0), 3=LEFT (0, -1), 4=RIGHT (0, +1)
        expected_dr = {1: -1, 2: 1, 3: 0, 4: 0}[action]
        expected_dc = {1: 0, 2: 0, 3: -1, 4: 1}[action]

        prev_fg = np.argwhere((prev_grid != bg_color) & (prev_grid > 0))
        curr_fg = np.argwhere((curr_grid != bg_color) & (curr_grid > 0))
        if len(prev_fg) == 0 or len(curr_fg) == 0:
            return 0.0

        prev_center = np.mean(prev_fg, axis=0)
        curr_center = np.mean(curr_fg, axis=0)
        dr = curr_center[0] - prev_center[0]
        dc = curr_center[1] - prev_center[1]

        # Dot product with expected unit direction
        dot = (dr * expected_dr + dc * expected_dc)
        if abs(dr) > 0.01 or abs(dc) > 0.01:
            return 1.0 if dot > 0.1 else -1.0
        return 0.0

    def _detect_downward_flux(self, prev_grid: np.ndarray, curr_grid: np.ndarray, bg_color: int) -> float:
        prev_fg = np.argwhere((prev_grid != bg_color) & (prev_grid > 0))
        curr_fg = np.argwhere((curr_grid != bg_color) & (curr_grid > 0))
        if len(prev_fg) == 0 or len(curr_fg) == 0:
            return 0.0
        dr = np.mean(curr_fg[:, 0]) - np.mean(prev_fg[:, 0])
        return 1.0 if dr > 0.5 else 0.0

    def _detect_push_displacement(self, prev_grid: np.ndarray, curr_grid: np.ndarray, bg_color: int) -> float:
        diff_coords = np.argwhere(prev_grid != curr_grid)
        if len(diff_coords) >= 4:
            # Multi-cell change suggests moving player AND displaced block
            return 1.0
        return 0.0

    def _extract_static_morphology(
        self,
        grid: np.ndarray,
        bg_color: int,
        floor_colors: Set[int],
        available_actions: List[int]
    ) -> Tuple[float, float, float, float]:
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        comps = []
        # Exclude HUD row 63
        for r in range(h - 1):
            for c in range(w):
                val = int(grid[r, c])
                if not visited[r, c] and val != bg_color and val not in floor_colors:
                    visited[r, c] = True
                    comp = [(r, c)]
                    q = [(r, c)]
                    while q:
                        cr, cc = q.pop(0)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h - 1 and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == val:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                                comp.append((nr, nc))
                    comps.append((val, comp))

        hollow_inner_sizes = set()
        blocks = []
        has_switch = 0.0

        for val, comp in comps:
            coords = np.array(comp)
            rmin, cmin = coords.min(axis=0)
            rmax, cmax = coords.max(axis=0)
            ch = int(rmax - rmin + 1)
            cw = int(cmax - cmin + 1)

            # Check hollow receptor
            if 4 <= ch <= 16 and 4 <= cw <= 16 and (ch * cw <= h * w * 0.40):
                inner = grid[rmin+1:rmax, cmin+1:cmax]
                if np.mean(np.isin(inner, list(floor_colors) + [bg_color])) >= 0.70:
                    hollow_inner_sizes.add((ch - 2, cw - 2))

            # Check compact block
            if 2 <= ch <= 8 and 2 <= cw <= 8 and len(comp) <= 64:
                blocks.append(((ch, cw), (int((rmin + rmax) // 2), int((cmin + cmax) // 2)), comp))

            # Check switch widgets (clickable cards)
            if 6 in available_actions:
                if 15 <= len(comp) <= 120:
                    if (4 <= ch <= 15 and 8 <= cw <= 25) or (8 <= ch <= 25 and 4 <= cw <= 15):
                        has_switch = 1.0

        # Receptor match: at least one block size matches a hollow receptor inner size
        has_receptors = 1.0 if any(b[0] in hollow_inner_sizes for b in blocks) else 0.0

        # Goal tile: rare color on floor that is not part of any block
        has_goal = 0.0
        counts = collections.Counter(grid[0:h-1].flatten())
        rare_colors = [c for c, cnt in counts.items() if c != bg_color and c not in floor_colors and 1 <= cnt <= 9]
        for rc in rare_colors:
            pts = np.argwhere(grid[0:h-1] == rc)
            if len(pts) > 0:
                rmin, cmin = pts.min(axis=0)
                rmax, cmax = pts.max(axis=0)
                ch = int(rmax - rmin + 1)
                cw = int(cmax - cmin + 1)
                if 1 <= ch <= 3 and 1 <= cw <= 3:
                    center = (int((rmin + rmax) // 2), int((cmin + cmax) // 2))
                    # Check if center falls inside any block
                    inside_block = False
                    for b_size, b_center, b_comp in blocks:
                        if abs(center[0] - b_center[0]) <= b_size[0] // 2 and abs(center[1] - b_center[1]) <= b_size[1] // 2:
                            inside_block = True
                            break
                    if not inside_block:
                        has_goal = 1.0
                        break

        # Gravity platforms: horizontal shelves with vertical air gaps
        has_gravity = 0.0
        if 7 in available_actions or any(a in [1, 2, 3, 4] for a in available_actions):
            for r in range(h - 10):
                row = grid[r]
                non_empty = (row != bg_color)
                runs = np.diff(np.where(np.concatenate(([non_empty[0]], non_empty[:-1] != non_empty[1:], [True])))[0])[::2] if np.any(non_empty) else []
                if any(run >= 8 for run in runs):
                    under = grid[r+1:min(r+6, h-1)]
                    if np.mean(under == bg_color) > 0.40:
                        has_gravity = 1.0
                        break

        return (has_receptors, has_goal, has_switch, has_gravity)

