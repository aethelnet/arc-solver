"""
=============================================================================
ARC-AGI CAUSAL ACTIVE INFERENCE: ONLINE KINEMATICS CALIBRATOR
=============================================================================
Learns on-the-fly actuator dynamics and compensates for action-channel
permutations sigma in S_4 (e.g. Action 1 -> Action 2).
Transforms nominal high-level directional intents (UP, DOWN, LEFT, RIGHT)
into true physical actuator channels via online Bayesian deduction.
=============================================================================
"""

from typing import Dict, Tuple, Optional, Set, List, Any
import numpy as np

# Canonical Cardinal Directions and their Nominal Action codes
# 1: UP    (dr = -1, dc = 0)
# 2: DOWN  (dr = +1, dc = 0)
# 3: LEFT  (dr = 0, dc = -1)
# 4: RIGHT (dr = 0, dc = +1)
INTENT_TO_DIR: Dict[int, Tuple[int, int]] = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1)
}
DIR_TO_INTENT: Dict[Tuple[int, int], int] = {v: k for k, v in INTENT_TO_DIR.items()}
CARDINAL_DELTAS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def detect_frame_displacement(
    prev_grid: Optional[np.ndarray],
    curr_grid: Optional[np.ndarray],
    avatar_pos: Optional[Tuple[int, int]] = None,
    candidate_strides: Optional[List[int]] = None
) -> Tuple[Optional[Tuple[int, int]], int]:
    """
    Detects rigid avatar or cluster displacement between consecutive frames.
    Returns:
        (best_cardinal_dir, best_stride) where best_cardinal_dir in [(-1,0), (1,0), (0,-1), (0,1)]
        or (None, 1) if no clear directional displacement detected.
    """
    if prev_grid is None or curr_grid is None:
        return (None, 1)

    p_arr = np.asarray(prev_grid)
    c_arr = np.asarray(curr_grid)
    while p_arr.ndim > 2:
        p_arr = p_arr[-1]
    while c_arr.ndim > 2:
        c_arr = c_arr[-1]

    H, W = c_arr.shape
    max_r = min(H, 60) # Preserve HUD rows 60..63
    p_crop = p_arr[:max_r, :]
    c_crop = c_arr[:max_r, :]

    diff_mask = (p_crop != c_crop)
    if not np.any(diff_mask):
        return (None, 1)

    diff_coords = np.argwhere(diff_mask)
    if len(diff_coords) == 0:
        return (None, 1)

    strides = candidate_strides if candidate_strides is not None else [1, 2, 3, 4, 5, 6, 8]

    # Method 1: If avatar_pos is known, evaluate local patch correlation
    if avatar_pos is not None:
        ar, ac = avatar_pos
        best_delta = None
        best_score = -1.0
        best_S = 1

        for S in strides:
            k = max(S // 2, 1)
            if not (k <= ar < max_r - k and k <= ac < W - k):
                continue
            old_patch = p_crop[ar - k : ar + k + 1, ac - k : ac + k + 1]

            for dr, dc in CARDINAL_DELTAS:
                nr, nc = ar + dr * S, ac + dc * S
                if k <= nr < max_r - k and k <= nc < W - k:
                    new_patch = c_crop[nr - k : nr + k + 1, nc - k : nc + k + 1]
                    score = float(np.mean(old_patch == new_patch))
                    if score > best_score and score >= 0.65:
                        best_score = score
                        best_delta = (dr, dc)
                        best_S = S

        if best_delta is not None:
            return (best_delta, best_S)

    # Method 2: Foreground cluster centroid displacement
    vals, counts = np.unique(c_crop, return_counts=True)
    freq = dict(zip(vals, counts))

    # Foreground colors in diff: colors with global frequency <= 80 (excludes floor and wall expanses)
    fg_colors_p = set(c for c in np.unique(p_crop[diff_mask]) if freq.get(c, 0) <= 80)
    fg_colors_c = set(c for c in np.unique(c_crop[diff_mask]) if freq.get(c, 0) <= 80)
    all_fg = fg_colors_p.union(fg_colors_c)

    if all_fg:
        # Check for opposing movements across colors (reflection symmetry / counter-moving entities)
        signs_dr = set()
        signs_dc = set()
        for col in all_fg:
            p_c = np.argwhere((p_crop == col) & diff_mask)
            c_c = np.argwhere((c_crop == col) & diff_mask)
            if len(p_c) > 0 and len(c_c) > 0:
                c_dr = int(np.sign(round(c_c[:, 0].mean() - p_c[:, 0].mean())))
                c_dc = int(np.sign(round(c_c[:, 1].mean() - p_c[:, 1].mean())))
                if c_dr != 0:
                    signs_dr.add(c_dr)
                if c_dc != 0:
                    signs_dc.add(c_dc)
        if (1 in signs_dr and -1 in signs_dr) or (1 in signs_dc and -1 in signs_dc):
            # Opposing movements detected (reflection symmetry / multi-agent). Unanchored displacement is ambiguous.
            return (None, 1)

        pts0 = np.argwhere(np.isin(p_crop, list(all_fg)) & diff_mask)
        pts1 = np.argwhere(np.isin(c_crop, list(all_fg)) & diff_mask)
        if len(pts0) > 0 and len(pts1) > 0:
            dr = int(round(pts1[:, 0].mean() - pts0[:, 0].mean()))
            dc = int(round(pts1[:, 1].mean() - pts0[:, 1].mean()))
            if abs(dr) > 0 or abs(dc) > 0:
                if abs(dr) >= abs(dc) and abs(dr) > 0:
                    return ((int(np.sign(dr)), 0), abs(dr))
                elif abs(dc) > abs(dr) and abs(dc) > 0:
                    return ((0, int(np.sign(dc))), abs(dc))

    # Method 3: Fallback to rarest common color in diff region
    diff_colors_p = set(np.unique(p_crop[diff_mask]))
    diff_colors_c = set(np.unique(c_crop[diff_mask]))
    common_colors = diff_colors_p.intersection(diff_colors_c)
    sorted_colors = sorted(common_colors, key=lambda col: freq.get(col, 999999))

    for col in sorted_colors:
        if freq.get(col, 0) > 80:
            continue
        pts0 = np.argwhere((p_crop == col) & diff_mask)
        pts1 = np.argwhere((c_crop == col) & diff_mask)
        if len(pts0) > 0 and len(pts1) > 0:
            dr = int(round(pts1[:, 0].mean() - pts0[:, 0].mean()))
            dc = int(round(pts1[:, 1].mean() - pts0[:, 1].mean()))
            if abs(dr) > 0 and abs(dc) == 0:
                return ((int(np.sign(dr)), 0), abs(dr))
            elif abs(dc) > 0 and abs(dr) == 0:
                return ((0, int(np.sign(dc))), abs(dc))

    return (None, 1)


class OnlineKinematicsCalibrator:
    """
    Online Bayesian Kinematics Estimator for S_4 Action Permutations.
    Tracks bijection sigma: {1..4} <-> {(-1,0), (1,0), (0,-1), (0,1)}.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        # Default standard arcade prior:
        # Action 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT
        self.action_to_dir: Dict[int, Tuple[int, int]] = {
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1)
        }
        self.dir_to_action: Dict[Tuple[int, int], int] = {
            (-1, 0): 1,
            (1, 0): 2,
            (0, -1): 3,
            (0, 1): 4
        }
        self.confirmed_actions: Set[int] = set()
        self.confirmed_dirs: Set[Tuple[int, int]] = set()
        self.transition_counts: Dict[int, Dict[Tuple[int, int], int]] = {
            a: {d: 0 for d in CARDINAL_DELTAS} for a in (1, 2, 3, 4)
        }

    def is_fully_calibrated(self, num_nav: int = 4) -> bool:
        return len(self.confirmed_actions) >= min(num_nav, 3) or len(self.confirmed_actions) >= 3

    def is_calibrated(self, num_nav: int = 4) -> bool:
        return self.is_fully_calibrated(num_nav)

    def observe(
        self,
        physical_action: int,
        observed_delta: Tuple[int, int],
        available_actions: Optional[List[int]] = None
    ) -> bool:
        """
        Updates kinematic beliefs from an observed physical transition.
        observed_delta: (dr, dc) displacement of the avatar / moved cluster.
        Returns True if a new mapping was confirmed.
        """
        if self.is_fully_calibrated():
            return False

        if physical_action not in (1, 2, 3, 4):
            return False

        dr, dc = observed_delta
        if dr == 0 and dc == 0:
            return False

        sign_dr = int(np.sign(dr))
        sign_dc = int(np.sign(dc))

        # Filter out passive environmental forces (e.g. gravity fall in 1D horizontal games)
        if available_actions is not None:
            has_vert = (1 in available_actions or 2 in available_actions)
            has_horiz = (3 in available_actions or 4 in available_actions)
            if not has_vert:
                sign_dr = 0
            if not has_horiz:
                sign_dc = 0

        cardinal = (sign_dr, sign_dc)

        if cardinal not in INTENT_TO_DIR.values():
            return False

        self.transition_counts[physical_action][cardinal] += 1

        # Check if already confirmed
        if self.action_to_dir.get(physical_action) == cardinal and physical_action in self.confirmed_actions:
            return False

        # Confirmation threshold:
        # If cardinal matches nominal prior for physical_action: threshold = 1
        # If cardinal conflicts with nominal prior for physical_action: threshold = 2
        nominal_dir = INTENT_TO_DIR.get(physical_action)
        req_count = 1 if cardinal == nominal_dir else 2
        if self.transition_counts[physical_action][cardinal] < req_count:
            return False

        # If some other action was previously confirmed for this cardinal direction, unconfirm it
        for act in list(self.confirmed_actions):
            if self.action_to_dir.get(act) == cardinal and act != physical_action:
                self.confirmed_actions.discard(act)

        # If this physical action was previously confirmed for a different direction, unconfirm that direction
        for d in list(self.confirmed_dirs):
            if self.dir_to_action.get(d) == physical_action and d != cardinal:
                self.confirmed_dirs.discard(d)

        # Update assignment
        self.action_to_dir[physical_action] = cardinal
        self.dir_to_action[cardinal] = physical_action
        self.confirmed_actions.add(physical_action)
        self.confirmed_dirs.add(cardinal)

        # Elimination deduction
        self._deduce_by_elimination()
        return True

    def _deduce_by_elimination(self):
        """If 3 cardinal directions are confirmed, the 4th is uniquely determined."""
        all_actions = {1, 2, 3, 4}
        all_dirs = set(CARDINAL_DELTAS)

        unconfirmed_actions = list(all_actions - self.confirmed_actions)
        unconfirmed_dirs = list(all_dirs - self.confirmed_dirs)

        if len(unconfirmed_actions) == 1 and len(unconfirmed_dirs) == 1:
            rem_act = unconfirmed_actions[0]
            rem_dir = unconfirmed_dirs[0]
            self.action_to_dir[rem_act] = rem_dir
            self.dir_to_action[rem_dir] = rem_act
            self.confirmed_actions.add(rem_act)
            self.confirmed_dirs.add(rem_dir)

    def get_action_for_direction(
        self,
        target_dir: Tuple[int, int],
        available_actions: Optional[List[int]] = None
    ) -> int:
        """
        Returns the physical action channel that executes the intended (dr, dc).
        """
        dr = int(np.sign(target_dir[0]))
        dc = int(np.sign(target_dir[1]))
        cardinal = (dr, dc)

        legal = available_actions if available_actions is not None else [1, 2, 3, 4]
        nav_legal = [a for a in [1, 2, 3, 4] if a in legal]

        # 1. If direction is confirmed and legal:
        phys_act = self.dir_to_action.get(cardinal)
        if phys_act is not None and phys_act in nav_legal and cardinal in self.confirmed_dirs:
            return phys_act

        # 2. Fallback to nominal prior action for this direction if legal:
        nominal_act = DIR_TO_INTENT.get(cardinal)
        if nominal_act is not None and nominal_act in nav_legal:
            return nominal_act

        # 3. Fallback to mapped action or first legal nav action
        if phys_act is not None and phys_act in nav_legal:
            return phys_act
        return nav_legal[0] if nav_legal else 1

    def get_action_for_intent(
        self,
        nominal_intent: int,
        available_actions: Optional[List[int]] = None
    ) -> int:
        """
        Translates nominal intent (1=UP, 2=DOWN, 3=LEFT, 4=RIGHT) to physical action.
        """
        if nominal_intent not in (1, 2, 3, 4):
            return nominal_intent
        target_dir = INTENT_TO_DIR.get(nominal_intent, (-1, 0))
        return self.get_action_for_direction(target_dir, available_actions)

    def get_delta_for_action(self, physical_action: int) -> Tuple[int, int]:
        """Returns the expected (dr, dc) for a physical action channel."""
        return self.action_to_dir.get(physical_action, (0, 0))

    def get_intent_for_action(self, physical_action: int) -> int:
        """Translates physical action back to nominal intent code (1..4)."""
        cardinal = self.action_to_dir.get(physical_action)
        if cardinal is not None:
            return DIR_TO_INTENT.get(cardinal, physical_action)
        return physical_action
