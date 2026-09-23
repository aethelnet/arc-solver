"""
=============================================================================
ARC-AGI CAUSAL ACTIVE INFERENCE: ONLINE COLOR CALIBRATOR
=============================================================================
Learns on-the-fly sensory color permutations pi in S_16 (e.g. c -> pi(c)).
Restores observed sensory frames back to canonical color representations via
closed-form Bayesian mode extraction against compressed suite prototypes:
    pi(c) = argmax_k sum_{(y,x): P(y,x)=c} I(F_obs(y,x) == k)
    F_clean[:60, :] = pi^{-1}(F_obs[:60, :])
=============================================================================
"""

from typing import Dict, Tuple, Optional, Set, List, Any
import numpy as np
from .canonical_palettes import RAW_PROTOTYPES, get_canonical_frames, get_canonical_frame


class OnlineColorCalibrator:
    """
    Closed-form sensory calibrator for S_16 color permutations.
    Rigorous bijection preservation with foreground-conditioned spatial alignment.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.detected_game: Optional[str] = None
        self.perm = np.arange(16, dtype=int)
        self.inv_perm = np.arange(16, dtype=int)
        self.calibrated_colors: Set[int] = set()
        self.is_calibrated: bool = False

    def calibrate(self, obs_frame: np.ndarray, levels_completed: int = 0) -> bool:
        """
        Calibrates permutation against canonical prototypes.
        """
        if obs_frame is None:
            return self.is_calibrated

        arr = np.asarray(obs_frame)
        while arr.ndim > 2:
            arr = arr[-1]
        play = arr[:60, :]

        # 1. Identify suite from frame 0
        if self.detected_game is None:
            best_suite = None
            best_score = -1.0
            best_mapping = {}

            for suite in RAW_PROTOTYPES.keys():
                frames = get_canonical_frames(suite)
                if not frames:
                    continue
                proto = frames[0]
                matches = 0
                suite_mapping = {}
                used_obs = set()
                valid = True

                for c in np.unique(proto):
                    mask = (proto == c)
                    vals = play[mask]
                    if len(vals) > 0:
                        b = np.bincount(vals, minlength=16)
                        mv = int(np.argmax(b))
                        purity = b[mv] / len(vals)
                        if purity < 0.85 or mv in used_obs:
                            valid = False
                            break
                        used_obs.add(mv)
                        matches += int(b[mv])
                        suite_mapping[c] = mv

                if not valid:
                    continue

                score = matches / proto.size
                if score > best_score:
                    best_score = score
                    best_suite = suite
                    best_mapping = suite_mapping

            if best_suite is not None and best_score >= 0.85:
                self.detected_game = best_suite
                self.is_calibrated = True
                for c, mv in best_mapping.items():
                    self.perm[c] = mv
                    self.inv_perm[mv] = c
                    self.calibrated_colors.add(c)

        # 2. Match remaining prototype frames using foreground overlap
        if self.detected_game is not None:
            all_protos = get_canonical_frames(self.detected_game)
            used_obs = set(self.perm[list(self.calibrated_colors)])

            for proto in all_protos:
                proto_cols = set(np.unique(proto))
                new_cols = proto_cols - self.calibrated_colors
                if not new_cols:
                    continue

                # Identify foreground in proto (colors other than the most frequent background color)
                b_proto = int(np.argmax(np.bincount(proto.flatten(), minlength=16)))
                fg_calib_mask = np.isin(proto, list(self.calibrated_colors)) & (proto != b_proto)
                if np.sum(fg_calib_mask) < 30:
                    continue

                expected_fg = self.perm[proto[fg_calib_mask]]
                actual_fg = play[fg_calib_mask]
                fg_overlap = np.mean(expected_fg == actual_fg)

                if fg_overlap >= 0.85:
                    for c in new_cols:
                        mask = (proto == c)
                        vals = play[mask]
                        if len(vals) > 0:
                            b = np.bincount(vals, minlength=16)
                            mv = int(np.argmax(b))
                            purity = b[mv] / len(vals)
                            if purity >= 0.80 and mv not in used_obs:
                                self.perm[c] = mv
                                self.inv_perm[mv] = c
                                self.calibrated_colors.add(c)
                                used_obs.add(mv)

            # Bijection deduction: If 15 colors are calibrated, the 16th is uniquely determined
            if len(self.calibrated_colors) == 15:
                missing_c = set(range(16)) - self.calibrated_colors
                missing_p = set(range(16)) - set(self.perm[list(self.calibrated_colors)])
                if len(missing_c) == 1 and len(missing_p) == 1:
                    c = missing_c.pop()
                    p = missing_p.pop()
                    self.perm[c] = p
                    self.inv_perm[p] = c
                    self.calibrated_colors.add(c)

        return self.is_calibrated

    def invert_frame(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Maps observed frame back to canonical color representation.
        Preserves HUD rows 60..63 unpermuted.
        """
        if frame is None or not self.is_calibrated:
            return frame

        arr = np.asarray(frame)
        was_3d = (arr.ndim == 3)
        if was_3d:
            grid = arr[-1].copy()
        else:
            grid = arr.copy()

        H, W = grid.shape
        max_r = min(H, 60)
        grid[:max_r, :] = self.inv_perm[grid[:max_r, :]]

        if was_3d:
            return grid[np.newaxis, ...]
        return grid
