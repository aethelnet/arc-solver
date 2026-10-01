"""
Exact Inductive Program Synthesis Solver for ARC-AGI-2 (ARC-1/2 Grid Transformation).
Finds deterministic symbolic rules that satisfy 100% of training pairs.
"""

import json
import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Callable
from scipy.ndimage import label


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


# -----------------------------------------------------------------------------
# 1. Geometric Primitives (D4 Group)
# -----------------------------------------------------------------------------
def geom_identity(g: np.ndarray) -> np.ndarray: return g.copy()
def geom_rot90(g: np.ndarray) -> np.ndarray: return np.rot90(g, -1)
def geom_rot180(g: np.ndarray) -> np.ndarray: return np.rot90(g, -2)
def geom_rot270(g: np.ndarray) -> np.ndarray: return np.rot90(g, -3)
def geom_flip_h(g: np.ndarray) -> np.ndarray: return np.flip(g, axis=0)
def geom_flip_v(g: np.ndarray) -> np.ndarray: return np.flip(g, axis=1)
def geom_transpose(g: np.ndarray) -> np.ndarray: return g.T
def geom_antitranspose(g: np.ndarray) -> np.ndarray: return np.rot90(g.T, 2)

GEOM_OPS: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("id", geom_identity),
    ("rot90", geom_rot90),
    ("rot180", geom_rot180),
    ("rot270", geom_rot270),
    ("flip_h", geom_flip_h),
    ("flip_v", geom_flip_v),
    ("transpose", geom_transpose),
    ("antitranspose", geom_antitranspose)
]


# -----------------------------------------------------------------------------
# 2. Cropping Primitives
# -----------------------------------------------------------------------------
def crop_box(g: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return g[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()

def crop_non_zero(g: np.ndarray) -> Optional[np.ndarray]:
    return crop_box(g, g != 0)

def crop_non_bg(g: np.ndarray) -> Optional[np.ndarray]:
    bg = get_bg(g)
    return crop_box(g, g != bg)

def crop_rarest_color(g: np.ndarray) -> Optional[np.ndarray]:
    vals, counts = np.unique(g, return_counts=True)
    bg = get_bg(g)
    non_bg = [(v, c) for v, c in zip(vals, counts) if v != bg and v != 0]
    if not non_bg:
        return None
    rarest = min(non_bg, key=lambda x: x[1])[0]
    return crop_box(g, g == rarest)

def crop_most_frequent_fg(g: np.ndarray) -> Optional[np.ndarray]:
    vals, counts = np.unique(g, return_counts=True)
    bg = get_bg(g)
    non_bg = [(v, c) for v, c in zip(vals, counts) if v != bg and v != 0]
    if not non_bg:
        return None
    most_freq = max(non_bg, key=lambda x: x[1])[0]
    return crop_box(g, g == most_freq)

CROP_OPS: List[Tuple[str, Callable[[np.ndarray], Optional[np.ndarray]]]] = [
    ("crop_non_zero", crop_non_zero),
    ("crop_non_bg", crop_non_bg),
    ("crop_rarest", crop_rarest_color),
    ("crop_most_freq", crop_most_frequent_fg)
]


# -----------------------------------------------------------------------------
# 3. Scaling & Tiling Primitives
# -----------------------------------------------------------------------------
def make_tile(fy: int, fx: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda g: np.tile(g, (fy, fx))

def make_upscale(f: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda g: np.repeat(np.repeat(g, f, axis=0), f, axis=1)

def fractal_expand(g: np.ndarray) -> Optional[np.ndarray]:
    """Kronecker product: self-similarity expansion."""
    bg = get_bg(g)
    mask = (g != bg).astype(int)
    h, w = g.shape
    if h > 10 or w > 10:
        return None
    res = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                res[r*h:(r+1)*h, c*w:(c+1)*w] = g
            else:
                res[r*h:(r+1)*h, c*w:(c+1)*w] = bg
    return res

TILE_OPS: List[Tuple[str, Callable[[np.ndarray], Optional[np.ndarray]]]] = [
    ("tile_2x2", make_tile(2, 2)),
    ("tile_3x3", make_tile(3, 3)),
    ("tile_1x2", make_tile(1, 2)),
    ("tile_2x1", make_tile(2, 1)),
    ("tile_1x3", make_tile(1, 3)),
    ("tile_3x1", make_tile(3, 1)),
    ("upscale_2", make_upscale(2)),
    ("upscale_3", make_upscale(3)),
    ("fractal", fractal_expand)
]


# -----------------------------------------------------------------------------
# 4. Color Mapping Inducer
# -----------------------------------------------------------------------------
def induce_color_map(train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Dict[int, int]]:
    """Checks if a consistent 1-to-1 or N-to-1 color mapping explains all train pairs."""
    mapping: Dict[int, int] = {}
    for x, y in train_pairs:
        if x.shape != y.shape:
            return None
        for u in range(x.shape[0]):
            for v in range(x.shape[1]):
                cx = int(x[u, v])
                cy = int(y[u, v])
                if cx in mapping and mapping[cx] != cy:
                    return None
                mapping[cx] = cy
    return mapping

def apply_color_map(g: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    res = g.copy()
    for src, dst in mapping.items():
        res[g == src] = dst
    return res


# -----------------------------------------------------------------------------
# 5. Connected Component / Object Selection
# -----------------------------------------------------------------------------
def get_connected_objects(g: np.ndarray, bg: int = 0) -> List[np.ndarray]:
    mask = (g != bg).astype(int)
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = label(mask, structure)
    objects = []
    for i in range(1, num_features + 1):
        obj_mask = (labeled == i)
        cropped = crop_box(g, obj_mask)
        if cropped is not None:
            objects.append(cropped)
    return objects

def extract_largest_obj(g: np.ndarray) -> Optional[np.ndarray]:
    objs = get_connected_objects(g, get_bg(g))
    if not objs:
        return None
    return max(objs, key=lambda o: o.size)

def extract_smallest_obj(g: np.ndarray) -> Optional[np.ndarray]:
    objs = get_connected_objects(g, get_bg(g))
    if not objs:
        return None
    return min(objs, key=lambda o: o.size)

OBJECT_OPS: List[Tuple[str, Callable[[np.ndarray], Optional[np.ndarray]]]] = [
    ("largest_obj", extract_largest_obj),
    ("smallest_obj", extract_smallest_obj)
]


# -----------------------------------------------------------------------------
# 6. Symmetry & Completion
# -----------------------------------------------------------------------------
def sym_complete_h(g: np.ndarray) -> np.ndarray:
    res = g.copy()
    bg = get_bg(g)
    flipped = np.flip(g, axis=1)
    mask = (res == bg) & (flipped != bg)
    res[mask] = flipped[mask]
    return res

def sym_complete_v(g: np.ndarray) -> np.ndarray:
    res = g.copy()
    bg = get_bg(g)
    flipped = np.flip(g, axis=0)
    mask = (res == bg) & (flipped != bg)
    res[mask] = flipped[mask]
    return res

SYMMETRY_OPS: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("sym_h", sym_complete_h),
    ("sym_v", sym_complete_v)
]


# -----------------------------------------------------------------------------
# 7. Gravity / Slide
# -----------------------------------------------------------------------------
def gravity_down(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    res = np.zeros_like(g)
    for c in range(w):
        col = g[:, c]
        non_zeros = col[col != 0]
        if len(non_zeros) > 0:
            res[h - len(non_zeros):, c] = non_zeros
    return res

GRAVITY_OPS: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("gravity_down", gravity_down)
]


# -----------------------------------------------------------------------------
# MASTER SOLVER ENGINE
# -----------------------------------------------------------------------------
class ARCSymbolicSolver:
    def __init__(self):
        self.atomic_ops: List[Tuple[str, Callable[[np.ndarray], Optional[np.ndarray]]]] = []
        self.atomic_ops.extend(GEOM_OPS)
        self.atomic_ops.extend(CROP_OPS)
        self.atomic_ops.extend(TILE_OPS)
        self.atomic_ops.extend(OBJECT_OPS)
        self.atomic_ops.extend(SYMMETRY_OPS)
        self.atomic_ops.extend(GRAVITY_OPS)

    def solve_task(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]], test_inputs: List[np.ndarray]) -> List[Dict[str, List[List[int]]]]:
        """
        Solves task by finding symbolic rules consistent with 100% of training pairs.
        Returns list of {'attempt_1': grid, 'attempt_2': grid} for each test input.
        """
        candidate_rules: List[Callable[[np.ndarray], Optional[np.ndarray]]] = []

        # 1. Check Global Color Map
        cmap = induce_color_map(train_pairs)
        if cmap is not None:
            # Verify if color map solves all train pairs
            valid = True
            for x, y in train_pairs:
                pred = apply_color_map(x, cmap)
                if not np.array_equal(pred, y):
                    valid = False
                    break
            if valid:
                candidate_rules.append(lambda g, m=cmap: apply_color_map(g, m))

        # 2. Check 1-Step Atomic Operations
        for name, op in self.atomic_ops:
            matches = True
            for x, y in train_pairs:
                try:
                    pred = op(x)
                    if pred is None or not np.array_equal(pred, y):
                        matches = False
                        break
                except Exception:
                    matches = False
                    break
            if matches:
                candidate_rules.append(op)

        # 3. Check 2-Step Compositions: op1 then op2
        if len(candidate_rules) < 2:
            composed_subset = [op for op in self.atomic_ops if op[0] in [
                "id", "rot90", "rot180", "rot270", "flip_h", "flip_v",
                "crop_non_zero", "crop_non_bg", "crop_rarest",
                "tile_2x2", "upscale_2", "sym_h", "sym_v"
            ]]
            for name1, op1 in composed_subset:
                for name2, op2 in composed_subset:
                    if name1 == "id" and name2 == "id":
                        continue
                    matches = True
                    for x, y in train_pairs:
                        try:
                            mid = op1(x)
                            if mid is None:
                                matches = False
                                break
                            pred = op2(mid)
                            if pred is None or not np.array_equal(pred, y):
                                matches = False
                                break
                        except Exception:
                            matches = False
                            break
                    if matches:
                        candidate_rules.append(lambda g, f1=op1, f2=op2: f2(f1(g)))
                        if len(candidate_rules) >= 4:
                            break
                if len(candidate_rules) >= 4:
                    break

        # 4. Check Crop + Color Mapping
        if len(candidate_rules) < 2:
            for cname, cop in CROP_OPS:
                cropped_train = []
                can_crop = True
                for x, y in train_pairs:
                    cx = cop(x)
                    if cx is None:
                        can_crop = False
                        break
                    cropped_train.append((cx, y))
                if can_crop:
                    cc_map = induce_color_map(cropped_train)
                    if cc_map is not None:
                        matches = True
                        for cx, y in cropped_train:
                            pred = apply_color_map(cx, cc_map)
                            if not np.array_equal(pred, y):
                                matches = False
                                break
                        if matches:
                            candidate_rules.append(lambda g, c=cop, m=cc_map: apply_color_map(c(g), m) if c(g) is not None else g)

        # Generate predictions for each test input
        predictions = []
        for test_in in test_inputs:
            h, w = test_in.shape
            attempt_1 = None
            attempt_2 = None

            # Attempt 1: primary consistent rule
            if len(candidate_rules) >= 1:
                try:
                    p1 = candidate_rules[0](test_in)
                    if p1 is not None and isinstance(p1, np.ndarray) and p1.size > 0:
                        attempt_1 = p1.tolist()
                except Exception:
                    pass

            # Attempt 2: secondary rule or identity
            if len(candidate_rules) >= 2:
                try:
                    p2 = candidate_rules[1](test_in)
                    if p2 is not None and isinstance(p2, np.ndarray) and p2.size > 0:
                        attempt_2 = p2.tolist()
                except Exception:
                    pass

            # Fallbacks if rules failed or not found
            if attempt_1 is None:
                attempt_1 = test_in.tolist()
            if attempt_2 is None:
                # Fallback to cropped or identity
                c = crop_non_zero(test_in)
                attempt_2 = c.tolist() if c is not None else test_in.tolist()

            predictions.append({
                "attempt_1": attempt_1,
                "attempt_2": attempt_2
            })

        return predictions
