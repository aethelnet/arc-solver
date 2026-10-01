"""
=============================================================================
AETHELNET DETERMINISTIC DSL ENGINE FOR ARC-AGI
=============================================================================
Pure deterministic, sub-millisecond execution engine.
Induces symbolic transforms that satisfy 100% of training pairs.
Zero hallucination, zero LLM syntax errors.
=============================================================================
"""

import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from scipy.ndimage import label, binary_fill_holes


# -----------------------------------------------------------------------------
# 1. Perception & Helpers
# -----------------------------------------------------------------------------
def get_bg(grid: np.ndarray) -> int:
    """ARC standard: 0 is black/background if present, else mode."""
    if 0 in grid:
        return 0
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])

def find_dividers(grid: np.ndarray) -> Dict[str, List[int]]:
    h, w = grid.shape
    bg = get_bg(grid)
    h_divs = [r for r in range(h) if len(np.unique(grid[r, :])) == 1 and grid[r, 0] != bg]
    v_divs = [c for c in range(w) if len(np.unique(grid[:, c])) == 1 and grid[0, c] != bg]
    return {'h': h_divs, 'v': v_divs}

def crop_box(g: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return g[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()


# -----------------------------------------------------------------------------
# 2. Geometric Primitives (D4 Group)
# -----------------------------------------------------------------------------
def geom_id(g: np.ndarray) -> np.ndarray: return g.copy()
def geom_rot90(g: np.ndarray) -> np.ndarray: return np.rot90(g, -1)
def geom_rot180(g: np.ndarray) -> np.ndarray: return np.rot90(g, -2)
def geom_rot270(g: np.ndarray) -> np.ndarray: return np.rot90(g, -3)
def geom_flip_h(g: np.ndarray) -> np.ndarray: return np.flip(g, axis=0)
def geom_flip_v(g: np.ndarray) -> np.ndarray: return np.flip(g, axis=1)
def geom_transpose(g: np.ndarray) -> np.ndarray: return g.T
def geom_antitranspose(g: np.ndarray) -> np.ndarray: return np.rot90(g.T, 2)

D4_OPS = [
    ("rot90", geom_rot90),
    ("rot180", geom_rot180),
    ("rot270", geom_rot270),
    ("flip_h", geom_flip_h),
    ("flip_v", geom_flip_v),
    ("transpose", geom_transpose),
    ("antitranspose", geom_antitranspose),
]


# -----------------------------------------------------------------------------
# 3. Cropping & Object Primitives
# -----------------------------------------------------------------------------
def crop_non_bg(g: np.ndarray) -> Optional[np.ndarray]:
    return crop_box(g, g != get_bg(g))

def crop_rarest_color(g: np.ndarray) -> Optional[np.ndarray]:
    bg = get_bg(g)
    vals, counts = np.unique(g, return_counts=True)
    non_bg = [(v, c) for v, c in zip(vals, counts) if v != bg]
    if not non_bg:
        return None
    rarest = min(non_bg, key=lambda x: x[1])[0]
    return crop_box(g, g == rarest)

def crop_most_freq_fg(g: np.ndarray) -> Optional[np.ndarray]:
    bg = get_bg(g)
    vals, counts = np.unique(g, return_counts=True)
    non_bg = [(v, c) for v, c in zip(vals, counts) if v != bg]
    if not non_bg:
        return None
    most_freq = max(non_bg, key=lambda x: x[1])[0]
    return crop_box(g, g == most_freq)

def get_connected_objects(g: np.ndarray, bg: Optional[int] = None) -> List[np.ndarray]:
    if bg is None:
        bg = get_bg(g)
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
    objs = get_connected_objects(g)
    return max(objs, key=lambda o: o.size) if objs else None

def extract_smallest_obj(g: np.ndarray) -> Optional[np.ndarray]:
    objs = get_connected_objects(g)
    return min(objs, key=lambda o: o.size) if objs else None


# -----------------------------------------------------------------------------
# 4. Scaling, Tiling, Fractal
# -----------------------------------------------------------------------------
def make_tile(fy: int, fx: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda g: np.tile(g, (fy, fx))

def make_upscale(f: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda g: np.repeat(np.repeat(g, f, axis=0), f, axis=1)

def fractal_expand(g: np.ndarray) -> Optional[np.ndarray]:
    bg = get_bg(g)
    h, w = g.shape
    if h > 10 or w > 10:
        return None
    mask = (g != bg).astype(int)
    res = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                res[r*h:(r+1)*h, c*w:(c+1)*w] = g
            else:
                res[r*h:(r+1)*h, c*w:(c+1)*w] = bg
    return res


# -----------------------------------------------------------------------------
# 5. Physics & Symmetry
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

def gravity_up(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    res = np.zeros_like(g)
    for c in range(w):
        col = g[:, c]
        non_zeros = col[col != 0]
        if len(non_zeros) > 0:
            res[:len(non_zeros), c] = non_zeros
    return res

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


# -----------------------------------------------------------------------------
# 6. Dividers & Multi-Panel Boolean Logic
# -----------------------------------------------------------------------------
def op_boolean_v(g: np.ndarray, mode: str, target_color: Optional[int] = None) -> Optional[np.ndarray]:
    divs = find_dividers(g)['v']
    if not divs:
        return None
    c_div = divs[0]
    left = g[:, :c_div]
    right = g[:, c_div+1:]
    min_w = min(left.shape[1], right.shape[1])
    h = g.shape[0]
    l_sub = left[:, :min_w]
    r_sub = right[:, :min_w]
    bg = get_bg(g)
    
    out = np.zeros((h, min_w), dtype=int) + bg
    l_mask = (l_sub != bg)
    r_mask = (r_sub != bg)
    
    if mode == "intersect":
        mask = l_mask & r_mask
        tc = target_color if target_color is not None else 2
        out[mask] = tc
    elif mode == "union":
        out[l_mask] = l_sub[l_mask]
        out[r_mask] = r_sub[r_mask]
        if target_color is not None:
            out[out != bg] = target_color
    elif mode == "xor":
        mask = l_mask ^ r_mask
        tc = target_color if target_color is not None else 2
        out[mask] = tc
    return out

def op_boolean_h(g: np.ndarray, mode: str, target_color: Optional[int] = None) -> Optional[np.ndarray]:
    divs = find_dividers(g)['h']
    if not divs:
        return None
    r_div = divs[0]
    top = g[:r_div, :]
    bot = g[r_div+1:, :]
    min_h = min(top.shape[0], bot.shape[0])
    w = g.shape[1]
    t_sub = top[:min_h, :]
    b_sub = bot[:min_h, :]
    bg = get_bg(g)
    
    out = np.zeros((min_h, w), dtype=int) + bg
    t_mask = (t_sub != bg)
    b_mask = (b_sub != bg)
    
    if mode == "intersect":
        mask = t_mask & b_mask
        tc = target_color if target_color is not None else 2
        out[mask] = tc
    elif mode == "union":
        out[t_mask] = t_sub[t_mask]
        out[b_mask] = b_sub[b_mask]
        if target_color is not None:
            out[out != bg] = target_color
    elif mode == "xor":
        mask = t_mask ^ b_mask
        tc = target_color if target_color is not None else 2
        out[mask] = tc
    return out

def extract_panel(g: np.ndarray, axis: str, index: int) -> Optional[np.ndarray]:
    divs = find_dividers(g)[axis]
    if not divs:
        return None
    if axis == 'v':
        splits = [0] + divs + [g.shape[1]]
        if index < 0 or index >= len(splits) - 1:
            return None
        return g[:, splits[index]:splits[index+1]]
    else:
        splits = [0] + divs + [g.shape[0]]
        if index < 0 or index >= len(splits) - 1:
            return None
        return g[splits[index]:splits[index+1], :]


# -----------------------------------------------------------------------------
# 7. Topology (Hole Filling)
# -----------------------------------------------------------------------------
def fill_holes(g: np.ndarray, fill_color: int) -> np.ndarray:
    bg = get_bg(g)
    fg_mask = (g != bg)
    filled_mask = binary_fill_holes(fg_mask)
    holes = filled_mask & (~fg_mask)
    res = g.copy()
    res[holes] = fill_color
    return res


# -----------------------------------------------------------------------------
# 8. Color Permutation Inducer
# -----------------------------------------------------------------------------
def induce_color_map(train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Dict[int, int]]:
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
# 9. Morphology & Shape Detection (Crosses, Frames, Squares, Singletons)
# -----------------------------------------------------------------------------
def op_recolor_cross(g: np.ndarray, src_col: int, dst_col: int) -> np.ndarray:
    out = g.copy()
    h, w = g.shape
    for r in range(1, h-1):
        for c in range(1, w-1):
            if (g[r, c] == src_col and g[r-1, c] == src_col and g[r+1, c] == src_col and
                g[r, c-1] == src_col and g[r, c+1] == src_col):
                out[r, c] = dst_col
                out[r-1, c] = dst_col
                out[r+1, c] = dst_col
                out[r, c-1] = dst_col
                out[r, c+1] = dst_col
    return out

def op_recolor_hollow_box(g: np.ndarray, src_col: int, dst_col: int) -> np.ndarray:
    out = g.copy()
    h, w = g.shape
    bg = get_bg(g)
    for r in range(1, h-1):
        for c in range(1, w-1):
            if g[r, c] == bg:
                neighbors = [
                    g[r-1, c-1], g[r-1, c], g[r-1, c+1],
                    g[r, c-1],              g[r, c+1],
                    g[r+1, c-1], g[r+1, c], g[r+1, c+1]
                ]
                if all(n == src_col for n in neighbors):
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr != 0 or dc != 0:
                                out[r+dr, c+dc] = dst_col
    return out

def op_recolor_2x2(g: np.ndarray, src_col: int, dst_col: int) -> np.ndarray:
    out = g.copy()
    h, w = g.shape
    for r in range(h-1):
        for c in range(w-1):
            if (g[r, c] == src_col and g[r+1, c] == src_col and
                g[r, c+1] == src_col and g[r+1, c+1] == src_col):
                out[r:r+2, c:c+2] = dst_col
    return out

def op_recolor_singletons(g: np.ndarray, src_col: int, dst_col: int) -> np.ndarray:
    out = g.copy()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if g[r, c] == src_col:
                is_isolated = True
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if g[nr, nc] == src_col:
                                is_isolated = False
                                break
                    if not is_isolated: break
                if is_isolated:
                    out[r, c] = dst_col
    return out


# -----------------------------------------------------------------------------
# 10. Ray Casting & Beam Extrapolation
# -----------------------------------------------------------------------------
def op_ray_cast_diagonals(g: np.ndarray, emitter_col: int, ray_col: int) -> np.ndarray:
    out = g.copy()
    h, w = g.shape
    bg = get_bg(g)
    ys, xs = np.where(g == emitter_col)
    for r, c in zip(ys, xs):
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            cr, cc = r + dr, c + dc
            while 0 <= cr < h and 0 <= cc < w:
                if out[cr, cc] != bg and out[cr, cc] != ray_col:
                    break
                out[cr, cc] = ray_col
                cr += dr
                cc += dc
    return out

def op_ray_cast_cardinals(g: np.ndarray, emitter_col: int, ray_col: int) -> np.ndarray:
    out = g.copy()
    h, w = g.shape
    bg = get_bg(g)
    ys, xs = np.where(g == emitter_col)
    for r, c in zip(ys, xs):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            cr, cc = r + dr, c + dc
            while 0 <= cr < h and 0 <= cc < w:
                if out[cr, cc] != bg and out[cr, cc] != ray_col:
                    break
                out[cr, cc] = ray_col
                cr += dr
                cc += dc
    return out

def op_connect_dots_axis(g: np.ndarray, dot_col: int, line_col: int, axis: int) -> np.ndarray:
    out = g.copy()
    h, w = g.shape
    if axis == 0:  # horizontal
        for r in range(h):
            cols = np.where(g[r, :] == dot_col)[0]
            if len(cols) >= 2:
                for i in range(len(cols) - 1):
                    c1, c2 = cols[i], cols[i+1]
                    out[r, c1+1:c2] = line_col
    else:  # vertical
        for c in range(w):
            rows = np.where(g[:, c] == dot_col)[0]
            if len(rows) >= 2:
                for i in range(len(rows) - 1):
                    r1, r2 = rows[i], rows[i+1]
                    out[r1+1:r2, c] = line_col
# -----------------------------------------------------------------------------
# 11. Stride-Based Periodic Extrapolation & Multi-Panel Strip Mapping
# -----------------------------------------------------------------------------
def op_extrapolate_1d(grid: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    h, w = grid.shape
    if axis is None:
        h_pairs = 0
        for r in range(h):
            nz = grid[r, :][grid[r, :] != 0]
            if len(nz) >= 2:
                _, counts = np.unique(nz, return_counts=True)
                if any(c == 2 for c in counts):
                    h_pairs += 1
        v_pairs = 0
        for c in range(w):
            nz = grid[:, c][grid[:, c] != 0]
            if len(nz) >= 2:
                _, counts = np.unique(nz, return_counts=True)
                if any(c == 2 for c in counts):
                    v_pairs += 1
        axis = 0 if h_pairs >= v_pairs else 1

    out = grid.copy()
    num_lines = h if axis == 0 else w
    line_len = w if axis == 0 else h

    for l in range(num_lines):
        line = grid[l, :] if axis == 0 else grid[:, l]
        nz_indices = np.where(line != 0)[0]
        if len(nz_indices) < 2:
            continue

        color_pos = {}
        for idx in nz_indices:
            c = int(line[idx])
            color_pos.setdefault(c, []).append(idx)

        pair_found = False
        stride = None
        pair_color = None
        pair_indices = None

        for c, pos_list in color_pos.items():
            if len(pos_list) == 2:
                pair_color = c
                pair_indices = sorted(pos_list)
                stride = pair_indices[1] - pair_indices[0]
                pair_found = True
                break

        if not pair_found or stride is None or stride <= 0:
            continue

        base_pos = pair_indices[0]
        endpoint = None
        fill_color = pair_color

        for idx in nz_indices:
            val = int(line[idx])
            if val != pair_color and (idx - base_pos) % stride == 0:
                endpoint = idx
                fill_color = val
                break

        if endpoint is not None:
            min_i = min(pair_indices[0], endpoint)
            max_i = max(pair_indices[1], endpoint)
            lattice_indices = [i for i in range(min_i, max_i + 1, stride)]
        else:
            k_min = - (base_pos // stride)
            k_max = (line_len - 1 - base_pos) // stride
            lattice_indices = [base_pos + k * stride for k in range(k_min, k_max + 1)]

        for idx in lattice_indices:
            if axis == 0:
                out[l, idx] = fill_color
            else:
                out[idx, l] = fill_color

    return out

def op_map_strips(grid: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], axis: str = 'v') -> Optional[np.ndarray]:
    divs = find_dividers(grid)[axis]
    if not divs:
        return None
    out = grid.copy()
    total_len = grid.shape[1] if axis == 'v' else grid.shape[0]
    splits = [0] + divs + [total_len]
    for i in range(len(splits) - 1):
        s, e = splits[i], splits[i+1]
        if s in divs: s += 1
        if e > s:
            if axis == 'v':
                sub = grid[:, s:e]
                res = fn(sub)
                if res is None or res.shape != sub.shape: return None
                out[:, s:e] = res
            else:
                sub = grid[s:e, :]
                res = fn(sub)
                if res is None or res.shape != sub.shape: return None
                out[s:e, :] = res
    return out

def op_wavelength_emission(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    best_c = None
    best_count = 0
    for c in range(w):
        col = grid[:, c]
        nz = col[col != 0]
        if len(nz) >= h * 0.7:
            u, counts = np.unique(nz, return_counts=True)
            if len(u) == 1 and counts[0] > best_count:
                best_count = counts[0]
                best_c = c

    if best_c is None:
        return out

    div_c = best_c
    left_nz = np.sum(grid[:, :div_c] != 0)
    right_nz = np.sum(grid[:, div_c+1:] != 0)
    prop_direction = 1 if left_nz >= right_nz else -1

    for r in range(h):
        if prop_direction == 1:
            seed_seg = grid[r, :div_c]
            prop_len = w - 1 - div_c
        else:
            seed_seg = grid[r, div_c+1:]
            prop_len = div_c

        if np.all(seed_seg == 0):
            continue

        runs = []
        curr_c = None
        curr_len = 0
        for val in seed_seg:
            if val != 0:
                if val == curr_c:
                    curr_len += 1
                else:
                    if curr_c is not None:
                        runs.append((curr_c, curr_len))
                    curr_c = val
                    curr_len = 1
            else:
                if curr_c is not None:
                    runs.append((curr_c, curr_len))
                    curr_c = None
                    curr_len = 0
        if curr_c is not None:
            runs.append((curr_c, curr_len))

        if not runs:
            continue

        prop_res = np.zeros(prop_len, dtype=int)
        ordered_runs = list(reversed(runs)) if prop_direction == 1 else runs

        for c, L in ordered_runs:
            if L <= 0: continue
            for x in range(prop_len):
                if x % L == 0:
                    if prop_res[x] == 0:
                        prop_res[x] = c

        if prop_direction == 1:
            out[r, div_c+1:] = prop_res
        else:
            out[r, :div_c] = prop_res[::-1]

    return out


# -----------------------------------------------------------------------------
# 12. Aethelnet DSL Engine Solver
# -----------------------------------------------------------------------------
class AethelnetDSLEngine:
    def __init__(self):
        pass

    def solve_task(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable[[np.ndarray], Optional[np.ndarray]]]:
        """
        Runs inductive search over DSL primitives.
        Returns a solver function if and only if it achieves 100% exact match on train_pairs.
        """
        # Target colors from train outputs
        out_colors = set()
        in_colors = set()
        for x, y in train_pairs:
            in_colors.update(np.unique(x).tolist())
            out_colors.update(np.unique(y).tolist())
        new_colors = sorted(list(out_colors - in_colors))
        all_colors = sorted(list(out_colors.union(in_colors)))

        # Rule 1: Global Color Map Inducer
        cmap = induce_color_map(train_pairs)
        if cmap is not None:
            valid = True
            for x, y in train_pairs:
                pred = apply_color_map(x, cmap)
                if not np.array_equal(pred, y):
                    valid = False
                    break
            if valid:
                return lambda g, m=cmap: apply_color_map(g, m)

        # Rule 2: Single-step Candidate Generators
        candidates: List[Tuple[str, Callable[[np.ndarray], Optional[np.ndarray]]]] = []

        # D4 Symmetries
        candidates.extend(D4_OPS)

        # Cropping
        candidates.append(("crop_non_bg", crop_non_bg))
        candidates.append(("crop_rarest", crop_rarest_color))
        candidates.append(("crop_most_freq", crop_most_freq_fg))

        # Objects
        candidates.append(("extract_largest_obj", extract_largest_obj))
        candidates.append(("extract_smallest_obj", extract_smallest_obj))

        # Tiling & Scaling
        candidates.append(("fractal", fractal_expand))
        for fy in [1, 2, 3]:
            for fx in [1, 2, 3]:
                if fy == 1 and fx == 1: continue
                candidates.append((f"tile_{fy}x{fx}", make_tile(fy, fx)))
        candidates.append(("upscale_2", make_upscale(2)))
        candidates.append(("upscale_3", make_upscale(3)))

        # Physics & Symmetry
        candidates.append(("gravity_down", gravity_down))
        candidates.append(("gravity_up", gravity_up))
        candidates.append(("sym_h", sym_complete_h))
        candidates.append(("sym_v", sym_complete_v))

        # Hole filling
        colors_to_test = new_colors if new_colors else all_colors
        for c in colors_to_test:
            candidates.append((f"hole_fill_{c}", lambda g, col=c: fill_holes(g, col)))

        # Boolean Divider splits
        for mode in ["intersect", "union", "xor"]:
            candidates.append((f"bool_v_{mode}_none", lambda g, m=mode: op_boolean_v(g, m, None)))
            candidates.append((f"bool_h_{mode}_none", lambda g, m=mode: op_boolean_h(g, m, None)))
            for c in colors_to_test:
                candidates.append((f"bool_v_{mode}_{c}", lambda g, m=mode, col=c: op_boolean_v(g, m, col)))
                candidates.append((f"bool_h_{mode}_{c}", lambda g, m=mode, col=c: op_boolean_h(g, m, col)))

        # Morphology (Crosses, Hollow Boxes, 2x2, Singletons)
        dst_colors = new_colors if new_colors else all_colors
        for sc in in_colors:
            for dc in dst_colors:
                if sc == dc: continue
                candidates.append((f"recolor_cross_{sc}_{dc}", lambda g, s=sc, d=dc: op_recolor_cross(g, s, d)))
                candidates.append((f"recolor_hollow_{sc}_{dc}", lambda g, s=sc, d=dc: op_recolor_hollow_box(g, s, d)))
                candidates.append((f"recolor_2x2_{sc}_{dc}", lambda g, s=sc, d=dc: op_recolor_2x2(g, s, d)))
                candidates.append((f"recolor_singletons_{sc}_{dc}", lambda g, s=sc, d=dc: op_recolor_singletons(g, s, d)))

        # Ray Casting & Dot Connections
        for ec in in_colors:
            for rc in dst_colors:
                candidates.append((f"ray_diag_{ec}_{rc}", lambda g, e=ec, r=rc: op_ray_cast_diagonals(g, e, r)))
                candidates.append((f"ray_card_{ec}_{rc}", lambda g, e=ec, r=rc: op_ray_cast_cardinals(g, e, r)))
                candidates.append((f"connect_h_{ec}_{rc}", lambda g, d=ec, l=rc: op_connect_dots_axis(g, d, l, 0)))
                candidates.append((f"connect_v_{ec}_{rc}", lambda g, d=ec, l=rc: op_connect_dots_axis(g, d, l, 1)))

        # Stride-Based 1D Periodic Extrapolation
        candidates.append(("extrapolate_1d_auto", lambda g: op_extrapolate_1d(g, axis=None)))
        candidates.append(("extrapolate_1d_h", lambda g: op_extrapolate_1d(g, axis=0)))
        candidates.append(("extrapolate_1d_v", lambda g: op_extrapolate_1d(g, axis=1)))

        # Wavelength Emission from Object Lengths
        candidates.append(("wavelength_emission", op_wavelength_emission))

        # Evaluate Level-1 atomic candidates
        for name, fn in candidates:
            match = True
            for x, y in train_pairs:
                try:
                    pred = fn(x)
                    if pred is None or pred.shape != y.shape or not np.array_equal(pred, y):
                        match = False
                        break
                except Exception:
                    match = False
                    break
            if match:
                return fn

        # Level 2 Compositions: Crop/Object -> D4 Symmetries
        base_extractors = [
            ("crop_non_bg", crop_non_bg),
            ("largest_obj", extract_largest_obj),
            ("smallest_obj", extract_smallest_obj),
        ]
        for b_name, b_fn in base_extractors:
            for d_name, d_fn in D4_OPS:
                comp_fn = lambda g, bf=b_fn, df=d_fn: df(bf(g)) if bf(g) is not None else None
                match = True
                for x, y in train_pairs:
                    try:
                        pred = comp_fn(x)
                        if pred is None or pred.shape != y.shape or not np.array_equal(pred, y):
                            match = False
                            break
                    except Exception:
                        match = False
                        break
                if match:
                    return comp_fn

        # Level 3 Compositions: Multi-panel Strip Mappings
        shape_preserving_ops = [
            ("extrapolate_1d_auto", lambda g: op_extrapolate_1d(g, axis=None)),
            ("sym_h", sym_complete_h),
            ("sym_v", sym_complete_v),
            ("gravity_down", gravity_down),
        ]
        for op_name, op_fn in shape_preserving_ops:
            for s_axis in ['v', 'h']:
                strip_fn = lambda g, fn=op_fn, ax=s_axis: op_map_strips(g, fn, axis=ax)
                match = True
                for x, y in train_pairs:
                    try:
                        pred = strip_fn(x)
                        if pred is None or pred.shape != y.shape or not np.array_equal(pred, y):
                            match = False
                            break
                    except Exception:
                        match = False
                        break
                if match:
                    return strip_fn

        return None
