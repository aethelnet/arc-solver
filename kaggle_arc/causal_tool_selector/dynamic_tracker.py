"""
=============================================================================
ARC-AGI CAUSAL META-TOOL SELECTOR: SPATIO-TEMPORAL DYNAMIC TRACKER & PLANNER
=============================================================================
Provides domain-agnostic tracking, trajectory forecasting, and space-time
reservation planning for moving obstacles, hazards, and autonomous NPCs:

1. DynamicEntityTracker:
   - Identifies non-avatar autonomous moving entities across consecutive frames.
   - Infers velocity vectors v = (Δr, Δc) and motion regimes (linear bounce, periodic patrol).
   - Generates dynamic 3D space-time reservation tables R(t) over horizon H.
2. space_time_a_star:
   - 3D Space-Time A* search over state space (r, c, t).
   - Vertex collision avoidance: (r, c) ∉ R(t).
   - Swap / Edge collision avoidance: ¬((r', c') ∈ R(t) ∧ (r, c) ∈ R(t+1)).
   - Non-destructive wait action synthesis (Action 5 or wall bump) for hazard clearance.
   - Coupled morphology support (avatar + carried entity joint collision avoidance).
=============================================================================
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional, Set
import collections
import heapq
import numpy as np


DIR_MAP = {
    1: (-1, 0),  # UP
    2: (1, 0),   # DOWN
    3: (0, -1),  # LEFT
    4: (0, 1)    # RIGHT
}

REV_DIR = {v: k for k, v in DIR_MAP.items()}


def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class MotionPattern(Enum):
    STATIC = "STATIC"
    LINEAR_BOUNCE = "LINEAR_BOUNCE"
    PERIODIC = "PERIODIC"
    AUTONOMOUS = "AUTONOMOUS"


@dataclass
class TrackedEntity:
    entity_id: int
    color: int
    centroid: Tuple[int, int]
    relative_pixels: Set[Tuple[int, int]]  # Footprint relative to centroid
    history: List[Tuple[int, int]] = field(default_factory=list)
    velocity: Optional[Tuple[int, int]] = None
    pattern: MotionPattern = MotionPattern.STATIC
    cycle: List[Tuple[int, int]] = field(default_factory=list)
    cycle_idx: int = 0
    last_seen_step: int = 0


@dataclass
class CarrierPlatform:
    """
    Spatiotemporal representation of a moving platform / carrier surface.
    """
    entity_id: int
    color: int
    body_by_time: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)
    support_by_time: Dict[int, Set[Tuple[int, int]]] = field(default_factory=dict)
    velocity_by_time: Dict[int, Tuple[int, int]] = field(default_factory=dict)


class DynamicEntityTracker:
    """
    Domain-agnostic tracker for moving hazards, patrol obstacles, and NPCs.
    Infers velocity and motion patterns from successive frame observations.
    """
    def __init__(self, history_capacity: int = 40):
        self.history_capacity = history_capacity
        self.entities: Dict[int, TrackedEntity] = {}
        self.next_entity_id: int = 1
        self.current_step: int = 0
        self.last_grid: Optional[np.ndarray] = None
        self.last_avatar_pos: Optional[Tuple[int, int]] = None

    def reset(self):
        """Resets tracker state across game resets."""
        self.entities.clear()
        self.next_entity_id = 1
        self.current_step = 0
        self.last_grid = None
        self.last_avatar_pos = None

    def _extract_components(
        self,
        grid: np.ndarray,
        bg_canvas: int,
        avatar_pos: Optional[Tuple[int, int]] = None,
        avatar_color: Optional[int] = None,
        floor_colors: Optional[Set[int]] = None
    ) -> List[Dict[str, Any]]:
        """Extracts connected components excluding background, floors, and avatar."""
        H, W = grid.shape
        floors = set(floor_colors) if floor_colors else set()
        visited = set()
        components = []

        for r in range(H):
            for c in range(W):
                val = int(grid[r, c])
                if val == bg_canvas or val in floors or (r, c) in visited:
                    continue
                # Check if this cell belongs to avatar
                if avatar_pos is not None and (r, c) == avatar_pos:
                    continue
                if avatar_color is not None and val == avatar_color:
                    # If avatar has unique color and matches this cell
                    if avatar_pos is not None and abs(r - avatar_pos[0]) + abs(c - avatar_pos[1]) <= 2:
                        continue

                # Flood-fill component
                comp_cells = set()
                q = [(r, c)]
                visited.add((r, c))
                comp_cells.add((r, c))

                while q:
                    cr, cc = q.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                            if int(grid[nr, nc]) == val:
                                visited.add((nr, nc))
                                comp_cells.add((nr, nc))
                                q.append((nr, nc))

                # If component includes avatar position, skip
                if avatar_pos is not None and avatar_pos in comp_cells:
                    continue

                cells_list = list(comp_cells)
                cr = int(round(sum(p[0] for p in cells_list) / len(cells_list)))
                cc = int(round(sum(p[1] for p in cells_list) / len(cells_list)))
                centroid = (cr, cc)
                rel_pixels = {(p[0] - cr, p[1] - cc) for p in cells_list}

                components.append({
                    "color": val,
                    "centroid": centroid,
                    "cells": comp_cells,
                    "rel_pixels": rel_pixels,
                    "size": len(comp_cells)
                })

        return components

    def observe_frame(
        self,
        grid: np.ndarray,
        avatar_pos: Optional[Tuple[int, int]] = None,
        avatar_color: Optional[int] = None,
        floor_colors: Optional[Set[int]] = None,
        action_id: Optional[int] = None
    ) -> List[TrackedEntity]:
        """
        Updates tracking state from a new frame observation.
        Matches components to existing tracklets and detects autonomous motion.
        """
        grid_2d = np.asarray(grid)
        while grid_2d.ndim > 2:
            grid_2d = grid_2d[-1]

        bg_canvas = int(collections.Counter(grid_2d.flatten()).most_common(1)[0][0])
        components = self._extract_components(grid_2d, bg_canvas, avatar_pos, avatar_color, floor_colors)
        self.current_step += 1

        matched_entity_ids = set()

        for comp in components:
            c_color = comp["color"]
            c_cent = comp["centroid"]
            c_size = comp["size"]
            c_rel = comp["rel_pixels"]

            # Find best match among existing entities
            best_id = None
            best_dist = float("inf")

            for eid, ent in self.entities.items():
                if eid in matched_entity_ids:
                    continue
                if ent.color != c_color:
                    continue
                dist = manhattan(ent.centroid, c_cent)
                # Proximity gate (max step distance 4)
                if dist < 4 and dist < best_dist:
                    best_dist = dist
                    best_id = eid

            if best_id is not None:
                matched_entity_ids.add(best_id)
                ent = self.entities[best_id]
                prev_cent = ent.centroid
                dr = c_cent[0] - prev_cent[0]
                dc = c_cent[1] - prev_cent[1]

                ent.centroid = c_cent
                ent.relative_pixels = c_rel
                ent.history.append(c_cent)
                if len(ent.history) > self.history_capacity:
                    ent.history.pop(0)
                ent.last_seen_step = self.current_step

                # Motion detection:
                if (dr != 0 or dc != 0):
                    # Check if avatar caused this motion via direct push
                    was_pushed = False
                    if self.last_avatar_pos is not None and action_id in DIR_MAP:
                        p_dr, p_dc = DIR_MAP[action_id]
                        # Check if entity was adjacent to avatar before motion
                        dist_to_prev = abs(self.last_avatar_pos[0] - prev_cent[0]) + abs(self.last_avatar_pos[1] - prev_cent[1])
                        is_aligned = (dr * p_dr >= 0 and dc * p_dc >= 0)
                        if dist_to_prev <= 6 and is_aligned:
                            was_pushed = True

                    if not was_pushed:
                        ent.velocity = (dr, dc)
                        self._update_motion_pattern(ent)
                    else:
                        ent.velocity = None
                        ent.pattern = MotionPattern.STATIC
            else:
                # New entity detected
                new_ent = TrackedEntity(
                    entity_id=self.next_entity_id,
                    color=c_color,
                    centroid=c_cent,
                    relative_pixels=c_rel,
                    history=[c_cent],
                    velocity=None,
                    pattern=MotionPattern.STATIC,
                    last_seen_step=self.current_step
                )
                self.entities[self.next_entity_id] = new_ent
                self.next_entity_id += 1

        # Prune stale entities not seen for > 5 steps
        stale_ids = [eid for eid, ent in self.entities.items() if self.current_step - ent.last_seen_step > 5]
        for eid in stale_ids:
            del self.entities[eid]

        self.last_grid = grid_2d.copy()
        self.last_avatar_pos = avatar_pos

        return [e for e in self.entities.values() if e.pattern != MotionPattern.STATIC]

    def _update_motion_pattern(self, entity: TrackedEntity):
        """Analyzes tracklet history to detect periodic cycles or linear bounces."""
        hist = entity.history
        if len(hist) < 4:
            entity.pattern = MotionPattern.LINEAR_BOUNCE
            return

        # Check for periodicity: cycle length P between 2 and len(hist) // 2
        for period in range(2, len(hist) // 2 + 1):
            is_cycle = True
            for i in range(period):
                if hist[-1 - i] != hist[-1 - i - period]:
                    is_cycle = False
                    break
            if is_cycle:
                entity.pattern = MotionPattern.PERIODIC
                entity.cycle = hist[-period:]
                entity.cycle_idx = entity.cycle.index(entity.centroid) if entity.centroid in entity.cycle else 0
                return

        # Default to linear bounce with current velocity
        entity.pattern = MotionPattern.LINEAR_BOUNCE

    def build_reservation_table(
        self,
        grid: np.ndarray,
        horizon: int = 40,
        static_blocked: Optional[Set[Tuple[int, int]]] = None,
        stride: int = 1
    ) -> Dict[int, Set[Tuple[int, int]]]:
        """
        Projects future occupations R(t) for t in [0, horizon].
        R(t) contains all cells occupied by dynamic entities at time step t.
        """
        grid_2d = np.asarray(grid)
        while grid_2d.ndim > 2:
            grid_2d = grid_2d[-1]
        H, W = grid_2d.shape
        blocked = set(static_blocked) if static_blocked else set()

        reservations: Dict[int, Set[Tuple[int, int]]] = {t: set() for t in range(horizon + 1)}

        for ent in self.entities.values():
            if ent.pattern == MotionPattern.STATIC or ent.velocity is None:
                continue

            if ent.pattern == MotionPattern.PERIODIC and ent.cycle:
                # Periodic cycle projection
                cycle_len = len(ent.cycle)
                for t in range(horizon + 1):
                    cent = ent.cycle[(ent.cycle_idx + t) % cycle_len]
                    for r_off, c_off in ent.relative_pixels:
                        occ = (cent[0] + r_off, cent[1] + c_off)
                        if 0 <= occ[0] < H and 0 <= occ[1] < W:
                            reservations[t].add(occ)

            elif ent.pattern in (MotionPattern.LINEAR_BOUNCE, MotionPattern.AUTONOMOUS):
                # Forward linear trajectory with wall/boundary reflection
                curr_cent = ent.centroid
                curr_vel = ent.velocity
                for t in range(horizon + 1):
                    # Add current footprint
                    for r_off, c_off in ent.relative_pixels:
                        occ = (curr_cent[0] + r_off, curr_cent[1] + c_off)
                        if 0 <= occ[0] < H and 0 <= occ[1] < W:
                            reservations[t].add(occ)

                    # Compute next position
                    next_cent = (curr_cent[0] + curr_vel[0] * stride, curr_cent[1] + curr_vel[1] * stride)
                    # Check boundary or static wall collision
                    hits_wall = False
                    for r_off, c_off in ent.relative_pixels:
                        check_cell = (next_cent[0] + r_off, next_cent[1] + c_off)
                        if not (0 <= check_cell[0] < H and 0 <= check_cell[1] < W):
                            hits_wall = True
                            break
                        if check_cell in blocked or int(grid_2d[check_cell[0], check_cell[1]]) in blocked:
                            hits_wall = True
                            break

                    if hits_wall:
                        # Bounce: reverse velocity
                        curr_vel = (-curr_vel[0], -curr_vel[1])
                        next_cent = (curr_cent[0] + curr_vel[0] * stride, curr_cent[1] + curr_vel[1] * stride)

                    curr_cent = next_cent

        return reservations

    def predict_spacetime_reservations(
        self,
        grid: np.ndarray,
        horizon: int = 40,
        static_blocked: Optional[Any] = None,
        stride: int = 1
    ) -> Dict[int, Set[Tuple[int, int]]]:
        """Alias for build_reservation_table for uniform spacetime interface."""
        return self.build_reservation_table(
            grid=grid,
            horizon=horizon,
            static_blocked=static_blocked,
            stride=stride
        )


    def get_carrier_platforms(
        self,
        grid: np.ndarray,
        horizon: int = 40,
        solid_colors: Optional[Set[int]] = None,
        hazard_colors: Optional[Set[int]] = None,
        static_blocked: Optional[Set[Tuple[int, int]]] = None,
        gravity: Tuple[int, int] = (1, 0),
        stride: int = 1
    ) -> List[CarrierPlatform]:
        """
        Identifies and spatiotemporally projects moving carrier platforms.
        An entity is a carrier platform if:
        1. Its color is not in hazard_colors.
        2. Its pattern is non-STATIC with active velocity.
        3. Its geometry forms a support surface (e.g. width >= 2 or solid_colors match).
        """
        grid_2d = np.asarray(grid)
        while grid_2d.ndim > 2:
            grid_2d = grid_2d[-1]
        H, W = grid_2d.shape
        blocked = set(static_blocked) if static_blocked else set()
        solids = set(solid_colors) if solid_colors else set()
        hazards = set(hazard_colors) if hazard_colors else set()

        platforms: List[CarrierPlatform] = []

        for ent in self.entities.values():
            if ent.pattern == MotionPattern.STATIC or ent.velocity is None:
                continue

            # If explicitly classified as lethal hazard, cannot be a platform
            if ent.color in hazards:
                continue

            # Check morphology: horizontal width or solid color
            cols = [c for r, c in ent.relative_pixels]
            width = (max(cols) - min(cols) + 1) if cols else 1
            is_platform_candidate = (ent.color in solids) or (width >= 2)

            if not is_platform_candidate:
                continue

            plat = CarrierPlatform(entity_id=ent.entity_id, color=ent.color)

            if ent.pattern == MotionPattern.PERIODIC and ent.cycle:
                cycle_len = len(ent.cycle)
                for t in range(horizon + 1):
                    cent = ent.cycle[(ent.cycle_idx + t) % cycle_len]
                    next_cent = ent.cycle[(ent.cycle_idx + t + 1) % cycle_len]
                    vel = (next_cent[0] - cent[0], next_cent[1] - cent[1])
                    plat.velocity_by_time[t] = vel

                    body = set()
                    for r_off, c_off in ent.relative_pixels:
                        b_cell = (cent[0] + r_off, cent[1] + c_off)
                        if 0 <= b_cell[0] < H and 0 <= b_cell[1] < W:
                            body.add(b_cell)
                    plat.body_by_time[t] = body

                    # Support surface: cells immediately opposite gravity
                    gr, gc = gravity
                    support = set()
                    for br, bc in body:
                        sr, sc = br - gr * stride, bc - gc * stride
                        if 0 <= sr < H and 0 <= sc < W and (sr, sc) not in body:
                            support.add((sr, sc))
                    plat.support_by_time[t] = support

            elif ent.pattern in (MotionPattern.LINEAR_BOUNCE, MotionPattern.AUTONOMOUS):
                curr_cent = ent.centroid
                curr_vel = ent.velocity
                for t in range(horizon + 1):
                    body = set()
                    for r_off, c_off in ent.relative_pixels:
                        b_cell = (curr_cent[0] + r_off, curr_cent[1] + c_off)
                        if 0 <= b_cell[0] < H and 0 <= b_cell[1] < W:
                            body.add(b_cell)
                    plat.body_by_time[t] = body

                    gr, gc = gravity
                    support = set()
                    for br, bc in body:
                        sr, sc = br - gr * stride, bc - gc * stride
                        if 0 <= sr < H and 0 <= sc < W and (sr, sc) not in body:
                            support.add((sr, sc))
                    plat.support_by_time[t] = support

                    # Next position and bounce detection
                    next_cent = (curr_cent[0] + curr_vel[0] * stride, curr_cent[1] + curr_vel[1] * stride)
                    hits_wall = False
                    for r_off, c_off in ent.relative_pixels:
                        check_cell = (next_cent[0] + r_off, next_cent[1] + c_off)
                        if not (0 <= check_cell[0] < H and 0 <= check_cell[1] < W) or check_cell in blocked:
                            hits_wall = True
                            break

                    actual_vel = curr_vel
                    if hits_wall:
                        curr_vel = (-curr_vel[0], -curr_vel[1])
                        next_cent = (curr_cent[0] + curr_vel[0] * stride, curr_cent[1] + curr_vel[1] * stride)
                        actual_vel = curr_vel

                    plat.velocity_by_time[t] = actual_vel
                    curr_cent = next_cent

            platforms.append(plat)

        return platforms


def space_time_a_star(
    grid: np.ndarray,
    start_pos: Tuple[int, int],
    goal_candidates: Set[Tuple[int, int]],
    reservations: Dict[int, Set[Tuple[int, int]]],
    static_blocked: Set[Tuple[int, int]],
    box_offset: Optional[Tuple[int, int]] = None,
    max_time: int = 50,
    available_actions: Optional[List[int]] = None,
    wait_action: Optional[int] = None,
    stride: int = 1,
    carrier_platforms: Optional[List[CarrierPlatform]] = None,
    gravity: Optional[Tuple[int, int]] = None
) -> Optional[List[int]]:
    """
    3D Space-Time A* Search: (r, c, t)
    Finds optimal collision-free trajectory navigating around dynamic obstacles.
    Supports non-destructive wait actions, composite rigid body footprints (carried items),
    and moving carrier platforms with passive momentum carriage.
    """
    grid_2d = np.asarray(grid)
    while grid_2d.ndim > 2:
        grid_2d = grid_2d[-1]
    H, W = grid_2d.shape

    actions_pool = available_actions or [1, 2, 3, 4, 5]
    platforms = carrier_platforms or []

    def heuristic(r: int, c: int) -> float:
        if not goal_candidates:
            return 0.0
        if box_offset is not None:
            br, bc = r + box_offset[0], c + box_offset[1]
            return min(abs(br - g[0]) + abs(bc - g[1]) for g in goal_candidates) / float(stride)
        return min(abs(r - g[0]) + abs(c - g[1]) for g in goal_candidates) / float(stride)

    def find_safe_wait_action(r: int, c: int) -> Optional[int]:
        """Synthesizes non-destructive wait: Action 5 or bumping adjacent solid wall."""
        if wait_action is not None and wait_action in actions_pool:
            return wait_action
        # If carrying, Action 5 places the box, so we prefer bumping a wall to wait!
        if box_offset is None and 5 in actions_pool:
            return 5
        # Find adjacent wall or boundary to bump non-destructively
        for act, (dr, dc) in DIR_MAP.items():
            if act not in actions_pool:
                continue
            adj = (r + dr * stride, c + dc * stride)
            is_wall = (adj in static_blocked) or not (0 <= adj[0] < H and 0 <= adj[1] < W)
            if box_offset is not None:
                b_adj = (adj[0] + box_offset[0], adj[1] + box_offset[1])
                b_is_wall = (b_adj in static_blocked) or not (0 <= b_adj[0] < H and 0 <= b_adj[1] < W)
                if is_wall or b_is_wall:
                    return act
            elif is_wall:
                return act
        # Fallback to Action 5 if available
        if 5 in actions_pool:
            return 5
        return None

    # Priority queue: (f_score, g_score, r, c, t, actions)
    h0 = heuristic(start_pos[0], start_pos[1])
    pq: List[Tuple[float, int, int, int, int, List[int]]] = [(h0, 0, start_pos[0], start_pos[1], 0, [])]
    visited: Set[Tuple[int, int, int]] = set()

    while pq:
        f, g, r, c, t, actions = heapq.heappop(pq)
        state_key = (r, c, t)
        if state_key in visited:
            continue
        visited.add(state_key)

        # Check goal condition
        effective_check = (r + box_offset[0], c + box_offset[1]) if box_offset is not None else (r, c)
        if effective_check in goal_candidates:
            return actions

        if t >= max_time:
            continue

        # Check if avatar is supported by a carrier platform at time t
        riding_plat: Optional[CarrierPlatform] = None
        plat_vel: Tuple[int, int] = (0, 0)
        for p in platforms:
            if (r, c) in p.support_by_time.get(t, set()):
                riding_plat = p
                plat_vel = p.velocity_by_time.get(t, (0, 0))
                break

        nt = t + 1
        res_nt = reservations.get(nt, set())

        # =============================================================
        # Branch 1: Avatar is Riding a Moving Platform at time t
        # =============================================================
        if riding_plat is not None:
            p_dr, p_dc = plat_vel

            # 1A. Passive Carriage (Wait Action on Platform)
            # Platform carries the avatar forward with its velocity
            nr = r + p_dr * stride
            nc = c + p_dc * stride
            in_bounds = (0 <= nr < H and 0 <= nc < W)
            not_static = ((nr, nc) not in static_blocked)
            not_hazard = ((nr, nc) not in res_nt)
            not_in_bodies = all((nr, nc) not in p.body_by_time.get(nt, set()) for p in platforms)

            box_ok = True
            if box_offset is not None:
                bnr, bnc = nr + box_offset[0], nc + box_offset[1]
                box_ok = (0 <= bnr < H and 0 <= bnc < W and
                          (bnr, bnc) not in static_blocked and
                          (bnr, bnc) not in res_nt)

            if in_bounds and not_static and not_hazard and not_in_bodies and box_ok:
                nxt_key = (nr, nc, nt)
                if nxt_key not in visited:
                    wait_act = find_safe_wait_action(r, c)
                    if wait_act is not None:
                        h_val = heuristic(nr, nc)
                        # Passive riding is productive transit towards goal
                        heapq.heappush(pq, (nt + h_val, nt, nr, nc, nt, actions + [wait_act]))

            # 1B. Active Walking Relative to Moving Platform
            for act, (dr, dc) in DIR_MAP.items():
                if act not in actions_pool:
                    continue
                nr = r + p_dr * stride + dr * stride
                nc = c + p_dc * stride + dc * stride

                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in static_blocked or (nr, nc) in res_nt:
                    continue
                if any((nr, nc) in p.body_by_time.get(nt, set()) for p in platforms):
                    continue

                if box_offset is not None:
                    bnr, bnc = nr + box_offset[0], nc + box_offset[1]
                    if not (0 <= bnr < H and 0 <= bnc < W) or (bnr, bnc) in static_blocked or (bnr, bnc) in res_nt:
                        continue

                nxt_key = (nr, nc, nt)
                if nxt_key not in visited:
                    h_val = heuristic(nr, nc)
                    heapq.heappush(pq, (nt + h_val, nt, nr, nc, nt, actions + [act]))

        # =============================================================
        # Branch 2: Avatar is on Static Ground (or Off Platform)
        # =============================================================
        else:
            # 2A. Cardinal Movement Transitions (Actions 1..4)
            for act, (dr, dc) in DIR_MAP.items():
                if act not in actions_pool:
                    continue
                nr, nc = r + dr * stride, c + dc * stride

                # Bounds & Static Obstacle Check
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in static_blocked:
                    continue

                # Cannot step inside platform solid body
                if any((nr, nc) in p.body_by_time.get(nt, set()) for p in platforms):
                    continue

                # Carried Box Bounds & Static Check
                if box_offset is not None:
                    bnr, bnc = nr + box_offset[0], nc + box_offset[1]
                    if not (0 <= bnr < H and 0 <= bnc < W) or (bnr, bnc) in static_blocked:
                        continue
                    if (bnr, bnc) in res_nt:
                        continue

                # Dynamic Vertex Collision Check
                if (nr, nc) in res_nt:
                    continue

                # Dynamic Swap / Edge Collision Check (head-on crossing)
                res_t = reservations.get(t, set())
                if (nr, nc) in res_t and (r, c) in res_nt:
                    continue
                if box_offset is not None:
                    br, bc = r + box_offset[0], c + box_offset[1]
                    if (bnr, bnc) in res_t and (br, bc) in res_nt:
                        continue

                nxt_key = (nr, nc, nt)
                if nxt_key not in visited:
                    h_val = heuristic(nr, nc)
                    heapq.heappush(pq, (nt + h_val, nt, nr, nc, nt, actions + [act]))

            # 2B. Wait Transition: Remain at (r, c) until t + 1 on Static Ground
            avatar_safe = ((r, c) not in res_nt and
                           all((r, c) not in p.body_by_time.get(nt, set()) for p in platforms))
            box_safe = True
            if box_offset is not None:
                br, bc = r + box_offset[0], c + box_offset[1]
                box_safe = ((br, bc) not in res_nt and
                            all((br, bc) not in p.body_by_time.get(nt, set()) for p in platforms))

            if avatar_safe and box_safe:
                nxt_key = (r, c, nt)
                if nxt_key not in visited:
                    wait_act = find_safe_wait_action(r, c)
                    if wait_act is not None:
                        h_val = heuristic(r, c)
                        # Slight wait penalty (+1.0) ensures agent prefers moving when path is clear
                        heapq.heappush(pq, (nt + 1.0 + h_val, nt, r, c, nt, actions + [wait_act]))

    return None
