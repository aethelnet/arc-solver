"""
=============================================================================
ARC-AGI CAUSAL META-TOOL SELECTOR: PLATFORMER DYNAMICS & BALLISTIC A*
=============================================================================
Provides domain-agnostic forward physical simulation and graph search for
environments with directional gravity, fall physics, and jumping mechanics:

1. Directional Gravity Field Modeling:
   - Constant or inverted gravity vectors g in {(1, 0), (-1, 0), (0, 1), (0, -1)}.
   - Support surface validation: standing on solid shelves/ground.
2. Forward Ballistic Simulators:
   - simulate_gravity_fall: Traces downward acceleration and terminal landing.
   - simulate_ballistic_jump: Traces parabolic jump arcs and safe landing detection.
3. platformer_a_star:
   - Macro-action A* search over platform states (walking, ledge drops, gap jumps,
     and ledge climbs).
=============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set
import collections
import heapq
import numpy as np


@dataclass
class GravityActuator:
    """
    Actuator that dynamically modifies directional gravity upon trigger or contact.
    """
    pos: Tuple[int, int]
    action_id: int = 5                  # 5 (Interact), 6 (Click), or 0 (step-on / portal)
    target_gravity: Tuple[int, int] = (-1, 0) # Target gravity vector
    requires_adjacent: bool = True      # If True, avatar must be at or adjacent (dist <= stride)
    action_data: Optional[Dict[str, Any]] = None
    cost: float = 1.0                   # Transition action cost


def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def is_supported(
    grid: np.ndarray,
    pos: Tuple[int, int],
    gravity: Tuple[int, int] = (1, 0),
    solid_colors: Optional[Set[int]] = None,
    stride: int = 1,
    carrier_platforms: Optional[List[Any]] = None,
    t: int = 0,
    hypothetical_solids: Optional[Set[Tuple[int, int]]] = None
) -> bool:
    """
    Checks if position (r, c) is resting on a solid support in direction of gravity,
    or on the support surface of a moving carrier platform at time t.
    Also supports hypothetical_solids (counterfactual bridges/terrain modifications).
    """
    r, c = pos
    gr, gc = gravity
    below_r = r + gr * stride
    below_c = c + gc * stride
    H, W = grid.shape

    # Check hypothetical solids (e.g. bridged pit/chasm)
    if hypothetical_solids and (below_r, below_c) in hypothetical_solids:
        return True

    # Check moving carrier platforms
    if carrier_platforms:
        for p in carrier_platforms:
            supp = getattr(p, 'support_by_time', {}).get(t, set())
            if (r, c) in supp:
                return True

    # Bottom edge of grid can act as boundary floor
    if not (0 <= below_r < H and 0 <= below_c < W):
        return True

    solids = solid_colors or set()
    val = int(grid[below_r, below_c])
    return val in solids


def simulate_gravity_fall(
    grid: np.ndarray,
    start_pos: Tuple[int, int],
    gravity: Tuple[int, int] = (1, 0),
    solid_colors: Optional[Set[int]] = None,
    hazard_colors: Optional[Set[int]] = None,
    suspect_hazard_colors: Optional[Set[int]] = None,
    bg_canvas: int = 0,
    stride: int = 1,
    max_fall: int = 35,
    hypothetical_solids: Optional[Set[Tuple[int, int]]] = None
) -> Tuple[Tuple[int, int], List[Tuple[int, int]], bool]:
    """
    Traces free-fall along gravity vector g until landing on a solid support
    or entering a lethal hazard / void / suspect dangerous surface.
    Returns:
        landing_pos: (r, c) final resting cell
        trajectory: list of coordinates traversed during fall
        hit_hazard: True if fall ended in death, void, or suspect hazard
    """
    grid_2d = np.asarray(grid)
    while grid_2d.ndim > 2:
        grid_2d = grid_2d[-1]
    H, W = grid_2d.shape

    solids = solid_colors or set()
    hazards = hazard_colors or set()
    suspects = suspect_hazard_colors or set()
    gr, gc = gravity

    curr_r, curr_c = start_pos
    trajectory = [(curr_r, curr_c)]

    for _ in range(max_fall):
        next_r = curr_r + gr * stride
        next_c = curr_c + gc * stride

        # Out of bounds fall = death into void
        if not (0 <= next_r < H and 0 <= next_c < W):
            return (curr_r, curr_c), trajectory, True

        cell_val = int(grid_2d[next_r, next_c])

        # Hit lethal hazard (spikes, lava) or suspect hazard
        if cell_val in hazards or cell_val in suspects:
            trajectory.append((next_r, next_c))
            return (next_r, next_c), trajectory, True

        # Hit solid ground/platform or counterfactual bridged block -> land on curr cell!
        if cell_val in solids or (hypothetical_solids and (next_r, next_c) in hypothetical_solids):
            # Safe-Set Inversion: check if the resting cell itself is in hazards or suspect!
            curr_val = int(grid_2d[curr_r, curr_c])
            if curr_val in hazards or curr_val in suspects:
                return (curr_r, curr_c), trajectory, True
            return (curr_r, curr_c), trajectory, False

        curr_r, curr_c = next_r, next_c
        trajectory.append((curr_r, curr_c))

    # Reached max fall without support -> hazardous drop
    return (curr_r, curr_c), trajectory, True


def simulate_ballistic_jump(
    grid: np.ndarray,
    start_pos: Tuple[int, int],
    lateral_dir: int,  # -1 (LEFT), 0 (VERTICAL), 1 (RIGHT)
    jump_height: int = 2,
    jump_reach: int = 2,
    gravity: Tuple[int, int] = (1, 0),
    solid_colors: Optional[Set[int]] = None,
    hazard_colors: Optional[Set[int]] = None,
    suspect_hazard_colors: Optional[Set[int]] = None,
    bg_canvas: int = 0,
    stride: int = 1,
    hypothetical_solids: Optional[Set[Tuple[int, int]]] = None
) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]], bool]:
    """
    Simulates a parabolic jump:
    1. Ascent: moves up against gravity while drifting laterally.
    2. Apex: reaches maximum height.
    3. Descent: free-fall under gravity until landing.
    Returns:
        landing_pos: (r, c) safe landing position or None
        trajectory: list of coordinates in jump trajectory
        hit_hazard: True if collided with hazard or suspect danger
    """
    grid_2d = np.asarray(grid)
    while grid_2d.ndim > 2:
        grid_2d = grid_2d[-1]
    H, W = grid_2d.shape

    solids = solid_colors or set()
    hazards = hazard_colors or set()
    suspects = suspect_hazard_colors or set()
    gr, gc = gravity
    up_r, up_c = -gr, -gc  # Counter-gravity impulse

    curr_r, curr_c = start_pos
    trajectory = [(curr_r, curr_c)]

    # 1. Ascent Phase
    for step in range(1, jump_height + 1):
        # Upward and lateral component
        lat_step = lateral_dir * stride if step <= jump_reach else 0
        nr = curr_r + up_r * stride
        nc = curr_c + lat_step

        if not (0 <= nr < H and 0 <= nc < W):
            return None, trajectory, True

        val = int(grid_2d[nr, nc])
        if val in hazards or val in suspects:
            trajectory.append((nr, nc))
            return None, trajectory, True

        if val in solids or (hypothetical_solids and (nr, nc) in hypothetical_solids):
            # Head-bump against ceiling: stop ascent early
            break

        curr_r, curr_c = nr, nc
        trajectory.append((curr_r, curr_c))

    # 2. Descent / Free-Fall Phase from Apex
    # Continue lateral drift for 1 step if lateral_dir was active
    if lateral_dir != 0:
        drift_r = curr_r
        drift_c = curr_c + lateral_dir * stride
        if 0 <= drift_r < H and 0 <= drift_c < W:
            val = int(grid_2d[drift_r, drift_c])
            if val not in solids and (not hypothetical_solids or (drift_r, drift_c) not in hypothetical_solids) and val not in hazards and val not in suspects:
                curr_r, curr_c = drift_r, drift_c
                trajectory.append((curr_r, curr_c))

    landing_pos, fall_traj, hit_hazard = simulate_gravity_fall(
        grid=grid_2d,
        start_pos=(curr_r, curr_c),
        gravity=gravity,
        solid_colors=solids,
        hazard_colors=hazards,
        suspect_hazard_colors=suspects,
        bg_canvas=bg_canvas,
        stride=stride,
        hypothetical_solids=hypothetical_solids
    )

    trajectory.extend(fall_traj[1:])

    if hit_hazard:
        return None, trajectory, True

    return landing_pos, trajectory, False


def is_kinematically_irreversible_drop(
    grid: np.ndarray,
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int],
    gravity: Tuple[int, int] = (1, 0),
    jump_height: int = 2,
    jump_reach: int = 2,
    solid_colors: Optional[Set[int]] = None,
    hazard_colors: Optional[Set[int]] = None,
    suspect_hazard_colors: Optional[Set[int]] = None,
    bg_canvas: int = 0,
    stride: int = 1,
    goal_candidates: Optional[Set[Tuple[int, int]]] = None,
    interactable_cells: Optional[Set[Tuple[int, int]]] = None,
    carrier_platforms: Optional[List[Any]] = None,
    max_escape_expansions: int = 100,
    hypothetical_solids: Optional[Set[Tuple[int, int]]] = None
) -> Tuple[bool, bool, str, Optional[Tuple[int, int]]]:
    """
    Counterfactual Kinematic Inversion / Safe-Set Filter:
    Evaluates whether moving from from_pos to to_pos induces an irreversible fall
    into an absorbing deadlock state (pit trap, unjumpable drop, or lethal void).

    Returns:
        is_irreversible: bool (True if drop cannot be escaped or has no forward affordance)
        is_fatal: bool (True if drop ends in hazard or out of bounds void)
        reason: str
        landing_pos: Optional[Tuple[int, int]] (final resting coordinates after fall)
    """
    grid_2d = np.asarray(grid)
    while grid_2d.ndim > 2:
        grid_2d = grid_2d[-1]
    H, W = grid_2d.shape

    solids = set(solid_colors) if solid_colors is not None else set()
    hazards = set(hazard_colors) if hazard_colors is not None else set()
    suspects = set(suspect_hazard_colors) if suspect_hazard_colors is not None else set()
    goals = set(goal_candidates) if goal_candidates is not None else set()
    interactables = set(interactable_cells) if interactable_cells is not None else set()
    gr, gc = gravity

    tr, tc = to_pos
    if not (0 <= tr < H and 0 <= tc < W):
        return True, True, f"Out of bounds destination {to_pos}", None

    val = int(grid_2d[tr, tc])
    if val in solids or (hypothetical_solids and (tr, tc) in hypothetical_solids):
        return False, False, "Solid tile (motion blocked, not a drop)", None

    if val in hazards or val in suspects:
        return True, True, f"Direct move into hazard/suspect tile at {to_pos}", to_pos

    # If to_pos is supported, it is a supported non-falling move
    if is_supported(grid_2d, to_pos, gravity, solids, stride, carrier_platforms, hypothetical_solids=hypothetical_solids):
        return False, False, "Supported move (no drop)", to_pos

    # Avatar is NOT supported at to_pos -> free fall under gravity
    landing_pos, fall_traj, hit_haz = simulate_gravity_fall(
        grid=grid_2d,
        start_pos=to_pos,
        gravity=gravity,
        solid_colors=solids,
        hazard_colors=hazards,
        suspect_hazard_colors=suspects,
        bg_canvas=bg_canvas,
        stride=stride,
        hypothetical_solids=hypothetical_solids
    )

    if hit_haz:
        return True, True, f"Fatal drop along trajectory to {landing_pos}", landing_pos

    if landing_pos == from_pos:
        return False, False, "No net displacement", landing_pos

    # Drop distance along gravity direction
    drop_dist = (landing_pos[0] - from_pos[0]) * gr + (landing_pos[1] - from_pos[1]) * gc
    if drop_dist <= 0:
        return False, False, "Safe non-drop move", landing_pos

    # If landing or any point along trajectory is a goal or interactable, drop is intentional & valid
    if landing_pos in goals or any(p in goals for p in fall_traj):
        return False, False, "Drop directly reaches goal target", landing_pos

    if landing_pos in interactables or any(p in interactables for p in fall_traj):
        return False, False, "Drop directly reaches interactable entity", landing_pos

    # Evaluate escape capability from landing_pos
    from_elevation = from_pos[0] * gr + from_pos[1] * gc

    # Fast BFS to test if landing_pos is inside an absorbing pit trap
    q = collections.deque([landing_pos])
    visited: Set[Tuple[int, int]] = {landing_pos}
    can_escape = False
    expansions = 0

    while q and expansions < max_escape_expansions:
        expansions += 1
        curr = q.popleft()

        curr_elevation = curr[0] * gr + curr[1] * gc
        # 1. Avatar reached elevation equal to or higher than origin (drop is reversible!)
        if curr_elevation <= from_elevation:
            can_escape = True
            break

        # 2. Reached goal or interactable
        if curr in goals or curr in interactables:
            can_escape = True
            break

        # 3. If reachable platform space exceeds large exploration threshold, it is an open lower floor, not a pit trap
        if len(visited) >= 30:
            can_escape = True
            break

        curr_supp = is_supported(grid_2d, curr, gravity, solids, stride, carrier_platforms, hypothetical_solids=hypothetical_solids)

        # Lateral moves from curr
        for dc_lat in [-1, 1]:
            nr_lat = curr[0]
            nc_lat = curr[1] + dc_lat * stride
            if not (0 <= nr_lat < H and 0 <= nc_lat < W):
                continue
            lat_val = int(grid_2d[nr_lat, nc_lat])
            if lat_val in solids or (hypothetical_solids and (nr_lat, nc_lat) in hypothetical_solids) or lat_val in hazards or lat_val in suspects:
                continue

            # Supported lateral step
            if is_supported(grid_2d, (nr_lat, nc_lat), gravity, solids, stride, carrier_platforms, hypothetical_solids=hypothetical_solids):
                if (nr_lat, nc_lat) not in visited:
                    visited.add((nr_lat, nc_lat))
                    q.append((nr_lat, nc_lat))
            else:
                # Secondary ledge drop
                sec_land, _, sec_haz = simulate_gravity_fall(
                    grid=grid_2d,
                    start_pos=(nr_lat, nc_lat),
                    gravity=gravity,
                    solid_colors=solids,
                    hazard_colors=hazards,
                    suspect_hazard_colors=suspects,
                    bg_canvas=bg_canvas,
                    stride=stride,
                    hypothetical_solids=hypothetical_solids
                )
                if not sec_haz and sec_land not in visited:
                    visited.add(sec_land)
                    q.append(sec_land)

        # Ballistic jump arcs from curr (if curr is supported)
        if curr_supp:
            for lat_d in [-1, 0, 1]:
                j_land, _, j_haz = simulate_ballistic_jump(
                    grid=grid_2d,
                    start_pos=curr,
                    lateral_dir=lat_d,
                    jump_height=jump_height,
                    jump_reach=jump_reach,
                    gravity=gravity,
                    solid_colors=solids,
                    hazard_colors=hazards,
                    suspect_hazard_colors=suspects,
                    bg_canvas=bg_canvas,
                    stride=stride,
                    hypothetical_solids=hypothetical_solids
                )
                if j_land is not None and not j_haz and j_land not in visited:
                    visited.add(j_land)
                    q.append(j_land)

    if can_escape:
        return False, False, f"Reversible drop (escape possible, drop_dist={drop_dist})", landing_pos
    else:
        return True, False, (
            f"Kinematic Irreversibility Veto: Drop into inescapable pit trap at {landing_pos} "
            f"(depth={drop_dist} > jump_height={jump_height}, reachable_states={len(visited)})"
        ), landing_pos


def is_kinematically_irreversible_jump(
    grid: np.ndarray,
    from_pos: Tuple[int, int],
    lateral_dir: int,
    jump_height: int = 2,
    jump_reach: int = 2,
    gravity: Tuple[int, int] = (1, 0),
    solid_colors: Optional[Set[int]] = None,
    hazard_colors: Optional[Set[int]] = None,
    suspect_hazard_colors: Optional[Set[int]] = None,
    bg_canvas: int = 0,
    stride: int = 1,
    goal_candidates: Optional[Set[Tuple[int, int]]] = None,
    interactable_cells: Optional[Set[Tuple[int, int]]] = None,
    carrier_platforms: Optional[List[Any]] = None,
    max_escape_expansions: int = 100,
    hypothetical_solids: Optional[Set[Tuple[int, int]]] = None
) -> Tuple[bool, bool, str, Optional[Tuple[int, int]]]:
    """
    Counterfactual Kinematic Inversion for Ballistic Jumps:
    Evaluates whether a jump in lateral_dir lands in a lethal hazard or an
    inescapable pit trap.
    """
    grid_2d = np.asarray(grid)
    while grid_2d.ndim > 2:
        grid_2d = grid_2d[-1]

    solids = set(solid_colors) if solid_colors is not None else set()
    hazards = set(hazard_colors) if hazard_colors is not None else set()
    suspects = set(suspect_hazard_colors) if suspect_hazard_colors is not None else set()

    landing_pos, traj, hit_haz = simulate_ballistic_jump(
        grid=grid_2d,
        start_pos=from_pos,
        lateral_dir=lateral_dir,
        jump_height=jump_height,
        jump_reach=jump_reach,
        gravity=gravity,
        solid_colors=solids,
        hazard_colors=hazards,
        suspect_hazard_colors=suspects,
        bg_canvas=bg_canvas,
        stride=stride,
        hypothetical_solids=hypothetical_solids
    )
    if hit_haz:
        return True, True, f"Fatal jump arc into hazard/void along trajectory to {landing_pos}", landing_pos
    if landing_pos is None:
        return True, True, "Jump trajectory failed to find safe landing", None
    if landing_pos == from_pos:
        return False, False, "Safe in-place jump", landing_pos

    return is_kinematically_irreversible_drop(
        grid=grid_2d,
        from_pos=from_pos,
        to_pos=landing_pos,
        gravity=gravity,
        jump_height=jump_height,
        jump_reach=jump_reach,
        solid_colors=solids,
        hazard_colors=hazards,
        suspect_hazard_colors=suspects,
        bg_canvas=bg_canvas,
        stride=stride,
        goal_candidates=goal_candidates,
        interactable_cells=interactable_cells,
        carrier_platforms=carrier_platforms,
        max_escape_expansions=max_escape_expansions,
        hypothetical_solids=hypothetical_solids
    )


def platformer_a_star(
    grid: np.ndarray,
    start_pos: Tuple[int, int],
    goal_candidates: Set[Tuple[int, int]],
    gravity: Tuple[int, int] = (1, 0),
    jump_height: int = 2,
    jump_reach: int = 2,
    solid_colors: Optional[Set[int]] = None,
    hazard_colors: Optional[Set[int]] = None,
    suspect_hazard_colors: Optional[Set[int]] = None,
    bg_canvas: int = 0,
    available_actions: Optional[List[int]] = None,
    stride: int = 1,
    max_expansions: int = 800,
    gravity_actuators: Optional[List[GravityActuator]] = None,
    carrier_platforms: Optional[List[Any]] = None,
    reservations: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
    max_time: int = 50,
    hypothetical_solids: Optional[Set[Tuple[int, int]]] = None
) -> Optional[List[int]]:
    """
    Ballistic Platformer A* Search over Hybrid 4D Spacetime State Space (r, c, t, g):
    Plans sequences of walking (Actions 3, 4), safe ledge drops, parabolic
    jumps, dynamic actuator-driven gravity inversions (e.g. switch flips
    Action 5 or remote click Action 6), and ground waiting actions across multi-tier
    platforms and ceilings.
    Evades dynamic moving hazards via 4D space-time collision pruning against reservations R(t).
    Enforces Safe-Set Inversion: avoids suspect unverified death hazards.
    """
    grid_2d = np.asarray(grid)
    while grid_2d.ndim > 2:
        grid_2d = grid_2d[-1]
    H, W = grid_2d.shape

    solids = solid_colors or set()
    hazards = hazard_colors or set()
    suspects = suspect_hazard_colors or set()
    actions_pool = available_actions or [1, 2, 3, 4, 5]

    def heuristic(pos: Tuple[int, int]) -> float:
        if not goal_candidates:
            return 0.0
        return min(manhattan(pos, g) for g in goal_candidates) / float(stride)

    def is_cell_dynamic_hazard(cell: Tuple[int, int], t_step: int) -> bool:
        if reservations is None:
            return False
        return cell in reservations.get(t_step, set())

    curr_g = gravity

    # Initial fall if start_pos is mid-air
    actual_start = start_pos
    initial_actions: List[int] = []
    start_t = 0
    if not is_supported(grid_2d, start_pos, curr_g, solids, stride, carrier_platforms, hypothetical_solids=hypothetical_solids):
        land_p, fall_traj, hit_haz = simulate_gravity_fall(
            grid_2d, start_pos, curr_g, solids, hazards, suspects, bg_canvas, stride, hypothetical_solids=hypothetical_solids
        )
        if hit_haz:
            return None
        if reservations is not None:
            for step_k, p in enumerate(fall_traj):
                if is_cell_dynamic_hazard(p, step_k):
                    return None
            start_t = len(fall_traj) - 1
        actual_start = land_p

    # Priority queue: (f_score, g_score, (r, c, t, g), actions)
    start_state = (actual_start[0], actual_start[1], start_t if reservations is not None else 0, curr_g)
    h0 = heuristic(actual_start)
    pq: List[Tuple[float, float, Tuple[int, int, int, Tuple[int, int]], List[int]]] = [
        (h0, 0.0, start_state, initial_actions)
    ]
    visited: Dict[Tuple[int, int, int, Tuple[int, int]], float] = {start_state: 0.0}
    expansions = 0

    while pq and expansions < max_expansions:
        expansions += 1
        f, g, (r, c, t, state_g), actions = heapq.heappop(pq)

        if (r, c) in goal_candidates:
            return actions

        state_key = (r, c, t if reservations is not None else 0, state_g)
        if g > visited.get(state_key, float("inf")):
            continue

        if reservations is not None and t >= max_time:
            continue

        curr_is_supp = is_supported(grid_2d, (r, c), state_g, solids, stride, carrier_platforms, hypothetical_solids=hypothetical_solids)

        # -------------------------------------------------------------
        # 0. Ground Wait Action (Wall Bump / Idle) when Dynamic Hazards present
        # -------------------------------------------------------------
        if reservations is not None and curr_is_supp:
            nt = t + 1
            if not is_cell_dynamic_hazard((r, c), nt):
                wait_act = None
                has_act_5_switch = False
                if gravity_actuators:
                    for ga in gravity_actuators:
                        if ga.action_id == 5 and ga.pos == (r, c):
                            has_act_5_switch = True
                            break
                if 5 in actions_pool and not has_act_5_switch:
                    wait_act = 5
                else:
                    for b_act, (dr_b, dc_b) in [(3, (0, -1)), (4, (0, 1)), (1, (-1, 0)), (2, (1, 0))]:
                        if b_act in actions_pool:
                            b_nr, b_nc = r + dr_b * stride, c + dc_b * stride
                            if (b_nr, b_nc) in solids or (hypothetical_solids and (b_nr, b_nc) in hypothetical_solids) or not (0 <= b_nr < H and 0 <= b_nc < W):
                                wait_act = b_act
                                break

                if wait_act is not None:
                    wait_state = (r, c, nt, state_g)
                    wait_actions = actions + [wait_act]
                    step_cost = 1.0
                    nxt_g = g + step_cost
                    if nxt_g < visited.get(wait_state, float("inf")):
                        visited[wait_state] = nxt_g
                        heapq.heappush(pq, (nxt_g + heuristic((r, c)), nxt_g, wait_state, wait_actions))

        # -------------------------------------------------------------
        # 1. Lateral Walk / Ledge Drop / Step-On Gravity Portals
        # -------------------------------------------------------------
        for act, dc in [(3, -1), (4, 1)]:
            if act not in actions_pool:
                continue
            nr, nc = r, c + dc * stride
            nt = t + 1
            if not (0 <= nr < H and 0 <= nc < W):
                continue

            val = int(grid_2d[nr, nc])
            if val in solids or (hypothetical_solids and (nr, nc) in hypothetical_solids) or val in hazards or (suspects and val in suspects and (nr, nc) not in goal_candidates):
                continue

            if reservations is not None:
                if is_cell_dynamic_hazard((nr, nc), nt):
                    continue
                if is_cell_dynamic_hazard((nr, nc), t) and is_cell_dynamic_hazard((r, c), nt):
                    continue

            # Check if stepping onto a step-on gravity inverter (action_id == 0)
            step_on_act = None
            if gravity_actuators:
                for ga in gravity_actuators:
                    if ga.action_id == 0 and ga.pos == (nr, nc):
                        step_on_act = ga
                        break

            if step_on_act is not None:
                inv_g = step_on_act.target_gravity
                fall_time = 0
                if is_supported(grid_2d, (nr, nc), inv_g, solids, stride, carrier_platforms, hypothetical_solids=hypothetical_solids):
                    nxt_pos = (nr, nc)
                    hit_haz = False
                else:
                    nxt_pos, fall_traj, hit_haz = simulate_gravity_fall(
                        grid_2d, (nr, nc), inv_g, solids, hazards, suspects, bg_canvas, stride, hypothetical_solids=hypothetical_solids
                    )
                    fall_time = len(fall_traj) - 1
                    if reservations is not None and not hit_haz:
                        for fk, fp in enumerate(fall_traj):
                            if is_cell_dynamic_hazard(fp, nt + fk):
                                hit_haz = True
                                break

                if not hit_haz:
                    nxt_t = (nt + fall_time) if reservations is not None else 0
                    nxt_state = (nxt_pos[0], nxt_pos[1], nxt_t, inv_g)
                    nxt_actions = actions + [act]
                    step_cost = 1.0 + (manhattan((nr, nc), nxt_pos) * 0.2 if nxt_pos != (nr, nc) else 0.0)
                    nxt_g = g + step_cost
                    if nxt_g < visited.get(nxt_state, float("inf")):
                        visited[nxt_state] = nxt_g
                        heapq.heappush(pq, (nxt_g + heuristic(nxt_pos), nxt_g, nxt_state, nxt_actions))
                continue

            # Standard walk / ledge drop under state_g
            if is_supported(grid_2d, (nr, nc), state_g, solids, stride, carrier_platforms, hypothetical_solids=hypothetical_solids):
                nxt_pos = (nr, nc)
                nxt_t = nt if reservations is not None else 0
                nxt_state = (nr, nc, nxt_t, state_g)
                nxt_actions = actions + [act]
                step_cost = 1.0
                nxt_g = g + step_cost
                if nxt_g < visited.get(nxt_state, float("inf")):
                    visited[nxt_state] = nxt_g
                    heapq.heappush(pq, (nxt_g + heuristic(nxt_pos), nxt_g, nxt_state, nxt_actions))
            else:
                # Ledge Drop: Step off platform and fall under state_g
                landing, fall_traj, hit_haz = simulate_gravity_fall(
                    grid_2d, (nr, nc), state_g, solids, hazards, suspects, bg_canvas, stride, hypothetical_solids=hypothetical_solids
                )
                fall_time = len(fall_traj) - 1
                if reservations is not None and not hit_haz:
                    for fk, fp in enumerate(fall_traj):
                        if is_cell_dynamic_hazard(fp, nt + fk):
                            hit_haz = True
                            break

                if not hit_haz and landing != (r, c):
                    # Safe-Set Inversion: prune ledge drops into irreversible pit traps if goal is outside
                    if goal_candidates and landing not in goal_candidates:
                        is_irrev, _, _, _ = is_kinematically_irreversible_drop(
                            grid=grid_2d,
                            from_pos=(r, c),
                            to_pos=(nr, nc),
                            gravity=state_g,
                            jump_height=jump_height,
                            jump_reach=jump_reach,
                            solid_colors=solids,
                            hazard_colors=hazards,
                            suspect_hazard_colors=suspects,
                            bg_canvas=bg_canvas,
                            stride=stride,
                            goal_candidates=goal_candidates,
                            carrier_platforms=carrier_platforms,
                            hypothetical_solids=hypothetical_solids
                        )
                        if is_irrev:
                            continue

                    nxt_t = (nt + fall_time) if reservations is not None else 0
                    nxt_state = (landing[0], landing[1], nxt_t, state_g)
                    nxt_actions = actions + [act]
                    step_cost = 1.5 + manhattan((nr, nc), landing) * 0.2
                    nxt_g = g + step_cost
                    if nxt_g < visited.get(nxt_state, float("inf")):
                        visited[nxt_state] = nxt_g
                        heapq.heappush(pq, (nxt_g + heuristic(landing), nxt_g, nxt_state, nxt_actions))

        # -------------------------------------------------------------
        # 2. Ballistic Jumps
        # -------------------------------------------------------------
        jump_act_candidates = []
        if 1 in actions_pool or 2 in actions_pool:
            if state_g[0] > 0 and 1 in actions_pool:
                jump_act_candidates.append(1)
            elif state_g[0] < 0 and 2 in actions_pool:
                jump_act_candidates.append(2)
            elif 1 in actions_pool:
                jump_act_candidates.append(1)
        elif 5 in actions_pool:
            jump_act_candidates.append(5)

        for j_act in jump_act_candidates:
            jump_options = [
                (0, [j_act]),                                                 # Vertical Jump
                (-1, [j_act, 3] if 3 in actions_pool else [j_act]),          # Jump Left
                (1, [j_act, 4] if 4 in actions_pool else [j_act])            # Jump Right
            ]

            for lat_dir, j_acts in jump_options:
                landing, traj, hit_haz = simulate_ballistic_jump(
                    grid=grid_2d,
                    start_pos=(r, c),
                    lateral_dir=lat_dir,
                    jump_height=jump_height,
                    jump_reach=jump_reach,
                    gravity=state_g,
                    solid_colors=solids,
                    hazard_colors=hazards,
                    suspect_hazard_colors=suspects,
                    bg_canvas=bg_canvas,
                    stride=stride,
                    hypothetical_solids=hypothetical_solids
                )

                dt = max(1, len(j_acts))
                if reservations is not None and not hit_haz and landing is not None:
                    for tk, tp in enumerate(traj):
                        check_t = min(t + tk, t + dt)
                        if is_cell_dynamic_hazard(tp, check_t):
                            hit_haz = True
                            break
                    if not hit_haz and is_cell_dynamic_hazard(landing, t + dt):
                        hit_haz = True

                if not hit_haz and landing is not None and landing != (r, c):
                    nxt_t = (t + dt) if reservations is not None else 0
                    nxt_state = (landing[0], landing[1], nxt_t, state_g)
                    nxt_actions = actions + j_acts
                    step_cost = 2.0 + len(j_acts)
                    nxt_g = g + step_cost
                    if nxt_g < visited.get(nxt_state, float("inf")):
                        visited[nxt_state] = nxt_g
                        heapq.heappush(pq, (nxt_g + heuristic(landing), nxt_g, nxt_state, nxt_actions))

        # -------------------------------------------------------------
        # 3. Dynamic Actuator-Driven Gravity Inversion (Action 5 or 6)
        # -------------------------------------------------------------
        if gravity_actuators:
            for ga in gravity_actuators:
                if ga.action_id == 0:
                    continue  # Handled above via step-on
                if ga.action_id not in actions_pool:
                    continue

                can_trigger = False
                if not ga.requires_adjacent:
                    can_trigger = True  # Remote click (Action 6)
                else:
                    dist = manhattan((r, c), ga.pos)
                    if dist <= stride:
                        can_trigger = True

                if can_trigger:
                    nxt_g = ga.target_gravity
                    if nxt_g == state_g:
                        continue  # Already in this gravity orientation

                    act_dt = 1
                    fall_time = 0
                    if is_supported(grid_2d, (r, c), nxt_g, solids, stride, carrier_platforms, hypothetical_solids=hypothetical_solids):
                        nxt_pos = (r, c)
                        hit_haz = False
                    else:
                        nxt_pos, fall_traj, hit_haz = simulate_gravity_fall(
                            grid_2d, (r, c), nxt_g, solids, hazards, suspects, bg_canvas, stride, hypothetical_solids=hypothetical_solids
                        )
                        fall_time = len(fall_traj) - 1
                        if reservations is not None and not hit_haz:
                            for fk, fp in enumerate(fall_traj):
                                if is_cell_dynamic_hazard(fp, t + act_dt + fk):
                                    hit_haz = True
                                    break

                    if not hit_haz:
                        nxt_t = (t + act_dt + fall_time) if reservations is not None else 0
                        nxt_state = (nxt_pos[0], nxt_pos[1], nxt_t, nxt_g)
                        nxt_actions = actions + [ga.action_id]
                        fall_dist = manhattan((r, c), nxt_pos)
                        act_cost = ga.cost + (fall_dist * 0.2 if nxt_pos != (r, c) else 0.0)
                        nxt_g_cost = g + act_cost
                        if nxt_g_cost < visited.get(nxt_state, float("inf")):
                            visited[nxt_state] = nxt_g_cost
                            heapq.heappush(pq, (nxt_g_cost + heuristic(nxt_pos), nxt_g_cost, nxt_state, nxt_actions))

    return None
