"""
=============================================================================
ARC-AGI CAUSAL META-TOOL SELECTOR: CAUSAL PRECONDITION INDUCER (PHASE 220)
=============================================================================
Enables Multi-Step Causal Reasoning and Teleological Subgoal DAGs (Hebel 1):
1. Precondition Chaining:
   - When path to goal/frontier is blocked by an impassable obstacle (gap,
     chasm, locked barrier, pit trap), induces the causal prerequisite.
2. Dynamic Terrain Modification:
   - Chasm Bridging: Identifies movable blocks within reachable component,
     computes push trajectory to chasm edge, and models counterfactual
     gravity fall where block fills pit cell (Δh -> Δh - 1), creating a bridge.
3. Causal Barrier Resolution:
   - Identifies blocking barrier color and chains key collection or switch
     toggle as formal DAG prerequisite.
=============================================================================
"""

import collections
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np

from .options_framework import Subgoal, SubgoalType, SubgoalStatus
from .platformer_planner import (
    is_supported,
    simulate_gravity_fall,
    simulate_ballistic_jump,
    platformer_a_star,
    is_kinematically_irreversible_drop
)


class PushableBlockCandidate:
    """
    Representation of a movable entity candidate for causal bridge building.
    """
    def __init__(
        self,
        color: int,
        coords: List[Tuple[int, int]],
        centroid: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        size: Tuple[int, int]
    ):
        self.color = color
        self.coords = coords
        self.centroid = centroid
        self.bbox = bbox
        self.size = size

    def __repr__(self) -> str:
        return f"PushableBlock(color={self.color}, center={self.centroid}, size={self.size})"


class CausalPreconditionInducer:
    """
    Synthesizes teleological precondition DAGs when direct path to primary goal is blocked.
    """
    def __init__(self):
        self.subgoal_counter = 0

    def compute_reachable_component(
        self,
        grid: np.ndarray,
        avatar_pos: Tuple[int, int],
        gravity: Tuple[int, int] = (1, 0),
        solid_colors: Optional[Set[int]] = None,
        hazard_colors: Optional[Set[int]] = None,
        suspect_hazard_colors: Optional[Set[int]] = None,
        jump_height: int = 2,
        jump_reach: int = 2,
        stride: int = 1,
        max_expansions: int = 500
    ) -> Set[Tuple[int, int]]:
        """
        Computes the forward reachable state set under platformer kinematics.
        """
        grid_2d = np.asarray(grid)
        while grid_2d.ndim > 2:
            grid_2d = grid_2d[-1]
        H, W = grid_2d.shape
        solids = solid_colors or set()
        hazards = hazard_colors or set()
        suspects = suspect_hazard_colors or set()

        # Start from supported landing position
        if not is_supported(grid_2d, avatar_pos, gravity, solids, stride):
            land_p, _, hit_haz = simulate_gravity_fall(
                grid_2d, avatar_pos, gravity, solids, hazard_colors=hazards, suspect_hazard_colors=suspects, stride=stride
            )
            if hit_haz:
                return {avatar_pos}
            start = land_p
        else:
            start = avatar_pos

        visited: Set[Tuple[int, int]] = {start}
        queue: List[Tuple[int, int]] = [start]

        while queue and len(visited) < max_expansions:
            curr = queue.pop(0)
            r, c = curr

            # 1. Lateral Walk / Ledge Drop
            for dc in (-1, 1):
                nc = c + dc * stride
                nr = r
                if 0 <= nr < H and 0 <= nc < W:
                    val = int(grid_2d[nr, nc])
                    if val not in solids and val not in hazards and val not in suspects:
                        if is_supported(grid_2d, (nr, nc), gravity, solids, stride):
                            if (nr, nc) not in visited:
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                        else:
                            # Ledge drop
                            is_irrev, is_fatal, _, land_p = is_kinematically_irreversible_drop(
                                grid=grid_2d,
                                from_pos=curr,
                                to_pos=(nr, nc),
                                gravity=gravity,
                                jump_height=jump_height,
                                jump_reach=jump_reach,
                                solid_colors=solids,
                                hazard_colors=hazards,
                                suspect_hazard_colors=suspects,
                                stride=stride
                            )
                            if not is_fatal and not is_irrev and land_p is not None:
                                if land_p not in visited:
                                    visited.add(land_p)
                                    queue.append(land_p)

            # 2. Jumps
            if is_supported(grid_2d, curr, gravity, solids, stride):
                for lat_d in [-1, 0, 1]:
                    land_p, _, hit_haz = simulate_ballistic_jump(
                        grid=grid_2d,
                        start_pos=curr,
                        lateral_dir=lat_d,
                        jump_height=jump_height,
                        jump_reach=jump_reach,
                        gravity=gravity,
                        solid_colors=solids,
                        hazard_colors=hazards,
                        suspect_hazard_colors=suspects,
                        stride=stride
                    )
                    if not hit_haz and land_p is not None and land_p not in visited:
                        visited.add(land_p)
                        queue.append(land_p)

        return visited

    def extract_pushable_blocks(
        self,
        grid: np.ndarray,
        reachable_cells: Set[Tuple[int, int]],
        avatar_pos: Optional[Tuple[int, int]] = None,
        avatar_color: Optional[int] = None,
        target_pos: Optional[Tuple[int, int]] = None,
        bg_canvas: int = 0,
        floor_colors: Optional[Set[int]] = None,
        solid_colors: Optional[Set[int]] = None
    ) -> List[PushableBlockCandidate]:
        """
        Extracts compact, movable components that can be manipulated by the avatar.
        Guarantees that neither the avatar nor the primary target are extracted as pushable blocks.
        """
        grid_2d = np.asarray(grid)
        while grid_2d.ndim > 2:
            grid_2d = grid_2d[-1]
        H, W = grid_2d.shape
        floors = floor_colors or set()
        solids = solid_colors or set()

        visited = np.zeros((H, W), dtype=bool)
        candidates: List[PushableBlockCandidate] = []

        for r in range(H):
            for c in range(W):
                val = int(grid_2d[r, c])
                if visited[r, c] or val == bg_canvas or val in floors or val in solids:
                    continue
                if avatar_color is not None and val == avatar_color:
                    continue
                if avatar_pos is not None and (r, c) == avatar_pos:
                    continue
                if target_pos is not None and (r, c) == target_pos:
                    continue

                # BFS component extraction
                comp = []
                q = [(r, c)]
                visited[r, c] = True
                while q:
                    cr, cc = q.pop(0)
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and int(grid_2d[nr, nc]) == val:
                            visited[nr, nc] = True
                            q.append((nr, nc))

                # Do not treat avatar or target component as pushable block
                if avatar_pos is not None and avatar_pos in comp:
                    continue
                if target_pos is not None and target_pos in comp:
                    continue

                coords = np.array(comp)
                rmin, cmin = coords.min(axis=0)
                rmax, cmax = coords.max(axis=0)
                ch = int(rmax - rmin + 1)
                cw = int(cmax - cmin + 1)

                # Pushable blocks are compact (<= 4x4, size <= 16 cells)
                if 1 <= ch <= 4 and 1 <= cw <= 4 and len(comp) <= 16:
                    centroid = (int((rmin + rmax) // 2), int((cmin + cmax) // 2))
                    # Check if block is accessible (adjacent to reachable component)
                    is_accessible = any(
                        (cr + dr, cc + dc) in reachable_cells
                        for cr, cc in comp
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    )
                    if is_accessible:
                        candidates.append(PushableBlockCandidate(
                            color=val,
                            coords=comp,
                            centroid=centroid,
                            bbox=(int(rmin), int(cmin), int(rmax), int(cmax)),
                            size=(ch, cw)
                        ))

        return candidates

    def induce_chasm_bridge_plan(
        self,
        grid: np.ndarray,
        avatar_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        avatar_color: Optional[int] = None,
        gravity: Tuple[int, int] = (1, 0),
        solid_colors: Optional[Set[int]] = None,
        hazard_colors: Optional[Set[int]] = None,
        suspect_hazard_colors: Optional[Set[int]] = None,
        bg_canvas: int = 0,
        floor_colors: Optional[Set[int]] = None,
        jump_height: int = 2,
        jump_reach: int = 2,
        available_actions: Optional[List[int]] = None,
        stride: int = 1
    ) -> Optional[List[Subgoal]]:
        """
        Synthesizes a teleological DAG for bridging a chasm / pit:
        1. Identifies reachable blocks.
        2. Searches for drop gaps where falling block creates a walkable bridge.
        3. Validates that counterfactual bridge connects avatar to target.
        """
        grid_2d = np.asarray(grid)
        while grid_2d.ndim > 2:
            grid_2d = grid_2d[-1]
        H, W = grid_2d.shape
        solids = solid_colors or set()

        # Check if direct path already exists
        direct_path = platformer_a_star(
            grid=grid_2d,
            start_pos=avatar_pos,
            goal_candidates={target_pos},
            gravity=gravity,
            jump_height=jump_height,
            jump_reach=jump_reach,
            solid_colors=solids,
            hazard_colors=hazard_colors,
            suspect_hazard_colors=suspect_hazard_colors,
            bg_canvas=bg_canvas,
            available_actions=available_actions,
            stride=stride
        )
        if direct_path is not None:
            return None # Target already reachable directly!

        reachable = self.compute_reachable_component(
            grid=grid_2d,
            avatar_pos=avatar_pos,
            gravity=gravity,
            solid_colors=solids,
            hazard_colors=hazard_colors,
            suspect_hazard_colors=suspect_hazard_colors,
            jump_height=jump_height,
            jump_reach=jump_reach,
            stride=stride
        )
        if target_pos in reachable:
            return None

        blocks = self.extract_pushable_blocks(
            grid=grid_2d,
            reachable_cells=reachable,
            avatar_pos=avatar_pos,
            avatar_color=avatar_color,
            target_pos=target_pos,
            bg_canvas=bg_canvas,
            floor_colors=floor_colors,
            solid_colors=solids
        )
        if not blocks:
            return None

        # Search candidate ledge drop columns between reachable component and target
        # Ledge drop cells are empty air cells adjacent to a supported floor in reachable set
        candidate_drop_edges = []
        for r, c in reachable:
            if is_supported(grid_2d, (r, c), gravity, solids, stride):
                for dc in (-1, 1):
                    nc = c + dc * stride
                    nr = r
                    if 0 <= nc < W:
                        # If cell is empty and unsupported, it is a drop ledge
                        if int(grid_2d[nr, nc]) not in solids and not is_supported(grid_2d, (nr, nc), gravity, solids, stride):
                            # Test where block would land if dropped from here (blocks can land on hazard cells to cover them)
                            land_p, _, _ = simulate_gravity_fall(
                                grid_2d, (nr, nc), gravity, solids, hazard_colors=None, suspect_hazard_colors=None, bg_canvas=bg_canvas, stride=stride
                            )
                            if land_p is not None:
                                candidate_drop_edges.append(((r, c), (nr, nc), land_p, dc))

        if not candidate_drop_edges:
            return None

        # Sort candidate drop edges by proximity to target
        candidate_drop_edges.sort(key=lambda item: abs(item[2][0] - target_pos[0]) + abs(item[2][1] - target_pos[1]))

        # 1. First test 1-block solutions
        for block in blocks:
            for ledge_p, drop_p, land_p, push_dir in candidate_drop_edges:
                hypothetical_solids = {land_p}

                # Also test if bridge allows standing on top of land_p
                test_path = platformer_a_star(
                    grid=grid_2d,
                    start_pos=avatar_pos,
                    goal_candidates={target_pos},
                    gravity=gravity,
                    jump_height=jump_height,
                    jump_reach=jump_reach,
                    solid_colors=solids,
                    hazard_colors=hazard_colors,
                    suspect_hazard_colors=suspect_hazard_colors,
                    bg_canvas=bg_canvas,
                    available_actions=available_actions,
                    stride=stride,
                    hypothetical_solids=hypothetical_solids
                )

                if test_path is not None:
                    # Verified 1-block Causal Solution Found!
                    self.subgoal_counter += 1
                    bridge_id = f"sg_causal_bridge_{self.subgoal_counter}"
                    target_id = f"sg_causal_target_{self.subgoal_counter}"

                    # Subgoal 1: Push block into chasm
                    bridge_sg = Subgoal(
                        subgoal_id=bridge_id,
                        subgoal_type=SubgoalType.BRIDGE_GAP,
                        target_pos=drop_p,
                        target_color=block.color,
                        max_steps=35,
                        metadata={
                            'block_pos': block.centroid,
                            'block_coords': block.coords,
                            'block_color': block.color,
                            'ledge_pos': ledge_p,
                            'drop_pos': drop_p,
                            'landing_pos': land_p,
                            'push_dir': push_dir,
                            'action': 'bridge_gap',
                            'stack_tier': 1
                        },
                        effects={
                            'hypothetical_solids': {land_p}
                        }
                    )

                    # Subgoal 2: Reach primary target (conditioned on bridge completion)
                    target_sg = Subgoal(
                        subgoal_id=target_id,
                        subgoal_type=SubgoalType.REACH_ENTITY,
                        target_pos=target_pos,
                        max_steps=50,
                        metadata={
                            'target_pos': target_pos,
                            'action': 'cross_bridge_to_goal'
                        },
                        precondition_ids=[bridge_id]
                    )

                    return [bridge_sg, target_sg]

        # 2. Multi-Box Stacking / Multi-Gap Bridging (2 Blocks)
        if len(blocks) >= 2:
            for i, b1 in enumerate(blocks):
                for ledge_p1, drop_p1, land_p1, push_dir1 in candidate_drop_edges:
                    solids1 = {land_p1}
                    for j, b2 in enumerate(blocks):
                        if i == j:
                            continue
                        for ledge_p2, drop_p2, _, push_dir2 in candidate_drop_edges:
                            # Drop second block taking into account solids1 as new solid ground
                            land_p2, _, _ = simulate_gravity_fall(
                                grid_2d, drop_p2, gravity, solids,
                                hazard_colors=None, suspect_hazard_colors=None,
                                bg_canvas=bg_canvas, stride=stride,
                                hypothetical_solids=solids1
                            )
                            if land_p2 is None or land_p2 in solids1:
                                continue
                            solids2 = solids1 | {land_p2}
                            test_path2 = platformer_a_star(
                                grid=grid_2d,
                                start_pos=avatar_pos,
                                goal_candidates={target_pos},
                                gravity=gravity,
                                jump_height=jump_height,
                                jump_reach=jump_reach,
                                solid_colors=solids,
                                hazard_colors=hazard_colors,
                                suspect_hazard_colors=suspect_hazard_colors,
                                bg_canvas=bg_canvas,
                                available_actions=available_actions,
                                stride=stride,
                                hypothetical_solids=solids2
                            )
                            if test_path2 is not None:
                                self.subgoal_counter += 1
                                b_id1 = f"sg_causal_bridge_{self.subgoal_counter}_1"
                                b_id2 = f"sg_causal_bridge_{self.subgoal_counter}_2"
                                target_id = f"sg_causal_target_{self.subgoal_counter}"

                                sg1 = Subgoal(
                                    subgoal_id=b_id1,
                                    subgoal_type=SubgoalType.BRIDGE_GAP,
                                    target_pos=drop_p1,
                                    target_color=b1.color,
                                    max_steps=35,
                                    metadata={
                                        'block_pos': b1.centroid,
                                        'block_coords': b1.coords,
                                        'block_color': b1.color,
                                        'ledge_pos': ledge_p1,
                                        'drop_pos': drop_p1,
                                        'landing_pos': land_p1,
                                        'push_dir': push_dir1,
                                        'action': 'bridge_gap',
                                        'stack_tier': 1
                                    },
                                    effects={'hypothetical_solids': {land_p1}}
                                )
                                sg2 = Subgoal(
                                    subgoal_id=b_id2,
                                    subgoal_type=SubgoalType.BRIDGE_GAP,
                                    target_pos=drop_p2,
                                    target_color=b2.color,
                                    max_steps=35,
                                    metadata={
                                        'block_pos': b2.centroid,
                                        'block_coords': b2.coords,
                                        'block_color': b2.color,
                                        'ledge_pos': ledge_p2,
                                        'drop_pos': drop_p2,
                                        'landing_pos': land_p2,
                                        'push_dir': push_dir2,
                                        'action': 'bridge_gap',
                                        'stack_tier': 2
                                    },
                                    precondition_ids=[b_id1],
                                    effects={'hypothetical_solids': {land_p2}}
                                )
                                target_sg = Subgoal(
                                    subgoal_id=target_id,
                                    subgoal_type=SubgoalType.REACH_ENTITY,
                                    target_pos=target_pos,
                                    max_steps=50,
                                    metadata={
                                        'target_pos': target_pos,
                                        'action': 'cross_bridge_to_goal'
                                    },
                                    precondition_ids=[b_id2]
                                )
                                return [sg1, sg2, target_sg]

        # 3. Multi-Box Stacking (3 Blocks)
        if len(blocks) >= 3:
            for i, b1 in enumerate(blocks):
                for ledge_p1, drop_p1, land_p1, push_dir1 in candidate_drop_edges:
                    solids1 = {land_p1}
                    for j, b2 in enumerate(blocks):
                        if i == j:
                            continue
                        for ledge_p2, drop_p2, _, push_dir2 in candidate_drop_edges:
                            land_p2, _, _ = simulate_gravity_fall(
                                grid_2d, drop_p2, gravity, solids,
                                hazard_colors=None, suspect_hazard_colors=None,
                                bg_canvas=bg_canvas, stride=stride,
                                hypothetical_solids=solids1
                            )
                            if land_p2 is None or land_p2 in solids1:
                                continue
                            solids2 = solids1 | {land_p2}
                            for k, b3 in enumerate(blocks):
                                if k == i or k == j:
                                    continue
                                for ledge_p3, drop_p3, _, push_dir3 in candidate_drop_edges:
                                    land_p3, _, _ = simulate_gravity_fall(
                                        grid_2d, drop_p3, gravity, solids,
                                        hazard_colors=None, suspect_hazard_colors=None,
                                        bg_canvas=bg_canvas, stride=stride,
                                        hypothetical_solids=solids2
                                    )
                                    if land_p3 is None or land_p3 in solids2:
                                        continue
                                    solids3 = solids2 | {land_p3}
                                    test_path3 = platformer_a_star(
                                        grid=grid_2d,
                                        start_pos=avatar_pos,
                                        goal_candidates={target_pos},
                                        gravity=gravity,
                                        jump_height=jump_height,
                                        jump_reach=jump_reach,
                                        solid_colors=solids,
                                        hazard_colors=hazard_colors,
                                        suspect_hazard_colors=suspect_hazard_colors,
                                        bg_canvas=bg_canvas,
                                        available_actions=available_actions,
                                        stride=stride,
                                        hypothetical_solids=solids3
                                    )
                                    if test_path3 is not None:
                                        self.subgoal_counter += 1
                                        b_id1 = f"sg_causal_bridge_{self.subgoal_counter}_1"
                                        b_id2 = f"sg_causal_bridge_{self.subgoal_counter}_2"
                                        b_id3 = f"sg_causal_bridge_{self.subgoal_counter}_3"
                                        target_id = f"sg_causal_target_{self.subgoal_counter}"

                                        sg1 = Subgoal(
                                            subgoal_id=b_id1,
                                            subgoal_type=SubgoalType.BRIDGE_GAP,
                                            target_pos=drop_p1,
                                            target_color=b1.color,
                                            max_steps=35,
                                            metadata={'block_pos': b1.centroid, 'block_coords': b1.coords, 'block_color': b1.color, 'ledge_pos': ledge_p1, 'drop_pos': drop_p1, 'landing_pos': land_p1, 'push_dir': push_dir1, 'action': 'bridge_gap', 'stack_tier': 1},
                                            effects={'hypothetical_solids': {land_p1}}
                                        )
                                        sg2 = Subgoal(
                                            subgoal_id=b_id2,
                                            subgoal_type=SubgoalType.BRIDGE_GAP,
                                            target_pos=drop_p2,
                                            target_color=b2.color,
                                            max_steps=35,
                                            metadata={'block_pos': b2.centroid, 'block_coords': b2.coords, 'block_color': b2.color, 'ledge_pos': ledge_p2, 'drop_pos': drop_p2, 'landing_pos': land_p2, 'push_dir': push_dir2, 'action': 'bridge_gap', 'stack_tier': 2},
                                            precondition_ids=[b_id1],
                                            effects={'hypothetical_solids': {land_p2}}
                                        )
                                        sg3 = Subgoal(
                                            subgoal_id=b_id3,
                                            subgoal_type=SubgoalType.BRIDGE_GAP,
                                            target_pos=drop_p3,
                                            target_color=b3.color,
                                            max_steps=35,
                                            metadata={'block_pos': b3.centroid, 'block_coords': b3.coords, 'block_color': b3.color, 'ledge_pos': ledge_p3, 'drop_pos': drop_p3, 'landing_pos': land_p3, 'push_dir': push_dir3, 'action': 'bridge_gap', 'stack_tier': 3},
                                            precondition_ids=[b_id2],
                                            effects={'hypothetical_solids': {land_p3}}
                                        )
                                        target_sg = Subgoal(
                                            subgoal_id=target_id,
                                            subgoal_type=SubgoalType.REACH_ENTITY,
                                            target_pos=target_pos,
                                            max_steps=50,
                                            metadata={'target_pos': target_pos, 'action': 'cross_bridge_to_goal'},
                                            precondition_ids=[b_id3]
                                        )
                                        return [sg1, sg2, sg3, target_sg]

        return None

    def induce_barrier_unlock_plan(
        self,
        grid: np.ndarray,
        avatar_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        bg_canvas: int = 0,
        floor_colors: Optional[Set[int]] = None,
        solid_colors: Optional[Set[int]] = None,
        available_actions: Optional[List[int]] = None,
        max_depth: int = 5
    ) -> Optional[List[Subgoal]]:
        """
        Synthesizes a teleological DAG for unlocking multi-step doors / barriers (Depth >= 3):
        Uses Fixed-Point Forward Reachability Chaining:
        1. Repeatedly expands the avatar's reachable component.
        2. At each frontier, checks if target is reached.
        3. If not, identifies barrier colors blocking paths to target, locates newly accessible
           matching keys/tokens, and adds them as formal DAG prerequisites.
        4. Terminates when target is reached or when no new keys can be accessed.
        """
        grid_2d = np.asarray(grid)
        while grid_2d.ndim > 2:
            grid_2d = grid_2d[-1]
        H, W = grid_2d.shape
        floors = floor_colors or set()
        solids = solid_colors or set()

        def is_wall(r: int, c: int) -> bool:
            return not (0 <= r < H and 0 <= c < W) or int(grid_2d[r, c]) in solids

        unlocked_barriers: Set[int] = set()
        collected_keys: List[Tuple[Tuple[int, int], int]] = []
        used_key_coords: Set[Tuple[int, int]] = set()

        for depth in range(max_depth):
            # 1. Compute reachable set with current unlocked_barriers
            reachable_cells: Set[Tuple[int, int]] = set()
            q = [avatar_pos]
            reachable_cells.add(avatar_pos)
            while q:
                cr, cc = q.pop(0)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not is_wall(nr, nc) and (nr, nc) not in reachable_cells:
                        val = int(grid_2d[nr, nc])
                        # Can traverse if floor, bg, already unlocked, or key position
                        if val in floors or val == bg_canvas or val in unlocked_barriers or (nr, nc) in used_key_coords:
                            reachable_cells.add((nr, nc))
                            q.append((nr, nc))
                        elif (nr, nc) == target_pos:
                            reachable_cells.add((nr, nc))

            # 2. Check if target_pos is reached
            if target_pos in reachable_cells:
                if not collected_keys:
                    return None  # Target already reachable directly without keys

                # Build Teleological Subgoal DAG chain
                chain: List[Subgoal] = []
                prev_id = None
                for idx, (k_pos, k_color) in enumerate(collected_keys):
                    self.subgoal_counter += 1
                    key_id = f"sg_key_{self.subgoal_counter}"
                    pre_ids = [prev_id] if prev_id is not None else []
                    sg = Subgoal(
                        subgoal_id=key_id,
                        subgoal_type=SubgoalType.REACH_ENTITY,
                        target_pos=k_pos,
                        target_color=k_color,
                        max_steps=25,
                        metadata={'action': 'collect_key', 'barrier_color': k_color, 'key_index': idx},
                        precondition_ids=pre_ids,
                        effects={'unlocked_barriers': {k_color}}
                    )
                    chain.append(sg)
                    prev_id = key_id

                self.subgoal_counter += 1
                target_id = f"sg_barrier_target_{self.subgoal_counter}"
                target_sg = Subgoal(
                    subgoal_id=target_id,
                    subgoal_type=SubgoalType.REACH_ENTITY,
                    target_pos=target_pos,
                    max_steps=50,
                    metadata={'action': 'reach_target_after_all_keys'},
                    precondition_ids=[prev_id] if prev_id is not None else []
                )
                chain.append(target_sg)
                return chain

            # 3. Find barriers blocking paths from reachable_cells to target_pos
            queue = []
            bfs_visited = set()
            for rp in reachable_cells:
                queue.append((rp[0], rp[1], []))
                bfs_visited.add(rp)

            blocking_barriers: Set[int] = set()
            found_path_barriers = False

            while queue:
                r, c, path_barriers = queue.pop(0)
                if (r, c) == target_pos:
                    blocking_barriers.update(path_barriers)
                    found_path_barriers = True
                    break

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not is_wall(nr, nc) and (nr, nc) not in bfs_visited:
                        bfs_visited.add((nr, nc))
                        v = int(grid_2d[nr, nc])
                        next_pb = list(path_barriers)
                        if v != bg_canvas and v not in floors and v not in unlocked_barriers and (nr, nc) != target_pos and (nr, nc) not in used_key_coords:
                            next_pb.append(v)
                        queue.append((nr, nc, next_pb))

            if not found_path_barriers:
                # No path exists even if barriers were unlocked
                return None

            # 4. Search accessible boundary for any key matching blocking_barriers
            found_key = False
            for b_color in blocking_barriers:
                key_coords = np.argwhere(grid_2d == b_color)
                for kr, kc in key_coords:
                    kp = (int(kr), int(kc))
                    if kp in used_key_coords:
                        continue
                    is_accessible = (kp in reachable_cells) or any(
                        (kp[0] + dr, kp[1] + dc) in reachable_cells
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    )
                    if is_accessible:
                        collected_keys.append((kp, b_color))
                        used_key_coords.add(kp)
                        unlocked_barriers.add(b_color)
                        found_key = True
                        break
                if found_key:
                    break

            if not found_key:
                # Search for any accessible uncollected entity that could open another room
                all_cells = np.argwhere(grid_2d != bg_canvas)
                for r_c in all_cells:
                    kp = (int(r_c[0]), int(r_c[1]))
                    c_val = int(grid_2d[kp])
                    if c_val in solids or c_val in floors or c_val in unlocked_barriers or kp in used_key_coords:
                        continue
                    is_accessible = (kp in reachable_cells) or any(
                        (kp[0] + dr, kp[1] + dc) in reachable_cells
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    )
                    if is_accessible:
                        collected_keys.append((kp, c_val))
                        used_key_coords.add(kp)
                        unlocked_barriers.add(c_val)
                        found_key = True
                        break

            if not found_key:
                return None

        return None
