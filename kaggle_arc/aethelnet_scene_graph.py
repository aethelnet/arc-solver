"""
=============================================================================
AETHELNET CAUSAL SCENE GRAPH SOLVER
=============================================================================
Unified, Domain-Agnostic Active Inference Scene Graph Engine for ARC-AGI-3.
Replaces fragmented, warring subsolvers with a single causal multigraph:
- Nodes: Discrete segmented entities (Avatars, Pushable Blocks, Receptors, Switches, Walls).
- Edges: Physical and symbolic affordances (Walk, Push, Click-Focus, Toggle).
- Planner: Unified Multi-Agent A* / BFS over induced physical regimes (Unit-Step vs Continuous-Slide).
=============================================================================
"""

import sys
import collections
import heapq
from enum import Enum
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np


class KinematicsRegime(Enum):
    UNKNOWN = "UNKNOWN"
    UNIT_STEP = "UNIT_STEP"                 # Moves fixed stride (1..3 cells) per step
    CONTINUOUS_SLIDE = "CONTINUOUS_SLIDE"   # Slides until colliding with obstacle (inertia/ice)


class SceneEntity:
    """Represents a segmented physical or functional component on the board."""
    def __init__(
        self,
        entity_id: int,
        color: int,
        cells: Set[Tuple[int, int]],
        bbox: Tuple[int, int, int, int]
    ):
        self.entity_id = entity_id
        self.color = color
        self.cells = set(cells)
        self.bbox = bbox # (min_r, min_c, max_r, max_c)
        self.is_active_avatar: bool = False
        self.is_selectable: bool = False
        self.role: str = 'unknown' # 'avatar', 'block', 'slot', 'wall'
        
    @property
    def center(self) -> Tuple[int, int]:
        if not self.cells:
            return (self.bbox[0], self.bbox[1])
        rs = [r for r, c in self.cells]
        cs = [c for r, c in self.cells]
        return (int(round(np.mean(rs))), int(round(np.mean(cs))))

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.bbox[2] - self.bbox[0] + 1, self.bbox[3] - self.bbox[1] + 1)


class AethelnetSceneGraph:
    """
    Online Causal Multigraph Engine:
    Maintains graph representation of game state and executes unified causal plans.
    """
    def __init__(self):
        self.entities: Dict[int, SceneEntity] = {}
        self.active_avatar_id: Optional[int] = None
        self.controllable_avatar_ids: Set[int] = set()
        
        # Kinematics & Regime
        self.regime: KinematicsRegime = KinematicsRegime.UNKNOWN
        self.stride: int = 1
        self.slide_observations: int = 0
        
        # Action mappings (1=UP, 2=DOWN, 3=LEFT, 4=RIGHT)
        self.action_to_dir = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        self.dir_to_action = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
        
        # Frame tracking
        self.prev_grid: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_action_data: Optional[Dict[str, Any]] = None
        self.probe_step_count: int = 0
        
        # Planned path queue: List of (action_id, action_data)
        self.plan_queue: List[Tuple[int, Optional[Dict[str, Any]]]] = []
        
        # Cycle detection & Spatial Exploration Heatmap
        self.state_history: List[int] = []
        self.visit_counts: Dict[Tuple[int, int], int] = collections.defaultdict(int)

    def reset(self):
        self.entities.clear()
        self.active_avatar_id = None
        self.controllable_avatar_ids.clear()
        self.regime = KinematicsRegime.UNKNOWN
        self.stride = 1
        self.slide_observations = 0
        self.prev_grid = None
        self.last_action = None
        self.last_action_data = None
        self.probe_step_count = 0
        self.plan_queue.clear()
        self.state_history.clear()
        self.visit_counts.clear()

    def segment_grid(self, grid: np.ndarray) -> List[SceneEntity]:
        """Connected-component entity extraction with background isolation."""
        h, w = grid.shape
        bg = int(collections.Counter(grid.flatten()).most_common(1)[0][0])
        h_max = h - 1 if h >= 64 else h
        
        visited = np.zeros((h, w), dtype=bool)
        entities: List[SceneEntity] = []
        ent_id = 0
        
        for r in range(h_max):
            for c in range(w):
                if grid[r, c] == bg or visited[r, c]:
                    continue
                color = int(grid[r, c])
                comp: Set[Tuple[int, int]] = set()
                q = [(r, c)]
                visited[r, c] = True
                min_r, max_r, min_c, max_c = r, r, c, c
                
                while q:
                    cr, cc = q.pop(0)
                    comp.add((cr, cc))
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)
                    
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h_max and 0 <= nc < w:
                            if not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                                
                entities.append(SceneEntity(
                    entity_id=ent_id,
                    color=color,
                    cells=comp,
                    bbox=(min_r, min_c, max_r, max_c)
                ))
                ent_id += 1
                
        return entities

    def update_observations(self, current_grid: np.ndarray):
        """Builds or refreshes the scene graph and induces kinematics/focus."""
        h, w = current_grid.shape
        h_play = h - 1 if h >= 64 else h
        current_hash = hash(current_grid[:h_play, :].tobytes())
        self.state_history.append(current_hash)
        if len(self.state_history) > 30:
            self.state_history.pop(0)

        curr_entities = self.segment_grid(current_grid)
        self.entities = {e.entity_id: e for e in curr_entities}

        if self.prev_grid is not None and self.last_action is not None:
            diff_mask = (current_grid[:h_play, :] != self.prev_grid[:h_play, :])
            if np.any(diff_mask):
                diff_coords = set((int(r), int(c)) for r, c in np.argwhere(diff_mask))
                
                # 1. Focus Transfer via Click (Action 6)
                if self.last_action == 6 and self.last_action_data:
                    cx = self.last_action_data.get('x', -1)
                    cy = self.last_action_data.get('y', -1)
                    for ent in curr_entities:
                        if ent.bbox[0] <= cy <= ent.bbox[2] and ent.bbox[1] <= cx <= ent.bbox[3]:
                            ent.is_selectable = True
                            self.controllable_avatar_ids.add(ent.entity_id)
                            self.active_avatar_id = ent.entity_id
                            break
                            
                # 2. Kinematics Displacement via Directional Actions (1..4)
                elif self.last_action in [1, 2, 3, 4]:
                    prev_entities = self.segment_grid(self.prev_grid)
                    dr, dc = self.action_to_dir.get(self.last_action, (0, 0))
                    color_counts = collections.Counter(current_grid[:h_play, :].flatten())
                    
                    active_p_ents = [p for p in prev_entities if (p.cells & diff_coords)]
                    active_p_ents.sort(key=lambda p: color_counts.get(p.color, 999999))
                    
                    for p_ent in active_p_ents:
                        if color_counts.get(p_ent.color, 0) > 200:
                            continue
                            
                        best_match = None
                        best_dist = 999999
                        for c_ent in curr_entities:
                            if c_ent.color == p_ent.color and abs(len(c_ent.cells) - len(p_ent.cells)) <= 3:
                                disp_r = c_ent.center[0] - p_ent.center[0]
                                disp_c = c_ent.center[1] - p_ent.center[1]
                                if dr != 0 and np.sign(disp_r) == np.sign(dr) and abs(disp_c) <= 2:
                                    if 0 < abs(disp_r) < best_dist:
                                        best_dist = abs(disp_r)
                                        best_match = c_ent
                                elif dc != 0 and np.sign(disp_c) == np.sign(dc) and abs(disp_r) <= 2:
                                    if 0 < abs(disp_c) < best_dist:
                                        best_dist = abs(disp_c)
                                        best_match = c_ent

                        if best_match is not None:
                            disp_r = best_match.center[0] - p_ent.center[0]
                            disp_c = best_match.center[1] - p_ent.center[1]
                            dist = abs(disp_r) + abs(disp_c)
                            if dist > 0:
                                self.active_avatar_id = best_match.entity_id
                                self.controllable_avatar_ids.add(best_match.entity_id)
                                if dist > 4:
                                    self.regime = KinematicsRegime.CONTINUOUS_SLIDE
                                    self.slide_observations += 1
                                elif dist <= 4 and self.slide_observations == 0:
                                    self.regime = KinematicsRegime.UNIT_STEP
                                    self.stride = dist
                                break

        # Controllable sister unit discovery: units sharing identical shape & rare color
        if self.active_avatar_id is not None and self.active_avatar_id in self.entities:
            active_ent = self.entities[self.active_avatar_id]
            for ent in self.entities.values():
                if ent.entity_id != active_ent.entity_id:
                    if ent.shape == active_ent.shape and abs(len(ent.cells) - len(active_ent.cells)) <= 2:
                        ent.is_selectable = True
                        self.controllable_avatar_ids.add(ent.entity_id)

    def simulate_step(
        self,
        pos: Tuple[int, int],
        shape: Tuple[int, int],
        direction: Tuple[int, int],
        solid_mask: np.ndarray
    ) -> Tuple[int, int]:
        """Simulates physical motion under the induced kinematics regime."""
        h, w = solid_mask.shape
        r, c = pos
        sh_r, sh_c = shape
        dr, dc = direction
        
        def is_blocked(tr: int, tc: int) -> bool:
            if tr < 0 or tr + sh_r > h or tc < 0 or tc + sh_c > w:
                return True
            return bool(np.any(solid_mask[tr:tr+sh_r, tc:tc+sh_c]))

        if self.regime == KinematicsRegime.CONTINUOUS_SLIDE:
            curr_r, curr_c = r, c
            while True:
                nr, nc = curr_r + dr, curr_c + dc
                if is_blocked(nr, nc):
                    break
                curr_r, curr_c = nr, nc
            return (curr_r, curr_c)
        else:
            s = self.stride
            nr, nc = r + dr * s, c + dc * s
            if not is_blocked(nr, nc):
                return (nr, nc)
            return (r, c)

    def plan_path_to_target(
        self,
        start_pos: Tuple[int, int],
        shape: Tuple[int, int],
        target_pos: Tuple[int, int],
        solid_mask: np.ndarray,
        available_actions: List[int],
        max_depth: int = 150
    ) -> List[int]:
        """Unified A* search on the induced kinematics (Unit-Step or Continuous-Slide)."""
        sr, sc = start_pos
        tr, tc = target_pos
        
        h0 = abs(sr - tr) + abs(sc - tc)
        heap = [(h0, 0, (sr, sc), [])]
        visited = {(sr, sc): 0}
        
        while heap:
            f, g, curr, path = heapq.heappop(heap)
            cr, cc = curr
            
            # Goal check: within collision boundary of target
            if abs(cr - tr) <= max(1, shape[0]) and abs(cc - tc) <= max(1, shape[1]):
                return path
                
            if g >= max_depth:
                continue
                
            for act in [1, 2, 3, 4]:
                if act not in available_actions:
                    continue
                d = self.action_to_dir[act]
                nr, nc = self.simulate_step((cr, cc), shape, d, solid_mask)
                if (nr, nc) == (cr, cc):
                    continue # Blocked
                    
                new_g = g + 1
                if (nr, nc) not in visited or new_g < visited[(nr, nc)]:
                    visited[(nr, nc)] = new_g
                    heur = abs(nr - tr) + abs(nc - tc)
                    heapq.heappush(heap, (new_g + heur, new_g, (nr, nc), path + [act]))
                    
        return []

    def choose_action(
        self,
        current_grid: np.ndarray,
        available_actions: List[int],
        levels_completed: int
    ) -> Tuple[int, Optional[Dict[str, Any]], str]:
        """
        Plans next optimal causal action:
        1. Follow active plan queue.
        2. Probe kinematics if unknown.
        3. Break periodic 2-step / 4-step oscillations via focus switch.
        4. Forward A* search on induced kinematics towards rarest goal entity.
        """
        self.update_observations(current_grid)
        
        # 1. Follow active plan queue
        if self.plan_queue:
            act, data = self.plan_queue.pop(0)
            if act in available_actions:
                self.last_action = act
                self.last_action_data = data
                self.prev_grid = current_grid.copy()
                return (act, data, "ExecutingPlan")
            self.plan_queue.clear()

        # 2. Kinematics Probing
        if self.active_avatar_id is None:
            probe_sequence = [4, 2, 3, 1]
            probe_candidates = [a for a in probe_sequence if a in available_actions]
            if probe_candidates:
                p_act = probe_candidates[self.probe_step_count % len(probe_candidates)]
                self.probe_step_count += 1
                self.last_action = p_act
                self.last_action_data = None
                self.prev_grid = current_grid.copy()
                return (p_act, None, f"KinematicProbe_{p_act}")

        # 3. Detect Periodic Oscillation
        is_oscillating = False
        if len(self.state_history) >= 6:
            if (self.state_history[-1] == self.state_history[-3] == self.state_history[-5] and
                self.state_history[-2] == self.state_history[-4] == self.state_history[-6]):
                is_oscillating = True
        if len(self.state_history) >= 8 and not is_oscillating:
            if (self.state_history[-1] == self.state_history[-5] and
                self.state_history[-2] == self.state_history[-6] and
                self.state_history[-3] == self.state_history[-7] and
                self.state_history[-4] == self.state_history[-8]):
                is_oscillating = True

        # 4. Escape Oscillation via Multi-Agent Focus Switch
        if is_oscillating and 6 in available_actions and len(self.controllable_avatar_ids) > 1:
            active_id = self.active_avatar_id
            candidates = [
                self.entities[eid] for eid in self.controllable_avatar_ids
                if eid != active_id and eid in self.entities
            ]
            if candidates:
                target_unit = candidates[0]
                cr, cc = target_unit.center
                self.active_avatar_id = target_unit.entity_id
                self.last_action = 6
                self.last_action_data = {'x': int(cc), 'y': int(cr)}
                self.prev_grid = current_grid.copy()
                return (6, {'x': int(cc), 'y': int(cr)}, f"SwitchAvatar_{target_unit.entity_id}")

        # 5. Goal-Directed Induced Kinematics A* Search
        avatar_ent = self.entities.get(self.active_avatar_id)
        if avatar_ent is not None:
            h, w = current_grid.shape
            h_play = h - 1 if h >= 64 else h
            counts = collections.Counter(current_grid[:h_play, :].flatten())
            bg = counts.most_common(1)[0][0]
            
            sorted_colors = [c for c, _ in counts.most_common() if c != bg]
            wall_color = sorted_colors[0] if sorted_colors else bg
            solid_mask = (current_grid[:h_play, :] == wall_color)
            
            # Identify goal candidates (rarest non-wall, non-avatar entities)
            goal_candidates = [
                e for e in self.entities.values()
                if e.entity_id != self.active_avatar_id and
                   e.entity_id not in self.controllable_avatar_ids and
                   e.color != wall_color and
                   e.color != bg
            ]
            goal_candidates.sort(key=lambda e: (len(e.cells), counts.get(e.color, 9999)))
            
            # Try planning to candidate goals via induced kinematics
            for goal_ent in goal_candidates[:3]:
                path = self.plan_path_to_target(
                    start_pos=avatar_ent.bbox[:2],
                    shape=avatar_ent.shape,
                    target_pos=goal_ent.bbox[:2],
                    solid_mask=solid_mask,
                    available_actions=available_actions
                )
                if path:
                    first_act = path[0]
                    self.plan_queue = [(a, None) for a in path[1:]]
                    self.last_action = first_act
                    self.last_action_data = None
                    self.prev_grid = current_grid.copy()
                    return (first_act, None, f"InducedKinematicsPlan_{first_act}")

            # Update visit count for current avatar position
            curr_pos = avatar_ent.bbox[:2]
            self.visit_counts[curr_pos] += 1

            # Fallback: exploratory legal move minimizing spatial visit count
            legal_moves = [a for a in [1, 2, 3, 4] if a in available_actions]
            opposite = {1: 2, 2: 1, 3: 4, 4: 3}
            opp = opposite.get(self.last_action)

            scored_moves = []
            for act in legal_moves:
                d = self.action_to_dir[act]
                next_pos = self.simulate_step(curr_pos, avatar_ent.shape, d, solid_mask)
                if next_pos != curr_pos:
                    # Penalize reversing direction if oscillating or visiting familiar tile
                    penalty = 50 if (is_oscillating and act == opp) else 0
                    score = self.visit_counts[next_pos] + penalty
                    scored_moves.append((score, act))

            if scored_moves:
                scored_moves.sort(key=lambda x: x[0])
                best_act = scored_moves[0][1]
                self.last_action = best_act
                self.last_action_data = None
                self.prev_grid = current_grid.copy()
                return (best_act, None, f"InducedKinematicsExplore_{best_act}")

        # 6. Fallback: select dynamically from legal actions
        candidates = [a for a in available_actions if a != 0]
        fallback = np.random.choice(candidates) if candidates else 1
        data = {'x': 32, 'y': 32} if fallback == 6 else None
        self.last_action = fallback
        self.last_action_data = data
        self.prev_grid = current_grid.copy()
        return (int(fallback), data, "SceneGraphFallback")
