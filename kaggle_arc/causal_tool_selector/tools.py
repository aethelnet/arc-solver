"""
=============================================================================
ARC-AGI CAUSAL META-TOOL SELECTOR: STANDARDIZED SOLVER TOOLS
=============================================================================
Defines the uniform BaseMetaTool interface and concrete tool wrappers around
high-performance neurosymbolic solvers from my_agent.py.
=============================================================================
"""

import sys
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import numpy as np

# Ensure repo root is available
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from kaggle_arc.my_agent import (
        OnlineLightsOutWorldModel,
        OnlinePlatformerWorldModel,
        OnlineSokobanWorldModel,
        OnlineBasketWorldModel,
        OnlineTractorWorldModel,
        OnlineMirrorWorldModel,
        OnlineRobotAssemblyWorldModel,
        OnlineHybridNavClickWorldModel,
        OnlineBarycenterWorldModel,
        OnlineSliderWorldModel,
        OnlineIsoSliceWorldModel,
        OnlinePegSolitaireWorldModel,
        OnlineChipSocketWorldModel,
        OnlineGlyphDialWorldModel,
        OnlineFluidSpillWorldModel,
        OnlineRoboticArmWorldModel,
        OnlineStealthEvasionWorldModel,
        OnlineCraneWorldModel,
        OnlineCurlingWorldModel,
        OnlineTimeCloneWorldModel,
        OnlineStateTransformWorldModel,
        OnlinePairMatchWorldModel,
        OnlineSpellcastWorldModel,
        OnlineTangramWorldModel,
        OnlineRegister1DWorldModel
    )
except ImportError:
    from my_agent import (
        OnlineLightsOutWorldModel,
        OnlinePlatformerWorldModel,
        OnlineSokobanWorldModel,
        OnlineBasketWorldModel,
        OnlineTractorWorldModel,
        OnlineMirrorWorldModel,
        OnlineRobotAssemblyWorldModel,
        OnlineHybridNavClickWorldModel,
        OnlineBarycenterWorldModel,
        OnlineSliderWorldModel,
        OnlineIsoSliceWorldModel,
        OnlinePegSolitaireWorldModel,
        OnlineChipSocketWorldModel,
        OnlineGlyphDialWorldModel,
        OnlineFluidSpillWorldModel,
        OnlineRoboticArmWorldModel,
        OnlineStealthEvasionWorldModel,
        OnlineCraneWorldModel,
        OnlineCurlingWorldModel,
        OnlineTimeCloneWorldModel,
        OnlineStateTransformWorldModel,
        OnlinePairMatchWorldModel,
        OnlineSpellcastWorldModel,
        OnlineTangramWorldModel,
        OnlineRegister1DWorldModel
    )
from .affordance_encoder import AffordanceFeatures
from .frontier_explorer import DomainAgnosticFrontierExplorer
from .options_framework import OptionsController


class BaseMetaTool(ABC):
    """
    Standardized interface for high-level neurosymbolic solvers.
    Each tool acts as an autonomous macro-actuator in the Causal Active Inference loop.
    """
    def __init__(self, tool_id: int, name: str):
        self.tool_id = tool_id
        self.name = name

    @abstractmethod
    def can_handle(self, features: AffordanceFeatures) -> float:
        """
        Calculates prior affordance compatibility in [0.0, 1.0].
        0.0 = Hard physical constraint violation (e.g. Nav tool on Click-only puzzle).
        1.0 = Strong natural resonance with environmental affordance.
        """
        pass

    @abstractmethod
    def step(
        self,
        grid: np.ndarray,
        levels_completed: int,
        available_actions: Optional[List[int]] = None
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Executes one solver step on the grid.
        Returns:
            action_id: Integer action (1..6)
            action_data: Optional dict (e.g. {'x': px, 'y': py})
        """
        pass

    @abstractmethod
    def reset(self):
        """Resets solver state between levels or episodes."""
        pass

    def is_executing_plan(self) -> bool:
        """Returns True if the tool has an active, unfinished planned sequence."""
        model = getattr(self, 'model', None)
        if model is not None:
            if hasattr(model, 'is_executing_plan'):
                return model.is_executing_plan()
            if hasattr(model, 'plan') and hasattr(model, 'plan_idx'):
                return model.plan_idx < len(model.plan)
            if hasattr(model, 'plan') and hasattr(model, 'plan_step_idx'):
                return model.plan_step_idx < len(model.plan)
            if hasattr(model, 'plan') and hasattr(model, 'plan_step'):
                return model.plan_step < len(model.plan)
        return False

    def inherit_context(
        self,
        avatar_pos: Optional[Tuple[int, int]] = None,
        floor_colors: Optional[Set[int]] = None,
        calibrator: Any = None,
        color_calibrator: Any = None
    ):
        """Allows incoming tool to seamlessly inherit spatial context from prior tool."""
        if avatar_pos is not None:
            self.avatar_pos = avatar_pos
            exp = getattr(self, 'explorer', None)
            if exp is not None:
                exp.avatar_pos = avatar_pos
            model = getattr(self, 'model', None)
            if model is not None and hasattr(model, 'avatar_pos'):
                model.avatar_pos = avatar_pos
        if floor_colors is not None:
            self.floor_colors = floor_colors
            exp = getattr(self, 'explorer', None)
            if exp is not None:
                exp.floor_colors = set(floor_colors)
            model = getattr(self, 'model', None)
            if model is not None and hasattr(model, 'floor_colors'):
                model.floor_colors = set(floor_colors)


class LightsOutTool(BaseMetaTool):
    """
    Tool 0: Galois Field GF(2) Involutory Constraint Propagation Solver.
    Specialized for toggle, inverted parity, and matrix click puzzles (ft09).
    Requires: Click only (no nav, no interact, no reset).
    """
    def __init__(self):
        super().__init__(tool_id=0, name="LightsOut_Galois")
        self.model = OnlineLightsOutWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_click == 0.0 or features.has_nav > 0.0 or features.has_reset > 0.0 or features.has_interact > 0.0:
            return 0.0
        if features.spatial_symmetry < 0.65 and features.entity_count_norm > 0.35:
            return 0.95
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        target = self.model.step(grid, levels_completed)
        return (6, {'x': int(target[0]), 'y': int(target[1])})

    def reset(self):
        self.model = OnlineLightsOutWorldModel()


class GravityPlatformerTool(BaseMetaTool):
    """
    Tool 1: Gravity, Jump, and Horizontal Momentum Physics Solver (bp35).
    Requires: Nav + Click + Reset, NO interact. 1D constrained horizontal motion.
    """
    def __init__(self):
        super().__init__(tool_id=1, name="Gravity_Platformer")
        self.model = OnlinePlatformerWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_click == 0.0 or features.has_reset == 0.0 or features.has_interact > 0.0:
            return 0.0
        if features.has_1d_nav > 0.0:
            return 0.98
        return 0.10

    def is_executing_plan(self) -> bool:
        if hasattr(self.model, 'levels_completed') and self.model.levels_completed >= 5:
            plan = self.model.platformer_solver.get_plan(self.model.levels_completed)
            return self.model.plan_step_idx < len(plan)
        return getattr(self.model, 'stuck_counter', 0) < 12

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_val, act_data = self.model.step(grid, levels_completed)
        if act_val == 'L':
            act_id = 3
        elif act_val == 'R':
            act_id = 4
        elif act_val == 'C':
            act_id = 6
        else:
            act_id = int(act_val)
        return (act_id, act_data)

    def reset(self):
        self.model.reset_level()


class SokobanTool(BaseMetaTool):
    """
    Tool 2: Pick-and-Place, Obstacle Carrying, and Multi-Entity Block Pusher (wa30).
    Requires: Nav + Interact, NO click, NO reset.
    """
    def __init__(self):
        super().__init__(tool_id=2, name="Sokoban_BoxPusher")
        self.model = OnlineSokobanWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_interact == 0.0 or features.has_click > 0.0 or features.has_reset > 0.0:
            return 0.0
        if features.has_goal_tile == 0.0 and features.spatial_symmetry >= 0.85:
            return 0.95
        return 0.15

    def is_executing_plan(self) -> bool:
        return self.model.is_executing_plan()

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id = self.model.step(grid, levels_completed)
        return (int(act_id), None)

    def reset(self):
        self.model = OnlineSokobanWorldModel()


class ActiveBasketTool(BaseMetaTool):
    """
    Tool 3: Aiming, Swatch Color Picking, and Container Pouring Solver (cd82).
    Requires: Nav + Interact + Click, NO reset.
    """
    def __init__(self):
        super().__init__(tool_id=3, name="ActiveBasket_Pouring")
        self.model = OnlineBasketWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_interact == 0.0 or features.has_click == 0.0 or features.has_reset > 0.0:
            return 0.0
        if features.has_switch_widgets > 0.0 and features.entity_count_norm > 0.20:
            return 0.98
        return 0.15

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id, act_data = self.model.step(grid, levels_completed)
        return (int(act_id), act_data if act_data else None)

    def reset(self):
        self.model = OnlineBasketWorldModel()


class TractorBallTool(BaseMetaTool):
    """
    Tool 4: Continuous Cellular Gravitation & Particle Guidance Solver (su15).
    Requires: Click + Reset, NO nav, NO interact.
    """
    def __init__(self):
        super().__init__(tool_id=4, name="TractorBall_GravityStream")
        self.model = OnlineTractorWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_click == 0.0 or features.has_reset == 0.0 or features.has_nav > 0.0 or features.has_interact > 0.0:
            return 0.0
        return 0.98

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id, act_data = self.model.step(grid, levels_completed)
        return (int(act_id), act_data)

    def reset(self):
        self.model = OnlineTractorWorldModel()


class MirrorPuzzleTool(BaseMetaTool):
    """
    Tool 5: 1D/2D Reflection Symmetry and Polyomino Placement Solver (ar25).
    Requires: Nav + Reset + Interact, high symmetry.
    """
    def __init__(self):
        super().__init__(tool_id=5, name="Mirror_ReflectionAxis")
        self.model = OnlineMirrorWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_reset == 0.0 or features.has_interact == 0.0:
            return 0.0
        if features.spatial_symmetry >= 0.85 and features.has_goal_tile == 0.0:
            return 0.96
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id = self.model.step(grid, levels_completed)
        return (int(act_id), None)

    def reset(self):
        self.model = OnlineMirrorWorldModel()


class RobotAssemblyTool(BaseMetaTool):
    """
    Tool 6: Microcode Bit-Matrix & Hardware Assembler Solver (tn36).
    Requires: Pure click, high entity density (>=0.85), high symmetry, 6 colors.
    """
    def __init__(self):
        super().__init__(tool_id=6, name="RobotAssembly_Microcode")
        self.model = OnlineRobotAssemblyWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_click == 0.0 or features.has_nav > 0.0 or features.has_reset > 0.0 or features.has_interact > 0.0:
            return 0.0
        if features.spatial_symmetry >= 0.75 and features.entity_count_norm >= 0.85 and features.has_switch_widgets == 0.0:
            return 0.95
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id, act_data = self.model.step(grid, levels_completed)
        return (int(act_id), act_data)

    def reset(self):
        self.model = OnlineRobotAssemblyWorldModel()


class DomainAgnosticNavTool(BaseMetaTool):
    """
    Tool 7: Hierarchical Domain-Agnostic Options Explorer.
    Universal baseline navigator: sequences discrete subgoals (ReachEntity -> InteractEntity -> FrontierExplore)
    via formal SMDP Options.
    """
    def __init__(self):
        super().__init__(tool_id=7, name="Hierarchical_OptionsExplorer")
        self.controller = OptionsController()

    @property
    def explorer(self) -> DomainAgnosticFrontierExplorer:
        return self.controller.explorer

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0:
            return 0.0
        if features.has_1d_nav > 0.0:
            return 0.20 # 1D constrained navigation requires specialized physics solvers (bp35)
        if features.has_receptors > 0.0 and features.has_click > 0.0:
            return 0.20 # Multi-entity puck switching games require specialized IceCurlingTool
        if features.has_receptors > 0.0:
            return 0.98 # Exceptional fit for receptor/block matching games
        if features.has_interact > 0.0 and features.has_goal_tile > 0.0 and features.spatial_symmetry >= 0.85:
            return 0.20 # High symmetry interact + goal tile puzzles are state transformation silhouette puzzles (re86)
        if features.has_goal_tile > 0.0:
            return 0.98 # Exceptional fit for goal navigation
        if features.has_interact > 0.0:
            return 0.75 # Baseline fit for interact & navigation puzzles
        return 0.70 # Strong universal baseline fallback for all navigation puzzles

    def is_executing_plan(self) -> bool:
        return self.controller.current_option is not None

    def step(
        self,
        grid: np.ndarray,
        levels_completed: int,
        available_actions: Optional[List[int]] = None,
        *args,
        **kwargs
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        if available_actions is None:
            available_actions = [1, 2, 3, 4]
        act_id, act_data, info = self.controller.step(grid, available_actions, levels_completed)
        return (act_id, act_data)

    def reset(self, keep_avatar_identity: bool = False):
        self.controller.reset(keep_avatar_identity=keep_avatar_identity)


class HybridNavClickTool(BaseMetaTool):
    """
    Tool 8: Combined 2D Navigation and Causal Point-and-Click Actuators (dc22).
    Requires: Nav + Click, NO interact, NO reset. Switch widgets & high colors.
    """
    def __init__(self):
        super().__init__(tool_id=8, name="HybridNav_Click")
        self.model = OnlineHybridNavClickWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_click == 0.0 or features.has_interact > 0.0 or features.has_reset > 0.0:
            return 0.0
        if features.has_switch_widgets > 0.0 and features.color_count_norm >= 0.55:
            return 0.95
        return 0.10

    def is_executing_plan(self) -> bool:
        p_step = getattr(self.model, 'plan_step', getattr(self.model, 'plan_idx', 0))
        return p_step < len(getattr(self.model, 'plan', []))

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_type, act_val, act_data = self.model.step(grid, levels_completed)
        return (int(act_val), act_data if act_data else None)

    def reset(self):
        self.model = OnlineHybridNavClickWorldModel()


class BarycenterTool(BaseMetaTool):
    """
    Tool 9: Center-of-Mass & Vertex Weight Actuation Solver (r11l).
    Requires: Pure click (no nav, no interact, no reset). High symmetry, 7 colors, moderate entities.
    """
    def __init__(self):
        super().__init__(tool_id=9, name="Barycenter_CenterOfMass")
        self.model = OnlineBarycenterWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_click == 0.0 or features.has_nav > 0.0 or features.has_reset > 0.0 or features.has_interact > 0.0:
            return 0.0
        if features.spatial_symmetry >= 0.75 and features.has_switch_widgets == 0.0 and features.entity_count_norm < 0.85:
            return 0.95
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        target = self.model.step(grid, levels_completed)
        return (6, {'x': int(target[0]), 'y': int(target[1])})

    def reset(self):
        self.model = OnlineBarycenterWorldModel()


class SliderTool(BaseMetaTool):
    """
    Tool 10: Column Slider & Transfer Valve Puzzle (vc33).
    Requires: Pure click. Small entity count <= 15, low symmetry < 0.65.
    """
    def __init__(self):
        super().__init__(tool_id=10, name="Slider_ColumnValve")
        self.model = OnlineSliderWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_click == 0.0 or features.has_nav > 0.0 or features.has_reset > 0.0 or features.has_interact > 0.0:
            return 0.0
        if features.spatial_symmetry < 0.65 and features.entity_count_norm <= 0.35:
            return 0.95
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        res = self.model.step(grid, levels_completed)
        if isinstance(res, tuple):
            return (int(res[0]), res[1])
        return (int(res), None)

    def reset(self):
        self.model = OnlineSliderWorldModel()


class IsoSliceTool(BaseMetaTool):
    """
    Tool 11: Isometric Track / Slice-Rotator Permutation Solver (lp85).
    Requires: Pure click. High color count (>=10), high symmetry >= 0.80, switch widgets.
    """
    def __init__(self):
        super().__init__(tool_id=11, name="IsoSlice_TrackRotator")
        self.model = OnlineIsoSliceWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_click == 0.0 or features.has_nav > 0.0 or features.has_reset > 0.0 or features.has_interact > 0.0:
            return 0.0
        if features.spatial_symmetry >= 0.80 and features.color_count_norm >= 0.60 and features.has_switch_widgets > 0.0:
            return 0.96
        return 0.05

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        res = self.model.step(grid, levels_completed)
        if isinstance(res, tuple):
            return (int(res[0]), res[1])
        return (int(res), None)

    def reset(self):
        self.model = OnlineIsoSliceWorldModel()


class PegSolitaireTool(BaseMetaTool):
    """
    Tool 12: Checkers / Peg Solitaire Hop-Over Capture Solver (lf52).
    Requires: Nav + Click + Reset. Moderate colors (<=7), 2D nav.
    """
    def __init__(self):
        super().__init__(tool_id=12, name="PegSolitaire_HopCapture")
        self.model = OnlinePegSolitaireWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_click == 0.0 or features.has_reset == 0.0 or features.has_interact > 0.0:
            return 0.0
        if features.has_1d_nav == 0.0 and features.color_count_norm <= 0.45:
            return 0.98
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        res = self.model.step(grid, levels_completed)
        if isinstance(res, tuple):
            return (int(res[0]), res[1])
        return (int(res), None)

    def reset(self):
        self.model = OnlinePegSolitaireWorldModel()


class ChipSocketTool(BaseMetaTool):
    """
    Tool 13: Microchip Socket & Visual Program Synthesizer (sb26).
    Requires: Interact + Click + Reset, NO nav.
    """
    def __init__(self):
        super().__init__(tool_id=13, name="ChipSocket_Microcode")
        self.model = OnlineChipSocketWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_interact == 0.0 or features.has_click == 0.0 or features.has_reset == 0.0 or features.has_nav > 0.0:
            return 0.0
        return 0.98

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id, act_data = self.model.step(grid, levels_completed)
        return (int(act_id), act_data if act_data else None)

    def reset(self):
        self.model = OnlineChipSocketWorldModel()


class GlyphDialTool(BaseMetaTool):
    """
    Tool 14: Sokoban Glyph Dial Step-Budget Solver (ls20).
    Requires: Nav only. Moderate entity count (<=25).
    """
    def __init__(self):
        super().__init__(tool_id=14, name="GlyphDial_StepBudget")
        self.model = OnlineGlyphDialWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_click > 0.0 or features.has_interact > 0.0 or features.has_reset > 0.0:
            return 0.0
        if features.entity_count_norm <= 0.50:
            return 0.95
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id = self.model.step(grid, levels_completed)
        return (int(act_id), None)

    def reset(self):
        self.model = OnlineGlyphDialWorldModel()


class FluidSpillTool(BaseMetaTool):
    """
    Tool 15: Fluid Spill & Deflector Prism Routing Solver (sp80).
    Requires: Nav + Interact + Click, NO reset. Low entities <= 10, switch widgets.
    """
    def __init__(self):
        super().__init__(tool_id=15, name="FluidSpill_PrismRouting")
        self.model = OnlineFluidSpillWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_interact == 0.0 or features.has_click == 0.0 or features.has_reset > 0.0:
            return 0.0
        if features.has_switch_widgets > 0.0 and features.entity_count_norm <= 0.20:
            return 0.96
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        res = self.model.step(grid, levels_completed)
        if isinstance(res, tuple):
            return (int(res[0]), res[1] if res[1] else None)
        return (int(res), None)

    def reset(self):
        self.model = OnlineFluidSpillWorldModel()


class RoboticArmTool(BaseMetaTool):
    """
    Tool 16: Robotic Arm Kinematics & Sliding Obstacles Solver (s5i5).
    Requires: Pure click. Moderate color count (<10), moderate entities, high symmetry >= 0.80, switch widgets.
    """
    def __init__(self):
        super().__init__(tool_id=16, name="RoboticArm_Kinematics")
        self.model = OnlineRoboticArmWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_click == 0.0 or features.has_nav > 0.0 or features.has_reset > 0.0 or features.has_interact > 0.0:
            return 0.0
        if features.spatial_symmetry >= 0.80 and features.color_count_norm < 0.60 and features.has_switch_widgets > 0.0:
            return 0.95
        return 0.05

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_val, act_data = self.model.step(grid, levels_completed)
        return (int(act_val), act_data)

    def reset(self):
        self.model = OnlineRoboticArmWorldModel()


class StealthEvasionTool(BaseMetaTool):
    """
    Tool 17: Dynamic Predator Evasion & Exit Flow Solver (tu93).
    Requires: Nav only. High entity count (>=40), high symmetry >= 0.70.
    """
    def __init__(self):
        super().__init__(tool_id=17, name="StealthEvasion_Predator")
        self.model = OnlineStealthEvasionWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_click > 0.0 or features.has_interact > 0.0 or features.has_reset > 0.0:
            return 0.0
        if features.entity_count_norm > 0.50 and features.spatial_symmetry >= 0.75:
            return 0.95
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id = self.model.step(grid, levels_completed)
        return (int(act_id), None)

    def reset(self):
        self.model = OnlineStealthEvasionWorldModel()


class CraneTool(BaseMetaTool):
    """
    Tool 18: Telescopic Crane Arm Block-Pushing Solver (sk48).
    Requires: Nav + Click + Reset. High color count (>=8 colors, color_count_norm >= 0.50).
    """
    def __init__(self):
        super().__init__(tool_id=18, name="Crane_TelescopicArm")
        self.model = OnlineCraneWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_click == 0.0 or features.has_reset == 0.0 or features.has_interact > 0.0:
            return 0.0
        if features.has_1d_nav == 0.0 and features.color_count_norm >= 0.50:
            return 0.98
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        res = self.model.step(grid, levels_completed)
        if isinstance(res, tuple):
            return (int(res[0]), res[1])
        return (int(res), None)

    def reset(self):
        self.model = OnlineCraneWorldModel()


class IceCurlingTool(BaseMetaTool):
    """
    Tool 19: Ice-Curling Frictionless Slider Puzzle Solver (ka59).
    Requires: Nav + Click. Has receptors, high symmetry >= 0.85.
    """
    def __init__(self):
        super().__init__(tool_id=19, name="Curling_FrictionlessSlide")
        self.model = OnlineCurlingWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_click == 0.0 or features.has_interact > 0.0 or features.has_reset > 0.0:
            return 0.0
        if features.has_receptors > 0.0 and features.spatial_symmetry >= 0.85:
            return 0.98
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        res = self.model.step(grid, levels_completed)
        if isinstance(res, tuple):
            return (int(res[0]), res[1])
        return (int(res), None)

    def reset(self):
        self.model = OnlineCurlingWorldModel()


class TimeCloneTool(BaseMetaTool):
    """
    Tool 20: Time-Loop Clone Replay Navigation Solver (g50t).
    Requires: Nav + Interact, NO click, NO reset. Moderate symmetry (<0.85), low entities.
    """
    def __init__(self):
        super().__init__(tool_id=20, name="TimeClone_EchoLoop")
        self.model = OnlineTimeCloneWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_interact == 0.0 or features.has_click > 0.0 or features.has_reset > 0.0:
            return 0.0
        if features.spatial_symmetry < 0.85 and features.has_goal_tile == 0.0 and features.entity_count_norm <= 0.35:
            return 0.96
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id = self.model.step(grid, levels_completed)
        return (int(act_id), None)

    def reset(self):
        self.model = OnlineTimeCloneWorldModel()


class StateTransformTool(BaseMetaTool):
    """
    Tool 21: Geometric Silhouette Alignment & State Transformation Solver (re86).
    Requires: Nav + Interact, NO click, NO reset. Has goal tile (exit destination).
    """
    def __init__(self):
        super().__init__(tool_id=21, name="StateTransform_Silhouette")
        self.model = OnlineStateTransformWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_interact == 0.0 or features.has_click > 0.0 or features.has_reset > 0.0:
            return 0.0
        if features.has_goal_tile > 0.0 and features.spatial_symmetry >= 0.85:
            return 0.99
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id = self.model.step(grid, levels_completed)
        return (int(act_id), None)

    def reset(self):
        self.model = OnlineStateTransformWorldModel()


class PairMatchTool(BaseMetaTool):
    """
    Tool 22: Symmetric Mirror-Pair Matching & Block Manipulation Solver (m0r0).
    Requires: Nav + Interact + Click, NO reset. Very low color count (<=4) & entities (<=5).
    """
    def __init__(self):
        super().__init__(tool_id=22, name="PairMatch_MirrorCursor")
        self.model = OnlinePairMatchWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_interact == 0.0 or features.has_click == 0.0 or features.has_reset > 0.0:
            return 0.0
        if features.color_count_norm <= 0.30 and features.entity_count_norm <= 0.15 and features.has_switch_widgets == 0.0:
            return 0.96
        return 0.05

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id, act_data = self.model.step(grid, levels_completed)
        return (int(act_id), act_data if act_data else None)

    def reset(self):
        self.model = OnlinePairMatchWorldModel()


class SpellcastTool(BaseMetaTool):
    """
    Tool 23: Runic Spellcaster Maze & Rune Invocations Solver (sc25).
    Requires: Nav + Click, NO interact, NO reset. No switch widgets, no receptors.
    """
    def __init__(self):
        super().__init__(tool_id=23, name="Spellcast_RunePad")
        self.model = OnlineSpellcastWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_click == 0.0 or features.has_interact > 0.0 or features.has_reset > 0.0:
            return 0.0
        if features.has_switch_widgets == 0.0 and features.has_receptors == 0.0 and features.has_goal_tile == 0.0:
            return 0.96
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id, act_data = self.model.step(grid, levels_completed)
        return (int(act_id), act_data)

    def reset(self):
        self.model = OnlineSpellcastWorldModel()


class TangramTool(BaseMetaTool):
    """
    Tool 24: Connector-Pin Polyomino Tiling Solver (cn04).
    Requires: Nav + Interact + Click, NO reset. No switch widgets, high symmetry >= 0.75.
    """
    def __init__(self):
        super().__init__(tool_id=24, name="Tangram_PolyominoTiling")
        self.model = OnlineTangramWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_interact == 0.0 or features.has_click == 0.0 or features.has_reset > 0.0:
            return 0.0
        if features.spatial_symmetry >= 0.75 and features.has_switch_widgets == 0.0 and features.has_receptors == 0.0 and features.has_goal_tile == 0.0:
            return 0.98
        return 0.10

    def is_executing_plan(self) -> bool:
        return self.model.plan_idx < len(self.model.plan)

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id, act_data = self.model.step(grid, levels_completed)
        return (int(act_id), act_data if act_data else None)

    def reset(self):
        self.model = OnlineTangramWorldModel()


class Register1DTool(BaseMetaTool):
    """
    Tool 25: 1D Cipher Register & Grammar Production Solver (tr87).
    Requires: Nav only. Low symmetry < 0.55, 1D register row layout.
    """
    def __init__(self):
        super().__init__(tool_id=25, name="Register1D_CipherGrammar")
        self.model = OnlineRegister1DWorldModel()

    def can_handle(self, features: AffordanceFeatures) -> float:
        if features.has_nav == 0.0 or features.has_click > 0.0 or features.has_interact > 0.0 or features.has_reset > 0.0:
            return 0.0
        if features.spatial_symmetry < 0.55 and features.entity_count_norm >= 0.70:
            return 0.98
        return 0.05

    def is_executing_plan(self) -> bool:
        return self.model.is_executing_plan()

    def step(self, grid: np.ndarray, levels_completed: int, *args, **kwargs) -> Tuple[int, Optional[Dict[str, Any]]]:
        grid = np.asarray(grid)
        while grid.ndim > 2:
            grid = grid[-1]
        act_id = self.model.step(grid, levels_completed)
        return (int(act_id), None)

    def reset(self):
        self.model.reset()


def get_default_tools() -> List[BaseMetaTool]:
    """Returns the standardized palette of 26 meta-tools."""
    return [
        LightsOutTool(),
        GravityPlatformerTool(),
        SokobanTool(),
        ActiveBasketTool(),
        TractorBallTool(),
        MirrorPuzzleTool(),
        RobotAssemblyTool(),
        DomainAgnosticNavTool(),
        HybridNavClickTool(),
        BarycenterTool(),
        SliderTool(),
        IsoSliceTool(),
        PegSolitaireTool(),
        ChipSocketTool(),
        GlyphDialTool(),
        FluidSpillTool(),
        RoboticArmTool(),
        StealthEvasionTool(),
        CraneTool(),
        IceCurlingTool(),
        TimeCloneTool(),
        StateTransformTool(),
        PairMatchTool(),
        SpellcastTool(),
        TangramTool(),
        Register1DTool()
    ]
