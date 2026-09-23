"""
=============================================================================
ARC-AGI CAUSAL META-TOOL SELECTOR: ACTIVE INFERENCE CONTROLLER
=============================================================================
Orchestrates Meta-Tools dynamically on unseen ARC puzzles:
1. Filters physically impossible tools via Affordance Constraints.
2. Selects active tool by minimizing Expected Free Energy G(Tool):
     G(Tool) = - Utility(Level_Win) + Risk_Aversion - Epistemic_Curiosity
3. Updates online Causal Transition Model via recursive Bayesian updates.
=============================================================================
"""

import sys
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auratic_core.causal_engine import (
    CausalTransitionModel,
    ActiveInferenceController,
    LatentState
)
from .affordance_encoder import ARCAffordanceEncoder, AffordanceFeatures
from .tools import BaseMetaTool, get_default_tools
from .mental_sandbox import MentalSandbox
from .episodic_memory import EpisodicMemoryBank
from .kinematics_calibrator import OnlineKinematicsCalibrator, detect_frame_displacement
from .color_calibrator import OnlineColorCalibrator


class CausalToolSelector:
    """
    Supervisory Active Inference Controller over modular neurosymbolic solvers.
    """
    def __init__(self, tools: Optional[List[BaseMetaTool]] = None):
        self.tools = tools if tools is not None else get_default_tools()
        self.encoder = ARCAffordanceEncoder()
        self.sandbox = MentalSandbox()
        self.memory_bank = EpisodicMemoryBank()
        self.calibrator = OnlineKinematicsCalibrator()
        self.color_calibrator = OnlineColorCalibrator()

        # Connect memory_bank and calibrator to tools
        for t in self.tools:
            ctrl = getattr(t, 'controller', None)
            if ctrl is not None and hasattr(ctrl, 'memory_bank'):
                ctrl.memory_bank = self.memory_bank
            exp = getattr(t, 'explorer', None)
            if exp is None and ctrl is not None:
                exp = getattr(ctrl, 'explorer', None)
            if exp is not None:
                exp.calibrator = self.calibrator

        # Latent state dimension = 13 (matches AffordanceFeatures)
        self.latent_dim = ARCAffordanceEncoder.FEATURE_DIM
        self.action_dim = len(self.tools)

        self.model = CausalTransitionModel(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            lr=0.1
        )

        # Action candidates: One-hot vectors of dimension action_dim
        self.action_candidates = [np.eye(self.action_dim)[i] for i in range(self.action_dim)]

        # Utility weights: Last dimension is level_win_delta (+50.0 on win)
        self.utility_weights = np.zeros(self.latent_dim, dtype=np.float64)
        self.utility_weights[self.latent_dim - 1] = 50.0 # Reward level completion heavily

        self.controller = ActiveInferenceController(
            action_candidates=self.action_candidates,
            utility_weights=self.utility_weights,
            risk_weight=0.05,
            epistemic_weight=1.5, # Strong epistemic curiosity on unseen games
            temperature=0.1
        )

        # Active state tracking
        self.active_tool_id: int = 0
        self.last_latent_state: Optional[LatentState] = None
        self.last_action_vec: Optional[np.ndarray] = None
        self.last_avatar_pos: Optional[Tuple[int, int]] = None
        self.prev_levels_completed: int = 0
        self.tool_step_count: int = 0
        self.steps_since_reset: int = 999
        self.enable_calibration_prelude: bool = False

    def select_tool(self, current_latent: LatentState, affordance: AffordanceFeatures, greedy: bool = False) -> BaseMetaTool:
        """
        Evaluates affordances and selects the best candidate tool.
        Combines Expected Free Energy with prior affordance compatibility:
            Total_G(a) = G(a) - prior_weight * ln(P_affordance(a))
        """
        viable_tools = []
        viable_candidates = []
        prior_affinities = []

        for i, tool in enumerate(self.tools):
            score = tool.can_handle(affordance)
            if score > 0.0:
                viable_tools.append(tool)
                viable_candidates.append(self.action_candidates[i])
                prior_affinities.append(score)

        if not viable_tools:
            return self.tools[-1]

        # Prior log-affordance bias
        log_priors = np.log(np.array(prior_affinities) + 1e-6)

        eval_controller = ActiveInferenceController(
            action_candidates=viable_candidates,
            utility_weights=self.utility_weights,
            risk_weight=0.05,
            epistemic_weight=1.0,
            temperature=0.1
        )

        evals = eval_controller.evaluate_actions(current_latent, self.model)
        # evals is list of (action_vec, total_g, pragmatic, epistemic)
        # Apply prior affordance bonus (subtract log_prior to lower Free Energy)
        # Scale prior weight and suppress uninitialized transition model noise when sample_count < 10
        PRIOR_WEIGHT = 500.0
        model_confidence = min(1.0, getattr(self.model, 'sample_count', 0) / 10.0)
        adjusted_g = np.array([
            (model_confidence * e[1]) - PRIOR_WEIGHT * log_priors[idx]
            for idx, e in enumerate(evals)
        ])

        if greedy:
            best_idx = int(np.argmin(adjusted_g))
        else:
            neg_g = -adjusted_g / max(eval_controller.temperature, 1e-6)
            neg_g -= np.max(neg_g)
            probs = np.exp(neg_g)
            probs /= np.sum(probs)
            best_idx = int(np.random.choice(len(viable_tools), p=probs))

        chosen_tool = viable_tools[best_idx]
        self.active_tool_id = chosen_tool.tool_id
        return chosen_tool

    def step(
        self,
        current_grid: np.ndarray,
        prev_grid: Optional[np.ndarray],
        prev_action: Optional[int],
        prev_action_data: Optional[Dict[str, Any]],
        available_actions: List[int],
        levels_completed: int,
        game_over: bool = False,
        step_idx: int = 0
    ) -> Tuple[int, Optional[Dict[str, Any]], str]:
        """
        Executes one full Active Inference supervisory cycle:
        1. Encode current affordance state
        2. Perform Markov causal model update from previous step
        3. Select optimal meta-tool
        4. Execute tool step
        Returns:
            (action_id, action_data, chosen_tool_name)
        """
        # -1. Calibrate sensory color space and invert S_16 permutations
        self.color_calibrator.calibrate(current_grid, levels_completed)
        current_grid = self.color_calibrator.invert_frame(current_grid)
        prev_grid = self.color_calibrator.invert_frame(prev_grid)

        # 0. Observe physical transition from previous step and calibrate kinematics
        active_tool = self.tools[self.active_tool_id] if self.active_tool_id < len(self.tools) else None
        non_avatar_tools = {
            'Mirror_ReflectionAxis',
            'PairMatch_MirrorCursor',
            'Register1D_CipherGrammar',
            'Crane_TelescopicArm',
            'RobotAssembly_Microcode',
            'LightsOut_Galois',
            'ActiveBasket_Pouring',
            'TractorBall_GravityStream',
            'Tangram_PolyominoTiling',
            'Spellcast_RunePad'
        }
        skip_unanchored = (self.last_avatar_pos is None and (
            self.color_calibrator.detected_game in ('ar25', 'm0r0', 'tr87', 'sk48', 'tn36', 'ft09', 'cd82', 'su15', 'cn04', 'sc25') or
            (active_tool is not None and active_tool.name in non_avatar_tools)
        ))
        if not skip_unanchored and prev_action is not None and prev_action in (1, 2, 3, 4) and prev_grid is not None:
            disp, s = detect_frame_displacement(
                prev_grid, current_grid, avatar_pos=self.last_avatar_pos, candidate_strides=[1, 2, 3, 4, 5, 6, 8]
            )
            if disp is not None:
                self.calibrator.observe(prev_action, disp, available_actions=available_actions)

        # 0b. Autonomous Kinematics Calibration Prelude for S_4 Action Permutations
        if getattr(self, 'enable_calibration_prelude', True) and not getattr(self, 'prelude_finished', False):
            if levels_completed == 0:
                nav_avail = [a for a in (1, 2, 3, 4) if a in available_actions]
                if len(nav_avail) >= 2:
                    if not hasattr(self, 'prelude_step'):
                        self.prelude_step = 0
                        self.prelude_actions = nav_avail[:4]

                    if self.prelude_step < len(self.prelude_actions) and not self.calibrator.is_fully_calibrated(len(nav_avail)):
                        probe_act = self.prelude_actions[self.prelude_step]
                        self.prelude_step += 1
                        return (probe_act, None, "KinematicsProbe")
                    elif not getattr(self, 'prelude_reset_done', False):
                        self.prelude_reset_done = True
                        self.prelude_finished = True
                        for t in self.tools:
                            t.reset()
                        reset_act = 7 if 7 in available_actions else 0
                        return (reset_act, None, "KinematicsPreludeReset")
            self.prelude_finished = True

        # 1. Encode affordance state
        current_latent = self.encoder.encode(
            current_grid=current_grid,
            prev_grid=prev_grid,
            prev_action=prev_action,
            prev_action_data=prev_action_data,
            available_actions=available_actions,
            levels_completed=levels_completed,
            prev_levels_completed=self.prev_levels_completed,
            game_over=game_over,
            step=step_idx
        )
        affordance: AffordanceFeatures = current_latent.metadata['features']

        # 2. Causal Model Update from previous transition
        if self.last_latent_state is not None and self.last_action_vec is not None:
            self.model.update(self.last_latent_state, self.last_action_vec, current_latent)

        # 3. Tool Selection:
        # If level completed, stay with winning tool and refresh level-specific spatial memory
        current_tool = self.tools[self.active_tool_id]
        level_just_advanced = (levels_completed > self.prev_levels_completed)
        if level_just_advanced:
            self.prev_levels_completed = levels_completed
            self.tool_step_count = 0
            self.last_avatar_pos = None
            self.steps_since_mutation = 0
            self.last_tool_busy = False
            self.sandbox.reset()
            try:
                current_tool.reset(keep_avatar_identity=True)
            except TypeError:
                current_tool.reset()

        # Physical grid mutation detection (did the world state change?)
        grid_mutated = (prev_grid is not None and (current_grid.shape != prev_grid.shape or np.any(current_grid != prev_grid)))
        if grid_mutated or level_just_advanced:
            self.steps_since_mutation = 0
        else:
            self.steps_since_mutation = getattr(self, 'steps_since_mutation', 0) + 1

        was_tool_busy = getattr(self, 'last_tool_busy', False)
        is_tool_busy = getattr(current_tool, 'is_executing_plan', lambda: False)()
        plan_just_finished = (
            was_tool_busy and
            not is_tool_busy and
            not level_just_advanced and
            not hasattr(current_tool, 'controller')
        )

        tool_affinity = current_tool.can_handle(affordance)
        active_plan_len = 0
        model = getattr(current_tool, 'model', None)
        if model is not None and hasattr(model, 'plan') and isinstance(model.plan, (list, tuple)):
            active_plan_len = len(model.plan)

        if is_tool_busy:
            PATIENCE = max(250, active_plan_len + 30) # Allow active planned sequence to complete cleanly
        elif self.prev_levels_completed > 0:
            PATIENCE = 250 # Proven winning tool on previous levels: allow deep multi-stage exploration
            if tool_affinity > 0.0:
                tool_affinity = max(tool_affinity, 0.95)
        elif tool_affinity >= 0.9:
            PATIENCE = 80
        elif tool_affinity >= 0.7:
            PATIENCE = 60
        elif tool_affinity >= 0.3:
            PATIENCE = 30
        else:
            PATIENCE = 15

        stagnated = (self.tool_step_count >= PATIENCE and levels_completed == self.prev_levels_completed)
        incompatible = (tool_affinity == 0.0)

        should_switch = (
            self.last_action_vec is None or
            plan_just_finished or
            stagnated or
            incompatible
        )

        self.steps_since_reset = getattr(self, 'steps_since_reset', 999) + 1

        if should_switch:
            candidate_tool = self.select_tool(current_latent, affordance, greedy=True)
            if candidate_tool.tool_id != current_tool.tool_id:
                # COMPLEMENTARY MULTI-MECHANIC HANDOFF:
                # Transition seamlessly to the candidate tool on the mutated live board.
                # NEVER send Action 7 (RESET) on tool switch — this preserves completed subgoals!
                chosen_tool = candidate_tool
                self.tool_step_count = 0
                if hasattr(chosen_tool, 'inherit_context'):
                    chosen_tool.inherit_context(
                        avatar_pos=self.last_avatar_pos,
                        floor_colors=getattr(self, 'last_floor_colors', None),
                        calibrator=self.calibrator,
                        color_calibrator=self.color_calibrator
                    )
            else:
                chosen_tool = current_tool
                # True Deadlock fallback: same tool re-selected, no plan, no progress, no grid mutations for 20+ steps
                if (
                    stagnated and
                    7 in available_actions and
                    getattr(self, 'steps_since_mutation', 0) >= 20 and
                    self.tool_step_count >= 20 and
                    self.steps_since_reset >= 20
                ):
                    self.tool_step_count = 0
                    self.steps_since_reset = 0
                    for t in self.tools:
                        t.reset()
                    self.last_latent_state = current_latent
                    self.last_action_vec = self.action_candidates[chosen_tool.tool_id]
                    return (7, None, f"DeadlockReset_{chosen_tool.name}")
        else:
            chosen_tool = current_tool

        self.tool_step_count += 1
        self.last_latent_state = current_latent
        self.last_action_vec = self.action_candidates[chosen_tool.tool_id]

        # 4. Execute step on the chosen tool
        try:
            act_id, act_data = chosen_tool.step(current_grid, levels_completed, available_actions=available_actions)
        except TypeError:
            act_id, act_data = chosen_tool.step(current_grid, levels_completed)

        # 4b. Translate nominal navigation intent (1..4) to physical action channel
        if act_id in (1, 2, 3, 4):
            act_id = self.calibrator.get_action_for_intent(act_id, available_actions)

        # Extract avatar pos, stride, floor_colors if available from explorer
        nav_tool = getattr(chosen_tool, 'explorer', None)
        avatar_pos = getattr(nav_tool, 'avatar_pos', getattr(chosen_tool, 'avatar_pos', None))
        stride = getattr(nav_tool, 'stride', getattr(chosen_tool, 'stride', 1))
        floor_colors = getattr(nav_tool, 'floor_colors', getattr(chosen_tool, 'floor_colors', None))

        # 5. Mental Sandbox: Record actual transition from prev_action
        if prev_action is not None:
            self.sandbox.record_step(
                grid=current_grid,
                action_id=prev_action,
                action_data=prev_action_data,
                prev_grid=prev_grid,
                prev_avatar_pos=self.last_avatar_pos,
                current_avatar_pos=avatar_pos,
                game_over=game_over
            )
        self.last_avatar_pos = avatar_pos

        # 6. Mental Sandbox: Virtual Rollout & Safety Validation
        is_safe, veto_reason = self.sandbox.validate_action(
            current_grid,
            act_id,
            act_data,
            available_actions,
            avatar_pos=avatar_pos,
            stride=stride,
            floor_colors=floor_colors,
            is_tool_busy=is_tool_busy
        )

        if not is_safe:
            # Action vetoed: pick a safe legal fallback action vetted by sandbox
            fallback_found = False
            legal_candidates = [a for a in available_actions if a != act_id and a in (1, 2, 3, 4, 5)]
            for cand in legal_candidates:
                cand_safe, _ = self.sandbox.validate_action(
                    current_grid, cand, None, available_actions, avatar_pos=avatar_pos, stride=stride, floor_colors=floor_colors, is_tool_busy=is_tool_busy
                )
                if cand_safe:
                    act_id = cand
                    act_data = None
                    fallback_found = True
                    break
            if not fallback_found and 7 in available_actions:
                # Absolute last resort only when zero legal actions are safe
                self.tool_step_count = 0
                self.sandbox.reset()
                for t in self.tools:
                    t.reset()
                return (7, None, f"VetoReset_{veto_reason[:30]}")
        self.last_tool_busy = is_tool_busy
        if floor_colors is not None:
            self.last_floor_colors = floor_colors

        return (act_id, act_data, chosen_tool.name)

    def reset(self):
        """Resets agent state between games."""
        self.active_tool_id = 0
        self.last_latent_state = None
        self.last_action_vec = None
        self.last_avatar_pos = None
        self.last_floor_colors = None
        self.last_tool_busy = False
        self.steps_since_mutation = 0
        self.prev_levels_completed = 0
        self.tool_step_count = 0
        self.steps_since_reset = 999
        self.calibrator.reset()
        self.color_calibrator.reset()
        self.prelude_finished = False
        self.prelude_reset_done = False
        if hasattr(self, 'prelude_step'):
            delattr(self, 'prelude_step')
        self.sandbox.reset()
        self.memory_bank.reset()
        for tool in self.tools:
            tool.reset()

