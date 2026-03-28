"""
TrustDeskEnv — Core OpenEnv-compatible environment.

Implements the standard:
    reset(task_id) -> ObservationModel
    step(action)   -> StepResponse (observation, reward, done, info)
    state()        -> StateModel

The environment is deterministic given a task_id and fixed action sequence.

Step info now includes a rich StepTrace with:
    - decision_trace: deterministic reasoning strings
    - violations: list of policy rule failures
    - subgoals_completed: cumulative episode progress
    - policy_checks: audit-trail of active policy conditions
    - loop_detected / invalid_action flags
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from app.models import (
    Action,
    ActionType,
    ObservationModel,
    PolicyChecks,
    RewardModel,
    StateModel,
    StepResponse,
    StepTrace,
)
from app.tasks import Task, get_task
from app.policies import (
    contains_unsafe_promise,
    contains_verification_language,
    ACTIONS_REQUIRING_CLASSIFICATION_FIRST,
)
from app.utils import clamp


# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------

ALL_ACTION_TYPES: List[str] = [a.value for a in ActionType]


# ---------------------------------------------------------------------------
# TrustDeskEnv
# ---------------------------------------------------------------------------


class TrustDeskEnv:
    """
    Policy-aware customer support and trust & safety operations environment.

    An AI agent must handle realistic enterprise support tickets by taking
    multi-step structured actions. Every step() returns rich explainability
    metadata in `info` alongside the standard observation, reward, and done flag.

    Episode lifecycle:
        1. reset(task_id)                 -> initial ObservationModel
        2. step(action) [repeated]        -> StepResponse
        3. state()                        -> StateModel (anytime, for inspection)
        4. grade (via /grader endpoint)   -> GraderOutput with score + failures
    """

    def __init__(self) -> None:
        self._task: Optional[Task] = None
        self._state: Optional[StateModel] = None
        # Running list of all subgoals achieved across the episode
        self._episode_subgoals: List[str] = []

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> ObservationModel:
        """
        Initialize a new episode for the given task.

        Args:
            task_id: Registered task identifier, or None for the default easy task.

        Returns:
            Initial ObservationModel visible to the agent.
        """
        self._task = get_task(task_id)
        task = self._task
        self._episode_subgoals = []

        self._state = StateModel(
            task_id=task.task_id,
            ticket_id=task.ticket_id,
            difficulty=task.difficulty,
            steps_taken=0,
            remaining_steps=task.step_budget,
            cumulative_reward=0.0,
        )
        return self._build_observation()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: Action) -> StepResponse:
        """
        Apply a structured action and advance the environment.

        Returns StepResponse containing:
            - observation: updated ObservationModel
            - reward: dense RewardModel with breakdown and penalties
            - done: whether episode is finished
            - info: StepTrace dict with decision_trace, violations,
                    policy_checks, and subgoal tracking
        """
        if self._task is None or self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        state = self._state
        task = self._task

        # ---- Step budget ----
        state.steps_taken += 1
        state.remaining_steps = max(0, task.step_budget - state.steps_taken)

        reward_value: float = 0.0
        score_breakdown: Dict[str, float] = {}
        penalties: Dict[str, float] = {}
        subgoals: List[str] = []
        violations: List[str] = []
        decision_trace: List[str] = []

        action_type = action.action_type
        if hasattr(action_type, "value"):
            action_type = action_type.value

        # ---- Build policy_checks for this step ----
        policy_checks = self._build_policy_checks(task, state)
        expected_outcome = task.expected_outcome

        # ---- Loop / repeat detection ----
        loop_detected = False
        if action_type == state.last_action_type:
            state.loop_count += 1
            loop_penalty = -0.05
            penalties["loop_penalty"] = loop_penalty
            reward_value += loop_penalty
            loop_detected = True
            violations.append("repeated_action_loop")
            decision_trace.append(
                f"Action '{action_type}' repeated consecutively — loop penalty applied."
            )

        state.last_action_type = action_type

        # ---- Ordering constraint: classification must precede certain actions ----
        order_invalid = False
        if action_type in ACTIONS_REQUIRING_CLASSIFICATION_FIRST and not state.classified:
            invalid_penalty = -0.08
            penalties["order_violation"] = invalid_penalty
            reward_value += invalid_penalty
            state.invalid_action_count += 1
            order_invalid = True
            violations.append("classification_required_first")
            decision_trace.append(
                f"'{action_type}' called before classify_ticket — ordering constraint violated."
            )

        # ---- Adversarial check: security flags active but attempting non-security action ----
        if (
            task.security_flags
            and not state.risk_detected
            and action_type in ("assign_team", "offer_resolution", "mark_resolved")
        ):
            penalty_v = -0.10
            penalties["adversarial_security_skip"] = penalty_v
            reward_value += penalty_v
            violations.append("security_risk_not_assessed")
            decision_trace.append(
                f"Security flags are active ({task.security_flags}) but risk has not been "
                f"detected before '{action_type}'. This is a critical ordering failure."
            )
            # Track that security was ignored in state for grader
            state.security_ignored = True

        # ---- Dispatch action ----
        if action_type == ActionType.CLASSIFY_TICKET.value:
            r, s, g, dt = self._handle_classify(action, state, task)
        elif action_type == ActionType.SET_PRIORITY.value:
            r, s, g, dt = self._handle_set_priority(action, state, task)
        elif action_type == ActionType.DETECT_RISK.value:
            r, s, g, dt = self._handle_detect_risk(action, state, task)
        elif action_type == ActionType.ASSIGN_TEAM.value:
            r, s, g, dt = self._handle_assign_team(action, state, task)
        elif action_type == ActionType.REQUEST_VERIFICATION.value:
            r, s, g, dt = self._handle_request_verification(action, state, task)
        elif action_type == ActionType.OFFER_RESOLUTION.value:
            r, s, g, dt = self._handle_offer_resolution(action, state, task)
        elif action_type == ActionType.ESCALATE.value:
            r, s, g, dt = self._handle_escalate(action, state, task)
        elif action_type == ActionType.DRAFT_REPLY.value:
            r, s, g, dt = self._handle_draft_reply(action, state, task)
        elif action_type == ActionType.MARK_RESOLVED.value:
            r, s, g, dt = self._handle_mark_resolved(action, state, task)
        else:
            r, s, g, dt = -0.1, {"invalid_action": -0.1}, [], [
                f"Unknown action_type '{action_type}' — not in action space."
            ]
            state.invalid_action_count += 1
            violations.append(f"unknown_action:{action_type}")

        reward_value += r
        score_breakdown.update(s)
        subgoals.extend(g)
        decision_trace.extend(dt)

        # Add newly completed subgoals to episode-level list
        for sg in g:
            if sg not in self._episode_subgoals:
                self._episode_subgoals.append(sg)

        # ---- Cumulative reward ----
        state.cumulative_reward = clamp(state.cumulative_reward + reward_value, -5.0, 5.0)

        # ---- Episode done? ----
        done = state.resolved or state.remaining_steps <= 0

        # ---- Build reward model ----
        reward_model = RewardModel(
            reward=reward_value,
            score_breakdown=score_breakdown,
            penalties=penalties,
            subgoals_achieved=subgoals,
            cumulative_reward=state.cumulative_reward,
        )

        # ---- Build StepTrace for info ----
        trace = StepTrace(
            action_type=action_type,
            score_breakdown=score_breakdown,
            violations=violations,
            subgoals_completed=list(self._episode_subgoals),
            decision_trace=decision_trace,
            loop_detected=loop_detected,
            invalid_action=(order_invalid or bool(violations)),
            policy_checks=policy_checks,
        )

        info: Dict[str, Any] = trace.model_dump()
        if done and not state.resolved:
            info["termination"] = "step_budget_exhausted"

        obs = self._build_observation()
        return StepResponse(observation=obs, reward=reward_model, done=done, info=info)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> StateModel:
        """Return a deep copy of the current internal state (safe for external reads)."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return copy.deepcopy(self._state)

    # ------------------------------------------------------------------
    # Internal action handlers
    # Each returns: (reward_delta, score_dict, subgoals_list, decision_trace_list)
    # ------------------------------------------------------------------

    def _handle_classify(
        self, action: Action, state: StateModel, task: Task
    ) -> tuple[float, Dict, List, List]:
        if state.classified:
            state.loop_count += 1
            return -0.03, {"reclassify_penalty": -0.03}, [], [
                "Ticket already classified — reclassification penalized."
            ]

        state.classified = True
        expected_cat = task.expected_outcome.get("expected_category", "")
        agent_cat = action.category
        if hasattr(agent_cat, "value"):
            agent_cat = agent_cat.value if agent_cat else None

        state.classification_correct = agent_cat == expected_cat

        if state.classification_correct:
            return 0.15, {"classification": 0.15}, ["classified_correctly"], [
                f"Customer message analyzed → category '{agent_cat}' matches expected "
                f"'{expected_cat}'. Correct classification."
            ]
        else:
            return 0.05, {"classification": 0.05}, ["classified_incorrectly"], [
                f"Category '{agent_cat}' assigned, but expected '{expected_cat}'. "
                f"Partial credit only — incorrect classification will cascade to routing."
            ]

    def _handle_set_priority(
        self, action: Action, state: StateModel, task: Task
    ) -> tuple[float, Dict, List, List]:
        if state.priority_set:
            return -0.03, {"repeat_priority_penalty": -0.03}, [], [
                "Priority already set — repeat action penalized."
            ]

        state.priority_set = True
        expected_pri = task.expected_outcome.get("expected_priority", "medium")
        agent_pri = action.priority
        if hasattr(agent_pri, "value"):
            agent_pri = agent_pri.value if agent_pri else None

        state.priority_correct = agent_pri == expected_pri
        if state.priority_correct:
            return 0.10, {"priority": 0.10}, ["priority_correct"], [
                f"Priority '{agent_pri}' correctly set (expected: '{expected_pri}'). "
                f"Tier={task.customer_tier} and risk signals were correctly weighed."
            ]

        adjacent = {
            "low": {"medium"}, "medium": {"low", "high"},
            "high": {"medium", "critical"}, "critical": {"high"}
        }
        if agent_pri in adjacent.get(expected_pri, set()):
            return 0.05, {"priority": 0.05}, ["priority_partial"], [
                f"Priority '{agent_pri}' is adjacent to expected '{expected_pri}' — partial credit."
            ]
        return 0.0, {"priority": 0.0}, [], [
            f"Priority '{agent_pri}' incorrect (expected '{expected_pri}'). "
            f"Customer tier '{task.customer_tier}' and security context were not weighed correctly."
        ]

    def _handle_detect_risk(
        self, action: Action, state: StateModel, task: Task
    ) -> tuple[float, Dict, List, List]:
        if state.risk_detected:
            return -0.03, {"repeat_risk_penalty": -0.03}, [], [
                "Risk already assessed — repeat detect_risk penalized."
            ]

        state.risk_detected = True
        # If we get here, security was not 'ignored' (agent did assess it)
        state.security_ignored = False
        expected_risk = task.expected_outcome.get("expected_risk", "none")
        agent_risk = action.risk_label
        if hasattr(agent_risk, "value"):
            agent_risk = agent_risk.value if agent_risk else None

        state.risk_correct = agent_risk == expected_risk

        if expected_risk == "critical" and agent_risk != "critical":
            state.risk_correct = False
            return -0.05, {"risk_miss_critical": -0.05}, [], [
                f"CRITICAL risk signal present (flags: {task.security_flags}) but agent "
                f"assessed risk as '{agent_risk}'. Missing critical security risk is a "
                f"high-severity failure — heavy penalty applied."
            ]

        if state.risk_correct:
            return 0.12, {"risk_detection": 0.12}, ["risk_detected_correctly"], [
                f"Risk assessed as '{agent_risk}' — matches expected '{expected_risk}'. "
                f"Security signals correctly evaluated: {task.security_flags or 'none'}."
            ]
        return 0.02, {"risk_detection": 0.02}, [], [
            f"Risk label '{agent_risk}' does not match expected '{expected_risk}'. "
            f"Partial credit only — incorrect risk level may lead to wrong team routing."
        ]

    def _handle_assign_team(
        self, action: Action, state: StateModel, task: Task
    ) -> tuple[float, Dict, List, List]:
        if state.team_assigned:
            return -0.03, {"repeat_team_penalty": -0.03}, [], [
                "Team already assigned — repeat action penalized."
            ]

        state.team_assigned = True
        expected_team = task.expected_outcome.get("expected_team", "general_support")
        agent_team = action.team
        if hasattr(agent_team, "value"):
            agent_team = agent_team.value if agent_team else None

        state.team_correct = agent_team == expected_team

        # Adversarial: wrong team on a security task (billing/general on security case)
        if (
            expected_team in ("trust_and_safety", "account_security")
            and agent_team in ("billing", "general_support", "refunds")
        ):
            state.wrong_team_on_security = True
            state.policy_violation_count += 1
            return -0.12, {"wrong_team_security": -0.12}, [], [
                f"ADVERSARIAL FAILURE: Security task requires '{expected_team}' but agent "
                f"assigned '{agent_team}'. Routing billing/general team to a security incident "
                f"violates triage policy and delays remediation."
            ]

        if state.team_correct:
            return 0.12, {"team_assignment": 0.12}, ["team_correct"], [
                f"Team '{agent_team}' correctly assigned. "
                f"Routing logic correctly mapped category + risk → team."
            ]
        return 0.0, {"team_assignment": 0.0}, [], [
            f"Team '{agent_team}' assigned but expected '{expected_team}'. "
            f"Incorrect routing — ticket may not be handled by the right specialist."
        ]

    def _handle_request_verification(
        self, action: Action, state: StateModel, task: Task
    ) -> tuple[float, Dict, List, List]:
        if state.verification_requested:
            return -0.03, {"repeat_verification": -0.03}, [], [
                "Verification already requested — repeat action penalized."
            ]

        state.verification_requested = True
        required = task.expected_outcome.get("verification_required", False)

        if required:
            return 0.12, {"verification": 0.12}, ["verification_requested"], [
                "Identity verification correctly requested. "
                "This is required before any resolution action in this scenario "
                "(account is locked / security flags active)."
            ]
        return -0.03, {"unnecessary_verification": -0.03}, [], [
            "Verification requested but not required for this scenario. "
            "Minor penalty — wastes SLA budget."
        ]

    def _handle_offer_resolution(
        self, action: Action, state: StateModel, task: Task
    ) -> tuple[float, Dict, List, List]:
        expected_outcome = task.expected_outcome

        # Adversarial: resolution in security scenario without verification
        if expected_outcome.get("verification_required") and not state.verification_requested:
            state.policy_violation_count += 1
            state.resolved_before_verification = True
            return -0.15, {"premature_resolution_penalty": -0.15}, [], [
                "POLICY VIOLATION: offer_resolution called before request_verification. "
                "Security policy mandates identity verification prior to any resolution "
                "on accounts with active security flags."
            ]

        agent_res = action.resolution_code
        if hasattr(agent_res, "value"):
            agent_res = agent_res.value if agent_res else None

        expected_code = expected_outcome.get("expected_resolution_code")
        allowed = task.allowed_resolutions
        disallowed_res = expected_outcome.get("disallowed_resolutions", [])

        if agent_res in disallowed_res:
            state.policy_violation_count += 1
            return -0.15, {"disallowed_resolution": -0.15}, [], [
                f"POLICY VIOLATION: Resolution '{agent_res}' is explicitly disallowed "
                f"for this scenario. Disallowed codes: {disallowed_res}."
            ]

        if agent_res == expected_code:
            state.resolution_correct = True
            return 0.15, {"resolution": 0.15}, ["resolution_correct"], [
                f"Resolution '{agent_res}' is the ideal outcome for this scenario."
            ]
        if agent_res in allowed:
            state.resolution_correct = True
            return 0.10, {"resolution": 0.10}, ["resolution_allowed"], [
                f"Resolution '{agent_res}' is an acceptable alternative "
                f"(ideal: '{expected_code}')."
            ]
        return 0.0, {"resolution": 0.0}, [], [
            f"Resolution '{agent_res}' is neither optimal nor explicitly disallowed. "
            f"No credit awarded."
        ]

    def _handle_escalate(
        self, action: Action, state: StateModel, task: Task
    ) -> tuple[float, Dict, List, List]:
        state.escalated = True
        required = task.expected_outcome.get("escalation_required", False)
        reason = action.escalation_reason or ""
        state.escalation_justified = bool(reason.strip()) and required

        if required and state.escalation_justified:
            return 0.10, {"escalation": 0.10}, ["escalated_correctly"], [
                f"Escalation correctly triggered. Reason provided: '{reason[:80]}'. "
                f"Case complexity justifies specialist involvement."
            ]
        if required and not reason.strip():
            return -0.03, {"escalation_no_reason": -0.03}, [], [
                "Escalation triggered but no justification provided. "
                "Support SOP requires a reason to escalate."
            ]
        if not required:
            return -0.05, {"unnecessary_escalation": -0.05}, [], [
                "Escalation not required for this scenario — ticket could be resolved "
                "at tier-1. Unnecessary escalation wastes specialist capacity."
            ]
        return 0.0, {}, [], []

    def _handle_draft_reply(
        self, action: Action, state: StateModel, task: Task
    ) -> tuple[float, Dict, List, List]:
        if state.reply_drafted:
            return -0.03, {"repeat_reply_penalty": -0.03}, [], [
                "Reply already drafted — sending multiple replies penalized."
            ]

        state.reply_drafted = True
        message = action.message or ""

        if not message.strip():
            return -0.05, {"empty_reply": -0.05}, [], [
                "Empty reply drafted — no customer-facing message provided."
            ]

        unsafe = contains_unsafe_promise(message)
        needs_verification = task.expected_outcome.get("verification_required", False)

        if unsafe:
            state.reply_safe = False
            state.policy_violation_count += 1
            state.unsafe_actions.append("unsafe_draft_reply")
            return -0.10, {"unsafe_reply": -0.10}, [], [
                "UNSAFE REPLY: Message contains a promise that violates policy "
                "(e.g. committing to refund/cancellation/restoration before verification). "
                "This creates legal/compliance liability."
            ]

        if needs_verification and not contains_verification_language(message):
            state.reply_safe = False
            return 0.03, {"reply_missing_verification_language": 0.03}, [], [
                "Reply drafted but lacks required verification language. "
                "In security scenarios, the reply must instruct the customer to verify identity. "
                "Partial credit only."
            ]

        state.reply_safe = True
        return 0.10, {"reply_drafted": 0.10}, ["reply_drafted_safely"], [
            "Reply drafted safely — no unsafe promises, policy-compliant language used."
        ]

    def _handle_mark_resolved(
        self, action: Action, state: StateModel, task: Task
    ) -> tuple[float, Dict, List, List]:
        expected_outcome = task.expected_outcome

        # Adversarial: resolving without verification in a security scenario
        if expected_outcome.get("verification_required") and not state.verification_requested:
            state.policy_violation_count += 1
            state.resolved_before_verification = True
            state.resolved = True
            return -0.20, {"resolve_before_verification": -0.20}, [], [
                "CRITICAL POLICY VIOLATION: mark_resolved called without prior "
                "request_verification. On accounts with active security flags, "
                "resolution before identity verification is forbidden. "
                "This is the highest-severity penalty in this environment."
            ]

        if not state.classified:
            return -0.05, {"resolve_before_classify": -0.05}, [], [
                "Attempted to resolve ticket before classification. "
                "Ticket must be classified before closing."
            ]

        agent_res = action.resolution_code
        if hasattr(agent_res, "value"):
            agent_res = agent_res.value if agent_res else None

        disallowed_res = expected_outcome.get("disallowed_resolutions", [])
        if agent_res and agent_res in disallowed_res:
            state.policy_violation_count += 1
            state.resolved = True
            return -0.15, {"disallowed_resolution_on_close": -0.15}, [], [
                f"POLICY VIOLATION: Resolution code '{agent_res}' is disallowed "
                f"(disallowed: {disallowed_res}). Closing with wrong disposition."
            ]

        allowed = task.allowed_resolutions
        expected_code = expected_outcome.get("expected_resolution_code")
        if agent_res == expected_code:
            state.resolution_correct = True
        elif agent_res in allowed:
            state.resolution_correct = True

        state.resolved = True
        if state.resolution_correct and state.reply_drafted:
            return 0.15, {"resolved_correctly": 0.15}, ["resolved_correctly"], [
                f"Episode resolved correctly with code '{agent_res}' and reply drafted. "
                f"Full resolution credit awarded."
            ]
        elif state.resolution_correct:
            return 0.08, {"resolved": 0.08}, ["resolved"], [
                f"Resolved with correct code '{agent_res}' but no customer reply was drafted. "
                f"Partial credit."
            ]
        return 0.03, {"resolved_incorrectly": 0.03}, [], [
            f"Resolved with code '{agent_res}' (expected '{expected_code}'). "
            f"Incorrect disposition but episode closed."
        ]

    # ------------------------------------------------------------------
    # Policy checks builder — deterministic per step
    # ------------------------------------------------------------------

    def _build_policy_checks(self, task: Task, state: StateModel) -> PolicyChecks:
        """Build a PolicyChecks object reflecting current task + state conditions."""
        pc = task.policy_context
        return PolicyChecks(
            refund_window_ok=self._check_refund_window(pc),
            verification_required=bool(task.expected_outcome.get("verification_required", False)),
            security_flags_active=bool(task.security_flags),
            prior_goodwill_refund_used=pc.get("goodwill_refund_used"),
            classification_required_first=not state.classified,
        )

    @staticmethod
    def _check_refund_window(policy_context: dict) -> Optional[bool]:
        """Return True/False if refund window data is available, else None."""
        days = policy_context.get("days_since_renewal")
        window = policy_context.get("refund_window_days")
        if days is not None and window is not None:
            return int(days) <= int(window)
        return None

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> ObservationModel:
        task = self._task
        state = self._state

        # Prune single-use actions that are already done
        available = list(ALL_ACTION_TYPES)
        if state.classified:
            available = [a for a in available if a != ActionType.CLASSIFY_TICKET.value]
        if state.priority_set:
            available = [a for a in available if a != ActionType.SET_PRIORITY.value]
        if state.risk_detected:
            available = [a for a in available if a != ActionType.DETECT_RISK.value]
        if state.team_assigned:
            available = [a for a in available if a != ActionType.ASSIGN_TEAM.value]
        if state.resolved:
            available = []

        status_parts = []
        if not state.classified:
            status_parts.append("Ticket unclassified.")
        else:
            status_parts.append(
                f"Ticket classified (correct={state.classification_correct}). "
                f"Priority set: {state.priority_set}."
            )
        if state.risk_detected:
            status_parts.append(f"Risk assessed (correct={state.risk_correct}).")
        if state.team_assigned:
            status_parts.append(f"Team assigned (correct={state.team_correct}).")
        if state.verification_requested:
            status_parts.append("Verification requested.")
        if state.reply_drafted:
            status_parts.append(f"Reply drafted (safe={state.reply_safe}).")
        if state.escalated:
            status_parts.append(f"Escalated (justified={state.escalation_justified}).")
        if state.resolved:
            status_parts.append("RESOLVED.")
        current_status = " ".join(status_parts) or "No actions taken yet."

        return ObservationModel(
            task_id=task.task_id,
            ticket_id=task.ticket_id,
            customer_message=task.customer_message,
            customer_tier=task.customer_tier,
            account_status=task.account_status,
            policy_context=task.policy_context,
            security_flags=task.security_flags,
            prior_history=task.prior_history,
            steps_taken=state.steps_taken,
            remaining_steps=state.remaining_steps,
            available_actions=available,
            current_status=current_status,
        )


# ---------------------------------------------------------------------------
# Module-level singleton — used by FastAPI
# ---------------------------------------------------------------------------

_env_instance: Optional[TrustDeskEnv] = None


def get_env() -> TrustDeskEnv:
    """Return (or create) the module-level singleton TrustDeskEnv instance."""
    global _env_instance
    if _env_instance is None:
        _env_instance = TrustDeskEnv()
    return _env_instance
