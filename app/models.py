"""
Pydantic models for TrustDeskEnv.
Defines all typed data structures for Observation, Action, Reward, State,
GraderOutput (with failure modes), TaskDetailResponse, and Step explainability.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CustomerTier(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class AccountStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"
    LOCKED = "locked"


class TicketCategory(str, Enum):
    BILLING = "billing"
    REFUND = "refund"
    ACCOUNT_SECURITY = "account_security"
    CANCELLATION = "cancellation"
    TECHNICAL = "technical"
    GENERAL = "general"
    FRAUD = "fraud"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLabel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SupportTeam(str, Enum):
    BILLING = "billing"
    REFUNDS = "refunds"
    ACCOUNT_SECURITY = "account_security"
    TRUST_AND_SAFETY = "trust_and_safety"
    CUSTOMER_SUCCESS = "customer_success"
    GENERAL_SUPPORT = "general_support"
    ESCALATIONS = "escalations"


class ResolutionCode(str, Enum):
    FULL_REFUND = "full_refund"
    PARTIAL_REFUND = "partial_refund"
    CREDIT_ISSUED = "credit_issued"
    NO_REFUND_POLICY = "no_refund_policy"
    ACCOUNT_UNLOCKED = "account_unlocked"
    ACCOUNT_SUSPENDED = "account_suspended"
    DUPLICATE_CHARGE_REVERSED = "duplicate_charge_reversed"
    ESCALATED_TO_SPECIALIST = "escalated_to_specialist"
    VERIFICATION_REQUIRED = "verification_required"
    CANCELLATION_PROCESSED = "cancellation_processed"
    GOODWILL_EXCEPTION = "goodwill_exception"
    POLICY_DECLINED = "policy_declined"
    FRAUD_REVIEW = "fraud_review"


class ActionType(str, Enum):
    CLASSIFY_TICKET = "classify_ticket"
    SET_PRIORITY = "set_priority"
    DETECT_RISK = "detect_risk"
    ASSIGN_TEAM = "assign_team"
    REQUEST_VERIFICATION = "request_verification"
    OFFER_RESOLUTION = "offer_resolution"
    ESCALATE = "escalate"
    DRAFT_REPLY = "draft_reply"
    MARK_RESOLVED = "mark_resolved"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class Action(BaseModel):
    """Structured action taken by the agent in the environment."""

    action_type: ActionType = Field(..., description="The type of action to perform.")
    category: Optional[TicketCategory] = Field(
        None, description="Ticket category (used with classify_ticket)."
    )
    priority: Optional[Priority] = Field(
        None, description="Priority level (used with set_priority)."
    )
    risk_label: Optional[RiskLabel] = Field(
        None, description="Risk assessment label (used with detect_risk)."
    )
    team: Optional[SupportTeam] = Field(
        None, description="Target team to assign ticket (used with assign_team)."
    )
    resolution_code: Optional[ResolutionCode] = Field(
        None, description="Resolution type (used with offer_resolution or mark_resolved)."
    )
    message: Optional[str] = Field(
        None, description="Draft reply or message to customer (used with draft_reply)."
    )
    escalation_reason: Optional[str] = Field(
        None, description="Justification for escalation (used with escalate)."
    )

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class ObservationModel(BaseModel):
    """What the agent sees at each step."""

    task_id: str = Field(..., description="Unique task identifier.")
    ticket_id: str = Field(..., description="Unique ticket identifier.")
    customer_message: str = Field(..., description="The customer's message or complaint.")
    customer_tier: CustomerTier = Field(..., description="Customer subscription tier.")
    account_status: AccountStatus = Field(..., description="Current account status.")
    policy_context: Dict[str, Any] = Field(
        ..., description="Relevant policy rules for this scenario."
    )
    security_flags: List[str] = Field(
        default_factory=list, description="Active security / fraud signals."
    )
    prior_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Previous support interactions."
    )
    steps_taken: int = Field(0, description="Number of steps completed so far.")
    remaining_steps: int = Field(..., description="Steps remaining in episode budget.")
    available_actions: List[str] = Field(
        ..., description="Action types currently available."
    )
    current_status: str = Field(..., description="Human-readable summary of current state.")

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


class RewardModel(BaseModel):
    """Dense reward with structured breakdown."""

    reward: float = Field(..., description="Scalar reward for this step.")
    score_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension contribution to reward.",
    )
    penalties: Dict[str, float] = Field(
        default_factory=dict, description="Penalties applied this step."
    )
    subgoals_achieved: List[str] = Field(
        default_factory=list, description="Subgoal identifiers completed."
    )
    cumulative_reward: float = Field(0.0, description="Total reward so far in episode.")


# ---------------------------------------------------------------------------
# Explainability trace (embedded in StepResponse.info)
# ---------------------------------------------------------------------------


class PolicyChecks(BaseModel):
    """Deterministic policy checks evaluated at each step."""

    refund_window_ok: Optional[bool] = Field(
        None, description="Whether the refund falls within policy window (if applicable)."
    )
    verification_required: bool = Field(
        False, description="Whether identity verification is required before resolution."
    )
    security_flags_active: bool = Field(
        False, description="Whether active security/fraud signals are present."
    )
    prior_goodwill_refund_used: Optional[bool] = Field(
        None, description="Whether the customer has already used their goodwill refund."
    )
    classification_required_first: bool = Field(
        False, description="Whether ticket must be classified before this action."
    )


class StepTrace(BaseModel):
    """
    Explainability trace attached to every step() info response.
    Provides deterministic reasoning, subgoal tracking, and policy audit trail.
    """

    action_type: str
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    violations: List[str] = Field(default_factory=list)
    subgoals_completed: List[str] = Field(default_factory=list)
    decision_trace: List[str] = Field(
        default_factory=list,
        description="Deterministic reasoning strings explaining the evaluation outcome.",
    )
    loop_detected: bool = False
    invalid_action: bool = False
    policy_checks: PolicyChecks = Field(default_factory=PolicyChecks)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class StateModel(BaseModel):
    """Full internal environment state — for debugging and grading."""

    task_id: str
    ticket_id: str
    difficulty: str
    steps_taken: int
    remaining_steps: int
    cumulative_reward: float

    # Classification & routing
    classified: bool = False
    classification_correct: bool = False
    priority_set: bool = False
    priority_correct: bool = False
    risk_detected: bool = False
    risk_correct: bool = False
    team_assigned: bool = False
    team_correct: bool = False

    # Workflow progress
    verification_requested: bool = False
    reply_drafted: bool = False
    reply_safe: bool = False
    escalated: bool = False
    escalation_justified: bool = False
    resolved: bool = False
    resolution_correct: bool = False

    # Compliance
    policy_violation_count: int = 0
    unsafe_actions: List[str] = Field(default_factory=list)
    loop_count: int = 0
    last_action_type: Optional[str] = None
    invalid_action_count: int = 0

    # Adversarial tracking (for hard penalty grading)
    resolved_before_verification: bool = False
    security_ignored: bool = False   # had security flags but never ran detect_risk
    wrong_team_on_security: bool = False  # assigned billing/general on a security task

    # Final
    done: bool = False
    episode_score: Optional[float] = None

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Step response
# ---------------------------------------------------------------------------


class StepResponse(BaseModel):
    """Full response returned from step()."""

    observation: ObservationModel
    reward: RewardModel
    done: bool
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Rich explainability metadata including StepTrace, decision_trace, "
            "violations, policy_checks, and subgoal progress."
        ),
    )


# ---------------------------------------------------------------------------
# Grader output — with failure modes
# ---------------------------------------------------------------------------


class FailureEntry(BaseModel):
    """A single documented failure in agent behavior."""

    failure_type: str = Field(
        ..., description="Category of failure: policy_violation, unsafe_resolution, etc."
    )
    detail: str = Field(..., description="Human-readable description of the failure.")
    penalty: float = Field(0.0, description="Score penalty applied for this failure.")


class GraderOutput(BaseModel):
    """Grader evaluation of a completed episode."""

    task_id: str
    final_score: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    verdict: str
    notes: List[str] = Field(default_factory=list)
    failures: List[FailureEntry] = Field(
        default_factory=list,
        description="Structured list of agent failures that reduced the score.",
    )


# ---------------------------------------------------------------------------
# Task listing — enriched
# ---------------------------------------------------------------------------


class TaskDetailInfo(BaseModel):
    """Rich task descriptor for the /tasks endpoint."""

    task_id: str
    title: str
    difficulty: str
    description: str
    max_steps: int
    competencies: List[str] = Field(
        default_factory=list,
        description="Skills the agent must demonstrate to score well.",
    )
    action_schema: Dict[str, Any] = Field(default_factory=dict)


class TaskInfo(BaseModel):
    """Minimal task descriptor (kept for backward compatibility)."""

    task_id: str
    title: str
    difficulty: str
    description: str


class TaskListResponse(BaseModel):
    """Response from the /tasks endpoint."""

    tasks: List[TaskDetailInfo]
    action_schema: Dict[str, Any]
