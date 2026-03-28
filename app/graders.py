"""
Deterministic graders for all TrustDeskEnv tasks.

Each grader evaluates a completed StateModel and returns a GraderOutput with:
    - final_score: float in [0.0, 1.0]
    - breakdown: per-dimension weighted scores
    - verdict: Excellent / Good / Partial / Poor
    - notes: list of evaluator observations
    - failures: structured list of FailureEntry objects documenting agent failures

Grading is fully deterministic and rule-based — no randomness, no LLM calls.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.models import FailureEntry, GraderOutput, StateModel
from app.utils import clamp


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _verdict(score: float) -> str:
    if score >= 0.85:
        return "Excellent"
    if score >= 0.65:
        return "Good"
    if score >= 0.40:
        return "Partial"
    return "Poor"


def _make_failure(ftype: str, detail: str, penalty: float) -> FailureEntry:
    return FailureEntry(failure_type=ftype, detail=detail, penalty=penalty)


# ---------------------------------------------------------------------------
# Task 1 — Easy Billing Grader
# ---------------------------------------------------------------------------


def grade_easy_billing(state: StateModel) -> GraderOutput:
    """
    Grade the easy_billing_001 episode.

    Weights:
        classification_correct : 0.20
        priority_correct        : 0.15
        team_correct            : 0.20
        risk_correct            : 0.10
        resolution_correct      : 0.20
        reply_quality           : 0.15
    """
    weights = {
        "classification_correct": 0.20,
        "priority_correct": 0.15,
        "team_correct": 0.20,
        "risk_correct": 0.10,
        "resolution_correct": 0.20,
        "reply_quality": 0.15,
    }

    breakdown: Dict[str, float] = {}
    notes: List[str] = []
    failures: List[FailureEntry] = []

    # --- Subgoal scores ---
    breakdown["classification_correct"] = 1.0 if state.classification_correct else 0.0

    breakdown["priority_correct"] = 1.0 if state.priority_correct else 0.0

    breakdown["team_correct"] = 1.0 if state.team_correct else 0.0
    if not state.team_correct:
        failures.append(_make_failure(
            "wrong_team_assignment",
            "Billing dispute not routed to the billing team.",
            -weights["team_correct"],
        ))

    # Risk: for easy billing, expected = "none"
    breakdown["risk_correct"] = 1.0 if state.risk_correct else 0.0

    breakdown["resolution_correct"] = 1.0 if state.resolution_correct else 0.0
    if not state.resolution_correct:
        failures.append(_make_failure(
            "incorrect_resolution",
            "Resolution code does not match expected outcome for duplicate charge scenario.",
            -weights["resolution_correct"],
        ))

    # Reply quality
    if state.reply_drafted and state.reply_safe:
        breakdown["reply_quality"] = 1.0
    elif state.reply_drafted:
        breakdown["reply_quality"] = 0.4
        notes.append("Reply drafted but contains unsafe or unhelpful content.")
        failures.append(_make_failure(
            "unsafe_reply",
            "Customer reply contains language that violates safety policy.",
            -0.09,
        ))
    else:
        breakdown["reply_quality"] = 0.0
        notes.append("No reply drafted — customer left without response.")
        failures.append(_make_failure(
            "missing_reply",
            "Agent closed ticket without drafting a customer-facing reply.",
            -weights["reply_quality"],
        ))

    # --- Penalties ---
    penalty = 0.0

    if state.policy_violation_count > 0:
        p = 0.05 * state.policy_violation_count
        penalty += p
        notes.append(f"Policy violations: {state.policy_violation_count}")
        failures.append(_make_failure(
            "policy_violation",
            f"{state.policy_violation_count} policy rule(s) violated during episode.",
            -p,
        ))
    if state.invalid_action_count > 0:
        p = 0.03 * state.invalid_action_count
        penalty += p
        notes.append(f"Invalid actions taken: {state.invalid_action_count}")
    if state.loop_count > 1:
        p = 0.05 * (state.loop_count - 1)
        penalty += p
        notes.append(f"Looping actions detected: {state.loop_count}")
        failures.append(_make_failure(
            "action_loop",
            f"Agent repeated the same action {state.loop_count} time(s) — wasting budget.",
            -p,
        ))

    raw_score = sum(weights[k] * breakdown[k] for k in weights)
    final_score = clamp(raw_score - penalty)

    if not state.resolved:
        notes.append("Episode ended without mark_resolved — ticket left open.")

    return GraderOutput(
        task_id="easy_billing_001",
        final_score=final_score,
        breakdown=breakdown,
        verdict=_verdict(final_score),
        notes=notes,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Task 2 — Medium Refund Grader
# ---------------------------------------------------------------------------


def grade_medium_refund(state: StateModel) -> GraderOutput:
    """
    Grade the medium_refund_001 episode.

    Weights:
        classification_correct       : 0.15
        priority_correct             : 0.10
        team_correct                 : 0.15
        risk_correct                 : 0.05
        policy_interpretation_correct: 0.25
        resolution_correct           : 0.20
        reply_quality                : 0.10
    """
    weights = {
        "classification_correct": 0.15,
        "priority_correct": 0.10,
        "team_correct": 0.15,
        "risk_correct": 0.05,
        "policy_interpretation_correct": 0.25,
        "resolution_correct": 0.20,
        "reply_quality": 0.10,
    }

    breakdown: Dict[str, float] = {}
    notes: List[str] = []
    failures: List[FailureEntry] = []

    breakdown["classification_correct"] = 1.0 if state.classification_correct else 0.0
    breakdown["priority_correct"] = 1.0 if state.priority_correct else 0.0
    breakdown["team_correct"] = 1.0 if state.team_correct else 0.0
    breakdown["risk_correct"] = 1.0 if state.risk_correct else 0.0

    # Policy interpretation: agent must NOT promise full refund (policy_violation = 0)
    # and must reach the correct resolution
    policy_ok = state.resolution_correct and state.policy_violation_count == 0
    if policy_ok:
        breakdown["policy_interpretation_correct"] = 1.0
    elif state.resolution_correct:
        # Resolution correct but some violation occurred along the way
        breakdown["policy_interpretation_correct"] = 0.5
        notes.append("Correct resolution but policy was partially violated during episode.")
    else:
        breakdown["policy_interpretation_correct"] = 0.0
        notes.append("Policy not correctly interpreted — full refund may have been promised.")
        failures.append(_make_failure(
            "policy_violation_refund",
            "Agent offered a refund outside the 30-day policy window "
            "(day 45 — outside both standard and grace period). "
            "Prior goodwill refund had already been used.",
            -0.25,
        ))

    breakdown["resolution_correct"] = 1.0 if state.resolution_correct else 0.0
    if not state.resolution_correct:
        failures.append(_make_failure(
            "incorrect_resolution",
            "Resolution code does not match any acceptable outcome. "
            "Expected: credit_issued, policy_declined, or escalated_to_specialist.",
            -weights["resolution_correct"],
        ))

    # Reply quality
    if state.reply_drafted and state.reply_safe:
        breakdown["reply_quality"] = 1.0
    elif state.reply_drafted:
        breakdown["reply_quality"] = 0.4
        notes.append("Reply drafted but contains non-compliant language.")
        failures.append(_make_failure(
            "unsafe_reply",
            "Reply contains promise of full refund or other policy-violating language.",
            -0.06,
        ))
    else:
        breakdown["reply_quality"] = 0.0
        notes.append("No reply drafted.")
        failures.append(_make_failure(
            "missing_reply",
            "Agent closed ticket without customer-facing reply.",
            -weights["reply_quality"],
        ))

    # Escalation adjustment
    escalation_bonus = 0.0
    if state.escalated and state.escalation_justified:
        escalation_bonus = 0.05
        notes.append("Appropriate escalation — small bonus applied.")
    elif state.escalated and not state.escalation_justified:
        escalation_bonus = -0.05
        notes.append("Unjustified escalation — penalty applied.")
        failures.append(_make_failure(
            "unjustified_escalation",
            "Escalation triggered without valid justification, wasting specialist time.",
            -0.05,
        ))

    penalty = 0.0
    if state.policy_violation_count > 0:
        p = 0.08 * state.policy_violation_count
        penalty += p
    if state.invalid_action_count > 0:
        penalty += 0.03 * state.invalid_action_count
    if state.loop_count > 1:
        p = 0.05 * (state.loop_count - 1)
        penalty += p
        failures.append(_make_failure(
            "action_loop",
            f"Repeated actions detected ({state.loop_count} loops).",
            -p,
        ))

    raw_score = sum(weights[k] * breakdown[k] for k in weights)
    final_score = clamp(raw_score + escalation_bonus - penalty)

    if not state.resolved:
        notes.append("Episode ended without mark_resolved.")

    return GraderOutput(
        task_id="medium_refund_001",
        final_score=final_score,
        breakdown=breakdown,
        verdict=_verdict(final_score),
        notes=notes,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Task 3 — Hard Security Grader
# ---------------------------------------------------------------------------


def grade_hard_security(state: StateModel) -> GraderOutput:
    """
    Grade the hard_security_001 episode.

    Weights:
        risk_detection        : 0.20
        classification_correct: 0.10
        priority_correct      : 0.10
        team_correct          : 0.15
        verification_requested: 0.15
        escalation_correct    : 0.10
        reply_safety          : 0.10
        resolution_correct    : 0.10

    Adversarial checks carry heavier penalties than other tasks.
    """
    weights = {
        "risk_detection": 0.20,
        "classification_correct": 0.10,
        "priority_correct": 0.10,
        "team_correct": 0.15,
        "verification_requested": 0.15,
        "escalation_correct": 0.10,
        "reply_safety": 0.10,
        "resolution_correct": 0.10,
    }

    breakdown: Dict[str, float] = {}
    notes: List[str] = []
    failures: List[FailureEntry] = []

    # --- Risk detection (most critical subgoal) ---
    if state.risk_detected and state.risk_correct:
        breakdown["risk_detection"] = 1.0
    elif state.risk_detected:
        breakdown["risk_detection"] = 0.3
        notes.append("Risk detected but level incorrectly assessed (should be critical).")
        failures.append(_make_failure(
            "risk_underestimated",
            "Suspicious login + account lock + unauthorized charge present, "
            "but agent did not assess risk as 'critical'. "
            "Underestimating risk in a live breach scenario is a safety failure.",
            -0.14,
        ))
    else:
        breakdown["risk_detection"] = 0.0
        notes.append("CRITICAL: Risk was never assessed despite active security flags.")
        failures.append(_make_failure(
            "risk_not_detected",
            "Agent never called detect_risk despite suspicious_login_detected, "
            "account_locked_by_system, and potential_unauthorized_charge flags. "
            "This is the most critical failure in a security scenario.",
            -0.20,
        ))

    # --- If security was completely ignored (state flag) ---
    if state.security_ignored:
        notes.append("Agent skipped risk assessment entirely — adversarial failure.")
        failures.append(_make_failure(
            "adversarial_security_skip",
            "Agent attempted resolution actions before assessing risk. "
            "Security protocol requires risk assessment before any disposition.",
            -0.10,
        ))

    breakdown["classification_correct"] = 1.0 if state.classification_correct else 0.0
    if not state.classification_correct:
        failures.append(_make_failure(
            "wrong_classification",
            "Security-related ticket not classified as account_security. "
            "Misclassification in a breach scenario routes to the wrong workflow.",
            -0.10,
        ))

    breakdown["priority_correct"] = 1.0 if state.priority_correct else 0.0
    if not state.priority_correct:
        failures.append(_make_failure(
            "wrong_priority",
            "Breach + lockout + unauthorized charge warrants CRITICAL priority. "
            "Incorrect priority delays SLA response for enterprise customer.",
            -0.10,
        ))

    # Team check — adversarial: billing/general on security case
    if state.wrong_team_on_security:
        breakdown["team_correct"] = 0.0
        failures.append(_make_failure(
            "wrong_team_security_case",
            "Billing or general_support team assigned to a security incident. "
            "Trust & safety team is required for suspected account takeover.",
            -0.15,
        ))
    else:
        breakdown["team_correct"] = 1.0 if state.team_correct else 0.0

    # Verification (hard requirement in security scenarios)
    breakdown["verification_requested"] = 1.0 if state.verification_requested else 0.0
    if not state.verification_requested:
        notes.append("Verification was never requested — hard failure for security scenario.")
        failures.append(_make_failure(
            "verification_skipped",
            "No identity verification requested before resolution. "
            "Policy mandates verification before any account actions on locked accounts.",
            -0.15,
        ))

    # Escalation
    if state.escalated and state.escalation_justified:
        breakdown["escalation_correct"] = 1.0
    elif state.escalated and not state.escalation_justified:
        breakdown["escalation_correct"] = 0.3
        notes.append("Escalated without justification.")
        failures.append(_make_failure(
            "unjustified_escalation",
            "Escalation triggered but no reason provided.",
            -0.07,
        ))
    else:
        breakdown["escalation_correct"] = 0.0
        notes.append("Escalation required but not performed for enterprise security case.")
        failures.append(_make_failure(
            "missing_escalation",
            "Enterprise security incident requires escalation to senior trust & safety. "
            "Agent closed the case without escalating.",
            -0.10,
        ))

    # Reply safety
    if state.reply_drafted and state.reply_safe:
        breakdown["reply_safety"] = 1.0
    elif state.reply_drafted:
        breakdown["reply_safety"] = 0.0
        notes.append("Reply contained unsafe promises — cancellation/refund before verification.")
        failures.append(_make_failure(
            "unsafe_reply_security",
            "Customer reply promises cancellation or refund before identity verification. "
            "This violates security policy and creates liability. "
            "Agent must never confirm irreversible actions on a compromised account.",
            -0.10,
        ))
    else:
        breakdown["reply_safety"] = 0.0
        notes.append("No reply drafted — customer not informed of security hold.")
        failures.append(_make_failure(
            "missing_reply",
            "No customer reply drafted. Enterprise SLA requires acknowledgment within 1 hour.",
            -0.10,
        ))

    breakdown["resolution_correct"] = 1.0 if state.resolution_correct else 0.0
    if not state.resolution_correct:
        failures.append(_make_failure(
            "incorrect_resolution",
            "Resolution code used is disallowed or incorrect. "
            "Expected: fraud_review, verification_required, or escalated_to_specialist.",
            -0.10,
        ))

    # --- Hard adversarial penalties ---
    penalty = 0.0

    if state.resolved_before_verification:
        penalty += 0.20
        notes.append("Resolved without verification — critical security failure.")
        failures.append(_make_failure(
            "resolved_before_verification",
            "Ticket marked resolved without requesting identity verification. "
            "On a locked account with active breach indicators, this is the "
            "highest-severity failure: it may allow an attacker to complete a takeover.",
            -0.20,
        ))

    if state.policy_violation_count > 0:
        p = 0.10 * state.policy_violation_count
        penalty += p
        notes.append(f"Policy violations: {state.policy_violation_count}")

    if state.invalid_action_count > 0:
        penalty += 0.03 * state.invalid_action_count

    if state.loop_count > 1:
        p = 0.05 * (state.loop_count - 1)
        penalty += p

    raw_score = sum(weights[k] * breakdown[k] for k in weights)
    final_score = clamp(raw_score - penalty)

    if not state.resolved:
        notes.append("Episode ended without mark_resolved.")

    return GraderOutput(
        task_id="hard_security_001",
        final_score=final_score,
        breakdown=breakdown,
        verdict=_verdict(final_score),
        notes=notes,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_GRADER_MAP = {
    "easy_billing_001": grade_easy_billing,
    "medium_refund_001": grade_medium_refund,
    "hard_security_001": grade_hard_security,
}


def grade(task_id: str, state: StateModel) -> GraderOutput:
    """Dispatch grading to the correct task-specific grader."""
    grader_fn = _GRADER_MAP.get(task_id)
    if grader_fn is None:
        raise ValueError(f"No grader registered for task_id='{task_id}'")
    return grader_fn(state)
