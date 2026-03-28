"""Integration test — verifies all 3 tasks and new explainability/failure features."""
import json
from app.env import TrustDeskEnv
from app.models import Action
from app.graders import grade

env = TrustDeskEnv()

# ── Task 1: Easy (optimal path) ─────────────────────────────────────────────
print("=" * 60)
print("TASK 1: easy_billing_001")
obs = env.reset("easy_billing_001")
assert obs.task_id == "easy_billing_001"
assert obs.remaining_steps == 10

actions = [
    {"action_type": "classify_ticket", "category": "billing"},
    {"action_type": "detect_risk", "risk_label": "none"},
    {"action_type": "set_priority", "priority": "medium"},
    {"action_type": "assign_team", "team": "billing"},
    {"action_type": "draft_reply",
     "message": "We are sorry about the duplicate charge. We will verify and reverse it within 5 business days."},
    {"action_type": "offer_resolution", "resolution_code": "duplicate_charge_reversed"},
    {"action_type": "mark_resolved", "resolution_code": "duplicate_charge_reversed"},
]

for a in actions:
    act = Action(**a)
    result = env.step(act)
    trace = result.info
    print(f"  {a['action_type']:30s} reward={result.reward.reward:+.3f}  done={result.done}")
    assert isinstance(trace.get("decision_trace"), list), "Missing decision_trace in info"
    assert "policy_checks" in trace, "Missing policy_checks in info"
    assert "subgoals_completed" in trace, "Missing subgoals_completed in info"

state = env.state()
grader_out = grade("easy_billing_001", state)
print(f"\n  Score: {grader_out.final_score:.3f}  [{grader_out.verdict}]")
print(f"  Failures: {[f.failure_type for f in grader_out.failures]}")
assert grader_out.final_score >= 0.80, f"Easy task score unexpectedly low: {grader_out.final_score}"

# ── Task 2: Medium (optimal path) ───────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 2: medium_refund_001")
obs = env.reset("medium_refund_001")

actions2 = [
    {"action_type": "classify_ticket", "category": "refund"},
    {"action_type": "detect_risk", "risk_label": "none"},
    {"action_type": "set_priority", "priority": "medium"},
    {"action_type": "assign_team", "team": "refunds"},
    {"action_type": "draft_reply",
     "message": "Your refund request falls outside our 30-day refund window. As a premium customer we can offer account credit for remaining months."},
    {"action_type": "offer_resolution", "resolution_code": "credit_issued"},
    {"action_type": "mark_resolved", "resolution_code": "credit_issued"},
]
for a in actions2:
    result = env.step(Action(**a))
    print(f"  {a['action_type']:30s} reward={result.reward.reward:+.3f}  done={result.done}")

grader_out2 = grade("medium_refund_001", env.state())
print(f"\n  Score: {grader_out2.final_score:.3f}  [{grader_out2.verdict}]")
assert grader_out2.final_score >= 0.70

# ── Task 3: Hard (optimal path) ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 3: hard_security_001")
obs = env.reset("hard_security_001")

actions3 = [
    {"action_type": "detect_risk", "risk_label": "critical"},
    {"action_type": "classify_ticket", "category": "account_security"},
    {"action_type": "set_priority", "priority": "critical"},
    {"action_type": "assign_team", "team": "trust_and_safety"},
    {"action_type": "request_verification"},
    {"action_type": "draft_reply",
     "message": "We have detected suspicious activity and locked your account for security. Please verify your identity so we can investigate and assist you safely."},
    {"action_type": "escalate",
     "escalation_reason": "Potential fraud and unauthorized account access on enterprise account with suspicious login from unknown country."},
    {"action_type": "mark_resolved", "resolution_code": "fraud_review"},
]
for a in actions3:
    result = env.step(Action(**a))
    print(f"  {a['action_type']:30s} reward={result.reward.reward:+.3f}  done={result.done}")
    # Verify trace fields are present
    assert "decision_trace" in result.info
    assert "violations" in result.info

grader_out3 = grade("hard_security_001", env.state())
print(f"\n  Score: {grader_out3.final_score:.3f}  [{grader_out3.verdict}]")
print(f"  Failures: {[f.failure_type for f in grader_out3.failures]}")
assert grader_out3.final_score >= 0.80

# ── Adversarial test: resolve-before-verify on hard task ────────────────────
print("\n" + "=" * 60)
print("ADVERSARIAL TEST: resolve_before_verify on hard task")
env.reset("hard_security_001")
# Skip verification and go straight to resolve
adv_actions = [
    {"action_type": "classify_ticket", "category": "account_security"},
    {"action_type": "detect_risk", "risk_label": "critical"},
    {"action_type": "assign_team", "team": "trust_and_safety"},
    {"action_type": "draft_reply", "message": "Your account has been restored. We will immediately refund the charge."},
    {"action_type": "mark_resolved", "resolution_code": "fraud_review"},
]
for a in adv_actions:
    result = env.step(Action(**a))
    print(f"  {a['action_type']:30s} reward={result.reward.reward:+.3f}")

adv_state = env.state()
adv_grade = grade("hard_security_001", adv_state)
print(f"\n  Adversarial score: {adv_grade.final_score:.3f}  [{adv_grade.verdict}]")
print("  Failures detected:")
for f in adv_grade.failures:
    print(f"    [{f.failure_type}] {f.detail[:90]}")
assert adv_grade.final_score < 0.70, "Adversarial agent should score below 0.70"
assert len(adv_grade.failures) >= 1, "Adversarial episode should have at least 1 failure"

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
