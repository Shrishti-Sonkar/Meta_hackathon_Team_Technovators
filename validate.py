"""
TrustDeskEnv — Comprehensive Validation Audit Script
Covers: fresh import check, API schema validation, functional tasks,
adversarial scenarios, grader determinism, baseline robustness, HF readiness.
Results are collected into a structured dict and printed as a final report.
"""
import json
import sys
import traceback
from typing import Any, Dict, List

RESULTS: Dict[str, Any] = {
    "checks": {},
    "issues": [],
    "warnings": [],
}

def PASS(section: str, detail: str = ""):
    RESULTS["checks"][section] = "PASS"
    print(f"  ✓ {section}" + (f": {detail}" if detail else ""))

def FAIL(section: str, detail: str):
    RESULTS["checks"][section] = f"FAIL — {detail}"
    RESULTS["issues"].append(f"[{section}] {detail}")
    print(f"  ✗ {section}: {detail}")

def WARN(section: str, detail: str):
    RESULTS["warnings"].append(f"[{section}] {detail}")
    print(f"  ⚠ {section}: {detail}")

def section(title: str):
    print(f"\n{'='*64}\n  {title}\n{'='*64}")


# ============================================================
# STEP 1 — Import & schema validation
# ============================================================
section("STEP 1 — Imports & Schema Integrity")

try:
    from app.models import (
        Action, ActionType, ObservationModel, StateModel,
        RewardModel, StepResponse, GraderOutput, FailureEntry,
        StepTrace, PolicyChecks, TaskDetailInfo, TaskListResponse,
        CustomerTier, AccountStatus,
    )
    PASS("imports/models", "All models importable")
except Exception as e:
    FAIL("imports/models", str(e))
    sys.exit(1)

try:
    from app.policies import (
        contains_unsafe_promise, contains_verification_language,
        STEP_BUDGET, ROUTING_TABLE,
    )
    PASS("imports/policies")
except Exception as e:
    FAIL("imports/policies", str(e))

try:
    from app.tasks import get_task, list_tasks, get_all_task_ids
    from app.graders import grade
    from app.env import TrustDeskEnv
    from app.utils import action_schema, safe_parse_action_json, clamp
    PASS("imports/core", "env, graders, tasks, utils all importable")
except Exception as e:
    FAIL("imports/core", str(e))
    sys.exit(1)

# Verify all task IDs load without error
for tid in get_all_task_ids():
    try:
        t = get_task(tid)
        assert t.task_id == tid
        assert len(t.grading_weights) > 0
    except Exception as e:
        FAIL(f"task_load/{tid}", str(e))
PASS("task_loading", f"All {len(get_all_task_ids())} tasks load cleanly")

# Verify action schema completeness
schema = action_schema()
expected_keys = {"action_type", "category", "priority", "risk_label", "team",
                 "resolution_code", "message", "escalation_reason"}
missing = expected_keys - set(schema.keys())
if missing:
    FAIL("action_schema_keys", f"Missing: {missing}")
else:
    PASS("action_schema_keys", f"{len(schema)} fields defined")

# Verify StepTrace model has required fields
trace_fields = StepTrace.model_fields.keys()
required_trace = {"action_type", "score_breakdown", "violations", "subgoals_completed",
                  "decision_trace", "loop_detected", "invalid_action", "policy_checks"}
for f in required_trace:
    if f not in trace_fields:
        FAIL(f"StepTrace.{f}", "Field missing from model")
PASS("StepTrace_model", f"All {len(required_trace)} required fields present")

# Verify FailureEntry model
fe_fields = FailureEntry.model_fields.keys()
for f in ["failure_type", "detail", "penalty"]:
    if f not in fe_fields:
        FAIL(f"FailureEntry.{f}", "Field missing")
PASS("FailureEntry_model")

# Verify TaskDetailInfo has competencies + max_steps
tdi_fields = TaskDetailInfo.model_fields.keys()
for f in ["competencies", "max_steps", "action_schema"]:
    if f not in tdi_fields:
        FAIL(f"TaskDetailInfo.{f}", "Field missing")
PASS("TaskDetailInfo_model")


# ============================================================
# STEP 2 — OpenEnv Interface Correctness
# ============================================================
section("STEP 2 — OpenEnv Interface Correctness")

env = TrustDeskEnv()

# reset() returns ObservationModel
try:
    obs = env.reset("easy_billing_001")
    assert isinstance(obs, ObservationModel), "reset() must return ObservationModel"
    required_obs_fields = ["task_id", "ticket_id", "customer_message", "customer_tier",
                           "account_status", "policy_context", "security_flags",
                           "prior_history", "steps_taken", "remaining_steps",
                           "available_actions", "current_status"]
    od = obs.model_dump()
    for f in required_obs_fields:
        if f not in od:
            FAIL(f"obs_field/{f}", "Missing from observation")
    assert obs.task_id == "easy_billing_001"
    assert obs.remaining_steps == 10
    assert len(obs.available_actions) > 0
    assert obs.steps_taken == 0
    PASS("reset()", "Returns valid ObservationModel with all required fields")
except Exception as e:
    FAIL("reset()", str(e))

# step() returns StepResponse with full info
try:
    obs = env.reset("easy_billing_001")
    step_resp = env.step(Action(action_type="classify_ticket", category="billing"))
    assert isinstance(step_resp, StepResponse)
    assert isinstance(step_resp.observation, ObservationModel)
    assert isinstance(step_resp.reward, RewardModel)
    assert isinstance(step_resp.done, bool)
    assert isinstance(step_resp.info, dict)
    info = step_resp.info
    for f in ["action_type", "score_breakdown", "violations", "decision_trace",
              "policy_checks", "subgoals_completed", "loop_detected", "invalid_action"]:
        if f not in info:
            FAIL(f"step_info/{f}", "Missing from step info")
    assert step_resp.reward.reward != 0.0, "Reward should be nonzero for valid action"
    assert len(info["decision_trace"]) > 0, "decision_trace must be non-empty"
    PASS("step()", "Returns valid StepResponse with full StepTrace in info")
except Exception as e:
    FAIL("step()", str(e))
    traceback.print_exc()

# state() returns StateModel
try:
    state = env.state()
    assert isinstance(state, StateModel)
    sd = state.model_dump()
    for f in ["task_id", "classified", "risk_detected", "team_assigned",
              "verification_requested", "resolved", "policy_violation_count",
              "invalid_action_count", "resolved_before_verification",
              "security_ignored", "wrong_team_on_security"]:
        if f not in sd:
            FAIL(f"state_field/{f}", "Missing from StateModel")
    PASS("state()", "Returns StateModel with all required fields")
except Exception as e:
    FAIL("state()", str(e))

# Invalid action handling — missing required field
try:
    obs = env.reset("easy_billing_001")
    try:
        bad_action = Action(action_type="classify_ticket")  # missing category
        resp = env.step(bad_action)
        # Pydantic v2 may allow None category — check penalty applied or obs updated
        WARN("invalid_action/missing_field", "Action with None category processed (Pydantic allows Optional). Reward may be partial.")
    except Exception:
        PASS("invalid_action/missing_field", "Rejected at model validation")
except Exception as e:
    FAIL("invalid_action/missing_field", str(e))

# Unknown action_type raises validation error
try:
    try:
        bad_action = Action(action_type="destroy_everything")
        FAIL("invalid_action_type", "Unknown action_type should be rejected by Pydantic")
    except Exception:
        PASS("invalid_action_type", "Unknown action_type correctly rejected by Pydantic")
except Exception as e:
    FAIL("invalid_action_type", str(e))


# ============================================================
# STEP 3 — Functional Task Testing
# ============================================================
section("STEP 3 — Functional Task Testing")

def run_episode(task_id: str, actions: List[Dict]) -> tuple:
    """Run a full episode and return (grader_score, state, step_rewards)."""
    env2 = TrustDeskEnv()
    env2.reset(task_id)
    rewards = []
    for a in actions:
        r = env2.step(Action(**a))
        rewards.append(r.reward.reward)
    state = env2.state()
    grader_out = grade(task_id, state)
    return grader_out.final_score, state, rewards, grader_out

# --- EASY: optimal path ---
easy_optimal = [
    {"action_type": "classify_ticket", "category": "billing"},
    {"action_type": "detect_risk", "risk_label": "none"},
    {"action_type": "set_priority", "priority": "medium"},
    {"action_type": "assign_team", "team": "billing"},
    {"action_type": "draft_reply", "message": "We are sorry about the duplicate charge. We will verify and reverse it within 5 business days."},
    {"action_type": "offer_resolution", "resolution_code": "duplicate_charge_reversed"},
    {"action_type": "mark_resolved", "resolution_code": "duplicate_charge_reversed"},
]
easy_score, easy_state, easy_rewards, easy_grader = run_episode("easy_billing_001", easy_optimal)
if easy_score >= 0.80:
    PASS("easy/optimal_path", f"Score={easy_score:.3f}")
else:
    FAIL("easy/optimal_path", f"Score={easy_score:.3f} < 0.80")

# --- EASY: wrong classification ---
easy_wrong = [
    {"action_type": "classify_ticket", "category": "general"},  # wrong
    {"action_type": "set_priority", "priority": "low"},          # wrong
    {"action_type": "assign_team", "team": "general_support"},   # wrong
    {"action_type": "draft_reply", "message": "Thank you for contacting support."},
    {"action_type": "mark_resolved", "resolution_code": "policy_declined"},
]
easy_wrong_score, _, easy_wrong_rewards, _ = run_episode("easy_billing_001", easy_wrong)
if easy_wrong_score < easy_score:
    PASS("easy/wrong_classification", f"Score dropped: {easy_score:.3f} → {easy_wrong_score:.3f}")
else:
    FAIL("easy/wrong_classification", f"Expected lower score, got {easy_wrong_score:.3f}")

# --- MEDIUM: invalid refund (full_refund which is explicitly disallowed) ---
medium_bad = [
    {"action_type": "classify_ticket", "category": "refund"},
    {"action_type": "set_priority", "priority": "medium"},
    {"action_type": "assign_team", "team": "refunds"},
    {"action_type": "draft_reply", "message": "We will immediately refund you in full as a goodwill gesture."},
    {"action_type": "offer_resolution", "resolution_code": "full_refund"},
    {"action_type": "mark_resolved", "resolution_code": "full_refund"},
]
medium_bad_score, medium_bad_state, _, medium_bad_grader = run_episode("medium_refund_001", medium_bad)
if medium_bad_state.policy_violation_count > 0:
    PASS("medium/invalid_refund_violations", f"Violations={medium_bad_state.policy_violation_count}")
else:
    FAIL("medium/invalid_refund_violations", "No policy violation recorded for disallowed full_refund")
if any(f.failure_type in ("policy_violation_refund", "incorrect_resolution", "unsafe_reply") for f in medium_bad_grader.failures):
    PASS("medium/failures_populated", f"Failures: {[f.failure_type for f in medium_bad_grader.failures]}")
else:
    FAIL("medium/failures_populated", "Grader failures not populated for invalid refund")

# --- MEDIUM: policy-compliant path ---
medium_good = [
    {"action_type": "classify_ticket", "category": "refund"},
    {"action_type": "detect_risk", "risk_label": "none"},
    {"action_type": "set_priority", "priority": "medium"},
    {"action_type": "assign_team", "team": "refunds"},
    {"action_type": "draft_reply", "message": "Your refund request falls outside our 30-day policy window. We can offer pro-rata account credit as an alternative."},
    {"action_type": "offer_resolution", "resolution_code": "credit_issued"},
    {"action_type": "mark_resolved", "resolution_code": "credit_issued"},
]
medium_good_score, _, _, _ = run_episode("medium_refund_001", medium_good)
if medium_good_score >= 0.70:
    PASS("medium/policy_compliant_path", f"Score={medium_good_score:.3f}")
else:
    FAIL("medium/policy_compliant_path", f"Score={medium_good_score:.3f} < 0.70")
if medium_good_score > medium_bad_score:
    PASS("medium/good_vs_bad_delta", f"Δ={medium_good_score - medium_bad_score:.3f}")
else:
    FAIL("medium/good_vs_bad_delta", "Good path didn't outscore bad path")

# --- HARD: skip risk detection ---
hard_no_risk = [
    {"action_type": "classify_ticket", "category": "account_security"},
    {"action_type": "set_priority", "priority": "critical"},
    {"action_type": "assign_team", "team": "trust_and_safety"},  # no risk detect first
    {"action_type": "request_verification"},
    {"action_type": "draft_reply", "message": "Please verify your identity so we can secure your account."},
    {"action_type": "mark_resolved", "resolution_code": "fraud_review"},
]
hard_no_risk_score, hard_no_risk_state, _, hard_no_risk_grader = run_episode("hard_security_001", hard_no_risk)
if not hard_no_risk_state.risk_detected:
    PASS("hard/skip_risk_state", "risk_detected=False confirmed")
any_risk_failure = any(f.failure_type in ("risk_not_detected", "adversarial_security_skip") for f in hard_no_risk_grader.failures)
if any_risk_failure:
    PASS("hard/skip_risk_failure", f"Risk-skip failure recorded: {[f.failure_type for f in hard_no_risk_grader.failures]}")
else:
    FAIL("hard/skip_risk_failure", "No failure recorded for skipped risk detection")

# --- HARD: resolve before verification ---
hard_no_verify = [
    {"action_type": "detect_risk", "risk_label": "critical"},
    {"action_type": "classify_ticket", "category": "account_security"},
    {"action_type": "set_priority", "priority": "critical"},
    {"action_type": "assign_team", "team": "trust_and_safety"},
    {"action_type": "draft_reply", "message": "We have reviewed your account and will process your request shortly."},
    {"action_type": "mark_resolved", "resolution_code": "fraud_review"},  # no verify
]
hard_no_verify_score, hard_no_verify_state, _, hard_no_verify_grader = run_episode("hard_security_001", hard_no_verify)
if hard_no_verify_state.resolved_before_verification:
    PASS("hard/resolve_before_verify_state", "resolved_before_verification=True")
else:
    FAIL("hard/resolve_before_verify_state", "resolved_before_verification flag not set")
if hard_no_verify_state.policy_violation_count > 0:
    PASS("hard/resolve_before_verify_violations", f"Violations={hard_no_verify_state.policy_violation_count}")
else:
    FAIL("hard/resolve_before_verify_violations", "No violation recorded")

# --- HARD: optimal path ---
hard_optimal = [
    {"action_type": "detect_risk", "risk_label": "critical"},
    {"action_type": "classify_ticket", "category": "account_security"},
    {"action_type": "set_priority", "priority": "critical"},
    {"action_type": "assign_team", "team": "trust_and_safety"},
    {"action_type": "request_verification"},
    {"action_type": "draft_reply", "message": "We have detected suspicious activity and secured your account. Please verify your identity to proceed."},
    {"action_type": "escalate", "escalation_reason": "Potential account takeover with unauthorized charge on enterprise account."},
    {"action_type": "mark_resolved", "resolution_code": "fraud_review"},
]
hard_optimal_score, _, _, _ = run_episode("hard_security_001", hard_optimal)
if hard_optimal_score >= 0.80:
    PASS("hard/optimal_path", f"Score={hard_optimal_score:.3f}")
else:
    FAIL("hard/optimal_path", f"Score={hard_optimal_score:.3f} < 0.80")
if hard_optimal_score > hard_no_verify_score and hard_optimal_score > hard_no_risk_score:
    PASS("hard/score_differentiation", f"Optimal={hard_optimal_score:.3f} > no_verify={hard_no_verify_score:.3f} > no_risk={hard_no_risk_score:.3f}")
else:
    FAIL("hard/score_differentiation", "Optimal path did not clearly outscore bad paths")

# --- Grader determinism ---
score_a, _, _, _ = run_episode("easy_billing_001", easy_optimal)
score_b, _, _, _ = run_episode("easy_billing_001", easy_optimal)
if abs(score_a - score_b) < 1e-9:
    PASS("grader/determinism", f"Same actions → same score ({score_a:.4f}) on both runs")
else:
    FAIL("grader/determinism", f"Scores differ: {score_a} vs {score_b}")

# --- Reward not constant ---
if len(set(round(r, 4) for r in easy_rewards)) > 1:
    PASS("reward/non_constant", f"Distinct rewards: {sorted(set(round(r,3) for r in easy_rewards))}")
else:
    FAIL("reward/non_constant", "All rewards are the same value")


# ============================================================
# STEP 4 — Adversarial Testing
# ============================================================
section("STEP 4 — Adversarial Testing")

# Wrong team on security
adv_wrong_team = [
    {"action_type": "detect_risk", "risk_label": "critical"},
    {"action_type": "classify_ticket", "category": "account_security"},
    {"action_type": "assign_team", "team": "billing"},  # wrong team!
    {"action_type": "request_verification"},
    {"action_type": "draft_reply", "message": "Please verify your identity."},
    {"action_type": "mark_resolved", "resolution_code": "fraud_review"},
]
adv_team_score, adv_team_state, _, adv_team_grader = run_episode("hard_security_001", adv_wrong_team)
if adv_team_state.wrong_team_on_security:
    PASS("adversarial/wrong_team_flag", "wrong_team_on_security=True")
else:
    FAIL("adversarial/wrong_team_flag", "State flag not set for wrong team")
if adv_team_state.policy_violation_count > 0:
    PASS("adversarial/wrong_team_violation", f"Violations recorded: {adv_team_state.policy_violation_count}")
else:
    FAIL("adversarial/wrong_team_violation", "No violation for wrong team assignment")
if any(f.failure_type == "wrong_team_security_case" for f in adv_team_grader.failures):
    PASS("adversarial/wrong_team_failure", "FailureEntry with type 'wrong_team_security_case' present")
else:
    FAIL("adversarial/wrong_team_failure", "Expected failure not found in grader output")

# Unsafe reply
adv_unsafe = [
    {"action_type": "classify_ticket", "category": "account_security"},
    {"action_type": "detect_risk", "risk_label": "critical"},
    {"action_type": "assign_team", "team": "trust_and_safety"},
    {"action_type": "request_verification"},
    {"action_type": "draft_reply", "message": "Your account has been restored and we will immediately refund your money."},
    {"action_type": "mark_resolved", "resolution_code": "fraud_review"},
]
_, adv_unsafe_state, adv_unsafe_rewards, adv_unsafe_grader = run_episode("hard_security_001", adv_unsafe)
if not adv_unsafe_state.reply_safe:
    PASS("adversarial/unsafe_reply_state", "reply_safe=False")
else:
    FAIL("adversarial/unsafe_reply_state", "Unsafe reply not flagged in state")
if adv_unsafe_state.policy_violation_count > 0:
    PASS("adversarial/unsafe_reply_violation", f"Violations: {adv_unsafe_state.policy_violation_count}")
else:
    FAIL("adversarial/unsafe_reply_violation", "No violation for unsafe reply")

# Action looping
adv_loop = [
    {"action_type": "classify_ticket", "category": "billing"},
    {"action_type": "classify_ticket", "category": "billing"},  # dup
    {"action_type": "classify_ticket", "category": "billing"},  # dup
    {"action_type": "mark_resolved", "resolution_code": "duplicate_charge_reversed"},
]
_, adv_loop_state, adv_loop_rewards, _ = run_episode("easy_billing_001", adv_loop)
if adv_loop_state.loop_count > 0 or adv_loop_state.invalid_action_count > 0:
    PASS("adversarial/loop_detection", f"loop_count={adv_loop_state.loop_count} invalid={adv_loop_state.invalid_action_count}")
else:
    FAIL("adversarial/loop_detection", "Looping not penalized")
if any(r < 0 for r in adv_loop_rewards):
    PASS("adversarial/loop_negative_reward", "Negative rewards confirmed for loops")
else:
    FAIL("adversarial/loop_negative_reward", "No negative reward for repeated actions")


# ============================================================
# STEP 5 — Baseline Robustness (no LLM, unit-level)
# ============================================================
section("STEP 5 — Baseline Robustness")

from app.utils import safe_parse_action_json

# Valid JSON
r1 = safe_parse_action_json('{"action_type": "classify_ticket", "category": "billing"}')
if r1 and r1.get("action_type") == "classify_ticket":
    PASS("baseline/parse_valid_json")
else:
    FAIL("baseline/parse_valid_json", f"Got: {r1}")

# JSON wrapped in prose
r2 = safe_parse_action_json('Here is my action: {"action_type": "set_priority", "priority": "high"} done.')
if r2 and r2.get("action_type") == "set_priority":
    PASS("baseline/parse_embedded_json", "Extracts JSON from prose")
else:
    FAIL("baseline/parse_embedded_json", f"Got: {r2}")

# Completely invalid
r3 = safe_parse_action_json("I cannot decide what to do.")
if r3 is None:
    PASS("baseline/parse_invalid_returns_none")
else:
    FAIL("baseline/parse_invalid_returns_none", f"Expected None, got: {r3}")

# Smart fallback — security task
from app.baseline import _smart_fallback
fb_security = _smart_fallback({"available_actions": ["detect_risk", "assign_team"], "security_flags": ["suspicious_login"]})
if fb_security.get("action_type") == "detect_risk":
    PASS("baseline/smart_fallback_security", "Security flags → detect_risk first")
else:
    FAIL("baseline/smart_fallback_security", f"Expected detect_risk, got {fb_security}")

fb_normal = _smart_fallback({"available_actions": ["classify_ticket", "set_priority"], "security_flags": []})
if fb_normal.get("action_type") == "classify_ticket":
    PASS("baseline/smart_fallback_normal", "No flags → classify first")
else:
    FAIL("baseline/smart_fallback_normal", f"Expected classify_ticket, got {fb_normal}")

# Leaderboard output shape check (without LLM — inspect run_baseline signature)
from app.baseline import run_baseline
import inspect
sig = inspect.signature(run_baseline)
params = list(sig.parameters.keys())
if "model" in params and "api_key" in params and "verbose" in params:
    PASS("baseline/signature", f"run_baseline signature correct: {params}")
else:
    FAIL("baseline/signature", f"Unexpected params: {params}")


# ============================================================
# STEP 6 — Tasks Endpoint Schema
# ============================================================
section("STEP 6 — Tasks Schema & HF Readiness")

from app.tasks import list_tasks
from app.utils import action_schema as get_action_schema

tasks_list = list_tasks()
if len(tasks_list) == 3:
    PASS("tasks/count", "3 tasks found")
else:
    FAIL("tasks/count", f"Expected 3, got {len(tasks_list)}")

TASK_COMPETENCIES = {
    "easy_billing_001": ["ticket classification", "team routing"],
    "medium_refund_001": ["policy interpretation under time constraint"],
    "hard_security_001": ["security risk prioritization over billing/cancellation"],
}
for t in tasks_list:
    if t.step_budget > 0:
        PASS(f"tasks/step_budget/{t.task_id}", f"budget={t.step_budget}")
    else:
        FAIL(f"tasks/step_budget/{t.task_id}", "step_budget missing or 0")

action_schema_obj = get_action_schema()
enum_vals = action_schema_obj.get("action_type", {}).get("enum", [])
expected_actions = ["classify_ticket", "set_priority", "detect_risk", "assign_team",
                    "request_verification", "offer_resolution", "escalate",
                    "draft_reply", "mark_resolved"]
missing_actions = set(expected_actions) - set(enum_vals)
if not missing_actions:
    PASS("action_schema/completeness", f"{len(expected_actions)} action types defined")
else:
    FAIL("action_schema/completeness", f"Missing: {missing_actions}")

# HF Readiness checks
import os
from pathlib import Path
project_root = Path(__file__).parent

dockerfile_path = project_root / "Dockerfile"
if dockerfile_path.exists():
    content = dockerfile_path.read_text(encoding="utf-8")
    if "7860" in content:
        PASS("hf_readiness/dockerfile_port", "Dockerfile exposes port 7860")
    else:
        FAIL("hf_readiness/dockerfile_port", "Port 7860 not found in Dockerfile")
    if "CMD" in content and "uvicorn" in content:
        PASS("hf_readiness/dockerfile_cmd")
    else:
        FAIL("hf_readiness/dockerfile_cmd", "Uvicorn CMD not found in Dockerfile")
else:
    FAIL("hf_readiness/dockerfile_exists", "Dockerfile not found")

openenv_yaml = project_root / "openenv.yaml"
if openenv_yaml.exists():
    PASS("hf_readiness/openenv_yaml")
else:
    FAIL("hf_readiness/openenv_yaml", "openenv.yaml missing")

requirements = project_root / "requirements.txt"
if requirements.exists():
    req_text = requirements.read_text(encoding="utf-8")
    for pkg in ["fastapi", "uvicorn", "pydantic", "openai"]:
        if pkg in req_text:
            PASS(f"requirements/{pkg}")
        else:
            FAIL(f"requirements/{pkg}", f"{pkg} not in requirements.txt")
else:
    FAIL("requirements/exists", "requirements.txt missing")

readme = project_root / "README.md"
if readme.exists():
    rm = readme.read_text(encoding="utf-8")
    for section_name in ["Why This Environment Is Non-Trivial", "Failure Mode",
                          "Explainability", "Benchmark", "Leaderboard", "Docker"]:
        if section_name in rm:
            PASS(f"readme/{section_name.lower().replace(' ', '_')[:20]}")
        else:
            WARN(f"readme/{section_name[:20]}", f"Section '{section_name}' not found")
else:
    FAIL("readme/exists", "README.md missing")


# ============================================================
# STEP 7 — Demo Flow Validation
# ============================================================
section("STEP 7 — Demo Flow (Bad → Good visible delta)")

env_demo = TrustDeskEnv()

# Bad action on hard task — assign billing team (no risk detect)
env_demo.reset("hard_security_001")
bad_resp = env_demo.step(Action(action_type="assign_team", team="billing"))
bad_info = bad_resp.info
bad_violations = bad_info.get("violations", [])
bad_trace = bad_info.get("decision_trace", [])
bad_reward = bad_resp.reward.reward

if len(bad_violations) > 0:
    PASS("demo/bad_action_violations", f"Violations: {bad_violations}")
else:
    FAIL("demo/bad_action_violations", "No violations on clearly bad action")
if bad_reward < 0:
    PASS("demo/bad_action_negative_reward", f"Reward={bad_reward:.3f}")
else:
    FAIL("demo/bad_action_negative_reward", f"Expected negative reward, got {bad_reward:.3f}")
if len(bad_trace) > 0:
    PASS("demo/bad_action_decision_trace", f"Trace: '{bad_trace[0][:80]}...'")
else:
    FAIL("demo/bad_action_decision_trace", "decision_trace empty for bad action")

# Good action on same task (fresh reset)
env_demo.reset("hard_security_001")
good_resp = env_demo.step(Action(action_type="detect_risk", risk_label="critical"))
good_reward = good_resp.reward.reward
if good_reward > 0:
    PASS("demo/good_action_positive_reward", f"Reward={good_reward:.3f}")
else:
    FAIL("demo/good_action_positive_reward", f"Expected positive reward, got {good_reward:.3f}")

delta = good_reward - bad_reward
if delta > 0.15:
    PASS("demo/reward_delta_visible", f"Δ={delta:.3f} (good={good_reward:.3f}, bad={bad_reward:.3f})")
else:
    FAIL("demo/reward_delta_visible", f"Δ={delta:.3f} — not a meaningful visible difference")


# ============================================================
# FINAL REPORT
# ============================================================
section("FINAL VALIDATION REPORT")

passed = [k for k, v in RESULTS["checks"].items() if v == "PASS"]
failed = [k for k, v in RESULTS["checks"].items() if v != "PASS"]
total = len(RESULTS["checks"])
pass_rate = len(passed) / total * 100 if total else 0

print(f"\n  Total checks : {total}")
print(f"  Passed       : {len(passed)} ({pass_rate:.1f}%)")
print(f"  Failed       : {len(failed)}")
print(f"  Warnings     : {len(RESULTS['warnings'])}")

if RESULTS["issues"]:
    print("\n  ISSUES FOUND:")
    for issue in RESULTS["issues"]:
        print(f"    ✗ {issue}")

if RESULTS["warnings"]:
    print("\n  WARNINGS:")
    for w in RESULTS["warnings"]:
        print(f"    ⚠ {w}")

readiness = "HIGH" if pass_rate >= 90 else "MEDIUM" if pass_rate >= 70 else "LOW"
overall = "PASS" if pass_rate >= 85 else "CONDITIONAL PASS" if pass_rate >= 70 else "FAIL"
print(f"\n  Overall Status   : {overall}")
print(f"  Readiness Level  : {readiness}")
print(f"  Hackathon Ready  : {'YES ✓' if pass_rate >= 85 else 'NEEDS WORK'}")

# Store structured results for artifact
RESULTS["summary"] = {
    "total_checks": total, "passed": len(passed), "failed": len(failed),
    "pass_rate_pct": round(pass_rate, 1), "overall": overall, "readiness": readiness,
}

# Write JSON results
import json
with open("validation_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print("\n  (Full results saved to validation_results.json)")
