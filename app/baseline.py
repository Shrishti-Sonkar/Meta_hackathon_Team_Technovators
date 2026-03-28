"""
Baseline inference agent for TrustDeskEnv.

Drives an OpenAI model through all 3 tasks deterministically.
Returns a leaderboard-style benchmark report with:
    - per-task grader scores
    - average score
    - efficiency metrics (avg steps, violations, invalid actions, completion rate)
    - model metadata

Robustness features:
    - strict JSON parsing with one retry using a correction prompt
    - task-aware deterministic fallback actions if model output is invalid
    - never crashes due to a single bad model output
    - fixed task order and max_step cap for reproducibility
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from app.env import TrustDeskEnv
from app.graders import grade
from app.models import Action
from app.tasks import get_all_task_ids, get_task
from app.utils import safe_parse_action_json, action_schema


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert policy-compliant customer support and trust & safety operations agent.

Your job is to handle enterprise support tickets by taking structured, policy-aware actions.

You must always output a SINGLE valid JSON object representing your NEXT action. No other text, no markdown, no explanation.

Available action_types:
- classify_ticket       → required fields: category
- set_priority          → required fields: priority
- detect_risk           → required fields: risk_label
- assign_team           → required fields: team
- request_verification  → no extra fields required
- offer_resolution      → required fields: resolution_code
- escalate              → required fields: escalation_reason
- draft_reply           → required fields: message (customer-facing, honest, policy-compliant)
- mark_resolved         → required fields: resolution_code

STRICT RULES:
1. ALWAYS classify the ticket as your FIRST action.
2. If security_flags are non-empty, run detect_risk BEFORE assign_team or any resolution.
3. NEVER promise a refund, cancellation, or account restoration before verification if security flags are active.
4. In security scenarios (security_flags non-empty), ALWAYS request_verification before mark_resolved.
5. Draft a helpful, safe, honest reply before resolving.
6. Output ONLY valid JSON — nothing else.

GOOD EXAMPLE:
{"action_type": "classify_ticket", "category": "billing"}

BAD (never do this):
Here is my action: {"action_type": ...}  ← bad, must be JSON only
"""

CORRECTION_PROMPT = """Your previous output was not valid JSON. 
You MUST output ONLY a single valid JSON object, nothing else — no prose, no markdown, no backticks.

Available action_types: classify_ticket, set_priority, detect_risk, assign_team, 
request_verification, offer_resolution, escalate, draft_reply, mark_resolved.

Output the action JSON now:"""


# ---------------------------------------------------------------------------
# Deterministic fallbacks — per task difficulty and progress
# ---------------------------------------------------------------------------

def _smart_fallback(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a safe deterministic fallback action based on current observation.
    Ordered by what would be most useful next given available_actions.
    """
    available = obs_dict.get("available_actions", [])
    security_flags = obs_dict.get("security_flags", [])

    priority_sequence = [
        "classify_ticket",
        "detect_risk",
        "set_priority",
        "assign_team",
        "request_verification",
        "draft_reply",
        "offer_resolution",
        "escalate",
        "mark_resolved",
    ]

    # If there are security flags and risk hasn't been detected yet,
    # prioritize detect_risk first
    if security_flags and "detect_risk" in available:
        return {"action_type": "detect_risk", "risk_label": "critical"}

    # Follow priority sequence
    for action_type in priority_sequence:
        if action_type not in available:
            continue
        if action_type == "classify_ticket":
            # Infer category from security flags presence
            cat = "account_security" if security_flags else "billing"
            return {"action_type": "classify_ticket", "category": cat}
        if action_type == "detect_risk":
            risk = "critical" if security_flags else "none"
            return {"action_type": "detect_risk", "risk_label": risk}
        if action_type == "set_priority":
            prio = "critical" if security_flags else "medium"
            return {"action_type": "set_priority", "priority": prio}
        if action_type == "assign_team":
            team = "trust_and_safety" if security_flags else "billing"
            return {"action_type": "assign_team", "team": team}
        if action_type == "request_verification":
            return {"action_type": "request_verification"}
        if action_type == "draft_reply":
            if security_flags:
                msg = (
                    "We have detected suspicious activity on your account. "
                    "For your security, we have placed a temporary hold. "
                    "Please verify your identity so we can assist you safely."
                )
            else:
                msg = (
                    "Thank you for contacting support. "
                    "We have reviewed your request and will process it in accordance "
                    "with our policy. Please allow 3–5 business days for resolution."
                )
            return {"action_type": "draft_reply", "message": msg}
        if action_type == "offer_resolution":
            return {"action_type": "offer_resolution", "resolution_code": "escalated_to_specialist"}
        if action_type == "escalate":
            return {
                "action_type": "escalate",
                "escalation_reason": "Complex case requires specialist review.",
            }
        if action_type == "mark_resolved":
            code = "fraud_review" if security_flags else "escalated_to_specialist"
            return {"action_type": "mark_resolved", "resolution_code": code}

    # Last resort
    return {"action_type": "mark_resolved", "resolution_code": "escalated_to_specialist"}


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(observation: Dict[str, Any]) -> str:
    """Construct the per-step model prompt from the current observation."""
    return (
        f"=== Current Ticket State ===\n"
        f"Task ID:          {observation.get('task_id')}\n"
        f"Ticket ID:        {observation.get('ticket_id')}\n"
        f"Customer Tier:    {observation.get('customer_tier')}\n"
        f"Account Status:   {observation.get('account_status')}\n"
        f"Security Flags:   {observation.get('security_flags')}\n"
        f"Customer Message:\n  \"{observation.get('customer_message')}\"\n\n"
        f"Policy Context:\n{json.dumps(observation.get('policy_context'), indent=2)}\n\n"
        f"Prior History:    {json.dumps(observation.get('prior_history'))}\n"
        f"Steps Taken:      {observation.get('steps_taken')} "
        f"(remaining: {observation.get('remaining_steps')})\n"
        f"Current Status:   {observation.get('current_status')}\n"
        f"Available Actions: {observation.get('available_actions')}\n\n"
        f"Output your next action as a single JSON object:"
    )


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_task_episode(
    env: TrustDeskEnv,
    task_id: str,
    client: Any,
    model: str = "gpt-4o-mini",
    max_steps: int = 15,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run one complete episode for a single task.

    Robust against:
        - malformed model JSON output (one retry with correction prompt)
        - empty model responses
        - API exceptions

    Returns structured result dict with step history and grader output.
    """
    obs = env.reset(task_id)
    obs_dict = obs.model_dump()

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    history: List[Dict[str, Any]] = []

    done = False
    step_count = 0
    total_reward = 0.0
    invalid_actions = 0
    violations = 0

    while not done and step_count < max_steps:
        user_prompt = _build_user_prompt(obs_dict)
        messages.append({"role": "user", "content": user_prompt})

        # ---- Model call with error handling ----
        raw_output = ""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=512,
            )
            raw_output = response.choices[0].message.content.strip()
        except Exception as exc:
            if verbose:
                print(f"  [API error step {step_count + 1}] {exc}")

        # ---- Parse JSON — attempt 1 ----
        action_dict = safe_parse_action_json(raw_output)

        # ---- Retry once with correction prompt if parse failed ----
        if action_dict is None and raw_output:
            if verbose:
                print(f"  [Step {step_count + 1}] JSON parse failed, retrying with correction prompt...")
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": CORRECTION_PROMPT})
            try:
                retry_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=256,
                )
                retry_raw = retry_response.choices[0].message.content.strip()
                action_dict = safe_parse_action_json(retry_raw)
                if action_dict:
                    raw_output = retry_raw
            except Exception as exc:
                if verbose:
                    print(f"  [Retry API error] {exc}")

        # ---- Fallback: deterministic safe action ----
        used_fallback = False
        if action_dict is None:
            action_dict = _smart_fallback(obs_dict)
            used_fallback = True
            if verbose:
                print(f"  [Step {step_count + 1}] Using smart fallback: {action_dict}")

        # ---- Record assistant message ----
        messages.append({"role": "assistant", "content": json.dumps(action_dict)})

        # ---- Construct Action model safely ----
        try:
            action_obj = Action(**action_dict)
        except Exception as exc:
            if verbose:
                print(f"  [Action parse error] {exc} — applying smart fallback")
            action_dict = _smart_fallback(obs_dict)
            action_obj = Action(**action_dict)
            used_fallback = True

        # ---- Execute step ----
        step_response = env.step(action_obj)
        obs = step_response.observation
        obs_dict = obs.model_dump()
        reward = step_response.reward.reward
        done = step_response.done
        step_info = step_response.info

        total_reward += reward
        step_count += 1

        # Track violations from step trace
        step_violations = step_info.get("violations", [])
        violations += len(step_violations)
        if step_info.get("invalid_action"):
            invalid_actions += 1

        history.append({
            "step": step_count,
            "action": action_dict,
            "reward": round(reward, 4),
            "subgoals": step_response.reward.subgoals_achieved,
            "violations": step_violations,
            "decision_trace": step_info.get("decision_trace", []),
            "used_fallback": used_fallback,
        })

        if verbose:
            print(
                f"  Step {step_count:2d}: {action_dict.get('action_type'):25s} "
                f"reward={reward:+.3f}  done={done}"
            )

    # ---- Grade episode ----
    final_state = env.state()
    grader_output = grade(task_id, final_state)

    return {
        "task_id": task_id,
        "steps": step_count,
        "total_reward": round(total_reward, 4),
        "grader_score": grader_output.final_score,
        "grader_verdict": grader_output.verdict,
        "grader_breakdown": grader_output.breakdown,
        "grader_notes": grader_output.notes,
        "failures": [f.model_dump() for f in grader_output.failures],
        "efficiency": {
            "steps_used": step_count,
            "violations": violations,
            "invalid_actions": invalid_actions,
        },
        "history": history,
    }


# ---------------------------------------------------------------------------
# Run all tasks — leaderboard-style output
# ---------------------------------------------------------------------------

def run_baseline(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the baseline agent across all 3 tasks in fixed order.

    Returns a leaderboard-style benchmark report:
    {
        "results": {"easy_billing_001": 0.92, ...},
        "average_score": 0.82,
        "efficiency": {"avg_steps": 7.0, "total_violations": 0, "total_invalid_actions": 1},
        "metadata": {"model": "gpt-4o-mini", "temperature": 0, "max_steps": 15},
        "details": [... per-task full results ...]
    }
    """
    from openai import OpenAI

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return {
            "error": (
                "OPENAI_API_KEY not set. "
                "Set it in your environment or pass api_key argument."
            )
        }

    client = OpenAI(api_key=key)
    env = TrustDeskEnv()
    task_ids = get_all_task_ids()  # fixed deterministic order

    per_task_results: List[Dict[str, Any]] = []

    for tid in task_ids:
        task_meta = get_task(tid)
        max_steps = task_meta.step_budget

        if verbose:
            print(f"\n{'='*64}")
            print(f"  Task: {tid}  [{task_meta.difficulty.upper()}]  (budget: {max_steps} steps)")
            print(f"{'='*64}")

        result = run_task_episode(
            env, tid, client, model=model, max_steps=max_steps, verbose=verbose
        )
        per_task_results.append(result)

        if verbose:
            print(
                f"\n  ✓ Score: {result['grader_score']:.3f} "
                f"({result['grader_verdict']})  |  "
                f"Steps: {result['steps']}  |  "
                f"Violations: {result['efficiency']['violations']}"
            )
            if result["failures"]:
                print("  Failures:")
                for f in result["failures"]:
                    print(f"    [{f['failure_type']}] {f['detail'][:80]}")

    # ---- Aggregate ----
    scores = [r["grader_score"] for r in per_task_results]
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    total_steps = sum(r["steps"] for r in per_task_results)
    total_violations = sum(r["efficiency"]["violations"] for r in per_task_results)
    total_invalid = sum(r["efficiency"]["invalid_actions"] for r in per_task_results)
    completed = sum(1 for r in per_task_results if r["grader_score"] >= 0.65)

    if verbose:
        print(f"\n{'='*64}")
        print(f"  BENCHMARK SUMMARY")
        print(f"  Average score : {avg_score:.3f}")
        print(f"  Completion    : {completed}/{len(task_ids)} tasks scored ≥ 0.65 (Good)")
        print(f"  Total steps   : {total_steps}")
        print(f"  Violations    : {total_violations}")
        print(f"  Invalid acts  : {total_invalid}")
        print(f"{'='*64}")

    return {
        "results": {r["task_id"]: r["grader_score"] for r in per_task_results},
        "average_score": avg_score,
        "efficiency": {
            "avg_steps": round(total_steps / len(task_ids), 2) if task_ids else 0,
            "total_violations": total_violations,
            "total_invalid_actions": total_invalid,
            "completion_rate": f"{completed}/{len(task_ids)}",
        },
        "metadata": {
            "model": model,
            "temperature": 0,
            "max_steps_per_task": "per-task budget",
            "task_order": task_ids,
        },
        "per_task_scores": {r["task_id"]: r["grader_score"] for r in per_task_results},
        "per_task_verdicts": {r["task_id"]: r["grader_verdict"] for r in per_task_results},
        "details": per_task_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_baseline(verbose=True)
    summary = {k: v for k, v in result.items() if k != "details"}
    print("\n--- Final Benchmark Report ---")
    print(json.dumps(summary, indent=2))
