"""
FastAPI application for TrustDeskEnv.

OpenEnv-compatible HTTP API:
    GET  /        — environment metadata and endpoint map
    POST /reset   — initialize episode, return initial observation
    POST /step    — submit action, receive observation + reward + explainability trace
    GET  /state   — current internal environment state
    GET  /tasks   — enriched task list with competencies, max_steps, action schema
    GET  /grader  — deterministic grader score for current episode (with failure modes)
    POST /baseline — leaderboard-style benchmark run across all tasks
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.env import TrustDeskEnv, get_env
from app.graders import grade
from app.models import (
    Action,
    GraderOutput,
    ObservationModel,
    StateModel,
    StepResponse,
    TaskDetailInfo,
    TaskListResponse,
)
from app.tasks import list_tasks
from app.utils import action_schema


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TrustDeskEnv",
    description=(
        "**OpenEnv-compatible** enterprise customer support and trust & safety operations benchmark. "
        "AI agents navigate realistic multi-step support scenarios involving billing disputes, "
        "out-of-window refunds, and security breach triage under real policy constraints. "
        "Every step returns a rich explainability trace — violations, decision_trace, policy_checks, "
        "and subgoal progress — making agent behaviour fully inspectable."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Task competency metadata (drives /tasks response)
# ---------------------------------------------------------------------------

TASK_COMPETENCIES: Dict[str, list] = {
    "easy_billing_001": [
        "ticket classification",
        "priority assignment",
        "team routing",
        "customer reply drafting",
        "resolution disposition",
    ],
    "medium_refund_001": [
        "policy interpretation under time constraint",
        "multi-variable decision making (tier + window + goodwill history)",
        "identifying disallowed vs. allowed resolution paths",
        "compliant reply drafting",
        "escalation judgment",
    ],
    "hard_security_001": [
        "security risk prioritization over billing/cancellation",
        "adversarial action ordering (risk before team before resolution)",
        "identity verification enforcement",
        "unsafe promise detection in customer reply",
        "multi-signal conflict resolution (security + billing + cancellation)",
        "enterprise SLA compliance",
    ],
}


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class BaselineRequest(BaseModel):
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", summary="Environment metadata and endpoint map")
async def root() -> Dict[str, Any]:
    """
    Return metadata about TrustDeskEnv, available tasks, and endpoint reference.
    Useful for automated discovery and demo displays.
    """
    return {
        "name": "TrustDeskEnv",
        "version": "1.1.0",
        "openenv_compatible": True,
        "description": (
            "Policy-aware enterprise support and trust & safety simulation. "
            "An AI agent receives a customer ticket and must take multi-step, "
            "policy-compliant actions to resolve it correctly. "
            "Every step returns a rich explainability trace."
        ),
        "tasks": [
            {
                "task_id": "easy_billing_001",
                "difficulty": "easy",
                "title": "Duplicate Charge Complaint",
            },
            {
                "task_id": "medium_refund_001",
                "difficulty": "medium",
                "title": "Out-of-Window Refund Dispute",
            },
            {
                "task_id": "hard_security_001",
                "difficulty": "hard",
                "title": "Security Breach + Billing + Cancellation Conflict",
            },
        ],
        "endpoints": {
            "POST /reset": "Initialize episode. Body: {task_id: string|null}",
            "POST /step": "Submit action. Body: Action object.",
            "GET  /state": "Return full internal state (for debugging/grading).",
            "GET  /tasks": "List all tasks with competencies, max_steps, action schema.",
            "GET  /grader": "Score current episode with failure mode breakdown.",
            "POST /baseline": "Run leaderboard benchmark across all tasks.",
        },
        "example_step_action": {
            "action_type": "classify_ticket",
            "category": "billing",
        },
        "docs_url": "/docs",
        "openenv_yaml": "/openenv.yaml",
    }


@app.post("/reset", response_model=ObservationModel, summary="Reset environment")
async def reset(request: Optional[ResetRequest] = None) -> ObservationModel:
    """
    Initialize a new episode for the given `task_id` (or the default easy task).

    Returns the initial **ObservationModel** including:
    - customer message, tier, account status
    - policy context and security flags
    - available action types
    - step budget
    """
    request = request or ResetRequest()
    env = get_env()
    try:
        obs = env.reset(task_id=request.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs


@app.post("/step", response_model=StepResponse, summary="Submit an action")
async def step(action: Action) -> StepResponse:
    """
    Apply a structured **Action** to the current episode.

    Returns a **StepResponse** containing:
    - `observation`: updated environment view
    - `reward`: dense reward with breakdown and penalties
    - `done`: whether episode is finished
    - `info`: rich **StepTrace** with decision_trace, violations,
      policy_checks, subgoals_completed, loop_detected, invalid_action
    """
    env = get_env()
    try:
        result = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/state", response_model=StateModel, summary="Get internal environment state")
async def state() -> StateModel:
    """
    Return the full internal **StateModel** — useful for debugging, unit tests,
    and building custom graders.

    Includes all subgoal flags, violation counts, adversarial tracking,
    and cumulative reward so far.
    """
    env = get_env()
    try:
        return env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/tasks", response_model=TaskListResponse, summary="List all tasks")
async def tasks() -> TaskListResponse:
    """
    Return all available tasks with enriched metadata:
    - `title`, `difficulty`, `description`
    - `max_steps`: step budget for this task
    - `competencies`: skills the agent must demonstrate to score well
    - `action_schema`: full action type enumeration with field requirements
    """
    task_list = list_tasks()
    schema = action_schema()
    task_infos = [
        TaskDetailInfo(
            task_id=t.task_id,
            title=t.title,
            difficulty=t.difficulty,
            description=t.description,
            max_steps=t.step_budget,
            competencies=TASK_COMPETENCIES.get(t.task_id, []),
            action_schema=schema,
        )
        for t in task_list
    ]
    return TaskListResponse(tasks=task_infos, action_schema=schema)


@app.get("/grader", response_model=GraderOutput, summary="Grade current episode")
async def grader() -> GraderOutput:
    """
    Score the current (or completed) episode using the task-specific deterministic grader.

    Returns:
    - `final_score`: float in [0.0, 1.0]
    - `breakdown`: per-dimension weighted scores
    - `verdict`: Excellent / Good / Partial / Poor
    - `failures`: structured list of agent failures with type, detail, and penalty
    - `notes`: additional evaluator observations
    """
    env = get_env()
    try:
        current_state = env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        result = grade(current_state.task_id, current_state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.post("/baseline", summary="Run leaderboard benchmark across all tasks")
async def baseline(request: BaselineRequest) -> Dict[str, Any]:
    """
    Run the OpenAI-based baseline agent across all 3 tasks and return a
    **leaderboard-style benchmark report** including:
    - per-task scores
    - average score
    - efficiency metrics (avg steps, total violations, invalid actions)
    - model metadata

    Requires `OPENAI_API_KEY` in environment or `api_key` in request body.
    """
    from app.baseline import run_baseline

    api_key = request.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=(
                "OPENAI_API_KEY is required. "
                "Set it as an environment variable or pass api_key in the request body."
            ),
        )

    result = run_baseline(model=request.model, api_key=api_key, verbose=False)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
