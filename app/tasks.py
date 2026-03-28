"""
Task definitions loader for TrustDeskEnv.
Each task is loaded from its JSON data file and wrapped in a Task dataclass.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from app.utils import load_task_data
from app.policies import STEP_BUDGET


@dataclass
class Task:
    """A single task loaded from the data directory."""

    task_id: str
    title: str
    difficulty: str
    description: str
    customer_message: str
    customer_tier: str
    account_status: str
    policy_context: Dict[str, Any]
    security_flags: List[str]
    prior_history: List[Dict[str, Any]]
    expected_outcome: Dict[str, Any]
    allowed_resolutions: List[str]
    disallowed_actions: List[str]
    grading_weights: Dict[str, float]

    @property
    def step_budget(self) -> int:
        return STEP_BUDGET.get(self.difficulty, 10)

    @property
    def ticket_id(self) -> str:
        return f"TKT-{self.task_id.upper()}"


def _load(filename: str) -> Task:
    data = load_task_data(filename)
    return Task(
        task_id=data["task_id"],
        title=data["title"],
        difficulty=data["difficulty"],
        description=data["description"],
        customer_message=data["customer_message"],
        customer_tier=data["customer_tier"],
        account_status=data["account_status"],
        policy_context=data["policy_context"],
        security_flags=data["security_flags"],
        prior_history=data["prior_history"],
        expected_outcome=data["expected_outcome"],
        allowed_resolutions=data["allowed_resolutions"],
        disallowed_actions=data["disallowed_actions"],
        grading_weights=data["grading_weights"],
    )


# Registry — loaded lazily when first accessed
_TASK_REGISTRY: Dict[str, Task] = {}

_TASK_FILES: Dict[str, str] = {
    "easy_billing_001": "easy_billing.json",
    "medium_refund_001": "medium_refund.json",
    "hard_security_001": "hard_security.json",
}

DEFAULT_TASK_ID = "easy_billing_001"


def get_task(task_id: Optional[str] = None) -> Task:
    """Return a Task by ID. Defaults to easy task if None."""
    tid = task_id or DEFAULT_TASK_ID
    if tid not in _TASK_REGISTRY:
        if tid not in _TASK_FILES:
            raise ValueError(
                f"Unknown task_id '{tid}'. Valid: {list(_TASK_FILES.keys())}"
            )
        _TASK_REGISTRY[tid] = _load(_TASK_FILES[tid])
    return _TASK_REGISTRY[tid]


def list_tasks() -> List[Task]:
    """Return all available tasks."""
    return [get_task(tid) for tid in _TASK_FILES]


def get_all_task_ids() -> List[str]:
    return list(_TASK_FILES.keys())
