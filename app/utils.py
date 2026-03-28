"""
Utility helpers for TrustDeskEnv.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


DATA_DIR = Path(__file__).parent.parent / "data"


def load_task_data(filename: str) -> Dict[str, Any]:
    """Load a JSON task data file from the data directory."""
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float value to [lo, hi]."""
    return max(lo, min(hi, value))


def safe_parse_action_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract a JSON object from raw model output.
    Returns None if parsing fails entirely.
    """
    # First try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON block with regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def normalize_score(raw: float, max_raw: float) -> float:
    """Normalize a raw score to [0.0, 1.0]."""
    if max_raw == 0:
        return 0.0
    return clamp(raw / max_raw)


def action_schema() -> Dict[str, Any]:
    """Return a JSON schema description of the Action model."""
    return {
        "action_type": {
            "type": "string",
            "enum": [
                "classify_ticket",
                "set_priority",
                "detect_risk",
                "assign_team",
                "request_verification",
                "offer_resolution",
                "escalate",
                "draft_reply",
                "mark_resolved",
            ],
            "description": "Required. The action to perform.",
        },
        "category": {
            "type": "string",
            "enum": [
                "billing", "refund", "account_security",
                "cancellation", "technical", "general", "fraud"
            ],
            "description": "Required for classify_ticket.",
        },
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
            "description": "Required for set_priority.",
        },
        "risk_label": {
            "type": "string",
            "enum": ["none", "low", "medium", "high", "critical"],
            "description": "Required for detect_risk.",
        },
        "team": {
            "type": "string",
            "enum": [
                "billing", "refunds", "account_security",
                "trust_and_safety", "customer_success",
                "general_support", "escalations",
            ],
            "description": "Required for assign_team.",
        },
        "resolution_code": {
            "type": "string",
            "enum": [
                "full_refund", "partial_refund", "credit_issued",
                "no_refund_policy", "account_unlocked", "account_suspended",
                "duplicate_charge_reversed", "escalated_to_specialist",
                "verification_required", "cancellation_processed",
                "goodwill_exception", "policy_declined", "fraud_review",
            ],
            "description": "Required for offer_resolution or mark_resolved.",
        },
        "message": {
            "type": "string",
            "description": "Required for draft_reply. The customer-facing message.",
        },
        "escalation_reason": {
            "type": "string",
            "description": "Required for escalate. Justification for escalation.",
        },
    }
