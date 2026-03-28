"""
Business policy rules for TrustDeskEnv.
Defines refund policies, routing rules, SLA thresholds, and safety constraints.
"""

from __future__ import annotations

from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# Refund Policy
# ---------------------------------------------------------------------------

REFUND_WINDOW_DAYS: Dict[str, int] = {
    "free": 14,
    "premium": 30,
    "enterprise": 60,
}

REFUND_GRACE_EXTENSION_DAYS: Dict[str, int] = {
    "free": 0,
    "premium": 7,   # grace period for premium
    "enterprise": 14,
}


def refund_eligible(days_since_purchase: int, tier: str) -> bool:
    """Return True if refund is within policy window."""
    window = REFUND_WINDOW_DAYS.get(tier, 14)
    return days_since_purchase <= window


def refund_grace_eligible(days_since_purchase: int, tier: str) -> bool:
    """Return True if within grace-period extension (requires manager approval)."""
    window = REFUND_WINDOW_DAYS.get(tier, 14)
    grace = REFUND_GRACE_EXTENSION_DAYS.get(tier, 0)
    return window < days_since_purchase <= (window + grace)


# ---------------------------------------------------------------------------
# SLA / Step Budgets
# ---------------------------------------------------------------------------

STEP_BUDGET: Dict[str, int] = {
    "easy": 10,
    "medium": 12,
    "hard": 15,
}

SLA_RESPONSE_MINUTES: Dict[str, int] = {
    "free": 1440,       # 24h
    "premium": 240,     # 4h
    "enterprise": 60,   # 1h
}

# ---------------------------------------------------------------------------
# Team Routing Rules
# ---------------------------------------------------------------------------

# Maps (category, risk_level) -> preferred team
ROUTING_TABLE: Dict[tuple, str] = {
    ("billing", "none"): "billing",
    ("billing", "low"): "billing",
    ("billing", "medium"): "billing",
    ("billing", "high"): "trust_and_safety",
    ("billing", "critical"): "trust_and_safety",
    ("refund", "none"): "refunds",
    ("refund", "low"): "refunds",
    ("refund", "medium"): "customer_success",
    ("refund", "high"): "escalations",
    ("account_security", "none"): "account_security",
    ("account_security", "low"): "account_security",
    ("account_security", "medium"): "account_security",
    ("account_security", "high"): "trust_and_safety",
    ("account_security", "critical"): "trust_and_safety",
    ("cancellation", "none"): "customer_success",
    ("cancellation", "low"): "customer_success",
    ("cancellation", "medium"): "customer_success",
    ("cancellation", "high"): "trust_and_safety",
    ("fraud", "high"): "trust_and_safety",
    ("fraud", "critical"): "trust_and_safety",
    ("technical", "none"): "general_support",
    ("general", "none"): "general_support",
}


def get_preferred_team(category: str, risk: str) -> str:
    """Look up preferred team from routing table."""
    key = (category, risk)
    return ROUTING_TABLE.get(key, "general_support")


# ---------------------------------------------------------------------------
# Priority Rules
# ---------------------------------------------------------------------------

PRIORITY_RULES: Dict[str, str] = {
    "critical_risk_any_tier": "critical",
    "high_risk_enterprise": "critical",
    "high_risk_premium": "high",
    "high_risk_free": "high",
    "medium_risk_enterprise": "high",
    "medium_risk_premium": "medium",
    "billing_enterprise": "high",
    "security_any": "critical",
    "default": "medium",
}


def get_expected_priority(risk: str, tier: str, category: str) -> str:
    """Derive expected priority from risk/tier/category."""
    if category == "account_security" or risk == "critical":
        return "critical"
    if risk == "high":
        return "high" if tier in ("premium", "enterprise") else "medium"
    if risk == "medium" and tier == "enterprise":
        return "high"
    if category == "billing" and tier == "enterprise":
        return "high"
    return "medium"


# ---------------------------------------------------------------------------
# Unsafe / Disallowed Phrases
# ---------------------------------------------------------------------------

UNSAFE_PROMISE_PHRASES: List[str] = [
    "we will immediately refund",
    "i will refund you now",
    "your account is safe",         # cannot promise before verification
    "we guarantee a full refund",
    "cancellation is done",
    "refund has been processed",    # before actual processing
    "your money is on the way",
    "i promise to refund",
    "your account has been restored",  # before security verification
]

REQUIRED_VERIFICATION_ACKNOWLEDGMENTS: List[str] = [
    "verify",
    "verification",
    "confirm your identity",
    "security check",
    "secure your account",
]


def contains_unsafe_promise(message: str) -> bool:
    """Return True if the message contains a policy-violating promise."""
    lower = message.lower()
    return any(phrase in lower for phrase in UNSAFE_PROMISE_PHRASES)


def contains_verification_language(message: str) -> bool:
    """Return True if message appropriately mentions verification."""
    lower = message.lower()
    return any(phrase in lower for phrase in REQUIRED_VERIFICATION_ACKNOWLEDGMENTS)


# ---------------------------------------------------------------------------
# Resolved-before-verified check
# ---------------------------------------------------------------------------

ACTIONS_REQUIRING_PRIOR_VERIFICATION: Set[str] = {
    "offer_resolution",
    "mark_resolved",
}

ACTIONS_REQUIRING_CLASSIFICATION_FIRST: Set[str] = {
    "set_priority",
    "assign_team",
    "offer_resolution",
    "escalate",
    "draft_reply",
    "mark_resolved",
}
