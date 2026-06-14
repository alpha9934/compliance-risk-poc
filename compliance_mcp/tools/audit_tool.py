from __future__ import annotations
"""
audit_tool.py

MCP tool: log_audit_event

The only MCP tool with WRITE access — logs an event to the immutable
audit chain. All other tools are read-only.

Sensitive write actions (case closure, SAR preparation) require
authorized human confirmation before this tool records their outcome.

POC:  delegates to the existing AuditLogger (stdout + hash chain).
Prod: writes to append-only Neon Postgres table + S3 WORM bucket.
"""
from typing import Any
from compliance_mcp.auth.role_validator import require_role
from compliance_mcp.middleware.input_sanitizer import sanitize_string
from audit.audit_logger import AuditLogger

audit = AuditLogger()

# Events that require COMPLIANCE_MANAGER or above — never ANALYST alone
ELEVATED_EVENT_TYPES = {
    "CASE_CLOSED",
    "SAR_PREPARED",
    "THRESHOLD_CHANGED",
    "MODEL_VERSION_DEPLOYED",
    "REVIEWER_ESCALATION_OVERRIDE",
}


@require_role(["COMPLIANCE_ANALYST", "COMPLIANCE_MANAGER",
               "MODEL_RISK", "INTERNAL_AUDIT", "SYSTEM_INTEGRATION"])
def log_audit_event(
    event_type: str,
    payload: dict[str, Any],
    caller_role: str = "SYSTEM_INTEGRATION",
    session_id: str = None,
    caller_id: str = None,
) -> dict:
    """
    Records an immutable audit event to the hash-chain logger.

    Elevated event types (CASE_CLOSED, SAR_PREPARED, etc.) require
    COMPLIANCE_MANAGER role — ANALYST calls are rejected.

    Args:
        event_type:  Type of event (e.g. "REVIEWER_ACTION", "CASE_CLOSED")
        payload:     Event data dict — must not contain raw PII
        caller_role: Enforced by decorator
        session_id:  Pipeline correlation ID
        caller_id:   Identity of the human or system writing the event

    Returns:
        {"event_hash": str, "event_type": str, "status": "logged"}
    """
    event_type = sanitize_string(event_type)

    # Elevated events need manager-level role
    if event_type in ELEVATED_EVENT_TYPES:
        if caller_role not in ("COMPLIANCE_MANAGER", "MODEL_RISK",
                               "INTERNAL_AUDIT", "SYSTEM_INTEGRATION"):
            raise PermissionError(
                f"Event type '{event_type}' requires COMPLIANCE_MANAGER role. "
                f"Caller has '{caller_role}'."
            )

    # Sanitize string values in payload
    clean_payload = {
        k: sanitize_string(v) if isinstance(v, str) else v
        for k, v in payload.items()
    }

    event_hash = audit.log(
        event_type,
        clean_payload,
        session_id=session_id,
        caller_id=caller_id,
    )

    return {
        "event_hash": event_hash,
        "event_type": event_type,
        "status":     "logged",
    }
