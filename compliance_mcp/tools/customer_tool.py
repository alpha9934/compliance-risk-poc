from __future__ import annotations
"""
customer_tool.py

MCP tool: get_customer_risk

Returns permitted KYC attributes and risk signals for a customer_id.
PII fields (name, address, DOB) are never returned — field minimisation.

POC:  reads from the feature store customer profiles.
Prod: queries KYC / CRM system via approved integration.
"""
from compliance_mcp.auth.role_validator import require_role
from compliance_mcp.middleware.input_sanitizer import sanitize_string
from compliance_mcp.middleware.output_filter import filter_output
from audit.audit_logger import AuditLogger

audit = AuditLogger()

PERMITTED_FIELDS = {
    "customer_id", "risk_rating", "prior_alert_count",
    "account_age_days", "onboarding_status", "occupation_category",
}


@require_role(["COMPLIANCE_ANALYST", "COMPLIANCE_MANAGER",
               "MODEL_RISK", "SYSTEM_INTEGRATION"])
def get_customer_risk(
    customer_id: str,
    caller_role: str = "COMPLIANCE_ANALYST",
    session_id: str = None,
) -> dict:
    """
    Returns permitted KYC risk attributes for a customer.
    Never exposes name, address, DOB, or passport details.

    Args:
        customer_id: Customer to look up
        caller_role: Enforced by role_validator decorator
        session_id:  Audit trail correlation ID

    Returns:
        Filtered customer risk profile dict
    """
    customer_id = sanitize_string(customer_id)

    audit.log("MCP_TOOL_CALL", {
        "tool":        "get_customer_risk",
        "customer_id": customer_id,
        "caller_role": caller_role,
    }, session_id=session_id)

    # Import here to avoid circular dependency at module load
    from features.redis_client import FeatureStoreClient
    store   = FeatureStoreClient()
    profile = store.get_customer_profile(customer_id)

    # Merge customer_id into the profile for completeness
    raw = {"customer_id": customer_id, **profile}
    result = filter_output(raw, permitted_fields=PERMITTED_FIELDS)

    audit.log("MCP_TOOL_RESULT", {
        "tool":            "get_customer_risk",
        "customer_id":     customer_id,
        "risk_rating":     result.get("risk_rating"),
        "prior_alerts":    result.get("prior_alert_count"),
    }, session_id=session_id)

    return result
