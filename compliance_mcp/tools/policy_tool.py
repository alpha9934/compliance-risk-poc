import json
import os
from compliance_mcp.auth.role_validator import require_role
from compliance_mcp.middleware.input_sanitizer import sanitize_string
from audit.audit_logger import AuditLogger

audit = AuditLogger()
POLICY_DIR = os.path.join(os.path.dirname(__file__), "../../policies")


@require_role(["COMPLIANCE_ANALYST", "COMPLIANCE_MANAGER", "MODEL_RISK", "SYSTEM_INTEGRATION"])
def retrieve_policy(
    policy_type: str,
    jurisdiction: str = "DEFAULT",
    product: str = "DEFAULT",
    caller_role: str = "COMPLIANCE_ANALYST",
    session_id: str = None,
) -> list[dict]:
    """
    Returns approved policy passages from versioned JSON files.
    POC: reads from /policies/*.json
    Production: queries versioned policy repository API
    """
    policy_type = sanitize_string(policy_type)
    audit.log("MCP_TOOL_CALL", {
        "tool": "retrieve_policy",
        "params": {"policy_type": policy_type, "jurisdiction": jurisdiction},
    }, session_id=session_id)

    policy_file = os.path.join(POLICY_DIR, f"{policy_type.lower()}.json")
    if not os.path.exists(policy_file):
        return []

    with open(policy_file) as f:
        all_passages = json.load(f)

    results = [
        p for p in all_passages
        if p.get("jurisdiction") in (jurisdiction, "ALL")
        and p.get("product") in (product, "ALL")
    ]

    audit.log("MCP_TOOL_RESULT", {
        "tool": "retrieve_policy",
        "result_count": len(results),
    }, session_id=session_id)

    return results
