import json
import os
from mcp.auth.role_validator import require_role
from mcp.middleware.input_sanitizer import sanitize_string
from audit.audit_logger import AuditLogger

audit = AuditLogger()
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "../../fixtures")


@require_role(["COMPLIANCE_ANALYST", "COMPLIANCE_MANAGER"])
def check_sanctions(
    entity_id: str = None,
    jurisdiction: str = None,
    caller_role: str = "COMPLIANCE_ANALYST",
    session_id: str = None,
) -> dict:
    """
    Checks mock sanctions watchlist.
    POC: reads from fixtures/sanctions_mock.json
    Production: calls OFAC / UN / EU / internal watchlist APIs
    NOTE: name is NOT logged — PII minimization
    """
    audit.log("MCP_TOOL_CALL", {
        "tool": "check_sanctions",
        "params": {"entity_id": entity_id, "jurisdiction": jurisdiction},
        # name intentionally excluded from audit log
    }, session_id=session_id)

    fixture_path = os.path.join(FIXTURES_DIR, "sanctions_mock.json")
    with open(fixture_path) as f:
        watchlist = json.load(f)

    match = next(
        (e for e in watchlist.get("entries", []) if e.get("entity_id") == entity_id),
        None,
    )
    result = {
        "match_found": match is not None,
        "confidence": match.get("confidence", 0.0) if match else 0.0,
        "watchlist_source": match.get("source", None) if match else None,
        "recommended_severity": "HIGH" if match and match.get("confidence", 0) > 0.8 else "LOW",
    }

    audit.log("MCP_TOOL_RESULT", {
        "tool": "check_sanctions",
        "match_found": result["match_found"],
        "confidence": result["confidence"],
    }, session_id=session_id)

    return result
