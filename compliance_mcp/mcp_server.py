from __future__ import annotations
"""
mcp_server.py

FastMCP server exposing all 6 compliance tools over HTTP.

Run as sidecar alongside FastAPI:
  uvicorn compliance_mcp.mcp_server:mcp_app --port 8001

Requires: pip install "mcp[cli]"
"""
import logging

logger = logging.getLogger(__name__)

# ── Import FastMCP — try multiple paths across mcp SDK versions ───────────
_FastMCP = None
_import_error = None

try:
    from mcp.server.fastmcp import FastMCP as _FastMCP
except ImportError:
    pass

if _FastMCP is None:
    try:
        from fastmcp import FastMCP as _FastMCP
    except ImportError:
        pass

if _FastMCP is None:
    raise ImportError(
        "FastMCP not found. Install with: pip install 'mcp[cli]'\n"
        "Tried: mcp.server.fastmcp, fastmcp"
    )

# ── Tool implementations ──────────────────────────────────────────────────
from compliance_mcp.tools.transaction_tool   import get_transaction_context
from compliance_mcp.tools.customer_tool      import get_customer_risk
from compliance_mcp.tools.policy_tool        import retrieve_policy
from compliance_mcp.tools.sanctions_tool     import check_sanctions
from compliance_mcp.tools.case_similarity_tool import get_similar_cases
from compliance_mcp.tools.audit_tool         import log_audit_event

# ── Server ────────────────────────────────────────────────────────────────
mcp = _FastMCP(
    name="compliance-risk-mcp",
    instructions=(
        "Compliance Risk MCP server. All tools require caller_role. "
        "Use SYSTEM_INTEGRATION for automated pipeline calls. "
        "Every call is audit-logged."
    ),
)


@mcp.tool()
def transaction_context(
    transaction_id: str,
    caller_role: str = "COMPLIANCE_ANALYST",
    session_id: str = "",
) -> dict:
    """
    Returns permitted transaction facts: amount, currency, channel,
    jurisdictions, timestamp. PII excluded. Requires COMPLIANCE_ANALYST+.
    """
    return get_transaction_context(
        transaction_id=transaction_id,
        caller_role=caller_role,
        session_id=session_id or None,
    )


@mcp.tool()
def customer_risk(
    customer_id: str,
    caller_role: str = "COMPLIANCE_ANALYST",
    session_id: str = "",
) -> dict:
    """
    Returns KYC risk attributes: risk_rating (1-5), prior_alert_count,
    account_age_days. Name and address excluded. Requires COMPLIANCE_ANALYST+.
    """
    return get_customer_risk(
        customer_id=customer_id,
        caller_role=caller_role,
        session_id=session_id or None,
    )


@mcp.tool()
def policy_retrieval(
    policy_type: str,
    jurisdiction: str = "ALL",
    product: str = "ALL",
    caller_role: str = "COMPLIANCE_ANALYST",
    session_id: str = "",
) -> list:
    """
    Returns approved policy passages from the versioned policy repository.
    policy_type: AML | KYC. Cite ONLY these passages — never invent references.
    """
    return retrieve_policy(
        policy_type=policy_type,
        jurisdiction=jurisdiction,
        product=product,
        caller_role=caller_role,
        session_id=session_id or None,
    )


@mcp.tool()
def sanctions_check(
    entity_id: str,
    jurisdiction: str = "",
    caller_role: str = "COMPLIANCE_ANALYST",
    session_id: str = "",
) -> dict:
    """
    Checks entity against OFAC, UN, EU watchlists.
    Returns match_found, confidence (0-1), watchlist_source.
    Requires COMPLIANCE_ANALYST+.
    """
    return check_sanctions(
        entity_id=entity_id,
        jurisdiction=jurisdiction or None,
        caller_role=caller_role,
        session_id=session_id or None,
    )


@mcp.tool()
def similar_cases(
    jurisdiction_destination: str = "",
    channel: str = "",
    risk_class: str = "",
    top_k: int = 3,
    caller_role: str = "COMPLIANCE_ANALYST",
    session_id: str = "",
) -> list:
    """
    Returns up to top_k historical cases with case_id, disposition,
    key_signal. Use for consistency checks. Requires COMPLIANCE_ANALYST+.
    """
    return get_similar_cases(
        jurisdiction_destination=jurisdiction_destination or None,
        channel=channel or None,
        risk_class=risk_class or None,
        top_k=top_k,
        caller_role=caller_role,
        session_id=session_id or None,
    )


@mcp.tool()
def audit_event(
    event_type: str,
    payload: dict,
    caller_role: str = "SYSTEM_INTEGRATION",
    session_id: str = "",
    caller_id: str = "",
) -> dict:
    """
    Records an immutable audit event (only write-capable tool).
    Elevated types (CASE_CLOSED, SAR_PREPARED) require COMPLIANCE_MANAGER.
    Returns event_hash for chain verification.
    """
    return log_audit_event(
        event_type=event_type,
        payload=payload,
        caller_role=caller_role,
        session_id=session_id or None,
        caller_id=caller_id or None,
    )


# ── ASGI app ──────────────────────────────────────────────────────────────
mcp_app = mcp.streamable_http_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(mcp_app, host="0.0.0.0", port=8001)
