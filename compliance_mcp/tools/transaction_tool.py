from __future__ import annotations
"""
transaction_tool.py

MCP tool: get_transaction_context

Returns permitted transaction facts for a given transaction_id.
Fields are filtered — only what a compliance reviewer needs,
not raw banking system data.

POC:  reads from an in-memory dict seeded by the ingestion pipeline.
Prod: queries approved core banking API with field-level access controls.
"""
import os
import json
from compliance_mcp.auth.role_validator import require_role
from compliance_mcp.middleware.input_sanitizer import sanitize_string
from compliance_mcp.middleware.output_filter import filter_output
from audit.audit_logger import AuditLogger

audit = AuditLogger()

# POC in-memory store — populated by ingest_transaction()
_TRANSACTION_STORE: dict[str, dict] = {}

# Only these fields are returned to the LLM — nothing else
PERMITTED_FIELDS = {
    "transaction_id", "amount", "currency", "channel",
    "jurisdiction_origin", "jurisdiction_destination",
    "product_type", "timestamp", "status",
}


def ingest_transaction(event_dict: dict) -> None:
    """Called by the ingestion pipeline to register a transaction in the store."""
    _TRANSACTION_STORE[event_dict["transaction_id"]] = event_dict


@require_role(["COMPLIANCE_ANALYST", "COMPLIANCE_MANAGER",
               "MODEL_RISK", "SYSTEM_INTEGRATION"])
def get_transaction_context(
    transaction_id: str,
    caller_role: str = "COMPLIANCE_ANALYST",
    session_id: str = None,
) -> dict:
    """
    Returns filtered transaction facts for a given transaction_id.

    Args:
        transaction_id: The transaction to look up
        caller_role:    Role of the requesting party (enforced by decorator)
        session_id:     Audit trail correlation ID

    Returns:
        Dict of permitted transaction fields, or empty dict if not found.
    """
    transaction_id = sanitize_string(transaction_id)

    audit.log("MCP_TOOL_CALL", {
        "tool":           "get_transaction_context",
        "transaction_id": transaction_id,
        "caller_role":    caller_role,
    }, session_id=session_id)

    raw = _TRANSACTION_STORE.get(transaction_id, {})
    result = filter_output(raw, permitted_fields=PERMITTED_FIELDS)

    audit.log("MCP_TOOL_RESULT", {
        "tool":           "get_transaction_context",
        "transaction_id": transaction_id,
        "found":          bool(raw),
        "fields_returned": list(result.keys()),
    }, session_id=session_id)

    return result
