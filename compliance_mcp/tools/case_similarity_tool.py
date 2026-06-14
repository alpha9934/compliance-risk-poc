from __future__ import annotations
"""
case_similarity_tool.py

MCP tool: get_similar_cases

Returns historical compliance cases similar to the current transaction,
with their final dispositions and key signals.

This gives the LLM Judge context like:
  "Three similar high-value wires to AE in the past 90 days were all
   escalated and confirmed as SAR-reportable."

POC:  returns from a small hardcoded fixture of realistic cases.
Prod: queries case history DB with vector similarity on feature embeddings.
"""
from compliance_mcp.auth.role_validator import require_role
from compliance_mcp.middleware.output_filter import filter_output
from audit.audit_logger import AuditLogger

audit = AuditLogger()

PERMITTED_FIELDS = {
    "case_id", "risk_class", "disposition",
    "key_signal", "jurisdiction_destination", "channel",
}

# POC fixture — representative historical cases
# In production these come from the case history database
_CASE_FIXTURES = [
    {
        "case_id":                  "CASE-HIST-001",
        "risk_class":               "HIGH",
        "disposition":              "ESCALATED",
        "key_signal":               "Large wire to AE, FATF greylist",
        "jurisdiction_destination": "AE",
        "channel":                  "WIRE",
        "amount_range":             "50K-200K",
    },
    {
        "case_id":                  "CASE-HIST-002",
        "risk_class":               "HIGH",
        "disposition":              "SAR_FILED",
        "key_signal":               "SWIFT to RU, sanctions-adjacent entity",
        "jurisdiction_destination": "RU",
        "channel":                  "SWIFT",
        "amount_range":             "100K+",
    },
    {
        "case_id":                  "CASE-HIST-003",
        "risk_class":               "MEDIUM",
        "disposition":              "CLOSED_FALSE_POSITIVE",
        "key_signal":               "ACH velocity spike, legitimate payroll",
        "jurisdiction_destination": "US",
        "channel":                  "ACH",
        "amount_range":             "10K-50K",
    },
    {
        "case_id":                  "CASE-HIST-004",
        "risk_class":               "HIGH",
        "disposition":              "ESCALATED",
        "key_signal":               "New beneficiary, round amount, night transaction",
        "jurisdiction_destination": "AE",
        "channel":                  "WIRE",
        "amount_range":             "50K-200K",
    },
    {
        "case_id":                  "CASE-HIST-005",
        "risk_class":               "HIGH",
        "disposition":              "SAR_FILED",
        "key_signal":               "Structuring pattern below CTR threshold",
        "jurisdiction_destination": "US",
        "channel":                  "ACH",
        "amount_range":             "9K-10K",
    },
]


@require_role(["COMPLIANCE_ANALYST", "COMPLIANCE_MANAGER",
               "MODEL_RISK", "SYSTEM_INTEGRATION"])
def get_similar_cases(
    jurisdiction_destination: str = None,
    channel: str = None,
    risk_class: str = None,
    top_k: int = 3,
    caller_role: str = "COMPLIANCE_ANALYST",
    session_id: str = None,
) -> list[dict]:
    """
    Returns up to top_k historical cases similar to the given signals.

    Matching priority:
      1. Same jurisdiction_destination AND channel
      2. Same jurisdiction_destination only
      3. Same risk_class only
      4. Most recent (fixture order)

    Args:
        jurisdiction_destination: Filter by destination jurisdiction
        channel:                  Filter by payment channel
        risk_class:               Filter by risk class (HIGH/MEDIUM/LOW)
        top_k:                    Max cases to return (default 3)
        caller_role:              Enforced by decorator
        session_id:               Audit trail correlation ID

    Returns:
        List of filtered historical case dicts
    """
    audit.log("MCP_TOOL_CALL", {
        "tool":                    "get_similar_cases",
        "jurisdiction_destination": jurisdiction_destination,
        "channel":                 channel,
        "risk_class":              risk_class,
        "top_k":                   top_k,
    }, session_id=session_id)

    candidates = _CASE_FIXTURES.copy()

    # Score each case by match quality
    def score(case: dict) -> int:
        s = 0
        if jurisdiction_destination and case.get("jurisdiction_destination") == jurisdiction_destination:
            s += 2
        if channel and case.get("channel") == channel:
            s += 2
        if risk_class and case.get("risk_class") == risk_class:
            s += 1
        return s

    ranked  = sorted(candidates, key=score, reverse=True)
    results = [
        filter_output(c, permitted_fields=PERMITTED_FIELDS)
        for c in ranked[:top_k]
    ]

    audit.log("MCP_TOOL_RESULT", {
        "tool":          "get_similar_cases",
        "cases_returned": len(results),
        "case_ids":      [r.get("case_id") for r in results],
    }, session_id=session_id)

    return results
