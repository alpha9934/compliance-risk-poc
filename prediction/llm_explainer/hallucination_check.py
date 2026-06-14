from __future__ import annotations
"""
hallucination_check.py

Verifies that every policy citation returned by the LLM Explainer
actually exists in the passages we supplied in the prompt.

This is the single most important safety check in the LLM pipeline.
If the LLM fabricates a policy reference (e.g. "AML-XX-99"), that
fake reference reaching a compliance reviewer would be a serious failure.

The check is intentionally strict: if a cited policy_id does not exactly
match any of the supplied passages, it is removed from the output and
logged as a hallucination. The explanation still proceeds — we never
silently pass bad citations to reviewers.
"""
import logging

logger = logging.getLogger(__name__)


def verify_policy_citations(
    claimed_citations: list[dict],
    supplied_passages: list[dict],
) -> list[dict]:
    """
    Filters out any citations not grounded in the supplied passages.

    Args:
        claimed_citations: List of {"policy_id": str, "passage": str}
                           returned by the LLM
        supplied_passages: List of policy dicts we sent in the prompt
                           (each must have "policy_id" key)

    Returns:
        Filtered list containing only verified citations.
        Hallucinated citations are dropped and logged.
    """
    supplied_ids = {p["policy_id"] for p in supplied_passages}
    verified = []

    for citation in claimed_citations:
        cited_id = citation.get("policy_id", "")
        if cited_id in supplied_ids:
            verified.append(citation)
        else:
            logger.warning(
                "HALLUCINATION DETECTED — LLM cited policy '%s' "
                "which was not in the supplied passages. Dropping citation.",
                cited_id,
            )

    if not verified and claimed_citations:
        logger.error(
            "ALL %d citation(s) from LLM were hallucinated. "
            "Returning empty citations list.",
            len(claimed_citations),
        )

    return verified


def check_reason_codes(
    claimed_codes: list[str],
    allowed_codes: set[str] | None = None,
) -> list[str]:
    """
    Ensures all reason codes are from the approved list.
    Unknown codes are dropped and logged.
    """
    if allowed_codes is None:
        allowed_codes = {
            "AML_VELOCITY", "AMOUNT_DEVIATION", "HIGH_RISK_JURISDICTION",
            "SANCTIONS_PROXIMITY", "PEP_RISK", "STRUCTURING_PATTERN",
            "NEW_BENEFICIARY", "CHANNEL_ANOMALY", "ROUND_AMOUNT",
            "PRIOR_ALERTS", "HIGH_RISK_CUSTOMER", "FATF_BLACKLIST",
            "FATF_GREYLIST",
        }

    verified = []
    for code in claimed_codes:
        if code in allowed_codes:
            verified.append(code)
        else:
            logger.warning("Unknown reason code '%s' dropped.", code)

    return verified
