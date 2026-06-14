from __future__ import annotations
"""
output_parser.py

Parses and validates the raw JSON string returned by the LLM Judge.
Enforces the allowed values for recommendation and confidence fields.
Provides safe defaults if the LLM returns unexpected structure.
"""
import json
import logging

logger = logging.getLogger(__name__)

VALID_RECOMMENDATIONS = {"ESCALATE", "FLAG", "MONITOR", "CLOSE"}
VALID_CONFIDENCES     = {"HIGH", "MEDIUM", "LOW"}


def parse_judge_response(raw_text: str) -> dict:
    """
    Parses the raw LLM response string into a validated dict.

    Returns a safe default dict on any parse or validation error.
    Never raises — errors are logged and a safe fallback is returned.
    """
    # Strip markdown fences
    text = raw_text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]

    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as e:
        logger.error("[Judge] JSON parse error: %s | Raw: %s", e, raw_text[:200])
        return _safe_default("JSON parse failed")

    # Validate recommendation
    rec = parsed.get("recommendation", "FLAG").upper()
    if rec not in VALID_RECOMMENDATIONS:
        logger.warning("[Judge] Invalid recommendation '%s' — defaulting to FLAG", rec)
        rec = "FLAG"

    # Validate confidence
    conf = parsed.get("confidence", "LOW").upper()
    if conf not in VALID_CONFIDENCES:
        logger.warning("[Judge] Invalid confidence '%s' — defaulting to LOW", conf)
        conf = "LOW"

    # Validate supporting signals (must be exactly 3, strings)
    signals = parsed.get("supporting_signals", [])
    if not isinstance(signals, list):
        signals = []
    signals = [str(s) for s in signals[:3]]
    while len(signals) < 3:
        signals.append("Insufficient signal data")

    return {
        "recommendation":     rec,
        "confidence":         conf,
        "narrative":          str(parsed.get("narrative", ""))[:500],  # cap at 500 chars
        "supporting_signals": signals,
    }


def _safe_default(reason: str) -> dict:
    return {
        "recommendation":     "FLAG",
        "confidence":         "LOW",
        "narrative":          f"Advisory unavailable ({reason}). Default to standard review.",
        "supporting_signals": [
            "LLM parse error — manual review required",
            "Refer to ML score and SHAP values",
            "Check sanctions and KYC signals manually",
        ],
    }
