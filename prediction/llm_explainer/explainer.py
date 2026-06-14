from __future__ import annotations
"""
explainer.py

Calls Google Gemini Flash to generate a plain-English compliance
explanation for a HIGH or MEDIUM risk transaction.

Role in the system:
  - Input:  ML scorer output (SHAP values) + MCP policy passages + tx context
  - Output: LLMExplainerOutput with explanation text, reason codes, policy citations
  - Has NO influence on the risk score — purely explains the ML model's decision

The LLM is given a strict JSON schema to follow and a hallucination check
runs on every response before the output is used downstream.

Falls back gracefully when GEMINI_API_KEY is not set (e.g. local dev):
returns a structured placeholder explanation so the rest of the pipeline
can still be tested end-to-end.
"""
import os
import json
import hashlib
import logging
from datetime import datetime, timezone
from pydantic import BaseModel

from prediction.llm_explainer.prompt_builder import build_explanation_prompt, SYSTEM_PROMPT
from prediction.llm_explainer.hallucination_check import (
    verify_policy_citations, check_reason_codes,
)
from prediction.ml_scorer.xgboost_scorer import ScorerOutput

logger = logging.getLogger(__name__)

# ── Output model ──────────────────────────────────────────────────────────

class LLMExplainerOutput(BaseModel):
    transaction_id: str
    explanation_text: str
    reason_codes: list[str]
    policy_citations: list[dict]
    model_id: str
    prompt_hash: str               # SHA-256 of user-turn prompt (for audit)
    generated_at: datetime
    fallback_used: bool = False    # True when API key not set or call failed
    fallback_reason: str = ""      # NO_API_KEY | API_ERROR: <msg> | INVALID_JSON: <msg> | ""


# ── Gemini client (lazy init) ─────────────────────────────────────────────

def _get_gemini_client():
    """Returns a configured Gemini GenerativeModel or None if key not set."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0.1,         # low temp → consistent structured output
                # gemini-2.5-flash has "thinking" enabled by default, and thinking
                # tokens are deducted from max_output_tokens. The deprecated
                # google.generativeai SDK's GenerationConfig protobuf does NOT
                # support a "thinking_config" field (raises "Unknown field" error),
                # so thinking cannot be disabled here. Instead, give a generous
                # token budget so thinking + the structured JSON both fit
                # comfortably without truncation.
                "max_output_tokens": 4096,
                "response_mime_type": "application/json",
            },
            system_instruction=SYSTEM_PROMPT,
        )
    except Exception as e:
        logger.error("Failed to initialise Gemini client: %s", e)
        return None


# ── Fallback explanation ──────────────────────────────────────────────────

def _fallback_explanation(
    scorer_output: ScorerOutput,
    prompt_hash: str,
    reason: str = "NO_API_KEY",
) -> LLMExplainerOutput:
    """
    Returns a structured placeholder when the LLM is unavailable.
    This keeps the pipeline runnable without API keys.
    """
    top = scorer_output.shap_explanation.top_features
    top_feature = top[0].feature if top else "unknown_feature"

    reason_codes = []
    if scorer_output.shap_explanation.top_features:
        for f in scorer_output.shap_explanation.top_features[:3]:
            if "jurisdiction" in f.feature or "risk_dest" in f.feature:
                reason_codes.append("HIGH_RISK_JURISDICTION")
            elif "zscore" in f.feature or "outlier" in f.feature:
                reason_codes.append("AMOUNT_DEVIATION")
            elif "fatf_blacklist" in f.feature:
                reason_codes.append("FATF_BLACKLIST")
            elif "fatf_greylist" in f.feature:
                reason_codes.append("FATF_GREYLIST")
            elif "new_beneficiary" in f.feature:
                reason_codes.append("NEW_BENEFICIARY")
            elif "structuring" in f.feature:
                reason_codes.append("STRUCTURING_PATTERN")
            elif "customer_risk" in f.feature or "prior_alert" in f.feature:
                reason_codes.append("HIGH_RISK_CUSTOMER")
    reason_codes = list(dict.fromkeys(reason_codes))[:3] or ["AMOUNT_DEVIATION"]

    text = (
        f"This transaction received a risk score of {scorer_output.risk_score:.3f} "
        f"({scorer_output.risk_class.value}). "
        f"The primary signal is '{top_feature}'. "
        f"[LLM explanation unavailable — {reason}]"
    )

    return LLMExplainerOutput(
        transaction_id=scorer_output.transaction_id,
        explanation_text=text,
        reason_codes=reason_codes,
        policy_citations=[],
        model_id="fallback-no-api-key",
        prompt_hash=prompt_hash,
        generated_at=datetime.now(timezone.utc),
        fallback_used=True,
        fallback_reason=reason,
    )


# ── Main explainer function ───────────────────────────────────────────────

def generate_explanation(
    scorer_output: ScorerOutput,
    policy_passages: list[dict],
    transaction_context: dict,
) -> LLMExplainerOutput:
    """
    Calls Gemini Flash to generate a reviewer-facing compliance explanation.

    Args:
        scorer_output:       Output from xgboost_scorer.score_transaction()
        policy_passages:     Passages from MCP retrieve_policy tool
        transaction_context: Dict from MCP get_transaction_context tool

    Returns:
        LLMExplainerOutput — verified, structured, ready for reviewer
    """
    # Build prompt and hash it for audit trail
    user_prompt = build_explanation_prompt(
        scorer_output, policy_passages, transaction_context
    )
    prompt_hash = hashlib.sha256(user_prompt.encode()).hexdigest()

    # Try Gemini
    client = _get_gemini_client()
    if client is None:
        reason = "NO_API_KEY" if not os.getenv("GEMINI_API_KEY") else "CLIENT_INIT_FAILED"
        logger.warning(
            "[Explainer] Gemini client unavailable (%s) — using fallback for txn %s",
            reason, scorer_output.transaction_id,
        )
        return _fallback_explanation(scorer_output, prompt_hash, reason=reason)

    try:
        response = client.generate_content(user_prompt)
        raw_text = response.text.strip()

        # Strip markdown fences if Gemini wraps in ```json ... ```
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        parsed = json.loads(raw_text)

        # Safety checks
        verified_citations = verify_policy_citations(
            parsed.get("policy_citations", []),
            policy_passages,
        )
        verified_codes = check_reason_codes(
            parsed.get("reason_codes", [])
        )

        return LLMExplainerOutput(
            transaction_id=scorer_output.transaction_id,
            explanation_text=parsed.get("explanation_text", ""),
            reason_codes=verified_codes,
            policy_citations=verified_citations,
            model_id="gemini-2.5-flash",
            prompt_hash=prompt_hash,
            generated_at=datetime.now(timezone.utc),
            fallback_used=False,
        )

    except json.JSONDecodeError as e:
        logger.error(
            "[Explainer] Gemini returned invalid JSON for txn %s: %s",
            scorer_output.transaction_id, e,
        )
        return _fallback_explanation(scorer_output, prompt_hash, reason=f"INVALID_JSON: {e}")

    except Exception as e:
        logger.error(
            "[Explainer] Gemini call failed for txn %s: %s",
            scorer_output.transaction_id, e,
        )
        return _fallback_explanation(scorer_output, prompt_hash, reason=f"API_ERROR: {e}")
