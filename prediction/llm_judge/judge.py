from __future__ import annotations
"""
judge.py

Calls Groq (Llama 3.1 70B) to provide a holistic advisory recommendation
for HIGH and MEDIUM risk transactions.

Role in the system:
  - Input:  Full case context (ML score, explainer output, customer, sanctions, similar cases)
  - Output: LLMJudgeOutput with recommendation + narrative + 3 supporting signals
  - Advisory only — output is NEVER used to auto-close or auto-escalate cases

Falls back gracefully when GROQ_API_KEY is not set.
"""
import os
import json
import hashlib
import logging
from datetime import datetime, timezone
from pydantic import BaseModel

from prediction.llm_judge.prompt_builder import build_judge_prompt, JUDGE_SYSTEM_PROMPT
from prediction.llm_judge.output_parser import parse_judge_response
from prediction.ml_scorer.xgboost_scorer import ScorerOutput
from prediction.llm_explainer.explainer import LLMExplainerOutput

logger = logging.getLogger(__name__)


# ── Output model ──────────────────────────────────────────────────────────

class LLMJudgeOutput(BaseModel):
    transaction_id: str
    recommendation: str             # ESCALATE | FLAG | MONITOR | CLOSE
    confidence: str                 # HIGH | MEDIUM | LOW
    narrative: str
    supporting_signals: list[str]   # always exactly 3
    model_id: str
    prompt_hash: str
    generated_at: datetime
    fallback_used: bool = False
    fallback_reason: str = ""        # NO_API_KEY | API_ERROR: <msg> | ""


# ── Groq client (lazy init) ───────────────────────────────────────────────

def _get_groq_client():
    """Returns a Groq client or None if GROQ_API_KEY not set."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception as e:
        logger.error("Failed to initialise Groq client: %s", e)
        return None


# ── Fallback ──────────────────────────────────────────────────────────────

def _fallback_judge(
    scorer_output: ScorerOutput,
    prompt_hash: str,
    reason: str = "NO_API_KEY",
) -> LLMJudgeOutput:
    """
    Heuristic advisory when Groq is unavailable.
    Mirrors what a judge would likely recommend given the risk class.
    """
    from ingestion.schemas.transaction_event import RiskClass
    rec = {
        RiskClass.HIGH:   "ESCALATE",
        RiskClass.MEDIUM: "FLAG",
        RiskClass.LOW:    "MONITOR",
    }.get(scorer_output.risk_class, "FLAG")

    return LLMJudgeOutput(
        transaction_id=scorer_output.transaction_id,
        recommendation=rec,
        confidence="LOW",
        narrative=(
            f"Heuristic advisory: ML score {scorer_output.risk_score:.3f} "
            f"({scorer_output.risk_class.value}). "
            f"[{reason}]"
        ),
        supporting_signals=[
            f"ML risk class: {scorer_output.risk_class.value}",
            f"Risk score: {scorer_output.risk_score:.4f}",
            f"Full advisory unavailable — {reason}",
        ],
        model_id="fallback-heuristic",
        prompt_hash=prompt_hash,
        generated_at=datetime.now(timezone.utc),
        fallback_used=True,
        fallback_reason=reason,
    )


# ── Main judge function ───────────────────────────────────────────────────

def run_judge(
    scorer_output: ScorerOutput,
    explainer_output: LLMExplainerOutput,
    customer_context: dict,
    sanctions_result: dict,
    similar_cases: list[dict],
    transaction_context: dict,
) -> LLMJudgeOutput:
    """
    Calls Groq / Llama 70B for holistic case advisory.

    Args:
        scorer_output:        ML scorer output
        explainer_output:     LLM Explainer output
        customer_context:     From MCP get_customer_risk
        sanctions_result:     From MCP check_sanctions
        similar_cases:        From MCP get_similar_cases
        transaction_context:  From MCP get_transaction_context

    Returns:
        LLMJudgeOutput — advisory recommendation for human reviewer
    """
    user_prompt = build_judge_prompt(
        scorer_output, explainer_output, customer_context,
        sanctions_result, similar_cases, transaction_context,
    )
    prompt_hash = hashlib.sha256(user_prompt.encode()).hexdigest()

    client = _get_groq_client()
    if client is None:
        reason = "NO_API_KEY" if not os.getenv("GROQ_API_KEY") else "CLIENT_INIT_FAILED"
        logger.warning(
            "[Judge] Groq client unavailable (%s) — using fallback for txn %s",
            reason, scorer_output.transaction_id,
        )
        return _fallback_judge(scorer_output, prompt_hash, reason=reason)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content
        parsed   = parse_judge_response(raw_text)

        return LLMJudgeOutput(
            transaction_id=scorer_output.transaction_id,
            recommendation=parsed["recommendation"],
            confidence=parsed["confidence"],
            narrative=parsed["narrative"],
            supporting_signals=parsed["supporting_signals"],
            model_id="llama-3.3-70b-versatile",
            prompt_hash=prompt_hash,
            generated_at=datetime.now(timezone.utc),
            fallback_used=False,
        )

    except Exception as e:
        logger.error(
            "[Judge] Groq call failed for txn %s: %s",
            scorer_output.transaction_id, e,
        )
        return _fallback_judge(scorer_output, prompt_hash, reason=f"API_ERROR: {e}")
