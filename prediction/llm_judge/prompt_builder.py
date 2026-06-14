from __future__ import annotations
"""
prompt_builder.py  (LLM Judge)

Builds the full case context prompt sent to Groq / Llama 70B
for the holistic advisory recommendation.

The judge gets a richer context than the explainer:
  - ML score + top SHAP features
  - LLM Explainer's assessment
  - Customer risk profile
  - Sanctions check result
  - Similar historical cases with their dispositions

This wider context is what allows the judge to catch patterns
that the ML model misses (e.g. "three similar cases last month
all turned out to be true positives and were escalated").
"""
import os
from prediction.ml_scorer.xgboost_scorer import ScorerOutput
from prediction.llm_explainer.explainer import LLMExplainerOutput

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "../prompts/judge_v1.txt")

with open(_PROMPT_PATH) as f:
    JUDGE_SYSTEM_PROMPT = f.read().strip()


def build_judge_prompt(
    scorer_output: ScorerOutput,
    explainer_output: LLMExplainerOutput,
    customer_context: dict,
    sanctions_result: dict,
    similar_cases: list[dict],
    transaction_context: dict,
) -> str:
    """
    Builds the user-turn prompt for the LLM-as-Judge.

    Args:
        scorer_output:       ML scorer output with SHAP values
        explainer_output:    Output from the LLM Explainer
        customer_context:    Customer profile from MCP get_customer_risk
        sanctions_result:    Sanctions check from MCP check_sanctions
        similar_cases:       Historical cases from MCP get_similar_cases
        transaction_context: Transaction facts from MCP get_transaction_context

    Returns:
        User-turn prompt string
    """
    shap = scorer_output.shap_explanation
    tx   = transaction_context
    cust = customer_context
    sanc = sanctions_result

    # Format SHAP top features
    feature_lines = "\n".join(
        f"  - {f.feature}: {f.value} (contribution: {f.contribution:+.4f})"
        for f in shap.top_features
    )

    # Format similar cases
    if similar_cases:
        case_lines = "\n".join(
            f"  [{c.get('case_id','?')}] "
            f"risk={c.get('risk_class','?')}, "
            f"disposition={c.get('disposition','?')}, "
            f"key_signal={c.get('key_signal','?')}"
            for c in similar_cases[:3]
        )
    else:
        case_lines = "  No similar historical cases found."

    # Format sanctions
    sanctions_line = (
        f"MATCH FOUND — confidence={sanc.get('confidence',0):.2f}, "
        f"source={sanc.get('watchlist_source','?')}"
        if sanc.get("match_found")
        else "No sanctions match found."
    )

    return f"""TRANSACTION SUMMARY:
  Amount:      {tx.get('amount','?')} {tx.get('currency','')}
  Channel:     {tx.get('channel','?')}
  Origin:      {tx.get('jurisdiction_origin','?')}
  Destination: {tx.get('jurisdiction_destination','?')}
  Timestamp:   {tx.get('timestamp','?')}

ML RISK ASSESSMENT:
  Score:       {scorer_output.risk_score:.4f}
  Class:       {scorer_output.risk_class.value}
  Model:       {scorer_output.model_version} ({scorer_output.scorer_mode})

TOP MODEL SIGNALS:
{feature_lines}

LLM EXPLAINER ASSESSMENT:
  Explanation: {explainer_output.explanation_text if explainer_output else 'Running concurrently — see SHAP signals above'}
  Reason codes: {', '.join(explainer_output.reason_codes) if explainer_output else 'N/A'}

CUSTOMER PROFILE:
  Risk rating:    {cust.get('risk_rating', '?')} / 5
  Prior alerts:   {cust.get('prior_alert_count', 0)}
  Account age:    {cust.get('account_age_days', '?')} days

SANCTIONS CHECK:
  {sanctions_line}

SIMILAR HISTORICAL CASES:
{case_lines}

Provide your advisory recommendation as JSON now."""
