from __future__ import annotations
"""
prompt_builder.py  (LLM Explainer)

Builds the prompt sent to Gemini Flash for generating plain-English
compliance explanations.

The prompt is constructed from:
  1. The versioned system prompt (prediction/prompts/explainer_v1.txt)
  2. The ML scorer output (risk score + top SHAP features)
  3. Policy passages retrieved via the MCP policy tool

Keeping prompt construction here (not in explainer.py) means you can
version-control, test, and iterate on prompts independently of API call logic.
"""
import os
from prediction.ml_scorer.xgboost_scorer import ScorerOutput
from prediction.ml_scorer.shap_explainer import SHAPExplanation

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "../prompts/explainer_v1.txt")

with open(_PROMPT_PATH) as f:
    SYSTEM_PROMPT = f.read().strip()


def build_explanation_prompt(
    scorer_output: ScorerOutput,
    policy_passages: list[dict],
    transaction_context: dict,
) -> str:
    """
    Builds the user-turn prompt for the LLM Explainer.
    The system prompt is sent separately in the API call.

    Args:
        scorer_output:       Full ScorerOutput from xgboost_scorer
        policy_passages:     List of policy dicts from MCP retrieve_policy tool
        transaction_context: Dict from MCP get_transaction_context tool

    Returns:
        User-turn prompt string
    """
    shap = scorer_output.shap_explanation

    # Format SHAP top features
    feature_lines = "\n".join(
        f"  - {f.feature}: value={f.value}, "
        f"contribution={f.contribution:+.4f} "
        f"({'raises' if f.contribution > 0 else 'lowers'} risk)"
        for f in shap.top_features
    )

    # Format policy passages (max 3 to stay within token budget)
    policy_lines = "\n".join(
        f"  [{p['policy_id']} v{p.get('version','?')}] {p['passage']}"
        for p in policy_passages[:3]
    ) or "  No matching policy passages found."

    # Format transaction context
    tx = transaction_context
    tx_lines = (
        f"  Amount:      {tx.get('amount','?')} {tx.get('currency','')}\n"
        f"  Channel:     {tx.get('channel','?')}\n"
        f"  Origin:      {tx.get('jurisdiction_origin','?')}\n"
        f"  Destination: {tx.get('jurisdiction_destination','?')}\n"
        f"  Timestamp:   {tx.get('timestamp','?')}"
    )

    return f"""TRANSACTION DETAILS:
{tx_lines}

RISK SCORE: {scorer_output.risk_score:.4f} ({scorer_output.risk_class.value} risk)
Scorer mode: {scorer_output.scorer_mode}

TOP MODEL SIGNALS (SHAP feature contributions):
{feature_lines}

APPROVED POLICY PASSAGES (cite only these):
{policy_lines}

Generate the JSON explanation now."""
