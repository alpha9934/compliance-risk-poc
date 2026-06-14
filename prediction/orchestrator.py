from __future__ import annotations
"""
orchestrator.py

Central coordinator of the prediction engine.
Runs the full pipeline for a single transaction:

  Step 1 — Feature engineering          (always, sync)
  Step 2 — ML scoring                   (always, sync, ~2ms)
  Step 3 — MCP tool calls               (HIGH/MEDIUM only)
  Step 4 — LLM Explainer + Judge        (HIGH/MEDIUM only, concurrent)
  Step 5 — Build UnifiedRiskOutput
  Step 6 — Create workbench case        (HIGH/MEDIUM only)

All MCP calls use the local tool implementations directly (POC).
In production they go through the FastMCP server over HTTP.
"""
import asyncio
import uuid
import logging
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Optional

from features.models import FeatureVector
from features.feature_pipeline import build_feature_vector
from features.redis_client import FeatureStoreClient
from ingestion.schemas.transaction_event import TransactionEvent, RiskClass
from prediction.ml_scorer.xgboost_scorer import score_transaction, ScorerOutput
from prediction.llm_explainer.explainer import generate_explanation, LLMExplainerOutput
from prediction.llm_judge.judge import run_judge, LLMJudgeOutput
from audit.audit_logger import AuditLogger

# All 6 MCP tools — called directly in POC
from compliance_mcp.tools.transaction_tool      import get_transaction_context, ingest_transaction
from compliance_mcp.tools.customer_tool         import get_customer_risk
from compliance_mcp.tools.policy_tool           import retrieve_policy
from compliance_mcp.tools.sanctions_tool        import check_sanctions
from compliance_mcp.tools.case_similarity_tool  import get_similar_cases
from compliance_mcp.tools.audit_tool            import log_audit_event

logger = logging.getLogger(__name__)
audit  = AuditLogger()


# ── Output model ──────────────────────────────────────────────────────────

class UnifiedRiskOutput(BaseModel):
    transaction_id:  str
    risk_score:      float
    risk_class:      RiskClass
    ml_output:       ScorerOutput
    llm_explanation: Optional[LLMExplainerOutput] = None
    llm_judge:       Optional[LLMJudgeOutput]      = None
    case_created:    bool = False
    case_id:         Optional[str] = None
    session_id:      str
    produced_at:     datetime


# ── LLM async wrapper ─────────────────────────────────────────────────────

async def _run_llm_components(
    scorer_output:       ScorerOutput,
    policy_passages:     list[dict],
    transaction_context: dict,
    customer_context:    dict,
    sanctions_result:    dict,
    similar_cases_list:  list[dict],
) -> tuple[LLMExplainerOutput, LLMJudgeOutput]:
    """
    Runs Explainer and Judge concurrently with asyncio.gather().
    Wall time = max(explainer, judge) — not their sum.
    """
    expl_task = asyncio.to_thread(
        generate_explanation,
        scorer_output,
        policy_passages,
        transaction_context,
    )
    judge_task = asyncio.to_thread(
        run_judge,
        scorer_output,
        None,                    # explainer runs in parallel — no output yet
        customer_context,
        sanctions_result,
        similar_cases_list,
        transaction_context,
    )
    return await asyncio.gather(expl_task, judge_task)


# ── Main orchestrator ─────────────────────────────────────────────────────

async def process_transaction(
    event:      TransactionEvent,
    store:      FeatureStoreClient,
    session_id: Optional[str] = None,
) -> UnifiedRiskOutput:
    """
    Runs the complete prediction pipeline for one transaction.

    Args:
        event:      Validated TransactionEvent
        store:      FeatureStoreClient (Upstash Redis or in-memory)
        session_id: Optional UUID — generated if not provided

    Returns:
        UnifiedRiskOutput — complete, audit-logged pipeline result
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Register transaction in MCP store so transaction_tool can look it up
    ingest_transaction(event.model_dump(mode="json", exclude={"schema_version"}))

    audit.log("PIPELINE_START", {
        "transaction_id": event.transaction_id,
        "session_id":     session_id,
        "amount":         event.amount,
        "channel":        event.channel,
    }, session_id=session_id)

    # ── Step 1: Feature engineering ───────────────────────────────────────
    fv = build_feature_vector(event, store)

    audit.log("FEATURES_COMPUTED", {
        "transaction_id":       event.transaction_id,
        "feature_version":      fv.feature_version,
        "amount_zscore":        fv.amount_zscore,
        "destination_risk":     fv.destination_risk_score,
        "is_new_beneficiary":   fv.is_new_beneficiary,
        "is_fatf_greylist":     fv.is_fatf_greylist,
        "is_fatf_blacklist":    fv.is_fatf_blacklist,
    }, session_id=session_id)

    # ── Step 2: ML scoring ────────────────────────────────────────────────
    scored = score_transaction(
        fv,
        product_type=event.product_type,
        jurisdiction_destination=event.jurisdiction_destination,
    )

    audit.log("ML_SCORE_PRODUCED", {
        "transaction_id": event.transaction_id,
        "risk_score":     scored.risk_score,
        "risk_class":     scored.risk_class.value,
        "model_version":  scored.model_version,
        "scorer_mode":    scored.scorer_mode,
        "latency_ms":     scored.latency_ms,
        "top_feature":    scored.shap_explanation.top_features[0].feature
                          if scored.shap_explanation.top_features else "unknown",
    }, session_id=session_id)

    llm_explanation: Optional[LLMExplainerOutput] = None
    llm_judge:       Optional[LLMJudgeOutput]      = None

    # ── Steps 3+4: MCP context + LLM (HIGH and MEDIUM only) ──────────────
    if scored.risk_class in (RiskClass.HIGH, RiskClass.MEDIUM):

        # MCP Tool 1: transaction context
        tx_context = get_transaction_context(
            transaction_id=event.transaction_id,
            caller_role="SYSTEM_INTEGRATION",
            session_id=session_id,
        )

        # MCP Tool 2: customer risk profile
        cust_context = get_customer_risk(
            customer_id=event.customer_id,
            caller_role="SYSTEM_INTEGRATION",
            session_id=session_id,
        )

        # MCP Tool 3: policy passages
        try:
            policy_passages = retrieve_policy(
                policy_type="AML",
                jurisdiction=event.jurisdiction_destination,
                product=event.product_type,
                caller_role="SYSTEM_INTEGRATION",
                session_id=session_id,
            )
        except Exception as e:
            logger.warning("Policy retrieval failed: %s", e)
            policy_passages = []

        # MCP Tool 4: sanctions check
        try:
            sanctions = check_sanctions(
                entity_id=event.destination_account,
                jurisdiction=event.jurisdiction_destination,
                caller_role="SYSTEM_INTEGRATION",
                session_id=session_id,
            )
        except Exception as e:
            logger.warning("Sanctions check failed: %s", e)
            sanctions = {"match_found": False, "confidence": 0.0}

        # MCP Tool 5: similar historical cases
        sim_cases = get_similar_cases(
            jurisdiction_destination=event.jurisdiction_destination,
            channel=event.channel,
            risk_class=scored.risk_class.value,
            top_k=3,
            caller_role="SYSTEM_INTEGRATION",
            session_id=session_id,
        )

        audit.log("MCP_CONTEXT_FETCHED", {
            "transaction_id":    event.transaction_id,
            "policy_count":      len(policy_passages),
            "sanctions_matched": sanctions.get("match_found", False),
            "similar_cases":     len(sim_cases),
        }, session_id=session_id)

        # Step 4: LLM Explainer + Judge concurrently
        llm_explanation, llm_judge = await _run_llm_components(
            scored, policy_passages, tx_context,
            cust_context, sanctions, sim_cases,
        )

        audit.log("LLM_OUTPUTS_PRODUCED", {
            "transaction_id":     event.transaction_id,
            "explainer_model":    llm_explanation.model_id,
            "explainer_fallback": llm_explanation.fallback_used,
            "judge_model":        llm_judge.model_id,
            "judge_fallback":     llm_judge.fallback_used,
            "recommendation":     llm_judge.recommendation,
            "confidence":         llm_judge.confidence,
            "explainer_hash":     llm_explanation.prompt_hash[:12],
            "judge_hash":         llm_judge.prompt_hash[:12],
        }, session_id=session_id)

    # ── Step 5: Build output ──────────────────────────────────────────────
    case_id      = None
    case_created = False

    if scored.risk_class in (RiskClass.HIGH, RiskClass.MEDIUM):
        case_id      = f"CASE-{uuid.uuid4().hex[:8].upper()}"
        case_created = True

        # MCP Tool 6: log case creation to audit trail
        log_audit_event(
            event_type="CASE_CREATED",
            payload={
                "case_id":        case_id,
                "transaction_id": event.transaction_id,
                "risk_class":     scored.risk_class.value,
                "risk_score":     scored.risk_score,
                "recommendation": llm_judge.recommendation if llm_judge else "N/A",
            },
            caller_role="SYSTEM_INTEGRATION",
            session_id=session_id,
        )

    output = UnifiedRiskOutput(
        transaction_id  = event.transaction_id,
        risk_score      = scored.risk_score,
        risk_class      = scored.risk_class,
        ml_output       = scored,
        llm_explanation = llm_explanation,
        llm_judge       = llm_judge,
        case_created    = case_created,
        case_id         = case_id,
        session_id      = session_id,
        produced_at     = datetime.now(timezone.utc),
    )

    audit.log("PIPELINE_COMPLETE", {
        "transaction_id": event.transaction_id,
        "risk_class":     output.risk_class.value,
        "case_id":        output.case_id,
        "case_created":   output.case_created,
    }, session_id=session_id)

    return output
