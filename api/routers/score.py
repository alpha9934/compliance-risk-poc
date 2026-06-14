from __future__ import annotations
"""
score.py  —  POST /score

Accepts a TransactionEvent, runs the full prediction pipeline
via the orchestrator, and returns UnifiedRiskOutput.

The feature store is kept as a module-level singleton so customer
history accumulates across requests within a process lifetime.
In production this is backed by Upstash Redis (persistent).
"""
import uuid
import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ingestion.schemas.transaction_event import TransactionEvent
from features.redis_client import FeatureStoreClient
from prediction.orchestrator import process_transaction, UnifiedRiskOutput
from audit.audit_logger import AuditLogger

router = APIRouter(prefix="/score", tags=["scoring"])
logger = logging.getLogger(__name__)
audit  = AuditLogger()

# Module-level feature store — reused across requests
_store = FeatureStoreClient()


@router.post("/", response_model=UnifiedRiskOutput)
async def score_transaction_endpoint(event: TransactionEvent):
    """
    POST /score

    Runs the complete compliance risk prediction pipeline:
      1. Feature engineering
      2. ML scoring (XGBoost + SHAP)
      3. MCP tool calls (policy, sanctions, similar cases)
      4. LLM Explainer + Judge (HIGH/MEDIUM only, concurrent)
      5. Audit logging

    Returns UnifiedRiskOutput with risk_score, risk_class,
    SHAP explanation, LLM explanation, judge recommendation,
    and case_id if a case was created.
    """
    session_id = str(uuid.uuid4())

    audit.log("API_SCORE_REQUEST", {
        "transaction_id": event.transaction_id,
        "amount":         event.amount,
        "channel":        event.channel,
        "dest":           event.jurisdiction_destination,
        "session_id":     session_id,
    })

    try:
        result = await process_transaction(
            event=event,
            store=_store,
            session_id=session_id,
        )
        return result

    except Exception as e:
        logger.exception("Pipeline failed for txn %s", event.transaction_id)
        audit.log("API_SCORE_ERROR", {
            "transaction_id": event.transaction_id,
            "error":          str(e),
            "session_id":     session_id,
        })
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@router.post("/batch")
async def score_batch(events: list[TransactionEvent]):
    """
    POST /score/batch

    Scores a list of transactions sequentially.
    Returns list of UnifiedRiskOutput.

    For large batches use the Anthropic Batch API pattern instead.
    """
    if len(events) > 50:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 50. For larger batches, use async processing."
        )

    results = []
    for event in events:
        session_id = str(uuid.uuid4())
        try:
            result = await process_transaction(
                event=event,
                store=_store,
                session_id=session_id,
            )
            results.append(result.model_dump(mode="json"))
        except Exception as e:
            logger.error("Batch item failed: %s — %s", event.transaction_id, e)
            results.append({
                "transaction_id": event.transaction_id,
                "error": str(e),
                "status": "failed",
            })

    return {"results": results, "total": len(results)}
