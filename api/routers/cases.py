from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from audit.audit_logger import AuditLogger

router = APIRouter(prefix="/cases", tags=["cases"])
audit = AuditLogger()


class ReviewerAction(BaseModel):
    action: str           # APPROVE | REJECT | ESCALATE | HOLD | CLOSE
    reason_code: str
    notes: Optional[str] = None
    reviewer_id: str


@router.get("/")
async def list_cases(status: str = "UNDER_REVIEW"):
    """GET /cases — list open cases by status."""
    return {"cases": [], "status": status}  # TODO: query Neon


@router.patch("/{case_id}")
async def update_case(case_id: str, action: ReviewerAction):
    """PATCH /cases/{case_id} — record reviewer decision."""
    audit.log("REVIEWER_ACTION", {
        "case_id": case_id,
        "reviewer_id": action.reviewer_id,
        "action": action.action,
        "reason_code": action.reason_code,
    })
    return {"case_id": case_id, "action": action.action, "status": "recorded"}
