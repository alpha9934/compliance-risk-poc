from fastapi import APIRouter

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/export")
async def export_audit(limit: int = 100, offset: int = 0):
    """GET /audit/export — export audit events for review."""
    return {"events": [], "limit": limit, "offset": offset}  # TODO: query Neon
