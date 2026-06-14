from pydantic import BaseModel
from datetime import datetime
from typing import Any, Optional


class AuditEvent(BaseModel):
    event_type: str
    payload: dict[str, Any]
    timestamp: datetime
    prev_hash: str
    event_hash: Optional[str] = None
    session_id: Optional[str] = None
    caller_id: Optional[str] = None
