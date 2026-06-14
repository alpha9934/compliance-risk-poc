import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from audit.models import AuditEvent


class AuditLogger:
    """
    Immutable hash-chain audit logger.
    Each event carries SHA-256 of the previous event — tampering breaks the chain.
    Writes to Neon Postgres (production) or prints to stdout (dev).
    """

    def __init__(self):
        self._last_hash = "GENESIS"

    def log(self, event_type: str, payload: dict[str, Any],
            session_id: str = None, caller_id: str = None) -> str:
        event = AuditEvent(
            event_type=event_type,
            payload=payload,
            timestamp=datetime.now(timezone.utc),
            prev_hash=self._last_hash,
            session_id=session_id,
            caller_id=caller_id,
        )
        serialized = json.dumps(
            event.model_dump(mode="json", exclude={"event_hash"}),
            sort_keys=True, default=str
        )
        event_hash = hashlib.sha256(serialized.encode()).hexdigest()
        event.event_hash = event_hash
        self._last_hash = event_hash
        self._persist(event)
        return event_hash

    def _persist(self, event: AuditEvent):
        # TODO: replace with async Neon Postgres insert in production
        # asyncpg: INSERT INTO audit_events (...) VALUES (...)
        print(f"[AUDIT] {event.event_type} | {event.event_hash[:12]}...")
