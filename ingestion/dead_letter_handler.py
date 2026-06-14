import os
import time
from upstash_redis import Redis

redis = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"), 
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
)

DLQ_STREAM = "compliance-transaction-dlq"

def handle_dead_letter(message_id: str, raw_payload: str, reason: str):
    """
    Routes unparseable/invalid data away from production streams for debugging.
    """
    print(f"⚠️ Routing message {message_id} to DLQ. Reason: {reason}")
    
    dlq_entry = {
        "original_id": message_id,
        "failed_at": str(time.time()),
        "reason": reason,
        "payload": raw_payload
    }
    
    # Push to DLQ stream for audit visibility
    try:
        redis.xadd(DLQ_STREAM, "*", dlq_entry)
    except Exception as e:
        print(f"Critical: Failed to log to DLQ stream. {e}")