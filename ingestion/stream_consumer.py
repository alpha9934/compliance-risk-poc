import os
import json
import time
from upstash_redis import Redis
from dotenv import load_dotenv
from ingestion.schema_validator import validate_schema
from ingestion.dead_letter_handler import handle_dead_letter
load_dotenv()
# Initialize Upstash HTTP Client
redis = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"), 
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
)

STREAM_NAME = "compliance-transactions"
CONSUMER_GROUP = "compliance-risk-poc-group"
CONSUMER_NAME = "worker-1"

def init_consumer_group():
    """Creates the consumer group if it doesn't already exist."""
    try:
        # RAW COMMAND: XGROUP CREATE compliance-transactions compliance-risk-poc-group $ MKSTREAM
        redis.execute(["XGROUP", "CREATE", STREAM_NAME, CONSUMER_GROUP, "$", "MKSTREAM"])
        print(f"Consumer group '{CONSUMER_GROUP}' initialized successfully.")
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            print(f"Group init info: {e}")

def consume_events():
    init_consumer_group()
    print(f"Listening for live events on Upstash stream: {STREAM_NAME}...")

    while True:
        try:
            # RAW COMMAND: XREADGROUP GROUP group consumer COUNT 5 BLOCK 2000 STREAMS stream >
            response = redis.execute([
                "XREADGROUP", "GROUP", CONSUMER_GROUP, CONSUMER_NAME,
                "COUNT", "5", "BLOCK", "2000", "STREAMS", STREAM_NAME, ">"
            ])

            if not response:
                continue

            # Upstash REST returns nested lists for streams: [[stream_name, [[msg_id, [field, val]]]]]
            for stream_data in response:
                messages = stream_data[1]
                for message in messages:
                    msg_id = message[0]
                    fields_list = message[1] # Altered to sequential list [key, value, key, value]
                    
                    # Convert Redis list back into a readable dictionary
                    payload = dict(zip(fields_list[::2], fields_list[1::2]))
                    raw_data = payload.get("data")
                    
                    print(f"\n📥 Received event [{msg_id}] from cloud")

                    is_valid, processed_data = validate_schema(raw_data)

                    if is_valid:
                        print(f"✅ Success: Verified transaction {processed_data.get('transaction_id')}")
                        # RAW COMMAND: XACK compliance-transactions compliance-risk-poc-group msg_id
                        redis.execute(["XACK", STREAM_NAME, CONSUMER_GROUP, msg_id])
                    else:
                        handle_dead_letter(msg_id, raw_data, reason="Schema Validation Failed")
                        redis.execute(["XACK", STREAM_NAME, CONSUMER_GROUP, msg_id])

        except Exception as err:
            print(f"Error reading stream: {err}")
            time.sleep(2)

if __name__ == "__main__":
    consume_events()