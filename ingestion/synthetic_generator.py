# ingestion/synthetic_generator.py
import os
import argparse
import json
import random
import uuid
from dotenv import load_dotenv
from upstash_redis import Redis

load_dotenv()

# Initialize Upstash HTTP Client
redis = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
)

STREAM_NAME = "compliance-transactions"

def generate_mock_transaction(high_risk=False):
    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": f"usr_{random.randint(1000, 9999)}",
        "amount": round(random.uniform(5000, 50000) if high_risk else random.uniform(10, 1000), 2),
        "location": random.choice(["IN", "US", "UK", "AE"]) if not high_risk else "High-Risk-Zone"
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    print(f"Generating {args.count} synthetic transactions...")
    
    for _ in range(args.count):
        # 30% chance of generating a high-risk transaction for testing
        is_high_risk = random.random() < 0.30
        tx_data = generate_mock_transaction(high_risk=is_high_risk)
        
        if args.publish:
            # RAW COMMAND: XADD compliance-transactions * data {"transaction_id": ...}
            redis.execute([
                "XADD", STREAM_NAME, "*", 
                "data", json.dumps(tx_data)
            ])
            print(f" [x] Sent to Upstash Stream: {tx_data['transaction_id']}")
        else:
            print(f" [x] Local Mode (No Publish): {tx_data}")

if __name__ == "__main__":
    main()