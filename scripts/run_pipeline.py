# scripts/run_pipeline.py
import argparse
import os
from ingestion.stream_consumer import consume_events  # Shared consumer logic from our previous step
from ingestion.synthetic_generator import main as run_synthetic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="local", choices=["local", "realtime"])
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--high-risk-pct", type=int, default=30)
    args = parser.parse_args()

    if args.source == "realtime":
        print("🚀 Starting Pipeline Mode: Consuming from Upstash Realtime Stream...")
        try:
            consume_events()
        except KeyboardInterrupt:
            print("\nPipeline gracefully stopped.")
            
    else:
        print(f"🤖 Starting Pipeline Mode: Running with local synthetic data...")
        # Patching args temporarily to trigger the local generator logic directly
        import sys
        sys.argv = [sys.argv[0], "--count", str(args.count)]
        run_synthetic()

if __name__ == "__main__":
    main()