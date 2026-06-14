from __future__ import annotations
"""
Redis client for the feature store.
POC:        Upstash Redis (free tier, HTTP-based, no persistent connection needed)
Production: AWS ElastiCache with persistent connection pool

The client stores:
  - customer transaction history (list of recent txn amounts + timestamps)
  - seen beneficiary accounts (set per customer)
  - customer channel history (list of last N channels used)

Keys:
  customer:txns:{customer_id}        → JSON list of {amount, timestamp, channel, dest}
  customer:beneficiaries:{customer_id} → JSON set of seen destination accounts
"""
import json
import os
from datetime import datetime, timezone
from typing import Any

# Try Upstash Redis first; fall back to in-memory dict for local dev without keys
try:
    from upstash_redis import Redis as UpstashRedis
    _upstash_available = True
except ImportError:
    _upstash_available = False


class FeatureStoreClient:
    """
    Thin wrapper around Upstash Redis for feature storage.
    Falls back to an in-memory dict when UPSTASH_REDIS_URL is not set —
    useful for local dev and unit tests.
    """

    def __init__(self):
        # Upstash's dashboard exports UPSTASH_REDIS_REST_URL / _REST_TOKEN.
        # Some setups use the shorter UPSTASH_REDIS_URL / _TOKEN instead.
        # Accept either naming convention — REST variant takes precedence
        # since that's what users copy directly from the Upstash console.
        redis_url = (
            os.getenv("UPSTASH_REDIS_REST_URL")
            or os.getenv("UPSTASH_REDIS_URL")
            or ""
        )
        redis_token = (
            os.getenv("UPSTASH_REDIS_REST_TOKEN")
            or os.getenv("UPSTASH_REDIS_TOKEN")
            or ""
        )

        # Reject placeholder values from .env.example — these are not real
        # credentials and would cause a DNS resolution error on first use
        # (e.g. "https://your-redis.upstash.io" has no real DNS record).
        is_placeholder = (
            "your-redis" in redis_url
            or "your-username" in redis_token
            or "your-redis-token" in redis_token
            or redis_url in ("", "https://")
        )

        if _upstash_available and redis_url and redis_token and not is_placeholder:
            try:
                client = UpstashRedis(url=redis_url, token=redis_token)
                # Verify the connection actually works before committing to it —
                # a bad hostname only fails on first real network call otherwise.
                client.ping()
                self._client = client
                self._mode = "upstash"
            except Exception as e:
                print(f"[FeatureStore] Upstash connection failed ({e}) — "
                      f"falling back to in-memory store")
                self._client = {}
                self._mode = "memory"
        else:
            self._client = {}      # in-memory fallback
            self._mode = "memory"
            if is_placeholder and redis_url:
                print("[FeatureStore] UPSTASH_REDIS_URL looks like a placeholder "
                      "value — using in-memory store. Set real credentials in .env "
                      "to enable persistence.")
            else:
                print("[FeatureStore] No Upstash credentials — using in-memory store")

    # ── Low-level get / set ───────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        if self._mode == "upstash":
            val = self._client.get(key)
            return json.loads(val) if val else None
        return self._client.get(key)

    def set(self, key: str, value: Any, ex: int = 86400 * 7) -> None:
        """Store value with TTL (default 7 days)."""
        if self._mode == "upstash":
            self._client.set(key, json.dumps(value, default=str), ex=ex)
        else:
            self._client[key] = value

    # ── Customer transaction history ──────────────────────────────────────

    def get_customer_txn_history(self, customer_id: str) -> list[dict]:
        """Returns list of past transactions for this customer (max 500)."""
        return self.get(f"customer:txns:{customer_id}") or []

    def append_customer_txn(self, customer_id: str, txn: dict) -> None:
        """Appends a transaction to customer history, capped at 500 entries."""
        history = self.get_customer_txn_history(customer_id)
        history.append(txn)
        history = history[-500:]        # keep most recent 500
        self.set(f"customer:txns:{customer_id}", history)

    # ── Seen beneficiaries ────────────────────────────────────────────────

    def get_seen_beneficiaries(self, customer_id: str) -> set[str]:
        data = self.get(f"customer:beneficiaries:{customer_id}") or []
        return set(data)

    def add_beneficiary(self, customer_id: str, dest_account: str) -> None:
        bens = self.get_seen_beneficiaries(customer_id)
        bens.add(dest_account)
        self.set(f"customer:beneficiaries:{customer_id}", list(bens))

    # ── Channel history ───────────────────────────────────────────────────

    def get_channel_history(self, customer_id: str, n: int = 5) -> list[str]:
        """Returns last N channels used by this customer."""
        data = self.get(f"customer:channels:{customer_id}") or []
        return data[-n:]

    def append_channel(self, customer_id: str, channel: str) -> None:
        history = self.get(f"customer:channels:{customer_id}") or []
        history.append(channel)
        history = history[-50:]
        self.set(f"customer:channels:{customer_id}", history)

    # ── Utility ───────────────────────────────────────────────────────────

    def seed_customer(
        self,
        customer_id: str,
        avg_amount: float,
        risk_rating: int,
        prior_alerts: int,
        account_age_days: int,
    ) -> None:
        """
        Seeds a synthetic customer profile used by deviation and KYC features.
        In production this data comes from the KYC system via MCP tool.
        """
        self.set(f"customer:profile:{customer_id}", {
            "avg_amount": avg_amount,
            "std_amount": avg_amount * 0.4,
            "risk_rating": risk_rating,
            "prior_alert_count": prior_alerts,
            "account_age_days": account_age_days,
        })

    def get_customer_profile(self, customer_id: str) -> dict:
        return self.get(f"customer:profile:{customer_id}") or {
            "avg_amount": 5000.0,
            "std_amount": 2000.0,
            "risk_rating": 2,
            "prior_alert_count": 0,
            "account_age_days": 365,
        }

    @property
    def mode(self) -> str:
        return self._mode
