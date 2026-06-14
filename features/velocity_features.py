from __future__ import annotations
"""
Velocity features — how many transactions and how much volume
has this customer generated over rolling time windows.

High velocity in a short window is a core AML signal (structuring, layering).
"""
from datetime import datetime, timezone, timedelta
from features.models import VelocityFeatures
from features.redis_client import FeatureStoreClient
from ingestion.schemas.transaction_event import TransactionEvent

# Approximate USD conversion rates for normalising amounts across currencies
# Production: use a live FX rate API. POC: static approximations.
FX_TO_USD = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0067,
    "AED": 0.27,
    "CHF": 1.12,
    "CAD": 0.74,
    "AUD": 0.65,
    "SGD": 0.74,
    "INR": 0.012,
}


def _to_usd(amount: float, currency: str) -> float:
    return amount * FX_TO_USD.get(currency, 1.0)


def compute_velocity_features(
    event: TransactionEvent,
    store: FeatureStoreClient,
) -> VelocityFeatures:
    """
    Counts and sums transactions in 1h, 24h, 7d windows for this customer.
    Uses transaction history stored in Redis/in-memory feature store.
    """
    now = event.timestamp.replace(tzinfo=timezone.utc) if event.timestamp.tzinfo is None \
        else event.timestamp

    history = store.get_customer_txn_history(event.customer_id)

    count_1h = count_24h = count_7d = 0
    sum_1h = sum_24h = sum_7d = 0.0
    dest_accounts_7d: set[str] = set()

    cutoff_1h  = now - timedelta(hours=1)
    cutoff_24h = now - timedelta(hours=24)
    cutoff_7d  = now - timedelta(days=7)

    for txn in history:
        try:
            ts = datetime.fromisoformat(str(txn["timestamp"]))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue

        usd_amount = _to_usd(txn.get("amount", 0), txn.get("currency", "USD"))

        if ts >= cutoff_7d:
            count_7d += 1
            sum_7d += usd_amount
            dest_accounts_7d.add(txn.get("destination_account", ""))

        if ts >= cutoff_24h:
            count_24h += 1
            sum_24h += usd_amount

        if ts >= cutoff_1h:
            count_1h += 1
            sum_1h += usd_amount

    return VelocityFeatures(
        count_1h=count_1h,
        count_24h=count_24h,
        count_7d=count_7d,
        sum_1h=round(sum_1h, 2),
        sum_24h=round(sum_24h, 2),
        sum_7d=round(sum_7d, 2),
        unique_dest_7d=len(dest_accounts_7d),
    )
