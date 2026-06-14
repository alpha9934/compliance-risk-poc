from __future__ import annotations
"""
Counterparty features — signals about the destination account and beneficiary.

A beneficiary never seen before in the customer's history is a key AML signal,
especially combined with high amounts or high-risk jurisdictions.
"""
from datetime import datetime, timezone, timedelta
from features.models import CounterpartyFeatures
from features.redis_client import FeatureStoreClient
from ingestion.schemas.transaction_event import TransactionEvent


def compute_counterparty_features(
    event: TransactionEvent,
    store: FeatureStoreClient,
) -> CounterpartyFeatures:
    """
    Checks whether the destination account has been seen before for this customer,
    and how many other senders have used this destination account.
    """
    seen_beneficiaries = store.get_seen_beneficiaries(event.customer_id)
    is_new = event.destination_account not in seen_beneficiaries

    # Approximate age of beneficiary relationship — uses transaction history
    history = store.get_customer_txn_history(event.customer_id)
    dest_txns = [
        t for t in history
        if t.get("destination_account") == event.destination_account
    ]

    if dest_txns:
        try:
            earliest = min(
                datetime.fromisoformat(str(t["timestamp"])) for t in dest_txns
            )
            if earliest.tzinfo is None:
                earliest = earliest.replace(tzinfo=timezone.utc)
            now = event.timestamp.replace(tzinfo=timezone.utc) \
                if event.timestamp.tzinfo is None else event.timestamp
            age_days = (now - earliest).days
        except (ValueError, KeyError):
            age_days = 0
    else:
        age_days = 999      # never seen — treat as brand new

    return CounterpartyFeatures(
        is_new_beneficiary=is_new,
        dest_account_age_days=age_days,
        unique_senders_to_dest=0,   # TODO: cross-customer lookup in production
    )
