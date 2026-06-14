from __future__ import annotations
"""
Deviation features — how far does this transaction deviate from
the customer's historical normal behaviour?

A wire transfer 4 standard deviations above a customer's average is a
much stronger signal than the same amount from a high-volume business client.
"""
import math
from features.models import DeviationFeatures
from features.redis_client import FeatureStoreClient
from ingestion.schemas.transaction_event import TransactionEvent

OUTLIER_ZSCORE_THRESHOLD = 2.5     # |z| > 2.5 → flagged as outlier


def compute_deviation_features(
    event: TransactionEvent,
    store: FeatureStoreClient,
) -> DeviationFeatures:
    """
    Computes z-score and ratio of current transaction vs customer's
    rolling 90-day average from the feature store profile.
    """
    profile = store.get_customer_profile(event.customer_id)
    avg    = profile.get("avg_amount", 5000.0)
    std    = profile.get("std_amount", 2000.0)

    # Avoid division by zero for customers with no transaction history
    if std < 1.0:
        std = max(avg * 0.3, 100.0)

    zscore = (event.amount - avg) / std
    ratio  = event.amount / avg if avg > 0 else 1.0

    return DeviationFeatures(
        amount_zscore=round(zscore, 4),
        amount_vs_avg_ratio=round(ratio, 4),
        customer_avg_amount=round(avg, 2),
        customer_std_amount=round(std, 2),
        is_amount_outlier=abs(zscore) > OUTLIER_ZSCORE_THRESHOLD,
    )
