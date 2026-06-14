from __future__ import annotations
"""
KYC features — customer-level risk signals from the KYC / onboarding system.

In production these come from the MCP `get_customer_risk` tool.
In the POC they come from the seeded customer profiles in the feature store.
"""
from features.models import KYCFeatures
from features.redis_client import FeatureStoreClient
from ingestion.schemas.transaction_event import TransactionEvent

HIGH_RISK_RATING_THRESHOLD = 4


def compute_kyc_features(
    event: TransactionEvent,
    store: FeatureStoreClient,
) -> KYCFeatures:
    """
    Pulls the customer risk profile from the feature store.
    Falls back to safe defaults if the customer has no profile
    (e.g. new customer not yet seeded).
    """
    profile = store.get_customer_profile(event.customer_id)

    risk_rating    = int(profile.get("risk_rating", 2))
    prior_alerts   = int(profile.get("prior_alert_count", 0))
    account_age    = int(profile.get("account_age_days", 365))

    return KYCFeatures(
        customer_risk_rating=risk_rating,
        prior_alert_count=prior_alerts,
        account_age_days=account_age,
        is_high_risk_customer=risk_rating >= HIGH_RISK_RATING_THRESHOLD,
        has_prior_alerts=prior_alerts > 0,
    )
