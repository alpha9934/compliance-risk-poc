from __future__ import annotations
"""
Feature pipeline orchestrator.

Calls all five feature builders in sequence and merges their outputs
into a single FeatureVector that the ML scorer consumes.

Also handles the post-feature side-effect of updating the feature store
with the current transaction (so future transactions can use this as history).
"""
from datetime import datetime, timezone
from features.models import FeatureVector
from features.redis_client import FeatureStoreClient
from features.velocity_features import compute_velocity_features
from features.deviation_features import compute_deviation_features
from features.geo_features import compute_geo_features
from features.counterparty_features import compute_counterparty_features
from features.kyc_features import compute_kyc_features
from features.behavioral_features import compute_behavioral_features
from ingestion.schemas.transaction_event import TransactionEvent


def build_feature_vector(
    event: TransactionEvent,
    store: FeatureStoreClient,
) -> FeatureVector:
    """
    Orchestrates all feature builders for a single transaction event.
    Returns a FeatureVector ready for the XGBoost ML scorer.

    Order matters for the store update at the end — we read history
    BEFORE appending so this transaction doesn't count in its own features.
    """

    # ── Build all feature groups ───────────────────────────────────────────
    velocity    = compute_velocity_features(event, store)
    deviation   = compute_deviation_features(event, store)
    geo         = compute_geo_features(event)
    counterpart = compute_counterparty_features(event, store)
    kyc         = compute_kyc_features(event, store)
    behavioral  = compute_behavioral_features(event, store)

    # ── Merge into flat FeatureVector ──────────────────────────────────────
    fv = FeatureVector(
        transaction_id=event.transaction_id,
        customer_id=event.customer_id,

        # Velocity
        count_24h=velocity.count_24h,
        count_7d=velocity.count_7d,
        sum_24h=velocity.sum_24h,
        sum_7d=velocity.sum_7d,
        unique_dest_7d=velocity.unique_dest_7d,

        # Deviation
        amount_zscore=deviation.amount_zscore,
        amount_vs_avg_ratio=deviation.amount_vs_avg_ratio,
        is_amount_outlier=deviation.is_amount_outlier,

        # Geo
        destination_risk_score=geo.destination_risk_score,
        risk_score_delta=geo.risk_score_delta,
        is_high_risk_dest=geo.is_high_risk_dest,
        is_fatf_greylist=geo.is_fatf_greylist,
        is_fatf_blacklist=geo.is_fatf_blacklist,
        is_cross_border=geo.is_cross_border,

        # Counterparty
        is_new_beneficiary=counterpart.is_new_beneficiary,
        dest_account_age_days=counterpart.dest_account_age_days,

        # KYC
        customer_risk_rating=kyc.customer_risk_rating,
        prior_alert_count=kyc.prior_alert_count,
        is_high_risk_customer=kyc.is_high_risk_customer,
        has_prior_alerts=kyc.has_prior_alerts,

        # Behavioral
        channel_switch_flag=behavioral.channel_switch_flag,
        time_of_day_risk=behavioral.time_of_day_risk,
        is_round_amount=behavioral.is_round_amount,
        is_structuring_pattern=behavioral.is_structuring_pattern,

        feature_version="1.0",
        computed_at=datetime.now(timezone.utc),
    )

    # ── Update feature store AFTER computing features ──────────────────────
    _update_store(event, store)

    return fv


def _update_store(event: TransactionEvent, store: FeatureStoreClient) -> None:
    """
    Persists the current transaction into the feature store
    so it influences future feature computations for this customer.
    """
    store.append_customer_txn(event.customer_id, {
        "transaction_id": event.transaction_id,
        "amount": event.amount,
        "currency": event.currency,
        "channel": event.channel,
        "destination_account": event.destination_account,
        "jurisdiction_destination": event.jurisdiction_destination,
        "timestamp": event.timestamp.isoformat(),
    })
    store.add_beneficiary(event.customer_id, event.destination_account)
    store.append_channel(event.customer_id, event.channel)
