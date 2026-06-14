"""
Unit tests for the feature engineering layer.
Runs entirely in-memory — no Redis, no database, no API keys needed.
"""
import pytest
from datetime import datetime, timezone, timedelta
from ingestion.schemas.transaction_event import TransactionEvent
from features.redis_client import FeatureStoreClient
from features.feature_pipeline import build_feature_vector
from features.velocity_features import compute_velocity_features
from features.deviation_features import compute_deviation_features
from features.geo_features import compute_geo_features
from features.behavioral_features import compute_behavioral_features, _is_round_amount, _is_structuring
from features.kyc_features import compute_kyc_features
from features.counterparty_features import compute_counterparty_features


# ── Fixtures ───────────────────────────────────────────────────────────────

def make_event(**overrides) -> TransactionEvent:
    defaults = dict(
        transaction_id="TXN-TEST-001",
        customer_id="CUST-TEST-42",
        amount=5000.00,
        currency="USD",
        channel="WIRE",
        origin_account="ACC-US-001",
        destination_account="ACC-AE-999",
        beneficiary_name="Test Corp",
        jurisdiction_origin="US",
        jurisdiction_destination="AE",
        product_type="WIRE_TRANSFER",
        timestamp=datetime(2026, 6, 13, 14, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return TransactionEvent(**defaults)


def make_store(
    avg_amount: float = 5000.0,
    risk_rating: int = 2,
    prior_alerts: int = 0,
    account_age: int = 365,
) -> FeatureStoreClient:
    store = FeatureStoreClient()   # in-memory mode (no env vars)
    store.seed_customer(
        "CUST-TEST-42",
        avg_amount=avg_amount,
        risk_rating=risk_rating,
        prior_alerts=prior_alerts,
        account_age_days=account_age,
    )
    return store


# ── Velocity tests ─────────────────────────────────────────────────────────

class TestVelocityFeatures:

    def test_empty_history_returns_zeros(self):
        event = make_event()
        store = make_store()
        v = compute_velocity_features(event, store)
        assert v.count_24h == 0
        assert v.sum_24h == 0.0
        assert v.count_7d == 0

    def test_counts_recent_transactions(self):
        event = make_event(timestamp=datetime(2026, 6, 13, 14, 0, 0, tzinfo=timezone.utc))
        store = make_store()

        # Add 3 transactions within 24h window
        for i in range(3):
            store.append_customer_txn("CUST-TEST-42", {
                "transaction_id": f"TXN-HIST-{i}",
                "amount": 1000.0,
                "currency": "USD",
                "channel": "WIRE",
                "destination_account": "ACC-US-001",
                "jurisdiction_destination": "US",
                "timestamp": (datetime(2026, 6, 13, 10, 0, 0, tzinfo=timezone.utc)).isoformat(),
            })

        v = compute_velocity_features(event, store)
        assert v.count_24h == 3
        assert v.sum_24h == 3000.0

    def test_excludes_old_transactions(self):
        event = make_event(timestamp=datetime(2026, 6, 13, 14, 0, 0, tzinfo=timezone.utc))
        store = make_store()

        # Old transaction — 8 days ago, outside 7d window
        store.append_customer_txn("CUST-TEST-42", {
            "transaction_id": "TXN-OLD",
            "amount": 99999.0,
            "currency": "USD",
            "channel": "WIRE",
            "destination_account": "ACC-OLD",
            "jurisdiction_destination": "US",
            "timestamp": datetime(2026, 6, 5, 14, 0, 0, tzinfo=timezone.utc).isoformat(),
        })
        v = compute_velocity_features(event, store)
        assert v.count_7d == 0
        assert v.sum_7d == 0.0

    def test_unique_destinations_counted(self):
        event = make_event(timestamp=datetime(2026, 6, 13, 14, 0, 0, tzinfo=timezone.utc))
        store = make_store()
        base_ts = datetime(2026, 6, 12, 10, 0, 0, tzinfo=timezone.utc)

        for i in range(4):
            store.append_customer_txn("CUST-TEST-42", {
                "amount": 500.0, "currency": "USD", "channel": "WIRE",
                "destination_account": f"ACC-DEST-{i}",
                "jurisdiction_destination": "US",
                "timestamp": base_ts.isoformat(),
            })
        v = compute_velocity_features(event, store)
        assert v.unique_dest_7d == 4


# ── Deviation tests ────────────────────────────────────────────────────────

class TestDeviationFeatures:

    def test_average_transaction_has_low_zscore(self):
        event = make_event(amount=5000.0)
        store = make_store(avg_amount=5000.0)
        d = compute_deviation_features(event, store)
        assert abs(d.amount_zscore) < 0.1
        assert not d.is_amount_outlier

    def test_high_amount_has_high_zscore(self):
        event = make_event(amount=50000.0)
        store = make_store(avg_amount=5000.0)
        d = compute_deviation_features(event, store)
        assert d.amount_zscore > 2.5
        assert d.is_amount_outlier

    def test_ratio_computed_correctly(self):
        event = make_event(amount=10000.0)
        store = make_store(avg_amount=5000.0)
        d = compute_deviation_features(event, store)
        assert abs(d.amount_vs_avg_ratio - 2.0) < 0.01

    def test_zero_std_doesnt_crash(self):
        event = make_event(amount=1000.0)
        store = make_store(avg_amount=1000.0)
        # Force std=0 by manipulating profile
        store.set("customer:profile:CUST-TEST-42", {
            "avg_amount": 1000.0, "std_amount": 0.0,
            "risk_rating": 2, "prior_alert_count": 0, "account_age_days": 365
        })
        d = compute_deviation_features(event, store)
        assert isinstance(d.amount_zscore, float)


# ── Geo tests ──────────────────────────────────────────────────────────────

class TestGeoFeatures:

    def test_us_to_us_low_risk(self):
        event = make_event(jurisdiction_origin="US", jurisdiction_destination="US")
        g = compute_geo_features(event)
        assert g.destination_risk_score == 1
        assert not g.is_high_risk_dest
        assert not g.is_cross_border

    def test_us_to_ru_high_risk(self):
        event = make_event(jurisdiction_origin="US", jurisdiction_destination="RU")
        g = compute_geo_features(event)
        assert g.destination_risk_score == 5
        assert g.is_high_risk_dest
        assert g.is_cross_border
        assert g.risk_score_delta == 4

    def test_blacklisted_jurisdiction(self):
        event = make_event(jurisdiction_destination="IR")
        g = compute_geo_features(event)
        assert g.is_fatf_blacklist

    def test_greylisted_jurisdiction(self):
        event = make_event(jurisdiction_destination="AE")
        g = compute_geo_features(event)
        assert g.is_fatf_greylist
        assert not g.is_fatf_blacklist

    def test_unknown_jurisdiction_defaults_to_moderate(self):
        event = make_event(jurisdiction_destination="ZZ")  # not in our list
        g = compute_geo_features(event)
        assert g.destination_risk_score == 3


# ── Behavioral tests ───────────────────────────────────────────────────────

class TestBehavioralFeatures:

    def test_round_amount_detection(self):
        assert _is_round_amount(100000.0) is True
        assert _is_round_amount(50000.0)  is True
        assert _is_round_amount(99999.0)  is False
        assert _is_round_amount(1234.56)  is False

    def test_structuring_detection(self):
        history = [{"amount": 9500.0}, {"amount": 9800.0}]
        assert _is_structuring(9700.0, history) is True

    def test_no_structuring_below_count(self):
        history = [{"amount": 9500.0}]    # only 1 past, need 2+
        assert _is_structuring(9700.0, history) is False

    def test_night_transaction_high_risk(self):
        event = make_event(
            timestamp=datetime(2026, 6, 13, 2, 30, 0, tzinfo=timezone.utc)
        )
        store = make_store()
        b = compute_behavioral_features(event, store)
        assert b.time_of_day_risk >= 0.7

    def test_business_hours_low_risk(self):
        event = make_event(
            timestamp=datetime(2026, 6, 13, 10, 0, 0, tzinfo=timezone.utc)
        )
        store = make_store()
        b = compute_behavioral_features(event, store)
        assert b.time_of_day_risk < 0.5

    def test_channel_switch_detected(self):
        event = make_event(channel="WIRE")
        store = make_store()
        # Seed history with ACH as majority channel
        for _ in range(4):
            store.append_channel("CUST-TEST-42", "ACH")
        b = compute_behavioral_features(event, store)
        assert b.channel_switch_flag is True

    def test_no_channel_switch_same_channel(self):
        event = make_event(channel="WIRE")
        store = make_store()
        for _ in range(4):
            store.append_channel("CUST-TEST-42", "WIRE")
        b = compute_behavioral_features(event, store)
        assert b.channel_switch_flag is False


# ── KYC tests ─────────────────────────────────────────────────────────────

class TestKYCFeatures:

    def test_high_risk_customer_flagged(self):
        event = make_event()
        store = make_store(risk_rating=4)
        k = compute_kyc_features(event, store)
        assert k.is_high_risk_customer is True
        assert k.customer_risk_rating == 4

    def test_low_risk_customer_not_flagged(self):
        event = make_event()
        store = make_store(risk_rating=1)
        k = compute_kyc_features(event, store)
        assert k.is_high_risk_customer is False

    def test_prior_alerts_flag(self):
        event = make_event()
        store = make_store(prior_alerts=3)
        k = compute_kyc_features(event, store)
        assert k.has_prior_alerts is True
        assert k.prior_alert_count == 3


# ── Full pipeline integration test ─────────────────────────────────────────

class TestFeaturePipeline:

    def test_full_pipeline_returns_feature_vector(self):
        event = make_event(
            amount=95000.0,
            jurisdiction_destination="AE",
            channel="WIRE",
        )
        store = make_store(avg_amount=5000.0, risk_rating=3)
        fv = build_feature_vector(event, store)

        assert fv.transaction_id == "TXN-TEST-001"
        assert fv.is_cross_border is True
        assert fv.destination_risk_score == 3
        assert fv.is_fatf_greylist is True
        assert fv.amount_zscore > 0       # 95K vs avg 5K
        assert fv.is_amount_outlier is True
        assert fv.is_new_beneficiary is True  # first time seeing dest account

    def test_pipeline_updates_store(self):
        event = make_event()
        store = make_store()
        assert store.get_customer_txn_history("CUST-TEST-42") == []

        build_feature_vector(event, store)

        history = store.get_customer_txn_history("CUST-TEST-42")
        assert len(history) == 1
        assert history[0]["transaction_id"] == "TXN-TEST-001"

    def test_feature_vector_to_model_input(self):
        event = make_event()
        store = make_store()
        fv = build_feature_vector(event, store)
        model_input = fv.to_model_input()

        assert isinstance(model_input, list)
        assert len(model_input) == 24        # must match FEATURE_ORDER length
        assert all(isinstance(v, float) for v in model_input)

    def test_second_transaction_uses_first_as_history(self):
        store = make_store(avg_amount=5000.0)

        # First transaction
        event1 = make_event(
            transaction_id="TXN-FIRST",
            amount=5000.0,
            timestamp=datetime(2026, 6, 13, 10, 0, 0, tzinfo=timezone.utc),
        )
        build_feature_vector(event1, store)

        # Second transaction — 2h later, same customer
        event2 = make_event(
            transaction_id="TXN-SECOND",
            amount=6000.0,
            timestamp=datetime(2026, 6, 13, 12, 0, 0, tzinfo=timezone.utc),
        )
        fv2 = build_feature_vector(event2, store)

        # The second transaction should see the first in its velocity window
        assert fv2.count_24h >= 1
        assert fv2.sum_24h >= 5000.0
