from __future__ import annotations
"""
Pydantic models for feature engineering layer.
Every feature group has its own model — they are merged into FeatureVector
by the pipeline orchestrator.
"""
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Optional


class VelocityFeatures(BaseModel):
    """Transaction frequency and volume over time windows."""
    count_1h: int = 0           # number of txns in last 1 hour
    count_24h: int = 0          # number of txns in last 24 hours
    count_7d: int = 0           # number of txns in last 7 days
    sum_1h: float = 0.0         # total amount in last 1 hour (USD equiv)
    sum_24h: float = 0.0        # total amount in last 24 hours
    sum_7d: float = 0.0         # total amount in last 7 days
    unique_dest_7d: int = 0     # unique destination accounts in 7 days


class DeviationFeatures(BaseModel):
    """How much this transaction deviates from the customer's historical pattern."""
    amount_zscore: float = 0.0          # z-score vs customer 90-day avg
    amount_vs_avg_ratio: float = 1.0    # amount / customer_avg_amount
    customer_avg_amount: float = 0.0    # rolling 90-day avg for this customer
    customer_std_amount: float = 0.0    # rolling 90-day std dev
    is_amount_outlier: bool = False     # True if |z-score| > 2.5


class GeoFeatures(BaseModel):
    """Geography and jurisdiction risk signals."""
    origin_risk_score: int = 1          # 1-5 rating from jurisdiction_risk.json
    destination_risk_score: int = 1     # 1-5 rating
    risk_score_delta: int = 0           # dest_risk - origin_risk (positive = escalating)
    is_high_risk_dest: bool = False     # destination risk >= 4
    is_fatf_greylist: bool = False      # destination on FATF grey list
    is_fatf_blacklist: bool = False     # destination on FATF black list
    is_cross_border: bool = False       # origin != destination jurisdiction


class CounterpartyFeatures(BaseModel):
    """Signals about the transaction counterparty and beneficiary."""
    is_new_beneficiary: bool = False    # never seen this dest account before
    dest_account_age_days: int = 999    # days since first seen (999 = never seen)
    unique_senders_to_dest: int = 0    # how many different senders to this dest


class KYCFeatures(BaseModel):
    """Customer-level KYC and onboarding risk signals."""
    customer_risk_rating: int = 1       # 1 (low) to 5 (high) from KYC system
    prior_alert_count: int = 0          # number of prior compliance alerts
    account_age_days: int = 365         # days since account opened
    is_high_risk_customer: bool = False # risk_rating >= 4
    has_prior_alerts: bool = False      # prior_alert_count > 0


class BehavioralFeatures(BaseModel):
    """Unusual patterns in transaction behaviour."""
    channel_switch_flag: bool = False   # different channel vs last 5 txns
    time_of_day_risk: float = 0.0       # 0-1 score (night/weekend = higher)
    is_round_amount: bool = False       # amount is suspiciously round (e.g. 100000.00)
    is_structuring_pattern: bool = False # multiple txns just below reporting threshold


class FeatureVector(BaseModel):
    """
    Complete set of features for a single transaction.
    This is the input to the XGBoost ML scorer.
    All field names must match FEATURE_ORDER in xgboost_scorer.py.
    """
    transaction_id: str
    customer_id: str

    # Velocity
    count_24h: int = 0
    count_7d: int = 0
    sum_24h: float = 0.0
    sum_7d: float = 0.0
    unique_dest_7d: int = 0

    # Deviation
    amount_zscore: float = 0.0
    amount_vs_avg_ratio: float = 1.0
    is_amount_outlier: bool = False

    # Geo
    destination_risk_score: int = 1
    risk_score_delta: int = 0
    is_high_risk_dest: bool = False
    is_fatf_greylist: bool = False
    is_fatf_blacklist: bool = False
    is_cross_border: bool = False

    # Counterparty
    is_new_beneficiary: bool = False
    dest_account_age_days: int = 999

    # KYC
    customer_risk_rating: int = 1
    prior_alert_count: int = 0
    is_high_risk_customer: bool = False
    has_prior_alerts: bool = False

    # Behavioral
    channel_switch_flag: bool = False
    time_of_day_risk: float = 0.0
    is_round_amount: bool = False
    is_structuring_pattern: bool = False

    # Metadata
    feature_version: str = "1.0"
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_model_input(self) -> list[float]:
        """
        Returns features as an ordered float list for XGBoost.
        Order must match FEATURE_ORDER in prediction/ml_scorer/xgboost_scorer.py
        """
        return [
            float(self.count_24h),
            float(self.count_7d),
            float(self.sum_24h),
            float(self.sum_7d),
            float(self.unique_dest_7d),
            float(self.amount_zscore),
            float(self.amount_vs_avg_ratio),
            float(self.is_amount_outlier),
            float(self.destination_risk_score),
            float(self.risk_score_delta),
            float(self.is_high_risk_dest),
            float(self.is_fatf_greylist),
            float(self.is_fatf_blacklist),
            float(self.is_cross_border),
            float(self.is_new_beneficiary),
            float(self.dest_account_age_days),
            float(self.customer_risk_rating),
            float(self.prior_alert_count),
            float(self.is_high_risk_customer),
            float(self.has_prior_alerts),
            float(self.channel_switch_flag),
            float(self.time_of_day_risk),
            float(self.is_round_amount),
            float(self.is_structuring_pattern),
        ]
