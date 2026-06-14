from __future__ import annotations
"""
Behavioral features — unusual patterns in how the customer is transacting.

These catch behaviours that don't show up in amounts alone:
- Suddenly switching from ACH to WIRE (channel switch)
- Transacting at 3am on a Sunday (time-of-day risk)
- Sending exactly $9,999 three times (structuring pattern)
- Round-number amounts (a known indicator of manual, laundering activity)
"""
from datetime import datetime, timezone
from features.models import BehavioralFeatures
from features.redis_client import FeatureStoreClient
from ingestion.schemas.transaction_event import TransactionEvent

# Amounts that are suspiciously round (multiples of these values)
ROUND_AMOUNT_THRESHOLDS = [1000, 5000, 10000, 25000, 50000, 100000]

# CTR filing threshold in USD — transactions just below are a structuring signal
CTR_THRESHOLD = 10_000.0
STRUCTURING_WINDOW = 0.15       # within 15% below CTR threshold


def _time_of_day_risk(ts: datetime) -> float:
    """
    Returns a 0–1 risk score based on time of day and day of week.
    Night hours (22:00–06:00) and weekends get higher scores.
    """
    hour    = ts.hour
    weekday = ts.weekday()          # 0=Monday, 6=Sunday

    time_score = 0.0
    if 22 <= hour or hour < 6:      # night window
        time_score = 0.8
    elif 6 <= hour < 9:             # early morning
        time_score = 0.4
    else:
        time_score = 0.1

    weekend_score = 0.3 if weekday >= 5 else 0.0
    return round(min(time_score + weekend_score, 1.0), 2)


def _is_round_amount(amount: float) -> bool:
    """True if amount is an exact multiple of a suspicious round number."""
    for threshold in ROUND_AMOUNT_THRESHOLDS:
        if amount >= threshold and amount % threshold == 0:
            return True
    return False


def _is_structuring(amount: float, history: list[dict]) -> bool:
    """
    Detects potential structuring — multiple transactions just below CTR threshold.
    Flags if 2+ transactions in history fall in the structuring window.
    """
    lower_bound = CTR_THRESHOLD * (1 - STRUCTURING_WINDOW)
    current_near_threshold = lower_bound <= amount < CTR_THRESHOLD

    if not current_near_threshold:
        return False

    near_threshold_count = sum(
        1 for t in history
        if lower_bound <= t.get("amount", 0) < CTR_THRESHOLD
    )
    return near_threshold_count >= 2


def compute_behavioral_features(
    event: TransactionEvent,
    store: FeatureStoreClient,
) -> BehavioralFeatures:
    channel_history = store.get_channel_history(event.customer_id, n=5)
    txn_history     = store.get_customer_txn_history(event.customer_id)

    # Channel switch: current channel differs from majority of last 5
    channel_switch = False
    if channel_history:
        most_common = max(set(channel_history), key=channel_history.count)
        channel_switch = event.channel != most_common

    ts = event.timestamp.replace(tzinfo=timezone.utc) \
        if event.timestamp.tzinfo is None else event.timestamp

    return BehavioralFeatures(
        channel_switch_flag=channel_switch,
        time_of_day_risk=_time_of_day_risk(ts),
        is_round_amount=_is_round_amount(event.amount),
        is_structuring_pattern=_is_structuring(event.amount, txn_history),
    )
