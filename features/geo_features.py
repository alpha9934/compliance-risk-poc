from __future__ import annotations
"""
Geography and jurisdiction risk features.

Jurisdiction risk is one of the strongest compliance signals:
- Is the destination on the FATF blacklist?
- Is this a transfer from a low-risk to a high-risk country?
- Does the risk delta suggest deliberate jurisdiction escalation?
"""
import json
import os
from features.models import GeoFeatures
from ingestion.schemas.transaction_event import TransactionEvent

_POLICY_PATH = os.path.join(
    os.path.dirname(__file__), "../policies/jurisdiction_risk.json"
)

def _load_jurisdiction_data() -> dict:
    with open(_POLICY_PATH) as f:
        return json.load(f)

_JURISDICTION_DATA = _load_jurisdiction_data()
_RISK_RATINGS: dict[str, int] = _JURISDICTION_DATA["risk_ratings"]
_FATF_GREYLIST: set[str]      = set(_JURISDICTION_DATA["fatf_greylist"])
_FATF_BLACKLIST: set[str]     = set(_JURISDICTION_DATA["fatf_blacklist"])

HIGH_RISK_THRESHOLD = 4         # rating >= 4 → high risk


def compute_geo_features(event: TransactionEvent) -> GeoFeatures:
    """
    Looks up jurisdiction risk scores for origin and destination.
    Unknown jurisdictions default to risk rating 3 (moderate) — fail-safe.
    """
    origin_risk = _RISK_RATINGS.get(event.jurisdiction_origin, 3)
    dest_risk   = _RISK_RATINGS.get(event.jurisdiction_destination, 3)
    delta       = dest_risk - origin_risk

    return GeoFeatures(
        origin_risk_score=origin_risk,
        destination_risk_score=dest_risk,
        risk_score_delta=delta,
        is_high_risk_dest=dest_risk >= HIGH_RISK_THRESHOLD,
        is_fatf_greylist=event.jurisdiction_destination in _FATF_GREYLIST,
        is_fatf_blacklist=event.jurisdiction_destination in _FATF_BLACKLIST,
        is_cross_border=event.jurisdiction_origin != event.jurisdiction_destination,
    )
