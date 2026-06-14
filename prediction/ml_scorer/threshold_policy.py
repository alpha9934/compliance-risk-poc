from __future__ import annotations
"""
threshold_policy.py

Loads risk score thresholds from config/thresholds.yaml and classifies
a raw float score into HIGH / MEDIUM / LOW.

Thresholds are segmented by product type and destination jurisdiction so
the same score means different things in different contexts:
  - A 0.65 score on a US domestic ACH is LOW
  - A 0.65 score on a SWIFT transfer to AE is HIGH

Any change to thresholds.yaml must be version-controlled and approved
before reaching production (governance requirement NFR-06).
"""
import os
import yaml
from ingestion.schemas.transaction_event import RiskClass

_THRESHOLDS_PATH = os.path.join(
    os.path.dirname(__file__), "../../config/thresholds.yaml"
)


def _load() -> dict:
    with open(_THRESHOLDS_PATH) as f:
        return yaml.safe_load(f)


# Load once at import time — reload only on service restart
_THRESHOLDS: dict = _load()


def classify_risk(
    score: float,
    product_type: str = "DEFAULT",
    jurisdiction_destination: str = "DEFAULT",
) -> RiskClass:
    """
    Returns the RiskClass for a given score, product, and destination.

    Lookup order:
      1. product_type → jurisdiction_destination  (most specific)
      2. product_type → DEFAULT
      3. DEFAULT     → DEFAULT                    (fallback)
    """
    product_config = _THRESHOLDS.get(product_type, {})
    config = (
        product_config.get(jurisdiction_destination)
        or product_config.get("DEFAULT")
        or _THRESHOLDS["DEFAULT"]["DEFAULT"]
    )

    high_threshold   = config["high"]
    medium_threshold = config["medium"]

    if score >= high_threshold:
        return RiskClass.HIGH
    elif score >= medium_threshold:
        return RiskClass.MEDIUM
    return RiskClass.LOW


def get_thresholds(
    product_type: str = "DEFAULT",
    jurisdiction_destination: str = "DEFAULT",
) -> dict:
    """Returns the raw threshold dict for a given product / jurisdiction pair."""
    product_config = _THRESHOLDS.get(product_type, {})
    return (
        product_config.get(jurisdiction_destination)
        or product_config.get("DEFAULT")
        or _THRESHOLDS["DEFAULT"]["DEFAULT"]
    )
