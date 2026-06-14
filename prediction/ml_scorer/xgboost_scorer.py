from __future__ import annotations
"""
xgboost_scorer.py

Loads a trained XGBoost model and scores a FeatureVector.

Uses xgb.Booster directly — NOT XGBClassifier — to avoid the
sklearn mixin __sklearn_tags__ error that breaks on sklearn >= 1.6
with older XGBoost versions.

xgb.Booster is also:
  - Faster (no sklearn wrapper overhead)
  - Python-version agnostic (no sklearn dependency at inference time)
  - Identical model format — same JSON file, same SHAP values

Design:
  - Booster loads ONCE at module import (singleton)
  - Heuristic fallback when model file not found
  - FEATURE_NAMES is a hard contract with FeatureVector.to_model_input()
"""
import os
import time
import numpy as np
import xgboost as xgb
from datetime import datetime, timezone
from pydantic import BaseModel

from features.models import FeatureVector
from ingestion.schemas.transaction_event import RiskClass
from prediction.ml_scorer.shap_explainer import (
    SHAPExplainer, SHAPExplanation, FeatureContribution,
)
from prediction.ml_scorer.threshold_policy import classify_risk
from config.settings import settings


# ── Feature contract ──────────────────────────────────────────────────────
# Order MUST match FeatureVector.to_model_input() exactly.
# Never reorder without retraining the model.

FEATURE_NAMES: list[str] = [
    "count_24h",
    "count_7d",
    "sum_24h",
    "sum_7d",
    "unique_dest_7d",
    "amount_zscore",
    "amount_vs_avg_ratio",
    "is_amount_outlier",
    "destination_risk_score",
    "risk_score_delta",
    "is_high_risk_dest",
    "is_fatf_greylist",
    "is_fatf_blacklist",
    "is_cross_border",
    "is_new_beneficiary",
    "dest_account_age_days",
    "customer_risk_rating",
    "prior_alert_count",
    "is_high_risk_customer",
    "has_prior_alerts",
    "channel_switch_flag",
    "time_of_day_risk",
    "is_round_amount",
    "is_structuring_pattern",
]


# ── Output model ──────────────────────────────────────────────────────────

class ScorerOutput(BaseModel):
    transaction_id: str
    risk_score:     float           # 0.0 – 1.0
    risk_class:     RiskClass
    shap_explanation: SHAPExplanation
    model_version:  str
    scored_at:      datetime
    latency_ms:     float
    scorer_mode:    str             # "xgboost" | "heuristic"


# ── Booster singleton ─────────────────────────────────────────────────────

def _load_booster() -> tuple[xgb.Booster | None, str]:
    """
    Loads xgb.Booster from disk once at startup.
    Returns (None, "heuristic-v1") if file not found — never raises.

    Why Booster not XGBClassifier?
      XGBClassifier._load_model_attributes() calls sklearn.base.is_classifier()
      which calls get_tags() → __sklearn_tags__(), added in sklearn 1.4.
      XGBoost < 2.1 doesn't implement __sklearn_tags__ → AttributeError.
      xgb.Booster has zero sklearn dependency — works on any version.
    """
    path = settings.model_path
    if os.path.exists(path):
        booster = xgb.Booster()
        booster.load_model(path)
        version = os.path.basename(path).replace(".json", "")
        print(f"[Scorer] Loaded Booster: {path} (version: {version})")
        return booster, version

    print(
        f"[Scorer] Model not found at '{path}' — "
        "using heuristic scorer. Run: python notebooks/02_model_training.py"
    )
    return None, "heuristic-v1"


_BOOSTER, _MODEL_VERSION = _load_booster()

# SHAP explainer wraps the Booster directly — works identically
_SHAP_EXPLAINER: SHAPExplainer | None = (
    SHAPExplainer(_BOOSTER, FEATURE_NAMES, _MODEL_VERSION)
    if _BOOSTER is not None else None
)


# ── Heuristic fallback scorer ─────────────────────────────────────────────

def _heuristic_score(fv: FeatureVector) -> float:
    """
    Rule-based risk scorer. Used when model file is not yet present.
    Weights approximate what XGBoost learns from the synthetic dataset.
    Capped at 0.99, floored at 0.0.
    """
    s = 0.0
    s += fv.destination_risk_score * 0.07
    s += min(abs(fv.amount_zscore) * 0.05, 0.25)
    if fv.is_fatf_blacklist:      s += 0.20
    if fv.is_structuring_pattern: s += 0.10
    if fv.is_fatf_greylist:       s += 0.08
    if fv.is_high_risk_dest:      s += 0.08
    if fv.is_amount_outlier:      s += 0.07
    if fv.is_high_risk_customer:  s += 0.06
    if fv.is_new_beneficiary:     s += 0.05
    if fv.has_prior_alerts:       s += 0.05
    if fv.is_round_amount:        s += 0.03
    if fv.channel_switch_flag:    s += 0.03
    if fv.count_24h > 5:          s += 0.05
    if fv.count_24h > 10:         s += 0.05
    s += (fv.customer_risk_rating - 1) * 0.01
    return round(min(max(s, 0.0), 0.99), 4)


def _heuristic_shap(fv: FeatureVector, score: float) -> SHAPExplanation:
    """Plausible SHAP-style explanation for the heuristic fallback."""
    raw = {
        "destination_risk_score":   fv.destination_risk_score * 0.07,
        "amount_zscore":            min(abs(fv.amount_zscore) * 0.05, 0.25),
        "is_fatf_blacklist":        0.20 if fv.is_fatf_blacklist else 0.0,
        "is_structuring_pattern":   0.10 if fv.is_structuring_pattern else 0.0,
        "is_fatf_greylist":         0.08 if fv.is_fatf_greylist else 0.0,
        "is_high_risk_dest":        0.08 if fv.is_high_risk_dest else 0.0,
        "is_amount_outlier":        0.07 if fv.is_amount_outlier else 0.0,
        "is_high_risk_customer":    0.06 if fv.is_high_risk_customer else 0.0,
        "is_new_beneficiary":       0.05 if fv.is_new_beneficiary else 0.0,
        "count_24h":                min(fv.count_24h * 0.005, 0.10),
    }
    top = sorted(
        [
            FeatureContribution(
                feature=k,
                value=round(float(getattr(fv, k, 0)), 4),
                contribution=round(v, 4),
                abs_contribution=round(abs(v), 4),
            )
            for k, v in raw.items()
        ],
        key=lambda c: c.abs_contribution,
        reverse=True,
    )[:5]
    return SHAPExplanation(
        top_features=top,
        base_value=0.12,
        predicted_score=score,
        model_version="heuristic-v1",
    )


# ── Public API ────────────────────────────────────────────────────────────

def score_transaction(
    fv: FeatureVector,
    product_type: str = "DEFAULT",
    jurisdiction_destination: str = "DEFAULT",
) -> ScorerOutput:
    """
    Scores a single FeatureVector.

    Uses xgb.Booster when model is loaded; heuristic scorer as fallback.
    Caller gets identical ScorerOutput shape either way.

    Args:
        fv:                       FeatureVector from feature_pipeline
        product_type:             e.g. "WIRE_TRANSFER" — for threshold lookup
        jurisdiction_destination: e.g. "AE" — for threshold lookup

    Returns:
        ScorerOutput with risk_score, risk_class, shap_explanation
    """
    start = time.perf_counter()

    if _BOOSTER is not None and _SHAP_EXPLAINER is not None:
        # ── Booster path (production) ──────────────────────────────────
        x = np.array([fv.to_model_input()], dtype=np.float32)

        # DMatrix with feature names enables name-based SHAP alignment
        dmatrix    = xgb.DMatrix(x, feature_names=FEATURE_NAMES)
        risk_score = float(_BOOSTER.predict(dmatrix)[0])
        risk_class = classify_risk(risk_score, product_type, jurisdiction_destination)
        shap_exp   = _SHAP_EXPLAINER.explain(x, risk_score)
        scorer_mode = "xgboost"

    else:
        # ── Heuristic path (fallback) ──────────────────────────────────
        risk_score  = _heuristic_score(fv)
        risk_class  = classify_risk(risk_score, product_type, jurisdiction_destination)
        shap_exp    = _heuristic_shap(fv, risk_score)
        scorer_mode = "heuristic"

    return ScorerOutput(
        transaction_id=fv.transaction_id,
        risk_score=round(risk_score, 4),
        risk_class=risk_class,
        shap_explanation=shap_exp,
        model_version=_MODEL_VERSION,
        scored_at=datetime.now(timezone.utc),
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        scorer_mode=scorer_mode,
    )
