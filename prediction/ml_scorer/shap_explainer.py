from __future__ import annotations
"""
shap_explainer.py

Computes SHAP values for a single xgb.Booster prediction.

shap.TreeExplainer accepts xgb.Booster natively — no sklearn wrapper needed.

For binary classification with xgb.Booster (objective=binary:logistic):
  - explainer.shap_values(x) returns shape (n_samples, n_features)
    as a single 2-D array (not a list of two arrays like XGBClassifier)
  - expected_value is a single float (log-odds space)
  - values are already the positive-class contributions
"""
import shap
import numpy as np
from pydantic import BaseModel


# ── Output models ─────────────────────────────────────────────────────────

class FeatureContribution(BaseModel):
    feature:          str    # e.g. "amount_zscore"
    value:            float  # raw feature value, e.g. 3.72
    contribution:     float  # SHAP value — positive raises score, negative lowers
    abs_contribution: float  # abs(contribution) — used for ranking


class SHAPExplanation(BaseModel):
    top_features:    list[FeatureContribution]
    base_value:      float   # model's average output over training data
    predicted_score: float   # final model output for this instance
    model_version:   str


# ── Explainer ─────────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    Wraps shap.TreeExplainer for xgb.Booster models.
    Instantiated once at startup and reused across all scoring calls.
    """

    def __init__(self, booster, feature_names: list[str], model_version: str):
        # model_output="raw_values" keeps outputs in log-odds space,
        # consistent with booster.predict() which returns probabilities.
        self._explainer = shap.TreeExplainer(
            booster,
        )
        self._feature_names = feature_names
        self._model_version = model_version

    def explain(
        self,
        x: np.ndarray,
        predicted_score: float,
        top_n: int = 5,
    ) -> SHAPExplanation:
        """
        Computes SHAP values for a single observation (shape 1 × N).
        Returns the top_n features by absolute SHAP contribution.

        Args:
            x:               numpy array shape (1, n_features), dtype float32
            predicted_score: already-computed probability from booster.predict()
            top_n:           how many top features to return (default 5)

        Returns:
            SHAPExplanation with ranked feature contributions
        """
        shap_values = self._explainer.shap_values(x)

        # xgb.Booster with binary:logistic returns shape (n_samples, n_features)
        # — a single 2-D array, NOT a list of two class arrays.
        # Handle all possible shapes defensively:
        if isinstance(shap_values, list):
            # Old-style: [neg_class, pos_class] — take positive class
            values = np.array(shap_values[1]).flatten()
        elif shap_values.ndim == 3:
            # (n_samples, n_features, n_classes) — take positive class
            values = shap_values[0, :, 1]
        else:
            # (n_samples, n_features) — standard Booster output
            values = shap_values[0]

        # Base value
        ev = self._explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base_value = float(ev[1] if len(ev) > 1 else ev[0])
        else:
            base_value = float(ev)

        # Build FeatureContribution for every feature
        contributions = [
            FeatureContribution(
                feature=self._feature_names[i],
                value=round(float(x[0, i]), 4),
                contribution=round(float(values[i]), 4),
                abs_contribution=round(abs(float(values[i])), 4),
            )
            for i in range(len(self._feature_names))
        ]

        # Sort descending by |contribution|, return top N
        top = sorted(
            contributions,
            key=lambda c: c.abs_contribution,
            reverse=True,
        )[:top_n]

        return SHAPExplanation(
            top_features=top,
            base_value=round(base_value, 4),
            predicted_score=round(predicted_score, 4),
            model_version=self._model_version,
        )
