"""
02_model_training.py

Trains an XGBoost binary classifier on synthetic compliance data.
Saves the trained model to models/xgboost_v1.json as an xgb.Booster.

Uses xgb.train() + xgb.Booster directly — avoids the XGBClassifier
sklearn mixin which breaks on sklearn >= 1.6 with older XGBoost versions.

Run once before starting the API:
    python notebooks/02_model_training.py
"""
import os
import sys
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score,
)

random.seed(42)
np.random.seed(42)

# ── Jurisdiction tables ───────────────────────────────────────────────────
HIGH  = {"RU": 5, "IR": 5, "KP": 5, "BY": 5, "SY": 5}
MED   = {"AE": 3, "SA": 3, "MM": 4, "CU": 4, "VE": 4}
LOW   = {"US": 1, "GB": 1, "DE": 1, "JP": 1, "AU": 1, "FR": 1}
ALL   = {**HIGH, **MED, **LOW}
GREY  = {"AE", "MM", "YE", "PK"}
BLACK = {"IR", "KP"}

# ── Step 1: Generate synthetic labelled dataset ───────────────────────────
print("Step 1: Generating 5,000 synthetic transactions...")

records = []
for _ in range(5000):
    fraud = random.random() < 0.15

    if fraud:
        dest      = random.choice(list({**HIGH, **MED}))
        avg       = random.uniform(2000, 8000)
        amount    = random.uniform(avg * 3, avg * 20)
        cust_risk = random.choices([1,2,3,4,5], weights=[5,10,20,35,30])[0]
        alerts    = random.choices([0,1,2,3,4], weights=[30,25,25,12,8])[0]
        new_ben   = random.random() < 0.75
        ch_sw     = random.random() < 0.40
        round_amt = random.random() < 0.30
        struct    = random.random() < 0.25
        t_risk    = random.uniform(0.4, 1.0)
        cnt_24h   = random.randint(0, 3)
    else:
        dest      = random.choice(list(LOW))
        avg       = random.uniform(3000, 10000)
        amount    = max(100, random.gauss(avg, avg * 0.3))
        cust_risk = random.choices([1,2,3,4,5], weights=[30,40,20,7,3])[0]
        alerts    = random.choices([0,1,2], weights=[80,15,5])[0]
        new_ben   = random.random() < 0.25
        ch_sw     = random.random() < 0.10
        round_amt = random.random() < 0.08
        struct    = False
        t_risk    = random.uniform(0.0, 0.4)
        cnt_24h   = random.randint(0, 2)

    dest_risk = ALL.get(dest, 3)
    std       = avg * 0.4
    zscore    = (amount - avg) / max(std, 100)

    records.append({
        "count_24h":              cnt_24h,
        "count_7d":               cnt_24h * random.randint(2, 6),
        "sum_24h":                amount * cnt_24h * 0.8,
        "sum_7d":                 amount * cnt_24h * random.uniform(3, 8),
        "unique_dest_7d":         random.randint(1, 5) if fraud else random.randint(1, 3),
        "amount_zscore":          round(zscore, 4),
        "amount_vs_avg_ratio":    round(amount / avg, 4),
        "is_amount_outlier":      float(abs(zscore) > 2.5),
        "destination_risk_score": dest_risk,
        "risk_score_delta":       dest_risk - 1,
        "is_high_risk_dest":      float(dest_risk >= 4),
        "is_fatf_greylist":       float(dest in GREY),
        "is_fatf_blacklist":      float(dest in BLACK),
        "is_cross_border":        1.0,
        "is_new_beneficiary":     float(new_ben),
        "dest_account_age_days":  random.randint(0, 30) if new_ben else random.randint(30, 999),
        "customer_risk_rating":   cust_risk,
        "prior_alert_count":      alerts,
        "is_high_risk_customer":  float(cust_risk >= 4),
        "has_prior_alerts":       float(alerts > 0),
        "channel_switch_flag":    float(ch_sw),
        "time_of_day_risk":       round(t_risk, 2),
        "is_round_amount":        float(round_amt),
        "is_structuring_pattern": float(struct),
        "label":                  int(fraud),
    })

df = pd.DataFrame(records)
print(f"  {len(df)} records — {df['label'].sum()} high-risk ({df['label'].mean()*100:.1f}%)")

# ── Step 2: Train / test split ────────────────────────────────────────────
FEATURE_NAMES = [c for c in df.columns if c != "label"]
X = df[FEATURE_NAMES].values.astype(np.float32)
y = df["label"].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nStep 2: train={len(X_train)}, test={len(X_test)}")

# ── Step 3: Build DMatrix and train Booster ───────────────────────────────
# Using xgb.train() + Booster directly:
#   - No sklearn mixin → no __sklearn_tags__ error on any Python/sklearn version
#   - Faster inference (no wrapper overhead)
#   - Identical model format — same JSON file, same SHAP output
print("\nStep 3: Training XGBoost Booster...")

spw = float((y_train == 0).sum() / (y_train == 1).sum())

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=FEATURE_NAMES)

params = {
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": spw,
    "seed":             42,
    "verbosity":        0,
}

booster = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=[(dtest, "test")],
    verbose_eval=False,
)
print("  Done.")

# ── Step 4: Evaluate ──────────────────────────────────────────────────────
print("\nStep 4: Evaluating on test set...")

y_proba = booster.predict(dtest)
y_pred  = (y_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_proba)
p   = precision_score(y_test, y_pred)
r   = recall_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)

print(f"""
  ROC-AUC   : {auc:.4f}
  Precision : {p:.4f}
  Recall    : {r:.4f}
  F1        : {f1:.4f}
""")
print(classification_report(y_test, y_pred, target_names=["LOW/MED", "HIGH"]))

# ── Step 5: Feature importance ────────────────────────────────────────────
print("Step 5: Top 10 features by gain:")
scores = booster.get_score(importance_type="gain")
for name, gain in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {name:<32} {gain:.2f}")

# ── Step 6: Save ──────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
booster.save_model("models/xgboost_v1.json")

with open("models/xgboost_v1_meta.json", "w") as f:
    json.dump({
        "model_version":   "xgboost_v1",
        "model_type":      "xgb.Booster",   # NOT XGBClassifier
        "trained_at":      datetime.now(timezone.utc).isoformat(),
        "n_train":         int(len(X_train)),
        "n_features":      int(len(FEATURE_NAMES)),
        "feature_names":   FEATURE_NAMES,
        "metrics": {
            "roc_auc":   round(auc, 4),
            "precision": round(p,   4),
            "recall":    round(r,   4),
            "f1":        round(f1,  4),
        },
        "params": params,
    }, f, indent=2)

print("\nSaved → models/xgboost_v1.json")
print("Saved → models/xgboost_v1_meta.json")
print("\nDone. The scorer will load this model automatically on startup.")
