"""Paths, constants, and operating thresholds."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("KESTREL_DATA_DIR", ROOT / "data")).expanduser().resolve()
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
WEB_DIR = ROOT / "web"
PREDICTIONS_PATH = ROOT / "predictions.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

MODEL_VERSION = "logit-v1"

# Policy costs (ops-policy §4 / §7)
C_RETURN = 1150.0
C_CALL = 45.0
CALL_PREVENT = 0.35
CANCEL_ON_HOLD = 0.12
# Assumption — not in the pack; always label as such
MARGIN_PCT = 0.25
MARGIN_PCT_SENSITIVITY = (0.15, 0.25, 0.40)

# Call vs ship break-even: C_CALL / (CALL_PREVENT * C_RETURN)
OPERATING_THRESHOLD = C_CALL / (CALL_PREVENT * C_RETURN)  # ≈ 0.1118

HOLDOUT_BOUNDARY = "2026-04-01"
RANDOM_STATE = 42

# HistGradientBoosting hyperparameters (validated)
HGB_PARAMS = dict(
    max_depth=3,
    learning_rate=0.05,
    max_iter=500,
    min_samples_leaf=60,
    l2_regularization=1.0,
    random_state=RANDOM_STATE,
)

NUMERIC_FEATURES = [
    "hour",
    "dow",
    "month",
    "log_value",
    "value",
    "discount_pct",
    "qty",
    "promised_delivery_days",
    "customer_prior_orders",
    "customer_prior_returns",
    "prior_ret_rate",
    "pin_missing",
    "note_missing",
    "cust_since_days",
    "price_ratio",
    "warranty_months",
]

CATEGORICAL_FEATURES = [
    "sales_channel",
    "payment_mode",
    "is_gift",
    "family",
    "state",
    "shield_member",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Excluded from features (documented in README / data_audit)
LEAKAGE_COLUMNS = ["last_service_event_type", "pickup_scheduled_at"]
IDENTIFIER_COLUMNS = ["order_id", "customer_id"]
UNUSED_COLUMNS = ["source", "delivery_note", "sku", "city"]
