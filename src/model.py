"""Model wrappers and builders (never run as __main__ — keeps joblib pickles stable)."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    HGB_PARAMS,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


class Scorer(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


def build_hgb() -> HistGradientBoostingClassifier:
    cat_idx = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_FEATURES]
    return HistGradientBoostingClassifier(
        categorical_features=cat_idx,
        **HGB_PARAMS,
    )


def build_logistic() -> Pipeline:
    """Interpretable sigmoid baseline — one-hot categoricals, no class_weight.

    class_weight='balanced' inflates scores and collapses the operating threshold
    into call-almost-everyone; unweighted LR is better calibrated for p*≈0.11.
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


class HGBBundle:
    """HistGradientBoosting with fixed category levels for train/serve encoding."""

    def __init__(self, model: HistGradientBoostingClassifier, cat_levels: dict[str, list[str]]):
        self.model = model
        self.cat_levels = cat_levels
        self.feature_columns = list(FEATURE_COLUMNS)
        self.categorical_features = list(CATEGORICAL_FEATURES)
        self.numeric_features = list(NUMERIC_FEATURES)
        self.model_version = "hgb-v1"

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        arr = np.empty((len(X), len(FEATURE_COLUMNS)), dtype=np.float64)
        for i, col in enumerate(FEATURE_COLUMNS):
            if col in CATEGORICAL_FEATURES:
                levels = self.cat_levels[col]
                level_to_code = {v: k for k, v in enumerate(levels)}
                vals = (
                    X[col]
                    .astype(str)
                    .map(lambda v: level_to_code[v] if v in level_to_code else np.nan)
                    .to_numpy(dtype=np.float64)
                )
                arr[:, i] = vals
            else:
                arr[:, i] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy()
        return arr

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._encode(X))

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "HGBBundle":
        self.cat_levels = {
            c: sorted(X[c].astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES
        }
        self.model.fit(self._encode(X), y)
        return self


class LogisticBundle:
    """Sklearn logistic pipeline over the shared feature frame."""

    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline if pipeline is not None else build_logistic()
        self.feature_columns = list(FEATURE_COLUMNS)
        self.categorical_features = list(CATEGORICAL_FEATURES)
        self.numeric_features = list(NUMERIC_FEATURES)
        self.model_version = MODEL_VERSION

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[FEATURE_COLUMNS].copy()
        for c in CATEGORICAL_FEATURES:
            out[c] = out[c].astype(str)
        for c in NUMERIC_FEATURES:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        return out

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogisticBundle":
        self.pipeline.fit(self._frame(X), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(self._frame(X))


def fit_production_model(X: pd.DataFrame, y: np.ndarray) -> LogisticBundle:
    """Final deliverable: logistic regression (wins holdout AUC/PR + economics)."""
    bundle = LogisticBundle()
    bundle.fit(X, y)
    return bundle
