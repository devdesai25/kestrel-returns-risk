"""Fit the final model on all cleaned training data and write model.joblib."""

from __future__ import annotations

import json

import joblib

from src.config import MODEL_PATH, MODELS_DIR, MODEL_VERSION, REPORTS_DIR
from src.features import (
    build_rate_lookup,
    data_paths,
    load_prepared,
    matrix_from_frame,
    reference_values,
)
from src.model import fit_production_model


def _dimension_lookups() -> tuple[dict, dict]:
    """Compact sku/customer tables so /score works with no data pack on disk."""
    import pandas as pd

    paths = data_paths()
    products = pd.read_csv(paths["products"])
    customers = pd.read_csv(paths["customers"])
    products_by_sku = {
        str(r["sku"]): {
            "family": str(r["family"]),
            "list_price_inr": float(r["list_price_inr"]),
            "warranty_months": float(r["warranty_months"]),
        }
        for _, r in products.iterrows()
    }
    customers_by_id = {
        str(r["customer_id"]): {
            "city": str(r["city"]),
            "state": str(r["state"]),
            "signup_date": str(r["signup_date"]),
            "shield_member": str(r["shield_member"]),
        }
        for _, r in customers.iterrows()
    }
    return products_by_sku, customers_by_id


def fit_final_model(train):
    X = matrix_from_frame(train)
    y = train["returned"].astype(int).to_numpy()
    bundle = fit_production_model(X, y)
    refs = reference_values(train)
    rates = build_rate_lookup(train)
    products_by_sku, customers_by_id = _dimension_lookups()
    # Modal defaults when an id is unknown at score time
    state_mode = train["state"].astype(str).mode().iloc[0]
    meta = {
        "model_version": MODEL_VERSION,
        "model_family": "logistic_regression",
        "n_train": int(len(train)),
        "feature_columns": list(X.columns),
        "references": refs,
        "rate_lookup": rates,
        "positive_rate": float(y.mean()),
        "products_by_sku": products_by_sku,
        "customers_by_id": customers_by_id,
        "defaults": {
            "state": state_mode,
            "city": "UNKNOWN",
            "shield_member": "N",
            "family": "UNKNOWN",
            "list_price_inr": float(train["value"].median()),
            "warranty_months": 12.0,
        },
    }
    return bundle, meta


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train, _ = load_prepared()
    bundle, meta = fit_final_model(train)
    payload = {"bundle": bundle, "meta": meta}
    joblib.dump(payload, MODEL_PATH)
    print(
        f"Wrote {MODEL_PATH} ({meta['model_family']}, n={meta['n_train']}, "
        f"pos_rate={meta['positive_rate']:.4f})"
    )
    # Fingerprint without the large customer map
    slim = {k: v for k, v in meta.items() if k not in ("customers_by_id", "products_by_sku")}
    slim["n_products"] = len(meta["products_by_sku"])
    slim["n_customers"] = len(meta["customers_by_id"])
    (REPORTS_DIR / "train_fingerprint.json").write_text(json.dumps(slim, indent=2, default=str))


if __name__ == "__main__":
    main()
