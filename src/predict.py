"""Score test_unlabelled.csv and write predictions.csv matching sample_submission order."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_DIR, MODEL_PATH, PREDICTIONS_PATH
from src.features import data_paths, load_prepared, matrix_from_frame


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"model not found at {MODEL_PATH}; run python -m src.train first")

    payload = joblib.load(MODEL_PATH)
    bundle = payload["bundle"]

    _, test = load_prepared()
    X = matrix_from_frame(test)
    scores = bundle.predict_proba(X)[:, 1]
    # Keep strictly inside (0, 1)
    scores = np.clip(scores, 1e-6, 1.0 - 1e-6)

    pred = pd.DataFrame({"order_id": test["order_id"].astype(str), "score": scores})

    sample_path = data_paths()["sample"]
    sample = pd.read_csv(sample_path)
    sample["order_id"] = sample["order_id"].astype(str)
    # Reorder to match sample_submission exactly
    ordered = sample[["order_id"]].merge(pred, on="order_id", how="left", validate="one_to_one")
    if ordered["score"].isna().any():
        missing = ordered.loc[ordered["score"].isna(), "order_id"].tolist()[:5]
        raise SystemExit(f"Missing scores for order_ids e.g. {missing}")

    ordered.to_csv(PREDICTIONS_PATH, index=False)
    print(
        f"Wrote {PREDICTIONS_PATH} rows={len(ordered)} "
        f"score[min,max]=[{ordered['score'].min():.4f},{ordered['score'].max():.4f}]"
    )

    # Sanity vs sample
    assert list(ordered["order_id"]) == list(sample["order_id"])
    assert list(ordered.columns) == ["order_id", "score"]


if __name__ == "__main__":
    main()
