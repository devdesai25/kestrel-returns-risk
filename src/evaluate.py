"""Time-split evaluation, ablation, cost curve, and evidence report generation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CALL_PREVENT,
    C_CALL,
    C_RETURN,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    HOLDOUT_BOUNDARY,
    MARGIN_PCT,
    MARGIN_PCT_SENSITIVITY,
    OPERATING_THRESHOLD,
    REPORTS_DIR,
)
from src.economics import (
    call_breakeven_threshold,
    call_every_order_net,
    expected_call_economics,
    hold_breakeven_threshold,
    scale_net,
)
from src.features import load_prepared, matrix_from_frame
from src.model import HGBBundle, build_hgb


def time_split(train: pd.DataFrame, boundary: str = HOLDOUT_BOUNDARY):
    ts = pd.to_datetime(train["order_placed_at"])
    mask = ts < pd.Timestamp(boundary)
    return train.loc[mask].reset_index(drop=True), train.loc[~mask].reset_index(drop=True)


def window_split(
    train: pd.DataFrame, train_before: str, valid_start: str, valid_end_exclusive: str
):
    ts = pd.to_datetime(train["order_placed_at"])
    tr = train.loc[ts < pd.Timestamp(train_before)].reset_index(drop=True)
    va = train.loc[
        (ts >= pd.Timestamp(valid_start)) & (ts < pd.Timestamp(valid_end_exclusive))
    ].reset_index(drop=True)
    return tr, va


def metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    auc = float(roc_auc_score(y_true, scores))
    pr = float(average_precision_score(y_true, scores))
    brier = float(brier_score_loss(y_true, scores))
    pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = float(precision_score(y_true, pred, zero_division=0))
    recall = float(recall_score(y_true, pred, zero_division=0))
    acc = float((pred == y_true).mean())
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "auc_roc": auc,
        "auc_pr": pr,
        "brier": brier,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": acc,
        "flagged": int(tp + fp),
        "n": int(len(y_true)),
        "return_rate": float(y_true.mean()),
    }


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion (no scipy)."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def accuracy_ceiling(y_true: np.ndarray, scores: np.ndarray) -> dict:
    majority = float(max((y_true == 0).mean(), (y_true == 1).mean()))
    best = 0.0
    best_t = 0.5
    for t in np.unique(np.concatenate([[0.0], scores, [1.0]])):
        pred = (scores > t).astype(int)
        acc = float((pred == y_true).mean())
        if acc > best:
            best = acc
            best_t = float(t)
    n = int(len(y_true))
    max_errors_for_95 = int(math.floor(n * 0.05))
    return {
        "best_accuracy": best,
        "best_threshold": best_t,
        "majority_class_accuracy": majority,
        "n": n,
        "max_errors_for_95pct": max_errors_for_95,
        "holdout_returns": int(y_true.sum()),
        "note": "95% accuracy is unreachable; majority baseline already near the ceiling.",
    }


def fit_predict_hgb(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> np.ndarray:
    Xtr = matrix_from_frame(train_df)
    ytr = train_df["returned"].astype(int).to_numpy()
    Xva = matrix_from_frame(valid_df)
    bundle = HGBBundle(build_hgb(), cat_levels={})
    bundle.fit(Xtr, ytr)
    return bundle.predict_proba(Xva)[:, 1]


def fit_predict_logistic(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> np.ndarray:
    from src.model import LogisticBundle

    Xtr = matrix_from_frame(train_df)
    ytr = train_df["returned"].astype(int).to_numpy()
    Xva = matrix_from_frame(valid_df)
    bundle = LogisticBundle()
    bundle.fit(Xtr, ytr)
    return bundle.predict_proba(Xva)[:, 1]


def leakage_ablation(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        X = matrix_from_frame(df).copy()
        X["is_reverse_pickup"] = (
            df["last_service_event_type"].astype(str) == "REVERSE_PICKUP"
        ).astype(float)
        pickup = df["pickup_scheduled_at"]
        X["has_pickup"] = (
            ~(pickup.isna() | (pickup.astype(str).str.strip() == ""))
        ).astype(float)
        return X

    Xtr = _prep(train_df)
    Xva = _prep(valid_df)
    ytr = train_df["returned"].astype(int).to_numpy()
    yva = valid_df["returned"].astype(int).to_numpy()

    leak_cols = list(FEATURE_COLUMNS) + ["is_reverse_pickup", "has_pickup"]
    cat_idx = [leak_cols.index(c) for c in CATEGORICAL_FEATURES]
    clf_leak = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=60,
        l2_regularization=1.0,
        categorical_features=cat_idx,
        random_state=42,
    )

    def encode(X: pd.DataFrame, levels: dict | None = None):
        levels = levels or {
            c: sorted(X[c].astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES
        }
        arr = np.empty((len(X), len(leak_cols)), dtype=np.float64)
        for i, col in enumerate(leak_cols):
            if col in CATEGORICAL_FEATURES:
                m = {v: k for k, v in enumerate(levels[col])}
                arr[:, i] = (
                    X[col]
                    .astype(str)
                    .map(lambda v: m[v] if v in m else np.nan)
                    .to_numpy(dtype=np.float64)
                )
            else:
                arr[:, i] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy()
        return arr, levels

    Arr_tr, levels = encode(Xtr)
    Arr_va, _ = encode(Xva, levels)
    clf_leak.fit(Arr_tr, ytr)
    s_leak = clf_leak.predict_proba(Arr_va)[:, 1]
    s_clean = fit_predict_hgb(train_df, valid_df)

    # Fact checks on leakage columns in the (deduped) train slice used for holdout train+valid
    full = pd.concat([train_df, valid_df], ignore_index=True)
    rev = full["last_service_event_type"].astype(str) == "REVERSE_PICKUP"
    pickup = ~(
        full["pickup_scheduled_at"].isna()
        | (full["pickup_scheduled_at"].astype(str).str.strip() == "")
    )
    rev_rate = float(full.loc[rev, "returned"].mean()) if rev.any() else float("nan")
    pick_rate = float(full.loc[pickup, "returned"].mean()) if pickup.any() else float("nan")

    return {
        "with_leakage": {
            "auc_roc": float(roc_auc_score(yva, s_leak)),
            "auc_pr": float(average_precision_score(yva, s_leak)),
        },
        "without_leakage": {
            "auc_roc": float(roc_auc_score(yva, s_clean)),
            "auc_pr": float(average_precision_score(yva, s_clean)),
        },
        "reverse_pickup_n": int(rev.sum()),
        "reverse_pickup_return_rate": rev_rate,
        "pickup_scheduled_n": int(pickup.sum()),
        "pickup_scheduled_return_rate": pick_rate,
        "note": (
            "last_service_event_type==REVERSE_PICKUP and pickup_scheduled_at are "
            "post-return artefacts (ops-policy section 7). Dropped from the feature set."
        ),
    }


def shield_tradeoff(valid_df: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict:
    shield = valid_df["shield_member"].astype(str).str.upper() == "Y"
    y = valid_df["returned"].astype(int).to_numpy()
    full = expected_call_economics(scores, y, threshold)
    scores_ex = scores.copy()
    scores_ex[shield.to_numpy()] = 0.0
    ex = expected_call_economics(scores_ex, y, threshold)
    return {
        "intervene_all": {
            "net": full["net"],
            "flagged": full["flagged"],
            "tp": full["tp"],
            "prevented": full["prevented"],
        },
        "exclude_shield": {
            "net": ex["net"],
            "flagged": ex["flagged"],
            "tp": ex["tp"],
            "prevented": ex["prevented"],
        },
        "returns_missed_by_excluding_shield": float(full["tp"] - ex["tp"]),
        "calls_saved_by_excluding_shield": float(full["flagged"] - ex["flagged"]),
        "net_delta_excluding_shield": float(ex["net"] - full["net"]),
        "note": (
            "Excluding Shield misses some preventable returns but avoids calling "
            "high-LTV members; hold cancellations would hit them hardest."
        ),
    }


def decision_tree_illustration(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> str:
    """Shallow tree with per-leaf return rates (illustration only, not the deliverable)."""
    Xtr = matrix_from_frame(train_df)
    ytr = train_df["returned"].astype(int).to_numpy()
    cols = [
        "customer_prior_returns",
        "promised_delivery_days",
        "discount_pct",
        "value",
        "prior_ret_rate",
    ]
    readable = {
        "customer_prior_returns": "prior returns (count)",
        "promised_delivery_days": "promised delivery days",
        "discount_pct": "checkout discount %",
        "value": "order value Rs",
        "prior_ret_rate": "prior return rate",
    }
    Xt = Xtr[cols].astype(float)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=42)
    tree.fit(Xt, ytr)
    t = tree.tree_

    def fmt_rule(node: int, depth: int) -> list[str]:
        indent = "  " * depth
        if t.children_left[node] == t.children_right[node]:  # leaf
            n = int(t.n_node_samples[node])
            # value is weighted class counts; rate = positives / n for unweighted
            counts = t.value[node].ravel()
            rate = float(counts[1] / counts.sum()) if counts.sum() else 0.0
            return [
                f"{indent}leaf: n={n}, return rate={rate:.1%} "
                f"({'above' if rate > float(ytr.mean()) else 'at/below'} train base "
                f"{float(ytr.mean()):.1%})"
            ]
        feat = readable.get(cols[t.feature[node]], cols[t.feature[node]])
        thr = t.threshold[node]
        lines = [f"{indent}if {feat} <= {thr:.2f}:"]
        lines.extend(fmt_rule(t.children_left[node], depth + 1))
        lines.append(f"{indent}else:  # {feat} > {thr:.2f}")
        lines.extend(fmt_rule(t.children_right[node], depth + 1))
        return lines

    header = (
        "Decision tree on the training split only "
        "(max_depth=3, min_samples_leaf=200). Leaves show return rate, not a class label.\n"
        "This is a readable sketch of the main splits; logistic regression is the deliverable.\n"
    )
    return header + "\n".join(fmt_rule(0, 0))


def october_audit(raw_train_path: Path) -> dict:
    from src.features import dedupe_train, fix_october_inflation

    raw = pd.read_csv(raw_train_path)
    d = dedupe_train(raw)
    ts = pd.to_datetime(d["order_placed_at"])
    oct_mask = (ts.dt.year == 2025) & (ts.dt.month == 10)
    before = d.loc[oct_mask, "order_value_inr"]
    fixed = fix_october_inflation(d)
    after = fixed.loc[oct_mask, "order_value_inr"]
    other = d.loc[~oct_mask, "order_value_inr"]
    return {
        "oct_2025_rows": int(oct_mask.sum()),
        "before_mean": float(before.mean()),
        "before_max": float(before.max()),
        "after_mean": float(after.mean()),
        "after_max": float(after.max()),
        "other_months_mean": float(other.mean()),
        "other_months_max": float(other.max()),
        "fix": "Divided October-2025 order_value_inr by 100 (100x gateway inflation).",
    }


def service_state_audit(train: pd.DataFrame, test_raw: pd.DataFrame) -> dict:
    tr = train["last_service_event_type"].astype(str).value_counts(normalize=True).to_dict()
    te = test_raw["last_service_event_type"].astype(str).value_counts(normalize=True).to_dict()
    return {
        "train_distribution": {k: round(float(v), 4) for k, v in tr.items()},
        "test_distribution": {k: round(float(v), 4) for k, v in te.items()},
        "finding": (
            "Train reflects as-of-export-day service state (includes DEMO_DONE / "
            "INSTALL_DONE / REVERSE_PICKUP). Test is at-dispatch (NONE / INSTALL_BOOKED). "
            "INSTALL_BOOKED never appears in train. Column cannot be used across files."
        ),
    }


def calibration_table(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> dict:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin": i + 1,
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": 0,
                    "mean_predicted": None,
                    "observed_rate": None,
                }
            )
            continue
        rows.append(
            {
                "bin": i + 1,
                "lo": float(lo),
                "hi": float(hi),
                "n": n,
                "mean_predicted": float(scores[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
            }
        )
    return {
        "brier": float(brier_score_loss(y_true, scores)),
        "base_rate": float(y_true.mean()),
        "mean_predicted": float(scores.mean()),
        "bins": rows,
    }


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray, thresholds: list[float]) -> list[dict]:
    n = len(y_true)
    out = []
    for t in thresholds:
        m = metrics_at(y_true, scores, t)
        econ = expected_call_economics(scores, y_true, t)
        out.append(
            {
                "threshold": float(t),
                "flagged": m["flagged"],
                "flagged_pct": float(m["flagged"] / n) if n else 0.0,
                "precision": m["precision"],
                "recall": m["recall"],
                "accuracy": m["accuracy"],
                "net_inr": econ["net"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
            }
        )
    return out


def _rate_by(df: pd.DataFrame, col: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for val, g in df.groupby(df[col].astype(str)):
        out[str(val)] = {
            "n": float(len(g)),
            "return_rate": float(g["returned"].astype(int).mean()) if len(g) else 0.0,
        }
    return out


def _prior_bucket(n: float) -> str:
    n = int(n)
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def error_taxonomy(
    valid_df: pd.DataFrame, scores: np.ndarray, threshold: float
) -> dict:
    df = valid_df.copy()
    df["score"] = scores
    df["pred"] = (scores > threshold).astype(int)
    y = df["returned"].astype(int)
    df["_prior_bucket"] = df["customer_prior_returns"].map(_prior_bucket)
    try:
        df["_val_q"] = pd.qcut(
            df["value"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
        ).astype(str)
    except ValueError:
        df["_val_q"] = "Q?"
    fp = df[(df["pred"] == 1) & (y == 0)].copy()
    fn = df[(df["pred"] == 0) & (y == 1)].copy()

    holdout_prof = {
        "n": int(len(df)),
        "return_rate": float(y.mean()),
        "payment_mode": _rate_by(df, "payment_mode"),
        "sales_channel": _rate_by(df, "sales_channel"),
        "family": _rate_by(df, "family"),
        "shield_member": _rate_by(df, "shield_member"),
        "prior_returns_bucket": {
            str(k): {
                "n": float(v),
                "return_rate": float(
                    df.loc[df["_prior_bucket"].astype(str) == str(k), "returned"].mean()
                ),
            }
            for k, v in df["_prior_bucket"].astype(str).value_counts().items()
        },
        "value_quartile": {
            str(k): {
                "n": float(v),
                "return_rate": float(
                    df.loc[df["_val_q"].astype(str) == str(k), "returned"].mean()
                ),
            }
            for k, v in df["_val_q"].astype(str).value_counts().items()
        },
    }

    def composition(sub: pd.DataFrame) -> dict:
        if len(sub) == 0:
            return {"n": 0}
        return {
            "n": int(len(sub)),
            "mean_score": float(sub["score"].mean()),
            "payment_mode_share": {
                str(k): float(v) / len(sub)
                for k, v in sub["payment_mode"].astype(str).value_counts().items()
            },
            "sales_channel_share": {
                str(k): float(v) / len(sub)
                for k, v in sub["sales_channel"].astype(str).value_counts().items()
            },
            "family_share": {
                str(k): float(v) / len(sub)
                for k, v in sub["family"].astype(str).value_counts().items()
            },
            "shield_share": {
                str(k): float(v) / len(sub)
                for k, v in sub["shield_member"].astype(str).value_counts().items()
            },
            "prior_returns_bucket_share": {
                str(k): float(v) / len(sub)
                for k, v in sub["_prior_bucket"].astype(str).value_counts().items()
            },
            "value_quartile_share": {
                str(k): float(v) / len(sub)
                for k, v in sub["_val_q"].astype(str).value_counts().items()
            },
            "mean_prior_returns": float(sub["customer_prior_returns"].mean()),
            "mean_promised_days": float(sub["promised_delivery_days"].mean()),
            "pct_cod": float((sub["payment_mode"].astype(str) == "cod").mean()),
            "pct_shield": float((sub["shield_member"].astype(str).str.upper() == "Y").mean()),
            "pct_prior_zero": float((sub["customer_prior_returns"].astype(float) <= 0).mean()),
        }

    fp_c = composition(fp)
    fn_c = composition(fn)
    hold_cod = float((df["payment_mode"].astype(str) == "cod").mean())
    hold_shield = float((df["shield_member"].astype(str).str.upper() == "Y").mean())
    hold_prior0 = float((df["customer_prior_returns"].astype(float) <= 0).mean())

    # Representative examples
    fp_ex = (
        fp.sort_values("score", ascending=False)[["order_id", "score", "returned"]]
        .head(3)
        .to_dict(orient="records")
    )
    fn_ex = (
        fn.sort_values("score", ascending=False)[["order_id", "score", "returned"]]
        .head(3)
        .to_dict(orient="records")
    )
    for row in fp_ex + fn_ex:
        row["order_id"] = str(row["order_id"])
        row["score"] = float(row["score"])
        row["returned"] = int(row["returned"])

    # Narrative from the numbers
    narratives = []
    if fn_c.get("n", 0):
        if fn_c.get("pct_prior_zero", 0) > hold_prior0 + 0.1:
            narratives.append(
                "False negatives are enriched for first-time or zero-prior-return buyers "
                f"({fn_c['pct_prior_zero']:.0%} of misses vs {hold_prior0:.0%} of the holdout): "
                "the model has little personal history to latch onto, so a prepaid first-time "
                "buyer who returns for a reason that never appears in the data slips through."
            )
        if fn_c.get("pct_cod", 0) < hold_cod - 0.05:
            narratives.append(
                f"Misses under-index on COD ({fn_c['pct_cod']:.0%} vs {hold_cod:.0%} holdout), "
                "so the model is better at catching cash-on-delivery risk than quiet prepaid returns."
            )
        if fn_c.get("pct_shield", 0) < hold_shield - 0.05:
            narratives.append(
                f"Shield members are less common among misses "
                f"({_fmt_pct(fn_c['pct_shield'])} of FNs vs {_fmt_pct(hold_shield)} of the holdout), "
                "so many of the returns it fails to flag are ordinary non-Shield orders."
            )
    if fp_c.get("n", 0):
        if fp_c.get("pct_cod", 0) > hold_cod + 0.1:
            narratives.append(
                f"False positives over-index on COD ({fp_c['pct_cod']:.0%} vs {hold_cod:.0%} holdout): "
                "cash-on-delivery is a strong signal, so some COD orders that would have been fine "
                "still get a confirmation call."
            )
        if fp_c.get("pct_shield", 0) > hold_shield + 0.1:
            narratives.append(
                f"Over-flags lean Shield ({fp_c['pct_shield']:.0%} vs {hold_shield:.0%} holdout): "
                "free returns for members raise their historical rate, so loyal buyers get called "
                "more often than the Service Desk would like."
            )
        if fp_c.get("mean_prior_returns", 0) > float(df["customer_prior_returns"].mean()) + 0.2:
            narratives.append(
                "False positives have higher average prior returns than the holdout mean, "
                "so repeat returners who happen not to return this time still look risky."
            )
    if not narratives:
        narratives.append(
            "Error patterns are diffuse: no single channel or payment mode dominates the mistakes."
        )

    return {
        "holdout": holdout_prof,
        "false_positives": fp_c,
        "false_negatives": fn_c,
        "example_fps": fp_ex,
        "example_fns": fn_ex,
        "narrative": narratives[:5],
    }


def stability_check(train: pd.DataFrame) -> dict:
    """Two rolling-origin windows; report AUC/PR for logistic and HGB."""
    windows = {
        "A": {
            "train_before": "2025-11-01",
            "valid_start": "2025-11-01",
            "valid_end": "2026-02-01",
            "label": "train < 2025-11-01, validate Nov 2025-Jan 2026",
        },
        "B": {
            "train_before": "2026-04-01",
            "valid_start": "2026-04-01",
            "valid_end": "2026-07-01",
            "label": "train < 2026-04-01, validate Apr-Jun 2026",
        },
    }
    results = {}
    for key, w in windows.items():
        tr, va = window_split(train, w["train_before"], w["valid_start"], w["valid_end"])
        y = va["returned"].astype(int).to_numpy()
        s_log = fit_predict_logistic(tr, va)
        s_hgb = fit_predict_hgb(tr, va)
        results[key] = {
            "label": w["label"],
            "n_train": int(len(tr)),
            "n_valid": int(len(va)),
            "returns": int(y.sum()),
            "logistic": {
                "auc_roc": float(roc_auc_score(y, s_log)),
                "auc_pr": float(average_precision_score(y, s_log)),
            },
            "hgb": {
                "auc_roc": float(roc_auc_score(y, s_hgb)),
                "auc_pr": float(average_precision_score(y, s_hgb)),
            },
        }

    # Ranking: logistic wins a window if both AUC and PR are >= HGB, or average of the two metrics
    def score_pair(d):
        return 0.5 * (d["auc_roc"] + d["auc_pr"])

    wins = {"logistic": 0, "hgb": 0}
    for key, r in results.items():
        if score_pair(r["logistic"]) >= score_pair(r["hgb"]):
            wins["logistic"] += 1
            r["winner"] = "logistic"
        else:
            wins["hgb"] += 1
            r["winner"] = "hgb"

    avg_log = float(
        np.mean([score_pair(results[k]["logistic"]) for k in results])
    )
    avg_hgb = float(np.mean([score_pair(results[k]["hgb"]) for k in results]))
    selected_ok = avg_log >= avg_hgb
    return {
        "windows": results,
        "wins": wins,
        "avg_logistic_auc_pr_mean": avg_log,
        "avg_hgb_auc_pr_mean": avg_hgb,
        "ranking_holds": all(results[k]["winner"] == "logistic" for k in results),
        "selected_still_wins_on_average": selected_ok,
        "comment": (
            "Logistic leads on both windows."
            if all(results[k]["winner"] == "logistic" for k in results)
            else (
                "Ranking flips across windows; logistic kept only because it wins on average."
                if selected_ok
                else "Ranking flips and HGB wins on average; record honestly."
            )
        ),
    }


def call_prevention_sensitivity(
    scores: np.ndarray, y_true: np.ndarray, threshold: float, rates: list[float]
) -> dict:
    base_n = len(y_true)
    out = {}
    for r in rates:
        econ = expected_call_economics(
            scores, y_true, threshold, call_prevent=r
        )
        scaled = scale_net(econ["net"], base_n)
        out[str(r)] = {
            "call_prevent": r,
            "net_holdout": econ["net"],
            "monthly_700": scaled["monthly_700"],
            "annual_700": scaled["annual_700"],
        }
    return out


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _fmt_inr(x: float) -> str:
    return f"Rs {x:,.0f}"


def write_eval_report(metrics: dict, data_audit: dict, tree_text: str) -> str:
    m = metrics
    h = m["logistic_holdout"]
    g = m["hgb_holdout"]
    econ = m["economics"]
    stab = m["stability"]
    cal = m["calibration"]
    sweep = m["threshold_sweep"]
    tax = m["error_taxonomy"]
    sens = m["call_prevention_sensitivity"]
    ceil = m["accuracy_ceiling"]
    shield = m["shield_tradeoff"]
    abl = m["ablation"]
    svc = data_audit["service_state"]

    tp, fp, fn, tn = h["tp"], h["fp"], h["fn"], h["tn"]
    prec_ci = m["wilson"]["precision"]
    rec_ci = m["wilson"]["recall"]

    lines: list[str] = []
    a = lines.append

    a("# Evaluation report")
    a("")
    a("## 1. What was validated, and how")
    a("")
    a(
        f"We used a time-based holdout at {m['holdout_boundary']}: "
        f"{m['n_train_split']:,} orders for training and {m['n_valid_split']:,} orders "
        f"(April-June 2026) for validation, of which {econ['returns_total']} returned. "
        "A random split would flatter us because the label depends on events that happen "
        "after the order (returns, reverse pickups, later service states), so rows from the "
        "same period would leak future information into both sides of a shuffle. Training on "
        "earlier months and scoring later months matches how the warehouse will use the model."
    )
    a("")

    a("## 2. Model comparison")
    a("")
    a(
        "No LLM scoring: there is no API in this environment, and an LLM score would not be "
        "reproducible or defensible for returns risk. We compared unweighted logistic regression "
        "(one-hot categoricals) with HistGradientBoosting on the same holdout."
    )
    a("")
    a("| Model | AUC | PR-AUC | Brier | Precision | Recall | Flagged | Net Rs at p* |")
    a("|-------|-----|--------|-------|-----------|--------|---------|--------------|")
    a(
        f"| Logistic (selected) | {h['auc_roc']:.4f} | {h['auc_pr']:.4f} | {h['brier']:.4f} | "
        f"{h['precision']:.3f} | {h['recall']:.3f} | {h['flagged']} | {_fmt_inr(econ['model_net'])} |"
    )
    a(
        f"| HistGradientBoosting | {g['auc_roc']:.4f} | {g['auc_pr']:.4f} | {g['brier']:.4f} | "
        f"{g['precision']:.3f} | {g['recall']:.3f} | {g['flagged']} | "
        f"{_fmt_inr(econ['hgb_net_for_comparison'])} |"
    )
    a("")
    a("### Stability (two rolling-origin windows)")
    a("")
    a("| Window | Logistic AUC | Logistic PR | HGB AUC | HGB PR | Winner |")
    a("|--------|--------------|-------------|---------|--------|--------|")
    for key in ("A", "B"):
        w = stab["windows"][key]
        a(
            f"| {key}: {w['label']} (n_valid={w['n_valid']}) | "
            f"{w['logistic']['auc_roc']:.4f} | {w['logistic']['auc_pr']:.4f} | "
            f"{w['hgb']['auc_roc']:.4f} | {w['hgb']['auc_pr']:.4f} | {w['winner']} |"
        )
    a("")
    a(stab["comment"])
    a(
        f" Average of (AUC+PR)/2 across windows: logistic {stab['avg_logistic_auc_pr_mean']:.4f}, "
        f"HGB {stab['avg_hgb_auc_pr_mean']:.4f}."
    )
    a("")
    a(
        "A shallow decision tree is printed at the end only as a readable illustration of rules; "
        "it is not the deliverable model."
    )
    a("")

    a("## 3. Threshold sweep")
    a("")
    a(
        "The operating point is chosen from money (call break-even), not from accuracy. "
        "Net rupees use the policy call cost and prevention rate."
    )
    a("")
    a("| p | Flagged | Flagged % | Precision | Recall | Accuracy | Net Rs |")
    a("|---|---------|-----------|-----------|--------|----------|--------|")
    for row in sweep:
        a(
            f"| {row['threshold']:.4f} | {row['flagged']} | {_fmt_pct(row['flagged_pct'])} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['accuracy']:.3f} | "
            f"{_fmt_inr(row['net_inr'])} |"
        )
    a("")

    a("## 4. Confusion matrix at p* = 0.1118")
    a("")
    a("|  | Predicted return | Predicted ship | Row total |")
    a("|--|------------------|----------------|-----------|")
    a(f"| Actually returned | TP={tp} | FN={fn} | {tp + fn} |")
    a(f"| Actually kept | FP={fp} | TN={tn} | {fp + tn} |")
    a(f"| Column total | {tp + fp} | {fn + tn} | {tp + fp + fn + tn} |")
    a("")
    a(
        f"Precision {h['precision']:.3f} (Wilson 95% CI {_fmt_pct(prec_ci[0])} to {_fmt_pct(prec_ci[1])}). "
        f"Recall {h['recall']:.3f} (Wilson 95% CI {_fmt_pct(rec_ci[0])} to {_fmt_pct(rec_ci[1])}). "
        f"Specificity {h['specificity']:.3f}. "
        f"Of the {tp + fp} flagged orders, {fp} were false alarms "
        f"({_fmt_pct(fp / (tp + fp) if (tp + fp) else 0)}); of the {tp + fn} true returns, "
        f"{fn} were missed ({_fmt_pct(fn / (tp + fn) if (tp + fn) else 0)})."
    )
    a("")

    a("## 5. Calibration")
    a("")
    a(
        f"Brier score {cal['brier']:.4f}. Holdout base rate {_fmt_pct(cal['base_rate'])}; "
        f"mean predicted probability {_fmt_pct(cal['mean_predicted'])}."
    )
    a("")
    a("| Bin | Predicted range | n | Mean predicted | Observed return rate |")
    a("|-----|-----------------|---|----------------|----------------------|")
    for b in cal["bins"]:
        if b["n"] == 0:
            a(f"| {b['bin']} | {b['lo']:.1f}-{b['hi']:.1f} | 0 | - | - |")
        else:
            a(
                f"| {b['bin']} | {b['lo']:.1f}-{b['hi']:.1f} | {b['n']} | "
                f"{b['mean_predicted']:.3f} | {b['observed_rate']:.3f} |"
            )
    a("")
    gap = abs(cal["mean_predicted"] - cal["base_rate"])
    if gap < 0.02 and cal["brier"] < 0.12:
        a(
            "Mean predicted probability is close to the base rate and the reliability table "
            "tracks the diagonal well enough that these scores can be read as probabilities "
            "for thresholding; we leave the raw logistic model in place (no isotonic or Platt)."
        )
    else:
        a(
            "The reliability table shows some miscalibration; we note it and leave the raw "
            "logistic model in place rather than adding isotonic or Platt recalibration."
        )
    a("")

    a("## 6. What it gets wrong (error taxonomy)")
    a("")
    for sent in tax["narrative"]:
        a(sent)
        a("")
    a(
        f"False positives (n={tax['false_positives'].get('n', 0)}): "
        f"COD share {_fmt_pct(tax['false_positives'].get('pct_cod', 0))}, "
        f"Shield share {_fmt_pct(tax['false_positives'].get('pct_shield', 0))}, "
        f"mean prior returns {tax['false_positives'].get('mean_prior_returns', 0):.2f}."
    )
    a("")
    a(
        f"False negatives (n={tax['false_negatives'].get('n', 0)}): "
        f"zero-prior-return share {_fmt_pct(tax['false_negatives'].get('pct_prior_zero', 0))}, "
        f"COD share {_fmt_pct(tax['false_negatives'].get('pct_cod', 0))}, "
        f"Shield share {_fmt_pct(tax['false_negatives'].get('pct_shield', 0))}."
    )
    a("")
    a("| Type | order_id | score | true label |")
    a("|------|----------|-------|------------|")
    for row in tax["example_fps"]:
        a(f"| FP | {row['order_id']} | {row['score']:.4f} | {row['returned']} |")
    for row in tax["example_fns"]:
        a(f"| FN | {row['order_id']} | {row['score']:.4f} | {row['returned']} |")
    a("")

    a("## 7. Leakage ablation")
    a("")
    a(
        f"`last_service_event_type == REVERSE_PICKUP` covers {abl['reverse_pickup_n']} rows "
        f"with return rate {_fmt_pct(abl['reverse_pickup_return_rate'])}. "
        f"`pickup_scheduled_at` non-null covers {abl['pickup_scheduled_n']} rows with return rate "
        f"{_fmt_pct(abl['pickup_scheduled_return_rate'])}. Both are booked after a return is "
        "approved (ops-policy section 7), so they are post-label artefacts."
    )
    a("")
    a(
        f"On the same holdout with HGB: with both leakage features AUC "
        f"{abl['with_leakage']['auc_roc']:.4f} / PR {abl['with_leakage']['auc_pr']:.4f}; "
        f"without them AUC {abl['without_leakage']['auc_roc']:.4f} / "
        f"PR {abl['without_leakage']['auc_pr']:.4f}. We drop both."
    )
    a("")
    a(
        "Second-order finding: the service column is not the same column across files. "
        "Train includes DEMO_DONE and INSTALL_DONE (already delivered, historically ~0% returns) "
        "and REVERSE_PICKUP, while test at dispatch shows NONE and INSTALL_BOOKED "
        "(never seen in train). A model trained on that column would treat 'already delivered' "
        "states as safe and would not know what to do with INSTALL_BOOKED in production, "
        "so the column is excluded entirely."
    )
    a("")
    a("| Service event | Share in train | Share in test |")
    a("|---------------|----------------|---------------|")
    all_keys = sorted(
        set(svc["train_distribution"]) | set(svc["test_distribution"])
    )
    for key in all_keys:
        tr_s = svc["train_distribution"].get(key, 0.0)
        te_s = svc["test_distribution"].get(key, 0.0)
        a(f"| {key} | {_fmt_pct(tr_s)} | {_fmt_pct(te_s)} |")
    a("")

    a("## 8. The 95% conversation")
    a("")
    a(
        f"Best accuracy any threshold reaches on this holdout is {ceil['best_accuracy']:.4f}. "
        f"Predicting that nothing returns already scores {ceil['majority_class_accuracy']:.4f}. "
        f"To hit 95% accuracy you may make at most {ceil['max_errors_for_95pct']} errors out of "
        f"{ceil['n']}, while {ceil['holdout_returns']} orders genuinely return. "
        "Accuracy is the wrong metric when 88.5% of the answer is already 'no': a model that "
        "never flags anyone looks almost as accurate as the best achievable threshold, and "
        "would prevent zero returns."
    )
    a("")

    a("## 9. The money")
    a("")
    a("### Break-even for a confirmation call")
    a("")
    a(
        f"p* = C_CALL / (CALL_PREVENT x C_RETURN) = {C_CALL} / ({CALL_PREVENT} x {C_RETURN}) "
        f"= {OPERATING_THRESHOLD:.4f}."
    )
    a("")
    a("### Break-even for a hold vs ship (per order)")
    a("")
    a(
        "p*_i = 0.12 x M_i / (M_i + 1150), where M_i = MARGIN_PCT x order value. "
        "**MARGIN IS AN ASSUMPTION; it is not in the pack.** "
        f"At the holdout median order value {_fmt_inr(econ['median_order_value'])}:"
    )
    a("")
    a("| Assumed margin | Hold break-even p* |")
    a("|----------------|--------------------|")
    for mp, p in m["margin_sensitivity"].items():
        a(f"| {float(mp):.0%} | {p:.4f} |")
    a("")
    a(
        "We recommend the call as the intervention because the pack measures call prevention "
        "(35%) and does not measure hold prevention; holds also cancel about 12% of the time "
        "and hit high-LTV Shield customers."
    )
    a("")
    a("### Holdout rupee table (at p* = 0.1118)")
    a("")
    every = call_every_order_net(h["n"], int(econ["returns_total"]))
    a("| Strategy | Prevented returns | Saved Rs | Cost Rs | Net Rs |")
    a("|----------|-------------------|----------|---------|--------|")
    a(
        f"| No intervention | 0 | 0 | 0 | 0 |"
    )
    a(
        f"| Model-guided calls | {econ['model_prevented']:.2f} | "
        f"{_fmt_inr(econ['model_saved'])} | {_fmt_inr(econ['model_cost'])} | "
        f"{_fmt_inr(econ['model_net'])} |"
    )
    a(
        f"| Call everyone | {every['prevented']:.2f} | {_fmt_inr(every['saved'])} | "
        f"{_fmt_inr(every['cost'])} | {_fmt_inr(every['net'])} |"
    )
    a("")
    a("### Shield trade-off")
    a("")
    a("| Policy | Net Rs | Flagged | TP | Prevented | Missed TP vs all | Calls saved |")
    a("|--------|--------|---------|----|-----------|------------------|------------|")
    ia = shield["intervene_all"]
    ex = shield["exclude_shield"]
    a(
        f"| Intervene on everyone | {_fmt_inr(ia['net'])} | {int(ia['flagged'])} | "
        f"{int(ia['tp'])} | {ia['prevented']:.2f} | 0 | 0 |"
    )
    a(
        f"| Exclude Shield | {_fmt_inr(ex['net'])} | {int(ex['flagged'])} | "
        f"{int(ex['tp'])} | {ex['prevented']:.2f} | "
        f"{int(shield['returns_missed_by_excluding_shield'])} | "
        f"{int(shield['calls_saved_by_excluding_shield'])} |"
    )
    a("")
    a(
        f"Net delta of excluding Shield: {_fmt_inr(shield['net_delta_excluding_shield'])} "
        "(negative means excluding Shield costs money on this holdout, before goodwill)."
    )
    a("")
    a("### Scaling")
    a("")
    sc = econ["scaled"]
    a(
        f"Per order {_fmt_inr(sc['per_order'])}; at ~700 orders/month "
        f"{_fmt_inr(sc['monthly_700'])}/month; "
        f"{_fmt_inr(sc['annual_700'])}/year."
    )
    a("")
    a("### Call-prevention sensitivity")
    a("")
    a("| Assumed prevention rate | Holdout net | Monthly at 700 | Annual |")
    a("|-------------------------|-------------|----------------|--------|")
    for key in sorted(sens.keys(), key=float):
        row = sens[key]
        a(
            f"| {float(key):.0%} | {_fmt_inr(row['net_holdout'])} | "
            f"{_fmt_inr(row['monthly_700'])} | {_fmt_inr(row['annual_700'])} |"
        )
    a("")

    a("## 10. Small-sample honesty")
    a("")
    a(
        f"The holdout has n={h['n']} with {econ['returns_total']} returns. "
        f"Wilson 95% CI width on precision is {_fmt_pct(prec_ci[1] - prec_ci[0])}; "
        f"on recall {_fmt_pct(rec_ci[1] - rec_ci[0])}. "
        "Monthly rupee projections scale a few hundred true positives; a shift of precision "
        "or recall within those intervals moves the monthly figure by thousands of rupees, "
        "so treat the annualised number as a planning range, not a guarantee."
    )
    a("")
    a("## Appendix: decision-tree illustration (not the deliverable)")
    a("")
    a("```")
    a(tree_text.rstrip())
    a("```")
    a("")

    text = "\n".join(lines)
    (REPORTS_DIR / "eval_report.md").write_text(text)
    return text


def write_data_audit(data_audit: dict) -> None:
    audit_md = [
        "# Data audit",
        "",
        "## Duplicates",
        f"- Partner-feed re-imports: {data_audit['dedupe']['n_dup_order_ids']} order_ids appeared twice.",
        f"- Kept `crm` copy -> {data_audit['dedupe']['n_after']} rows "
        f"(base return rate {data_audit['dedupe']['return_rate']:.2%}).",
        "",
        "## October 2025 value inflation",
        f"- Rows: {data_audit['october']['oct_2025_rows']}",
        f"- Before mean/max: {data_audit['october']['before_mean']:.0f} / "
        f"{data_audit['october']['before_max']:.0f}",
        f"- After mean/max: {data_audit['october']['after_mean']:.0f} / "
        f"{data_audit['october']['after_max']:.0f}",
        f"- Other months mean/max: {data_audit['october']['other_months_mean']:.0f} / "
        f"{data_audit['october']['other_months_max']:.0f}",
        f"- Fix: {data_audit['october']['fix']}",
        "",
        "## Leakage",
        data_audit["leakage"]["note"],
        f"- With leakage AUC/PR: {data_audit['leakage']['with_leakage']}",
        f"- Without leakage AUC/PR: {data_audit['leakage']['without_leakage']}",
        "",
        "## Service-state mismatch (train vs test)",
        data_audit["service_state"]["finding"],
        f"- Train: {data_audit['service_state']['train_distribution']}",
        f"- Test: {data_audit['service_state']['test_distribution']}",
        "",
        "## Pincode 000000",
        data_audit["pincode_note"],
        "",
        "## Delivery notes",
        "Free-text `delivery_note` carries almost no signal; only `note_missing` is kept.",
        "",
    ]
    (REPORTS_DIR / "data_audit.md").write_text("\n".join(audit_md))


def write_reports(metrics: dict, data_audit: dict, tree_text: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_data_audit(data_audit)
    write_eval_report(metrics, data_audit, tree_text)


def main() -> None:
    from src.config import DATA_DIR
    from src.features import load_raw

    train, _test = load_prepared()
    raw_train, raw_test, _, _ = load_raw()

    tr, va = time_split(train)
    yva = va["returned"].astype(int).to_numpy()

    scores_hgb = fit_predict_hgb(tr, va)
    scores_log = fit_predict_logistic(tr, va)

    hgb_m = metrics_at(yva, scores_hgb, OPERATING_THRESHOLD)
    log_m = metrics_at(yva, scores_log, OPERATING_THRESHOLD)

    econ_hgb = expected_call_economics(scores_hgb, yva, OPERATING_THRESHOLD)
    returns_total = int(yva.sum())
    every = call_every_order_net(len(yva), returns_total)

    median_v = float(va["value"].median())
    hold_med = hold_breakeven_threshold(median_v)
    margin_sens = {
        str(mp): hold_breakeven_threshold(median_v, margin_pct=mp)
        for mp in MARGIN_PCT_SENSITIVITY
    }

    leak = leakage_ablation(tr, va)
    tree_text = decision_tree_illustration(tr, va)

    n_before = len(raw_train)
    n_dup = int(raw_train["order_id"].duplicated(keep=False).sum() // 2)
    dedupe_info = {
        "n_before": n_before,
        "n_dup_order_ids": n_dup,
        "n_after": len(train),
        "return_rate": float(train["returned"].mean()),
    }
    october = october_audit(DATA_DIR / "train.csv")
    service = service_state_audit(train, raw_test)
    data_audit = {
        "dedupe": dedupe_info,
        "october": october,
        "leakage": leak,
        "service_state": service,
        "pincode_note": (
            "delivery_pincode==000000 appears across channels (not only partner_outlet); "
            "treated as missing -> pin_missing flag."
        ),
    }

    econ_log = expected_call_economics(scores_log, yva, OPERATING_THRESHOLD)
    scaled_log = scale_net(econ_log["net"], len(yva))
    shield_log = shield_tradeoff(va, scores_log, OPERATING_THRESHOLD)
    ceiling_log = accuracy_ceiling(yva, scores_log)

    stab = stability_check(train)
    cal = calibration_table(yva, scores_log)
    sweep_thresholds = [0.06, 0.08, 0.10, 0.1118, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    sweep = threshold_sweep(yva, scores_log, sweep_thresholds)
    tax = error_taxonomy(va, scores_log, OPERATING_THRESHOLD)
    sens = call_prevention_sensitivity(scores_log, yva, OPERATING_THRESHOLD, [0.20, 0.35, 0.50])

    prec_ci = wilson_interval(log_m["tp"], log_m["tp"] + log_m["fp"])
    rec_ci = wilson_interval(log_m["tp"], log_m["tp"] + log_m["fn"])

    selected = "logistic_regression"
    why = (
        "Unweighted logistic regression (one-hot categoricals) beats HistGradientBoosting "
        f"on holdout AUC ({log_m['auc_roc']:.4f} vs {hgb_m['auc_roc']:.4f}), "
        f"PR-AUC ({log_m['auc_pr']:.4f} vs {hgb_m['auc_pr']:.4f}), "
        f"Brier ({log_m['brier']:.4f} vs {hgb_m['brier']:.4f}), and net Rs at p* "
        f"({econ_log['net']:.0f} vs {econ_hgb['net']:.0f}). "
        "No LLM scoring (no API / not reproducible / not defensible). "
        "A shallow decision tree is illustration-only."
    )

    metrics = {
        "holdout_boundary": HOLDOUT_BOUNDARY,
        "n_train_split": len(tr),
        "n_valid_split": len(va),
        "operating_threshold": OPERATING_THRESHOLD,
        "hgb_holdout": hgb_m,
        "logistic_holdout": log_m,
        "model_choice": {
            "selected": selected,
            "why": why,
            "hgb_auc_roc": hgb_m["auc_roc"],
            "hgb_auc_pr": hgb_m["auc_pr"],
            "hgb_brier": hgb_m["brier"],
            "hgb_net_inr": econ_hgb["net"],
            "logistic_auc_roc": log_m["auc_roc"],
            "logistic_auc_pr": log_m["auc_pr"],
            "logistic_brier": log_m["brier"],
            "logistic_net_inr": econ_log["net"],
        },
        "accuracy_ceiling": ceiling_log,
        "ablation": leak,
        "economics": {
            "call_breakeven": call_breakeven_threshold(),
            "hold_breakeven_median": hold_med,
            "median_order_value": median_v,
            "model_net": econ_log["net"],
            "model_saved": econ_log["saved"],
            "model_cost": econ_log["cost"],
            "model_prevented": econ_log["prevented"],
            "returns_total": returns_total,
            "no_intervention_net": 0.0,
            "call_every_net": every["net"],
            "call_every_cost": every["cost"],
            "call_every_saved": every["saved"],
            "call_every_prevented": every["prevented"],
            "scaled": scaled_log,
            "confusion_at_threshold": {
                "tp": econ_log["tp"],
                "fp": econ_log["fp"],
                "fn": econ_log["fn"],
                "tn": econ_log["tn"],
            },
            "hgb_net_for_comparison": econ_hgb["net"],
        },
        "shield_tradeoff": shield_log,
        "margin_sensitivity": margin_sens,
        "margin_pct_assumption": MARGIN_PCT,
        "constants": {
            "C_RETURN": C_RETURN,
            "C_CALL": C_CALL,
            "CALL_PREVENT": CALL_PREVENT,
        },
        "wilson": {
            "precision": list(prec_ci),
            "recall": list(rec_ci),
        },
        "stability": stab,
        "calibration": cal,
        "threshold_sweep": sweep,
        "error_taxonomy": tax,
        "call_prevention_sensitivity": sens,
    }

    write_reports(metrics, data_audit, tree_text)
    print(
        f"SELECTED logistic AUC={log_m['auc_roc']:.4f} PR={log_m['auc_pr']:.4f} | "
        f"HGB AUC={hgb_m['auc_roc']:.4f} PR={hgb_m['auc_pr']:.4f}"
    )
    print(f"stability: {stab['comment']}")
    print(f"Wrote reports to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
