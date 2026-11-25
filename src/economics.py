"""Cost model for pre-dispatch interventions (ops-policy §4 / §7)."""

from __future__ import annotations

from typing import Iterable

from src.config import (
    CALL_PREVENT,
    CANCEL_ON_HOLD,
    C_CALL,
    C_RETURN,
    MARGIN_PCT,
)


def call_breakeven_threshold(
    c_call: float = C_CALL,
    call_prevent: float = CALL_PREVENT,
    c_return: float = C_RETURN,
) -> float:
    """p* such that expected savings from a call equals call cost."""
    return c_call / (call_prevent * c_return)


def hold_breakeven_threshold(
    order_value: float,
    margin_pct: float = MARGIN_PCT,
    cancel_on_hold: float = CANCEL_ON_HOLD,
    c_return: float = C_RETURN,
) -> float:
    """
    p*_i = CANCEL_ON_HOLD * M_i / (M_i + C_RETURN)
    where M_i = margin_pct * order_value.
    """
    margin = margin_pct * float(order_value)
    denom = margin + c_return
    if denom <= 0:
        return 1.0
    return cancel_on_hold * margin / denom


def expected_call_economics(
    scores: Iterable[float],
    labels: Iterable[int] | None,
    threshold: float,
    *,
    c_call: float = C_CALL,
    call_prevent: float = CALL_PREVENT,
    c_return: float = C_RETURN,
) -> dict[str, float]:
    """
    At a call threshold: flag if score > threshold.
    prevented returns ≈ TP * call_prevent
    saved = prevented * c_return
    cost = (TP+FP) * c_call
    net = saved - cost
    """
    scores_l = list(scores)
    n = len(scores_l)
    if labels is None:
        flagged = sum(1 for s in scores_l if s > threshold)
        return {
            "n": float(n),
            "flagged": float(flagged),
            "tp": float("nan"),
            "fp": float("nan"),
            "fn": float("nan"),
            "tn": float("nan"),
            "prevented": float("nan"),
            "saved": float("nan"),
            "cost": flagged * c_call,
            "net": float("nan"),
        }

    labels_l = list(labels)
    tp = fp = fn = tn = 0
    for s, y in zip(scores_l, labels_l):
        pred = s > threshold
        if pred and y == 1:
            tp += 1
        elif pred and y == 0:
            fp += 1
        elif (not pred) and y == 1:
            fn += 1
        else:
            tn += 1

    prevented = tp * call_prevent
    saved = prevented * c_return
    cost = (tp + fp) * c_call
    return {
        "n": float(n),
        "flagged": float(tp + fp),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "prevented": float(prevented),
        "saved": float(saved),
        "cost": float(cost),
        "net": float(saved - cost),
        "returns_total": float(tp + fn),
    }


def call_every_order_net(
    n_orders: int,
    n_returns: int,
    *,
    c_call: float = C_CALL,
    call_prevent: float = CALL_PREVENT,
    c_return: float = C_RETURN,
) -> dict[str, float]:
    cost = n_orders * c_call
    prevented = n_returns * call_prevent
    saved = prevented * c_return
    return {
        "cost": float(cost),
        "prevented": float(prevented),
        "saved": float(saved),
        "net": float(saved - cost),
    }


def scale_net(net: float, holdout_n: int, monthly_orders: float = 700.0) -> dict[str, float]:
    per_order = net / holdout_n if holdout_n else 0.0
    monthly = per_order * monthly_orders
    return {
        "per_order": per_order,
        "monthly_700": monthly,
        "annual_700": monthly * 12.0,
    }


def per_order_call_expectation(
    score: float,
    order_value: float,
    *,
    threshold: float | None = None,
    c_call: float = C_CALL,
    call_prevent: float = CALL_PREVENT,
    c_return: float = C_RETURN,
    margin_pct: float = MARGIN_PCT,
) -> dict[str, float]:
    """Expected rupees if we call this order (used by /score)."""
    thr = call_breakeven_threshold() if threshold is None else threshold
    expected_saving = float(score) * call_prevent * c_return
    expected_cost = c_call
    return {
        "threshold": thr,
        "expected_saving_inr": round(expected_saving, 2),
        "expected_call_cost_inr": round(expected_cost, 2),
        "hold_breakeven": round(hold_breakeven_threshold(order_value, margin_pct), 4),
        "net_if_called": round(expected_saving - expected_cost, 2),
    }
