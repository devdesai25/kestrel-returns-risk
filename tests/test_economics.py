"""Unit tests for the cost model."""

from src.config import CALL_PREVENT, C_CALL, C_RETURN, OPERATING_THRESHOLD
from src.economics import (
    call_breakeven_threshold,
    call_every_order_net,
    expected_call_economics,
    hold_breakeven_threshold,
)


def test_call_breakeven_matches_policy_arithmetic():
    p_star = call_breakeven_threshold()
    assert abs(p_star - (45 / (0.35 * 1150))) < 1e-9
    assert abs(p_star - OPERATING_THRESHOLD) < 1e-12
    assert abs(p_star - 0.111801242236) < 1e-6


def test_hold_breakeven_medianish_order():
    # ≈ 0.059 for a median ₹4,742 order at 25% margin
    p = hold_breakeven_threshold(4742.0, margin_pct=0.25)
    margin = 0.25 * 4742
    expected = 0.12 * margin / (margin + 1150)
    assert abs(p - expected) < 1e-9
    assert 0.05 < p < 0.07


def test_expected_call_economics_counts():
    scores = [0.2, 0.05, 0.3, 0.01]
    labels = [1, 0, 0, 1]
    out = expected_call_economics(scores, labels, threshold=0.1118)
    assert out["tp"] == 1  # only first
    assert out["fp"] == 1  # third
    assert out["fn"] == 1  # fourth
    assert out["tn"] == 1
    assert out["flagged"] == 2
    assert abs(out["prevented"] - 1 * CALL_PREVENT) < 1e-9
    assert abs(out["cost"] - 2 * C_CALL) < 1e-9
    assert abs(out["saved"] - CALL_PREVENT * C_RETURN) < 1e-9


def test_call_every_order_net():
    out = call_every_order_net(2126, 245)
    assert out["cost"] == 2126 * 45
    assert abs(out["prevented"] - 245 * 0.35) < 1e-9
