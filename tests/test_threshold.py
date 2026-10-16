import numpy as np
import pytest

from riskwatch.model import build_random_forest
from riskwatch.threshold import threshold_for_fpr, threshold_table

# 6 legit (label 0) and 2 fraud (label 1) transactions.
Y = [0, 0, 0, 0, 0, 0, 1, 1]
SCORES = [0.1, 0.2, 0.3, 0.4, 0.6, 0.9, 0.7, 0.95]
AMOUNTS = [10, 10, 10, 10, 10, 10, 100, 300]


def test_threshold_table_counts():
    table = threshold_table(Y, SCORES, AMOUNTS, thresholds=[0.5, 0.8])
    at_05, at_08 = table.iloc[0], table.iloc[1]
    # >= 0.5 flags 0.6, 0.9 (legit) and 0.7, 0.95 (fraud)
    assert (at_05["fraud_caught"], at_05["false_alarms"]) == (2, 2)
    assert at_05["fraud_dollars_caught_pct"] == 1.0
    # >= 0.8 flags 0.9 (legit) and 0.95 (the $300 fraud)
    assert (at_08["fraud_caught"], at_08["false_alarms"]) == (1, 1)
    assert at_08["recall"] == 0.5
    assert at_08["fraud_dollars_caught_pct"] == pytest.approx(0.75)


def test_higher_threshold_never_adds_alerts():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    scores = rng.random(200)
    alerts = threshold_table(y, scores, np.ones(200))["alerts"]
    assert alerts.is_monotonic_decreasing


@pytest.mark.parametrize("max_fpr, expected_false_alarms", [(0.0, 0), (1 / 6, 1), (2 / 6, 2)])
def test_threshold_for_fpr_respects_budget(max_fpr, expected_false_alarms):
    t = threshold_for_fpr(Y, SCORES, max_fpr)
    fp = sum(s >= t and y == 0 for s, y in zip(SCORES, Y))
    assert fp == expected_false_alarms


def test_threshold_for_fpr_is_as_low_as_possible():
    # Budget of 1 false alarm: threshold sits just above 0.6, so the 0.7 fraud is still caught.
    t = threshold_for_fpr(Y, SCORES, 1 / 6)
    assert 0.6 < t <= 0.7


def test_random_forest_is_reproducible():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 1).astype(int)
    a = build_random_forest(n_estimators=10).fit(X, y).predict_proba(X)
    b = build_random_forest(n_estimators=10).fit(X, y).predict_proba(X)
    # allclose, not exact equality: parallel trees can sum probabilities in a different order.
    assert np.allclose(a, b)
