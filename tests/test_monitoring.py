import numpy as np
import pandas as pd

from riskwatch.model import build_logistic_regression
from riskwatch.monitoring import benchmark_latency, feature_drift, psi, psi_label


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=20_000), rng.normal(size=20_000)) < 0.01


def test_psi_flags_a_shifted_distribution():
    rng = np.random.default_rng(0)
    shifted = psi(rng.normal(size=20_000), rng.normal(loc=1.0, size=20_000))
    assert shifted > 0.25
    assert psi_label(shifted) == "significant shift"


def test_psi_counts_values_outside_baseline_range():
    baseline = np.linspace(0, 1, 1000)
    # Every new value is above the baseline max, so all land in the top bin.
    assert psi(baseline, np.full(1000, 5.0)) > 1


def test_feature_drift_sorts_worst_first():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.normal(size=5000)})
    curr = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.normal(loc=2, size=5000)})
    out = feature_drift(base, curr, ["a", "b"])
    assert out["feature"].tolist() == ["b", "a"]
    assert out["status"].tolist() == ["significant shift", "stable"]


def test_benchmark_latency_reports_batch_size():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["x", "y", "z"])
    model = build_logistic_regression().fit(X, (X["x"] > 0).astype(int))
    result = benchmark_latency(model, X, batch_size=100, repeats=3)
    assert result["batch_size"] == 100
    assert 0 < result["median_ms"] <= result["p95_ms"]
