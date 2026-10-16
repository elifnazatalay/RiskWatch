"""Model monitoring: distribution drift (PSI) and inference latency."""

import time

import numpy as np
import pandas as pd


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between a baseline and a new sample.

    Bins come from the baseline's quantiles, so each bin holds ~10% of the
    baseline. PSI sums how far each bin's share moved. Common rule of thumb
    in credit-risk model monitoring:
        < 0.10  stable
        0.10-0.25  moderate shift, investigate
        > 0.25  significant shift, consider retraining
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    # Open-ended outer bins so new values outside the baseline range still count.
    edges[0], edges[-1] = -np.inf, np.inf
    e_share = np.histogram(expected, edges)[0] / len(expected)
    a_share = np.histogram(actual, edges)[0] / len(actual)
    # Avoid log(0) for empty bins.
    e_share = np.clip(e_share, 1e-6, None)
    a_share = np.clip(a_share, 1e-6, None)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def psi_label(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate shift"
    return "significant shift"


def feature_drift(baseline: pd.DataFrame, current: pd.DataFrame, columns) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(baseline[c], current[c])} for c in columns]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
    out["status"] = out["psi"].map(psi_label)
    return out


def benchmark_latency(model, X: pd.DataFrame, batch_size: int = 1000, repeats: int = 20) -> dict:
    """Median and p95 milliseconds to score one batch of transactions."""
    batch = X.iloc[:batch_size]
    model.predict_proba(batch)  # warm-up run, not timed
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_proba(batch)
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "batch_size": len(batch),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
    }
