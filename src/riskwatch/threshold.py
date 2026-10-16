"""Decision-threshold analysis: trading fraud caught against false alarms.

A model outputs a fraud score between 0 and 1. The threshold decides which
scores become alerts, and choosing it is a business decision: every alert
costs an analyst review or a declined customer card.
"""

import numpy as np
import pandas as pd


def threshold_table(y_true, scores, amounts, thresholds=None) -> pd.DataFrame:
    """Recall, precision, false-positive rate, and fraud dollars caught per threshold."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    amt = np.asarray(amounts, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)

    total_fraud = y.sum()
    total_legit = len(y) - total_fraud
    total_fraud_dollars = amt[y == 1].sum()

    rows = []
    for t in thresholds:
        flagged = s >= t
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        rows.append({
            "threshold": float(t),
            "alerts": tp + fp,
            "fraud_caught": tp,
            "false_alarms": fp,
            "recall": tp / total_fraud if total_fraud else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "false_positive_rate": fp / total_legit if total_legit else 0.0,
            "fraud_dollars_caught_pct": (
                amt[flagged & (y == 1)].sum() / total_fraud_dollars if total_fraud_dollars else 0.0
            ),
        })
    return pd.DataFrame(rows)


def threshold_for_fpr(y_true, scores, max_fpr: float) -> float:
    """Lowest threshold (so highest recall) whose false-positive rate stays within max_fpr.

    Lowering the threshold only adds alerts, so FPR never decreases as the threshold
    drops. The answer is therefore set by the legit transactions' score distribution:
    we can allow at most floor(max_fpr * n_legit) legit scores at or above it.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores)
    legit = np.sort(s[y == 0])[::-1]  # highest legit scores first
    allowed = int(np.floor(max_fpr * len(legit)))
    if allowed >= len(legit):
        return 0.0
    # Any threshold just above the (allowed+1)-th highest legit score keeps
    # exactly `allowed` false alarms.
    return float(np.nextafter(legit[allowed], np.inf))
