import numpy as np
import pandas as pd
import pytest

from riskwatch.config import EXPECTED_COLUMNS
from riskwatch.model import (
    FEATURES,
    build_logistic_regression,
    evaluate,
    prepare,
    time_split,
    train_and_evaluate,
)


@pytest.fixture
def synthetic_df():
    # Fraud rows are shifted on V1 so a linear model can separate them.
    rng = np.random.default_rng(0)
    n = 400
    labels = (np.arange(n) % 20 == 0).astype(int)
    data = rng.normal(size=(n, 28))
    data[:, 0] += labels * 4
    df = pd.DataFrame(data, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.arange(n, dtype=float))
    df["Amount"] = rng.uniform(0, 200, n)
    df["Class"] = labels
    df.insert(0, "transaction_id", range(1, n + 1))
    return df[["transaction_id", *EXPECTED_COLUMNS]]


def test_prepare_drops_duplicates_and_adds_log_amount(synthetic_df):
    dup = synthetic_df.iloc[[0]].assign(transaction_id=999)
    out = prepare(pd.concat([synthetic_df, dup]))
    assert len(out) == len(synthetic_df)
    assert out["log_amount"].iloc[0] == pytest.approx(np.log1p(out["Amount"].iloc[0]))


def test_time_split_puts_later_rows_in_test(synthetic_df):
    train, test = time_split(prepare(synthetic_df), test_size=0.25)
    assert len(train) == 300 and len(test) == 100
    assert train["Time"].max() < test["Time"].min()


def test_evaluate_known_confusion_matrix():
    y = [0, 0, 0, 1, 1]
    scores = [0.1, 0.6, 0.2, 0.9, 0.3]  # 1 TP, 1 FP, 1 FN, 2 TN
    m = evaluate(y, scores, threshold=0.5)
    assert (m.true_positives, m.false_positives, m.false_negatives, m.true_negatives) == (1, 1, 1, 2)
    assert m.precision == 0.5
    assert m.recall == 0.5
    assert m.false_positive_rate == pytest.approx(1 / 3)


def test_always_legit_has_high_accuracy_but_zero_recall():
    y = [0] * 99 + [1]
    m = evaluate(y, np.zeros(100))
    assert m.accuracy == 0.99
    assert m.recall == 0.0


def test_balanced_logreg_learns_synthetic_signal(synthetic_df):
    train, test = time_split(prepare(synthetic_df))
    metrics = train_and_evaluate(build_logistic_regression("balanced"), train, test)
    assert metrics["recall"] >= 0.75
    assert metrics["pr_auc"] > 0.5
    assert set(FEATURES) <= set(train.columns)
