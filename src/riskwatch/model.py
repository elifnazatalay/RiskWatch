"""Train and evaluate fraud-detection models."""

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from riskwatch.config import DB_PATH, LABEL, PCA_FEATURES, TABLE_NAME

FEATURES = [*PCA_FEATURES, "log_amount"]


def load_dataset(db_path: Path = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    return prepare(df)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate transactions and add model features."""
    feature_cols = [c for c in df.columns if c != "transaction_id"]
    # Duplicates could land on both sides of the split and inflate test scores.
    df = df.drop_duplicates(subset=feature_cols).copy()
    # Amount is heavily right-skewed ($0 to $25k); log1p keeps large values from dominating.
    df["log_amount"] = np.log1p(df["Amount"])
    return df


def time_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier transactions and test on later ones.

    A random split would let the model see the "future", which overstates how it
    would perform in production, where it always scores new transactions.
    """
    df = df.sort_values("Time", kind="stable")
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut], df.iloc[cut:]


def build_logistic_regression(class_weight: str | None = "balanced") -> Pipeline:
    # class_weight="balanced" weights each fraud ~580x a legit transaction, so the
    # model can't score well by predicting "not fraud" for everything.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight=class_weight, max_iter=1000),
    )


def build_random_forest(n_estimators: int = 200, random_state: int = 42) -> RandomForestClassifier:
    # Trees split on thresholds, so no scaling is needed. "balanced_subsample"
    # reweights classes inside each tree's bootstrap sample. min_samples_leaf
    # stops trees from memorizing individual fraud rows.
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )


@dataclass
class Metrics:
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    false_positive_rate: float
    accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


def evaluate(y_true, scores, threshold: float = 0.5) -> Metrics:
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return Metrics(
        pr_auc=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        precision=float(tp / (tp + fp)) if tp + fp else 0.0,
        recall=float(tp / (tp + fn)) if tp + fn else 0.0,
        false_positive_rate=float(fp / (fp + tn)) if fp + tn else 0.0,
        accuracy=float((tp + tn) / len(y_pred)),
        true_positives=int(tp),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_negatives=int(tn),
    )


def train_and_evaluate(model: Pipeline, train: pd.DataFrame, test: pd.DataFrame,
                       threshold: float = 0.5) -> dict:
    model.fit(train[FEATURES], train[LABEL])
    scores = model.predict_proba(test[FEATURES])[:, 1]
    return asdict(evaluate(test[LABEL], scores, threshold))
