"""Train the Logistic Regression baseline and compare it with naive alternatives.

Usage:
    python scripts/train_baseline.py
"""

import json
from dataclasses import asdict

import joblib
import numpy as np

from riskwatch.config import LABEL, PROJECT_ROOT
from riskwatch.model import (
    build_logistic_regression,
    evaluate,
    load_dataset,
    time_split,
    train_and_evaluate,
)

REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_metrics.json"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "logreg_balanced.joblib"


def main() -> None:
    df = load_dataset()
    train, test = time_split(df)
    print(f"Train: {len(train):,} rows ({train[LABEL].sum()} fraud) | "
          f"Test: {len(test):,} rows ({test[LABEL].sum()} fraud)")

    results = {
        # Predicts "not fraud" for everything: shows why accuracy is misleading.
        "always_legit": asdict(evaluate(test[LABEL], np.zeros(len(test)))),
        "logreg_unweighted": train_and_evaluate(build_logistic_regression(None), train, test),
    }
    balanced = build_logistic_regression("balanced")
    results["logreg_balanced"] = train_and_evaluate(balanced, train, test)

    print(f"\n{'model':<20}{'PR-AUC':>8}{'recall':>8}{'precision':>11}{'FPR':>8}{'accuracy':>10}")
    for name, m in results.items():
        print(f"{name:<20}{m['pr_auc']:>8.3f}{m['recall']:>8.1%}{m['precision']:>11.1%}"
              f"{m['false_positive_rate']:>8.2%}{m['accuracy']:>10.2%}")

    m = results["logreg_balanced"]
    print(f"\nBalanced LR confusion matrix (threshold 0.5):"
          f"\n  caught fraud: {m['true_positives']}  missed fraud: {m['false_negatives']}"
          f"\n  false alarms: {m['false_positives']}  correct legit: {m['true_negatives']:,}")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(balanced, MODEL_PATH)
    print(f"\nSaved metrics to reports/{REPORT_PATH.name} and model to artifacts/{MODEL_PATH.name}")


if __name__ == "__main__":
    main()
