"""Compare Logistic Regression and Random Forest at a fixed false-alarm budget.

The threshold is chosen on a validation period, then applied unchanged to the
later test period. Picking it on the test set would leak test information and
overstate results.

Usage:
    python scripts/compare_models.py
"""

import json
import time
from dataclasses import asdict

import joblib
import pandas as pd

from riskwatch.config import LABEL, PROJECT_ROOT
from riskwatch.model import (
    FEATURES,
    build_logistic_regression,
    build_random_forest,
    evaluate,
    load_dataset,
    time_split,
)
from riskwatch.threshold import threshold_for_fpr, threshold_table

# Alert budget: at most 0.1% of legitimate transactions may be flagged
# (about 1 false alarm per 1,000 good customers).
MAX_FPR = 0.001

REPORTS = PROJECT_ROOT / "reports"
ARTIFACTS = PROJECT_ROOT / "artifacts"


def main() -> None:
    df = load_dataset()
    train_full, test = time_split(df, test_size=0.2)
    train, val = time_split(train_full, test_size=0.2)
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        print(f"{name:<11}{len(part):>8,} rows  {part[LABEL].sum():>4} fraud")

    models = {
        "logistic_regression": build_logistic_regression("balanced"),
        "random_forest": build_random_forest(),
    }
    results, curves = {}, []
    REPORTS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(train[FEATURES], train[LABEL])
        fit_seconds = time.perf_counter() - start

        val_scores = model.predict_proba(val[FEATURES])[:, 1]
        threshold = threshold_for_fpr(val[LABEL], val_scores, MAX_FPR)

        test_scores = model.predict_proba(test[FEATURES])[:, 1]
        metrics = asdict(evaluate(test[LABEL], test_scores, threshold))
        dollars = threshold_table(test[LABEL], test_scores, test["Amount"], [threshold])
        metrics["fraud_dollars_caught_pct"] = float(dollars["fraud_dollars_caught_pct"].iloc[0])
        metrics["threshold"] = threshold
        metrics["fit_seconds"] = round(fit_seconds, 1)
        results[name] = metrics

        curve = threshold_table(test[LABEL], test_scores, test["Amount"])
        curve.insert(0, "model", name)
        curves.append(curve)
        joblib.dump(model, ARTIFACTS / f"{name}.joblib")

    print(f"\nTest results at the threshold that kept validation FPR <= {MAX_FPR:.1%}:")
    print(f"{'model':<22}{'PR-AUC':>8}{'thresh':>8}{'caught':>9}{'recall':>8}"
          f"{'precision':>11}{'FPR':>8}{'$ caught':>10}")
    for name, m in results.items():
        caught = f"{m['true_positives']}/{m['true_positives'] + m['false_negatives']}"
        print(f"{name:<22}{m['pr_auc']:>8.3f}{m['threshold']:>8.3f}{caught:>9}{m['recall']:>8.1%}"
              f"{m['precision']:>11.1%}{m['false_positive_rate']:>8.3%}"
              f"{m['fraud_dollars_caught_pct']:>10.1%}")

    (REPORTS / "model_comparison.json").write_text(
        json.dumps({"max_validation_fpr": MAX_FPR, "results": results}, indent=2) + "\n"
    )
    pd.concat(curves).round(4).to_csv(REPORTS / "threshold_curve.csv", index=False)
    print("\nSaved reports/model_comparison.json and reports/threshold_curve.csv")


if __name__ == "__main__":
    main()
