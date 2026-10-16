"""Export the small files the dashboard needs, so it runs without the raw data.

The raw CSV (150 MB) and trained models are not committed to Git. This script
saves only the results the dashboard displays (about 1-2 MB) to
reports/dashboard/, which are committed and used by the deployed app.

Run after scripts/compare_models.py.

Usage:
    python scripts/export_dashboard_data.py
"""

import json
from dataclasses import asdict

import joblib
import pandas as pd

from riskwatch.analysis import run_query
from riskwatch.config import LABEL, PCA_FEATURES, PROJECT_ROOT
from riskwatch.ingest import load_csv
from riskwatch.model import FEATURES, load_dataset, time_split
from riskwatch.monitoring import feature_drift
from riskwatch.quality import run_all_checks

OUT = PROJECT_ROOT / "reports" / "dashboard"
MODELS = ["random_forest", "logistic_regression"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    for name in ["01_overview", "02_amount_bands", "03_hourly_risk"]:
        run_query(name).to_csv(OUT / f"{name[3:]}.csv", index=False)

    train_full, test = time_split(load_dataset())
    train, val = time_split(train_full)
    models = {m: joblib.load(PROJECT_ROOT / "artifacts" / f"{m}.joblib") for m in MODELS}

    # Test-period labels, amounts and each model's fraud score: everything the
    # threshold slider and precision-recall curves need.
    scores = test[["Class", "Amount"]].reset_index(drop=True)
    for m, model in models.items():
        scores[f"score_{m}"] = model.predict_proba(test[FEATURES])[:, 1]
    scores.to_csv(OUT / "test_scores.csv.gz", index=False, float_format="%.6g")

    feature_drift(train, test, [*PCA_FEATURES, "log_amount"]).to_csv(
        OUT / "feature_drift.csv", index=False)

    comparison = json.loads((PROJECT_ROOT / "reports" / "model_comparison.json").read_text())
    rf_threshold = comparison["results"]["random_forest"]["threshold"]
    val_scores = models["random_forest"].predict_proba(val[FEATURES])[:, 1]

    raw = load_csv().drop(columns="transaction_id")
    summary = {
        "quality_checks": [asdict(c) for c in run_all_checks(raw)],
        "rf_alert_rate_validation": float((val_scores >= rf_threshold).mean()),
        "fraud_rate_train": float(train[LABEL].mean()),
        "fraud_rate_test": float(test[LABEL].mean()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for path in sorted(OUT.iterdir()):
        print(f"{path.relative_to(PROJECT_ROOT)}  {path.stat().st_size / 1024:,.0f} KB")


if __name__ == "__main__":
    main()
