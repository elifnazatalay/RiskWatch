"""Measure batch-inference latency of the trained models.

Run scripts/compare_models.py first so the models exist in artifacts/.

Usage:
    python scripts/benchmark_latency.py
"""

import json
import platform

import joblib

from riskwatch.config import PROJECT_ROOT
from riskwatch.model import FEATURES, load_dataset, time_split
from riskwatch.monitoring import benchmark_latency

MODELS = ["logistic_regression", "random_forest"]
REPORT_PATH = PROJECT_ROOT / "reports" / "latency.json"


def main() -> None:
    _, test = time_split(load_dataset())
    results = {}
    for name in MODELS:
        model = joblib.load(PROJECT_ROOT / "artifacts" / f"{name}.joblib")
        results[name] = benchmark_latency(model, test[FEATURES])
        r = results[name]
        print(f"{name:<22} {r['batch_size']:,} transactions: "
              f"median {r['median_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms")

    # Latency depends on hardware, so record where it was measured.
    report = {"machine": f"{platform.system()} {platform.machine()}", "results": results}
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved reports/{REPORT_PATH.name}")


if __name__ == "__main__":
    main()
