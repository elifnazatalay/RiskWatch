"""Run data-quality checks on the raw CSV and load it into SQLite.

Usage:
    python scripts/load_data.py
"""

import sys
import time

from riskwatch.config import DB_PATH
from riskwatch.ingest import load_csv, write_to_sqlite
from riskwatch.quality import run_all_checks


def main() -> int:
    start = time.perf_counter()
    df = load_csv()
    print(f"Loaded {len(df):,} rows from CSV")

    results = run_all_checks(df.drop(columns="transaction_id"))
    print("\nData-quality checks:")
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")

    if not all(r.passed for r in results):
        print("\nStopping: fix the failed checks before loading.")
        return 1

    count = write_to_sqlite(df)
    elapsed = time.perf_counter() - start
    print(f"\nStored {count:,} rows in {DB_PATH.relative_to(DB_PATH.parents[2])} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
