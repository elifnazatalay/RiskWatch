"""Print the results of every SQL analysis query.

Usage:
    python scripts/run_analysis.py
"""

import pandas as pd

from riskwatch.analysis import list_queries, run_query


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    for name in list_queries():
        print(f"\n=== {name} ===")
        print(run_query(name).to_string(index=False))


if __name__ == "__main__":
    main()
