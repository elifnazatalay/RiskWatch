"""Load the raw Kaggle CSV and store it in a SQLite database."""

import sqlite3
from pathlib import Path

import pandas as pd

from riskwatch.config import DB_PATH, RAW_CSV, TABLE_NAME


def load_csv(path: Path = RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from Kaggle into data/raw/."
        )
    df = pd.read_csv(path)
    # Give every row a stable ID so predictions can be traced back to a transaction.
    df.insert(0, "transaction_id", range(1, len(df) + 1))
    return df


def write_to_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH, table: str = TABLE_NAME) -> int:
    """Replace the table with df and return the number of rows stored."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_class ON {table}(Class)")
        (count,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return count
