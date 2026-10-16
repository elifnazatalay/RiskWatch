"""Run the SQL analysis queries stored in the sql/ folder."""

import sqlite3
from pathlib import Path

import pandas as pd

from riskwatch.config import DB_PATH, PROJECT_ROOT

SQL_DIR = PROJECT_ROOT / "sql"


def list_queries(sql_dir: Path = SQL_DIR) -> list[str]:
    """Query names in run order, e.g. ['01_overview', '02_amount_bands', ...]."""
    return sorted(p.stem for p in sql_dir.glob("*.sql"))


def run_query(name: str, db_path: Path = DB_PATH, sql_dir: Path = SQL_DIR) -> pd.DataFrame:
    sql = (sql_dir / f"{name}.sql").read_text()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(sql, conn)
