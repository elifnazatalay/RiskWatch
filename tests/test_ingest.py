import sqlite3

import pandas as pd
import pytest

from riskwatch.config import EXPECTED_COLUMNS
from riskwatch.ingest import load_csv, write_to_sqlite


def make_csv(tmp_path, n=5):
    rows = [[float(i)] * 29 + [1.0, 0] for i in range(n)]
    path = tmp_path / "creditcard.csv"
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(path, index=False)
    return path


def test_load_csv_adds_sequential_ids(tmp_path):
    df = load_csv(make_csv(tmp_path))
    assert df["transaction_id"].tolist() == [1, 2, 3, 4, 5]


def test_load_csv_missing_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_csv(tmp_path / "nope.csv")


def test_write_to_sqlite_round_trip(tmp_path):
    df = load_csv(make_csv(tmp_path))
    db = tmp_path / "test.db"
    assert write_to_sqlite(df, db_path=db) == 5
    # Running twice must replace, not append.
    assert write_to_sqlite(df, db_path=db) == 5
    with sqlite3.connect(db) as conn:
        stored = pd.read_sql("SELECT * FROM transactions", conn)
    assert list(stored.columns) == ["transaction_id", *EXPECTED_COLUMNS]
