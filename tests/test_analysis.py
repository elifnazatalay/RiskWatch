import pandas as pd
import pytest

from riskwatch.analysis import list_queries, run_query
from riskwatch.config import EXPECTED_COLUMNS
from riskwatch.ingest import write_to_sqlite


@pytest.fixture
def small_db(tmp_path):
    # 10 transactions, 2 frauds: one $5 fraud in hour 0 of day 1,
    # one $800 fraud in hour 2 of day 2.
    rows = []
    for i in range(10):
        time = 86400 + 2 * 3600 if i == 9 else 0.0
        amount = {0: 5.0, 9: 800.0}.get(i, 20.0)
        label = 1 if i in (0, 9) else 0
        rows.append([time] + [0.0] * 28 + [amount, label])
    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    df.insert(0, "transaction_id", range(1, 11))
    db = tmp_path / "test.db"
    write_to_sqlite(df, db_path=db)
    return db


def test_every_query_runs(small_db):
    names = list_queries()
    assert len(names) == 5
    for name in names:
        assert not run_query(name, db_path=small_db).empty


def test_overview_counts(small_db):
    row = run_query("01_overview", db_path=small_db).iloc[0]
    assert row["total_transactions"] == 10
    assert row["fraud_transactions"] == 2
    assert row["fraud_rate_pct"] == 20.0
    assert row["fraud_amount"] == 805.0


def test_amount_band_fraud_shares_sum_to_100(small_db):
    bands = run_query("02_amount_bands", db_path=small_db)
    assert bands["share_of_all_fraud_pct"].sum() == pytest.approx(100.0)
    assert bands["transactions"].sum() == 10


def test_hourly_riskiest_hour_is_ranked_first(small_db):
    hourly = run_query("03_hourly_risk", db_path=small_db)
    top = hourly.iloc[0]
    assert top["hour"] == 2
    assert top["risk_rank"] == 1


def test_daily_drift_splits_by_day(small_db):
    daily = run_query("05_daily_drift", db_path=small_db)
    assert daily["day"].tolist() == [1, 2]
    assert daily["transactions"].tolist() == [9, 1]
