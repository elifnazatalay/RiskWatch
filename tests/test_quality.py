import pandas as pd
import pytest

from riskwatch.config import EXPECTED_COLUMNS
from riskwatch.quality import run_all_checks


@pytest.fixture
def clean_df():
    rows = [[float(i)] * 29 + [10.0 * i, i % 2] for i in range(4)]
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def results_by_name(df):
    return {r.name: r for r in run_all_checks(df)}


def test_clean_data_passes_every_check(clean_df):
    assert all(r.passed for r in run_all_checks(clean_df))


def test_missing_column_stops_after_schema_check(clean_df):
    results = run_all_checks(clean_df.drop(columns="Amount"))
    assert len(results) == 1
    assert not results[0].passed
    assert "Amount" in results[0].detail


def test_missing_values_fail(clean_df):
    clean_df.loc[0, "V3"] = None
    assert not results_by_name(clean_df)["missing_values"].passed


def test_invalid_label_fails(clean_df):
    clean_df.loc[0, "Class"] = 2
    assert not results_by_name(clean_df)["valid_labels"].passed


def test_negative_amount_fails(clean_df):
    clean_df.loc[0, "Amount"] = -5.0
    assert not results_by_name(clean_df)["non_negative_amount"].passed


def test_duplicates_are_reported_not_failed(clean_df):
    df = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
    result = results_by_name(df)["duplicate_rows"]
    assert result.passed
    assert result.detail.startswith("1 ")
