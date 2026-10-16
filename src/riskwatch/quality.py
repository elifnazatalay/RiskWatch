"""Data-quality checks run before any analysis or modeling.

Each check returns a CheckResult so the results can be printed in the
terminal now and shown on the dashboard later.
"""

from dataclasses import dataclass

import pandas as pd

from riskwatch.config import EXPECTED_COLUMNS, LABEL


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_schema(df: pd.DataFrame) -> CheckResult:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    passed = not missing and not extra
    detail = "all 31 expected columns present" if passed else f"missing={missing}, extra={extra}"
    return CheckResult("schema", passed, detail)


def check_missing_values(df: pd.DataFrame) -> CheckResult:
    counts = df.isna().sum()
    bad = counts[counts > 0]
    if bad.empty:
        return CheckResult("missing_values", True, "no missing values")
    return CheckResult("missing_values", False, f"{int(bad.sum())} missing cells in {list(bad.index)}")


def check_duplicates(df: pd.DataFrame) -> CheckResult:
    # Duplicates are reported, not failed: identical rows can be real repeat
    # transactions, so we decide how to handle them at modeling time.
    n = int(df.duplicated().sum())
    return CheckResult("duplicate_rows", True, f"{n} fully duplicated rows ({n / len(df):.2%})")


def check_labels(df: pd.DataFrame) -> CheckResult:
    invalid = ~df[LABEL].isin([0, 1])
    n = int(invalid.sum())
    if n:
        return CheckResult("valid_labels", False, f"{n} rows with Class not in {{0, 1}}")
    return CheckResult("valid_labels", True, "every Class value is 0 or 1")


def check_amounts(df: pd.DataFrame) -> CheckResult:
    n = int((df["Amount"] < 0).sum())
    if n:
        return CheckResult("non_negative_amount", False, f"{n} rows with negative Amount")
    return CheckResult("non_negative_amount", True, "no negative amounts")


def run_all_checks(df: pd.DataFrame) -> list[CheckResult]:
    schema = check_schema(df)
    if not schema.passed:
        # The other checks need the expected columns, so stop here.
        return [schema]
    return [
        schema,
        check_missing_values(df),
        check_duplicates(df),
        check_labels(df),
        check_amounts(df),
    ]
