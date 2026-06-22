import pandas as pd

from datalens.analysis.eda import (
    duplicate_summary,
    infer_semantic_type,
    join_quality,
    profile_table,
)


def test_profile_table_reports_missingness_and_semantic_types() -> None:
    frame = pd.DataFrame(
        {
            "vendor_id": ["A", "B"],
            "amount": [10.0, None],
            "category": ["x", "x"],
        }
    )

    profile = profile_table(frame).set_index("field")

    assert profile.loc["amount", "missing_percent"] == 50.0
    assert profile.loc["vendor_id", "semantic_type"] == "identifier"
    assert profile.loc["amount", "semantic_type"] == "numeric"


def test_duplicate_summary_counts_key_problems() -> None:
    frame = pd.DataFrame({"transaction_key": ["A", "A", None], "value": [1, 2, 3]})

    summary = duplicate_summary(frame, key="transaction_key")

    assert summary["duplicate_key_rows"] == 2
    assert summary["duplicate_key_values"] == 1
    assert summary["missing_key_rows"] == 1


def test_join_quality_distinguishes_missing_and_unmatched_keys() -> None:
    vendors = pd.DataFrame({"vendor_id": ["UEI:A"]})
    transactions = pd.DataFrame({"vendor_id": ["UEI:A", "UEI:B", None]})

    summary = join_quality(vendors, transactions)

    assert summary["matched_transaction_rows"] == 1
    assert summary["unmatched_vendor_key_rows"] == 1
    assert summary["missing_vendor_key_rows"] == 1
    assert summary["match_rate_percent"] == 33.333


def test_semantic_type_distinguishes_categories_dates_and_text() -> None:
    dates = pd.Series(pd.to_datetime(["2024-01-01"]))
    categories = pd.Series(["x"] * 100)
    text = pd.Series([f"value-{index}" for index in range(100)])

    assert infer_semantic_type("observed", dates) == "date"
    assert infer_semantic_type("category", categories) == "categorical"
    assert infer_semantic_type("description", text) == "text"
