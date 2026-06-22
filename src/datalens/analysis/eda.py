"""Reusable exploratory data analysis summaries."""

from __future__ import annotations

from typing import Any

import pandas as pd


def infer_semantic_type(column: str, series: pd.Series) -> str:
    """Infer a compact semantic type for EDA documentation."""
    lower = column.lower()
    if lower.endswith("_date") or "date_" in lower or pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if lower.endswith("_id") or lower.endswith("_key") or lower.endswith("_code"):
        return "identifier"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    if unique_ratio < 0.05:
        return "categorical"
    return "text"


def profile_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return field-level type, completeness, and cardinality metrics."""
    rows = []
    for column in frame.columns:
        series = frame[column]
        rows.append(
            {
                "field": column,
                "dtype": str(series.dtype),
                "semantic_type": infer_semantic_type(column, series),
                "missing_count": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().mean() * 100), 3),
                "unique_count": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["missing_percent", "field"],
        ascending=[False, True],
        ignore_index=True,
    )


def duplicate_summary(
    frame: pd.DataFrame,
    *,
    key: str,
) -> dict[str, Any]:
    """Summarize exact rows and duplicate candidate keys."""
    duplicated_keys = frame[key].duplicated(keep=False) & frame[key].notna()
    return {
        "rows": len(frame),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_key_rows": int(duplicated_keys.sum()),
        "duplicate_key_values": int(frame.loc[duplicated_keys, key].nunique()),
        "missing_key_rows": int(frame[key].isna().sum()),
    }


def join_quality(vendors: pd.DataFrame, transactions: pd.DataFrame) -> dict[str, Any]:
    """Measure transaction-to-vendor relationship quality."""
    vendor_keys = set(vendors["vendor_id"].dropna())
    missing = transactions["vendor_id"].isna()
    unmatched = transactions["vendor_id"].notna() & ~transactions["vendor_id"].isin(vendor_keys)
    matched = ~(missing | unmatched)
    return {
        "transaction_rows": len(transactions),
        "vendor_rows": len(vendors),
        "missing_vendor_key_rows": int(missing.sum()),
        "unmatched_vendor_key_rows": int(unmatched.sum()),
        "matched_transaction_rows": int(matched.sum()),
        "match_rate_percent": round(float(matched.mean() * 100), 3),
    }
