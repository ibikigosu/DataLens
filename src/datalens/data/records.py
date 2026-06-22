"""Shared record identity helpers for procurement datasets."""

import pandas as pd


def add_record_id(frame: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    """Add deterministic row identity independent of mutable business keys."""
    identified = frame.copy()
    identified["_record_id"] = pd.Series(
        (f"{table_name}:{position:08d}" for position in range(len(identified))),
        index=identified.index,
        dtype="string",
    )
    return identified


def add_record_ids(
    vendors: pd.DataFrame,
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add table-scoped row identifiers without mutating source frames."""
    return (
        add_record_id(vendors, table_name="vendor"),
        add_record_id(transactions, table_name="transaction"),
    )
