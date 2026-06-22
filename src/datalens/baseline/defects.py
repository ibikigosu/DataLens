"""Inject reproducible controlled defects into real-shaped procurement data."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from datalens.baseline.issues import issue
from datalens.data.records import add_record_ids


@dataclass(frozen=True)
class ControlledDefect:
    """One issue-level ground-truth label."""

    target_table: str
    record_id: str
    issue_type: str
    severity: str
    field: str
    original_value: Any
    injected_value: Any
    seed: int


def _value_type(value: Any) -> str:
    if value is None or pd.isna(value):
        return "null"
    return type(value).__name__


def _serialize_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "null"
    if isinstance(value, pd.Timestamp):
        return json.dumps(value.isoformat())
    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, default=str)


def _select_indices(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    count: int,
    used_record_ids: set[str],
    eligible: pd.Series | None = None,
) -> list[int]:
    mask = ~frame["_record_id"].isin(used_record_ids)
    if eligible is not None:
        mask &= eligible.fillna(False)
    candidates = frame.index[mask].to_numpy()
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} eligible records are available for {count} defects"
        )
    return rng.choice(candidates, size=count, replace=False).tolist()


def _record_mutation(
    frame: pd.DataFrame,
    indices: list[int],
    *,
    issue_type: str,
    field: str,
    value_factory: Callable[[pd.Series, int], Any],
    seed: int,
) -> list[ControlledDefect]:
    defects = []
    definition = issue(issue_type)
    for offset, index in enumerate(indices):
        original = frame.at[index, field]
        injected = value_factory(frame.loc[index], offset)
        frame.at[index, field] = injected
        defects.append(
            ControlledDefect(
                target_table=definition.target_table,
                record_id=str(frame.at[index, "_record_id"]),
                issue_type=issue_type,
                severity=definition.severity.value,
                field=field,
                original_value=original,
                injected_value=injected,
                seed=seed,
            )
        )
    return defects


def _append_duplicates(
    frame: pd.DataFrame,
    indices: list[int],
    *,
    issue_type: str,
    key_field: str,
    seed: int,
) -> tuple[pd.DataFrame, list[ControlledDefect]]:
    duplicates = frame.loc[indices].copy()
    defects = []
    definition = issue(issue_type)
    for offset, index in enumerate(indices):
        duplicate_record_id = f"controlled:{issue_type}:{seed}:{offset}"
        duplicates.loc[index, "_record_id"] = duplicate_record_id
        defects.append(
            ControlledDefect(
                target_table=definition.target_table,
                record_id=duplicate_record_id,
                issue_type=issue_type,
                severity=definition.severity.value,
                field=key_field,
                original_value=frame.at[index, key_field],
                injected_value=frame.at[index, key_field],
                seed=seed,
            )
        )
    return pd.concat([frame, duplicates], ignore_index=True), defects


def inject_controlled_defects(
    vendors: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    fiscal_year: int,
    seed: int,
    defects_per_type: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inject the controlled-defect catalog and return issue-level labels."""
    vendor_frame, transaction_frame = add_record_ids(vendors, transactions)
    rng = np.random.default_rng(seed + fiscal_year)
    used_vendor_ids: set[str] = set()
    used_transaction_ids: set[str] = set()
    labels: list[ControlledDefect] = []

    vendor_indices = _select_indices(
        vendor_frame,
        rng,
        defects_per_type,
        used_vendor_ids,
        vendor_frame["recipient_name"].notna(),
    )
    labels.extend(
        _record_mutation(
            vendor_frame,
            vendor_indices,
            issue_type="missing_vendor_name",
            field="recipient_name",
            value_factory=lambda row, offset: pd.NA,
            seed=seed,
        )
    )
    used_vendor_ids.update(vendor_frame.loc[vendor_indices, "_record_id"].astype(str))

    vendor_indices = _select_indices(
        vendor_frame,
        rng,
        defects_per_type,
        used_vendor_ids,
        vendor_frame["recipient_uei"].notna(),
    )
    labels.extend(
        _record_mutation(
            vendor_frame,
            vendor_indices,
            issue_type="invalid_vendor_uei",
            field="recipient_uei",
            value_factory=lambda row, offset: f"BAD{offset:03d}!",
            seed=seed,
        )
    )
    used_vendor_ids.update(vendor_frame.loc[vendor_indices, "_record_id"].astype(str))

    vendor_indices = _select_indices(
        vendor_frame,
        rng,
        defects_per_type,
        used_vendor_ids,
        vendor_frame["recipient_state_code"].notna(),
    )
    labels.extend(
        _record_mutation(
            vendor_frame,
            vendor_indices,
            issue_type="invalid_domestic_state",
            field="recipient_state_code",
            value_factory=lambda row, offset: "ZZ",
            seed=seed,
        )
    )
    vendor_frame.loc[vendor_indices, "recipient_country_code"] = "USA"
    used_vendor_ids.update(vendor_frame.loc[vendor_indices, "_record_id"].astype(str))

    vendor_indices = _select_indices(
        vendor_frame,
        rng,
        defects_per_type,
        used_vendor_ids,
    )
    vendor_frame, duplicate_labels = _append_duplicates(
        vendor_frame,
        vendor_indices,
        issue_type="duplicate_vendor_id",
        key_field="vendor_id",
        seed=seed,
    )
    labels.extend(duplicate_labels)

    transaction_indices = _select_indices(
        transaction_frame,
        rng,
        defects_per_type,
        used_transaction_ids,
    )
    labels.extend(
        _record_mutation(
            transaction_frame,
            transaction_indices,
            issue_type="orphan_vendor_reference",
            field="vendor_id",
            value_factory=lambda row, offset: f"UEI:CONTROLLEDORPHAN{offset:04d}",
            seed=seed,
        )
    )
    used_transaction_ids.update(
        transaction_frame.loc[transaction_indices, "_record_id"].astype(str)
    )

    transaction_indices = _select_indices(
        transaction_frame,
        rng,
        defects_per_type,
        used_transaction_ids,
    )
    transaction_frame, duplicate_labels = _append_duplicates(
        transaction_frame,
        transaction_indices,
        issue_type="duplicate_transaction_key",
        key_field="contract_transaction_unique_key",
        seed=seed,
    )
    labels.extend(duplicate_labels)

    transaction_indices = _select_indices(
        transaction_frame,
        rng,
        defects_per_type,
        used_transaction_ids,
        transaction_frame["period_of_performance_start_date"].notna()
        & transaction_frame["period_of_performance_current_end_date"].notna(),
    )
    labels.extend(
        _record_mutation(
            transaction_frame,
            transaction_indices,
            issue_type="invalid_performance_date_order",
            field="period_of_performance_current_end_date",
            value_factory=lambda row, offset: row["period_of_performance_start_date"]
            - pd.Timedelta(days=1),
            seed=seed,
        )
    )
    used_transaction_ids.update(
        transaction_frame.loc[transaction_indices, "_record_id"].astype(str)
    )

    transaction_indices = _select_indices(
        transaction_frame,
        rng,
        defects_per_type,
        used_transaction_ids,
    )
    labels.extend(
        _record_mutation(
            transaction_frame,
            transaction_indices,
            issue_type="negative_offer_count",
            field="number_of_offers_received",
            value_factory=lambda row, offset: -1,
            seed=seed,
        )
    )
    used_transaction_ids.update(
        transaction_frame.loc[transaction_indices, "_record_id"].astype(str)
    )

    transaction_indices = _select_indices(
        transaction_frame,
        rng,
        defects_per_type,
        used_transaction_ids,
    )
    labels.extend(
        _record_mutation(
            transaction_frame,
            transaction_indices,
            issue_type="action_date_outside_fiscal_year",
            field="action_date",
            value_factory=lambda row, offset: pd.Timestamp(
                year=fiscal_year - 2,
                month=1,
                day=1,
                tz="UTC",
            ),
            seed=seed,
        )
    )

    label_frame = pd.DataFrame([defect.__dict__ for defect in labels])
    label_frame["original_value_type"] = label_frame["original_value"].map(_value_type)
    label_frame["injected_value_type"] = label_frame["injected_value"].map(_value_type)
    label_frame["original_value"] = label_frame["original_value"].map(_serialize_value)
    label_frame["injected_value"] = label_frame["injected_value"].map(_serialize_value)
    label_frame["severity_rank"] = label_frame["issue_type"].map(
        lambda issue_type: issue(issue_type).severity_rank
    )
    return vendor_frame, transaction_frame, label_frame
