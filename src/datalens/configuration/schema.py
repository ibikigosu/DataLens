"""DataFrame validation against the approved procurement schema."""

from __future__ import annotations

import pandas as pd

from datalens.configuration.models import DataType, ProcurementSchema, TableConfiguration


def validate_procurement_frames(
    vendors: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    schema: ProcurementSchema,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and normalize a paired procurement dataset."""
    validated_vendors = _validate_table(vendors, schema.vendor, table_name="vendor")
    validated_transactions = _validate_table(
        transactions,
        schema.transaction,
        table_name="transaction",
    )
    relationship = schema.transaction_vendor_relationship
    vendor_keys = set(validated_vendors[relationship.to_column].dropna().astype(str))
    transaction_vendor_keys = set(
        validated_transactions[relationship.from_column].dropna().astype(str)
    )
    missing_vendor_keys = sorted(transaction_vendor_keys - vendor_keys)
    if missing_vendor_keys:
        preview = missing_vendor_keys[:5]
        raise ValueError(
            f"Transaction records reference {len(missing_vendor_keys)} undeclared vendor keys; "
            f"examples={preview}"
        )
    return validated_vendors, validated_transactions


def validate_table_frame(
    frame: pd.DataFrame,
    *,
    schema: ProcurementSchema,
    table_name: str,
) -> pd.DataFrame:
    """Validate one table without evaluating cross-table relationships."""
    if table_name == "vendor":
        return _validate_table(frame, schema.vendor, table_name=table_name)
    if table_name == "transaction":
        return _validate_table(frame, schema.transaction, table_name=table_name)
    raise ValueError(f"Unknown table name: {table_name}")


def _validate_table(
    frame: pd.DataFrame,
    table: TableConfiguration,
    *,
    table_name: str,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"{table_name.title()} input must contain at least one record")
    if frame.columns.has_duplicates:
        raise ValueError(f"{table_name.title()} input column names must be unique")
    missing_columns = sorted(set(table.required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{table_name.title()} input is missing required columns: {missing_columns}"
        )

    validated = frame.copy()
    for column in table.columns:
        if column.name not in validated:
            continue
        validated[column.name] = _coerce_column(
            validated[column.name],
            data_type=column.data_type,
            field_name=f"{table_name}.{column.name}",
        )
        if not column.nullable and validated[column.name].isna().any():
            raise ValueError(f"{table_name}.{column.name} cannot contain missing values")

    business_keys = validated[table.business_key].astype("string").str.strip()
    if business_keys.isna().any() or business_keys.eq("").any():
        raise ValueError(f"{table_name}.{table.business_key} cannot contain blank values")
    validated[table.business_key] = business_keys
    return validated


def _coerce_column(
    values: pd.Series,
    *,
    data_type: DataType,
    field_name: str,
) -> pd.Series:
    if data_type is DataType.STRING:
        return values.astype("string").str.strip().mask(lambda series: series.eq(""))
    if data_type is DataType.DATETIME:
        converted = pd.to_datetime(values, errors="coerce", utc=True)
    else:
        converted = pd.to_numeric(values, errors="coerce")
        if data_type is DataType.INTEGER:
            fractional = converted.dropna() % 1
            if fractional.ne(0).any():
                raise ValueError(f"{field_name} must contain integers")
            converted = converted.astype("Int64")
        else:
            converted = converted.astype("Float64")
    invalid = values.notna() & converted.isna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).head(3).tolist()
        raise ValueError(f"{field_name} contains invalid {data_type.value} values: {examples}")
    return converted
