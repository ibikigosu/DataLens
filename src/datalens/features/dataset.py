"""Typed construction of development-period feature datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datalens.data.plan import DatasetPlan, load_dataset_plan
from datalens.features.builders import build_transaction_features, build_vendor_features


@dataclass(frozen=True)
class DevelopmentFeatureDataset:
    """Vendor and transaction feature frames approved for fitting."""

    fiscal_year: str
    vendors: pd.DataFrame
    transactions: pd.DataFrame

    @classmethod
    def from_records(
        cls,
        vendors: pd.DataFrame,
        transactions: pd.DataFrame,
        *,
        fiscal_year: int | str,
        dataset_plan: DatasetPlan | None = None,
    ) -> DevelopmentFeatureDataset:
        """Validate period ownership and build both table-specific feature frames."""
        plan = dataset_plan or load_dataset_plan()
        period = plan.period(fiscal_year)
        if period.role != "development":
            raise ValueError(
                f"Feature pipelines may only be fitted on a development period, "
                f"but FY{period.fiscal_year} is {period.role}"
            )
        return cls(
            fiscal_year=period.fiscal_year,
            vendors=build_vendor_features(vendors),
            transactions=build_transaction_features(transactions),
        )
