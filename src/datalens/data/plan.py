"""Typed configuration for the reproducible USAspending dataset plan."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datalens.paths import CONFIG_DIR

DEFAULT_DATA_PLAN_PATH = CONFIG_DIR / "data" / "usaspending-pbs.json"


@dataclass(frozen=True)
class AgencyFilter:
    """USAspending agency filter."""

    type: str
    tier: str
    name: str
    toptier_name: str

    def as_api_filter(self) -> dict[str, str]:
        return {
            "type": self.type,
            "tier": self.tier,
            "name": self.name,
            "toptier_name": self.toptier_name,
        }


@dataclass(frozen=True)
class FiscalPeriod:
    """One named dataset period and its evaluation role."""

    fiscal_year: str
    role: str
    start_date: str
    end_date: str

    def as_time_filter(self, date_type: str) -> dict[str, str]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "date_type": date_type,
        }


@dataclass(frozen=True)
class DatasetPlan:
    """Canonical acquisition and period plan for a procurement dataset."""

    source_name: str
    api_base_url: str
    dataset_name: str
    agency: AgencyFilter
    award_type_codes: tuple[str, ...]
    date_type: str
    periods: tuple[FiscalPeriod, ...]

    def period(self, fiscal_year: str | int) -> FiscalPeriod:
        requested = str(fiscal_year)
        for period in self.periods:
            if period.fiscal_year == requested:
                return period
        raise KeyError(f"Fiscal year {requested} is not declared in the dataset plan")

    @property
    def fiscal_years(self) -> tuple[str, ...]:
        return tuple(period.fiscal_year for period in self.periods)

    def build_filters(self, fiscal_year: str | int) -> dict[str, Any]:
        period = self.period(fiscal_year)
        return {
            "time_period": [period.as_time_filter(self.date_type)],
            "agencies": [self.agency.as_api_filter()],
            "award_type_codes": list(self.award_type_codes),
        }


def load_dataset_plan(path: Path = DEFAULT_DATA_PLAN_PATH) -> DatasetPlan:
    """Load and validate the versioned dataset plan."""
    with path.open(encoding="utf-8") as config_file:
        payload = json.load(config_file)
    periods = tuple(
        FiscalPeriod(
            fiscal_year=str(fiscal_year),
            role=values["role"],
            start_date=values["start_date"],
            end_date=values["end_date"],
        )
        for fiscal_year, values in payload["periods"].items()
    )
    if not periods:
        raise ValueError("The dataset plan must declare at least one fiscal period")
    agency = AgencyFilter(**payload["agency"])
    return DatasetPlan(
        source_name=payload["source_name"],
        api_base_url=payload["api_base_url"],
        dataset_name=payload["dataset_name"],
        agency=agency,
        award_type_codes=tuple(payload["award_type_codes"]),
        date_type=payload["date_type"],
        periods=periods,
    )
