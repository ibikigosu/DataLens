"""Registry-driven deterministic data-quality rules."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from datalens.baseline.issues import ISSUES, IssueDefinition, TargetTable, issue

VALID_US_STATE_CODES = {
    "AK",
    "AL",
    "AR",
    "AS",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "FM",
    "GA",
    "GU",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MH",
    "MI",
    "MN",
    "MO",
    "MP",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "PR",
    "PW",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VA",
    "VI",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
}

MaskFactory = Callable[["RuleContext"], pd.Series]
EvidenceFactory = Callable[[pd.Series, "RuleContext"], str]


@dataclass(frozen=True)
class RuleContext:
    vendors: pd.DataFrame
    transactions: pd.DataFrame
    fiscal_year: int

    def frame_for(self, definition: IssueDefinition) -> pd.DataFrame:
        return self.vendors if definition.target_table == "vendor" else self.transactions

    @property
    def fiscal_year_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (
            pd.Timestamp(year=self.fiscal_year - 1, month=10, day=1, tz="UTC"),
            pd.Timestamp(year=self.fiscal_year, month=9, day=30, tz="UTC"),
        )


@dataclass(frozen=True)
class QualityRule:
    """One detector tied to a canonical issue definition."""

    definition: IssueDefinition
    mask: MaskFactory
    evidence: EvidenceFactory
    requires_relationship_context: bool = False

    def findings(self, context: RuleContext) -> list[dict[str, Any]]:
        frame = context.frame_for(self.definition)
        mask = self.mask(context).fillna(False)
        return [
            {
                "target_table": self.definition.target_table,
                "record_id": str(row["_record_id"]),
                "issue_type": self.definition.issue_type,
                "severity": self.definition.severity.value,
                "severity_rank": self.definition.severity_rank,
                "evidence": self.evidence(row, context),
            }
            for _, row in frame.loc[mask].iterrows()
        ]


def _duplicate_mask(frame: pd.DataFrame, key: str) -> pd.Series:
    return frame[key].notna() & frame[key].duplicated(keep=False)


RULES = (
    QualityRule(
        issue("missing_vendor_name"),
        lambda context: context.vendors["recipient_name"].isna()
        | context.vendors["recipient_name"].astype("string").str.strip().eq(""),
        lambda row, context: "recipient_name is missing",
    ),
    QualityRule(
        issue("invalid_vendor_uei"),
        lambda context: context.vendors["recipient_uei"].notna()
        & ~context.vendors["recipient_uei"].astype("string").str.fullmatch(r"[A-Z0-9]{12}"),
        lambda row, context: (
            f"recipient_uei={row['recipient_uei']} is not 12 alphanumeric characters"
        ),
    ),
    QualityRule(
        issue("invalid_domestic_state"),
        lambda context: context.vendors["recipient_country_code"].eq("USA")
        & ~context.vendors["recipient_state_code"].isin(VALID_US_STATE_CODES),
        lambda row, context: (
            f"recipient_state_code={row['recipient_state_code']} is invalid for USA"
        ),
    ),
    QualityRule(
        issue("duplicate_vendor_id"),
        lambda context: _duplicate_mask(context.vendors, "vendor_id"),
        lambda row, context: f"vendor_id={row['vendor_id']} appears more than once",
    ),
    QualityRule(
        issue("orphan_vendor_reference"),
        lambda context: context.transactions["vendor_id"].isna()
        | ~context.transactions["vendor_id"].isin(set(context.vendors["vendor_id"].dropna())),
        lambda row, context: (f"vendor_id={row['vendor_id']} does not identify a vendor record"),
        requires_relationship_context=True,
    ),
    QualityRule(
        issue("duplicate_transaction_key"),
        lambda context: _duplicate_mask(
            context.transactions,
            "contract_transaction_unique_key",
        ),
        lambda row, context: (
            f"contract_transaction_unique_key={row['contract_transaction_unique_key']} "
            "appears more than once"
        ),
    ),
    QualityRule(
        issue("invalid_performance_date_order"),
        lambda context: context.transactions["period_of_performance_start_date"].notna()
        & context.transactions["period_of_performance_current_end_date"].notna()
        & (
            context.transactions["period_of_performance_current_end_date"]
            < context.transactions["period_of_performance_start_date"]
        ),
        lambda row, context: (
            "period_of_performance_current_end_date is before period_of_performance_start_date"
        ),
    ),
    QualityRule(
        issue("negative_offer_count"),
        lambda context: context.transactions["number_of_offers_received"].notna()
        & (context.transactions["number_of_offers_received"] < 0),
        lambda row, context: (
            f"number_of_offers_received={row['number_of_offers_received']} is negative"
        ),
    ),
    QualityRule(
        issue("action_date_outside_fiscal_year"),
        lambda context: (
            context.transactions["action_date"].isna()
            | (context.transactions["action_date"] < context.fiscal_year_bounds[0])
            | (context.transactions["action_date"] > context.fiscal_year_bounds[1])
        ),
        lambda row, context: (
            f"action_date={row['action_date']} is outside FY{context.fiscal_year}"
        ),
    ),
)

if {rule.definition.issue_type for rule in RULES} != set(ISSUES):
    raise RuntimeError("Every canonical quality issue must have exactly one deterministic rule")


def run_rules(
    vendors: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    fiscal_year: int,
    scoring_weights: Mapping[str, int],
    target_tables: Collection[TargetTable] | None = None,
    include_relationship_rules: bool = True,
) -> pd.DataFrame:
    """Run every registered deterministic quality rule."""
    if set(scoring_weights) != set(ISSUES):
        raise ValueError("Scoring weights must be declared for every canonical quality issue")
    context = RuleContext(vendors=vendors, transactions=transactions, fiscal_year=fiscal_year)
    selected_tables = set(target_tables or ("vendor", "transaction"))
    selected_rules = (
        rule
        for rule in RULES
        if rule.definition.target_table in selected_tables
        and (include_relationship_rules or not rule.requires_relationship_context)
    )
    finding_frame = pd.DataFrame(
        finding for rule in selected_rules for finding in rule.findings(context)
    )
    if finding_frame.empty:
        return pd.DataFrame(
            columns=[
                "target_table",
                "record_id",
                "issue_type",
                "severity",
                "severity_rank",
                "evidence",
                "risk_score",
            ]
        )
    issue_counts = finding_frame.groupby(["target_table", "record_id"]).size()
    finding_frame["risk_score"] = finding_frame.apply(
        lambda row: min(
            100,
            scoring_weights[row["issue_type"]]
            + 5 * (issue_counts.loc[(row["target_table"], row["record_id"])] - 1),
        ),
        axis=1,
    )
    return finding_frame.sort_values(
        ["risk_score", "severity_rank", "target_table", "record_id", "issue_type"],
        ascending=[False, False, True, True, True],
        ignore_index=True,
    )


def finding_keys(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    """Return unique issue-level finding keys."""
    return set(
        frame[["target_table", "record_id", "issue_type"]]
        .astype("string")
        .itertuples(index=False, name=None)
    )


def record_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """Return unique record-level keys."""
    return set(
        frame[["target_table", "record_id"]].astype("string").itertuples(index=False, name=None)
    )
