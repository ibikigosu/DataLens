"""Canonical quality-issue definitions shared by rules and controlled defects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

TargetTable = Literal["vendor", "transaction"]


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class IssueDefinition:
    """Stable identity and business severity for one quality issue."""

    issue_type: str
    target_table: TargetTable
    severity: Severity

    @property
    def severity_rank(self) -> int:
        return SEVERITY_RANK[self.severity]

    @property
    def base_risk_score(self) -> int:
        return SEVERITY_SCORE[self.severity]


SEVERITY_RANK = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}
SEVERITY_SCORE = {
    Severity.LOW: 25,
    Severity.MEDIUM: 50,
    Severity.HIGH: 75,
    Severity.CRITICAL: 100,
}

ISSUES = {
    definition.issue_type: definition
    for definition in (
        IssueDefinition("missing_vendor_name", "vendor", Severity.HIGH),
        IssueDefinition("invalid_vendor_uei", "vendor", Severity.CRITICAL),
        IssueDefinition("invalid_domestic_state", "vendor", Severity.HIGH),
        IssueDefinition("duplicate_vendor_id", "vendor", Severity.CRITICAL),
        IssueDefinition("orphan_vendor_reference", "transaction", Severity.CRITICAL),
        IssueDefinition("duplicate_transaction_key", "transaction", Severity.CRITICAL),
        IssueDefinition("invalid_performance_date_order", "transaction", Severity.HIGH),
        IssueDefinition("negative_offer_count", "transaction", Severity.MEDIUM),
        IssueDefinition("action_date_outside_fiscal_year", "transaction", Severity.HIGH),
    )
}


def issue(issue_type: str) -> IssueDefinition:
    """Return one canonical issue definition."""
    try:
        return ISSUES[issue_type]
    except KeyError as error:
        raise KeyError(f"Unknown quality issue: {issue_type}") from error
