"""Canonical single-record and paired-dataset scoring behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from datalens.baseline.rules import run_rules
from datalens.configuration.loader import RuntimeConfig
from datalens.configuration.schema import (
    validate_procurement_frames,
    validate_table_frame,
)
from datalens.data.records import add_record_id, add_record_ids

TableName = Literal["vendor", "transaction"]


@dataclass(frozen=True)
class ScoringResult:
    """Validated records and their ranked deterministic findings."""

    vendors: pd.DataFrame
    transactions: pd.DataFrame
    findings: pd.DataFrame

    @property
    def summary(self) -> dict[str, object]:
        severities = (
            self.findings["severity"].value_counts().sort_index().astype(int).to_dict()
            if not self.findings.empty
            else {}
        )
        return {
            "vendor_records": len(self.vendors),
            "transaction_records": len(self.transactions),
            "finding_count": len(self.findings),
            "findings_by_severity": severities,
        }


class ScoringService:
    """Score procurement inputs using the approved schema and deterministic rules."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config

    def score_batch(
        self,
        vendors: pd.DataFrame,
        transactions: pd.DataFrame,
        *,
        fiscal_year: int,
    ) -> ScoringResult:
        validated_vendors, validated_transactions = validate_procurement_frames(
            vendors,
            transactions,
            schema=self._config.schema,
        )
        identified_vendors, identified_transactions = add_record_ids(
            validated_vendors,
            validated_transactions,
        )
        findings = run_rules(
            identified_vendors,
            identified_transactions,
            fiscal_year=fiscal_year,
            scoring_weights=self._config.schema.scoring_weights,
        )
        return ScoringResult(
            vendors=identified_vendors,
            transactions=identified_transactions,
            findings=findings,
        )

    def score_record(
        self,
        record: pd.DataFrame,
        *,
        table: TableName,
        fiscal_year: int,
    ) -> ScoringResult:
        validated = validate_table_frame(
            record,
            schema=self._config.schema,
            table_name=table,
        )
        identified = add_record_id(validated, table_name=table)
        empty = pd.DataFrame()
        vendors, transactions = (identified, empty) if table == "vendor" else (empty, identified)
        findings = run_rules(
            vendors,
            transactions,
            fiscal_year=fiscal_year,
            scoring_weights=self._config.schema.scoring_weights,
            target_tables=(table,),
            include_relationship_rules=False,
        )
        return ScoringResult(
            vendors=vendors,
            transactions=transactions,
            findings=findings,
        )
