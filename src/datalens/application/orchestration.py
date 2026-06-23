"""Atomic application orchestration for scoring and persistence."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from datalens.application.persistence import (
    FindingRecord,
    Repository,
    ScoringRunRecord,
)
from datalens.application.scoring import ScoringResult, ScoringService
from datalens.configuration.loader import RuntimeConfig
from datalens.modeling.reranker import RerankerStore

TableName = Literal["vendor", "transaction"]


class ScoringCoordinator:
    """Keep validation, scoring, and persistence in one application-owned flow."""

    def __init__(
        self,
        config: RuntimeConfig,
        scoring: ScoringService,
        repository: Repository,
        reranker_store: RerankerStore,
    ) -> None:
        self._config = config
        self._scoring = scoring
        self._repository = repository
        self._reranker_store = reranker_store

    def score_batch(
        self,
        vendors: pd.DataFrame,
        transactions: pd.DataFrame,
        *,
        fiscal_year: int,
    ) -> tuple[ScoringRunRecord, list[FindingRecord]]:
        result = self._scoring.score_batch(
            vendors,
            transactions,
            fiscal_year=fiscal_year,
        )
        return self._persist(result, fiscal_year=fiscal_year)

    def score_record(
        self,
        record: pd.DataFrame,
        *,
        table: TableName,
        fiscal_year: int,
    ) -> tuple[ScoringRunRecord, list[FindingRecord]]:
        result = self._scoring.score_record(
            record,
            table=table,
            fiscal_year=fiscal_year,
        )
        return self._persist(result, fiscal_year=fiscal_year)

    def _persist(
        self,
        result: ScoringResult,
        *,
        fiscal_year: int,
    ) -> tuple[ScoringRunRecord, list[FindingRecord]]:
        ranked_result = ScoringResult(
            vendors=result.vendors,
            transactions=result.transactions,
            findings=self._reranker_store.rank(result.findings),
        )
        run = self._repository.save_scoring_run(
            ranked_result,
            fiscal_year=fiscal_year,
            schema_version=self._config.schema.schema_version,
            model_version=(
                self._reranker_store.active_version() or self._config.model.active_model_version
            ),
        )
        return run, self._repository.list_findings(run.id)
