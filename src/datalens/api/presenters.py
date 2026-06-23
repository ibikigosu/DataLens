"""Response mapping kept separate from HTTP routes and ORM records."""

from __future__ import annotations

import json

from datalens.api.contracts import (
    FeedbackResponse,
    FindingResponse,
    ScoringRunResponse,
    ScoringRunSummaryResponse,
)
from datalens.application.persistence import (
    FeedbackRecord,
    FindingRecord,
    ScoringRunRecord,
)


def finding_response(record: FindingRecord) -> FindingResponse:
    return FindingResponse(
        finding_id=record.id,
        run_id=record.run_id,
        target_table=record.target_table,
        record_id=record.record_id,
        issue_type=record.issue_type,
        severity=record.severity,
        risk_score=record.risk_score,
        review_priority=record.review_priority,
        model_confidence=record.model_confidence,
        evidence=record.evidence,
    )


def run_summary_response(record: ScoringRunRecord) -> ScoringRunSummaryResponse:
    return ScoringRunSummaryResponse(
        run_id=record.id,
        created_at=record.created_at,
        fiscal_year=record.fiscal_year,
        schema_version=record.schema_version,
        model_version=record.model_version,
        summary=json.loads(record.summary_json),
    )


def run_response(
    record: ScoringRunRecord,
    findings: list[FindingRecord],
) -> ScoringRunResponse:
    summary = run_summary_response(record)
    return ScoringRunResponse(
        **summary.model_dump(),
        findings=[finding_response(finding) for finding in findings],
    )


def feedback_response(record: FeedbackRecord) -> FeedbackResponse:
    return FeedbackResponse(
        feedback_id=record.id,
        finding_id=record.finding_id,
        verdict=record.verdict,
        corrected_issue_type=record.corrected_issue_type,
        notes=record.notes,
        created_at=record.created_at,
    )
