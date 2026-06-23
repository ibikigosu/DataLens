"""Persisted scoring run and finding retrieval routes."""

from __future__ import annotations

import csv
from io import StringIO

from fastapi import APIRouter, HTTPException, Response

from datalens.api.contracts import FindingResponse, ScoringRunSummaryResponse
from datalens.api.dependencies import AppServices, Services
from datalens.api.presenters import finding_response, run_summary_response

router = APIRouter(tags=["runs"])


@router.get("/runs/{run_id}", response_model=ScoringRunSummaryResponse)
def get_run(
    run_id: str,
    services: Services,
) -> ScoringRunSummaryResponse:
    run = services.repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Scoring run not found")
    return run_summary_response(run)


@router.get("/runs/{run_id}/findings", response_model=list[FindingResponse])
def get_findings(
    run_id: str,
    services: Services,
) -> list[FindingResponse]:
    _require_run(run_id, services)
    return [finding_response(finding) for finding in services.repository.list_findings(run_id)]


@router.get("/runs/{run_id}/findings.csv")
def download_findings(
    run_id: str,
    services: Services,
) -> Response:
    _require_run(run_id, services)
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "finding_id",
            "run_id",
            "target_table",
            "record_id",
            "issue_type",
            "severity",
            "risk_score",
            "evidence",
        ]
    )
    for finding in services.repository.list_findings(run_id):
        writer.writerow(
            [
                finding.id,
                finding.run_id,
                finding.target_table,
                finding.record_id,
                finding.issue_type,
                finding.severity,
                finding.risk_score,
                finding.evidence,
            ]
        )
    return Response(
        output.getvalue(),
        media_type="text/csv",
        headers={"content-disposition": f'attachment; filename="datalens-{run_id}-findings.csv"'},
    )


def _require_run(run_id: str, services: AppServices) -> None:
    if services.repository.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Scoring run not found")
