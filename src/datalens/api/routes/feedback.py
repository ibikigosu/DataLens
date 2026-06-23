"""Reviewer feedback route."""

from fastapi import APIRouter, HTTPException, status

from datalens.api.contracts import (
    FeedbackBatchRequest,
    FeedbackBatchResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from datalens.api.dependencies import Services
from datalens.api.presenters import feedback_response

router = APIRouter(tags=["feedback"])


@router.post(
    "/findings/{finding_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_feedback(
    finding_id: str,
    request: FeedbackRequest,
    services: Services,
) -> FeedbackResponse:
    feedback = services.repository.add_feedback(
        finding_id,
        verdict=request.verdict.value,
        corrected_issue_type=request.corrected_issue_type,
        notes=request.notes,
    )
    if feedback is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    return feedback_response(feedback)


@router.post(
    "/runs/{run_id}/feedback/batch",
    response_model=FeedbackBatchResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_feedback_batch(
    run_id: str,
    request: FeedbackBatchRequest,
    services: Services,
) -> FeedbackBatchResponse:
    records = services.repository.add_feedback_batch(
        run_id,
        [
            {
                "finding_id": item.finding_id,
                "verdict": item.verdict.value,
                "corrected_issue_type": item.corrected_issue_type,
                "notes": item.notes,
            }
            for item in request.feedback
        ],
    )
    if records is None:
        raise HTTPException(
            status_code=404,
            detail="One or more findings do not belong to the scoring run",
        )
    return FeedbackBatchResponse(run_id=run_id, saved_feedback=len(records))
