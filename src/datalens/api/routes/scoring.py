"""Single-record and paired CSV scoring routes."""

from __future__ import annotations

from io import BytesIO
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from datalens.api.contracts import (
    ScoringRunResponse,
    TransactionScoreRequest,
    VendorScoreRequest,
)
from datalens.api.dependencies import AppServices, Services
from datalens.api.presenters import run_response

router = APIRouter(prefix="/score", tags=["scoring"])
FiscalYearForm = Annotated[int, Form(ge=2000, le=2100)]
CsvUpload = Annotated[UploadFile, File()]


@router.post("/vendor", response_model=ScoringRunResponse, status_code=status.HTTP_201_CREATED)
def score_vendor(
    request: VendorScoreRequest,
    services: Services,
) -> ScoringRunResponse:
    return _score_single(
        pd.DataFrame([request.record.model_dump()]),
        table="vendor",
        fiscal_year=request.fiscal_year,
        services=services,
    )


@router.post(
    "/transaction",
    response_model=ScoringRunResponse,
    status_code=status.HTTP_201_CREATED,
)
def score_transaction(
    request: TransactionScoreRequest,
    services: Services,
) -> ScoringRunResponse:
    return _score_single(
        pd.DataFrame([request.record.model_dump()]),
        table="transaction",
        fiscal_year=request.fiscal_year,
        services=services,
    )


@router.post("/batch", response_model=ScoringRunResponse, status_code=status.HTTP_201_CREATED)
async def score_batch(
    fiscal_year: FiscalYearForm,
    vendors: CsvUpload,
    transactions: CsvUpload,
    services: Services,
) -> ScoringRunResponse:
    try:
        vendor_frame = await _read_csv(vendors, services.config.settings.maximum_upload_bytes)
        transaction_frame = await _read_csv(
            transactions,
            services.config.settings.maximum_upload_bytes,
        )
        run, findings = services.scoring_coordinator.score_batch(
            vendor_frame,
            transaction_frame,
            fiscal_year=fiscal_year,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return run_response(run, findings)


def _score_single(
    frame: pd.DataFrame,
    *,
    table: str,
    fiscal_year: int,
    services: AppServices,
) -> ScoringRunResponse:
    try:
        run, findings = services.scoring_coordinator.score_record(
            frame,
            table=table,
            fiscal_year=fiscal_year,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return run_response(run, findings)


async def _read_csv(upload: UploadFile, maximum_bytes: int) -> pd.DataFrame:
    if not upload.filename or not upload.filename.lower().endswith(".csv"):
        raise ValueError("Uploads must be CSV files")
    payload = await upload.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError(f"{upload.filename} exceeds the configured upload limit")
    try:
        return pd.read_csv(BytesIO(payload))
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ValueError(f"{upload.filename} is not a valid UTF-8 CSV file") from error
