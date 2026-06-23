"""Typed public API request and response contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VendorRecord(ApiModel):
    vendor_id: str = Field(min_length=1)
    recipient_name: str | None = None
    recipient_uei: str | None = None
    recipient_country_code: str | None = None
    recipient_state_code: str | None = None
    source_transaction_count: int = Field(ge=0)
    address_variant_count: int = Field(ge=0)
    contracting_officers_determination_of_business_size_code: str | None = None


class TransactionRecord(ApiModel):
    contract_transaction_unique_key: str = Field(min_length=1)
    vendor_id: str = Field(min_length=1)
    federal_action_obligation: float | None = None
    total_dollars_obligated: float | None = None
    number_of_offers_received: int | None = None
    action_date: datetime
    period_of_performance_start_date: datetime | None = None
    period_of_performance_current_end_date: datetime | None = None
    award_type_code: str | None = None
    type_of_contract_pricing_code: str | None = None
    action_type_code: str | None = None
    product_or_service_code: str | None = None
    naics_code: str | None = None
    extent_competed_code: str | None = None
    solicitation_procedures_code: str | None = None
    type_of_set_aside_code: str | None = None


class VendorScoreRequest(ApiModel):
    fiscal_year: int = Field(ge=2000, le=2100)
    record: VendorRecord


class TransactionScoreRequest(ApiModel):
    fiscal_year: int = Field(ge=2000, le=2100)
    record: TransactionRecord


class FindingResponse(ApiModel):
    finding_id: str
    run_id: str
    target_table: str
    record_id: str
    issue_type: str
    severity: str
    risk_score: float
    review_priority: float
    model_confidence: float | None
    evidence: str


class ScoringRunResponse(ApiModel):
    run_id: str
    created_at: datetime
    fiscal_year: int
    schema_version: str
    model_version: str
    summary: dict[str, object]
    findings: list[FindingResponse]


class ScoringRunSummaryResponse(ApiModel):
    run_id: str
    created_at: datetime
    fiscal_year: int
    schema_version: str
    model_version: str
    summary: dict[str, object]


class FeedbackVerdict(StrEnum):
    CORRECT_FLAG = "correct_flag"
    FALSE_ALARM = "false_alarm"
    WRONG_ISSUE_TYPE = "wrong_issue_type"
    MISSED_ISSUE = "missed_issue"
    UNSURE = "unsure"


class FeedbackRequest(ApiModel):
    verdict: FeedbackVerdict
    corrected_issue_type: str | None = None
    notes: str | None = Field(default=None, max_length=2_000)


class FeedbackResponse(ApiModel):
    feedback_id: str
    finding_id: str
    verdict: FeedbackVerdict
    corrected_issue_type: str | None
    notes: str | None
    created_at: datetime


class FeedbackBatchItem(ApiModel):
    finding_id: str
    verdict: FeedbackVerdict
    corrected_issue_type: str | None = None
    notes: str | None = Field(default=None, max_length=2_000)


class FeedbackBatchRequest(ApiModel):
    feedback: list[FeedbackBatchItem] = Field(min_length=1, max_length=2_000)


class FeedbackBatchResponse(ApiModel):
    run_id: str
    saved_feedback: int


class HealthResponse(ApiModel):
    status: str
    environment: str


class RetrainingResponse(ApiModel):
    retraining_run_id: str
    active_model_version: str
    candidate_model_version: str
    promoted: bool
    promotion: dict[str, object]
    active: dict[str, object]
    candidate: dict[str, object]
    training_examples: int
    validation_examples: int
    evaluation_top_k: int
