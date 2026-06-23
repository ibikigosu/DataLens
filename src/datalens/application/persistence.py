"""Transactional persistence for scoring runs, findings, feedback, and retraining."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pandas as pd
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from datalens.application.scoring import ScoringResult


class Base(DeclarativeBase):
    pass


class ScoringRunRecord(Base):
    __tablename__ = "scoring_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    vendor_records: Mapped[int] = mapped_column(Integer, nullable=False)
    transaction_records: Mapped[int] = mapped_column(Integer, nullable=False)
    finding_count: Mapped[int] = mapped_column(Integer, nullable=False)
    summary_json: Mapped[str] = mapped_column(Text, nullable=False)
    findings: Mapped[list[FindingRecord]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class FindingRecord(Base):
    __tablename__ = "findings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("scoring_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    target_table: Mapped[str] = mapped_column(String(32), nullable=False)
    record_id: Mapped[str] = mapped_column(String(128), nullable=False)
    issue_type: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    review_priority: Mapped[float] = mapped_column(Float, nullable=False)
    model_confidence: Mapped[float | None] = mapped_column(Float)
    evidence: Mapped[str] = mapped_column(Text, nullable=False)
    run: Mapped[ScoringRunRecord] = relationship(back_populates="findings")
    feedback: Mapped[list[FeedbackRecord]] = relationship(
        back_populates="finding",
        cascade="all, delete-orphan",
    )


class FeedbackRecord(Base):
    __tablename__ = "review_feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    finding_id: Mapped[str] = mapped_column(
        ForeignKey("findings.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    corrected_issue_type: Mapped[str | None] = mapped_column(String(128))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finding: Mapped[FindingRecord] = relationship(back_populates="feedback")


class RetrainingRecord(Base):
    __tablename__ = "retraining_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    active_model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)


class Repository:
    """Small transactional repository shared by API services."""

    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self._engine = create_engine(
            database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        self._sessions = sessionmaker(self._engine, expire_on_commit=False)

    def initialize(self) -> None:
        Base.metadata.create_all(self._engine)
        self._add_finding_ranking_columns()

    def ready(self) -> bool:
        try:
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_scoring_run(
        self,
        result: ScoringResult,
        *,
        fiscal_year: int,
        schema_version: str,
        model_version: str,
    ) -> ScoringRunRecord:
        run_id = str(uuid4())
        run = ScoringRunRecord(
            id=run_id,
            created_at=datetime.now(UTC),
            fiscal_year=fiscal_year,
            schema_version=schema_version,
            model_version=model_version,
            vendor_records=len(result.vendors),
            transaction_records=len(result.transactions),
            finding_count=len(result.findings),
            summary_json=json.dumps(result.summary, sort_keys=True),
        )
        for finding in result.findings.itertuples(index=False):
            run.findings.append(
                FindingRecord(
                    id=str(uuid4()),
                    run_id=run_id,
                    target_table=str(finding.target_table),
                    record_id=str(finding.record_id),
                    issue_type=str(finding.issue_type),
                    severity=str(finding.severity),
                    risk_score=float(finding.risk_score),
                    review_priority=float(getattr(finding, "review_priority", finding.risk_score)),
                    model_confidence=_optional_float(getattr(finding, "model_confidence", None)),
                    evidence=str(finding.evidence),
                )
            )
        with self.session() as session:
            session.add(run)
        return run

    def get_run(self, run_id: str) -> ScoringRunRecord | None:
        with self.session() as session:
            return session.get(ScoringRunRecord, run_id)

    def list_findings(self, run_id: str) -> list[FindingRecord]:
        with self.session() as session:
            return list(
                session.scalars(
                    select(FindingRecord)
                    .where(FindingRecord.run_id == run_id)
                    .order_by(
                        FindingRecord.review_priority.desc(),
                        FindingRecord.target_table,
                        FindingRecord.record_id,
                        FindingRecord.issue_type,
                    )
                )
            )

    def add_feedback(
        self,
        finding_id: str,
        *,
        verdict: str,
        corrected_issue_type: str | None,
        notes: str | None,
    ) -> FeedbackRecord | None:
        with self.session() as session:
            finding = session.get(FindingRecord, finding_id)
            if finding is None:
                return None
            feedback = FeedbackRecord(
                id=str(uuid4()),
                finding_id=finding_id,
                verdict=verdict,
                corrected_issue_type=corrected_issue_type,
                notes=notes,
                created_at=datetime.now(UTC),
            )
            session.add(feedback)
        return feedback

    def add_feedback_batch(
        self,
        run_id: str,
        feedback_items: list[dict[str, str | None]],
    ) -> list[FeedbackRecord] | None:
        """Atomically retain a reviewed batch only when every finding belongs to the run."""
        finding_ids = [str(item["finding_id"]) for item in feedback_items]
        with self.session() as session:
            findings = {
                finding.id: finding
                for finding in session.scalars(
                    select(FindingRecord).where(
                        FindingRecord.run_id == run_id,
                        FindingRecord.id.in_(finding_ids),
                    )
                )
            }
            if set(findings) != set(finding_ids):
                return None
            records = [
                FeedbackRecord(
                    id=str(uuid4()),
                    finding_id=finding_id,
                    verdict=str(item["verdict"]),
                    corrected_issue_type=item.get("corrected_issue_type"),
                    notes=item.get("notes"),
                    created_at=datetime.now(UTC),
                )
                for item in feedback_items
                for finding_id in [str(item["finding_id"])]
            ]
            session.add_all(records)
        return records

    def feedback_examples(self) -> list[dict[str, Any]]:
        """Return issue-level feedback joined to the finding features it reviewed."""
        with self.session() as session:
            rows = session.execute(
                select(FindingRecord, FeedbackRecord)
                .join(FeedbackRecord, FeedbackRecord.finding_id == FindingRecord.id)
                .order_by(FeedbackRecord.created_at, FeedbackRecord.id)
            ).all()
        return [
            {
                "finding_id": finding.id,
                "target_table": finding.target_table,
                "issue_type": finding.issue_type,
                "severity": finding.severity,
                "risk_score": finding.risk_score,
                "verdict": feedback.verdict,
                "feedback_created_at": feedback.created_at,
            }
            for finding, feedback in rows
        ]

    def save_retraining_run(
        self,
        *,
        active_model_version: str,
        candidate_model_version: str,
        result: dict[str, Any],
    ) -> RetrainingRecord:
        record = RetrainingRecord(
            id=str(uuid4()),
            created_at=datetime.now(UTC),
            status="completed",
            active_model_version=active_model_version,
            candidate_model_version=candidate_model_version,
            promoted=bool(result["promotion"]["promoted"]),
            result_json=json.dumps(result, sort_keys=True),
        )
        with self.session() as session:
            session.add(record)
        return record

    def _add_finding_ranking_columns(self) -> None:
        columns = {column["name"] for column in inspect(self._engine).get_columns("findings")}
        statements = []
        if "review_priority" not in columns:
            statements.append("ALTER TABLE findings ADD COLUMN review_priority FLOAT")
        if "model_confidence" not in columns:
            statements.append("ALTER TABLE findings ADD COLUMN model_confidence FLOAT")
        if not statements:
            return
        with self._engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))
            connection.execute(
                text(
                    "UPDATE findings SET review_priority = risk_score WHERE review_priority IS NULL"
                )
            )


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
