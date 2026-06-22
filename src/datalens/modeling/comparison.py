"""Development-only model comparison, reranking, and locked selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from datalens.features.builders import build_transaction_features, build_vendor_features
from datalens.features.dataset import DevelopmentFeatureDataset
from datalens.features.pipeline import FeatureTable
from datalens.modeling.alternatives import train_one_class_svms
from datalens.modeling.evaluation import RankingEvaluation, evaluate_record_ranking
from datalens.modeling.models import train_isolation_forests
from datalens.modeling.promotion import PromotionDecision, assess_promotion
from datalens.modeling.reranking import (
    FalseAlarmReranker,
    build_finding_features,
    build_guarded_review_queue,
    simulated_feedback,
    train_false_alarm_reranker,
)
from datalens.modeling.workflow import EvaluationPeriod


class ScoreableTableModel(Protocol):
    def score(self, feature_frame: pd.DataFrame) -> pd.DataFrame: ...


class ScoreableModels(Protocol):
    def for_table(self, table: FeatureTable) -> ScoreableTableModel: ...


@dataclass(frozen=True)
class ComparisonResult:
    table_winners: dict[str, str]
    model_metrics: dict[str, RankingEvaluation]
    selected_metrics: RankingEvaluation
    reranked_metrics: RankingEvaluation
    guarded_metrics: RankingEvaluation
    promotion: PromotionDecision
    selected_scores: pd.DataFrame
    reranked_findings: pd.DataFrame
    guarded_queue: pd.DataFrame
    candidates: dict[str, ScoreableModels]
    reranker: FalseAlarmReranker

    def summary(self) -> dict[str, object]:
        return {
            "table_winners": self.table_winners,
            "model_metrics": {
                name: metrics.as_dict() for name, metrics in self.model_metrics.items()
            },
            "selected_metrics": self.selected_metrics.as_dict(),
            "reranked_metrics": self.reranked_metrics.as_dict(),
            "guarded_metrics": self.guarded_metrics.as_dict(),
            "promotion": self.promotion.as_dict(),
        }


def compare_development_models(
    dataset: DevelopmentFeatureDataset,
    period: EvaluationPeriod,
) -> ComparisonResult:
    """Compare two anomaly approaches and train one issue-level reranker."""
    candidates: dict[str, ScoreableModels] = {
        "isolation_forest": train_isolation_forests(dataset),
        "one_class_svm": train_one_class_svms(dataset),
    }
    scores = {name: _score_models(models, period) for name, models in candidates.items()}
    metrics = {
        name: evaluate_record_ranking(
            period.labels,
            candidate_scores,
            evaluated_records_by_table=period.record_counts,
        )
        for name, candidate_scores in scores.items()
    }
    table_winners = _select_table_winners(metrics)
    selected_scores = _combine_table_winners(scores, table_winners)
    selected_metrics = evaluate_record_ranking(
        period.labels,
        selected_scores,
        evaluated_records_by_table=period.record_counts,
    )
    finding_features = build_finding_features(
        period.deterministic_findings,
        selected_scores,
    )
    feedback = simulated_feedback(period.deterministic_findings, period.labels)
    reranker = train_false_alarm_reranker(finding_features, feedback)
    reranked_findings = reranker.rerank(finding_features)
    reranked_records = (
        reranked_findings.groupby(["target_table", "record_id"], as_index=False)
        .agg(priority_score=("priority_score", "max"))
        .assign(predicted=True)
    )
    reranked_metrics = evaluate_record_ranking(
        period.labels,
        reranked_records,
        evaluated_records_by_table=period.record_counts,
    )
    guarded_queue = build_guarded_review_queue(
        reranked_findings,
        selected_scores,
    )
    guarded_metrics = evaluate_record_ranking(
        period.labels,
        guarded_queue,
        evaluated_records_by_table=period.record_counts,
    )
    baseline_metrics = _baseline_metrics(period)
    return ComparisonResult(
        table_winners=table_winners,
        model_metrics=metrics,
        selected_metrics=selected_metrics,
        reranked_metrics=reranked_metrics,
        guarded_metrics=guarded_metrics,
        promotion=assess_promotion(
            selected_metrics,
            baseline_metrics,
            guarded_metrics,
            selection_period_role=period.role,
        ),
        selected_scores=selected_scores,
        reranked_findings=reranked_findings,
        guarded_queue=guarded_queue,
        candidates=candidates,
        reranker=reranker,
    )


def persist_comparison(result: ComparisonResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "comparison-summary.json",
        "scores": output_dir / "selected-scores.parquet",
        "reranked_findings": output_dir / "reranked-findings.parquet",
        "guarded_queue": output_dir / "guarded-review-queue.parquet",
    }
    paths["summary"].write_text(
        json.dumps(result.summary(), indent=2) + "\n",
        encoding="utf-8",
    )
    result.selected_scores.to_parquet(paths["scores"], index=False)
    result.reranked_findings.to_parquet(paths["reranked_findings"], index=False)
    result.guarded_queue.to_parquet(paths["guarded_queue"], index=False)
    return paths


def _score_models(
    models: ScoreableModels,
    period: EvaluationPeriod,
) -> pd.DataFrame:
    frames = {
        FeatureTable.VENDOR: build_vendor_features(period.vendors),
        FeatureTable.TRANSACTION: build_transaction_features(period.transactions),
    }
    return pd.concat(
        [models.for_table(table).score(frames[table]) for table in FeatureTable],
        ignore_index=True,
    ).sort_values(
        ["priority_score", "target_table", "record_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def _select_table_winners(
    metrics: dict[str, RankingEvaluation],
) -> dict[str, str]:
    winners = {}
    for table in FeatureTable:
        table_name = table.value
        winners[table_name] = max(
            sorted(metrics),
            key=lambda name: (
                metrics[name].per_table[table_name].top_k_precision,
                metrics[name].per_table[table_name].classification.f1,
                metrics[name].per_table[table_name].classification.precision,
                -metrics[name].per_table[table_name].classification.false_positives,
                name,
            ),
        )
    return winners


def _combine_table_winners(
    scores: dict[str, pd.DataFrame],
    winners: dict[str, str],
) -> pd.DataFrame:
    return pd.concat(
        [
            scores[model_name].loc[scores[model_name]["target_table"].eq(table)]
            for table, model_name in winners.items()
        ],
        ignore_index=True,
    ).sort_values(
        ["priority_score", "target_table", "record_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def _baseline_metrics(period: EvaluationPeriod) -> RankingEvaluation:
    ranking = (
        period.deterministic_findings.groupby(
            ["target_table", "record_id"],
            as_index=False,
        )
        .agg(priority_score=("risk_score", "max"))
        .assign(predicted=True)
    )
    return evaluate_record_ranking(
        period.labels,
        ranking,
        evaluated_records_by_table=period.record_counts,
    )
