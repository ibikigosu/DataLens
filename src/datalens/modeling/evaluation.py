"""Record-level comparison between anomaly rankings and controlled defects."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

RecordKey = tuple[str, str]


@dataclass(frozen=True)
class ClassificationMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class TableEvaluation:
    classification: ClassificationMetrics
    top_k_precision: float
    false_alarms_per_1000_records: float


@dataclass(frozen=True)
class RankingEvaluation:
    evaluated_records: int
    controlled_defect_records: int
    predicted_records: int
    classification: ClassificationMetrics
    macro_f1: float
    high_critical_recall: float
    top_k_precision: float
    false_alarms_per_1000_records: float
    per_table: dict[str, TableEvaluation]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_record_ranking(
    labels: pd.DataFrame,
    ranking: pd.DataFrame,
    *,
    evaluated_records_by_table: dict[str, int],
    top_k: int = 50,
) -> RankingEvaluation:
    """Evaluate shared record-level ranking metrics with correct table denominators."""
    truth = _record_keys(labels)
    predicted = _record_keys(ranking.loc[ranking["predicted"]])
    classification = _classification_metrics(truth, predicted)
    deduplicated = ranking.sort_values(
        ["priority_score", "target_table", "record_id"],
        ascending=[False, True, True],
    ).drop_duplicates(["target_table", "record_id"])
    top_records = _ordered_keys(deduplicated.head(top_k))
    per_table = {
        table: _evaluate_table(
            table,
            truth=truth,
            predicted=predicted,
            ranking=deduplicated,
            evaluated_records=evaluated_records,
            top_k=top_k,
        )
        for table, evaluated_records in sorted(evaluated_records_by_table.items())
    }
    high_critical_truth = _record_keys(labels.loc[labels["severity"].isin(["high", "critical"])])
    evaluated_records = sum(evaluated_records_by_table.values())
    return RankingEvaluation(
        evaluated_records=evaluated_records,
        controlled_defect_records=len(truth),
        predicted_records=len(predicted),
        classification=classification,
        macro_f1=sum(item.classification.f1 for item in per_table.values()) / len(per_table),
        high_critical_recall=(
            len(high_critical_truth & predicted) / len(high_critical_truth)
            if high_critical_truth
            else 0.0
        ),
        top_k_precision=_precision_at_k(top_records, truth),
        false_alarms_per_1000_records=(
            classification.false_positives / evaluated_records * 1_000 if evaluated_records else 0.0
        ),
        per_table=per_table,
    )


def _evaluate_table(
    table: str,
    *,
    truth: set[RecordKey],
    predicted: set[RecordKey],
    ranking: pd.DataFrame,
    evaluated_records: int,
    top_k: int,
) -> TableEvaluation:
    table_truth = {key for key in truth if key[0] == table}
    table_predicted = {key for key in predicted if key[0] == table}
    classification = _classification_metrics(table_truth, table_predicted)
    table_top = _ordered_keys(ranking.loc[ranking["target_table"].eq(table)].head(top_k))
    return TableEvaluation(
        classification=classification,
        top_k_precision=_precision_at_k(table_top, table_truth),
        false_alarms_per_1000_records=(
            classification.false_positives / evaluated_records * 1_000 if evaluated_records else 0.0
        ),
    )


def _classification_metrics(
    truth: set[RecordKey],
    predicted: set[RecordKey],
) -> ClassificationMetrics:
    true_positives = len(truth & predicted)
    false_positives = len(predicted - truth)
    false_negatives = len(truth - predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(truth) if truth else 0.0
    return ClassificationMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=(2 * precision * recall / (precision + recall) if precision + recall else 0.0),
    )


def _record_keys(frame: pd.DataFrame) -> set[RecordKey]:
    return set(
        frame[["target_table", "record_id"]].astype("string").itertuples(index=False, name=None)
    )


def _ordered_keys(frame: pd.DataFrame) -> list[RecordKey]:
    return list(
        frame[["target_table", "record_id"]].astype("string").itertuples(index=False, name=None)
    )


def _precision_at_k(records: list[RecordKey], truth: set[RecordKey]) -> float:
    return sum(record in truth for record in records) / len(records) if records else 0.0
