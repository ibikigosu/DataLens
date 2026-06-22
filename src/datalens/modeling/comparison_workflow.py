"""Locked Milestone 6 comparison and temporal evaluation workflows."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from datalens.baseline.run import BaselinePlan, load_config
from datalens.data.plan import DatasetPlan, load_dataset_plan
from datalens.features.builders import build_transaction_features, build_vendor_features
from datalens.features.dataset import DevelopmentFeatureDataset
from datalens.features.pipeline import FeatureTable
from datalens.features.schemas import TRANSACTION_FEATURE_SCHEMA, VENDOR_FEATURE_SCHEMA
from datalens.modeling.alternatives import (
    OneClassSvmModels,
    load_one_class_svm,
    save_one_class_svm,
)
from datalens.modeling.comparison import (
    ComparisonResult,
    compare_development_models,
    persist_comparison,
)
from datalens.modeling.evaluation import evaluate_record_ranking
from datalens.modeling.models import (
    TrainedModels,
    load_trained_model,
    save_trained_model,
)
from datalens.modeling.promotion import assess_promotion
from datalens.modeling.reranking import (
    build_finding_features,
    build_guarded_review_queue,
    load_false_alarm_reranker,
    save_false_alarm_reranker,
)
from datalens.modeling.tracking import ExperimentTracker
from datalens.modeling.workflow import (
    EvaluationPeriod,
    baseline_record_evaluation,
    load_clean_records,
    prepare_evaluation_period,
)
from datalens.paths import ARTIFACTS_DIR

DEFAULT_COMPARISON_DIR = ARTIFACTS_DIR / "modeling" / "milestone-06"
COMPARISON_EXPERIMENT_NAME = "datalens-model-comparison"
COMPARISON_LOCK_FILENAME = "comparison-lock.json"


def run_development_comparison(
    tracker: ExperimentTracker,
    *,
    output_dir: Path = DEFAULT_COMPARISON_DIR,
    dataset_plan: DatasetPlan | None = None,
    baseline_plan: BaselinePlan | None = None,
) -> dict[str, object]:
    plan = dataset_plan or load_dataset_plan()
    defects = baseline_plan or load_config()
    periods = [period for period in plan.periods if period.role == "development"]
    if len(periods) != 1:
        raise ValueError("Exactly one development period is required")
    period = periods[0]
    clean_vendors, clean_transactions = load_clean_records(period.fiscal_year)
    dataset = DevelopmentFeatureDataset.from_records(
        clean_vendors,
        clean_transactions,
        fiscal_year=period.fiscal_year,
        dataset_plan=plan,
    )
    evaluation_period = prepare_evaluation_period(
        period.fiscal_year,
        role=period.role,
        clean_vendors=clean_vendors,
        clean_transactions=clean_transactions,
        defects=defects,
    )
    result = compare_development_models(dataset, evaluation_period)
    paths = persist_comparison(result, output_dir)
    package_paths = _persist_candidates(result, output_dir)
    reranker_path = output_dir / "reranker" / "false-alarm-reranker.skops"
    save_false_alarm_reranker(result.reranker, reranker_path)
    run_ids = {
        name: tracker.log_run(
            run_name=f"{name}-fy{period.fiscal_year}",
            params={
                "model_family": name,
                "selection_period": period.fiscal_year,
                "selection_period_role": period.role,
            },
            metrics=_model_metrics(result, name),
            artifacts=tuple(package_paths[name].values()),
        )
        for name in sorted(result.model_metrics)
    }
    selection_run_id = tracker.log_run(
        run_name=f"selection-and-reranking-fy{period.fiscal_year}",
        params={
            "vendor_winner": result.table_winners["vendor"],
            "transaction_winner": result.table_winners["transaction"],
            "promoted": result.promotion.promoted,
            "selection_period_role": period.role,
        },
        metrics={
            "selected_top_50_precision": result.selected_metrics.top_k_precision,
            "reranked_top_50_precision": result.reranked_metrics.top_k_precision,
            "guarded_high_critical_recall": result.guarded_metrics.high_critical_recall,
        },
        artifacts=(*paths.values(), reranker_path),
    )
    lock = {
        "schema_version": 1,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "development_fiscal_year": period.fiscal_year,
        "development_role": period.role,
        "table_winners": result.table_winners,
        "candidate_run_ids": run_ids,
        "selection_run_id": selection_run_id,
        "promotion": result.promotion.as_dict(),
        "packages": {
            model_name: {
                table.value: {
                    "path": str(
                        package_paths[model_name][f"{table.value}_estimator"].parent.resolve()
                    ),
                    "sha256": _directory_digest(
                        package_paths[model_name][f"{table.value}_estimator"].parent
                    ),
                }
                for table in FeatureTable
            }
            for model_name in package_paths
        },
        "reranker": {
            "path": str(reranker_path.resolve()),
            "sha256": _file_digest(reranker_path),
        },
    }
    lock_path = output_dir / COMPARISON_LOCK_FILENAME
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"lock_path": str(lock_path), "summary": result.summary()}


def evaluate_locked_comparison(
    lock_path: Path,
    tracker: ExperimentTracker,
    *,
    output_dir: Path = DEFAULT_COMPARISON_DIR,
    dataset_plan: DatasetPlan | None = None,
    baseline_plan: BaselinePlan | None = None,
) -> dict[str, object]:
    plan = dataset_plan or load_dataset_plan()
    defects = baseline_plan or load_config()
    holdouts = [period for period in plan.periods if period.role == "temporal_holdout"]
    if len(holdouts) != 1:
        raise ValueError("Exactly one temporal holdout is required")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    period = holdouts[0]
    clean_vendors, clean_transactions = load_clean_records(period.fiscal_year)
    evaluation_period = prepare_evaluation_period(
        period.fiscal_year,
        role=period.role,
        clean_vendors=clean_vendors,
        clean_transactions=clean_transactions,
        defects=defects,
    )
    selected_scores = _score_locked_winners(lock, evaluation_period)
    selected_metrics = evaluate_record_ranking(
        evaluation_period.labels,
        selected_scores,
        evaluated_records_by_table=evaluation_period.record_counts,
    )
    reranker_info = lock["reranker"]
    reranker_path = Path(str(reranker_info["path"]))
    if _file_digest(reranker_path) != reranker_info["sha256"]:
        raise ValueError("Locked false-alarm reranker has changed")
    reranker = load_false_alarm_reranker(reranker_path)
    finding_features = build_finding_features(
        evaluation_period.deterministic_findings,
        selected_scores,
    )
    reranked_findings = reranker.rerank(finding_features)
    guarded_queue = build_guarded_review_queue(reranked_findings, selected_scores)
    guarded_metrics = evaluate_record_ranking(
        evaluation_period.labels,
        guarded_queue,
        evaluated_records_by_table=evaluation_period.record_counts,
    )
    baseline_metrics = baseline_record_evaluation(evaluation_period)
    promotion = assess_promotion(
        selected_metrics,
        baseline_metrics,
        guarded_metrics,
        selection_period_role=str(lock["development_role"]),
    )
    holdout_dir = output_dir / f"fy{period.fiscal_year}-evaluation"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    summary_path = holdout_dir / "comparison-summary.json"
    scores_path = holdout_dir / "selected-scores.parquet"
    queue_path = holdout_dir / "guarded-review-queue.parquet"
    summary = {
        "table_winners": lock["table_winners"],
        "selected_metrics": selected_metrics.as_dict(),
        "guarded_metrics": guarded_metrics.as_dict(),
        "baseline_metrics": baseline_metrics.as_dict(),
        "promotion": promotion.as_dict(),
        "development_lock_sha256": _file_digest(lock_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    selected_scores.to_parquet(scores_path, index=False)
    guarded_queue.to_parquet(queue_path, index=False)
    run_id = tracker.log_run(
        run_name=f"locked-comparison-fy{period.fiscal_year}",
        params={
            "evaluation_period": period.fiscal_year,
            "development_lock_sha256": _file_digest(lock_path),
        },
        metrics={
            "selected_top_50_precision": selected_metrics.top_k_precision,
            "guarded_high_critical_recall": guarded_metrics.high_critical_recall,
        },
        artifacts=(summary_path, scores_path, queue_path, lock_path),
    )
    return {"mlflow_run_id": run_id, "summary": summary}


def _persist_candidates(
    result: ComparisonResult,
    output_dir: Path,
) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    isolation = result.candidates["isolation_forest"]
    assert isinstance(isolation, TrainedModels)
    one_class = result.candidates["one_class_svm"]
    assert isinstance(one_class, OneClassSvmModels)
    for name, models in (
        ("isolation_forest", isolation),
        ("one_class_svm", one_class),
    ):
        paths[name] = {}
        for table in FeatureTable:
            directory = output_dir / "models" / name / table.value
            if name == "isolation_forest":
                save_trained_model(models.for_table(table), directory)
            else:
                save_one_class_svm(models.for_table(table), directory)
            paths[name][f"{table.value}_estimator"] = directory / "estimator.skops"
            paths[name][f"{table.value}_pipeline"] = directory / "feature-pipeline.json"
    return paths


def _score_locked_winners(
    lock: dict[str, object],
    period: EvaluationPeriod,
) -> pd.DataFrame:
    frames = {
        "vendor": build_vendor_features(period.vendors),
        "transaction": build_transaction_features(period.transactions),
    }
    schemas = {
        "vendor": VENDOR_FEATURE_SCHEMA,
        "transaction": TRANSACTION_FEATURE_SCHEMA,
    }
    scores = []
    for table, model_name in lock["table_winners"].items():
        package = lock["packages"][model_name][table]
        directory = Path(str(package["path"]))
        if _directory_digest(directory) != package["sha256"]:
            raise ValueError(f"Locked {table} candidate package has changed")
        if model_name == "isolation_forest":
            model = load_trained_model(directory, schema=schemas[table])
        else:
            model = load_one_class_svm(directory, schema=schemas[table])
        scores.append(model.score(frames[table]))
    return pd.concat(scores, ignore_index=True).sort_values(
        ["priority_score", "target_table", "record_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def _model_metrics(
    result: ComparisonResult,
    model_name: str,
) -> dict[str, float]:
    metrics = result.model_metrics[model_name]
    return {
        "precision": metrics.classification.precision,
        "recall": metrics.classification.recall,
        "macro_f1": metrics.macro_f1,
        "top_50_precision": metrics.top_k_precision,
        "false_alarms_per_1000": metrics.false_alarms_per_1000_records,
    }


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
