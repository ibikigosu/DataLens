"""Separated development training and temporal holdout evaluation workflows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from datalens.baseline.defects import inject_controlled_defects
from datalens.baseline.rules import run_rules
from datalens.baseline.run import BaselinePlan, load_config
from datalens.data.plan import DatasetPlan, load_dataset_plan
from datalens.features.builders import build_transaction_features, build_vendor_features
from datalens.features.dataset import DevelopmentFeatureDataset
from datalens.features.pipeline import FeatureTable
from datalens.features.schemas import TRANSACTION_FEATURE_SCHEMA, VENDOR_FEATURE_SCHEMA
from datalens.modeling.evaluation import RankingEvaluation, evaluate_record_ranking
from datalens.modeling.models import (
    TrainedModels,
    load_trained_model,
    save_trained_model,
    train_isolation_forests,
)
from datalens.modeling.tracking import ExperimentTracker
from datalens.paths import ARTIFACTS_DIR, MANIFEST_DIR, PROCESSED_DATA_DIR

DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "modeling" / "milestone-05"
EXPERIMENT_NAME = "datalens-isolation-forest"
LOCK_FILENAME = "development-lock.json"


@dataclass(frozen=True)
class EvaluationPeriod:
    fiscal_year: str
    role: str
    vendors: pd.DataFrame
    transactions: pd.DataFrame
    labels: pd.DataFrame
    deterministic_findings: pd.DataFrame
    dataset_identity: dict[str, object]

    @property
    def record_counts(self) -> dict[str, int]:
        return {
            FeatureTable.VENDOR.value: len(self.vendors),
            FeatureTable.TRANSACTION.value: len(self.transactions),
        }


def run_development_experiment(
    tracker: ExperimentTracker,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dataset_plan: DatasetPlan | None = None,
    baseline_plan: BaselinePlan | None = None,
) -> dict[str, object]:
    """Fit FY2024 models, evaluate development results, and write a lock manifest."""
    plan = dataset_plan or load_dataset_plan()
    defects = baseline_plan or load_config()
    development_periods = [period for period in plan.periods if period.role == "development"]
    if len(development_periods) != 1:
        raise ValueError("Exactly one development period is required")
    period = development_periods[0]
    clean_vendors, clean_transactions = _load_clean_records(period.fiscal_year)
    dataset = DevelopmentFeatureDataset.from_records(
        clean_vendors,
        clean_transactions,
        fiscal_year=period.fiscal_year,
        dataset_plan=plan,
    )
    models = train_isolation_forests(dataset)
    evaluation_period = _build_evaluation_period(
        period.fiscal_year,
        role=period.role,
        clean_vendors=clean_vendors,
        clean_transactions=clean_transactions,
        defects=defects,
    )
    scores = _score_period(models, evaluation_period)
    metrics = evaluate_record_ranking(
        evaluation_period.labels,
        scores,
        evaluated_records_by_table=evaluation_period.record_counts,
    )
    baseline_metrics = _baseline_record_evaluation(evaluation_period)
    paths = _persist_development_artifacts(
        output_dir,
        models=models,
        scores=scores,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        dataset_identity=evaluation_period.dataset_identity,
    )
    run_id = tracker.log_run(
        run_name=f"isolation-forest-fy{period.fiscal_year}",
        params={
            "model_family": "isolation_forest",
            "fit_fiscal_year": period.fiscal_year,
            "fit_period_role": period.role,
            "vendor_schema_version": VENDOR_FEATURE_SCHEMA.schema_version,
            "transaction_schema_version": TRANSACTION_FEATURE_SCHEMA.schema_version,
            "vendor_seed": models.vendor.spec.seed,
            "transaction_seed": models.transaction.spec.seed,
        },
        metrics=_tracking_metrics(metrics, baseline_metrics),
        artifacts=tuple(paths.values()),
    )
    lock = {
        "schema_version": 1,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "development_fiscal_year": period.fiscal_year,
        "development_role": period.role,
        "mlflow_run_id": run_id,
        "candidate_packages": {
            table.value: {
                "path": str((output_dir / "models" / table.value).resolve()),
                "sha256": _directory_digest(output_dir / "models" / table.value),
            }
            for table in FeatureTable
        },
    }
    lock_path = output_dir / LOCK_FILENAME
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return {"lock_path": str(lock_path), "metrics": metrics.as_dict()}


def evaluate_temporal_holdout(
    lock_path: Path,
    tracker: ExperimentTracker,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dataset_plan: DatasetPlan | None = None,
    baseline_plan: BaselinePlan | None = None,
) -> dict[str, object]:
    """Evaluate the locked FY2024 candidates once against the declared FY2025 holdout."""
    plan = dataset_plan or load_dataset_plan()
    defects = baseline_plan or load_config()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    holdouts = [period for period in plan.periods if period.role == "temporal_holdout"]
    if len(holdouts) != 1:
        raise ValueError("Exactly one temporal holdout is required")
    period = holdouts[0]
    models = _load_locked_models(lock)
    clean_vendors, clean_transactions = _load_clean_records(period.fiscal_year)
    evaluation_period = _build_evaluation_period(
        period.fiscal_year,
        role=period.role,
        clean_vendors=clean_vendors,
        clean_transactions=clean_transactions,
        defects=defects,
    )
    scores = _score_period(models, evaluation_period)
    metrics = evaluate_record_ranking(
        evaluation_period.labels,
        scores,
        evaluated_records_by_table=evaluation_period.record_counts,
    )
    baseline_metrics = _baseline_record_evaluation(evaluation_period)
    holdout_dir = output_dir / f"fy{period.fiscal_year}-evaluation"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    scores_path = holdout_dir / "scores.parquet"
    metrics_path = holdout_dir / "metrics.json"
    scores.to_parquet(scores_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "model": metrics.as_dict(),
                "baseline": baseline_metrics.as_dict(),
                "development_lock_sha256": _file_digest(lock_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run_id = tracker.log_run(
        run_name=f"temporal-evaluation-fy{period.fiscal_year}",
        params={
            "evaluation_fiscal_year": period.fiscal_year,
            "evaluation_period_role": period.role,
            "development_lock_sha256": _file_digest(lock_path),
            "candidate_run_id": str(lock["mlflow_run_id"]),
        },
        metrics=_tracking_metrics(metrics, baseline_metrics),
        artifacts=(scores_path, metrics_path, lock_path),
    )
    return {"mlflow_run_id": run_id, "metrics": metrics.as_dict()}


def _build_evaluation_period(
    fiscal_year: str,
    *,
    role: str,
    clean_vendors: pd.DataFrame,
    clean_transactions: pd.DataFrame,
    defects: BaselinePlan,
) -> EvaluationPeriod:
    vendors, transactions, labels = inject_controlled_defects(
        clean_vendors,
        clean_transactions,
        fiscal_year=int(fiscal_year),
        seed=defects.seed,
        defects_per_type=defects.defects_per_type,
    )
    findings = run_rules(vendors, transactions, fiscal_year=int(fiscal_year))
    manifest_path = MANIFEST_DIR / f"prepared_pbs_fy{fiscal_year}.json"
    return EvaluationPeriod(
        fiscal_year=fiscal_year,
        role=role,
        vendors=vendors,
        transactions=transactions,
        labels=labels,
        deterministic_findings=findings,
        dataset_identity=json.loads(manifest_path.read_text(encoding="utf-8")),
    )


def _score_period(models: TrainedModels, period: EvaluationPeriod) -> pd.DataFrame:
    return pd.concat(
        [
            models.vendor.score(build_vendor_features(period.vendors)),
            models.transaction.score(build_transaction_features(period.transactions)),
        ],
        ignore_index=True,
    ).sort_values(
        ["priority_score", "target_table", "record_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def _baseline_record_evaluation(period: EvaluationPeriod) -> RankingEvaluation:
    findings = (
        period.deterministic_findings.groupby(
            ["target_table", "record_id"],
            as_index=False,
        )
        .agg(priority_score=("risk_score", "max"))
        .assign(predicted=True)
    )
    return evaluate_record_ranking(
        period.labels,
        findings,
        evaluated_records_by_table=period.record_counts,
    )


def _persist_development_artifacts(
    output_dir: Path,
    *,
    models: TrainedModels,
    scores: pd.DataFrame,
    metrics: RankingEvaluation,
    baseline_metrics: RankingEvaluation,
    dataset_identity: dict[str, object],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for table in FeatureTable:
        save_trained_model(
            models.for_table(table),
            output_dir / "models" / table.value,
        )
    scores_path = output_dir / "development-scores.parquet"
    metrics_path = output_dir / "development-metrics.json"
    identity_path = output_dir / "dataset-identity.json"
    scores.to_parquet(scores_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {"model": metrics.as_dict(), "baseline": baseline_metrics.as_dict()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    identity_path.write_text(
        json.dumps(dataset_identity, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "scores": scores_path,
        "metrics": metrics_path,
        "dataset_identity": identity_path,
        "vendor_estimator": output_dir / "models" / "vendor" / "estimator.skops",
        "vendor_pipeline": output_dir / "models" / "vendor" / "feature-pipeline.json",
        "transaction_estimator": output_dir / "models" / "transaction" / "estimator.skops",
        "transaction_pipeline": output_dir / "models" / "transaction" / "feature-pipeline.json",
    }


def _load_locked_models(lock: dict[str, object]) -> TrainedModels:
    packages = lock["candidate_packages"]
    assert isinstance(packages, dict)
    vendor = packages["vendor"]
    transaction = packages["transaction"]
    assert isinstance(vendor, dict)
    assert isinstance(transaction, dict)
    vendor_path = Path(str(vendor["path"]))
    transaction_path = Path(str(transaction["path"]))
    if _directory_digest(vendor_path) != vendor["sha256"]:
        raise ValueError("Locked vendor candidate package has changed")
    if _directory_digest(transaction_path) != transaction["sha256"]:
        raise ValueError("Locked transaction candidate package has changed")
    return TrainedModels(
        vendor=load_trained_model(vendor_path, schema=VENDOR_FEATURE_SCHEMA),
        transaction=load_trained_model(
            transaction_path,
            schema=TRANSACTION_FEATURE_SCHEMA,
        ),
    )


def _tracking_metrics(
    model: RankingEvaluation,
    baseline: RankingEvaluation,
) -> dict[str, float]:
    return {
        "model_precision": model.classification.precision,
        "model_recall": model.classification.recall,
        "model_macro_f1": model.macro_f1,
        "model_top_50_precision": model.top_k_precision,
        "model_false_alarms_per_1000": model.false_alarms_per_1000_records,
        "baseline_precision": baseline.classification.precision,
        "baseline_recall": baseline.classification.recall,
        "baseline_macro_f1": baseline.macro_f1,
        "baseline_top_50_precision": baseline.top_k_precision,
        "baseline_false_alarms_per_1000": baseline.false_alarms_per_1000_records,
    }


def _load_clean_records(fiscal_year: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_dir = PROCESSED_DATA_DIR / f"fy{fiscal_year}"
    return (
        pd.read_parquet(period_dir / "vendors.parquet"),
        pd.read_parquet(period_dir / "transactions.parquet"),
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
