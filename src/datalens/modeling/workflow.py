"""End-to-end development selection and FY2025 temporal evaluation workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from datalens.baseline.defects import inject_controlled_defects
from datalens.baseline.rules import run_rules
from datalens.baseline.run import BaselinePlan, load_config
from datalens.data.plan import DatasetPlan, load_dataset_plan
from datalens.features.builders import build_transaction_features, build_vendor_features
from datalens.features.dataset import DevelopmentFeatureDataset
from datalens.features.pipeline import FeatureTable
from datalens.features.schemas import TRANSACTION_FEATURE_SCHEMA, VENDOR_FEATURE_SCHEMA
from datalens.modeling.evaluation import (
    PromotionCriteria,
    combine_selected_scores,
    evaluate_record_ranking,
    select_table_winners,
)
from datalens.modeling.models import (
    ModelFamily,
    ModelSpec,
    TableModelBundle,
    save_model_bundle,
    skops_trusted_types_for_model,
    train_table_model,
)
from datalens.modeling.scoring import baseline_record_ranking, build_guarded_review_queue
from datalens.paths import ARTIFACTS_DIR, MANIFEST_DIR, PROCESSED_DATA_DIR

DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "modeling"
DEFAULT_EXPERIMENT_NAME = "datalens-anomaly-model-comparison"
DEFAULT_TRACKING_DATABASE = ARTIFACTS_DIR / "mlflow.db"


@dataclass(frozen=True)
class PeriodData:
    """Clean records, controlled defects, and deterministic evidence for one period."""

    fiscal_year: str
    role: str
    clean_vendors: pd.DataFrame
    clean_transactions: pd.DataFrame
    defective_vendors: pd.DataFrame
    defective_transactions: pd.DataFrame
    labels: pd.DataFrame
    deterministic_findings: pd.DataFrame
    dataset_identity: dict[str, Any]

    @property
    def evaluated_records(self) -> int:
        return len(self.defective_vendors) + len(self.defective_transactions)

    @property
    def evaluated_records_by_table(self) -> dict[str, int]:
        return {
            FeatureTable.VENDOR.value: len(self.defective_vendors),
            FeatureTable.TRANSACTION.value: len(self.defective_transactions),
        }


def default_model_specs(seed: int = 42) -> dict[str, dict[FeatureTable, ModelSpec]]:
    """Return the fixed development comparison grid."""
    review_fractions = {
        FeatureTable.VENDOR: 0.025,
        FeatureTable.TRANSACTION: 0.005,
    }
    return {
        ModelFamily.ISOLATION_FOREST.value: {
            table: ModelSpec(
                family=ModelFamily.ISOLATION_FOREST,
                review_fraction=review_fraction,
                seed=seed,
                parameters={"n_estimators": 200, "max_samples": "auto", "n_jobs": -1},
            )
            for table, review_fraction in review_fractions.items()
        },
        ModelFamily.LOCAL_OUTLIER_FACTOR.value: {
            table: ModelSpec(
                family=ModelFamily.LOCAL_OUTLIER_FACTOR,
                review_fraction=review_fraction,
                seed=seed,
                parameters={"n_neighbors": 35, "n_jobs": -1},
            )
            for table, review_fraction in review_fractions.items()
        },
    }


def load_period_data(
    fiscal_year: int | str,
    *,
    dataset_plan: DatasetPlan,
    baseline_plan: BaselinePlan,
) -> PeriodData:
    """Load one declared dataset period and create reproducible controlled defects."""
    period = dataset_plan.period(fiscal_year)
    period_dir = PROCESSED_DATA_DIR / f"fy{period.fiscal_year}"
    clean_vendors = pd.read_parquet(period_dir / "vendors.parquet")
    clean_transactions = pd.read_parquet(period_dir / "transactions.parquet")
    defective_vendors, defective_transactions, labels = inject_controlled_defects(
        clean_vendors,
        clean_transactions,
        fiscal_year=int(period.fiscal_year),
        seed=baseline_plan.seed,
        defects_per_type=baseline_plan.defects_per_type,
    )
    deterministic_findings = run_rules(
        defective_vendors,
        defective_transactions,
        fiscal_year=int(period.fiscal_year),
    )
    manifest_path = MANIFEST_DIR / f"prepared_pbs_fy{period.fiscal_year}.json"
    dataset_identity = json.loads(manifest_path.read_text(encoding="utf-8"))
    return PeriodData(
        fiscal_year=period.fiscal_year,
        role=period.role,
        clean_vendors=clean_vendors,
        clean_transactions=clean_transactions,
        defective_vendors=defective_vendors,
        defective_transactions=defective_transactions,
        labels=labels,
        deterministic_findings=deterministic_findings,
        dataset_identity=dataset_identity,
    )


def run_experiment(
    development: PeriodData,
    temporal_holdout: PeriodData,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    model_specs: dict[str, dict[FeatureTable, ModelSpec]] | None = None,
    top_k: int = 50,
) -> dict[str, Any]:
    """Train on development data, select winners, then evaluate the sealed holdout."""
    if development.role != "development":
        raise ValueError("Model selection requires a development period")
    if temporal_holdout.role != "temporal_holdout":
        raise ValueError("Final evaluation requires the declared temporal holdout")
    if development.fiscal_year == temporal_holdout.fiscal_year:
        raise ValueError("Development and temporal evaluation periods must differ")

    specs = model_specs or default_model_specs()
    development_features = DevelopmentFeatureDataset.from_records(
        development.clean_vendors,
        development.clean_transactions,
        fiscal_year=development.fiscal_year,
    )
    defective_feature_frames = {
        development.fiscal_year: {
            FeatureTable.VENDOR: build_vendor_features(development.defective_vendors),
            FeatureTable.TRANSACTION: build_transaction_features(
                development.defective_transactions
            ),
        },
        temporal_holdout.fiscal_year: {
            FeatureTable.VENDOR: build_vendor_features(temporal_holdout.defective_vendors),
            FeatureTable.TRANSACTION: build_transaction_features(
                temporal_holdout.defective_transactions
            ),
        },
    }
    training_frames = {
        FeatureTable.VENDOR: development_features.vendors,
        FeatureTable.TRANSACTION: development_features.transactions,
    }
    schemas = {
        FeatureTable.VENDOR: VENDOR_FEATURE_SCHEMA,
        FeatureTable.TRANSACTION: TRANSACTION_FEATURE_SCHEMA,
    }

    bundles: dict[str, dict[FeatureTable, TableModelBundle]] = {}
    development_scores: dict[str, pd.DataFrame] = {}
    development_metrics: dict[str, dict[str, Any]] = {}
    for model_name, table_specs in specs.items():
        bundles[model_name] = {
            table: train_table_model(
                training_frames[table],
                schema=schemas[table],
                spec=table_specs[table],
                fit_fiscal_year=development.fiscal_year,
                period_role=development.role,
            )
            for table in FeatureTable
        }
        development_scores[model_name] = _score_period(
            bundles[model_name],
            defective_feature_frames[development.fiscal_year],
        )
        development_metrics[model_name] = evaluate_record_ranking(
            development.labels,
            development_scores[model_name],
            evaluated_records_by_table=development.evaluated_records_by_table,
            top_k=top_k,
        )

    table_winners = select_table_winners(development_metrics, top_k=top_k)
    selected_development_scores = combine_selected_scores(
        development_scores,
        table_winners,
    )
    selected_development_metrics = evaluate_record_ranking(
        development.labels,
        selected_development_scores,
        evaluated_records_by_table=development.evaluated_records_by_table,
        top_k=top_k,
    )

    holdout_scores = {
        model_name: _score_period(
            model_bundles,
            defective_feature_frames[temporal_holdout.fiscal_year],
        )
        for model_name, model_bundles in bundles.items()
    }
    holdout_metrics = {
        model_name: evaluate_record_ranking(
            temporal_holdout.labels,
            scores,
            evaluated_records_by_table=temporal_holdout.evaluated_records_by_table,
            top_k=top_k,
        )
        for model_name, scores in holdout_scores.items()
    }
    selected_holdout_scores = combine_selected_scores(holdout_scores, table_winners)
    selected_holdout_metrics = evaluate_record_ranking(
        temporal_holdout.labels,
        selected_holdout_scores,
        evaluated_records_by_table=temporal_holdout.evaluated_records_by_table,
        top_k=top_k,
    )

    baseline_development_ranking = baseline_record_ranking(development.deterministic_findings)
    baseline_holdout_ranking = baseline_record_ranking(temporal_holdout.deterministic_findings)
    baseline_development_metrics = evaluate_record_ranking(
        development.labels,
        baseline_development_ranking,
        evaluated_records_by_table=development.evaluated_records_by_table,
        top_k=top_k,
    )
    baseline_holdout_metrics = evaluate_record_ranking(
        temporal_holdout.labels,
        baseline_holdout_ranking,
        evaluated_records_by_table=temporal_holdout.evaluated_records_by_table,
        top_k=top_k,
    )

    guarded_development_queue = build_guarded_review_queue(
        selected_development_scores,
        development.deterministic_findings,
    )
    guarded_holdout_queue = build_guarded_review_queue(
        selected_holdout_scores,
        temporal_holdout.deterministic_findings,
    )
    guarded_development_metrics = evaluate_record_ranking(
        development.labels,
        guarded_development_queue,
        evaluated_records_by_table=development.evaluated_records_by_table,
        top_k=top_k,
    )
    guarded_holdout_metrics = evaluate_record_ranking(
        temporal_holdout.labels,
        guarded_holdout_queue,
        evaluated_records_by_table=temporal_holdout.evaluated_records_by_table,
        top_k=top_k,
    )
    promotion = PromotionCriteria().assess(
        selected_development_metrics,
        baseline_development_metrics,
        guarded_development_metrics,
        selection_period_role=development.role,
        top_k=top_k,
    )

    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection_policy": {
            "development_fiscal_year": development.fiscal_year,
            "development_role": development.role,
            "temporal_evaluation_fiscal_year": temporal_holdout.fiscal_year,
            "temporal_evaluation_role": temporal_holdout.role,
            "holdout_used_for_selection": False,
            "selection_order": [
                f"Select table winners from FY{development.fiscal_year} only",
                f"Lock winners before scoring FY{temporal_holdout.fiscal_year}",
            ],
        },
        "table_winners": table_winners,
        "promotion": promotion,
        "development": {
            "baseline": baseline_development_metrics,
            "models": development_metrics,
            "selected_workflow": selected_development_metrics,
            "guarded_queue": guarded_development_metrics,
        },
        "temporal_holdout": {
            "baseline": baseline_holdout_metrics,
            "models": holdout_metrics,
            "selected_workflow": selected_holdout_metrics,
            "guarded_queue": guarded_holdout_metrics,
        },
        "dataset_identity": {
            development.fiscal_year: development.dataset_identity,
            temporal_holdout.fiscal_year: temporal_holdout.dataset_identity,
        },
    }
    artifact_paths = _persist_artifacts(
        output_dir=output_dir,
        summary=summary,
        selected_development_scores=selected_development_scores,
        selected_holdout_scores=selected_holdout_scores,
        guarded_holdout_queue=guarded_holdout_queue,
        bundles=bundles,
    )
    _log_experiments(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        development=development,
        temporal_holdout=temporal_holdout,
        development_metrics=development_metrics,
        holdout_metrics=holdout_metrics,
        table_winners=table_winners,
        promotion=promotion,
        bundles=bundles,
        training_frames=training_frames,
        artifact_paths=artifact_paths,
    )
    return summary


def run_modeling_experiment(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> dict[str, Any]:
    """Load the project dataset plan and run the reproducible model comparison."""
    dataset_plan = load_dataset_plan()
    baseline_plan = load_config()
    development_periods = [
        period for period in dataset_plan.periods if period.role == "development"
    ]
    holdout_periods = [
        period for period in dataset_plan.periods if period.role == "temporal_holdout"
    ]
    if len(development_periods) != 1 or len(holdout_periods) != 1:
        raise ValueError(
            "Modeling requires exactly one development period and one temporal holdout"
        )
    development = load_period_data(
        development_periods[0].fiscal_year,
        dataset_plan=dataset_plan,
        baseline_plan=baseline_plan,
    )
    temporal_holdout = load_period_data(
        holdout_periods[0].fiscal_year,
        dataset_plan=dataset_plan,
        baseline_plan=baseline_plan,
    )
    return run_experiment(
        development,
        temporal_holdout,
        output_dir=output_dir,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )


def _score_period(
    bundles: dict[FeatureTable, TableModelBundle],
    feature_frames: dict[FeatureTable, pd.DataFrame],
) -> pd.DataFrame:
    return pd.concat(
        [bundles[table].score(feature_frames[table]) for table in FeatureTable],
        ignore_index=True,
    ).sort_values(
        ["priority_score", "target_table", "record_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def _persist_artifacts(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    selected_development_scores: pd.DataFrame,
    selected_holdout_scores: pd.DataFrame,
    guarded_holdout_queue: pd.DataFrame,
    bundles: dict[str, dict[FeatureTable, TableModelBundle]],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "comparison-summary.json",
        "development_scores": output_dir / "development-selected-scores.parquet",
        "holdout_scores": output_dir / "fy2025-selected-scores.parquet",
        "guarded_queue": output_dir / "fy2025-guarded-review-queue.parquet",
        "evidence": output_dir / "fy2025-bounded-anomaly-evidence.json",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_development_scores.to_parquet(paths["development_scores"], index=False)
    selected_holdout_scores.to_parquet(paths["holdout_scores"], index=False)
    guarded_holdout_queue.to_parquet(paths["guarded_queue"], index=False)
    top_evidence = (
        selected_holdout_scores.loc[selected_holdout_scores["predicted"]]
        .head(50)[
            [
                "target_table",
                "record_id",
                "model_name",
                "anomaly_score",
                "rank_percentile",
                "evidence",
            ]
        ]
        .to_dict(orient="records")
    )
    paths["evidence"].write_text(
        json.dumps(top_evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    for model_name, table_bundles in bundles.items():
        for table, bundle in table_bundles.items():
            bundle_path = output_dir / "models" / model_name / table.value
            save_model_bundle(bundle, bundle_path)
            paths[f"bundle_{model_name}_{table.value}"] = bundle_path
    return paths


def _log_experiments(
    *,
    tracking_uri: str | None,
    experiment_name: str,
    development: PeriodData,
    temporal_holdout: PeriodData,
    development_metrics: dict[str, dict[str, Any]],
    holdout_metrics: dict[str, dict[str, Any]],
    table_winners: dict[str, str],
    promotion: dict[str, Any],
    bundles: dict[str, dict[FeatureTable, TableModelBundle]],
    training_frames: dict[FeatureTable, pd.DataFrame],
    artifact_paths: dict[str, Path],
) -> None:
    resolved_tracking_uri = tracking_uri or _sqlite_tracking_uri(DEFAULT_TRACKING_DATABASE)
    mlflow.set_tracking_uri(resolved_tracking_uri)
    client = MlflowClient(tracking_uri=resolved_tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_root = artifact_paths["summary"].parent / "mlflow-artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_root.resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    for model_name, table_bundles in bundles.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(
                {
                    "model_family": model_name,
                    "development_fiscal_year": development.fiscal_year,
                    "temporal_holdout_fiscal_year": temporal_holdout.fiscal_year,
                    "holdout_used_for_selection": False,
                }
            )
            mlflow.log_metrics(
                _flatten_numeric_metrics(
                    development_metrics[model_name],
                    prefix="development",
                )
            )
            mlflow.log_metrics(
                _flatten_numeric_metrics(
                    holdout_metrics[model_name],
                    prefix="fy2025",
                )
            )
            mlflow.log_dict(
                {
                    "development": development.dataset_identity,
                    "temporal_holdout": temporal_holdout.dataset_identity,
                },
                "dataset-identity.json",
            )
            mlflow.log_dict(
                {
                    "development": development_metrics[model_name],
                    "temporal_holdout": holdout_metrics[model_name],
                },
                "metrics.json",
            )
            for table, bundle in table_bundles.items():
                with mlflow.start_run(run_name=table.value, nested=True):
                    mlflow.log_params(
                        {
                            "table": table.value,
                            "schema_version": bundle.model.schema_version,
                            "review_fraction": bundle.model.spec.review_fraction,
                            "seed": bundle.model.spec.seed,
                            **bundle.model.spec.parameters,
                        }
                    )
                    mlflow.log_dict(
                        bundle.pipeline.metadata(),
                        "feature-pipeline.json",
                    )
                    input_example = bundle.pipeline.transform(
                        training_frames[table].head(5)
                    ).values.to_numpy(dtype=float)
                    mlflow.sklearn.log_model(
                        bundle.model.estimator,
                        name=f"{table.value}-model",
                        serialization_format="skops",
                        input_example=input_example,
                        skops_trusted_types=skops_trusted_types_for_model(bundle.model),
                    )
                    mlflow.log_artifacts(
                        str(artifact_paths[f"bundle_{model_name}_{table.value}"]),
                        artifact_path="model-bundle",
                    )

    with mlflow.start_run(run_name="selection-and-promotion"):
        mlflow.log_dict(table_winners, "table-winners.json")
        mlflow.log_dict(promotion, "promotion-decision.json")
        mlflow.log_params(
            {
                "vendor_winner": table_winners["vendor"],
                "transaction_winner": table_winners["transaction"],
                "promoted": promotion["promoted"],
                "selection_period": development.fiscal_year,
                "selection_period_role": development.role,
            }
        )
        for name in (
            "summary",
            "development_scores",
            "holdout_scores",
            "guarded_queue",
            "evidence",
        ):
            mlflow.log_artifact(str(artifact_paths[name]))


def _flatten_numeric_metrics(
    payload: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        metric_name = f"{prefix}_{key}"
        if isinstance(value, bool | int | float):
            flattened[metric_name] = float(value)
        elif isinstance(value, dict) and key != "rank_calibration":
            flattened.update(
                _flatten_numeric_metrics(
                    value,
                    prefix=metric_name,
                )
            )
    return flattened


def _sqlite_tracking_uri(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve().as_posix()}"
