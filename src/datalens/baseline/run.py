"""Run the controlled-defect deterministic baseline for FY2024 and FY2025."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from datalens.baseline.defects import inject_controlled_defects
from datalens.baseline.evaluation import evaluate_baseline
from datalens.baseline.rules import run_rules
from datalens.data.plan import load_dataset_plan
from datalens.paths import ARTIFACTS_DIR, CONFIG_DIR, PROCESSED_DATA_DIR

DEFAULT_CONFIG_PATH = CONFIG_DIR / "baseline" / "controlled-defects.json"


@dataclass(frozen=True)
class BaselinePlan:
    """Sampling controls for reproducible controlled-defect evaluation."""

    schema_version: int
    seed: int
    defects_per_type: int


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> BaselinePlan:
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    return BaselinePlan(
        schema_version=int(config["schema_version"]),
        seed=int(config["seed"]),
        defects_per_type=int(config["defects_per_type"]),
    )


def run_period(fiscal_year: int, config: BaselinePlan) -> dict[str, object]:
    """Inject, score, evaluate, and persist one fiscal-year baseline run."""
    period_dir = PROCESSED_DATA_DIR / f"fy{fiscal_year}"
    vendors = pd.read_parquet(period_dir / "vendors.parquet")
    transactions = pd.read_parquet(period_dir / "transactions.parquet")
    defective_vendors, defective_transactions, labels = inject_controlled_defects(
        vendors,
        transactions,
        fiscal_year=fiscal_year,
        seed=config.seed,
        defects_per_type=config.defects_per_type,
    )
    findings = run_rules(
        defective_vendors,
        defective_transactions,
        fiscal_year=fiscal_year,
    )
    metrics = evaluate_baseline(
        labels,
        findings,
        evaluated_records=len(defective_vendors) + len(defective_transactions),
    )

    output_dir = ARTIFACTS_DIR / "baseline" / f"fy{fiscal_year}"
    output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output_dir / "controlled_defect_labels.parquet", index=False)
    findings.to_parquet(output_dir / "findings.parquet", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    config = load_config()
    dataset_plan = load_dataset_plan()
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": config.seed,
        "defects_per_type": config.defects_per_type,
        "periods": {
            fiscal_year: run_period(int(fiscal_year), config)
            for fiscal_year in dataset_plan.fiscal_years
        },
    }
    output_path = ARTIFACTS_DIR / "baseline" / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
