"""Command-line entry point for model comparison and MLflow tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datalens.modeling.workflow import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_OUTPUT_DIR,
    run_modeling_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, compare, track, and evaluate DataLens anomaly models."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated comparison and model artifacts.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI. Defaults to artifacts/mlflow.db using SQLite.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_modeling_experiment(
        output_dir=args.output_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
