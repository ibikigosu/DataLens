"""Command-line entry points for locked development and holdout workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datalens.modeling.comparison_workflow import (
    COMPARISON_EXPERIMENT_NAME,
    DEFAULT_COMPARISON_DIR,
    evaluate_locked_comparison,
    run_development_comparison,
)
from datalens.modeling.tracking import ExperimentTracker, require_postgresql_tracking_uri
from datalens.modeling.workflow import (
    DEFAULT_OUTPUT_DIR,
    EXPERIMENT_NAME,
    evaluate_temporal_holdout,
    run_development_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DataLens Isolation Forest experiments.")
    parser.add_argument(
        "command",
        choices=(
            "train",
            "evaluate-holdout",
            "compare",
            "evaluate-comparison",
        ),
    )
    parser.add_argument("--lock-path", type=Path)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        tracking_uri = require_postgresql_tracking_uri(args.tracking_uri)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    comparison_command = args.command in {"compare", "evaluate-comparison"}
    output_dir = args.output_dir or (
        DEFAULT_COMPARISON_DIR if comparison_command else DEFAULT_OUTPUT_DIR
    )
    tracker = ExperimentTracker(
        tracking_uri,
        experiment_name=(COMPARISON_EXPERIMENT_NAME if comparison_command else EXPERIMENT_NAME),
        artifact_root=output_dir / "mlflow-artifacts",
    )
    if args.command == "train":
        result = run_development_experiment(tracker, output_dir=output_dir)
    elif args.command == "evaluate-holdout":
        if args.lock_path is None:
            raise SystemExit("--lock-path is required for evaluate-holdout")
        result = evaluate_temporal_holdout(
            args.lock_path,
            tracker,
            output_dir=output_dir,
        )
    elif args.command == "compare":
        result = run_development_comparison(tracker, output_dir=output_dir)
    else:
        if args.lock_path is None:
            raise SystemExit("--lock-path is required for evaluate-comparison")
        result = evaluate_locked_comparison(
            args.lock_path,
            tracker,
            output_dir=output_dir,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
