"""Prepare demo files and seed clearly labeled simulated review feedback."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import requests

from datalens.demo import demo_frames, write_demo_files


def seed_simulated_feedback(base_url: str) -> dict[str, Any]:
    """Create one scoring run and label findings through public API routes."""
    vendors, transactions = demo_frames()
    base = base_url.rstrip("/")
    response = requests.post(
        f"{base}/api/v1/score/batch",
        data={"fiscal_year": "2024"},
        files={
            "vendors": (
                "vendors.csv",
                io.BytesIO(vendors.to_csv(index=False).encode()),
                "text/csv",
            ),
            "transactions": (
                "transactions.csv",
                io.BytesIO(transactions.to_csv(index=False).encode()),
                "text/csv",
            ),
        },
        timeout=30,
    )
    response.raise_for_status()
    run = response.json()
    counts = {"correct_flag": 0, "false_alarm": 0}
    feedback_items = []
    for finding in run["findings"]:
        verdict = "correct_flag" if finding["issue_type"] == "invalid_vendor_uei" else "false_alarm"
        feedback_items.append(
            {
                "finding_id": finding["finding_id"],
                "verdict": verdict,
                "notes": (
                    "Simulated historical feedback for the mentor demonstration. "
                    "This is not verified production ground truth."
                ),
            }
        )
        counts[verdict] += 1
    feedback = requests.post(
        f"{base}/api/v1/runs/{run['run_id']}/feedback/batch",
        json={"feedback": feedback_items},
        timeout=30,
    )
    feedback.raise_for_status()
    return {
        "run_id": run["run_id"],
        "finding_count": len(run["findings"]),
        "saved_feedback": feedback.json()["saved_feedback"],
        "simulated_feedback": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("demo"))
    parser.add_argument("--write-files", action="store_true")
    parser.add_argument("--seed-feedback", action="store_true")
    args = parser.parse_args()
    result: dict[str, Any] = {}
    try:
        if args.write_files:
            write_demo_files(args.output_dir)
            result["files"] = {
                "vendors": str(args.output_dir / "vendors.csv"),
                "transactions": str(args.output_dir / "transactions.csv"),
            }
        if args.seed_feedback:
            result["seed"] = seed_simulated_feedback(args.base_url)
    except requests.RequestException as error:
        print(f"Demo preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
