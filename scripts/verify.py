"""Run the repository's mentor-facing local verification commands."""

from __future__ import annotations

import subprocess
import sys

COMMANDS = (
    ("format", ["ruff", "format", "--check", "."]),
    ("lint", ["ruff", "check", "."]),
    (
        "tests",
        [
            "pytest",
            "--cov=datalens",
            "--cov-report=term-missing",
            "--cov-fail-under=85",
        ],
    ),
)


def main() -> int:
    """Run each verification command and stop on the first failure."""
    for label, command in COMMANDS:
        print(f"\n[{label}] {' '.join(command)}")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
