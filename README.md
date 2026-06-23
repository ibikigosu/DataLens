# DataLens

DataLens is a human-in-the-loop data quality assistant for procurement records.
It combines deterministic checks with statistical anomaly detection to help analysts find, explain, and prioritize issues in vendor and transaction data.

> [!IMPORTANT]
> DataLens identifies potential data quality issues for human review.
> It does not detect fraud, automatically correct source records, or treat unusual activity as proof of an error.

## What it does

- Acquires reproducible GSA Public Buildings Service contract data from USAspending.
- Prepares separate vendor and transaction datasets.
- Detects known quality problems with deterministic rules.
- Compares Isolation Forest and Local Outlier Factor anomaly models.
- Tracks experiments and evaluation results with MLflow.
- Protects critical deterministic findings when building the review queue.
- Produces bounded explanations that can be traced back to source records.

## How it works

```text
USAspending data
       |
       v
Acquisition and preparation
       |
       +--> Deterministic quality rules
       |
       +--> Vendor and transaction feature pipelines
                         |
                         v
                  Anomaly models
                         |
                         v
              Guarded review queue
```

FY2024 is used for development and model selection.
FY2025 remains sealed until temporal evaluation so its distributions cannot influence training or preprocessing.

The deterministic baseline is currently the primary issue detector.
Statistical anomalies are supplemental evidence because the evaluated models have not passed the promotion criteria.

## Quick start

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) 0.11 or later
- Git 2.50 or later

### Install

```powershell
git clone https://github.com/ibikigosu/DataLens.git
Set-Location DataLens
uv sync --all-groups
```

### Acquire and prepare data

```powershell
uv run python -m datalens.data.usaspending acquire
uv run python -m datalens.data.prepare
```

Downloaded data stays local and is excluded from Git.
Versioned manifests preserve provenance and integrity metadata for reproducibility.

### Run the analysis

Run the deterministic baseline:

```powershell
uv run python -m datalens.baseline.run
```

Train, compare, and evaluate the anomaly models:

```powershell
uv run python -m datalens.modeling.run
```

Open the local MLflow experiment viewer:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
```

## Notebooks

The notebooks are reproducible analysis drivers.
Reusable behavior lives in the `datalens` package rather than notebook state.

Run the FY2024 exploratory analysis:

```powershell
uv run jupyter nbconvert --to notebook --execute notebooks/01_usaspending_pbs_eda.ipynb --inplace
```

Run the model comparison analysis:

```powershell
uv run jupyter nbconvert --to notebook --execute notebooks/02_model_comparison.ipynb --inplace
```

## Evaluation approach

DataLens evaluates deterministic rules and statistical models against controlled defects injected into real-shaped procurement data.
This provides reproducible labels without claiming that naturally unusual public records are incorrect.

The evaluation includes:

- Precision, recall, and macro F1
- Top-50 precision
- False alarms per 1,000 records
- High and critical issue recall
- Temporal performance on FY2025

Separate models are trained for vendor and transaction records because the tables have different schemas and quality signals.
Model thresholds and preprocessing parameters are learned from FY2024 only.

## Project structure

```text
config/           Data acquisition and evaluation settings
data/             Local datasets and versioned provenance manifests
docs/             Decisions, analysis results, and delivery roadmap
notebooks/        Reproducible analysis drivers
scripts/          Repository verification tools
src/datalens/     Acquisition, features, rules, models, and evaluation
tests/            Automated unit and workflow tests
artifacts/        Generated reports, models, and MLflow state
```

## Verify the project

Run formatting, linting, tests, and coverage checks together:

```powershell
uv run python scripts/verify.py
```

Or run each check separately:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run pytest --cov=datalens --cov-report=term-missing
```

## Current status

The data acquisition, exploratory analysis, deterministic baseline, feature pipelines, anomaly-model comparison, MLflow tracking, and notebook-to-module cleanup milestones are complete.

The next milestone focuses on configuration and reproducibility before the scoring workflow is exposed through FastAPI.
See the [internship roadmap](docs/planning/internship-roadmap.md) for acceptance criteria and delivery evidence.
