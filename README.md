<div align="center">

# DataLens

*Human-in-the-loop data quality analysis for procurement records*

[![Python](https://img.shields.io/badge/Python->=3.11-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-f7931e?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Ruff](https://img.shields.io/badge/Ruff-261230?style=flat-square&logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)

[Overview](#overview) • [Features](#features) • [Getting started](#getting-started) • [API](#api) • [Project structure](#project-structure)

</div>

## Overview

DataLens is a human-in-the-loop data quality assistant for structured procurement data.
It combines deterministic checks with statistical anomaly detection to help analysts find, explain, and prioritize issues in vendor and transaction records.

> [!IMPORTANT]
> DataLens identifies potential data quality issues for human review.
> It does not detect fraud, automatically correct source records, or treat unusual activity as proof of an error.

## Features

- **Deterministic quality rules** - Detect known data quality problems with typed, severity-ranked rules
- **Statistical anomaly detection** - Compare Isolation Forest and Local Outlier Factor models for supplemental evidence
- **Guarded review queue** - Protect critical deterministic findings when ranking records for review
- **Bounded explanations** - Every finding includes traceable evidence with at most three feature deviations
- **Versioned configuration** - Schema contracts, scoring weights, and model parameters live under version control
- **Experiment tracking** - MLflow logs parameters, metrics, and artifacts for every model comparison
- **FastAPI scoring service** - Versioned REST API for validation, batch scoring, feedback, and retraining checks
- **Streamlit review workspace** - Thin UI client for uploads, ranked findings, review feedback, and model evidence
- **Containerized local stack** - Docker Compose runs PostgreSQL, MLflow, FastAPI, and Streamlit together
- **Reproducible notebooks** - Jupyter notebooks act as thin analysis drivers over tested Python modules

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

## Getting started

### Prerequisites

- [Python](https://www.python.org) 3.11 or later
- [uv](https://docs.astral.sh/uv/) 0.11 or later
- [Git](https://git-scm.com) 2.50 or later

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

### Run the API

Start FastAPI locally:

```powershell
uv run uvicorn datalens.api.app:app --host 0.0.0.0 --port 8000
```

OpenAPI documentation is available at `http://localhost:8000/docs`.
All public application routes are versioned under `/api/v1`.

### Run the review UI

Start Streamlit locally after the API is running:

```powershell
uv run streamlit run src/datalens/ui/streamlit_app.py
```

Open the review workspace at `http://localhost:8501`.
Use `demo/vendors.csv` and `demo/transactions.csv` for the mentor walkthrough.

### Run the full stack

Start PostgreSQL, MLflow, FastAPI, and Streamlit together:

```powershell
docker compose up --build
```

The container stack exposes:

- Streamlit review workspace at `http://localhost:8501`
- FastAPI documentation at `http://localhost:8000/docs`
- MLflow UI at `http://localhost:5000`

## Configuration

Version-controlled defaults live under `config/`.
The procurement schema owns required columns, relationships, and quality scoring weights.
The model configuration owns feature and model versions, model parameters, review fractions, and promotion gates.
Application paths and service endpoints are supplied by `config/application/default.json`.

Copy `.env.example` to `.env` when a local environment needs different service values.
Every application setting can also be overridden with a `DATALENS_` environment variable without editing source code.

For example:

```powershell
$env:DATALENS_ARTIFACT_DIR = "C:\datalens-artifacts"
$env:DATALENS_DATABASE_URL = "sqlite:///C:/datalens-artifacts/datalens.db"
uv run python -m datalens.modeling.run
```

Invalid JSON, unknown fields, missing schema columns, invalid types, and undeclared scoring weights fail before training or scoring.
Model comparison artifacts record the dataset, schema, feature, and model versions used by the run.

## API

### Health checks

Check readiness:

```powershell
curl.exe http://localhost:8000/api/v1/health/ready
```

### Batch scoring

Score paired CSV files:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/score/batch `
  -F "fiscal_year=2024" `
  -F "vendors=@vendors.csv;type=text/csv" `
  -F "transactions=@transactions.csv;type=text/csv"
```

Retrieve findings as JSON or CSV:

```powershell
curl.exe http://localhost:8000/api/v1/runs/RUN_ID/findings
curl.exe -OJ http://localhost:8000/api/v1/runs/RUN_ID/findings.csv
```

### Single-record validation

The following request demonstrates validation failing before scoring:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/score/vendor `
  -H "Content-Type: application/json" `
  -d '{"fiscal_year":2024,"record":{"vendor_id":"V1"}}'
```

The response is HTTP 422 because the record does not satisfy the approved vendor schema.
Scoring runs, findings, feedback, and retraining decisions are persisted through SQLAlchemy.
SQLite is the local default and PostgreSQL is used by the container stack.

### Model Registry and MLflow

Open the local MLflow experiment viewer:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
```

Model runs persist metrics, artifacts, promotion evidence, and model-card data.
Promotion gates keep statistical models supplemental unless evaluation criteria are met.

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
demo/             Small vendor and transaction files for product walkthroughs
docker/           Container database initialization
notebooks/        Reproducible analysis drivers
scripts/          Repository verification tools
src/datalens/     Acquisition, features, rules, models, and evaluation
tests/            Automated unit and workflow tests
artifacts/        Generated reports, models, and MLflow state
```

## Verification

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

The instructional implementation is complete through data acquisition, exploratory analysis, deterministic baseline, feature pipelines, anomaly-model comparison, MLflow tracking, model registry evidence, configuration, FastAPI scoring, containerization, and a Streamlit review interface.

The repository now keeps generated artifacts, downloaded datasets, and internal planning documents outside Git while preserving versioned source, configuration, tests, demo data, and provenance manifests.
