# DataLens

DataLens is a human-in-the-loop data quality assistant for structured procurement data.
It helps data stewards and procurement analysts find, understand, review, and prioritize quality issues in vendor records and transaction records.
It is explicitly a data quality project, not a fraud detection system.

## Current delivery stage

The current delivery stage is feature engineering.
The first feature-pipeline slice establishes development-only fitting, stable model matrices, and separate vendor and transaction feature schemas.
Milestones remain in progress until their acceptance criteria are verified and their pull requests are merged, or completion is explicitly confirmed.

## Requirements

- Git 2.50 or later
- uv 0.11 or later
- Python 3.11

uv installs and manages the project environment from the checked-in lock file.

## Setup

```powershell
git clone https://github.com/ibikigosu/DataLens.git
Set-Location DataLens
uv sync --all-groups
```

## Verify the repository

Run the complete local verification:

```powershell
uv run python scripts/verify.py
```

The individual checks are:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run pytest --cov=datalens --cov-report=term-missing
```

## Repository layout

```text
artifacts/        Generated analysis and baseline outputs
config/           Versioned data acquisition and evaluation settings
data/             Local raw and processed data plus versioned provenance manifests
notebooks/        Reproducible exploratory analysis
scripts/          Mentor-facing verification helpers
src/datalens/     Reusable Python package code
tests/            Automated tests
```

Downloaded data and generated artifacts are intentionally excluded from Git.
Small metadata manifests remain versioned so a mentor can see exactly how a local dataset was produced.

## Data workflow

DataLens uses GSA Public Buildings Service contract transactions from USAspending.
FY2024 is the development period and FY2025 is the sealed temporal evaluation period.
USAspending provides this public data through an unauthenticated API.
No US citizenship, residency, or USAspending account is required.

Acquire and prepare both periods:

```powershell
uv run python -m datalens.data.usaspending acquire
uv run python -m datalens.data.prepare
```

Execute the FY2024 EDA notebook:

```powershell
uv run jupyter nbconvert --to notebook --execute notebooks/01_usaspending_pbs_eda.ipynb --inplace
```

Run the deterministic controlled-defect baseline:

```powershell
uv run python -m datalens.baseline.run
```

## Feature engineering

`DevelopmentFeatureDataset` is the fitting boundary and accepts only a period declared with the `development` role.
For the current dataset plan, that period is FY2024.
FY2025 feature frames may be transformed for temporal evaluation, but their distributions cannot influence imputation, scaling, or category frequencies.

Numeric features use an explicit missing-value indicator and median imputation in transformed space.
Skewed count and amount fields use a signed `log1p` transformation followed by robust scaling with the median and interquartile range.
The signed transformation preserves legitimate negative procurement adjustments instead of treating them as invalid automatically.

Categorical features use frequencies learned from the development data.
Missing categories receive an explicit missing-value indicator, and categories not seen during fitting receive a frequency of zero.
Frequency encoding keeps high-cardinality procurement codes bounded without creating an unstable one-hot feature space.

Vendor and transaction records use separate schemas because their quality signals and future anomaly models have different meanings.
Table-specific builders own domain feature construction before statistical preprocessing.
Both builders assign deterministic row identities independent of duplicate-prone business keys.
The `_record_id` remains outside the numeric model matrix so findings can be mapped back to source rows while retaining business keys as evidence.

## Important boundaries

- FastAPI will own application and scoring behavior in later milestones.
- Streamlit will remain a thin API client.
- Naturally unusual USAspending records are not labeled as defects without evidence.
- Controlled defects are retained separately from reviewer feedback.
- No source dataset is automatically corrected or overwritten.
