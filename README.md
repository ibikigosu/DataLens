# DataLens

DataLens is a human-in-the-loop data quality assistant for structured procurement data.
It helps data stewards and procurement analysts find, understand, review, and prioritize quality issues in vendor records and transaction records.
It is explicitly a data quality project, not a fraud detection system.

## Current delivery stage

The current delivery stage is statistical model comparison and notebook-to-module cleanup.
Separate vendor and transaction models are trained on FY2024, tracked in MLflow, and evaluated on FY2025 only after development selection is locked.
The deterministic baseline remains the primary issue detector because the current statistical candidates did not pass promotion criteria.
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
The source data is retrieved through the public USAspending API.

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

Run the statistical model comparison and MLflow tracking workflow:

```powershell
uv run python -m datalens.modeling.run
```

Inspect the local MLflow experiment:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
```

Execute the thin model comparison notebook:

```powershell
uv run jupyter nbconvert --to notebook --execute notebooks/02_model_comparison.ipynb --inplace
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

## Statistical modeling

DataLens trains separate vendor and transaction models because the tables have different feature spaces and review meanings.
Isolation Forest is the first statistical model.
Local Outlier Factor is the comparison model.
Both approaches use fixed seeds and thresholds learned from FY2024 training scores.

Model selection uses FY2024 controlled-defect results only.
The selected workflow uses Isolation Forest for vendors and Local Outlier Factor for transactions.
FY2025 is evaluated only after those choices are locked.

The statistical workflow is not promoted over the deterministic baseline.
It has materially lower precision, recall, macro F1, and top-50 precision, with more false alarms.
Statistical anomalies therefore remain supplemental review evidence.

Every anomaly explanation is bounded to three feature deviations and 1,000 serialized characters.
The 0 to 100 anomaly score is a relative training-score rank.
It is not business severity, a defect probability, or proof of a specific issue.

The guarded review queue always retains deterministic findings and ranks deterministic critical findings ahead of statistical-only candidates.
This prevents the model layer from suppressing critical structural findings.

Reusable model behavior lives under `src/datalens/modeling/`.
`models.py` owns fitting, scoring, and serializable table bundles.
`evaluation.py` owns record-level metrics, table winner selection, and promotion gates.
`scoring.py` owns deterministic guardrails and review-queue construction.
`evidence.py` owns bounded anomaly evidence.
`workflow.py` owns development selection, FY2025 evaluation, artifacts, and MLflow tracking.
`run.py` is the command-line entry point.

## Important boundaries

- FastAPI will own application and scoring behavior in later milestones.
- Streamlit will remain a thin API client.
- Naturally unusual USAspending records are not labeled as defects without evidence.
- Controlled defects are retained separately from reviewer feedback.
- No source dataset is automatically corrected or overwritten.
