# DataLens

DataLens is a human-in-the-loop data quality assistant for structured procurement data.
It helps data stewards and procurement analysts find, understand, review, and prioritize quality issues in vendor records and transaction records.
It is explicitly a data quality project, not a fraud detection system.

## Current delivery stage

The current delivery stage is model iteration and comparison.
Isolation Forest and One-Class SVM candidates are compared separately for vendor and transaction records.
One feedback-trained false-alarm reranker improves deterministic finding order without changing issue identity or suppressing critical findings.
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

## Isolation Forest experiment

MLflow uses PostgreSQL for experiment state.
Set the tracking URI explicitly before running an experiment:

```powershell
Copy-Item .env.example .env
$env:MLFLOW_TRACKING_URI = "postgresql+psycopg://datalens:datalens@localhost:5432/datalens"
```

Train separate vendor and transaction Isolation Forests on FY2024:

```powershell
uv run python -m datalens.modeling.run train
```

The training command creates `artifacts/modeling/milestone-05/development-lock.json`.
The lock records the MLflow run and SHA-256 digests of both candidate packages.

Evaluate the locked candidates on FY2025:

```powershell
uv run python -m datalens.modeling.run evaluate-holdout `
  --lock-path artifacts/modeling/milestone-05/development-lock.json
```

Training and temporal evaluation are separate commands.
The evaluation command refuses candidate packages that changed after the development lock was written.
There is no SQLite production fallback.

## Model comparison and reranking

Run the FY2024 development comparison:

```powershell
uv run python -m datalens.modeling.run compare
```

The comparison:

- evaluates Isolation Forest and One-Class SVM through the same record-level metrics;
- selects vendor and transaction winners independently from FY2024 only;
- trains one logistic false-alarm reranker from reproducible simulated issue-level feedback;
- evaluates precision, recall, macro F1, top-50 precision, and false alarms;
- writes a comparison lock containing candidate, reranker, and MLflow run digests.

Evaluate the locked comparison on FY2025:

```powershell
uv run python -m datalens.modeling.run evaluate-comparison `
  --lock-path artifacts/modeling/milestone-06/comparison-lock.json
```

FY2025 cannot change the selected model families or reranker.
The evaluation command rejects changed candidate packages and changed reranker artifacts.

Promotion requires the selected anomaly workflow to be non-inferior to the deterministic baseline for top-50 precision, macro F1, and false alarms.
The guarded review queue must also preserve 100% high and critical controlled-defect recall.
Failed gates produce an explicit non-promotion decision rather than silently activating the latest model.

The reranker is not an issue classifier.
It changes finding order only.
Issue type, severity, and critical protection remain deterministic.

The verified FY2024 comparison selected One-Class SVM for both tables.
It achieved 80% record-level top-50 precision, compared with 0% for Isolation Forest, but its macro F1 remained 15.94% and its false-alarm rate was 7.61 per 1,000 records.
The deterministic baseline remained stronger overall, so the candidate failed promotion.
The guarded queue retained 100% high and critical controlled-defect recall.
FY2025 evaluation preserved that critical recall and produced 80% top-50 precision for the locked statistical workflow.

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

## Statistical model boundaries

Isolation Forest is the first statistical model because it can rank unusual combinations without requiring production defect labels.
Vendor and transaction models remain separate because their feature spaces and review meanings differ.
Fixed random seeds make repeated fitting reproducible.

The anomaly score is a bounded 0 to 100 rank against FY2024 training scores.
It is not business severity, a defect probability, or proof of a quality issue.
Each evidence payload contains at most three feature deviations and 1,000 serialized characters.

Candidate packages use a skops estimator and explicit JSON preprocessing state.
Pickle and joblib are not used for candidate persistence.
Model parameters, evaluation metrics, dataset identity, feature schema versions, and candidate artifacts are logged to MLflow.

## Important boundaries

- FastAPI will own application and scoring behavior in later milestones.
- Streamlit will remain a thin API client.
- Naturally unusual USAspending records are not labeled as defects without evidence.
- Controlled defects are retained separately from reviewer feedback.
- No source dataset is automatically corrected or overwritten.
