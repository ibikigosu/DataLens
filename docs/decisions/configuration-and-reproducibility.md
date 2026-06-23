# Configuration and reproducibility

## Decision

DataLens separates version-controlled policy from environment-specific service values.

The files under `config/schema`, `config/model`, `config/data`, and `config/baseline` are version controlled because they define reproducible behavior.
They contain the approved procurement input contract, scoring weights, dataset periods, controlled-defect settings, feature version, model parameters, and promotion gates.

Application defaults are version controlled in `config/application/default.json`.
Environment-specific paths, ports, database URLs, and MLflow endpoints can override those defaults through `DATALENS_` environment variables.
Secrets and machine-specific `.env` files remain outside version control.

## Rationale

The procurement schema is the canonical boundary shared by training, API validation, batch scoring, and the user interface.
Keeping the schema typed and versioned prevents each client from inventing its own interpretation of required fields.

Model parameters and promotion criteria are policy decisions that must be reviewable in a pull request.
They are not operational secrets and should not depend on the machine running the code.

Service locations and credentials vary by environment.
Environment variables allow local, containerized, and hosted deployments to use the same source tree without edits.

## Reproducibility record

Every model comparison records:

- Dataset version.
- Procurement schema version.
- Feature version.
- Model policy version.
- Random seed and model parameters.
- Development and temporal holdout identities.

The documented commands use the same typed configuration loader as production entry points.
Invalid configuration fails with a file-specific error before training or scoring starts.
