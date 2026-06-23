# FastAPI application boundary

## Decision

FastAPI owns validation, scoring, persisted runs, finding retrieval, feedback, and manual retraining behavior.
Streamlit and any future client must call the versioned API instead of importing scoring or persistence modules directly.

Routes translate HTTP requests and responses only.
Application services own scoring and retraining orchestration.
The repository owns transactional persistence.
The procurement schema remains the canonical validation contract.

## Persistence boundary

PostgreSQL is introduced when scoring runs, findings, and reviewer feedback become shared application state.
Those records require stable identifiers, referential integrity, atomic updates, and concurrent access from API and interface processes.

SQLite remains the zero-setup local and test default.
The same SQLAlchemy repository is used for SQLite and PostgreSQL.
The container stack supplies PostgreSQL through `DATALENS_DATABASE_URL`.

## Single-record behavior

Single-record scoring evaluates table-local rules.
Cross-table relationship rules require a paired procurement dataset and therefore run only during batch scoring.
This prevents a transaction from being labeled as an orphan merely because the single-record endpoint does not include its vendor table.

## Current model ownership

The deterministic baseline remains the active scoring model because the anomaly candidates have not passed promotion gates.
Manual retraining runs the configured comparison workflow and retains the promotion or rejection evidence.
The API does not silently activate a rejected candidate.
