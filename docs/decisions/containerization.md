# Containerization

## Decision

DataLens uses one immutable application image for FastAPI, Streamlit, and MLflow commands.
Docker Compose adds PostgreSQL and provides the demonstration topology, service ordering, health checks, ports, and persistent volumes.

The application image installs the locked production dependency set with `uv sync --frozen --no-dev --no-editable`.
The runtime process uses an unprivileged `datalens` user.

## Persistent data

The proof of concept uses named local volumes for:

- PostgreSQL application and MLflow metadata.
- MLflow artifacts.
- DataLens scoring and retraining artifacts.

Local volume storage is sufficient for a single-machine mentor demonstration because the stack has one deployment boundary, modest artifacts, and no availability objective.

Object storage becomes justified when artifacts must survive host replacement, be shared across deployments, support retention policies, or scale beyond one machine.
An asynchronous worker becomes justified when scoring or retraining exceeds acceptable request duration, jobs must survive API restarts, or concurrent workloads require queueing and resource isolation.

## Startup contract

PostgreSQL must be healthy before MLflow and FastAPI start.
MLflow must be healthy before FastAPI becomes available.
FastAPI must be healthy before Streamlit starts.

The smoke test uses public API routes to verify readiness, paired CSV scoring, persisted run retrieval, and expected findings.
