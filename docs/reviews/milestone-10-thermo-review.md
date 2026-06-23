# Milestone 10 thermo-nuclear code quality review

## Scope

This review covers the application image, Docker Compose stack, smoke test, and initial thin Streamlit client.

## Findings and resolutions

### Multiple service images would duplicate dependency and security maintenance

Separate images for FastAPI, Streamlit, and MLflow would repeat the same locked Python environment.

Resolution: one non-root application image is reused with explicit service commands.

### Container readiness needed dependency-aware startup

Port exposure alone would allow clients to reach services before their dependencies were usable.

Resolution: every service has a health check and Compose dependencies require healthy upstream services.

### The interface could have duplicated product behavior

Reading CSV files and running scoring logic inside Streamlit would violate the FastAPI ownership boundary.

Resolution: Streamlit uses a small versioned API client and contains presentation behavior only.

### Smoke verification needed to test behavior, not only open ports

A successful TCP connection would not prove validation, persistence, and finding retrieval work together.

Resolution: `scripts/smoke_stack.py` uploads a paired dataset, checks expected findings, and retrieves the persisted run through public routes.

### Non-root MLflow needed an owned artifact mount point

The first image created the DataLens artifact directory but not the MLflow artifact root.
A new named volume mounted at a missing root directory could be unwritable by the non-root runtime user.

Resolution: the image creates and owns both `/app/artifacts` and `/mlartifacts`.

### MLflow metadata should not share the application database

The first Compose draft placed application and MLflow tables in the same PostgreSQL database.
That coupled independent schemas and made lifecycle operations less clear.

Resolution: PostgreSQL initialization creates a separate `mlflow` database while retaining one database service for the proof of concept.

## Approval

Container definitions remain declarative and small.
The initial Streamlit entry point is below the file-size threshold and delegates all application behavior.
No structural blocker remains after the resolutions above.
