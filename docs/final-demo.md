# Final DataLens demo

## Setup

Start the complete container stack:

```powershell
docker compose up --build --wait
```

Prepare the versioned demo files and seed clearly labeled simulated historical feedback:

```powershell
uv run python scripts/demo_workflow.py --write-files --seed-feedback
```

Open Streamlit at `http://localhost:8501`.

If Docker is unavailable, start FastAPI and Streamlit in separate terminals:

```powershell
uv run uvicorn datalens.api.app:app --host 127.0.0.1 --port 8000
```

```powershell
uv run streamlit run src/datalens/ui/streamlit_app.py
```

## Walkthrough

1. Click **Run demo dataset**, or upload `demo/vendors.csv` and `demo/transactions.csv`.
2. Point out that validation passes before a scoring run is persisted.
3. Show the stable run identifier, schema version, and active model version.
4. Filter the ranked findings and select one row.
5. Explain the difference between configured risk score, review priority, and model confidence.
6. Submit one reviewer decision with a short note.
7. Click **Train feedback reranker**.
8. Compare active and candidate top-50 precision, top-50 recall, false alarms, and overall recall.
9. Show every promotion gate and the final active model identity.
10. Open MLflow at `http://localhost:5000` and show the registered anomaly candidates and retained rejection evidence.

## Expected evidence

The anomaly candidates remain rejected because they underperform the deterministic baseline.
The feedback reranker uses simulated historical review decisions and should improve held-out top-50 precision from approximately 50% to 100% on the controlled demo.
The exact split can change only if the versioned seed or demo data changes.
Overall recall remains 100% because reranking does not remove deterministic findings.

The simulated feedback result is proof of workflow behavior, not production validation.
The model card and retrospective state the limitations explicitly.

## Shutdown

Stop the stack while preserving local volumes:

```powershell
docker compose down
```

Reset every local service volume only when a clean demonstration state is required:

```powershell
docker compose down --volumes --remove-orphans
```
