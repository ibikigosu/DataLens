from pathlib import Path

import yaml


def test_compose_declares_complete_healthy_persistent_stack() -> None:
    compose = yaml.safe_load(Path("compose.yaml").read_text(encoding="utf-8"))

    assert set(compose["services"]) == {"postgres", "mlflow", "api", "streamlit"}
    assert set(compose["volumes"]) == {
        "postgres_data",
        "mlflow_artifacts",
        "datalens_artifacts",
    }
    assert all("healthcheck" in service for service in compose["services"].values())
    assert compose["services"]["api"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert compose["services"]["streamlit"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert any(
        str(argument).endswith("/mlflow") for argument in compose["services"]["mlflow"]["command"]
    )
    assert any(
        "init-databases.sql" in volume for volume in compose["services"]["postgres"]["volumes"]
    )


def test_dockerfile_runs_api_as_non_root_user() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "uv sync --frozen --no-dev --no-editable" in dockerfile
    assert "USER datalens" in dockerfile
    assert "mkdir -p /app/artifacts /mlartifacts" in dockerfile
    assert 'CMD ["uvicorn", "datalens.api.app:app"' in dockerfile
