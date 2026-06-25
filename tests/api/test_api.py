import json
import logging
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from datalens.api.app import create_app
from datalens.application.persistence import RetrainingRecord
from datalens.application.scoring import ScoringResult
from datalens.configuration.loader import RuntimeConfig, load_runtime_config


def _runtime_config(tmp_path: Path) -> RuntimeConfig:
    config = load_runtime_config()
    settings = config.settings.model_copy(
        update={
            "artifact_dir": tmp_path / "artifacts",
            "database_url": f"sqlite:///{(tmp_path / 'datalens.db').as_posix()}",
            "mlflow_tracking_uri": (f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"),
        }
    )
    return RuntimeConfig(settings=settings, schema=config.schema, model=config.model)


def _vendor() -> dict[str, object]:
    return {
        "vendor_id": "V1",
        "recipient_name": "Vendor One",
        "recipient_uei": "BAD!",
        "recipient_country_code": "USA",
        "recipient_state_code": "VA",
        "source_transaction_count": 1,
        "address_variant_count": 1,
        "contracting_officers_determination_of_business_size_code": "S",
    }


def _transaction() -> dict[str, object]:
    return {
        "contract_transaction_unique_key": "T1",
        "vendor_id": "V1",
        "federal_action_obligation": 100.0,
        "total_dollars_obligated": 100.0,
        "number_of_offers_received": -2,
        "action_date": "2024-01-15T00:00:00Z",
        "period_of_performance_start_date": "2024-01-01T00:00:00Z",
        "period_of_performance_current_end_date": "2024-12-31T00:00:00Z",
        "award_type_code": "A",
        "type_of_contract_pricing_code": "J",
        "action_type_code": "A",
        "product_or_service_code": "X",
        "naics_code": "1",
        "extent_competed_code": "A",
        "solicitation_procedures_code": "A",
        "type_of_set_aside_code": "SBA",
    }


def _csv_bytes(records: list[dict[str, object]]) -> bytes:
    return pd.DataFrame(records).to_csv(index=False).encode("utf-8")


def test_health_single_record_scoring_and_structured_request_log(
    tmp_path: Path,
    caplog,
) -> None:
    app = create_app(_runtime_config(tmp_path))

    with caplog.at_level(logging.INFO, logger="datalens.api"), TestClient(app) as client:
        assert client.get("/api/v1/health/live").json()["status"] == "ok"
        assert client.get("/api/v1/health/ready").json()["status"] == "ready"

        vendor_response = client.post(
            "/api/v1/score/vendor",
            json={"fiscal_year": 2024, "record": _vendor()},
        )
        transaction_response = client.post(
            "/api/v1/score/transaction",
            json={"fiscal_year": 2024, "record": _transaction()},
        )

    assert vendor_response.status_code == 201
    assert vendor_response.json()["findings"][0]["issue_type"] == "invalid_vendor_uei"
    assert transaction_response.status_code == 201
    assert transaction_response.json()["findings"][0]["issue_type"] == "negative_offer_count"
    request_logs = [
        json.loads(record.message) for record in caplog.records if record.name == "datalens.api"
    ]
    assert any(log["path"] == "/api/v1/score/vendor" for log in request_logs)
    assert all("request_id" in log for log in request_logs)


def test_batch_scoring_persists_run_findings_csv_and_feedback(tmp_path: Path) -> None:
    app = create_app(_runtime_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/score/batch",
            data={"fiscal_year": "2024"},
            files={
                "vendors": ("vendors.csv", _csv_bytes([_vendor()]), "text/csv"),
                "transactions": (
                    "transactions.csv",
                    _csv_bytes([_transaction()]),
                    "text/csv",
                ),
            },
        )
        payload = response.json()
        run_id = payload["run_id"]
        finding_id = payload["findings"][0]["finding_id"]

        run_response = client.get(f"/api/v1/runs/{run_id}")
        findings_response = client.get(f"/api/v1/runs/{run_id}/findings")
        csv_response = client.get(f"/api/v1/runs/{run_id}/findings.csv")
        batch_feedback_response = client.post(
            f"/api/v1/runs/{run_id}/feedback/batch",
            json={
                "feedback": [
                    {
                        "finding_id": finding["finding_id"],
                        "verdict": "correct_flag",
                        "notes": "Imported review batch.",
                    }
                    for finding in payload["findings"]
                ]
            },
        )
        feedback_response = client.post(
            f"/api/v1/findings/{finding_id}/feedback",
            json={"verdict": "correct_flag", "notes": "Confirmed in source system."},
        )

    assert response.status_code == 201
    assert payload["summary"]["vendor_records"] == 1
    assert payload["summary"]["transaction_records"] == 1
    assert payload["summary"]["finding_count"] == 2
    assert run_response.status_code == 200
    assert len(findings_response.json()) == 2
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert "invalid_vendor_uei" in csv_response.text
    assert batch_feedback_response.status_code == 201
    assert batch_feedback_response.json()["saved_feedback"] == 2
    assert feedback_response.status_code == 201
    assert feedback_response.json()["verdict"] == "correct_flag"


def test_batch_validation_fails_before_scoring_and_missing_resources_return_404(
    tmp_path: Path,
) -> None:
    app = create_app(_runtime_config(tmp_path))
    invalid_transaction = _transaction()
    invalid_transaction.pop("action_date")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/score/batch",
            data={"fiscal_year": "2024"},
            files={
                "vendors": ("vendors.csv", _csv_bytes([_vendor()]), "text/csv"),
                "transactions": (
                    "transactions.csv",
                    _csv_bytes([invalid_transaction]),
                    "text/csv",
                ),
            },
        )
        missing_run = client.get("/api/v1/runs/not-found")
        missing_feedback = client.post(
            "/api/v1/findings/not-found/feedback",
            json={"verdict": "unsure"},
        )

    assert response.status_code == 422
    assert "action_date" in response.json()["detail"]
    assert missing_run.status_code == 404
    assert missing_feedback.status_code == 404


def test_manual_retraining_endpoint_returns_active_candidate_comparison(
    tmp_path: Path,
) -> None:
    app = create_app(_runtime_config(tmp_path))
    result = {
        "promotion": {"promoted": False, "gates": {"top_k_precision_non_inferior": False}},
        "active": {"top_k_precision": 0.8},
        "candidate": {"top_k_precision": 0.7},
        "training_examples": 20,
        "validation_examples": 10,
        "evaluation_top_k": 5,
    }

    with TestClient(app) as client:
        client.app.state.services.retraining.run = lambda: RetrainingRecord(
            id="retrain-1",
            active_model_version="deterministic-baseline-v1",
            candidate_model_version="anomaly-v1-test",
            promoted=False,
            result_json=json.dumps(result),
        )
        response = client.post("/api/v1/models/retrain")

    assert response.status_code == 200
    assert response.json()["active_model_version"] == "deterministic-baseline-v1"
    assert response.json()["candidate_model_version"] == "anomaly-v1-test"
    assert response.json()["promoted"] is False


def test_deactivate_active_reranker_endpoint_returns_baseline(tmp_path: Path) -> None:
    app = create_app(_runtime_config(tmp_path))

    with TestClient(app) as client:
        response = client.post("/api/v1/models/active-reranker/deactivate")

    assert response.status_code == 200
    assert response.json() == {
        "active_model_version": "deterministic-baseline-v1",
        "deactivated": False,
    }


def test_feedback_retraining_promotes_reranker_used_by_later_scoring(
    tmp_path: Path,
) -> None:
    app = create_app(_runtime_config(tmp_path))
    findings = pd.DataFrame(
        [
            {
                "target_table": "vendor",
                "record_id": f"vendor:{index:08d}",
                "issue_type": ("invalid_vendor_uei" if index % 2 == 0 else "duplicate_vendor_id"),
                "severity": "critical",
                "risk_score": 100.0,
                "evidence": "Simulated historical review example.",
            }
            for index in range(80)
        ]
    )

    with TestClient(app) as client:
        repository = client.app.state.services.repository
        run = repository.save_scoring_run(
            ScoringResult(
                vendors=pd.DataFrame(),
                transactions=pd.DataFrame(),
                findings=findings,
            ),
            fiscal_year=2024,
            schema_version="1.0.0",
            model_version="deterministic-baseline-v1",
        )
        for finding in repository.list_findings(run.id):
            positive = finding.issue_type == "invalid_vendor_uei"
            repository.add_feedback(
                finding.id,
                verdict="correct_flag" if positive else "false_alarm",
                corrected_issue_type=None,
                notes="Simulated historical feedback.",
            )

        retraining = client.post("/api/v1/models/retrain")
        scored = client.post(
            "/api/v1/score/vendor",
            json={"fiscal_year": 2024, "record": _vendor()},
        )

    assert retraining.status_code == 200
    assert retraining.json()["promoted"]
    assert (
        retraining.json()["candidate"]["top_k_precision"]
        > retraining.json()["active"]["top_k_precision"]
    )
    assert scored.status_code == 201
    assert scored.json()["model_version"].startswith("feedback-reranker-")
    assert scored.json()["findings"][0]["model_confidence"] is not None
