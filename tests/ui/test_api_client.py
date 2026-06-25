import io

import pytest
import requests

from datalens.ui.api_client import ApiError, DataLensApiClient, Upload


class StubResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)
        self.ok = 200 <= status_code < 300

    def json(self) -> dict[str, object]:
        return self._payload


class StubSession:
    def __init__(self, response: StubResponse | Exception) -> None:
        self.response = response
        self.request_call: tuple[object, ...] | None = None

    def request(self, *args, **kwargs):
        self.request_call = (args, kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_api_client_posts_paired_uploads_to_versioned_api() -> None:
    session = StubSession(StubResponse(201, {"run_id": "run-1"}))
    client = DataLensApiClient("http://api:8000/", session=session)

    result = client.score_batch(
        fiscal_year=2024,
        vendors=Upload("vendors.csv", io.BytesIO(b"vendor")),
        transactions=Upload("transactions.csv", io.BytesIO(b"transaction")),
    )

    assert result["run_id"] == "run-1"
    assert session.request_call is not None
    args, kwargs = session.request_call
    assert args == ("POST", "http://api:8000/api/v1/score/batch")
    assert kwargs["data"] == {"fiscal_year": "2024"}


def test_api_client_posts_batch_feedback_to_run_endpoint() -> None:
    session = StubSession(StubResponse(201, {"saved_feedback": 2}))
    client = DataLensApiClient("http://api:8000/", session=session)

    result = client.submit_feedback_batch(
        "run-1",
        [
            {"finding_id": "finding-1", "verdict": "correct_flag", "notes": None},
            {"finding_id": "finding-2", "verdict": "false_alarm", "notes": "Demo label."},
        ],
    )

    assert result["saved_feedback"] == 2
    assert session.request_call is not None
    args, kwargs = session.request_call
    assert args == ("POST", "http://api:8000/api/v1/runs/run-1/feedback/batch")
    assert kwargs["json"]["feedback"][0]["verdict"] == "correct_flag"


def test_api_client_deactivates_active_reranker() -> None:
    session = StubSession(
        StubResponse(
            200,
            {
                "active_model_version": "deterministic-baseline-v1",
                "deactivated": True,
            },
        )
    )
    client = DataLensApiClient("http://api:8000/", session=session)

    result = client.deactivate_active_reranker()

    assert result["deactivated"] is True
    assert session.request_call is not None
    args, _ = session.request_call
    assert args == ("POST", "http://api:8000/api/v1/models/active-reranker/deactivate")


def test_api_client_returns_bounded_api_and_network_errors() -> None:
    api_client = DataLensApiClient(
        "http://api:8000",
        session=StubSession(StubResponse(422, {"detail": "invalid input"})),
    )
    network_client = DataLensApiClient(
        "http://api:8000",
        session=StubSession(requests.ConnectionError("offline")),
    )

    with pytest.raises(ApiError, match="HTTP 422: invalid input"):
        api_client.health()
    with pytest.raises(ApiError, match="API is unavailable"):
        network_client.health()
