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
