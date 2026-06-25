"""HTTP client used by Streamlit without duplicating application behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, BinaryIO

import requests


class ApiError(RuntimeError):
    """A bounded user-facing API failure."""


@dataclass(frozen=True)
class Upload:
    name: str
    content: BinaryIO


class DataLensApiClient:
    """Small typed wrapper around the public versioned API."""

    def __init__(
        self,
        base_url: str,
        *,
        session: requests.Session | None = None,
        timeout_seconds: float = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/health/ready")

    def score_batch(
        self,
        *,
        fiscal_year: int,
        vendors: Upload,
        transactions: Upload,
    ) -> dict[str, Any]:
        vendors.content.seek(0)
        transactions.content.seek(0)
        return self._request(
            "POST",
            "/api/v1/score/batch",
            data={"fiscal_year": str(fiscal_year)},
            files={
                "vendors": (vendors.name, vendors.content, "text/csv"),
                "transactions": (
                    transactions.name,
                    transactions.content,
                    "text/csv",
                ),
            },
        )

    def submit_feedback(
        self,
        finding_id: str,
        *,
        verdict: str,
        notes: str | None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/v1/findings/{finding_id}/feedback",
            json={"verdict": verdict, "notes": notes or None},
        )

    def submit_feedback_batch(
        self,
        run_id: str,
        feedback: list[dict[str, str | None]],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/v1/runs/{run_id}/feedback/batch",
            json={"feedback": feedback},
        )

    def retrain(self) -> dict[str, Any]:
        return self._request("POST", "/api/v1/models/retrain", timeout=300)

    def deactivate_active_reranker(self) -> dict[str, Any]:
        return self._request("POST", "/api/v1/models/active-reranker/deactivate")

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            response = self._session.request(
                method,
                f"{self._base_url}{path}",
                timeout=timeout or self._timeout_seconds,
                **kwargs,
            )
        except requests.RequestException as error:
            raise ApiError(f"DataLens API is unavailable: {error}") from error
        if response.ok:
            return response.json()
        try:
            detail = response.json().get("detail", response.text)
        except requests.JSONDecodeError:
            detail = response.text
        raise ApiError(f"DataLens API returned HTTP {response.status_code}: {detail}")
