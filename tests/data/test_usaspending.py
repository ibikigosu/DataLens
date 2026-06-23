import json
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
import requests

from datalens.data import usaspending
from datalens.data.plan import AgencyFilter, DatasetPlan, FiscalPeriod, load_dataset_plan


class FakeResponse:
    def __init__(self, payload=None, content: bytes = b"", status_code: int = 200) -> None:
        self.payload = payload
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self.payload

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class RequestSession:
    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = responses
        self.calls = []

    def request(self, method, url, json=None, timeout=None):
        self.calls.append((method, url, json, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _zip_bytes() -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("Contracts_PrimeTransactions_part_1.csv", "id,value\n1,x\n")
        archive.writestr("Contracts_Subawards_part_1.csv", "id,value\n2,y\n")
    return buffer.getvalue()


def _plan() -> DatasetPlan:
    return DatasetPlan(
        schema_version=1,
        dataset_version="test-dataset-v1",
        source_name="USAspending",
        dataset_name="test",
        api_base_url="https://api.example.test",
        date_type="action_date",
        agency=AgencyFilter(
            type="awarding",
            tier="subtier",
            name="Public Buildings Service",
            toptier_name="General Services Administration",
        ),
        award_type_codes=("A", "B", "C", "D"),
        periods=(
            FiscalPeriod(
                fiscal_year="2024",
                role="development",
                start_date="2023-10-01",
                end_date="2024-09-30",
            ),
        ),
    )


def test_load_config_and_build_filters() -> None:
    plan = load_dataset_plan()

    filters = usaspending.build_filters(plan, "2024")

    assert filters["award_type_codes"] == ["A", "B", "C", "D"]
    assert filters["agencies"][0]["name"] == "Public Buildings Service"
    assert filters["time_period"][0]["start_date"] == "2023-10-01"


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("DataLens\n", encoding="utf-8")

    assert (
        usaspending.sha256_file(source)
        == "7c67648534b924c8a496b679e3a586ccf7e4aee368c34b41b25ace2ecf9e71fd"
    )


def test_request_json_validates_object_payload() -> None:
    session = RequestSession([FakeResponse(payload=["not", "an", "object"])])

    with pytest.raises(TypeError, match="Expected a JSON object"):
        usaspending._request_json(session, "GET", "https://example.test")


def test_request_json_retries_transient_connection_failures(monkeypatch) -> None:
    session = RequestSession(
        [
            requests.ConnectionError("server closed connection"),
            FakeResponse(payload={"status": "finished"}),
        ]
    )
    monkeypatch.setattr(usaspending.time, "sleep", lambda seconds: None)

    result = usaspending._request_json(session, "GET", "https://example.test")

    assert result == {"status": "finished"}
    assert len(session.calls) == 2


def test_poll_download_handles_running_then_finished(monkeypatch) -> None:
    statuses = iter(
        [
            {"file_name": "file.zip", "status": "running"},
            {"file_name": "file.zip", "status": "finished", "file_url": "https://files/file.zip"},
        ]
    )
    monkeypatch.setattr(usaspending, "_request_json", lambda *args, **kwargs: next(statuses))
    monkeypatch.setattr(usaspending.time, "sleep", lambda seconds: None)

    result = usaspending._poll_download(
        object(),
        "https://api.example.test",
        "file.zip",
        poll_interval_seconds=0,
    )

    assert result["status"] == "finished"


def test_poll_download_raises_for_failed_job(monkeypatch) -> None:
    monkeypatch.setattr(
        usaspending,
        "_request_json",
        lambda *args, **kwargs: {"status": "failed", "message": "bad query"},
    )

    with pytest.raises(RuntimeError, match="bad query"):
        usaspending._poll_download(object(), "https://api.example.test", "file.zip")


def test_download_and_extract_transaction_files(tmp_path: Path) -> None:
    class DownloadSession:
        def get(self, url, stream, timeout):
            assert url == "https://files.example.test/data.zip"
            assert stream is True
            assert timeout == 180
            return FakeResponse(content=_zip_bytes())

    archive_path = tmp_path / "data.zip"
    usaspending._download_file(
        DownloadSession(),
        "https://files.example.test/data.zip",
        archive_path,
    )
    extracted = usaspending._extract_transaction_files(archive_path, tmp_path / "raw")

    assert len(extracted) == 1
    assert extracted[0].read_text(encoding="utf-8") == "id,value\n1,x\n"


def test_extract_rejects_archive_without_transactions(tmp_path: Path) -> None:
    archive_path = tmp_path / "data.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("Contracts_Subawards.csv", "id\n1\n")

    with pytest.raises(FileNotFoundError, match="did not contain"):
        usaspending._extract_transaction_files(archive_path, tmp_path / "raw")


def test_acquire_period_writes_reproducible_manifest(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    manifest_dir = tmp_path / "manifests"
    monkeypatch.setattr(usaspending, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(usaspending, "MANIFEST_DIR", manifest_dir)

    responses = iter(
        [
            {
                "calculated_count": 2,
                "rows_gt_limit": False,
                "calculated_transaction_count": 2,
            },
            {
                "file_name": "download.zip",
                "file_url": "https://files.example.test/download.zip",
                "download_request": {"limit": 2},
            },
            {"last_updated": "2026-06-21"},
        ]
    )
    monkeypatch.setattr(usaspending, "_request_json", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(
        usaspending,
        "_poll_download",
        lambda *args, **kwargs: {
            "file_name": "download.zip",
            "status": "finished",
            "file_url": "https://files.example.test/download.zip",
        },
    )

    def fake_download(session, url, destination):
        del session, url
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_zip_bytes())

    monkeypatch.setattr(usaspending, "_download_file", fake_download)
    manifest_path = usaspending.acquire_period(_plan(), "2024", session=object())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["fiscal_year"] == 2024
    assert manifest["count_response"]["calculated_count"] == 2
    assert manifest["files"][0]["path"].endswith(".csv")
    assert not (raw_dir / "download.zip").exists()


def test_acquire_period_reuses_existing_snapshot(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    manifest_dir = tmp_path / "manifests"
    period_dir = raw_dir / "fy2024"
    period_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    raw_file = period_dir / "Contracts_PrimeTransactions.csv"
    raw_file.write_text("id\n1\n", encoding="utf-8")
    manifest = manifest_dir / "usaspending_pbs_fy2024.json"
    monkeypatch.setattr(usaspending, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(usaspending, "MANIFEST_DIR", manifest_dir)
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "raw/fy2024/Contracts_PrimeTransactions.csv",
                        "bytes": raw_file.stat().st_size,
                        "sha256": usaspending.sha256_file(raw_file),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = usaspending.acquire_period(_plan(), "2024", session=object())

    assert result == manifest


def test_snapshot_integrity_rejects_modified_file(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    period_dir = raw_dir / "fy2024"
    period_dir.mkdir(parents=True)
    raw_file = period_dir / "Contracts_PrimeTransactions.csv"
    raw_file.write_text("id\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(usaspending, "RAW_DATA_DIR", raw_dir)
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "raw/fy2024/Contracts_PrimeTransactions.csv",
                        "bytes": raw_file.stat().st_size,
                        "sha256": usaspending.sha256_file(raw_file),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    raw_file.write_text("id\nchanged\n", encoding="utf-8")

    assert not usaspending.snapshot_is_valid(manifest, period_dir)
