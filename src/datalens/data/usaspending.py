"""Acquire reproducible USAspending contract transaction extracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import requests

from datalens.data.plan import DatasetPlan, load_dataset_plan
from datalens.paths import MANIFEST_DIR, RAW_DATA_DIR

COUNT_ENDPOINT = "/api/v2/download/count/"
DOWNLOAD_ENDPOINT = "/api/v2/download/transactions/"
STATUS_ENDPOINT = "/api/v2/download/status"
LAST_UPDATED_ENDPOINT = "/api/v2/awards/last_updated/"
NETWORK_ATTEMPTS = 5
NETWORK_BACKOFF_SECONDS = 1.0

SELECTED_COLUMNS = (
    "contract_transaction_unique_key",
    "contract_award_unique_key",
    "award_id_piid",
    "modification_number",
    "transaction_number",
    "parent_award_agency_id",
    "parent_award_agency_name",
    "parent_award_id_piid",
    "federal_action_obligation",
    "total_dollars_obligated",
    "base_and_exercised_options_value",
    "current_total_value_of_award",
    "base_and_all_options_value",
    "potential_total_value_of_award",
    "action_date",
    "action_date_fiscal_year",
    "period_of_performance_start_date",
    "period_of_performance_current_end_date",
    "period_of_performance_potential_end_date",
    "solicitation_date",
    "awarding_agency_code",
    "awarding_agency_name",
    "awarding_sub_agency_code",
    "awarding_sub_agency_name",
    "awarding_office_code",
    "awarding_office_name",
    "funding_agency_code",
    "funding_agency_name",
    "funding_sub_agency_code",
    "funding_sub_agency_name",
    "funding_office_code",
    "funding_office_name",
    "recipient_uei",
    "recipient_duns",
    "recipient_name",
    "recipient_name_raw",
    "recipient_doing_business_as_name",
    "cage_code",
    "recipient_parent_uei",
    "recipient_parent_duns",
    "recipient_parent_name",
    "recipient_country_code",
    "recipient_address_line_1",
    "recipient_address_line_2",
    "recipient_city_name",
    "recipient_county_name",
    "recipient_state_code",
    "recipient_zip_4_code",
    "primary_place_of_performance_country_code",
    "primary_place_of_performance_city_name",
    "primary_place_of_performance_county_name",
    "primary_place_of_performance_state_code",
    "primary_place_of_performance_zip_4",
    "award_type_code",
    "award_type",
    "type_of_contract_pricing_code",
    "type_of_contract_pricing",
    "transaction_description",
    "action_type_code",
    "action_type",
    "product_or_service_code",
    "product_or_service_code_description",
    "naics_code",
    "naics_description",
    "extent_competed_code",
    "extent_competed",
    "solicitation_procedures_code",
    "solicitation_procedures",
    "type_of_set_aside_code",
    "type_of_set_aside",
    "number_of_offers_received",
    "contracting_officers_determination_of_business_size_code",
    "contracting_officers_determination_of_business_size",
    "usaspending_permalink",
    "initial_report_date",
    "last_modified_date",
)


def build_filters(plan: DatasetPlan, fiscal_year: str) -> dict[str, Any]:
    """Build the exact advanced-search filters for one configured fiscal year."""
    return plan.build_filters(fiscal_year)


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    json_body: dict[str, Any] | None = None,
    timeout: float = 60,
    attempts: int = NETWORK_ATTEMPTS,
) -> dict[str, Any]:
    for attempt in range(1, attempts + 1):
        try:
            response = session.request(method, url, json=json_body, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError(f"Expected a JSON object from {url}")
            return payload
        except (requests.ConnectionError, requests.Timeout):
            if attempt == attempts:
                raise
            time.sleep(NETWORK_BACKOFF_SECONDS * 2 ** (attempt - 1))
    raise RuntimeError("Unreachable network retry state")


def _poll_download(
    session: requests.Session,
    api_base_url: str,
    file_name: str,
    *,
    timeout_seconds: int = 600,
    poll_interval_seconds: float = 3,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    status_url = f"{api_base_url}{STATUS_ENDPOINT}?file_name={file_name}"
    while time.monotonic() < deadline:
        status = _request_json(
            session,
            "GET",
            status_url,
            json_body=None,
            timeout=60,
        )
        if status.get("status") == "finished":
            return status
        if status.get("status") == "failed":
            raise RuntimeError(status.get("message") or f"USAspending download failed: {file_name}")
        time.sleep(poll_interval_seconds)
    raise TimeoutError(f"USAspending download did not finish within {timeout_seconds} seconds")


def _download_file(session: requests.Session, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(f"{destination.suffix}.part")
    with session.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    temporary_path.replace(destination)


def _extract_transaction_files(archive_path: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with ZipFile(archive_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if "PrimeTransactions" in Path(name).name and name.lower().endswith(".csv")
        ]
        if not members:
            raise FileNotFoundError("The USAspending archive did not contain transaction CSV files")
        for member in sorted(members):
            output_path = destination / Path(member).name
            with archive.open(member) as source, output_path.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            extracted.append(output_path)
    return extracted


def snapshot_is_valid(manifest_path: Path, raw_period_dir: Path) -> bool:
    """Return whether every file declared by a snapshot manifest is intact."""
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    if not files:
        return False
    for file_metadata in files:
        relative_path = Path(file_metadata["path"])
        path = RAW_DATA_DIR.parent / relative_path
        try:
            path.relative_to(raw_period_dir)
        except ValueError:
            return False
        if (
            not path.is_file()
            or path.stat().st_size != int(file_metadata["bytes"])
            or sha256_file(path) != file_metadata["sha256"]
        ):
            return False
    return True


def acquire_period(
    plan: DatasetPlan,
    fiscal_year: str,
    *,
    session: requests.Session | None = None,
    force: bool = False,
) -> Path:
    """Acquire and document one configured fiscal-year extract."""
    manifest_path = MANIFEST_DIR / f"usaspending_pbs_fy{fiscal_year}.json"
    raw_period_dir = RAW_DATA_DIR / f"fy{fiscal_year}"
    if snapshot_is_valid(manifest_path, raw_period_dir) and not force:
        return manifest_path

    period = plan.period(fiscal_year)
    api_base_url = plan.api_base_url.rstrip("/")
    client = session or requests.Session()
    filters = build_filters(plan, fiscal_year)
    count_payload = {"filters": filters, "spending_level": "transactions"}
    count_response = _request_json(
        client,
        "POST",
        f"{api_base_url}{COUNT_ENDPOINT}",
        json_body=count_payload,
    )
    row_count = int(count_response["calculated_count"])
    if row_count <= 0:
        raise ValueError(f"No transactions matched FY{fiscal_year}")
    if count_response.get("rows_gt_limit"):
        raise ValueError(f"FY{fiscal_year} exceeds the USAspending row-limited download maximum")

    download_payload = {
        "filters": filters,
        "columns": list(SELECTED_COLUMNS),
        "file_format": "csv",
        "limit": row_count,
    }
    request_response = _request_json(
        client,
        "POST",
        f"{api_base_url}{DOWNLOAD_ENDPOINT}",
        json_body=download_payload,
    )
    file_name = request_response["file_name"]
    status_response = _poll_download(client, api_base_url, file_name)
    file_url = status_response.get("file_url") or request_response["file_url"]
    if file_url.startswith("/"):
        file_url = f"{api_base_url}{file_url}"

    archive_path = RAW_DATA_DIR / file_name
    _download_file(client, file_url, archive_path)
    archive_sha256 = sha256_file(archive_path)
    extracted_files = _extract_transaction_files(archive_path, raw_period_dir)
    archive_path.unlink()

    last_updated = _request_json(
        client,
        "GET",
        f"{api_base_url}{LAST_UPDATED_ENDPOINT}",
    )
    manifest = {
        "schema_version": 1,
        "source": plan.source_name,
        "dataset_name": plan.dataset_name,
        "api_base_url": api_base_url,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "source_last_updated": last_updated,
        "fiscal_year": int(fiscal_year),
        "period_role": period.role,
        "filters": filters,
        "selected_columns": list(SELECTED_COLUMNS),
        "count_response": count_response,
        "download_request": request_response["download_request"],
        "download_status": status_response,
        "archive_sha256": archive_sha256,
        "files": [
            {
                "path": str(path.relative_to(RAW_DATA_DIR.parent)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in extracted_files
        ],
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def acquire_all(*, force: bool = False) -> list[Path]:
    """Acquire every period declared in the versioned configuration."""
    plan = load_dataset_plan()
    with requests.Session() as session:
        return [
            acquire_period(plan, fiscal_year, session=session, force=force)
            for fiscal_year in plan.fiscal_years
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["acquire"])
    parser.add_argument("--fiscal-year")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_dataset_plan()
    fiscal_years = [args.fiscal_year] if args.fiscal_year else list(plan.fiscal_years)
    with requests.Session() as session:
        for fiscal_year in fiscal_years:
            manifest = acquire_period(plan, fiscal_year, session=session, force=args.force)
            print(f"Acquired FY{fiscal_year}: {manifest}")


if __name__ == "__main__":
    main()
