"""Smoke test a running DataLens stack through public HTTP behavior."""

from __future__ import annotations

import argparse
import io
import sys
from typing import Any

import pandas as pd
import requests


def _vendor_csv() -> bytes:
    return (
        pd.DataFrame(
            [
                {
                    "vendor_id": "V1",
                    "recipient_name": "Vendor One",
                    "recipient_uei": "BAD!",
                    "recipient_country_code": "USA",
                    "recipient_state_code": "VA",
                    "source_transaction_count": 1,
                    "address_variant_count": 1,
                    "contracting_officers_determination_of_business_size_code": "S",
                }
            ]
        )
        .to_csv(index=False)
        .encode()
    )


def _transaction_csv() -> bytes:
    return (
        pd.DataFrame(
            [
                {
                    "contract_transaction_unique_key": "T1",
                    "vendor_id": "V1",
                    "federal_action_obligation": 100,
                    "total_dollars_obligated": 100,
                    "number_of_offers_received": -1,
                    "action_date": "2024-01-15",
                    "period_of_performance_start_date": "2024-01-01",
                    "period_of_performance_current_end_date": "2024-12-31",
                    "award_type_code": "A",
                    "type_of_contract_pricing_code": "J",
                    "action_type_code": "A",
                    "product_or_service_code": "X",
                    "naics_code": "1",
                    "extent_competed_code": "A",
                    "solicitation_procedures_code": "A",
                    "type_of_set_aside_code": "SBA",
                }
            ]
        )
        .to_csv(index=False)
        .encode()
    )


def run_smoke_test(base_url: str) -> dict[str, Any]:
    base = base_url.rstrip("/")
    health = requests.get(f"{base}/api/v1/health/ready", timeout=10)
    health.raise_for_status()
    scored = requests.post(
        f"{base}/api/v1/score/batch",
        data={"fiscal_year": "2024"},
        files={
            "vendors": ("vendors.csv", io.BytesIO(_vendor_csv()), "text/csv"),
            "transactions": (
                "transactions.csv",
                io.BytesIO(_transaction_csv()),
                "text/csv",
            ),
        },
        timeout=30,
    )
    scored.raise_for_status()
    payload = scored.json()
    run_id = payload["run_id"]
    findings = requests.get(
        f"{base}/api/v1/runs/{run_id}/findings",
        timeout=10,
    )
    findings.raise_for_status()
    finding_types = {finding["issue_type"] for finding in findings.json()}
    expected = {"invalid_vendor_uei", "negative_offer_count"}
    if not expected.issubset(finding_types):
        raise RuntimeError(f"Smoke scoring missed expected findings: {expected - finding_types}")
    return {
        "health": health.json(),
        "run_id": run_id,
        "finding_count": len(findings.json()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    try:
        result = run_smoke_test(args.base_url)
    except (requests.RequestException, RuntimeError, KeyError, ValueError) as error:
        print(f"Smoke test failed: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
