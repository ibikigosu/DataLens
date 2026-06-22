import json
from pathlib import Path

import pytest

from datalens.data.plan import load_dataset_plan


def test_dataset_plan_owns_period_selection_and_filters(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "source_name": "source",
                "api_base_url": "https://example.test",
                "dataset_name": "dataset",
                "agency": {
                    "type": "awarding",
                    "tier": "subtier",
                    "name": "Agency",
                    "toptier_name": "Parent",
                },
                "award_type_codes": ["A"],
                "date_type": "action_date",
                "periods": {
                    "2030": {
                        "role": "development",
                        "start_date": "2029-10-01",
                        "end_date": "2030-09-30",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    plan = load_dataset_plan(path)

    assert plan.fiscal_years == ("2030",)
    assert plan.period("2030").role == "development"
    assert plan.build_filters("2030")["time_period"][0]["start_date"] == "2029-10-01"


def test_dataset_plan_rejects_unknown_period() -> None:
    plan = load_dataset_plan()

    with pytest.raises(KeyError, match="not declared"):
        plan.period("1900")
