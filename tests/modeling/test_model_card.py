from pathlib import Path


def test_model_card_states_intended_use_metrics_and_limitations() -> None:
    model_card = Path("docs/model-card.md").read_text(encoding="utf-8")

    required_statements = (
        "DataLens does not detect fraud.",
        "controlled defects",
        "simulated",
        "top-50 precision",
        "temporal holdout",
        "The anomaly candidates were rejected.",
        "Future validation",
    )
    assert all(statement in model_card for statement in required_statements)
