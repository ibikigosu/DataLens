from datalens.application.scoring import ScoringService
from datalens.configuration.loader import load_runtime_config
from datalens.demo import demo_frames, simulated_feedback_verdict


def test_demo_dataset_provides_balanced_feedback_examples() -> None:
    vendors, transactions = demo_frames()

    assert len(vendors) == 80
    assert len(transactions) == 50
    assert vendors["recipient_uei"].eq("BAD!").sum() == 40
    assert vendors["vendor_id"].duplicated(keep=False).sum() == 40
    assert transactions["contract_transaction_unique_key"].duplicated(keep=False).sum() == 20
    assert transactions["number_of_offers_received"].lt(0).sum() == 10


def test_demo_dataset_scores_vendor_and_transaction_examples() -> None:
    vendors, transactions = demo_frames()

    result = ScoringService(load_runtime_config()).score_batch(
        vendors,
        transactions,
        fiscal_year=2024,
    )

    assert set(result.findings["target_table"]) == {"vendor", "transaction"}
    assert result.findings["target_table"].value_counts().to_dict() == {
        "vendor": 80,
        "transaction": 50,
    }


def test_simulated_feedback_treats_critical_uniqueness_findings_as_correct() -> None:
    assert simulated_feedback_verdict("invalid_vendor_uei") == "correct_flag"
    assert simulated_feedback_verdict("duplicate_vendor_id") == "correct_flag"
    assert simulated_feedback_verdict("duplicate_transaction_key") == "correct_flag"
    assert simulated_feedback_verdict("negative_offer_count") == "false_alarm"
