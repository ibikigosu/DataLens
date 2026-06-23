from datalens.demo import demo_frames


def test_demo_dataset_provides_balanced_feedback_examples() -> None:
    vendors, transactions = demo_frames()

    assert len(vendors) == 400
    assert len(transactions) == 1
    assert vendors["recipient_uei"].eq("BAD!").sum() == 200
    assert vendors["vendor_id"].duplicated(keep=False).sum() == 200
