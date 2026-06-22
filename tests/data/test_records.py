import pandas as pd

from datalens.data.records import add_record_ids


def test_add_record_ids_preserves_source_frames_and_business_keys() -> None:
    vendors = pd.DataFrame({"vendor_id": ["UEI:A", "UEI:A"]})
    transactions = pd.DataFrame(
        {
            "contract_transaction_unique_key": ["T1", "T1"],
            "vendor_id": ["UEI:A", "UEI:A"],
        }
    )

    identified_vendors, identified_transactions = add_record_ids(vendors, transactions)

    assert "_record_id" not in vendors
    assert "_record_id" not in transactions
    assert identified_vendors["vendor_id"].tolist() == ["UEI:A", "UEI:A"]
    assert identified_transactions["contract_transaction_unique_key"].tolist() == ["T1", "T1"]
    assert identified_vendors["_record_id"].tolist() == ["vendor:00000000", "vendor:00000001"]
    assert identified_transactions["_record_id"].tolist() == [
        "transaction:00000000",
        "transaction:00000001",
    ]
    assert identified_vendors["_record_id"].is_unique
    assert identified_transactions["_record_id"].is_unique
