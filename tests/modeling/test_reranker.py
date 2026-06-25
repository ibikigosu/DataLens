from pathlib import Path

import pandas as pd

from datalens.configuration.loader import load_runtime_config
from datalens.modeling.reranker import (
    RerankerStore,
    train_feedback_reranker,
)


def _feedback_examples() -> pd.DataFrame:
    rows = []
    for index in range(80):
        positive = index % 2 == 0
        rows.append(
            {
                "finding_id": f"F{index:03d}",
                "record_id": f"vendor:{index:08d}",
                "target_table": "vendor",
                "issue_type": ("invalid_vendor_uei" if positive else "duplicate_vendor_id"),
                "severity": "critical",
                "risk_score": 100.0,
                "verdict": "correct_flag" if positive else "false_alarm",
                "feedback_created_at": pd.Timestamp("2026-06-23", tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def test_feedback_reranker_improves_ranking_and_activates_safe_artifact(
    tmp_path: Path,
) -> None:
    config = load_runtime_config()

    training = train_feedback_reranker(
        _feedback_examples(),
        config=config.model.feedback_reranker,
        top_k=50,
        seed=42,
    )

    assert training.model is not None
    assert training.result["promotion"]["promoted"]
    assert (
        training.result["candidate"]["top_k_precision"]
        > training.result["active"]["top_k_precision"]
    )

    store = RerankerStore(tmp_path)
    store.save_candidate(
        training.model,
        version="feedback-reranker-test",
        result=training.result,
    )
    findings = _feedback_examples().drop(
        columns=["verdict", "feedback_created_at", "label"],
        errors="ignore",
    )
    ranked = store.rank(findings)

    assert store.active_version() == "feedback-reranker-test"
    assert ranked.iloc[0]["issue_type"] == "invalid_vendor_uei"
    assert ranked["model_confidence"].notna().all()

    assert store.deactivate()
    assert store.active_version() is None
    reset_ranked = store.rank(findings)
    assert reset_ranked["model_confidence"].isna().all()


def test_feedback_reranker_deactivate_is_idempotent(tmp_path: Path) -> None:
    store = RerankerStore(tmp_path)

    assert not store.deactivate()
    assert store.active_version() is None


def test_feedback_reranker_rejects_insufficient_feedback() -> None:
    config = load_runtime_config()

    training = train_feedback_reranker(
        _feedback_examples().head(4),
        config=config.model.feedback_reranker,
        top_k=50,
        seed=42,
    )

    assert training.model is None
    assert not training.result["promotion"]["promoted"]
