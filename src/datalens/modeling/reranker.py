"""Supervised feedback reranking with explicit promotion evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import skops.io
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from datalens.configuration.models import FeedbackRerankerConfiguration

CATEGORICAL_FEATURES = ["target_table", "issue_type", "severity"]
NUMERIC_FEATURES = ["risk_score"]
POSITIVE_VERDICTS = {"correct_flag"}
NEGATIVE_VERDICTS = {"false_alarm", "wrong_issue_type"}
TRUSTED_TYPE_PREFIXES = ("numpy.", "scipy.", "sklearn.")


@dataclass(frozen=True)
class RerankerTrainingResult:
    """Candidate model plus active-versus-candidate ranking evidence."""

    model: Pipeline | None
    result: dict[str, Any]


def train_feedback_reranker(
    examples: pd.DataFrame,
    *,
    config: FeedbackRerankerConfiguration,
    top_k: int,
    seed: int,
) -> RerankerTrainingResult:
    """Train and evaluate a candidate using the latest decisive feedback."""
    prepared = _prepare_examples(examples)
    sufficiency = _sufficiency(prepared, config)
    if not all(sufficiency.values()):
        return RerankerTrainingResult(
            model=None,
            result={
                "promotion": {
                    "promoted": False,
                    "gates": sufficiency,
                    "reason": "Insufficient decisive feedback for candidate training.",
                },
                "active": {},
                "candidate": {},
                "training_examples": len(prepared),
            },
        )

    train, validation = train_test_split(
        prepared,
        test_size=config.validation_fraction,
        random_state=seed,
        stratify=prepared["label"],
    )
    model = _pipeline(seed)
    model.fit(train[CATEGORICAL_FEATURES + NUMERIC_FEATURES], train["label"])
    validation = validation.copy()
    validation["model_confidence"] = model.predict_proba(
        validation[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    )[:, 1]
    validation["candidate_priority"] = (
        validation["risk_score"] * 1_000 + validation["model_confidence"]
    )
    validation["active_priority"] = validation["risk_score"] * 1_000
    evaluation_k = min(top_k, max(1, len(validation) // 2))
    active_metrics = _ranking_metrics(
        validation,
        priority_column="active_priority",
        top_k=evaluation_k,
    )
    candidate_metrics = _ranking_metrics(
        validation,
        priority_column="candidate_priority",
        top_k=evaluation_k,
    )
    gates = {
        **sufficiency,
        "top_k_precision_improved": (
            candidate_metrics["top_k_precision"] > active_metrics["top_k_precision"]
        ),
        "top_k_recall_non_inferior": (
            candidate_metrics["top_k_recall"] >= active_metrics["top_k_recall"]
        ),
        "false_alarms_non_inferior": (
            candidate_metrics["false_alarms_in_top_k"] <= active_metrics["false_alarms_in_top_k"]
        ),
        "deterministic_findings_preserved": True,
    }
    promoted = all(gates.values())
    return RerankerTrainingResult(
        model=model,
        result={
            "promotion": {
                "promoted": promoted,
                "gates": gates,
                "reason": (
                    "All feedback reranker gates passed."
                    if promoted
                    else (
                        "The feedback reranker remains a candidate because one or more "
                        "gates failed."
                    )
                ),
            },
            "active": active_metrics,
            "candidate": candidate_metrics,
            "training_examples": len(train),
            "validation_examples": len(validation),
            "evaluation_top_k": evaluation_k,
        },
    )


class RerankerStore:
    """Persist safe reranker artifacts and apply the active version when present."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._cached_version: str | None = None
        self._cached_model: Pipeline | None = None

    def save_candidate(
        self,
        model: Pipeline,
        *,
        version: str,
        result: dict[str, Any],
    ) -> Path:
        candidate_dir = self._root / version
        candidate_dir.mkdir(parents=True, exist_ok=True)
        model_path = candidate_dir / "model.skops"
        model_bytes = skops.io.dumps(model)
        trusted_types = _validated_trusted_types(skops.io.get_untrusted_types(data=model_bytes))
        _write_bytes_atomically(model_path, model_bytes)
        metadata = {
            "schema_version": 1,
            "model_version": version,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "trusted_types": trusted_types,
            "result": result,
        }
        _write_text_atomically(
            candidate_dir / "metadata.json",
            json.dumps(metadata, indent=2) + "\n",
        )
        if result["promotion"]["promoted"]:
            _write_text_atomically(
                self._root / "active.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_version": version,
                        "artifact_path": f"{version}/model.skops",
                    },
                    indent=2,
                )
                + "\n",
            )
        return candidate_dir

    def rank(self, findings: pd.DataFrame) -> pd.DataFrame:
        ranked = findings.copy()
        ranked["review_priority"] = ranked["risk_score"].astype(float) * 1_000
        ranked["model_confidence"] = pd.NA
        active_version = self.active_version()
        if findings.empty or active_version is None:
            return ranked
        model = self._load(active_version)
        confidence = model.predict_proba(ranked[CATEGORICAL_FEATURES + NUMERIC_FEATURES])[:, 1]
        ranked["model_confidence"] = confidence
        ranked["review_priority"] = ranked["risk_score"].astype(float) * 1_000 + confidence
        return ranked.sort_values(
            ["review_priority", "target_table", "record_id", "issue_type"],
            ascending=[False, True, True, True],
            ignore_index=True,
        )

    def active_version(self) -> str | None:
        active_path = self._root / "active.json"
        if not active_path.exists():
            return None
        active = json.loads(active_path.read_text(encoding="utf-8"))
        return str(active["model_version"])

    def deactivate(self) -> bool:
        """Return future scoring to the deterministic baseline."""
        active_path = self._root / "active.json"
        was_active = active_path.exists()
        if was_active:
            active_path.unlink()
        self._cached_version = None
        self._cached_model = None
        return was_active

    def _load(self, version: str) -> Pipeline:
        if self._cached_version == version and self._cached_model is not None:
            return self._cached_model
        model_path = self._root / version / "model.skops"
        metadata = json.loads((model_path.parent / "metadata.json").read_text(encoding="utf-8"))
        model_bytes = model_path.read_bytes()
        untrusted_types = _validated_trusted_types(skops.io.get_untrusted_types(data=model_bytes))
        if untrusted_types != metadata["trusted_types"]:
            raise ValueError("Reranker artifact types do not match trusted metadata")
        model = skops.io.loads(model_bytes, trusted=untrusted_types)
        if not isinstance(model, Pipeline):
            raise ValueError("Reranker artifact did not contain a sklearn Pipeline")
        self._cached_version = version
        self._cached_model = model
        return model


def _prepare_examples(examples: pd.DataFrame) -> pd.DataFrame:
    if examples.empty:
        return pd.DataFrame(
            columns=[
                "finding_id",
                *CATEGORICAL_FEATURES,
                *NUMERIC_FEATURES,
                "label",
            ]
        )
    decisive = examples.loc[examples["verdict"].isin(POSITIVE_VERDICTS | NEGATIVE_VERDICTS)].copy()
    decisive = decisive.sort_values(["feedback_created_at", "finding_id"]).drop_duplicates(
        "finding_id", keep="last"
    )
    decisive["label"] = decisive["verdict"].isin(POSITIVE_VERDICTS).astype(int)
    return decisive


def _sufficiency(
    examples: pd.DataFrame,
    config: FeedbackRerankerConfiguration,
) -> dict[str, bool]:
    counts = examples["label"].value_counts() if "label" in examples else pd.Series()
    return {
        "minimum_feedback_examples": len(examples) >= config.minimum_examples,
        "minimum_positive_examples": int(counts.get(1, 0)) >= config.minimum_class_examples,
        "minimum_negative_examples": int(counts.get(0, 0)) >= config.minimum_class_examples,
    }


def _pipeline(seed: int) -> Pipeline:
    features = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _ranking_metrics(
    examples: pd.DataFrame,
    *,
    priority_column: str,
    top_k: int,
) -> dict[str, float | int]:
    ranked = examples.sort_values(
        [priority_column, "finding_id"],
        ascending=[False, True],
    )
    top = ranked.head(top_k)
    positives = int(examples["label"].sum())
    true_positives = int(top["label"].sum())
    return {
        "top_k": top_k,
        "top_k_precision": true_positives / len(top) if len(top) else 0.0,
        "top_k_recall": true_positives / positives if positives else 0.0,
        "false_alarms_in_top_k": int((1 - top["label"]).sum()),
        "overall_recall": 1.0,
    }


def _validated_trusted_types(types: list[str]) -> list[str]:
    unexpected = sorted(
        type_name for type_name in types if not type_name.startswith(TRUSTED_TYPE_PREFIXES)
    )
    if unexpected:
        raise ValueError(f"Unexpected reranker artifact types: {unexpected}")
    return sorted(types)


def _write_bytes_atomically(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _write_text_atomically(path: Path, text: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
